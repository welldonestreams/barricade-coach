#!/usr/bin/env python3
"""Train a schema-2 neural policy/value candidate.

Reads teacher-format targets (jsonl rows, as produced by teacher_data.py and
mine_to_teacher.py) and fits a small numpy MLP with Adam. Never touches the
live champion.

Data pipeline (corrected 2026-09-08): mined loss positions are converted to
SOFT teacher-format rows (full top-k distribution + graded value) instead of
16x one-hot targets. Training then:

- tags every row with a source class ('teacher' or 'loss') by filename,
- merges rows that reach the same canonical board state (pawns + walls +
  reserves + side), averaging the target distributions,
- samples batches STRATIFIED by source class (--teacher-frac of each batch
  from teacher rows, the rest from loss rows), so corpus dominance is
  controlled by sampling, not by multiplying gradient weight,
- weights ordinary mined rows 2-3x, with the upper bound only when the depth-2
  search was decisive. Completed staged depth-3/4 refinements receive 4-5x.

The old one-hot .json regression format is still accepted for backwards
compatibility but demoted to weight 2.0; prefer converted jsonl rows.
"""
import argparse
import json
import math
import random
import time
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import coach as c
import nn_model as nn

ROOT = Path(__file__).resolve().parent


def _np():
    import numpy
    return numpy


def load_rows(paths, split='train'):
    """Yield training rows tagged with a source class.

    'teacher' rows come from jsonl teacher corpora; 'loss' rows come either
    from converted mined-case jsonl (soft targets) or legacy one-hot .json
    case files (weight 2.0, kept for compatibility only).
    """
    seen = set()
    try:
        import repair_gate
        held_out_histories = repair_gate.histories()
    except ImportError:
        held_out_histories = frozenset()
    for p in paths:
        p = Path(p)
        source = 'loss' if 'loss' in p.stem.casefold() else 'teacher'
        if p.suffix == '.json':
            # Legacy one-hot case files (old miner output). Softly demoted;
            # run mine_to_teacher.py to convert them to jsonl distributions.
            try:
                data = json.loads(p.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, list):
                continue
            sources = []
            for case in data:
                if not isinstance(case, dict) or not case.get('history') or not case.get('best'):
                    continue
                sources.append(dict(history=case['history'], split='train',
                                    player_holdout=False,
                                    game_hash=f"regression:{case.get('code')}:{case.get('ply')}",
                                    policy={case['best']: 1.0}, weight=2.0,
                                    source='loss'))
        else:
            sources = []
            with p.open(encoding='utf-8') as src:
                for line in src:
                    try:
                        loaded = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sources.append(loaded)
        for loaded in sources:
            nested = loaded.get('samples') if isinstance(loaded, dict) else None
            batch = nested if isinstance(nested, list) else [loaded]
            for row in batch:
                if nested is not None:
                    row = {**row, 'split': 'train'}
                try:
                    g = c.Game(row['history'])
                except (ValueError, KeyError, TypeError):
                    continue
                # Suppress byte-for-byte-equivalent supervision rows while
                # allowing a deeper teacher to refine the same game position.
                # The old (game_hash, ply) key discarded all staged depth-3/4
                # labels because they intentionally retain their shallow
                # source hash for auditability.
                policy_sig=tuple(sorted((str(m),float(v)) for m,v in
                                        (row.get('policy') or {}).items()))
                teacher_depth=(row.get('teacher') or {}).get('depth')
                key=(row.get('game_hash'),len(g.history),teacher_depth,policy_sig)
                if row.get('split') != split or row.get('player_holdout') or key in seen:
                    continue
                # The repair gate evaluates generalization on positions the
                # candidate must NEVER have seen in training.
                if tuple(g.history) in held_out_histories:
                    continue
                seen.add(key)
                row.setdefault('source', source)
                row['_game']=g  # reuse validated replay during canonical merge
                yield row


def position_key(g):
    """Canonical board state (transposition-independent)."""
    return (g.pawns[c.RED], g.pawns[c.BLUE], tuple(sorted(g.walls)),
            g.remaining[c.RED], g.remaining[c.BLUE], g.to_move)


def _featurize(acc):
    pz=sum(acc['policy'].values())
    if not pz:return None
    policy={m:v/pz for m,v in acc['policy'].items()}
    context=nn.action_context(acc['g'])
    return dict(sv=nn.state_features(acc['g']),legal=acc['legal'],
                av={m:nn.action_features(acc['g'],m,context=context)
                    for m in acc['legal']},target=policy,
                vt=(acc['vt_acc']/acc['w_acc']) if acc['w_acc'] else None,
                weight=min(12.0,max(1.0,acc['w_acc'])),kind=acc['kind'])


def build_examples(rows,workers=1):
    """Return list of dicts: state_vec, legal[], action_vecs, policy_probs,
    value_target, weight, kind. Duplicate canonical board states from
    different sources are merged: target distributions are averaged
    (weighted by each row's weight), value targets weighted-averaged.
    """
    merged = {}
    order = []
    for row in rows:
        try:
            g=row.get('_game') or c.Game(row['history'])
        except (ValueError, KeyError, TypeError):
            continue
        legal = g.moves(g.to_move)
        if not legal:
            continue
        target = {m: float(v) for m, v in row.get('policy', {}).items() if m in legal and v >= 0}
        z = sum(target.values())
        if not z:
            continue
        target = {m: v / z for m, v in target.items()}
        kind = 'loss' if row.get('source') == 'loss' else 'teacher'
        w = float(row.get('weight', 1.0))
        if kind == 'loss':
            # Mined rows: 2x base, 3x when the search was decisive (top
            # policy mass sharp => large stable score margin).
            top_mass = max(target.values())
            base = 3.0 if top_mass >= 0.6 else 2.0
            depth = int((row.get('teacher') or {}).get('depth', 2))
            depth_weight = 5.0 if depth >= 4 else 4.0 if depth >= 3 else base
            w = max(w, depth_weight)
        vt = row.get('value')
        if vt is not None:
            vt = max(-1.0, min(1.0, float(vt)))
        else:
            winner = row.get('winner')
            if winner in (c.RED, c.BLUE):
                vt = 1.0 if winner == g.to_move else -1.0
            else:
                vt = None
        key = position_key(g)
        acc = merged.get(key)
        if acc is None:
            acc = dict(sv=None, legal=legal, av=None, policy=None,
                       vt_acc=0.0, w_acc=0.0, kind=kind, g=g)
            merged[key] = acc
            order.append(key)
        elif kind=='loss':
            # Any hard-position supervision keeps the merged state in the
            # stratified loss pool. Previously the first ordinary teacher row
            # won this tag, so overlapping hard examples quietly disappeared
            # from the guaranteed hard-example share of each batch.
            acc['kind']='loss'
        acc['policy'] = acc['policy'] or {}
        for m, pr in target.items():
            acc['policy'][m] = acc['policy'].get(m, 0.0) + pr * w
        if vt is not None:
            acc['vt_acc'] += vt * w
            acc['w_acc'] += w
    pending=[merged[key] for key in order]
    if workers>1 and len(pending)>1:
        with ProcessPoolExecutor(max_workers=min(16,int(workers))) as pool:
            made=pool.map(_featurize,pending,chunksize=8)
            return [row for row in made if row is not None]
    return [row for row in map(_featurize,pending) if row is not None]


def train(model, examples, epochs=8, lr=0.003, batch=64, seed=20260907,
          teacher_frac=0.7, verbose=True):
    """Fit with Adam over source-stratified batches.

    Each epoch shuffles the teacher and loss pools and consumes batches that
    mix ``teacher_frac`` teacher rows with the rest loss rows, so a small
    high-value loss corpus is seen every epoch without drowning ordinary
    positions. If there are no loss rows the epoch is a plain sequential
    pass over the teacher pool (old behaviour).
    """
    np = _np()
    rng = random.Random(seed)
    d = model['metadata']['dims']
    S, A, H = d['state'], d['action'], d['hidden']

    def unflatten(layers, in_dim):
        W1 = np.asarray(layers['W1'], dtype=np.float64).reshape(in_dim, H)
        b1 = np.asarray(layers['b1'], dtype=np.float64)
        W2 = np.asarray(layers['W2'], dtype=np.float64).reshape(H, 1)
        b2 = np.asarray(layers['b2'], dtype=np.float64)
        return W1, b1, W2, b2

    pW1, pb1, pW2, pb2 = unflatten(model['policy'], S + A)
    vW1, vb1, vW2, vb2 = unflatten(model['value'], S)

    def adam():
        return dict(t=0)

    m_p = {k: np.zeros_like(v) for k, v in
           {'W1': pW1, 'b1': pb1, 'W2': pW2, 'b2': pb2}.items()}
    v_p = {k: np.zeros_like(v) for k, v in
           {'W1': pW1, 'b1': pb1, 'W2': pW2, 'b2': pb2}.items()}
    m_v = {k: np.zeros_like(v) for k, v in
           {'W1': vW1, 'b1': vb1, 'W2': vW2, 'b2': vb2}.items()}
    v_v = {k: np.zeros_like(v) for k, v in
           {'W1': vW1, 'b1': vb1, 'W2': vW2, 'b2': vb2}.items()}
    t = 0
    b1e = 0.9
    b2e = 0.999
    eps = 1e-8

    def adam_step(params, grads, ms, vs, lr):
        for k in params:
            ms[k] = b1e * ms[k] + (1 - b1e) * grads[k]
            vs[k] = b2e * vs[k] + (1 - b2e) * grads[k] ** 2
            mh = ms[k] / (1 - b1e ** t)
            vh = vs[k] / (1 - b2e ** t)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)

    teachers = [ex for ex in examples if ex['kind'] != 'loss']
    losses = [ex for ex in examples if ex['kind'] == 'loss']
    stratify = bool(teachers) and bool(losses)
    if not stratify:
        teachers = examples
    for epoch in range(epochs):
        rng.shuffle(teachers)
        rng.shuffle(losses)
        tot_loss = 0.0
        n = 0
        if stratify:
            n_teacher = max(1, int(round(batch * teacher_frac)))
            n_loss = max(1, batch - n_teacher)
            n_batches = math.ceil(len(teachers) / n_teacher)
        else:
            n_batches = math.ceil(max(1, len(teachers)) / batch)
        for b in range(n_batches):
            if stratify:
                t_off = (b * n_teacher) % len(teachers)
                teacher_chunk = [teachers[(t_off + j) % len(teachers)]
                                 for j in range(n_teacher)]
                l_off = (b * n_loss) % len(losses)
                loss_chunk = [losses[(l_off + j) % len(losses)]
                              for j in range(n_loss)]
                chunk = teacher_chunk + loss_chunk
            else:
                start = (b * batch) % len(teachers)
                chunk = [teachers[(start + j) % len(teachers)]
                         for j in range(min(batch, len(teachers)))]
            gp = {k: np.zeros_like(v) for k, v in
                  {'W1': pW1, 'b1': pb1, 'W2': pW2, 'b2': pb2}.items()}
            gv = {k: np.zeros_like(v) for k, v in
                  {'W1': vW1, 'b1': vb1, 'W2': vW2, 'b2': vb2}.items()}
            loss = 0.0
            cnt = 0
            for ex in chunk:
                w = max(0.1, min(12.0, ex['weight']))
                sv = np.asarray(ex['sv'], dtype=np.float64)
                X = np.vstack([np.concatenate([sv, np.asarray(ex['av'][m])]) for m in ex['legal']])
                h = np.tanh(X @ pW1 + pb1)
                logits = (h @ pW2 + pb2).ravel()
                logits = logits - logits.max()
                probs = np.exp(logits)
                probs = probs / probs.sum()
                idx = {m: k for k, m in enumerate(ex['legal'])}
                tgt = np.zeros(len(ex['legal']))
                for m, p in ex['target'].items():
                    tgt[idx[m]] = p
                ce = -np.sum(tgt * np.log(probs + 1e-12))
                loss += w * ce
                dl = (probs - tgt) * w
                gp['W2'] += h.T @ dl[:, None]
                gp['b2'] += dl.sum()
                dh = dl[:, None] @ pW2.T
                dz = dh * (1 - h ** 2)
                gp['W1'] += X.T @ dz
                gp['b1'] += dz.sum(axis=0)
                if ex['vt'] is not None:
                    hv = np.tanh(sv @ vW1 + vb1)
                    yhat = float((hv @ vW2 + vb2)[0])
                    err = yhat - ex['vt']
                    gv['W2'] += (hv * w * err)[:, None]
                    gv['b2'] += np.array([w * err])
                    dhv = (w * err) * vW2.ravel()
                    dzv = dhv * (1 - hv ** 2)
                    gv['W1'] += sv[:, None] @ dzv[None, :]
                    gv['b1'] += dzv
                cnt += 1
            if cnt:
                loss /= cnt
                tot_loss += loss
                n += 1
            t += 1
            lr_t = lr
            adam_step({'W1': pW1, 'b1': pb1, 'W2': pW2, 'b2': pb2}, gp, m_p, v_p, lr_t)
            adam_step({'W1': vW1, 'b1': vb1, 'W2': vW2, 'b2': vb2}, gv, m_v, v_v, lr_t)
        if verbose:
            print(json.dumps(dict(epoch=epoch + 1, examples=len(examples),
                                  teachers=len(teachers), losses=len(losses),
                                  loss=round(tot_loss / max(1, n), 5))), flush=True)

    model['policy'] = {'W1': pW1.reshape(-1).tolist(), 'b1': pb1.tolist(),
                       'W2': pW2.reshape(-1).tolist(), 'b2': pb2.tolist(),
                       'in': S + A, 'hidden': H, 'out': 1}
    model['value'] = {'W1': vW1.reshape(-1).tolist(), 'b1': vb1.tolist(),
                      'W2': vW2.reshape(-1).tolist(), 'b2': vb2.tolist(),
                      'in': S, 'hidden': H, 'out': 1}
    model['metadata'] = dict(kind='mlp', dims=d, seed=seed,
                             trained=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                             examples=len(examples), epochs=epochs,
                             teacher_frac=teacher_frac,
                             stratified=stratify)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=20260908)
    ap.add_argument('--teacher-frac', type=float, default=0.7)
    ap.add_argument('--hidden',type=int,default=nn.HIDDEN,
                    help='hidden units in each policy/value network (16-1024)')
    ap.add_argument('--feature-workers',type=int,
                    default=max(1,min(8,(os.cpu_count() or 2)-2)))
    ap.add_argument('--output', default=None)
    args = ap.parse_args()
    if not 16<=args.hidden<=1024 or not 1<=args.feature_workers<=16:
        ap.error('--hidden must be 16-1024 and --feature-workers 1-16')
    rows = list(load_rows(args.inputs))
    print(json.dumps(dict(rows=len(rows))), flush=True)
    ex = build_examples(rows,workers=args.feature_workers)
    kinds = {}
    for e in ex:
        kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
    print(json.dumps(dict(examples=len(ex), kinds=kinds)), flush=True)
    model = nn.new_model(args.seed,hidden=args.hidden)
    model = train(model, ex, epochs=args.epochs, lr=args.lr, batch=args.batch,
                  seed=args.seed, teacher_frac=args.teacher_frac)
    out = Path(args.output) if args.output else ROOT / 'memory' / 'candidates' / f'{time.time_ns()}' / 'model.json'
    nn.save(model, out)
    print(json.dumps(dict(candidate=str(out), id=nn.model_id(model))), flush=True)


if __name__ == '__main__':
    main()
