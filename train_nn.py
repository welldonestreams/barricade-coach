#!/usr/bin/env python3
"""Train a schema-2 neural policy/value candidate.

Reads the SAME teacher target format as train_policy.py (including the
. json regression cases as weighted examples) and the graded value label,
then fits a small numpy MLP with Adam. Never touches the live champion.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import coach as c
import nn_model as nn

ROOT = Path(__file__).resolve().parent


def _np():
    import numpy
    return numpy


def load_rows(paths, split='train'):
    """Yield training rows (history, policy dict, value, weight). Mirrors
    train_policy.examples() so teacher and regression files are consumed
    identically."""
    seen = set()
    for p in paths:
        p = Path(p)
        if p.suffix == '.json':
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
                sources.append(dict(history=case['history'], split='train', player_holdout=False,
                                    game_hash=f"regression:{case.get('code')}:{case.get('ply')}",
                                    policy={case['best']: 1.0}, weight=8.0,
                                    source='independent-loss-regression'))
        else:
            sources = []
            with p.open(encoding='utf-8') as src:
                for line in src:
                    try:
                        sources.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        for loaded in sources:
            nested = loaded.get('samples') if isinstance(loaded, dict) else None
            source = nested if isinstance(nested, list) else [loaded]
            for row in source:
                if nested is not None:
                    row = {**row, 'split': 'train'}
                try:
                    g = c.Game(row['history'])
                except (ValueError, KeyError, TypeError):
                    continue
                key = (row.get('game_hash'), len(g.history))
                if row.get('split') != split or row.get('player_holdout') or key in seen:
                    continue
                seen.add(key)
                yield row


def build_examples(rows):
    """Return list of dicts: state_vec, legal[], action_vecs, policy_probs,
    value_target, weight. Policy target is the teacher rank distribution over
    legal moves; value target is the graded tanh label (or +-1 from winner)."""
    out = []
    for row in rows:
        try:
            g = c.Game(row['history'])
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
        sv = nn.state_features(g)
        av = {m: nn.action_features(g, m) for m in legal}
        # value target
        vt = row.get('value')
        if vt is not None:
            vt = max(-1.0, min(1.0, float(vt)))
        else:
            w = row.get('winner')
            if w in (c.RED, c.BLUE):
                vt = 1.0 if w == g.to_move else -1.0
            else:
                vt = None
        out.append(dict(sv=sv, legal=legal, av=av, target=target,
                        vt=vt, weight=float(row.get('weight', 1.0))))
    return out


def train(model, examples, epochs=8, lr=0.003, batch=64, seed=20260907, verbose=True):
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

    # Adam state
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
        nonlocal t
        t += 1
        for k in params:
            ms[k] = b1e * ms[k] + (1 - b1e) * grads[k]
            vs[k] = b2e * vs[k] + (1 - b2e) * grads[k] ** 2
            mh = ms[k] / (1 - b1e ** t)
            vh = vs[k] / (1 - b2e ** t)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)

    for epoch in range(epochs):
        rng.shuffle(examples)
        tot_loss = 0.0
        n = 0
        for i in range(0, len(examples), batch):
            batch_ex = examples[i:i + batch]
            # accumulate grads
            gp = {k: np.zeros_like(v) for k, v in
                  {'W1': pW1, 'b1': pb1, 'W2': pW2, 'b2': pb2}.items()}
            gv = {k: np.zeros_like(v) for k, v in
                  {'W1': vW1, 'b1': vb1, 'W2': vW2, 'b2': vb2}.items()}
            loss = 0.0
            cnt = 0
            for ex in batch_ex:
                w = max(0.1, min(12.0, ex['weight']))
                sv = np.asarray(ex['sv'], dtype=np.float64)
                # ---- policy forward over all legal moves in one matrix ----
                X = np.vstack([np.concatenate([sv, np.asarray(ex['av'][m])]) for m in ex['legal']])
                h = np.tanh(X @ pW1 + pb1)
                logits = (h @ pW2 + pb2).ravel()
                logits = logits - logits.max()
                probs = np.exp(logits)
                probs = probs / probs.sum()
                # cross-entropy vs target distribution
                idx = {m: k for k, m in enumerate(ex['legal'])}
                tgt = np.zeros(len(ex['legal']))
                for m, p in ex['target'].items():
                    tgt[idx[m]] = p
                ce = -np.sum(tgt * np.log(probs + 1e-12))
                loss += w * ce
                # gradient of CE wrt logits
                dl = (probs - tgt) * w
                # backprop policy head (single tanh hidden, linear out)
                gp['W2'] += h.T @ dl[:, None]
                gp['b2'] += dl.sum()
                dh = dl[:, None] @ pW2.T
                dz = dh * (1 - h ** 2)
                gp['W1'] += X.T @ dz
                gp['b1'] += dz.sum(axis=0)
                # ---- value head ----
                if ex['vt'] is not None:
                    hv = np.tanh(sv @ vW1 + vb1)
                    yhat = float((hv @ vW2 + vb2)[0])
                    err = yhat - ex['vt']
                    # value output is raw (tanh applied later), linear loss
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
            lr_t = lr / (1 + 0.05 * (epoch * (len(examples) // batch) + n))
            adam_step({'W1': pW1, 'b1': pb1, 'W2': pW2, 'b2': pb2}, gp, m_p, v_p, lr_t)
            adam_step({'W1': vW1, 'b1': vb1, 'W2': vW2, 'b2': vb2}, gv, m_v, v_v, lr_t)
        if verbose:
            print(json.dumps(dict(epoch=epoch + 1, examples=len(examples),
                                  loss=round(tot_loss / max(1, n), 5))), flush=True)

    # write back flattened (keep in/hidden/out dims on each layer dict)
    model['policy'] = {'W1': pW1.reshape(-1).tolist(), 'b1': pb1.tolist(),
                       'W2': pW2.reshape(-1).tolist(), 'b2': pb2.tolist(),
                       'in': S + A, 'hidden': H, 'out': 1}
    model['value'] = {'W1': vW1.reshape(-1).tolist(), 'b1': vb1.tolist(),
                      'W2': vW2.reshape(-1).tolist(), 'b2': vb2.tolist(),
                      'in': S, 'hidden': H, 'out': 1}
    model['metadata'] = dict(kind='mlp', dims=d, seed=seed,
                             trained=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                             examples=len(examples), epochs=epochs)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=20260907)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()
    rows = list(load_rows(args.inputs))
    print(json.dumps(dict(rows=len(rows))), flush=True)
    ex = build_examples(rows)
    print(json.dumps(dict(examples=len(ex))), flush=True)
    model = nn.new_model(args.seed)
    model = train(model, ex, epochs=args.epochs, lr=args.lr, batch=args.batch, seed=args.seed)
    out = Path(args.output) if args.output else ROOT / 'memory' / 'candidates' / f'{time.time_ns()}' / 'model.json'
    nn.save(model, out)
    print(json.dumps(dict(candidate=str(out), id=nn.model_id(model))), flush=True)


if __name__ == '__main__':
    main()
