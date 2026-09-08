#!/usr/bin/env python3
"""Evaluate a policy candidate against frozen unguided MCTS before promotion.

Two-part repair gate (2026-09-08):

1. RAW SCREEN (cheap, model-only): for every held-out repair position the
   acceptable set must reach the raw model's top-``RAW_TOP_K`` or carry at
   least ``RAW_MIN_MASS`` probability in ``search_priors``. Thresholds are
   frozen module constants -- chosen before evaluating any corrected
   candidate -- so the screen measures model signal, not candidate admission.
2. LIVE CHECK (production path): every repair position is run through the
   exact call the live coach makes -- ``advice.advise(engine='mcts', ...)``
   with the candidate as ``policy_model``, the real 4s budget (MCTS budget
   minus the tactical-cross-check reserve), and five fixed seeds. The final
   recommendation must be acceptable on EVERY run with no timeout and no
   illegal move. The same positions are run unguided (``policy_model=None``)
   at the same budget: the candidate must not reduce repair reliability
   (guided acceptable runs >= unguided per case). Each run records the move,
   elapsed time, simulations, cross-check depth, and whether the tactical
   override fired.

The 100-pair arena still decides broad strength. A candidate is never
promoted on repair-gate results alone: if the deterministic tactical
override rescued every neural miss, the arena still has to prove the model
improves overall play (lower-95 > 0.5 vs unguided MCTS, zero illegal moves,
zero decision failures, p95 < 5s).

All gate cases are validated fail-closed: a missing file, malformed case, or
case whose expected move is illegal raises instead of being skipped.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import advice
import coach as c
import policy_value as pv
import repair_gate

ROOT = Path(__file__).resolve().parent

# The gate's repair check MUST use positions disjoint from the training set.
# repair-gate-cases.json is held out from training entirely and reserved for
# this gate only (see mine_walls.py / teacher_data.py for the exclusion
# machinery).
GATE_CASES = repair_gate.CASES

# Frozen raw-screen thresholds and live-check parameters (fixed before any
# corrected candidate was evaluated; do not tune to admit a candidate).
RAW_TOP_K = 10          # acceptable move must rank within top-10, OR
RAW_MIN_MASS = 0.05     # acceptable set carries >=5% of prior mass.
REPAIR_SECONDS = 4.0    # live_coach.query default budget (production path).
REPAIR_SEEDS = (2026090701, 2026090702, 2026090703, 2026090704, 2026090705)


def decision(g, seconds, seed, model):
    started = time.monotonic()
    try:
        result = advice.advise(g.history, g.to_move, seconds=seconds,
                               engine='mcts', seed=seed, policy_model=model)
    except c.SearchTimeout:
        return g.moves(g.to_move)[0], time.monotonic() - started, dict(arena_timeout=True)
    return result['scored'][0][1], time.monotonic() - started, result


def play(history, candidate_side, model, seconds, seed, max_plies=140, opponent=False):
    g = c.Game(history)
    latencies = []
    illegal = 0
    decision_failures = 0
    while g.winner is None and len(g.history) < max_plies:
        use = model if g.to_move == candidate_side else opponent
        move, elapsed, result = decision(g, seconds, seed + len(g.history) * 7919, use)
        if result.get('arena_timeout'):
            decision_failures += 1
        if g.to_move == candidate_side:
            latencies.append(elapsed)
        if move not in g.moves(g.to_move):
            illegal += 1
            break
        g.apply(move)
    points = .5 if g.winner is None else float(g.winner == candidate_side)
    return dict(points=points, winner=g.winner, plies=len(g.history),
                illegal=illegal, decision_failures=decision_failures,
                candidate_latencies=latencies)


def holdout_positions(paths, count, seed, player_only=False):
    rows = []
    seen = set()
    rng = random.Random(seed)
    repair_histories = repair_gate.histories()
    for path in paths:
        with Path(path).open(encoding='utf-8') as src:
            for line in src:
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        continue
                    hist = c.Game(row['history']).history
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
                key = ','.join(hist)
                # Keep the arena and the separate repair gate statistically
                # independent even when common openings recur in archive games.
                if tuple(hist) in repair_histories:
                    continue
                eligible = row.get('player_holdout') if player_only else (row.get('split') == 'holdout')
                if eligible and key not in seen:
                    seen.add(key)
                    rows.append(hist)
    rng.shuffle(rows)
    return rows[:count]


def lower95(values):
    if len(values) < 2:
        return 0.0
    return statistics.mean(values) - 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _validate_cases(paths):
    """Fail-closed case validation: returns [(case, game, legal, acceptable)]."""
    out = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'repair gate case file missing: {path}')
        try:
            cases = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f'unreadable repair gate case file {path}: {e}') from e
        if not isinstance(cases, list):
            raise ValueError(f'repair gate case file {path} is not a JSON list')
        for i, case in enumerate(cases):
            if (not isinstance(case, dict) or not case.get('best')
                    or not isinstance(case.get('history'), list)):
                raise ValueError(f'repair gate: malformed case #{i} in {path}: {case!r}')
            tag = f"{case.get('code')} ply {case.get('ply')}"
            try:
                game = c.Game(case['history'])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f'repair gate: case {tag} history invalid: {e}') from e
            legal = game.moves(game.to_move)
            expected = case['best']
            if expected not in legal:
                raise ValueError(f'repair gate: case {tag} expects {expected}, illegal after '
                                 f'{len(case["history"])} plies (off-by-one history?)')
            acceptable = case.get('acceptable', [expected])
            if (not isinstance(acceptable, list) or not acceptable
                    or any(move not in legal for move in acceptable)):
                raise ValueError(f'repair gate: case {tag} acceptable set {acceptable!r} invalid')
            out.append((case, game, legal, acceptable))
    return out


def repair_screen(model, paths, top_k=RAW_TOP_K, min_mass=RAW_MIN_MASS):
    """Raw-model screen: acceptable set within top-k or >= min mass.

    Rows carry the best acceptable-move rank, the acceptable-set probability
    mass, and ``correct`` (the frozen screen predicate). Fail closed via
    ``_validate_cases``.
    """
    rows = []
    for case, game, legal, acceptable in _validate_cases(paths):
        priors = pv.search_priors(model, game, legal)
        ranked = [m for m, _ in sorted(priors.items(), key=lambda kv: kv[1], reverse=True)] if priors else []
        chosen = ranked[0] if ranked else None
        ranks = [ranked.index(m) + 1 for m in acceptable if m in ranked]
        best_rank = min(ranks) if ranks else None
        acc_mass = sum(priors.get(m, 0.0) for m in acceptable)
        passed = bool(priors) and ((best_rank is not None and best_rank <= top_k)
                                   or acc_mass >= min_mass)
        rows.append(dict(code=case.get('code'), ply=case.get('ply'),
                         expected=case['best'], acceptable=acceptable,
                         chosen=chosen, raw_rank=best_rank, acc_mass=acc_mass,
                         passed=passed, correct=passed))
    return rows


def _advise_run(hist, side, seconds, seed, model, acceptable, legal):
    """One production-path decision; returns a full record (never raises).

    A run FAILS only when no real search completed (SearchTimeout exception
    or the engine's fallback), when the recommendation is illegal, or when it
    is not in the acceptable set. ``timed_out`` alone is the normal
    budget-exhausted state of production MCTS and is recorded, not failed.
    """
    started = time.monotonic()
    try:
        result = advice.advise(hist, side, seconds=seconds, engine='mcts',
                               seed=seed, policy_model=model)
    except c.SearchTimeout:
        return dict(move=None, ok=False, reason='timeout-exception',
                    elapsed=round(time.monotonic() - started, 3))
    rec = dict(elapsed=round(time.monotonic() - started, 3),
               simulations=result.get('simulations', 0),
               crosscheck_depth=result.get('crosscheck_depth'),
               override=result.get('tactical_override'),
               timed_out=bool(result.get('timed_out')),
               fallback=bool(result.get('fallback')),
               policy_model_id=result.get('policy_model_id'))
    scored = result.get('scored') or []
    if not scored or rec['fallback']:
        return {**rec, 'move': None, 'ok': False,
                'reason': 'fallback' if rec['fallback'] else 'no-recommendation'}
    move = scored[0][1]
    rec['move'] = move
    rec['illegal'] = move not in legal
    if rec['illegal']:
        rec.update(ok=False, reason='illegal')
    elif move not in acceptable:
        rec.update(ok=False, reason='not-acceptable')
    else:
        rec['ok'] = True
    return rec


def live_repair_check(model, paths, seconds=REPAIR_SECONDS, seeds=REPAIR_SEEDS,
                      workers=2):
    """Production-path repair check, guided vs unguided at equal budget.

    Returns one row per case: ``correct`` requires the guided final
    recommendation to be acceptable on EVERY seed with no timeout and no
    illegal move, and the guided acceptable-run count to be >= the unguided
    count (the model must not reduce repair reliability). Per-run detail is
    kept for the report.
    """
    cases = _validate_cases(paths)
    tasks = []
    for case, game, legal, acceptable in cases:
        hist = game.history
        for seed in seeds:
            for guided in (True, False):
                tasks.append((case, hist, game.to_move, legal, acceptable,
                              seed, seconds, model if guided else None, guided))
    records = {}

    def run(task):
        case, hist, side, legal, acceptable, seed, seconds, use_model, guided = task
        rec = _advise_run(hist, side, seconds, seed, use_model, acceptable, legal)
        return case.get('code'), case.get('ply'), seed, guided, rec

    with ThreadPoolExecutor(max_workers=max(1, min(4, int(workers)))) as pool:
        for code, ply, seed, guided, rec in pool.map(run, tasks):
            records.setdefault((code, ply), {})[seed, guided] = rec
    rows = []
    for case, game, legal, acceptable in cases:
        code = case.get('code')
        ply = case.get('ply')
        per = records[(code, ply)]
        guided_runs = [per[seed, True] for seed in seeds]
        unguided_runs = [per[seed, False] for seed in seeds]
        guided_ok = sum(1 for r in guided_runs if r['ok'])
        unguided_ok = sum(1 for r in unguided_runs if r['ok'])
        # guided_ok == len(seeds) already implies no timeout-exception, no
        # fallback, and no illegal move (each of those marks ok=False).
        parity = guided_ok >= unguided_ok
        correct = guided_ok == len(seeds) and parity
        rows.append(dict(code=code, ply=ply, expected=case['best'],
                         acceptable=acceptable, correct=correct,
                         guided_ok=guided_ok, unguided_ok=unguided_ok,
                         parity_ok=parity,
                         runs=[dict(seed=seed,
                                    guided=per[seed, True],
                                    unguided=per[seed, False])
                               for seed in seeds]))
    return rows


def repair_gate_passes(regressions):
    return bool(regressions) and all(row['correct'] for row in regressions)


def passes_promotion(report):
    return (report['arena_pairs'] >= 100 and report['score_lower_95'] > .5
            and report['illegal_moves'] == 0 and report['p95_seconds'] < 5
            and report['decision_failures'] == 0
            and report['screen_cases'] > 0
            and report['screen_correct'] == report['screen_cases']
            and report['live_cases'] > 0
            and report['live_correct'] == report['live_cases']
            and report['live_parity_ok'])


def evaluate(model, starts, seconds, seed, baselines=None, workers=2):
    baselines = baselines or [('raw-mcts', False)]
    tasks = [(baseline_name, baseline, i, hist)
             for baseline_name, baseline in baselines
             for i, hist in enumerate(starts)]

    def run_pair(task):
        baseline_name, baseline, i, hist = task
        g = c.Game(hist)
        scores = []
        pair_records = []
        pair_latencies = []
        for candidate_side in (g.to_move, 1 - g.to_move):
            row = play(hist, candidate_side, model, seconds, seed + i * 104729 + candidate_side,
                       opponent=baseline)
            row.update(start_hash=hashlib.sha256(','.join(hist).encode()).hexdigest(),
                       candidate_side=candidate_side, baseline=baseline_name)
            pair_records.append(row)
            scores.append(row['points'])
            pair_latencies.extend(row['candidate_latencies'])
        return pair_records, sum(scores) / 2, pair_latencies

    with ThreadPoolExecutor(max_workers=max(1, min(4, int(workers)))) as pool:
        completed = list(pool.map(run_pair, tasks))
    records = [row for pair_rows, _, _ in completed for row in pair_rows]
    pairs = [pair for _, pair, _ in completed]
    latencies = [latency for _, _, pair_latencies in completed for latency in pair_latencies]
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(.95 * len(ordered)))] if ordered else math.inf
    return dict(arena_pairs=len(pairs), games=len(records),
                score=sum(r['points'] for r in records) / max(1, len(records)),
                score_lower_95=lower95(pairs), p95_seconds=p95,
                illegal_moves=sum(r['illegal'] for r in records),
                decision_failures=sum(r['decision_failures'] for r in records),
                records=records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('candidate')
    ap.add_argument('targets', nargs='+')
    ap.add_argument('--pairs', type=int, default=100)
    ap.add_argument('--seconds', type=float, default=1)
    ap.add_argument('--seed', type=int, default=20260907)
    ap.add_argument('--promote', action='store_true')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--frozen', nargs='*', default=[],
                    help='older policy models included in the arena')
    ap.add_argument('--player-holdout-only', action='store_true')
    # The gate's repair check MUST use positions disjoint from the training
    # set (see GATE_CASES above for the full rationale).
    ap.add_argument('--regression', nargs='*', default=[GATE_CASES])
    ap.add_argument('--repair-seconds', type=float, default=REPAIR_SECONDS,
                    help='time budget for the live repair check (production default)')
    ap.add_argument('--repair-workers', type=int, default=2)
    # Measurement mode: run the live repair check even when the raw screen
    # fails, so a rejected candidate still gets its production-path numbers.
    ap.add_argument('--measure', action='store_true',
                    help='run the live repair check even if the raw screen fails')
    args = ap.parse_args()
    model = pv.load(args.candidate)
    screen = repair_screen(model, args.regression)
    screen_ok = repair_gate_passes(screen)
    live = None
    if screen_ok or args.measure:
        live = live_repair_check(model, args.regression,
                                 seconds=args.repair_seconds,
                                 workers=args.repair_workers)
    live_ok = bool(live) and repair_gate_passes(live)
    parity_ok = bool(live) and all(row['parity_ok'] for row in live)
    gate_ok = screen_ok and live_ok and parity_ok
    repair_fields = dict(screen_cases=len(screen),
                         screen_correct=sum(row['correct'] for row in screen),
                         screen=screen,
                         live_cases=len(live) if live else 0,
                         live_correct=sum(row['correct'] for row in live) if live else 0,
                         live_parity_ok=parity_ok,
                         live=live or [],
                         repair_seconds=args.repair_seconds,
                         repair_seeds=list(REPAIR_SEEDS))
    if not gate_ok:
        reasons = []
        if not screen_ok:
            reasons.append('raw screen failed')
        if live is not None and not live_ok:
            reasons.append('live repair check failed')
        if live is not None and not parity_ok:
            reasons.append('guided worse than unguided on a repair case')
        report = dict(arena_pairs=0, games=0, score=0.0, score_lower_95=0.0,
                      p95_seconds=0.0, illegal_moves=0, decision_failures=0,
                      records=[], candidate_sha256=hashlib.sha256(
                          Path(args.candidate).read_bytes()).hexdigest(),
                      policy_code_sha256=pv.code_hash(), seconds=args.seconds,
                      workers=args.workers, baselines=['raw-mcts'],
                      promoted=False, passed=False,
                      skipped_arena='repair gate failed: ' + ', '.join(reasons),
                      limitation='Frozen unguided MCTS field; no human Elo estimate',
                      **repair_fields)
    else:
        starts = holdout_positions(args.targets, args.pairs, args.seed,
                                   args.player_holdout_only)
        if len(starts) < args.pairs:
            ap.error(f'Need {args.pairs} unique held-out starts; found {len(starts)}')
        baselines = [('raw-mcts', False)] + [(str(path), pv.load(path))
                                             for path in args.frozen]
        report = evaluate(model, starts, args.seconds, args.seed, baselines, args.workers)
        report.update(candidate_sha256=hashlib.sha256(
                          Path(args.candidate).read_bytes()).hexdigest(),
                      policy_code_sha256=pv.code_hash(), seconds=args.seconds,
                      workers=args.workers,
                      baselines=[name for name, _ in baselines], promoted=False,
                      limitation='Frozen unguided MCTS field; no human Elo estimate',
                      **repair_fields)
    passed = passes_promotion(report)
    report['passed'] = passed
    if args.promote and passed:
        model['report'] = {**{k: v for k, v in report.items() if k != 'records'},
                           'promoted': True}
        pv.save(model, pv.CHAMPION)
        report['promoted'] = True
    output = Path(args.candidate).with_name('arena-report.json')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary = {k: v for k, v in report.items()
               if k not in ('records', 'screen', 'live')}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
