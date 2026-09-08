#!/usr/bin/env python3
"""Convert mined one-hot loss cases into teacher-format graded rows.

The old miner wrote {code, ply, history, best, played, delta} and train_nn
turned each into a one-hot target at 16x weight -- which made wall and pawn
corpora fight each other at shared-feature positions (81s8yr p11/p13 vs
52s6kd p16 flipped with whichever corpus dominated). This converter replays
each mined position with the same deterministic depth-2 search and emits
teacher_data.py-format rows: a soft policy distribution over the top moves
(score-spread-scaled temperature, so tied tiers share mass and decisive
margins sharpen) plus the graded position value tanh(-best/250).

Gate games and exact gate histories stay excluded (the mined files already
never contained them; re-verified here). Outputs are committed under study/
so the corrected training data is reproducible from git.

Usage:
  python mine_to_teacher.py study/training-loss-cases.json --out study/loss-teacher-walls.jsonl
  python mine_to_teacher.py study/pawn-loss-cases.json --out study/loss-teacher-pawns.jsonl
"""
import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import coach as c

ROOT = Path(__file__).resolve().parent


def position_key(g):
    """Canonical board state: pawns + walls + reserves + side to move."""
    return (g.pawns[c.RED], g.pawns[c.BLUE], tuple(sorted(g.walls)),
            g.remaining[c.RED], g.remaining[c.BLUE], g.to_move)


def convert_case(task):
    case, search_s, gate_hists, target_depth, min_depth = task
    code, ply = case.get('code'), case.get('ply')
    try:
        g = c.Game(case['history'])
    except (ValueError, KeyError, TypeError):
        return None, f'{code} p{ply}: history invalid'
    hist = g.history
    if tuple(hist) in gate_hists:
        return None, f'{code} p{ply}: matches a gate history (excluded)'
    try:
        if target_depth <= 2:
            r = c.search(hist, g.to_move, 2, search_s)
        else:
            # Wide tactical scouting prevents the deep beam from excluding a
            # remote defensive wall before it sees an opponent reply. The
            # second pass narrows only after that completed depth-2 ranking.
            scout_s = min(2.5, max(1.0, search_s * .25))
            scout = c.candidate_search(hist, g.to_move, depth=2,
                                       time_limit=scout_s, beam=12,
                                       root_slack=4, wide_root=True)
            if scout.get('depth', 0) < 2:
                return None, f'{code} p{ply}: wide scout incomplete'
            roots = [move for _, move in scout['scored'][:16]]
            r = c.candidate_search(hist, g.to_move, depth=target_depth,
                                   time_limit=max(.1, search_s-scout_s), beam=8,
                                   root_moves=roots, root_slack=4,
                                   wide_root=False)
    except c.SearchTimeout:
        return None, f'{code} p{ply}: search timeout'
    if r.get('depth', 0) < min_depth or not r.get('scored'):
        return None, f'{code} p{ply}: depth-{min_depth} incomplete'
    scores = r['scored']  # [(score, move)] best-first, lower is better
    best_score = scores[0][0]
    kept = scores[:min(12, len(scores))]
    # Score-spread-scaled temperature: tied tiers share mass, decisive
    # margins sharpen. Floor keeps near-ties from being noise; cap avoids
    # overconfidence from one wide outlier.
    spread = kept[-1][0] - best_score if kept else 0.0
    temp = max(60.0, min(300.0, spread * 0.5 + 50.0))
    raw = {m: math.exp(-min(12.0, (s - best_score) / temp)) for s, m in kept}
    total = sum(raw.values())
    policy = {m: v / total for m, v in raw.items()}
    row = dict(
        game_hash=f'mined:{code}:{ply}',
        split='train',
        player_holdout=False,
        source='loss',
        weight=(5.0 if r.get('depth', 2) >= 4 else
                4.0 if r.get('depth', 2) >= 3 else 1.0),
        history=hist,
        value=math.tanh(-best_score / 250.0),
        teacher=dict(engine=('full-width-minimax' if target_depth <= 2
                             else 'wide-scout-selective-minimax'),
                     depth=r.get('depth', 2), target_depth=target_depth,
                     nodes=r.get('nodes', 0), tt_hits=r.get('tt_hits', 0),
                     seconds=r.get('elapsed', 0.0),
                     coach_sha256=hashlib.sha256(
                         Path(c.__file__).read_bytes()).hexdigest()),
        policy=policy,
    )
    return row, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', help='mined cases .json (old format)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--search-s', type=float, default=4.0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--depth', type=int, default=2, choices=(2,3,4))
    ap.add_argument('--min-depth', type=int, default=None, choices=(2,3,4))
    ap.add_argument('--limit', type=int, default=None,
                    help='convert only the first N input rows')
    args = ap.parse_args()

    import repair_gate
    gate_hists = {tuple(r['history']) for r in repair_gate.rows()}

    cases = json.loads(Path(args.input).read_text(encoding='utf-8-sig'))
    if args.limit is not None:
        cases=cases[:max(0,args.limit)]
    min_depth=args.min_depth if args.min_depth is not None else args.depth
    print(json.dumps(dict(input_cases=len(cases), workers=args.workers,
                          depth=args.depth,min_depth=min_depth)), flush=True)
    start = time.monotonic()
    done = 0
    skipped = []
    out_path = ROOT / args.out
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(convert_case, (case, args.search_s, gate_hists,
                                           args.depth,min_depth))
                for case in cases]
        with out_path.open('w', encoding='utf-8') as out:
            for fut in futs:
                row, err = fut.result()
                if row is None:
                    skipped.append(err)
                    continue
                out.write(json.dumps(row, separators=(',', ':')) + '\n')
                done += 1
                if done % 250 == 0:
                    print(json.dumps(dict(converted=done)), flush=True)
    print(json.dumps(dict(converted=done, skipped=len(skipped),
                          seconds=round(time.monotonic() - start, 1))), flush=True)
    for s in skipped[:10]:
        print('  skip:', s, flush=True)


if __name__ == '__main__':
    main()
