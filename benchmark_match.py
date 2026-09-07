#!/usr/bin/env python3
"""Measured strength: play the live coach against fixed-strength opponents.

This is the honest "does this actually win" harness the improvement plan calls
for. It pits the coach (parallel MCTS, the live engine) against a set of fixed
baselines at equal thinking time, swapping colors, and reports a win rate --
so a change can be shown to help or hurt instead of just accumulating data.

Opponents are specified as 'depth=N' (Python minimax at depth N) or
'mcts=R' (single-tree MCTS at R rollouts). Results go to study/match-benchmark.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import coach as c
import mcts_coach
import live_coach

ROOT = Path(__file__).resolve().parent


def opponent_move(hist, side, spec, seconds):
    if spec.startswith('depth='):
        depth = int(spec.split('=')[1])
        r = c.search(hist, side, depth, seconds)
        return r['scored'][0][1], r
    if spec.startswith('mcts='):
        rollouts = int(spec.split('=')[1])
        r = mcts_coach.search(hist, side, time_limit=seconds, rollouts=rollouts, workers=1)
        return r['scored'][0][1], r
    raise ValueError(f'Unknown opponent spec: {spec}')


def coach_move(hist, side, seconds):
    g=c.Game(hist)
    if side!=g.to_move: raise ValueError('Wrong side to move')
    params=live_coach.position(g)
    params.update(h=','.join(hist),walls=','.join(sorted(g.walls)),seconds=str(seconds))
    r=live_coach.query(params, record_trace=False)
    return r['top'][0][1], r


def play(hist, coach_side, opponent_spec, seconds, max_plies=160):
    g = c.Game(hist)
    while g.winner is None and len(g.history) < max_plies:
        side = g.to_move
        if side == coach_side:
            mv, _ = coach_move(g.history, side, seconds)
        else:
            mv, _ = opponent_move(g.history, side, opponent_spec, seconds)
        g.apply(mv)
    return g.winner, len(g.history)


def run(opponents, seconds, games_per_side, openers, seed0):
    records = []
    for spec in opponents:
        for opener in openers:
            for coach_side in (0, 1):
                for k in range(games_per_side):
                    # vary the RNG indirectly by rotating openers + side + k
                    winner, plies = play(opener, coach_side, spec, seconds)
                    coach_won = winner == coach_side
                    records.append(dict(opponent=spec, opener=','.join(opener),
                                        coach_side=coach_side, winner=winner,
                                        coach_won=coach_won, plies=plies))
    return records


def summarize(records):
    by_opp = {}
    for r in records:
        by_opp.setdefault(r['opponent'], [0, 0, 0])  # wins, losses, unresolved
        if r['winner'] is None:
            by_opp[r['opponent']][2] += 1
        elif r['coach_won']:
            by_opp[r['opponent']][0] += 1
        else:
            by_opp[r['opponent']][1] += 1
    out = {}
    for spec, (w, l, d) in by_opp.items():
        n = w + l + d
        out[spec] = dict(games=n, wins=w, losses=l, unresolved=d,
                         winrate=round(w / n, 3) if n else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--opponents', default='depth=2,depth=3,mcts=5000',
                    help='comma list of "depth=N" or "mcts=R"')
    ap.add_argument('--seconds', type=float, default=2.0, help='thinking time per move')
    ap.add_argument('--games-per-side', type=int, default=2)
    ap.add_argument('--openers', default='e2,e8', help='comma list of opening histories (comma-joined plies)')
    args = ap.parse_args()
    opponents = [s.strip() for s in args.opponents.split(',') if s.strip()]
    openers = [[mv for mv in o.split(',') if mv] for o in args.openers.split(';') if o.strip()]
    if not openers:
        openers = [[]]
    t = time.monotonic()
    records = run(opponents, args.seconds, args.games_per_side, openers, None)
    summary = summarize(records)
    output = dict(coach_build=live_coach.BUILD, coach_path='live_coach.query', seconds_per_move=args.seconds, games_per_side=args.games_per_side,
                  opponents=opponents, summary=summary, records=records,
                  elapsed_s=round(time.monotonic() - t, 1))
    (ROOT / 'study' / 'match-benchmark.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print(f"elapsed {output['elapsed_s']}s", flush=True)


if __name__ == '__main__':
    main()
