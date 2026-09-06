#!/usr/bin/env python3
"""Overnight self-play: the coach plays itself and records every game into
the learning outcome DB (position -> move -> won/lost), so search has real
experience to draw on once priors are wired in.

Pure-Python engine on both sides by default (fast, no Node subprocess per
move). MCTS self-play is possible via --engine mcts but is much slower.

Resumable in spirit: every completed game is written to memory/ immediately,
so an interrupted run keeps everything it finished.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import coach as c

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import learning  # noqa: E402


def play_one_game(depth, seconds, engine, random_plies):
    """Engine vs engine from a clean board. Returns (history, winner_side)."""
    g = c.Game()
    # Random legal moves for the first few plies so self-play explores many
    # openings instead of replaying the same deterministic line (which would
    # all dedupe into one outcome record).
    while g.winner is None and len(g.history) < 160:
        side = g.to_move
        if len(g.history) < random_plies:
            legal = g.moves(side)
            move = random.choice(legal)
        elif engine == 'mcts':
            import mcts_coach
            result = mcts_coach.search(g.history, side, time_limit=seconds, rollouts=20000)
            scored = result.get('scored') or []
            if not scored:
                break
            move = scored[0][1]
        else:
            result = c.search(g.history, side, depth=depth, time_limit=seconds)
            scored = result.get('scored') or []
            if not scored:
                break
            move = scored[0][1]
        g.apply(move)
    return g.history, g.winner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--games', type=int, default=1000, help='how many self-play games to run')
    ap.add_argument('--depth', type=int, default=2, help='python-engine search depth')
    ap.add_argument('--seconds', type=float, default=0.5, help='search time budget per move (seconds)')
    ap.add_argument('--engine', choices=('python', 'mcts'), default='python')
    ap.add_argument('--random-plies', type=int, default=6,
                    help='play random legal moves for the first N plies (opening variety)')
    ap.add_argument('--red', default='coach-red', help='name recorded for the red side')
    ap.add_argument('--blue', default='coach-blue', help='name recorded for the blue side')
    args = ap.parse_args()

    data = learning.load_outcomes()
    started = data['games']
    print(f'Starting self-play: {args.games} games, engine={args.engine}, '
          f'depth={args.depth}, {args.seconds}s/move. {started} games already in memory.', flush=True)

    wins = {0: 0, 1: 0, None: 0}
    t0 = time.monotonic()
    for i in range(args.games):
        hist, winner = play_one_game(args.depth, args.seconds, args.engine, args.random_plies)
        winner_name = {0: 'red', 1: 'blue', None: None}[winner]
        try:
            recorded = learning.record_game(','.join(hist), winner_name, red_name=args.red,
                                            blue_name=args.blue, source='selfplay')
            if recorded:
                # Causal "why" signal: per-move eval swing tagged by outcome.
                learning.record_evals(','.join(hist), winner_name)
        except ValueError as exc:
            print(f'game {started + i + 1}: record skipped ({exc})', flush=True)
            continue
        wins[winner] += 1
        if (i + 1) % 25 == 0:
            rate = (i + 1) / max(1e-9, time.monotonic() - t0)
            print(f'{started + i + 1} games done '
                  f'(red {wins[0]} / blue {wins[1]} / draw {wins[None]}), {rate:.1f} games/s', flush=True)

    total = learning.load_outcomes()['games']
    print(f'Done. {total} total games in memory (added {total - started}).', flush=True)


if __name__ == '__main__':
    main()
