#!/usr/bin/env python3
"""Self-play with position seeding: the coach plays itself, but instead of
always starting from an empty board, it frequently resumes from a real
mid-game position sampled from the study archive (top players' games). This
breaks the self-vs-self feedback loop — the engine is forced to solve
positions it did not generate, so it can't just replay its own narrow line.

Also supports N parallel workers (separate processes, safe via SQLite WAL),
each assigned a worker id for log separation.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import coach as c

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import learning  # noqa: E402

ARCHIVE = ROOT / 'study' / 'archive'


def sample_seed(rng, min_ply=6, max_ply=40):
    """Pick a random archived game and a random ply, returning the history up to
    that ply (the position the engine will solve from). Returns [] on failure."""
    files = list(ARCHIVE.glob('*.json'))
    if not files:
        return []
    for _ in range(20):  # a few tries to find a long-enough game
        f = rng.choice(files)
        try:
            data = json_load(f)
        except Exception:
            continue
        hist = data.get('historyCsv', '')
        if not hist:
            continue
        try:
            moves = c.parse_history(hist)
        except ValueError:
            continue
        if len(moves) <= min_ply:
            continue
        cut = rng.randint(min_ply, min(max_ply, len(moves) - 1))
        return moves[:cut]
    return []


def json_load(path):
    import json
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def play_from(seed_hist, depth, seconds, engine, random_plies, rng):
    """Play engine-vs-engine from an arbitrary starting position (seed_hist).
    random_plies random legal moves are inserted first for exploration variety.
    Returns (full_history, winner_side)."""
    g = c.Game(seed_hist)
    while g.winner is None and len(g.history) < 160:
        side = g.to_move
        if len(g.history) < len(seed_hist) + random_plies:
            legal = g.moves(side)
            if not legal:
                break
            move = rng.choice(legal)
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
    ap.add_argument('--games', type=int, default=1000, help='games per worker')
    ap.add_argument('--depth', type=int, default=3, help='python-engine search depth')
    ap.add_argument('--seconds', type=float, default=0.5, help='search budget per move (s)')
    ap.add_argument('--engine', choices=('python', 'mcts'), default='python')
    ap.add_argument('--random-plies', type=int, default=3, help='random legal moves inserted after seed for variety')
    ap.add_argument('--seed-prob', type=float, default=0.7, help='probability a game starts from an imported position')
    ap.add_argument('--seed-min-ply', type=int, default=6)
    ap.add_argument('--seed-max-ply', type=int, default=40)
    ap.add_argument('--worker', type=int, default=0, help='worker id for log separation')
    ap.add_argument('--seed', type=int, default=None, help='RNG seed')
    args = ap.parse_args()

    rng = random.Random(args.seed if args.seed is not None else (time.time_ns() ^ (args.worker * 7919)))
    tag = f'[w{args.worker}]'

    started = learning.memory_stats()['games']
    print(f'{tag} self-play: {args.games} games, depth={args.depth}, {args.seconds}s/move, '
          f'seed-prob={args.seed_prob}, engine={args.engine}. {started} games in memory.', flush=True)

    wins = {0: 0, 1: 0, None: 0}
    seeded_count = 0
    t0 = time.monotonic()
    for i in range(args.games):
        seed_hist = sample_seed(rng, args.seed_min_ply, args.seed_max_ply) if rng.random() < args.seed_prob else []
        if seed_hist:
            seeded_count += 1
        hist, winner = play_from(seed_hist, args.depth, args.seconds, args.engine, args.random_plies, rng)
        winner_name = {0: 'red', 1: 'blue', None: None}[winner]
        try:
            recorded = learning.record_game(','.join(hist), winner_name,
                                            red_name='coach-red', blue_name='coach-blue',
                                            source='selfplay', seed_len=len(seed_hist))
            if recorded:
                learning.record_evals(','.join(hist), winner_name, seed_len=len(seed_hist))
        except ValueError as exc:
            print(f'{tag} game {i + 1}: record skipped ({exc})', flush=True)
            continue
        wins[winner] += 1
        if (i + 1) % 25 == 0:
            rate = (i + 1) / max(1e-9, time.monotonic() - t0)
            total = learning.memory_stats()['games']
            print(f'{tag} {i + 1} done (red {wins[0]}/blue {wins[1]}/draw {wins[None]}, '
                  f'{seeded_count} seeded), {rate:.2f} games/s, {total} total in memory', flush=True)

    total = learning.memory_stats()['games']
    print(f'{tag} done. {total} total games in memory (added {total - started}).', flush=True)


if __name__ == '__main__':
    main()
