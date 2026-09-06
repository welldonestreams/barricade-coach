#!/usr/bin/env python3
"""Harvest every top player's games from a barricade.gg leaderboard export.

Usage:
  python harvest_leaderboard.py "Leaderboard - Top Barricade Players.txt" [--limit N] [--min-games M] [--only NAME,...]

Parses the pasted leaderboard (rank, username, rating, games), collects each
profile's game list (cached under study/profiles/), then downloads every
unique game (cached under study/archive/) with pacing and resume. Re-running
continues where it left off. Afterwards rebuilds the opening book and offers
to build per-player learning models.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import collect_games

ROOT = Path(__file__).resolve().parent

ROW = re.compile(r'user/([A-Za-z0-9_-]+)>\s*(\d+(?:\.\d+)?)\s+(\d+)')


def parse_leaderboard(path, limit=None, min_games=0, only=None):
    players = []
    seen = set()
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        m = ROW.search(line)
        if not m:
            continue
        name, rating, games = m.group(1), float(m.group(2)), int(m.group(3))
        if name in seen:
            continue
        seen.add(name)
        players.append((name, rating, games))
    if only:
        wanted = {n.strip() for n in only.split(',') if n.strip()}
        players = [p for p in players if p[0] in wanted]
    players = [p for p in players if p[2] >= min_games]
    players.sort(key=lambda p: -p[1])
    if limit:
        players = players[:limit]
    return players


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('leaderboard', help='path to the pasted leaderboard text file')
    ap.add_argument('--limit', type=int, default=None, help='only the top N by rating')
    ap.add_argument('--min-games', type=int, default=0, help='skip players with fewer games')
    ap.add_argument('--only', default=None, help='comma list of specific usernames')
    ap.add_argument('--max-attempts', type=int, default=50, help='resume loops before giving up')
    args = ap.parse_args()

    players = parse_leaderboard(args.leaderboard, args.limit, args.min_games, args.only)
    if not players:
        sys.exit('No players parsed from that file - is it the leaderboard export?')
    print(f'Parsed {len(players)} players:', flush=True)
    for name, rating, games in players[:8]:
        print(f'  {name:22s} {rating:7.0f}  {games} games', flush=True)
    if len(players) > 8:
        print(f'  ... and {len(players)-8} more', flush=True)

    usernames = [p[0] for p in players]
    summary_path = ROOT / 'study' / 'collection-summary.json'
    done = False
    for attempt in range(1, args.max_attempts + 1):
        if attempt > 1:
            print(f'\n=== resume attempt {attempt} ===', flush=True)
        try:
            collect_games.collect(usernames)
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            if summary.get('complete'):
                done = True
                break
            print('Not complete yet; pausing before resuming...', flush=True)
            time.sleep(30)
        except KeyboardInterrupt:
            sys.exit('Interrupted - run again to resume.')
        except SystemExit:
            raise
        except Exception as exc:
            print(f'Collector error: {exc}; pausing before resume...', flush=True)
            time.sleep(30)
    if not done:
        print('Gave up after attempts; run again to continue (everything is cached/resumable).', flush=True)
        sys.exit(2)
    print('\n=== collection complete ===', flush=True)
    try:
        from build_book import build
        build()
        print('Opening book rebuilt.', flush=True)
    except Exception as exc:
        print(f'Opening book rebuild failed: {exc}', flush=True)


if __name__ == '__main__':
    main()
