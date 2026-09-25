#!/usr/bin/env python3
"""Generate study/top-players.json from the leaderboard export.

Usage: python make_top_players.py <leaderboard.txt> [--top N]
Writes {"players": [username, ...]} for the top N by rating.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import harvest_leaderboard as hl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('leaderboard', help='path to pasted leaderboard text')
    ap.add_argument('--top', type=int, default=20, help='how many players to keep (default 20)')
    args = ap.parse_args()
    players = hl.parse_leaderboard(args.leaderboard, limit=args.top)
    if not players:
        sys.exit('No players parsed')
    names = [p[0] for p in players]
    out = {'players': names, 'source': 'leaderboard', 'top': args.top}
    (ROOT / 'study' / 'top-players.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'Wrote {len(names)} players: {", ".join(names[:10])}...')


if __name__ == '__main__':
    main()
