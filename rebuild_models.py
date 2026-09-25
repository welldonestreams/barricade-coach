#!/usr/bin/env python3
"""Rebuild all existing opponent models to format 2 (color-aware) from the archive."""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import learning

archive = ROOT / 'study' / 'archive'
opp_dir = ROOT / 'memory' / 'opponents'
if not opp_dir.is_dir():
    print('no opponents dir')
    sys.exit(0)

# index archive games by player once
by_player = {}
for f in archive.glob('*.json'):
    try:
        d = json.loads(f.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        continue
    for name in (d.get('player1Username'), d.get('player2Username')):
        if name:
            by_player.setdefault(name, []).append(d)

for f in sorted(opp_dir.glob('*.json')):
    name = f.stem
    records = by_player.get(name, [])
    if not records:
        print(f'{name}: no archive games, skipping (will rebuild on live fetch)')
        continue
    model = learning.build_opponent_model(name, records)
    learning.save_opponent(name, model)
    print(f'{name} -> {model["games"]} games, format {model["format"]}, '
          f'red {model["color_games"]["red"]}/blue {model["color_games"]["blue"]}')
print('rebuild done')
