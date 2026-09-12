#!/usr/bin/env python3
"""Build empirical opening references from the full archive, not a hard lookup.

Every game in study/archive (and the study/*.json fixtures) contributes; moves
are ranked by (games, winrate) with a minimum sample floor. The book is a
*reference*: the live coach may play the dominant opening move instantly when
one is overwhelmingly supported, and otherwise falls through to full search.
"""
from collections import defaultdict
import json
from pathlib import Path
import coach

ROOT = Path(__file__).resolve().parent / 'study'


def build():
    nodes = defaultdict(lambda: defaultdict(lambda: dict(games=0, wins=0)))
    used = 0
    paths = list(ROOT.glob('??????.json')) + list((ROOT / 'archive').glob('*.json'))
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            if data.get('boardSize') != 9:
                continue
            coach.Game(data['historyCsv'])
        except (ValueError, KeyError):
            continue
        h = coach.parse_history(data['historyCsv'])
        winner = data.get('winner')
        used += 1
        # Record every ply of every game up to max_ply. A ply is attributed to
        # the side that moved, tagged won/lost by whether that side won.
        for i, move in enumerate(h[:16]):
            side = i % 2
            row = nodes[','.join(h[:i])][move]
            row['games'] += 1
            row['wins'] += int(str(winner) == str(side + 1))
    book = {}
    for key, moves in nodes.items():
        rows = [dict(move=move, games=r['games'], wins=r['wins'],
                     winrate=round(r['wins'] / r['games'], 3))
                for move, r in moves.items() if r['games'] >= 3]
        if rows:
            book[key] = sorted(rows, key=lambda r: (-r['games'], -r['winrate'], r['move']))[:6]
    output = dict(source_games=used, max_ply=16, min_samples=3, positions=book)
    target = ROOT / 'opening-book.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(output, indent=2), encoding='utf-8')
    tmp.replace(target)
    print(f'Opening book: {used} games, {len(book)} positions', flush=True)
    return output


if __name__ == '__main__':
    build()
