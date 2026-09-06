#!/usr/bin/env python3
"""Build empirical opening references, never a forced best-move lookup."""
from collections import defaultdict
import json
from pathlib import Path
import coach

ROOT=Path(__file__).resolve().parent/'study'
PLAYERS={'MikeJordan','RejeCted_','owen12345'}

def build():
    nodes=defaultdict(lambda:defaultdict(lambda:dict(games=0,wins=0)))
    games={}
    for path in list(ROOT.glob('??????.json'))+list((ROOT/'archive').glob('*.json')):
        try:
            data=json.loads(path.read_text(encoding='utf-8-sig'))
            if data['boardSize']!=9: continue
            coach.Game(data['historyCsv'])
            games[data['shareCode']]=data
        except (ValueError,KeyError):
            continue
    used=0
    for data in games.values():
        h=coach.parse_history(data['historyCsv'])
        if not any(data[f'player{s+1}Username'] in PLAYERS for s in (0,1)): continue
        used+=1
        for i,move in enumerate(h[:12]):
            side=i%2
            if data[f'player{side+1}Username'] not in PLAYERS: continue
            row=nodes[','.join(h[:i])][move]
            row['games']+=1
            row['wins']+=int(str(data['winner'])==str(side+1))
    book={key: sorted((dict(move=move,**row) for move,row in moves.items() if row['games']>=3),
                      key=lambda row:(-row['games'],row['move']))[:5] for key,moves in nodes.items()}
    book={key:rows for key,rows in book.items() if rows}
    output=dict(source_games=used,max_ply=12,min_samples=3,positions=book)
    target=ROOT/'opening-book.json'
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(output,indent=2),encoding='utf-8')
    temporary.replace(target)
    print(f'Opening references: {used} games, {len(book)} positions',flush=True)
    return output

if __name__=='__main__': build()
