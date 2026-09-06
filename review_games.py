#!/usr/bin/env python3
"""Reproducible local review of the ten supplied games. No official-engine grades."""
import json
from pathlib import Path
import coach as c

ROOT=Path(__file__).resolve().parent/'study'

def review():
    reports=[]
    for path in sorted(ROOT.glob('??????.json')):
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        history=data['historyCsv'].split(',')
        g=c.Game()
        facts=[]
        for i,move in enumerate(history):
            side=g.to_move
            before=[c.shortest(g.walls,g.pawns[s],c.GOALS[s]) for s in (0,1)]
            g.apply(move)
            after=[c.shortest(g.walls,g.pawns[s],c.GOALS[s]) for s in (0,1)]
            facts.append(dict(ply=i+1,side=side,move=move,paths_before=before,paths_after=after,
                              path_changes=[a-b for a,b in zip(after,before)],remaining=dict(g.remaining)))
        # Grade the user's side, or both sides of the supplied high-Elo games.
        sides=[s for s in (0,1) if data[f'player{s+1}Username']=='steak2222'] or [0,1]
        grades=[]
        for side in sides:
            grades.extend(c.grade_game(history,side,depth=2,time_limit=5))
        report=dict(code=path.stem,players=[data['player1Username'],data['player2Username']],
                    winner=data['winner'],finished_reason=data['finishedReason'],plies=len(history),
                    facts=facts,grades=sorted(grades,key=lambda r:r['ply']))
        reports.append(report)
        print(path.stem,'reviewed',len(grades),'decisions',flush=True)
        (ROOT/'local-review.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print('Review complete',flush=True)

if __name__=='__main__': review()
