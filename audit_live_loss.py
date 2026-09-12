#!/usr/bin/env python3
"""Compare recorded live advice with a deterministic post-game search."""
import argparse
import json
from pathlib import Path

import coach as c

ROOT=Path(__file__).resolve().parent


def traces(code):
    rows=[];seen=set()
    for line in (ROOT/'logs/live-advice.jsonl').open(encoding='utf-8'):
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if code not in row.get('game',''):continue
        key=tuple(row.get('history',[]))
        if key in seen:continue
        seen.add(key);rows.append(row)
    return sorted(rows,key=lambda row:len(row['history']))


def audit(code,seconds=3.0):
    findings=[]
    for row in traces(code):
        history=row['history'];g=c.Game(history);live=row.get('top') or []
        if not live:continue
        live_move=live[0][1]
        if sum(g.remaining.values())<=6:
            result=c.search(history,g.to_move,depth=2,time_limit=seconds)
        else:
            result=c.candidate_search(history,g.to_move,depth=3,time_limit=seconds,
                                      beam=16,root_moves=[move for _,move in live[:5]])
        scores={move:score for score,move in result.get('scored',[])}
        best=result['scored'][0] if result.get('scored') else (None,None)
        live_score=scores.get(live_move)
        gap=None if live_score is None or best[0] is None else round(live_score-best[0],1)
        findings.append(dict(ply=len(history)+1,live=live_move,best=best[1],gap=gap,
                             depth=result.get('depth'),selective=result.get('selective',False),
                             top=[move for _,move in result.get('scored',[])[:5]]))
    return findings


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('codes',nargs='+')
    ap.add_argument('--seconds',type=float,default=3.0);args=ap.parse_args()
    for code in args.codes:
        rows=audit(code,args.seconds)
        print(json.dumps(dict(code=code,positions=len(rows),findings=rows),indent=2),flush=True)


if __name__=='__main__':main()
