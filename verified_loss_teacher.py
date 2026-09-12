#!/usr/bin/env python3
"""Build deep soft targets from fully coach-followed losses only.

The input reports are produced by update_real_games.py. They remain candidate
evidence until this independent depth-3/4 pass completes. Held-out repair
positions are excluded by mine_to_teacher.convert_case.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import coach as c
import mine_to_teacher
import repair_gate

ROOT=Path(__file__).resolve().parent


def load_cases(pattern):
    rows=[]
    for path in sorted(ROOT.glob(pattern)):
        try:data=json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError,json.JSONDecodeError):continue
        if data.get('source')!='verified_coach_trace':continue
        for case in data.get('blunders') or []:
            if not isinstance(case,dict) or not isinstance(case.get('history'),list):continue
            rows.append({**case,'code':data.get('code'),'coach_build':data.get('coach_build')})
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reports',default='study/regressions/verified-*.json')
    ap.add_argument('--output',required=True)
    ap.add_argument('--seconds',type=float,default=30.0)
    ap.add_argument('--depth',type=int,default=4,choices=(3,4))
    ap.add_argument('--min-depth',type=int,default=3,choices=(3,4))
    ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args()
    cases=load_cases(args.reports)
    gate_hists=set(repair_gate.histories())
    tasks=[(case,args.seconds,gate_hists,args.depth,args.min_depth) for case in cases]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    made=[];skipped=[];seen=set();started=time.monotonic()
    with output.open('w',encoding='utf-8') as dst:
      with ProcessPoolExecutor(max_workers=max(1,min(16,args.workers))) as pool:
        for case,(row,error) in zip(cases,pool.map(mine_to_teacher.convert_case,tasks)):
            if row is None:
                skipped.append(error);continue
            g=c.Game(row['history'])
            key=(g.pawns[0],g.pawns[1],tuple(sorted(g.walls)),
                 g.remaining[0],g.remaining[1],g.to_move)
            if key in seen:continue
            seen.add(key)
            row['source']='loss'
            row['origin']='verified_coach_loss'
            row['coach_build']=case.get('coach_build')
            dst.write(json.dumps(row,separators=(',',':'))+'\n');dst.flush();made.append(row)
    meta=dict(reports=len({case.get('code') for case in cases}),candidates=len(cases),
              targets=len(made),skipped=len(skipped),depth=args.depth,
              min_depth=args.min_depth,seconds=args.seconds,
              elapsed=round(time.monotonic()-started,2))
    output.with_suffix(output.suffix+'.meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta),flush=True)
    for error in skipped[:10]:print('skip:',error,flush=True)


if __name__=='__main__':main()
