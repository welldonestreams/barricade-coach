#!/usr/bin/env python3
"""Generate isolated league games; never writes the live champion."""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import advice
import coach as c
import policy_value as pv

ROOT=Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def archive_files():
    try:
        import repair_gate
        excluded={str(row.get('code','')).casefold() for row in repair_gate.rows()}
    except (ImportError,ValueError,OSError):
        excluded=set()
    return tuple(path for path in (ROOT/'study/archive').glob('*.json')
                 if path.stem.casefold() not in excluded)


def sample_start(rng):
    files=archive_files()
    if not files:return []
    for _ in range(50):
        try:
            data=json.loads(rng.choice(files).read_text(encoding='utf-8-sig'))
            hist=c.Game(c.parse_history(data['historyCsv'])).history
            cut=rng.randrange(6,min(45,len(hist)-1));return hist[:cut]
        except (ValueError,KeyError,OSError):pass
    return []


def distribution(result):
    if result.get('tactical_override') or result.get('forced_loss') or result.get('tactical')=='immediate win':
        return {result['scored'][0][1]:1.0}
    rows=result.get('candidates') or []
    total=sum(max(0,r.get('visits',0)) for r in rows)
    return ({r['move']:r['visits']/total for r in rows if r.get('visits',0)>0}
            if total else {result['scored'][0][1]:1.0})


def play(start,models,seconds,rng,max_plies=140,production_safety=True):
    g=c.Game(start);samples=[];names=[m.get('metadata',{}).get('name','model') if m else 'raw-mcts' for m in models]
    while g.winner is None and len(g.history)<max_plies:
        result=advice.advise(g.history,g.to_move,seconds=seconds,engine='mcts',
                             seed=rng.randrange(2**32),policy_model=models[g.to_move],
                             production_safety=production_safety)
        samples.append(dict(history=g.history[:],side=g.to_move,policy=distribution(result)))
        g.apply(result['scored'][0][1])
    for row in samples:row['winner']=g.winner
    return dict(start=start,players=names,winner=g.winner,plies=len(g.history),samples=samples)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--games',type=int,default=100)
    ap.add_argument('--seconds',type=float,default=.5);ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--models',nargs='*',default=[]);ap.add_argument('--output',default=None)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--raw-only',action='store_true',
                    help='exclude every learned model, including the live champion')
    ap.add_argument('--no-production-safety',action='store_true',
                    help='collect fast frozen-MCTS data without live tactical overrides')
    args=ap.parse_args();pool=[False];seen=set()
    champion=None if args.raw_only else pv.load_champion()
    if champion:pool.append(champion);seen.add(pv.model_id(champion))
    for path in args.models:
        model=pv.load(path);ident=pv.model_id(model)
        if ident not in seen:pool.append(model);seen.add(ident)
    rng=random.Random(args.seed);output=Path(args.output) if args.output else ROOT/'memory/league'/f'{time.time_ns()}.jsonl'
    output.parent.mkdir(parents=True,exist_ok=True)
    def run_one(i):
        local=random.Random(args.seed+i*104729)
        players=[local.choice(pool),local.choice(pool)]
        return play(sample_start(local),players,args.seconds,local,
                    production_safety=not args.no_production_safety)
    with output.open('w',encoding='utf-8') as dst, \
         ThreadPoolExecutor(max_workers=max(1,min(4,args.workers))) as workers:
        for i,row in enumerate(workers.map(run_one,range(args.games)),1):
            dst.write(json.dumps(row,separators=(',',':'))+'\n');dst.flush()
            if i%10==0:print(f'{i}/{args.games}',flush=True)
    print(json.dumps(dict(games=args.games,output=str(output),pool=len(pool),
                          workers=max(1,min(4,args.workers)),
                          production_safety=not args.no_production_safety,
                          raw_only=args.raw_only)),flush=True)


if __name__=='__main__':main()
