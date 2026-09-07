#!/usr/bin/env python3
"""Generate isolated league games; never writes the live champion."""
import argparse
import json
import random
import time
from pathlib import Path

import advice
import coach as c
import policy_value as pv

ROOT=Path(__file__).resolve().parent


def sample_start(rng):
    files=list((ROOT/'study/archive').glob('*.json'))
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


def play(start,models,seconds,rng,max_plies=140):
    g=c.Game(start);samples=[];names=[m.get('metadata',{}).get('name','model') if m else 'raw-mcts' for m in models]
    while g.winner is None and len(g.history)<max_plies:
        result=advice.advise(g.history,g.to_move,seconds=seconds,engine='mcts',
                             seed=rng.randrange(2**32),policy_model=models[g.to_move])
        samples.append(dict(history=g.history[:],side=g.to_move,policy=distribution(result)))
        g.apply(result['scored'][0][1])
    for row in samples:row['winner']=g.winner
    return dict(start=start,players=names,winner=g.winner,plies=len(g.history),samples=samples)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--games',type=int,default=100)
    ap.add_argument('--seconds',type=float,default=.5);ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--models',nargs='*',default=[]);ap.add_argument('--output',default=None)
    args=ap.parse_args();pool=[False];seen=set()
    champion=pv.load_champion()
    if champion:
        pool.append(champion);seen.add(pv.model_id(champion))
    for path in args.models:
        model=pv.load(path);ident=pv.model_id(model)
        if ident not in seen:pool.append(model);seen.add(ident)
    rng=random.Random(args.seed);output=Path(args.output) if args.output else ROOT/'memory/league'/f'{time.time_ns()}.jsonl'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8') as dst:
        for i in range(args.games):
            players=[rng.choice(pool),rng.choice(pool)];row=play(sample_start(rng),players,args.seconds,rng)
            dst.write(json.dumps(row,separators=(',',':'))+'\n');dst.flush()
            if (i+1)%10==0:print(f'{i+1}/{args.games}',flush=True)
    print(json.dumps(dict(games=args.games,output=str(output),pool=len(pool))),flush=True)


if __name__=='__main__':main()
