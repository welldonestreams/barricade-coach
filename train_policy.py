#!/usr/bin/env python3
"""Train a sparse policy/value candidate; never changes the live champion."""
import argparse
import json
import math
import random
import hashlib
import time
from pathlib import Path

import coach as c
import policy_value as pv

ROOT=Path(__file__).resolve().parent


def examples(paths,split='train'):
    seen=set()
    for path in paths:
        with Path(path).open(encoding='utf-8') as src:
            for line in src:
                try:loaded=json.loads(line)
                except json.JSONDecodeError:continue
                nested=loaded.get('samples') if isinstance(loaded,dict) else None
                source=nested if isinstance(nested,list) else [loaded]
                for row in source:
                    if nested is not None:
                        stable=hashlib.sha256(','.join(loaded.get('start',[])).encode()).hexdigest()
                        row={**row,'split':'train','game_hash':loaded.get('game_hash') or stable}
                    try:g=c.Game(row['history'])
                    except (ValueError,KeyError,TypeError):continue
                    key=(row.get('game_hash'),len(g.history))
                    if row.get('split')!=split or row.get('player_holdout') or key in seen:continue
                    seen.add(key);yield row


def train(rows,epochs=3,rate=.03,value_rate=.01,seed=1,base=None):
    model=pv.new_model(base);rng=random.Random(seed);policy=model['policy'];value=model['value']
    rows=list(rows)
    for epoch in range(epochs):
        rng.shuffle(rows);policy_loss=value_loss=count=0
        step=rate/(1+.35*epoch)
        for row in rows:
            g=c.Game(row['history']);legal=g.moves(g.to_move)
            target={m:float(v) for m,v in row.get('policy',{}).items() if m in legal and v>=0}
            z=sum(target.values())
            if not z:continue
            target={m:v/z for m,v in target.items()}
            probs=pv.policy_priors(model,g,legal)
            for move,p in probs.items():
                delta=p-target.get(move,0.0)
                for key in pv.policy_keys(g,move):policy[key]=policy.get(key,0.0)-step*delta
            policy_loss-=sum(t*math.log(max(1e-12,probs.get(m,1e-12))) for m,t in target.items())
            winner=row.get('winner')
            # Prefer the teacher's graded search value when present; fall back
            # to the eventual winner for legacy/league rows that lack it.
            wanted=row.get('value')
            if wanted is not None:
                wanted=max(-1.0,min(1.0,float(wanted)))
            elif winner in (c.RED,c.BLUE):
                wanted=1.0 if winner==g.to_move else -1.0
            if wanted is not None:
                pred=pv.value(model,g)
                grad=(pred-wanted)*(1-pred*pred)
                for key in pv.state_features(g):value[key]=value.get(key,0.0)-value_rate*grad
                value_loss+=(pred-wanted)**2
            count+=1
        print(json.dumps(dict(epoch=epoch+1,examples=count,
              policy_loss=policy_loss/max(1,count),value_loss=value_loss/max(1,count))),flush=True)
    model['metadata']=dict(trained=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                           examples=len(rows),epochs=epochs,policy_weights=len(policy),
                           value_weights=len(value),seed=seed)
    return model


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs',nargs='+');ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--rate',type=float,default=.03);ap.add_argument('--value-rate',type=float,default=.01)
    ap.add_argument('--seed',type=int,default=20260907);ap.add_argument('--warm-start',action='store_true')
    ap.add_argument('--output',default=None)
    args=ap.parse_args();base=pv.load_champion() if args.warm_start else None
    model=train(examples(args.inputs),args.epochs,args.rate,args.value_rate,args.seed,base)
    output=Path(args.output) if args.output else ROOT/'memory/candidates'/str(time.time_ns())/'model.json'
    pv.save(model,output);print(json.dumps(dict(candidate=str(output),metadata=model['metadata'])),flush=True)


if __name__=='__main__':main()
