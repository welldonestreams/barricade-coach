#!/usr/bin/env python3
"""Evaluate a policy candidate against frozen unguided MCTS before promotion."""
import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import advice
import coach as c
import policy_value as pv

ROOT=Path(__file__).resolve().parent


def decision(g,seconds,seed,model):
    started=time.monotonic()
    result=advice.advise(g.history,g.to_move,seconds=seconds,engine='mcts',seed=seed,
                         policy_model=model)
    return result['scored'][0][1],time.monotonic()-started,result


def play(history,candidate_side,model,seconds,seed,max_plies=140,opponent=False):
    g=c.Game(history);latencies=[];illegal=0
    while g.winner is None and len(g.history)<max_plies:
        use=model if g.to_move==candidate_side else opponent
        move,elapsed,_=decision(g,seconds,seed+len(g.history)*7919,use)
        if g.to_move==candidate_side:latencies.append(elapsed)
        if move not in g.moves(g.to_move):illegal+=1;break
        g.apply(move)
    points=.5 if g.winner is None else float(g.winner==candidate_side)
    return dict(points=points,winner=g.winner,plies=len(g.history),illegal=illegal,
                candidate_latencies=latencies)


def holdout_positions(paths,count,seed,player_only=False):
    rows=[];seen=set();rng=random.Random(seed)
    for path in paths:
        with Path(path).open(encoding='utf-8') as src:
            for line in src:
                try:row=json.loads(line);hist=c.Game(row['history']).history
                except (ValueError,KeyError,json.JSONDecodeError):continue
                key=','.join(hist)
                eligible=row.get('player_holdout') if player_only else (row.get('split')=='holdout')
                if eligible and key not in seen:
                    seen.add(key);rows.append(hist)
    rng.shuffle(rows)
    return rows[:count]


def lower95(values):
    if len(values)<2:return 0.0
    return statistics.mean(values)-1.96*statistics.stdev(values)/math.sqrt(len(values))


def evaluate(model,starts,seconds,seed,baselines=None):
    baselines=baselines or [('raw-mcts',False)]
    records=[];pairs=[];latencies=[]
    for baseline_name,baseline in baselines:
        for i,hist in enumerate(starts):
            g=c.Game(hist);scores=[]
            for candidate_side in (g.to_move,1-g.to_move):
                row=play(hist,candidate_side,model,seconds,seed+i*104729+candidate_side,
                         opponent=baseline)
                row.update(start_hash=hashlib.sha256(','.join(hist).encode()).hexdigest(),
                           candidate_side=candidate_side,baseline=baseline_name)
                records.append(row);scores.append(row['points']);latencies.extend(row['candidate_latencies'])
            pairs.append(sum(scores)/2)
    ordered=sorted(latencies);p95=ordered[min(len(ordered)-1,int(.95*len(ordered)))] if ordered else math.inf
    return dict(arena_pairs=len(pairs),games=len(records),score=sum(r['points'] for r in records)/max(1,len(records)),
                score_lower_95=lower95(pairs),p95_seconds=p95,
                illegal_moves=sum(r['illegal'] for r in records),records=records)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('candidate');ap.add_argument('targets',nargs='+')
    ap.add_argument('--pairs',type=int,default=100);ap.add_argument('--seconds',type=float,default=1)
    ap.add_argument('--seed',type=int,default=20260907);ap.add_argument('--promote',action='store_true')
    ap.add_argument('--frozen',nargs='*',default=[],help='older policy models included in the arena')
    ap.add_argument('--player-holdout-only',action='store_true')
    args=ap.parse_args();model=pv.load(args.candidate)
    starts=holdout_positions(args.targets,args.pairs,args.seed,args.player_holdout_only)
    if len(starts)<args.pairs:ap.error(f'Need {args.pairs} unique held-out starts; found {len(starts)}')
    baselines=[('raw-mcts',False)]+[(str(path),pv.load(path)) for path in args.frozen]
    report=evaluate(model,starts,args.seconds,args.seed,baselines)
    report.update(candidate_sha256=hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),
                  policy_code_sha256=pv.code_hash(),seconds=args.seconds,
                  baselines=[name for name,_ in baselines],
                  promoted=False,limitation='Frozen unguided MCTS field; no human Elo estimate')
    passed=(report['arena_pairs']>=100 and report['score_lower_95']>.5
            and report['illegal_moves']==0 and report['p95_seconds']<5)
    report['passed']=passed
    if args.promote and passed:
        model['report']={**{k:v for k,v in report.items() if k!='records'},'promoted':True}
        pv.save(model,pv.CHAMPION);report['promoted']=True
    output=Path(args.candidate).with_name('arena-report.json');output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()
