#!/usr/bin/env python3
"""Evaluate a policy candidate against frozen unguided MCTS before promotion."""
import argparse
import hashlib
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import advice
import coach as c
import policy_value as pv
import repair_gate

ROOT=Path(__file__).resolve().parent

# The gate's repair check MUST use positions disjoint from the training set.
# tactical-loss-cases.json / recent-loss-cases.json are 8x-weighted TRAINING
# examples (see improve.py inputs and train_policy.py); testing on them would
# reward memorization, not generalization. repair-gate-cases.json is held out
# from training entirely and reserved for this gate only.
GATE_CASES = repair_gate.CASES


def decision(g,seconds,seed,model):
    started=time.monotonic()
    try:
        result=advice.advise(g.history,g.to_move,seconds=seconds,engine='mcts',seed=seed,
                             policy_model=model)
    except c.SearchTimeout:
        # A saturated CPU can make the tactical check hit its internal deadline.
        # Record that as a failed arena decision instead of aborting the report.
        return g.moves(g.to_move)[0],time.monotonic()-started,dict(arena_timeout=True)
    return result['scored'][0][1],time.monotonic()-started,result


def play(history,candidate_side,model,seconds,seed,max_plies=140,opponent=False):
    g=c.Game(history);latencies=[];illegal=0;decision_failures=0
    while g.winner is None and len(g.history)<max_plies:
        use=model if g.to_move==candidate_side else opponent
        move,elapsed,result=decision(g,seconds,seed+len(g.history)*7919,use)
        if result.get('arena_timeout'):decision_failures+=1
        if g.to_move==candidate_side:latencies.append(elapsed)
        if move not in g.moves(g.to_move):illegal+=1;break
        g.apply(move)
    points=.5 if g.winner is None else float(g.winner==candidate_side)
    return dict(points=points,winner=g.winner,plies=len(g.history),illegal=illegal,decision_failures=decision_failures,
                candidate_latencies=latencies)


def holdout_positions(paths,count,seed,player_only=False):
    rows=[];seen=set();rng=random.Random(seed);repair_histories=repair_gate.histories()
    for path in paths:
        with Path(path).open(encoding='utf-8') as src:
            for line in src:
                try:
                    row=json.loads(line)
                    if not isinstance(row,dict):continue
                    hist=c.Game(row['history']).history
                except (ValueError,KeyError,TypeError,json.JSONDecodeError):continue
                key=','.join(hist)
                # Keep the arena and the separate repair gate statistically
                # independent even when common openings recur in archive games.
                if tuple(hist) in repair_histories:continue
                eligible=row.get('player_holdout') if player_only else (row.get('split')=='holdout')
                if eligible and key not in seen:
                    seen.add(key);rows.append(hist)
    rng.shuffle(rows)
    return rows[:count]


def lower95(values):
    if len(values)<2:return 0.0
    return statistics.mean(values)-1.96*statistics.stdev(values)/math.sqrt(len(values))


def regression_results(model, paths):
    """Check the model's live root preference on confirmed loss positions.

    The arena's held-out games measure broad strength.  These cases guard a
    separate promise: a promoted model must retain the tactical repairs made
    after a real bad recommendation.  Use ``search_priors`` because that is
    exactly the policy/value signal passed to live MCTS at the root.
    """
    rows=[]
    for path in paths:
        try:
            cases=json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):
            continue
        if not isinstance(cases,list):
            continue
        for case in cases:
            if not isinstance(case,dict) or not case.get('best'):
                continue
            try:
                game=c.Game(case['history'])
            except (KeyError,TypeError,ValueError):
                continue
            legal=game.moves(game.to_move);expected=case['best']
            if expected not in legal:
                continue
            acceptable=case.get('acceptable',[expected])
            if not isinstance(acceptable,list) or not acceptable or any(move not in legal for move in acceptable):
                continue
            priors=pv.search_priors(model,game,legal)
            chosen=max(priors,key=priors.get) if priors else None
            rows.append(dict(code=case.get('code'),ply=case.get('ply'),
                             expected=expected,acceptable=acceptable,chosen=chosen,
                             correct=chosen in acceptable))
    return rows


def passes_promotion(report):
    return (report['arena_pairs']>=100 and report['score_lower_95']>.5
            and report['illegal_moves']==0 and report['p95_seconds']<5
            and report['decision_failures']==0
            and report['regression_cases']>0
            and report['regression_correct']==report['regression_cases'])


def repair_gate_passes(regressions):
    return bool(regressions) and all(row['correct'] for row in regressions)


def evaluate(model,starts,seconds,seed,baselines=None,workers=2):
    baselines=baselines or [('raw-mcts',False)]
    tasks=[(baseline_name,baseline,i,hist) for baseline_name,baseline in baselines
           for i,hist in enumerate(starts)]
    def run_pair(task):
        baseline_name,baseline,i,hist=task
        g=c.Game(hist);scores=[]
        pair_records=[];pair_latencies=[]
        for candidate_side in (g.to_move,1-g.to_move):
            row=play(hist,candidate_side,model,seconds,seed+i*104729+candidate_side,
                     opponent=baseline)
            row.update(start_hash=hashlib.sha256(','.join(hist).encode()).hexdigest(),
                       candidate_side=candidate_side,baseline=baseline_name)
            pair_records.append(row);scores.append(row['points']);pair_latencies.extend(row['candidate_latencies'])
        return pair_records,sum(scores)/2,pair_latencies
    with ThreadPoolExecutor(max_workers=max(1,min(4,int(workers)))) as pool:
        completed=list(pool.map(run_pair,tasks))
    records=[row for pair_rows,_,_ in completed for row in pair_rows]
    pairs=[pair for _,pair,_ in completed]
    latencies=[latency for _,_,pair_latencies in completed for latency in pair_latencies]
    ordered=sorted(latencies);p95=ordered[min(len(ordered)-1,int(.95*len(ordered)))] if ordered else math.inf
    return dict(arena_pairs=len(pairs),games=len(records),score=sum(r['points'] for r in records)/max(1,len(records)),
                score_lower_95=lower95(pairs),p95_seconds=p95,
                illegal_moves=sum(r['illegal'] for r in records),
                decision_failures=sum(r['decision_failures'] for r in records),records=records)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('candidate');ap.add_argument('targets',nargs='+')
    ap.add_argument('--pairs',type=int,default=100);ap.add_argument('--seconds',type=float,default=1)
    ap.add_argument('--seed',type=int,default=20260907);ap.add_argument('--promote',action='store_true')
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--frozen',nargs='*',default=[],help='older policy models included in the arena')
    ap.add_argument('--player-holdout-only',action='store_true')
    # The gate's repair check MUST use positions disjoint from the training set
    # (see GATE_CASES above for the full rationale).
    ap.add_argument('--regression',nargs='*',default=[GATE_CASES])
    args=ap.parse_args();model=pv.load(args.candidate)
    regressions=regression_results(model,args.regression)
    repair_fields=dict(regression_cases=len(regressions),
                       regression_correct=sum(row['correct'] for row in regressions),
                       regressions=regressions)
    # This is an independent, deterministic candidate check.  Running a costly
    # 200-game arena cannot rehabilitate a model that already fails a required
    # repair.  Keep a complete failed report for auditability and save the CPU
    # for generating the next teacher/candidate.
    if not repair_gate_passes(regressions):
        report=dict(arena_pairs=0,games=0,score=0.0,score_lower_95=0.0,
                    p95_seconds=0.0,illegal_moves=0,decision_failures=0,records=[],
                    candidate_sha256=hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),
                    policy_code_sha256=pv.code_hash(),seconds=args.seconds,workers=args.workers,
                    baselines=['raw-mcts'],promoted=False,passed=False,
                    skipped_arena='held-out repair gate failed',
                    limitation='Frozen unguided MCTS field; no human Elo estimate',**repair_fields)
    else:
        starts=holdout_positions(args.targets,args.pairs,args.seed,args.player_holdout_only)
        if len(starts)<args.pairs:ap.error(f'Need {args.pairs} unique held-out starts; found {len(starts)}')
        baselines=[('raw-mcts',False)]+[(str(path),pv.load(path)) for path in args.frozen]
        report=evaluate(model,starts,args.seconds,args.seed,baselines,args.workers)
        report.update(candidate_sha256=hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),
                      policy_code_sha256=pv.code_hash(),seconds=args.seconds,
                      workers=args.workers,
                      baselines=[name for name,_ in baselines],promoted=False,
                      limitation='Frozen unguided MCTS field; no human Elo estimate',**repair_fields)
    passed=passes_promotion(report)
    report['passed']=passed
    if args.promote and passed:
        model['report']={**{k:v for k,v in report.items() if k!='records'},'promoted':True}
        pv.save(model,pv.CHAMPION);report['promoted']=True
    output=Path(args.candidate).with_name('arena-report.json');output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':main()
