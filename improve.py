#!/usr/bin/env python3
"""Run one or more gated improvement rounds without touching a failed champion."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def run(args):
    print('+',' '.join(map(str,args)),flush=True)
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rounds',type=int,default=1);ap.add_argument('--positions',type=int,default=2000)
    ap.add_argument('--teacher-depth',type=int,default=3);ap.add_argument('--teacher-seconds',type=float,default=8)
    ap.add_argument('--teacher-workers',type=int,default=8)
    ap.add_argument('--arena-pairs',type=int,default=100);ap.add_argument('--arena-seconds',type=float,default=1)
    ap.add_argument('--arena-workers',type=int,default=2)
    ap.add_argument('--league-games',type=int,default=200);ap.add_argument('--league-seconds',type=float,default=.5)
    ap.add_argument('--run-dir',help='resume an existing run directory')
    args=ap.parse_args();run_dir=(Path(args.run_dir).resolve() if args.run_dir else ROOT/'memory'/'runs'/str(time.time_ns()))
    run_dir.mkdir(parents=True,exist_ok=True)
    targets=run_dir/'teacher.jsonl'
    run(['teacher_data.py','--positions',args.positions,'--depth',args.teacher_depth,
         '--seconds',args.teacher_seconds,'--workers',args.teacher_workers,
         '--output',targets.relative_to(ROOT)])
    inputs=[targets,ROOT/'study'/'tactical-loss-cases.json',
            ROOT/'study'/'recent-loss-cases.json'];promoted=[]
    for round_no in range(1,args.rounds+1):
        candidate=run_dir/f'round-{round_no}'/'model.json';candidate.parent.mkdir(exist_ok=True)
        if candidate.exists():
            print(json.dumps(dict(resumed_candidate=str(candidate))),flush=True)
        else:
            command=['train_policy.py',*inputs,'--output',candidate]
            if round_no>1:command.append('--warm-start')
            run(command)
        frozen=[]
        if promoted:frozen=['--frozen',*promoted[-3:]]
        run(['arena.py',candidate,targets,'--pairs',args.arena_pairs,'--seconds',args.arena_seconds,
             '--workers',args.arena_workers,'--promote','--player-holdout-only',*frozen])
        report=json.loads((candidate.parent/'arena-report.json').read_text(encoding='utf-8'))
        if not report.get('promoted'):
            print(json.dumps(dict(stopped='candidate did not pass promotion',round=round_no)),flush=True);break
        promoted.append(str(candidate));league=run_dir/f'round-{round_no}'/'league.jsonl'
        run(['league_selfplay.py','--games',args.league_games,'--seconds',args.league_seconds,
             '--output',league,'--models',*promoted]);inputs.append(league)
    print(json.dumps(dict(run=str(run_dir),promoted_rounds=len(promoted))),flush=True)


if __name__=='__main__':main()
