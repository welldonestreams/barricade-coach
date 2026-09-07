"""Parallel experiments with a frozen baseline and game-disjoint promotion gate.

python training.py --games 100 --workers 2 --evaluate 100
No experiment changes live advice until held-out paired matches pass the gate.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import coach as c
from learning import position_key

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / 'memory' / 'experiments'


def split_for(history):
    # Split by full normalized game content: renamed files and duplicate share
    # codes cannot put the same game on both sides of the holdout boundary.
    return 'holdout' if int(hashlib.sha256(','.join(history).encode()).hexdigest()[:8],16)%5==0 else 'train'


def dataset():
    top = set(json.loads((ROOT/'study/top-players.json').read_text(encoding='utf-8'))['players'][:100])
    pools = {'train': [], 'holdout': []}
    seen = set()
    for path in sorted((ROOT/'study/archive').glob('*.json')):
        try:
            record=json.loads(path.read_text(encoding='utf-8'))
            if not top.intersection([record.get('player1Username'),record.get('player2Username')]): continue
            hist=c.Game(c.parse_history(record.get('historyCsv',''))).history
            key=','.join(hist)
            if len(hist)<12 or key in seen: continue
            seen.add(key);pools[split_for(hist)].append(hist)
        except (ValueError,OSError,TypeError): continue
    # Also exclude exact seed positions occurring in *any* holdout game.
    held_positions=set()
    for hist in pools['holdout']:
        g=c.Game()
        for mv in hist:
            held_positions.add(position_key(g));g.apply(mv)
    return pools,held_positions


def choose(game, seconds, prior=None):
    result=c.search(game.history,game.to_move,depth=3,time_limit=seconds)
    scored=result['scored']
    if prior and result['depth']>=2:
        cells=prior.get(position_key(game),{})
        # Experience can only break an exact nonterminal tactical tie.
        best=scored[0][0]
        if abs(best)<c.WIN-100:
            tied=[mv for score,mv in scored if score==best]
            def rate(mv):
                won,total=cells.get(mv,[0,0]);return (won+2)/(total+4)
            return max(tied,key=rate)
    return scored[0][1]


def run_match(job):
    hist,seed,seconds,prior,candidate_side,explore=job
    # Leave resources for the foreground coach.
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    rng=random.Random(seed);g=c.Game(hist);rows=[];visits=defaultdict(int)
    for ply in range(100):
        if g.winner is not None: break
        key=position_key(g);visits[key]+=1
        if visits[key]>=3: break
        side=g.to_move
        if explore and ply<3:
            move=rng.choice(g.moves(side))
        else:
            move=choose(g,seconds,prior if side==candidate_side else None)
        rows.append((key,move,side));g.apply(move)
    return dict(winner=g.winner,rows=rows,seed=seed,candidate_side=candidate_side)


def aggregate(matches, held_positions):
    prior={}
    for match in matches:
        if match['winner'] is None: continue  # cutoff/repetition is not a loss
        for key,move,side in match['rows']:
            if key in held_positions: continue
            cell=prior.setdefault(key,{}).setdefault(move,[0,0])
            cell[0]+=int(side==match['winner']);cell[1]+=1
    return prior


def promotion_gate(pair_scores):
    # The two colors from one starting position are correlated. Count each
    # *pair* as one trial, not two independent wins; ties are conservative losses.
    n=len(pair_scores);wins=sum(s>1 for s in pair_scores)
    if not n: return False,0
    z=1.96;p=wins/n
    low=(p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
    return n>=100 and low>.5,low


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--games',type=int,default=100)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--evaluate',type=int,default=100,help='paired held-out positions')
    ap.add_argument('--seconds',type=float,default=.25)
    ap.add_argument('--seed',type=int,default=20260906)
    args=ap.parse_args()
    if min(args.games,args.workers,args.evaluate)<1 or not 0<args.seconds<=4: ap.error('Positive counts; seconds at most 4')
    workers=min(args.workers,max(1,(os.cpu_count() or 2)-2),4)
    pools,held=dataset()
    if not all(pools.values()): ap.error('Need validated train and holdout games')
    rng=random.Random(args.seed)
    def sample(pool):
        hist=rng.choice(pool);cut=rng.randrange(6,min(40,len(hist)-1));return hist[:cut]
    jobs=[(sample(pools['train']),rng.randrange(2**32),args.seconds,None,None,True) for _ in range(args.games)]
    EXPERIMENTS.mkdir(parents=True,exist_ok=True)
    run=EXPERIMENTS/str(time.time_ns());run.mkdir()
    baseline_hash=hashlib.sha256((ROOT/'coach.py').read_bytes()).hexdigest()
    print(f'Training {args.games} games using {workers} workers; {len(pools["train"])} train / {len(pools["holdout"])} holdout games',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        matches=list(pool.map(run_match,jobs))
        prior=aggregate(matches,held)
        (run/'candidate.json').write_text(json.dumps(prior),encoding='utf-8')
        # Unique held-out seed positions, with exactly the same board/budget in
        # both colors. They never write training outcomes or learned labels.
        eval_jobs=[];seen=set()
        for _ in range(args.evaluate*30):
            hist=sample(pools['holdout']);key=position_key(c.Game(hist))
            if key in seen: continue
            seen.add(key);seed=rng.randrange(2**32)
            eval_jobs.extend((hist,seed,args.seconds,prior,side,False) for side in (0,1))
            if len(seen)>=args.evaluate:break
        results=list(pool.map(run_match,eval_jobs))
    scores=[.5 if r['winner'] is None else float(r['winner']==r['candidate_side']) for r in results]
    pairs=[sum(scores[i:i+2]) for i in range(0,len(scores),2)]
    passed,low=promotion_gate(pairs)
    passed=passed and baseline_hash==hashlib.sha256((ROOT/'coach.py').read_bytes()).hexdigest()
    report=dict(engine="python",seed=args.seed,workers=workers,training_games=len(matches),positions=len(prior),
                holdout_pairs=len(pairs),candidate_points=sum(scores),matches=len(scores),
                pair_win_lower_95=low,promoted=passed,baseline_sha256=baseline_hash,
                limitation='One fixed tactical baseline; no leaderboard Elo estimate')
    (run/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if passed:
        champion=ROOT/'memory/champion.json';tmp=champion.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(report=report,positions=prior)),encoding='utf-8');tmp.replace(champion)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
