#!/usr/bin/env python3
"""Create held-out, deeply searched policy/value targets from legal games."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import coach as c

ROOT=Path(__file__).resolve().parent


def split_for(history):
    digest=hashlib.sha256(','.join(history).encode()).digest()[0]
    return 'holdout' if digest<51 else 'train'


def heldout_player(name):
    return bool(name) and hashlib.sha256(name.casefold().encode()).digest()[0]<51


def record(path,rng,min_ply,max_ply,depth,seconds):
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    history=c.Game(c.parse_history(data.get('historyCsv',''))).history
    if len(history)<=min_ply:return None
    ply=rng.randrange(min_ply,min(max_ply,len(history)-1)+1)
    g=c.Game(history[:ply]);result=c.search(g.history,g.to_move,depth,seconds)
    if result.get('depth',0)<depth:return None
    scores=result['scored'];best=scores[0][0]
    # A rank distribution is stable across heuristic scale changes and teaches
    # alternatives, unlike copying one human move or eventual outcome alone.
    kept=scores[:min(12,len(scores))]
    raw={move:math.exp(-min(12,(score-best)/100)) for score,move in kept}
    total=sum(raw.values());policy={m:v/total for m,v in raw.items()}
    winner={'1':c.RED,'2':c.BLUE,1:c.RED,2:c.BLUE}.get(data.get('winner'))
    players=[data.get('player1Username'),data.get('player2Username')]
    # The deep search's own score is a graded position value, far stronger than
    # the eventual game winner alone. Lower-is-better -> tanh(-best/250) in
    # [-1,1], matching the value head's convention (+1 = side-to-move winning).
    value=math.tanh(-best/250.0)
    return dict(game_hash=hashlib.sha256(','.join(history).encode()).hexdigest(),
                split=split_for(history),history=g.history,side=g.to_move,winner=winner,
                value=value,
                players=players,player_holdout=any(heldout_player(name) for name in players),
                teacher=dict(engine='full-width-minimax',depth=result['depth'],
                             nodes=result['nodes'],tt_hits=result.get('tt_hits',0),
                             seconds=result['elapsed']),policy=policy)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--positions',type=int,default=1000)
    ap.add_argument('--depth',type=int,default=3)
    ap.add_argument('--seconds',type=float,default=8)
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--min-ply',type=int,default=8);ap.add_argument('--max-ply',type=int,default=70)
    ap.add_argument('--output',default='memory/teacher/targets.jsonl')
    args=ap.parse_args()
    if args.positions<1 or not 1<=args.depth<=8 or not 0<args.seconds<=60:ap.error('Invalid limits')
    files=list((ROOT/'study/archive').glob('*.json'))+list((ROOT/'study/additional').glob('*.json'))
    if not files:ap.error('No validated game files')
    rng=random.Random(args.seed);rng.shuffle(files);out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    made=0;started=time.monotonic()
    with out.open('a',encoding='utf-8') as target:
        for path in files:
            if made>=args.positions:break
            try:row=record(path,rng,args.min_ply,args.max_ply,args.depth,args.seconds)
            except (OSError,ValueError,TypeError,KeyError):continue
            if row:
                target.write(json.dumps(row,separators=(',',':'))+'\n');target.flush();made+=1
                if made%25==0:print(f'{made}/{args.positions} completed targets',flush=True)
    print(json.dumps(dict(targets=made,elapsed=round(time.monotonic()-started,2),output=str(out))),flush=True)


if __name__=='__main__':main()
