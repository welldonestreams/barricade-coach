#!/usr/bin/env python3
"""Create held-out, deeply searched policy/value targets from legal games."""
import argparse
import hashlib
import json
import math
import random
import time
import os
from concurrent.futures import ProcessPoolExecutor
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
    completed=result.get('depth',0)
    if completed<2:return None  # even depth-2 unfinished -> unusable
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
                teacher=dict(engine='full-width-minimax',depth=completed,
                             nodes=result['nodes'],tt_hits=result.get('tt_hits',0),
                             seconds=result['elapsed'],
                             coach_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest()),
                policy=policy)


def record_task(task):
    path,seed,min_ply,max_ply,depth,seconds=task
    try:return record(Path(path),random.Random(seed),min_ply,max_ply,depth,seconds)
    except (OSError,ValueError,TypeError,KeyError):return None


def gate_game_files():
    """Archive game files that contain a held-out repair-gate position. These
    games must never be sampled by the teacher: a row drawn from one could leak
    the exact gate history into training, letting a candidate pass the gate by
    memorization instead of generalization. Gate cases carry the archive game
    code (their history is a ply-prefix of that game), so exclusion is a direct
    shareCode lookup -- fast even over 80k+ files."""
    try:
        import repair_gate
        gate_rows = repair_gate.rows()
    except ImportError:
        return set()
    codes = {row.get('code') for row in gate_rows if isinstance(row, dict) and row.get('code')}
    if not codes:
        return set()
    bad = set()
    for root in (ROOT / 'study' / 'archive', ROOT / 'study' / 'additional'):
        for path in root.glob('*.json'):
            stem = path.stem
            if stem in codes or stem.casefold() in codes:
                bad.add(path)
    return bad


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--positions',type=int,default=1000)
    ap.add_argument('--depth',type=int,default=3)
    ap.add_argument('--seconds',type=float,default=8)
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--workers',type=int,default=max(1,min(8,(os.cpu_count() or 2)-2)))
    ap.add_argument('--min-ply',type=int,default=8);ap.add_argument('--max-ply',type=int,default=70)
    ap.add_argument('--output',default='memory/teacher/targets.jsonl')
    args=ap.parse_args()
    if args.positions<1 or not 1<=args.depth<=8 or not 0<args.seconds<=60 or not 1<=args.workers<=16:ap.error('Invalid limits')
    files=list((ROOT/'study/archive').glob('*.json'))+list((ROOT/'study/additional').glob('*.json'))
    if not files:ap.error('No validated game files')
    # Exclude whole games that contain a held-out repair-gate position so the
    # teacher can never leak a gate history into training.
    bad = gate_game_files()
    if bad:
        files = [f for f in files if f not in bad]
        print(json.dumps(dict(excluded_gate_games=len(bad))), flush=True)
    if not files:ap.error('All candidate games excluded as repair-gate games')
    rng=random.Random(args.seed);rng.shuffle(files);out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    existing=set()
    if out.exists():
        with out.open(encoding='utf-8') as src:
            for line in src:
                try:
                    row=json.loads(line);existing.add((row['game_hash'],len(row['history'])))
                except (json.JSONDecodeError,KeyError,TypeError):pass
    made=len(existing);started=time.monotonic()
    if made>=args.positions:
        print(json.dumps(dict(targets=made,elapsed=0.0,workers=args.workers,
                              resumed=True,output=str(out))),flush=True)
        return
    tasks=[(str(path),args.seed+i*104729,args.min_ply,args.max_ply,args.depth,args.seconds)
           for i,path in enumerate(files)]
    with out.open('a',encoding='utf-8') as target:
      with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(record_task,tasks,chunksize=1):
            if made>=args.positions:break
            if row and (row['game_hash'],len(row['history'])) not in existing:
                existing.add((row['game_hash'],len(row['history'])))
                target.write(json.dumps(row,separators=(',',':'))+'\n');target.flush();made+=1
                if made%25==0:print(f'{made}/{args.positions} completed targets',flush=True)
    print(json.dumps(dict(targets=made,elapsed=round(time.monotonic()-started,2),
                          workers=args.workers,resumed=bool(existing),output=str(out))),flush=True)


if __name__=='__main__':main()
