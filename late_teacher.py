#!/usr/bin/env python3
"""Create color-balanced deep labels for one-sided-wall late games."""
import argparse
import hashlib
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import coach as c
import repair_gate

ROOT=Path(__file__).resolve().parent


def canonical(g):
    return (g.pawns[0],g.pawns[1],tuple(sorted(g.walls)),
            g.remaining[0],g.remaining[1],g.to_move)


def collect_positions(limit,scan_games,seed,min_ply,max_walls):
    rng=random.Random(seed);paths=list((ROOT/'study/archive').glob('*.json'));rng.shuffle(paths)
    gate_codes={str(row.get('code','')).casefold() for row in repair_gate.rows()}
    gate_hists=repair_gate.histories();gate_positions=repair_gate.positions()
    quota={c.RED:(limit+1)//2,c.BLUE:limit//2}
    found={c.RED:[],c.BLUE:[]};seen=set()
    for path in paths[:scan_games]:
        if path.stem.casefold() in gate_codes:continue
        try:
            data=json.loads(path.read_text(encoding='utf-8-sig'))
            history=c.Game(c.parse_history(data.get('historyCsv',''))).history
        except (OSError,ValueError,TypeError):continue
        g=c.Game();choices={c.RED:[],c.BLUE:[]}
        for move in history:
            side=g.to_move
            if (len(g.history)>=min_ply and g.remaining[side]==0
                    and 0<sum(g.remaining.values())<=max_walls
                    and tuple(g.history) not in gate_hists
                    and canonical(g) not in gate_positions):
                choices[side].append(g.history[:])
            try:g.apply(move)
            except ValueError:break
        for side in (c.RED,c.BLUE):
            if len(found[side])>=quota[side] or not choices[side]:continue
            hist=rng.choice(choices[side]);key=canonical(c.Game(hist))
            if key not in seen:seen.add(key);found[side].append((path.stem,hist))
        if all(len(found[s])>=quota[s] for s in quota):break
    rows=[]
    for side in (c.RED,c.BLUE):rows.extend(found[side][:quota[side]])
    rng.shuffle(rows)
    return rows


def solve(task):
    code,hist,depth,min_depth,seconds=task
    try:
        g=c.Game(hist);legal=g.moves(g.to_move)
        result=c.candidate_search(hist,g.to_move,depth=depth,time_limit=seconds,
                                  beam=16,root_moves=legal,wide_root=True,
                                  preserve_depth2_pawn_margin=None)
    except (ValueError,c.SearchTimeout) as error:return None,str(error)
    if result.get('depth',0)<min_depth or not result.get('scored'):
        return None,f'{code}: completed depth {result.get("depth",0)}'
    scores=result['scored'];best=scores[0][0];kept=scores[:min(12,len(scores))]
    spread=kept[-1][0]-best if kept else 0.0;temp=max(50.0,min(250.0,spread*.5+40.0))
    raw={move:math.exp(-min(12.0,(score-best)/temp)) for score,move in kept}
    total=sum(raw.values());policy={move:value/total for move,value in raw.items()}
    row=dict(game_hash=f'late:{code}:{len(hist)}',split='train',player_holdout=False,
             history=hist,side=g.to_move,value=math.tanh(-best/250.0),source='teacher',
             weight=3.0,policy=policy,
             teacher=dict(engine='one-sided-wall-selective-minimax',
                          depth=result.get('depth'),target_depth=depth,
                          nodes=result.get('nodes',0),tt_hits=result.get('tt_hits',0),
                          seconds=result.get('elapsed',0),
                          coach_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest()))
    return row,None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--positions',type=int,default=500);ap.add_argument('--scan-games',type=int,default=20000)
    ap.add_argument('--seconds',type=float,default=20.0);ap.add_argument('--depth',type=int,default=6)
    ap.add_argument('--min-depth',type=int,default=5);ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--max-opponent-walls',type=int,default=4);ap.add_argument('--min-ply',type=int,default=20)
    ap.add_argument('--seed',type=int,default=20260910);ap.add_argument('--output',required=True)
    args=ap.parse_args();started=time.monotonic()
    positions=collect_positions(args.positions,args.scan_games,args.seed,args.min_ply,
                                args.max_opponent_walls)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    tasks=[(code,hist,args.depth,args.min_depth,args.seconds) for code,hist in positions]
    made=[];skipped=[]
    with output.open('w',encoding='utf-8') as dst:
      with ProcessPoolExecutor(max_workers=max(1,min(16,args.workers))) as pool:
        for row,error in pool.map(solve,tasks):
            if row is None:skipped.append(error);continue
            dst.write(json.dumps(row,separators=(',',':'))+'\n');dst.flush();made.append(row)
            if len(made)%25==0:print(f'{len(made)}/{len(positions)} late targets',flush=True)
    colors={c.RED:0,c.BLUE:0}
    for row in made:colors[row['side']]+=1
    meta=dict(candidates=len(positions),targets=len(made),red=colors[c.RED],blue=colors[c.BLUE],
              skipped=len(skipped),depth=args.depth,min_depth=args.min_depth,
              seconds=args.seconds,elapsed=round(time.monotonic()-started,2))
    output.with_suffix(output.suffix+'.meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
