#!/usr/bin/env python3
"""Small, reproducible paired engine comparison; not an Elo estimate."""
import argparse
import json
from pathlib import Path
import coach as c
import mcts_coach

def run(seconds=1.0):
    games=[]
    for seed in (7,19):
        for mcts_side in (0,1):
            g=c.Game('e2,e8')
            decisions=[]
            while g.winner is None and len(g.history)<160:
                side=g.to_move
                if side==mcts_side:
                    result=mcts_coach.search(g.history,side,time_limit=seconds,rollouts=60000,seed=seed+len(g.history))
                else:
                    result=c.search(g.history,side,depth=2,time_limit=seconds)
                move=result['scored'][0][1]
                decisions.append(dict(ply=len(g.history)+1,engine='mcts' if side==mcts_side else 'python',
                                      move=move,elapsed=result['elapsed'],depth=result['depth'],
                                      simulations=result.get('simulations'),timed_out=result['timed_out']))
                g.apply(move)
            record=dict(seed=seed,mcts_side=mcts_side,winner=g.winner,
                        winner_engine=None if g.winner is None else ('mcts' if g.winner==mcts_side else 'python'),
                        history=g.history,decisions=decisions)
            games.append(record)
            output=dict(seconds_per_move=seconds,max_plies=160,games=games)
            (Path(__file__).resolve().parent/'study/benchmark.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
            print(f'seed {seed}, MCTS side {mcts_side}: {record["winner_engine"]} wins, {len(g.history)} plies',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=1.0)
    run(parser.parse_args().seconds)
