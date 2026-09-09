#!/usr/bin/env python3
"""Measured strength: play the live coach against fixed-strength opponents.

This is the honest "does this actually win" harness the improvement plan calls
for. It pits the coach (parallel MCTS, the live engine) against a set of fixed
baselines at equal thinking time, swapping colors, and reports a win rate --
so a change can be shown to help or hurt instead of just accumulating data.

Opponents are specified as 'depth=N' (Python minimax at depth N) or
'mcts=R' (single-tree MCTS at R rollouts). The output path is configurable.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import coach as c
import mcts_coach
import live_coach

ROOT = Path(__file__).resolve().parent


def opponent_move(hist, side, spec, seconds, seed=None):
    if spec.startswith('depth='):
        depth = int(spec.split('=')[1])
        r = c.search(hist, side, depth, seconds)
        return r['scored'][0][1], r
    if spec.startswith('mcts='):
        rollouts = int(spec.split('=')[1])
        r = mcts_coach.search(hist, side, time_limit=seconds, rollouts=rollouts,
                              workers=1, seed=seed)
        return r['scored'][0][1], r
    raise ValueError(f'Unknown opponent spec: {spec}')


def coach_move(hist, side, seconds, seed=None, opening_seconds=1.2):
    g=c.Game(hist)
    if side!=g.to_move: raise ValueError('Wrong side to move')
    params=live_coach.position(g)
    params.update(h=','.join(hist),walls=','.join(sorted(g.walls)),seconds=str(seconds),
                  seed=str(seed) if seed is not None else '',
                  opening_seconds=str(opening_seconds))
    r=live_coach.query(params, record_trace=False)
    return r['top'][0][1], r


def play(hist, coach_side, opponent_spec, seconds, seed, opening_seconds=1.2,
         max_plies=160):
    g = c.Game(hist)
    decisions = []
    while g.winner is None and len(g.history) < max_plies:
        side = g.to_move
        move_seed = (seed + len(g.history) * 7919) & 0xffffffff
        if side == coach_side:
            mv, result = coach_move(g.history, side, seconds, move_seed,
                                    opening_seconds)
            search = result.get('search', {})
            detail = dict(elapsed=round(search.get('elapsed', 0), 4),
                          simulations=search.get('simulations', 0),
                          early=search.get('early', False),
                          budget=search.get('budget'),
                          override=bool(search.get('tactical_override')),
                          fallback=bool(search.get('fallback')))
        else:
            mv, result = opponent_move(g.history, side, opponent_spec, seconds,
                                       move_seed)
            detail = dict(elapsed=round(result.get('elapsed', 0), 4),
                          depth=result.get('depth'),
                          simulations=result.get('simulations', 0),
                          fallback=bool(result.get('fallback')))
        decisions.append(dict(ply=len(g.history)+1,side=side,
                              actor='coach' if side==coach_side else 'opponent',
                              move=mv,seed=move_seed,**detail))
        g.apply(mv)
    return g.winner, len(g.history), g.history, decisions


def run(opponents, seconds, games_per_side, openers, seed0,
        opening_seconds=1.2, coach_sides=(0,1), on_record=None):
    records = []
    for spec in opponents:
        for opener in openers:
            for coach_side in coach_sides:
                for k in range(games_per_side):
                    # A red/blue pair shares its seed so color comparisons do
                    # not accidentally compare unrelated random samples.
                    identity=f'{spec}|{",".join(opener)}|{k}'
                    offset=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4],'big')
                    seed=(seed0+offset)&0xffffffff
                    winner, plies, history, decisions = play(
                        opener,coach_side,spec,seconds,seed,opening_seconds)
                    coach_won = winner == coach_side
                    records.append(dict(opponent=spec, opener=','.join(opener),
                                        coach_side=coach_side, winner=winner,
                                        coach_won=coach_won, plies=plies,seed=seed,
                                        pair_id=identity,
                                        history=history,decisions=decisions))
                    if on_record:on_record(records)
    return records


def summarize(records):
    by_opp = {}
    for r in records:
        by_opp.setdefault(r['opponent'], [0, 0, 0])  # wins, losses, unresolved
        if r['winner'] is None:
            by_opp[r['opponent']][2] += 1
        elif r['coach_won']:
            by_opp[r['opponent']][0] += 1
        else:
            by_opp[r['opponent']][1] += 1
    out = {}
    for spec, (w, l, d) in by_opp.items():
        n = w + l + d
        out[spec] = dict(games=n, wins=w, losses=l, unresolved=d,
                         winrate=round(w / n, 3) if n else 0.0)
    by_side={}
    by_opp_side={}
    for r in records:
        color='red' if r['coach_side']==c.RED else 'blue'
        for bucket,key in ((by_side,color),(by_opp_side,f"{r['opponent']}:{color}")):
            row=bucket.setdefault(key,dict(games=0,wins=0,losses=0,unresolved=0))
            row['games']+=1
            if r['winner'] is None: row['unresolved']+=1
            elif r['coach_won']: row['wins']+=1
            else: row['losses']+=1
    for bucket in (by_side,by_opp_side):
        for row in bucket.values():
            row['winrate']=round(row['wins']/row['games'],3) if row['games'] else 0
            row['winrate_95']=wilson(row['wins'],row['games'])
    paired={}
    for r in records:
        if 'pair_id' not in r:continue
        paired.setdefault(r['pair_id'],{})[r['coach_side']]=r
    pair_summary=dict(pairs=0,both_won=0,both_lost=0,red_only=0,blue_only=0,
                      unresolved=0)
    for pair in paired.values():
        if set(pair)!={c.RED,c.BLUE}:continue
        pair_summary['pairs']+=1
        if pair[c.RED]['winner'] is None or pair[c.BLUE]['winner'] is None:
            pair_summary['unresolved']+=1;continue
        red=pair[c.RED]['coach_won'];blue=pair[c.BLUE]['coach_won']
        key=('both_won' if red and blue else 'red_only' if red else
             'blue_only' if blue else 'both_lost')
        pair_summary[key]+=1
    return dict(by_opponent=out,by_coach_color=by_side,
                by_opponent_and_color=by_opp_side,paired_color=pair_summary)


def wilson(wins, games):
    if games<=0:return [0.0,0.0]
    z=1.959963984540054;p=wins/games
    denominator=1+z*z/games
    center=(p+z*z/(2*games))/denominator
    margin=z*math.sqrt((p*(1-p)+z*z/(4*games))/games)/denominator
    return [round(max(0,center-margin),3),round(min(1,center+margin),3)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--opponents', default='depth=2,depth=3,mcts=5000',
                    help='comma list of "depth=N" or "mcts=R"')
    ap.add_argument('--seconds', type=float, default=2.0, help='thinking time per move')
    ap.add_argument('--games-per-side', type=int, default=2)
    ap.add_argument('--openers', default='e2,e8', help='comma list of opening histories (comma-joined plies)')
    ap.add_argument('--seed',type=int,default=20260908)
    ap.add_argument('--opening-seconds',type=float,default=1.2,
                    help='coach budget during the uncontested center opening')
    ap.add_argument('--output',default='study/match-benchmark-diagnostic.json')
    ap.add_argument('--coach-colors',default='red,blue',
                    help='red, blue, or both (comma separated)')
    args = ap.parse_args()
    opponents = [s.strip() for s in args.opponents.split(',') if s.strip()]
    openers = [[mv for mv in o.split(',') if mv] for o in args.openers.split(';') if o.strip()]
    if not openers:
        openers = [[]]
    color_map={'red':c.RED,'blue':c.BLUE}
    try:coach_sides=tuple(color_map[x.strip().lower()] for x in args.coach_colors.split(',') if x.strip())
    except KeyError:ap.error('--coach-colors accepts only red and blue')
    if not coach_sides or len(set(coach_sides))!=len(coach_sides):ap.error('--coach-colors must contain unique red and/or blue')
    t = time.monotonic()
    if not 0<=args.seed<=0xffffffff:ap.error('--seed must be a 32-bit unsigned integer')
    if not 0<args.opening_seconds<=args.seconds:ap.error('--opening-seconds must be >0 and <= --seconds')
    output_path=Path(args.output)
    if not output_path.is_absolute():output_path=ROOT/output_path
    output_path.parent.mkdir(parents=True,exist_ok=True)
    partial_path=output_path.with_suffix(output_path.suffix+'.partial')
    def document(records,complete):
        return dict(coach_build=live_coach.BUILD,coach_path='live_coach.query',
                    seconds_per_move=args.seconds,opening_seconds=args.opening_seconds,
                    seed=args.seed,games_per_side=args.games_per_side,
                    opponents=opponents,complete=complete,summary=summarize(records),
                    records=records,elapsed_s=round(time.monotonic()-t,1))
    def checkpoint(records):
        partial_path.write_text(json.dumps(document(records,False),indent=2),encoding='utf-8')
        print(f'completed {len(records)} games',flush=True)
    records = run(opponents, args.seconds, args.games_per_side, openers,
                  args.seed,args.opening_seconds,coach_sides,checkpoint)
    output=document(records,True)
    output_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    partial_path.unlink(missing_ok=True)
    print(json.dumps(output['summary'], indent=2), flush=True)
    print(f"elapsed {output['elapsed_s']}s", flush=True)


if __name__ == '__main__':
    main()
