#!/usr/bin/env python3
"""Measure completed steak-account results by coach build and rating band."""
import argparse
import json
import re
import math
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def bucket(rating):
    if not isinstance(rating,(int,float)):return 'unknown'
    lo=int(rating)//100*100;return f'{lo}-{lo+99}'


def interval(wins,games):
    if not games:return [None,None]
    z=1.96;p=wins/games;d=1+z*z/games
    center=(p+z*z/(2*games))/d
    radius=z*math.sqrt(p*(1-p)/games+z*z/(4*games*games))/d
    return [max(0,center-radius),min(1,center+radius)]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prefix',default='steak')
    args=ap.parse_args();builds={}
    for log in (ROOT/'logs').glob('live-advice*.jsonl'):
        for line in log.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line);game=r.get('game')
            except json.JSONDecodeError:continue
            if isinstance(game,str) and re.fullmatch(r'[A-Za-z0-9]{6}',game):
                builds.setdefault(game,r.get('build','unknown'))
    stats=defaultdict(lambda:[0,0,0])
    paths=[]
    for code in builds:
        paths.extend(p for p in (ROOT/'study/archive'/f'{code}.json',ROOT/'study/additional'/f'{code}.json') if p.exists())
    for path in paths:
        try:d=json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError,ValueError):continue
        code=d.get('shareCode') or path.stem
        if code not in builds:continue
        names=[d.get('player1Username',''),d.get('player2Username','')]
        sides=[i for i,n in enumerate(names) if n.casefold().startswith(args.prefix.casefold())]
        if not sides:continue
        side=sides[0];opp=1-side;winner={'1':0,'2':1}.get(str(d.get('winner')))
        rating=d.get('p2Rating' if opp else 'p1Rating')
        key=(builds[code],bucket(rating),'red' if side==0 else 'blue')
        stats[key][2]+=1
        if winner==side:stats[key][0]+=1
        elif winner is not None:stats[key][1]+=1
    rows=[]
    for (build,band,color),(wins,losses,games) in sorted(stats.items()):
        rows.append(dict(build=build,opponent_rating=band,color=color,games=games,
                         wins=wins,losses=losses,winrate=wins/games if games else None,
                         winrate_95=interval(wins,games),target_85_supported=interval(wins,games)[0]>=.85))
    print(json.dumps(dict(rows=rows,limitation='Only games with a local advice trace and archived result'),indent=2))


if __name__=='__main__':main()
