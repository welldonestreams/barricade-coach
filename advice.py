"""Preserve engine rankings; independently validated evidence breaks exact ties.

MCTS visits, outcome frequencies and heuristic route scores have different units.
Raw self-play and repeated static evaluations are research data, not live priors.
"""
from functools import lru_cache
import hashlib
import json
import time
from pathlib import Path
import coach as c
import learning


@lru_cache(maxsize=1)
def _approved(path, modified):
    data=json.loads(path.read_text(encoding='utf-8'))
    report=data.get('report',{})
    if not (report.get('promoted') is True and report.get('holdout_pairs',0)>=100
            and report.get('pair_win_lower_95',0)>.5 and report.get('engine')=='python'):
        return {}
    digest=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest()
    if report.get('baseline_sha256')!=digest:
        return {}
    return data.get('positions',{})


def approved_prior(game, engine):
    # training.py currently measures only this Python tie-break policy. An MCTS
    # experiment needs its own held-out evaluation before it can supply a prior.
    if engine!='python': return {}
    try:
        path=learning.MEMORY/'champion.json'
        return _approved(path,path.stat().st_mtime_ns).get(learning.position_key(game),{})
    except (OSError,ValueError,TypeError,KeyError):
        return {}


def advise(history, side=None, depth=2, seconds=5.0, engine='python',
           opp_name=None, opp_color=None, rollouts=60000, seed=None):
    started=time.monotonic();deadline=started+seconds
    hist=list(history);game=c.Game(hist)
    side=game.to_move if side is None else side
    if engine=='mcts':
        import mcts_coach
        result=(mcts_coach.search(hist,side,seconds) if rollouts==60000 and seed is None
                else mcts_coach.search(hist,side,seconds,rollouts,seed))
    elif engine=='python':
        result=c.search(hist,side,depth,max(.001,seconds-.1))
    else:
        raise ValueError('Engine must be python or mcts')
    result['engine']=engine
    tactical=list(result.get('scored') or [])
    result.update(tactical=tactical,blend=[],learning_policy='validated exact ties only')
    if not tactical or result.get('fallback') or (engine=='python' and result.get('depth',0)<2):
        return result
    best=tactical[0][0]
    prior=approved_prior(game,engine) if time.monotonic()<deadline else {}
    rows=[]
    for score,move in tactical:
        preference=0; evidence=None; learned=0
        # Never reorder different tactical scores, including proven outcomes.
        eligible=score==best and game.winner is None and not (engine=='python' and abs(score)>=c.WIN-100)
        if eligible and move in prior:
            won,n=prior[move]
            if 0<=won<=n:
                learned=-(won+2)/(n+4)
        if eligible and opp_name and time.monotonic()<deadline:
            child=game.copy();child._play(move)
            if child.winner is None:
                insight=learning.opponent_insight(opp_name,child.history,
                                  'red' if child.to_move==c.RED else 'blue')
                legal=set(child.moves(child.to_move)) if insight and insight.get('known') else set()
                counts=[(mv,n) for mv,n in (insight or {}).get('top',[]) if mv in legal and n>0]
                total=sum(n for _,n in counts)
                if total>=3:
                    values=[]
                    for reply,n in counts:
                        after=child.copy();after._play(reply);values.append((after.eval_side(side),n))
                    preference=sum(v*n for v,n in values)/total
                    evidence=dict(samples=total,replies=counts)
        rows.append(dict(move=move,tactical=score,final=score,own_exp=0,opponent=0,
                         learned=learned,preference=preference,evidence=evidence))
    rows.sort(key=lambda r:(r['tactical'],r['learned'],r['preference']))
    result['scored']=[(r['tactical'],r['move']) for r in rows];result['blend']=rows
    if result['scored'][0][1]!=tactical[0][1]:
        result['principal_variation']=[result['scored'][0][1]]
    result['elapsed']=time.monotonic()-started
    return result
