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

# Tactical cross-check: MCTS rollouts are weak in sharp wall-rich positions --
# they converge on pawn moves and miss precise defensive walls. A fast full-width
# minimax (same legal-move model) catches those. We only override when minimax
# sees a move decisively better than MCTS's top pick, and only when walls are on
# the board (MCTS is fine in the open opening/middlegame).
# A measured blue-side loss missed the defensive wall family by four points at
# the old 80-point boundary, then spent the wall two plies too late. Keep the
# safeguard conservative while including that independently reproduced 76-point
# tempo loss.
TACTICAL_GAP_CP = 75.0


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
           opp_name=None, opp_color=None, rollouts=60000, seed=None,
           policy_model=None):
    started=time.monotonic();deadline=started+seconds
    hist=list(history);game=c.Game(hist)
    side=game.to_move if side is None else side
    if engine=='mcts':
        import mcts_coach
        import policy_value
        model=policy_value.load_champion() if policy_model is None else policy_model
        priors=policy_value.search_priors(model,game) if model else None
        # Give broad MCTS the full nominal clock. The old 3s tactical reserve
        # left it only ~1s and caused the measured 5r7nxm positional error.
        budget=max(.001,seconds-.08)
        result=(mcts_coach.search(hist,side,budget,root_priors=priors)
                if rollouts==60000 and seed is None else
                mcts_coach.search(hist,side,budget,rollouts,seed,root_priors=priors))
        result['policy_model']=bool(model)
        result['policy_model_id']=policy_value.model_id(model) if model else None
    elif engine=='python':
        result=c.search(hist,side,depth,max(.001,seconds-.1))
    else:
        raise ValueError('Engine must be python or mcts')
    result['engine']=engine
    if (engine=='mcts' and game.walls and any(game.remaining.values())
            and result.get('scored') and not result.get('fallback')
            and not result.get('forced_loss') and result.get('tactical')!='immediate win'):
        # Pawn recommendations in wall positions are where rollout search has
        # repeatedly missed a precise defensive wall, so retain the full 3s
        # deterministic guard there. A wall recommendation gets a bounded 2s
        # check, enough to catch the measured wasted-wall/pawn-tempo case.
        top_move=result['scored'][0][1]
        tactical_budget=3.0 if len(top_move)==2 else 2.0
        result=_tactical_crosscheck(hist,side,result,tactical_budget)
        result['mcts_budget']=budget
        result['tactical_budget']=tactical_budget
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
        eligible=not result.get('tactical_override') and score==best and game.winner is None and not (engine=='python' and abs(score)>=c.WIN-100)
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


def _tactical_crosscheck(hist, side, mcts_result, seconds):
    """Compare completed selective heuristic scores within the remaining budget.

    This is a selective heuristic safeguard, not an exact wall-game solution.
    A changed recommendation uses one consistent minimax score scale throughout.
    """
    if seconds <= .01:
        return mcts_result
    mcts_moves=[move for _,move in mcts_result.get('scored',[])[:8]]
    game=c.Game(hist)
    if game.remaining[side]==0 and sum(game.remaining.values())<=3:
        # With no walls of our own, look through the opponent's last placements
        # before recommending a backwards-looking pawn move.
        mm=c.candidate_search(hist,side,depth=5,time_limit=seconds,
                              beam=16,root_moves=mcts_moves,wide_root=True)
    elif sum(game.remaining.values())<=6:
        mm=c.search(hist,side,depth=2,time_limit=seconds)
        mm['selective']=False
    else:
        mm = c.candidate_search(hist, side, depth=3, time_limit=seconds,
                                beam=8,root_moves=mcts_moves,
                                preserve_depth2_pawn_margin=TACTICAL_GAP_CP)
    mcts_result['crosscheck_depth'] = mm.get('depth', 0)
    mcts_result['crosscheck_root_candidates'] = mm.get('root_candidates', 0)
    mcts_result['crosscheck_initial_root_candidates'] = mm.get('initial_root_candidates',0)
    mcts_result['crosscheck_staged_root_narrowing'] = mm.get('staged_root_narrowing',False)
    mcts_result['crosscheck_stopped_on_decisive_pawn'] = mm.get('stopped_on_decisive_pawn',False)
    mcts_result['crosscheck_root_slack'] = mm.get('root_slack')
    if not mm.get('scored') or mm.get('depth', 0) < 2:
        return mcts_result
    mm_best = mm['scored'][0]
    if (game.remaining[side]==0 and sum(game.remaining.values())<=3
            and mm.get('depth',0)>=5 and mm_best[0]>=500):
        mcts_result['position_warning']=(
            f"Position looks badly losing; no saving line was found through "
            f"the completed depth-{mm['depth']} check. The move shown is the best resistance found.")
        mcts_result['position_warning_depth']=mm['depth']
    mcts_top = mcts_result['scored'][0][1]
    mm_scores = {m: s for s, m in mm['scored']}
    if mcts_top not in mm_scores or mm_best[1] == mcts_top:
        return mcts_result
    gap = mm_scores[mcts_top] - mm_best[0]
    if gap <= TACTICAL_GAP_CP:
        return mcts_result
    mcts_result['tactical_override'] = dict(
        mcts_top=mcts_top, minimax_best=mm_best[1], gap=round(gap, 1), depth=mm['depth'],
        score_units='heuristic points', proven=False, selective=mm.get('selective',False),
        beam=mm.get('beam'))
    mcts_result['mcts_scored'] = list(mcts_result['scored'])
    mcts_result['scored'] = list(mm['scored'])
    mcts_result['score_units'] = 'heuristic points'
    mcts_result['principal_variation'] = list(mm['principal_variation'])
    kind='selective' if mm.get('selective') else 'full-width'
    mcts_result['crosscheck'] = f"completed {kind} depth-{mm['depth']} heuristic comparison"
    return mcts_result
