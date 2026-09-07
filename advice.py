"""Tactical advice with bounded, exact-position opponent tie-breaking.
Self-play and heuristic deltas are experimental, not causal evidence.
"""
import coach as c
import learning
import json
from functools import lru_cache

@lru_cache(maxsize=1)
def _champion(modified):
    data=json.loads((learning.MEMORY/'champion.json').read_text(encoding='utf-8'))
    return data.get('positions',{}) if data.get('report',{}).get('promoted') else {}

def approved_prior(game):
    try:
        path=learning.MEMORY/'champion.json'
        return _champion(path.stat().st_mtime_ns).get(learning.position_key(game),{})
    except (OSError, ValueError):
        return {}


def advise(history, side=None, depth=2, seconds=5.0, engine='python',
           opp_name=None, opp_color=None, rollouts=60000, seed=None):
    """Keep tactical scores authoritative. Empirical replies break exact ties only.

    Legacy mixed self-play/eval-delta tables are not independent evidence and
    must not influence live recommendations. MCTS visit counts are not tempos.
    """
    import time
    start = time.monotonic()
    hist = list(history)
    game = c.Game(hist)
    side = game.to_move if side is None else side
    if engine == 'mcts':
        import mcts_coach
        result = (mcts_coach.search(hist, side, seconds) if rollouts == 60000 and seed is None
                  else mcts_coach.search(hist, side, seconds, rollouts, seed))
    else:
        result = c.search(hist, side, depth, max(.001, seconds - .15))
    result['engine'] = engine
    tactical = list(result.get('scored') or [])
    result['tactical'] = tactical
    result['blend'] = []
    if not tactical:
        return result
    if engine == 'python' and result.get('depth', 0) < 2:
        return result
    if engine == 'mcts' and result.get('simulations', 0) < 200:
        return result
    best_score = tactical[0][0]
    prior=approved_prior(game)
    rows = []
    for score, move in tactical:
        preference, evidence = 0, None
        if score == best_score and abs(score) < c.WIN - 100 and opp_name and time.monotonic()-start < seconds:
            child = game.copy()
            child._play(move)
            insight = learning.opponent_insight(opp_name, child.history,
                                                'red' if child.to_move == c.RED else 'blue')
            legal = set(child.moves(child.to_move)) if insight and insight.get('known') else set()
            counts = [(mv, count) for mv, count in (insight or {}).get('top', []) if mv in legal]
            total = sum(n for _, n in counts)
            if total >= 3:
                values = []
                for reply, count in counts:
                    after = child.copy(); after._play(reply)
                    values.append((after.eval_side(side), count))
                preference = sum(v*n for v,n in values)/total
                evidence = dict(samples=total, replies=counts)
        won,total=prior.get(move,[0,0])
        learned=-(won+2)/(total+4) if score==best_score else 0
        rows.append(dict(learned=learned, move=move, tactical=score, own_exp=0, opponent=0,
                         final=score, preference=preference, evidence=evidence))
    rows.sort(key=lambda r: (r['tactical'], r['learned'], r['preference']))
    result['scored'] = [(r['tactical'], r['move']) for r in rows]
    result['blend'] = rows
    if result['scored'][0][1] != tactical[0][1]:
        # The original principal variation belongs to a different root.
        result['principal_variation'] = [result['scored'][0][1]]
    result['elapsed'] = time.monotonic()-start
    return result
