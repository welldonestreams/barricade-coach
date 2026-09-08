"""Dependency-free sparse policy/value model for guided Barricade search.

Candidates live under memory/candidates. Only memory/policy-champion.json is
eligible for live use, and only after arena.py writes a passing report.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from functools import lru_cache

import coach as c

ROOT=Path(__file__).resolve().parent
MEMORY=ROOT/'memory'
CHAMPION=MEMORY/'policy-champion.json'
SCHEMA=1


def code_hash():
    # Hash both model implementations so any code change invalidates a stale
    # champion of either schema.
    import nn_model
    return hashlib.sha256(
        Path(__file__).read_bytes() + nn_model.code_hash().encode()).hexdigest()


def model_id(model):
    if isinstance(model, dict) and model.get('schema') == 2:
        import nn_model
        return nn_model.model_id(model)
    body=json.dumps({'policy':model.get('policy',{}),'value':model.get('value',{})},
                    sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(body).hexdigest()[:12]


def _flip_rank(rank, side, wall=False):
    if side==c.RED:
        return rank
    return (9-rank) if wall else (10-rank)


def normalize_move(move, side):
    if side==c.RED:
        return move
    if len(move)==2:
        return move[0]+str(_flip_rank(int(move[1]),side))
    return move[:2]+str(_flip_rank(int(move[2]),side,wall=True))


def state_features(g, side=None):
    side=g.to_move if side is None else side
    mine=g.pawns[side];opp=g.pawns[1-side]
    my_rank=mine[1]+1 if side==c.RED else 9-mine[1]
    op_rank=opp[1]+1 if side==c.RED else 9-opp[1]
    my_route=g.race_distance(side);op_route=g.race_distance(1-side)
    delta=max(-12,min(12,my_route-op_route))
    # Signed "who is ahead" signals, distinct from path-length delta: raw pawn
    # advancement and wall-reserve lead. These let the value model learn
    # "I'm ahead -> conserve walls" vs "I'm behind -> must spend".
    progress_lead=max(-8,min(8,my_rank-op_rank))
    wall_lead=max(-10,min(10,g.remaining[side]-g.remaining[1-side]))
    stage=min(4,len(g.walls)//5)
    my_res=c.path_resilience(frozenset(g.walls),mine,c.GOALS[side])
    op_res=c.path_resilience(frozenset(g.walls),opp,c.GOALS[1-side])
    out=['bias',f'my_sq:{c.LETTERS[mine[0]]}{my_rank}',
         f'op_sq:{c.LETTERS[opp[0]]}{op_rank}',f'my_file:{mine[0]}',
         f'op_file:{opp[0]}',f'file_gap:{abs(mine[0]-opp[0])}',
         f'rank_gap:{abs(my_rank-op_rank)}',f'my_left:{g.remaining[side]}',
         f'op_left:{g.remaining[1-side]}',f'stage:{stage}',f'route_delta:{delta}',
         f'progress_lead:{progress_lead}',f'wall_lead:{wall_lead}',
         f'my_resilience:{my_res}',f'op_resilience:{op_res}',
         f'my_route_options:{g.route_options(side)}',
         f'op_route_options:{g.route_options(1-side)}']
    for wall in sorted(g.walls):
        out.append('wall:'+normalize_move(wall,side))
    return out


def action_features(g, move, side=None):
    side=g.to_move if side is None else side
    opp=g.pawns[1-side]
    norm=normalize_move(move,side)
    out=[f'a:{norm}',f'type:{"pawn" if len(move)==2 else move[0]}',
         f'afile:{norm[-2] if len(norm)==3 else norm[0]}',f'arank:{norm[-1]}']
    child=g.copy();child._play(move)
    before_me=g.race_distance(side);before_op=g.race_distance(1-side)
    after_me=child.race_distance(side);after_op=child.race_distance(1-side)
    out.extend((f'my_route_change:{max(-4,min(4,after_me-before_me))}',
                f'op_route_change:{max(-4,min(4,after_op-before_op))}'))
    if len(move)==3:
        net=(after_op-before_op)-(after_me-before_me)
        # Wall-placement quality: does it cut the opponent's number of distinct
        # shortest-path routes (path flexibility), not just lengthen one route?
        # This is the defensive-wall concept top players use and what the coach
        # was blind to (game 76d51f).
        before_res=c.path_resilience(frozenset(g.walls),opp,c.GOALS[1-side])
        after_res=c.path_resilience(frozenset(child.walls),child.pawns[1-side],c.GOALS[1-side])
        res_drop=before_res-after_res
        out.extend((f'wall_net:{max(-4,min(4,net))}',
                    f'wall_opp_resilience_change:{max(-2,min(2,res_drop))}',
                    'wall_changes_route' if net else 'wall_no_immediate_gain'))
    if child.winner==side: out.append('wins_now')
    return out


def policy_keys(g, move):
    state=state_features(g)
    action=action_features(g,move)
    keys=['ab:'+a for a in action]
    # Cross compact contextual features with coarse action features. Exact wall
    # layout features stay in the value model to keep policy artifacts bounded.
    coarse=[x for x in state if not x.startswith('wall:')]
    keys.extend('x:'+s+'|'+a for s in coarse for a in action[1:])
    return keys


def _dot(weights, keys):
    return sum(weights.get(k,0.0) for k in keys)


def policy_logits(model, g, legal=None):
    if isinstance(model, dict) and model.get('schema') == 2:
        import nn_model
        return nn_model.policy_logits(model, g, legal)
    legal=g.moves(g.to_move) if legal is None else legal
    weights=model.get('policy',{})
    return {move:_dot(weights,policy_keys(g,move)) for move in legal}


def policy_priors(model, g, legal=None, temperature=1.0):
    logits=policy_logits(model,g,legal)
    if not logits:return {}
    temperature=max(.05,float(temperature));peak=max(logits.values())
    raw={m:math.exp(max(-40,min(40,(v-peak)/temperature))) for m,v in logits.items()}
    total=sum(raw.values())
    return {m:v/total for m,v in raw.items()}


def search_priors(model,g,legal=None,temperature=1.0,value_scale=.75):
    """Combine policy preference with the opponent-perspective child value."""
    legal=g.moves(g.to_move) if legal is None else legal
    logits=policy_logits(model,g,legal)
    for move in legal:
        child=g.copy();child._play(move)
        terminal=1.0 if child.winner==g.to_move else -value(model,child)
        logits[move]+=value_scale*terminal
    if not logits:return {}
    peak=max(logits.values());temperature=max(.05,float(temperature))
    raw={m:math.exp(max(-40,min(40,(score-peak)/temperature))) for m,score in logits.items()}
    total=sum(raw.values());return {m:v/total for m,v in raw.items()}


def value(model,g,side=None):
    if isinstance(model, dict) and model.get('schema') == 2:
        import nn_model
        return nn_model.value(model, g, side)
    side=g.to_move if side is None else side
    return math.tanh(_dot(model.get('value',{}),state_features(g,side)))


def new_model(seed_from=None):
    if seed_from:
        return dict(schema=SCHEMA,policy=dict(seed_from.get('policy',{})),
                    value=dict(seed_from.get('value',{})),metadata={})
    return dict(schema=SCHEMA,policy={},value={},metadata={})


def validate(model):
    if isinstance(model, dict) and model.get('schema') == 2:
        import nn_model
        return nn_model.validate(model)
    if not isinstance(model,dict) or model.get('schema')!=SCHEMA:
        raise ValueError('Unsupported policy model')
    for name in ('policy','value'):
        weights=model.get(name)
        if not isinstance(weights,dict) or len(weights)>1_000_000:
            raise ValueError('Invalid model weights')
        if any(not isinstance(k,str) or not isinstance(v,(int,float)) or not math.isfinite(v)
               for k,v in weights.items()):
            raise ValueError('Invalid model coefficient')
    return model


def load(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data, dict) and data.get('schema') == 2:
        import nn_model
        return nn_model.load(path)
    return validate(data)


@lru_cache(maxsize=2)
def _load_champion(path,modified):
    model=load(path);report=model.get('report',{})
    if (report.get('promoted') is True and report.get('arena_pairs',0)>=100
            and report.get('score_lower_95',0)>.5
            and report.get('policy_code_sha256')==code_hash()):
        return model
    return None


def load_champion():
    try:
        return _load_champion(CHAMPION, CHAMPION.stat().st_mtime_ns)
    except (OSError, ValueError, TypeError):
        return None


def load_champion_any():
    """Return the champion regardless of schema (1 sparse or 2 neural), so the
    live advice path can use whichever model type earned promotion. Falls back
    to schema-1 load_champion() semantics."""
    model = load_champion()
    if model is None:
        # A schema-2 champion is stored identically; try the neural loader.
        try:
            import nn_model
            cand = nn_model.load(CHAMPION)
            report = cand.get('report', {})
            if (report.get('promoted') is True and report.get('arena_pairs', 0) >= 100
                    and report.get('score_lower_95', 0) > .5
                    and report.get('policy_code_sha256') == nn_model.code_hash()):
                return cand
        except (OSError, ValueError, TypeError, ImportError):
            return None
    return model


def save(model,path):
    if isinstance(model, dict) and model.get('schema') == 2:
        import nn_model
        return nn_model.save(model, path)
    validate(model);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(model,separators=(',',':')),encoding='utf-8')
    tmp.replace(path)
