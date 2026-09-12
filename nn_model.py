"""Schema-2 neural policy/value model for barricade-coach.

A small dependency-light (numpy-only) feed-forward network that replaces the
sparse linear schema-1 model. It plugs into the SAME policy_value interface
(value / policy_logits), so the arena, promotion gate, and load_champion() all
work unchanged. numpy is imported lazily so the running schema-1 live server
never needs it until a schema-2 champion is actually promoted.

Model shape (schema 2):
    policy:  dict of flat weight/bias lists + dims  (state+action -> h -> 1)
    value:   dict of flat weight/bias lists + dims  (state -> h -> 1)
    metadata: {dims: {state, action, hidden}, kind: 'mlp', ...}

All board features are color-normalized to the side-to-move's perspective so a
single set of weights serves both colors.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import coach as c

SCHEMA = 2
HIDDEN = 256
MAXW = 4_000_000
ACTION_SLOTS = 81 + 128


def _np():
    import numpy  # lazy: schema-1 live server never needs it
    return numpy


# ----------------------------------------------------------------------------
# Feature extraction (dense, color-normalized to `side`)
# ----------------------------------------------------------------------------

def _advance(rank, side):
    """Progress toward goal, 0..8, from `side`'s perspective."""
    return rank if side == c.RED else 8 - rank


def state_features(g, side=None):
    """Dense float list; length = dims['state']."""
    side = g.to_move if side is None else side
    mine = g.pawns[side]
    opp = g.pawns[1 - side]
    my_adv = _advance(mine[1], side)
    op_adv = _advance(opp[1], side)
    my_route = g.race_distance(side)
    op_route = g.race_distance(1 - side)
    delta = max(-12, min(12, my_route - op_route))
    my_res = c.path_resilience(frozenset(g.walls), mine, c.GOALS[side])
    op_res = c.path_resilience(frozenset(g.walls), opp, c.GOALS[1 - side])
    f = [
        # pawn positions (normalized)
        (mine[0] - 4.0) / 4.0, (my_adv - 4.0) / 4.0,
        (opp[0] - 4.0) / 4.0, (op_adv - 4.0) / 4.0,
        # relative geometry
        (mine[0] - opp[0]) / 8.0, (my_adv - op_adv) / 8.0,
        # wall reserves
        g.remaining[side] / 10.0, g.remaining[1 - side] / 10.0,
        # route / flexibility
        min(24.0, float(my_route)) / 24.0,
        min(24.0, float(op_route)) / 24.0, delta / 12.0,
        my_res / float(c.RESILIENCE_CAP), op_res / float(c.RESILIENCE_CAP),
        g.route_options(side) / float(c.RESILIENCE_CAP),
        g.route_options(1 - side) / float(c.RESILIENCE_CAP),
        # stage
        min(4, len(g.walls) // 5) / 4.0,
    ]
    # wall layout, normalized: 64 h-slots + 64 v-slots
    wall_bits = _wall_bits(g.walls, side)
    return f + wall_bits


def _wall_bits(walls, side):
    bits = [0.0] * 128
    for w in walls:
        horiz = w[0] == 'h'
        file_idx = c.FILES[w[1]]
        rank = int(w[2])
        if side == c.BLUE:
            rank = 9 - rank  # mirror toward blue's goal
        # h walls: file 0..7, rank 1..8 ; v walls: file 0..7, rank 1..8
        if 0 <= file_idx <= 7 and 1 <= rank <= 8:
            idx = file_idx * 8 + (rank - 1)
            if not horiz:
                idx += 64
            bits[idx] = 1.0
    return bits


def _action_bits(move, side):
    """One exact, color-normalized destination slot for every legal action."""
    bits = [0.0] * ACTION_SLOTS
    if len(move) == 3:
        file_idx = c.FILES[move[1]]
        rank = int(move[2])
        if side == c.BLUE:
            rank = 9 - rank
        slot = file_idx * 8 + (rank - 1)
        if move[0] == 'v':
            slot += 64
        bits[81 + slot] = 1.0
    else:
        file_idx = c.FILES[move[0]]
        rank = int(move[1]) - 1
        bits[_advance(rank, side) * 9 + file_idx] = 1.0
    return bits


def action_context(g,side=None):
    """Metrics shared by every candidate action at one root position."""
    side=g.to_move if side is None else side
    return dict(side=side,my_route=g.race_distance(side),
                op_route=g.race_distance(1-side))


def action_features(g, move, side=None, context=None):
    """Dense float list for a candidate move; length = dims['action']."""
    side = g.to_move if side is None else side
    context=action_context(g,side) if context is None else context
    is_wall = len(move) == 3
    if is_wall:
        file_idx = c.FILES[move[1]]
        rank = int(move[2])
        if side == c.BLUE:
            rank = 9 - rank
        typ = [1.0, 1.0 if move[0] == 'v' else 0.0]
        pos = [(file_idx - 3.5) / 3.5, (rank - 4.5) / 3.5]
    else:
        file_idx = c.FILES[move[0]]
        rank = int(move[1]) - 1  # 0-indexed row
        adv = _advance(rank, side)
        typ = [0.0, 0.0]
        pos = [(file_idx - 4.0) / 4.0, (adv - 4.0) / 4.0]
    child = g.copy()
    child._play(move)
    before_me=context['my_route'];before_op=context['op_route']
    after_me = child.race_distance(side)
    after_op = child.race_distance(1 - side)
    my_change = max(-4, min(4, after_me - before_me))
    op_change = max(-4, min(4, after_op - before_op))
    f = typ + pos + [my_change / 4.0, op_change / 4.0]
    if is_wall:
        net = (after_op - before_op) - (after_me - before_me)
        before_res = c.path_resilience(frozenset(g.walls), g.pawns[1 - side], c.GOALS[1 - side])
        after_res = c.path_resilience(frozenset(child.walls), child.pawns[1 - side], c.GOALS[1 - side])
        res_drop = before_res - after_res
        # Relative placement geometry generalizes a useful wall pattern to
        # other ranks and to the opposite color after normalization.
        opp_adv=_advance(g.pawns[1-side][1],side)
        file_distance=abs((file_idx+.5)-g.pawns[1-side][0])
        rank_distance=abs((rank-.5)-opp_adv)
        f += [max(-4, min(4, net)) / 4.0,
              max(-2, min(2, res_drop)) / 2.0,
              file_distance / 8.0, rank_distance / 8.0]
    else:
        f += [0.0,0.0,0.0,0.0]
    f.append(1.0 if child.winner == side else 0.0)
    return f + _action_bits(move, side)


def dims():
    s = len(state_features(c.Game()))
    a = len(action_features(c.Game(), 'e2'))
    return {'state': s, 'action': a, 'hidden': HIDDEN}


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------

def _mlp_forward(x, layers):
    """x: list/1d. layers: {'W1','b1','W2','b2','in','hidden','out'} flat lists."""
    np = _np()
    X = np.asarray(x, dtype=np.float64)
    W1 = np.asarray(layers['W1']).reshape(layers['in'], layers['hidden'])
    W2 = np.asarray(layers['W2']).reshape(layers['hidden'], layers['out'])
    b1 = np.asarray(layers['b1'])
    b2 = np.asarray(layers['b2'])
    h = np.tanh(X @ W1 + b1)
    return float((h @ W2 + b2)[0])


def _logit(model, state_vec, action_vec):
    np = _np()
    x = np.concatenate([np.asarray(state_vec), np.asarray(action_vec)])
    return _mlp_forward(x, model['policy'])


def value(model, g, side=None):
    side = g.to_move if side is None else side
    return math.tanh(_mlp_forward(state_features(g, side), model['value']))


def policy_logits(model, g, legal=None):
    legal = g.moves(g.to_move) if legal is None else legal
    sv = state_features(g)
    context=action_context(g)
    return {move:_logit(model,sv,action_features(g,move,context=context)) for move in legal}


# ----------------------------------------------------------------------------
# Model identity / validation / io
# ----------------------------------------------------------------------------

def _body(model):
    return {'policy': model.get('policy', {}), 'value': model.get('value', {})}


def model_id(model):
    return hashlib.sha256(json.dumps(_body(model), sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:12]


def code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate(model):
    if not isinstance(model, dict) or model.get('schema') != SCHEMA:
        raise ValueError('Unsupported neural model')
    md = model.get('metadata', {})
    if md.get('kind') != 'mlp':
        raise ValueError('Neural model must be kind=mlp')
    expected=dims();saved=md.get('dims',{})
    if (saved.get('state'),saved.get('action'))!=(expected['state'],expected['action']):
        raise ValueError('Neural model feature dimensions do not match this engine build')
    for name in ('policy', 'value'):
        layers = model.get(name)
        if not isinstance(layers, dict):
            raise ValueError(f'Missing {name} layers')
        for k in ('in', 'hidden', 'out'):
            if not isinstance(layers.get(k), int) or layers[k] <= 0:
                raise ValueError(f'Invalid {name}.{k}')
        for k in ('W1', 'b1', 'W2', 'b2'):
            arr = layers.get(k)
            if not isinstance(arr, list) or len(arr) > MAXW:
                raise ValueError(f'Invalid {name}.{k}')
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in arr):
                raise ValueError(f'Non-finite coefficient in {name}.{k}')
        expected_in=expected['state']+(expected['action'] if name=='policy' else 0)
        if (layers['in']!=expected_in or layers['out']!=1
                or layers['hidden']!=saved.get('hidden')
                or len(layers['W1'])!=layers['in']*layers['hidden']
                or len(layers['b1'])!=layers['hidden']
                or len(layers['W2'])!=layers['hidden']*layers['out']
                or len(layers['b2'])!=layers['out']):
            raise ValueError(f'Invalid {name} layer dimensions')
    return model


def save(model, path):
    validate(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(model, separators=(',', ':')), encoding='utf-8')
    tmp.replace(path)


def load(path):
    return validate(json.loads(Path(path).read_text(encoding='utf-8')))


def new_model(seed=20260907, hidden=HIDDEN):
    np = _np()
    rng = np.random.default_rng(seed)
    d = dims()
    d['hidden']=hidden

    def make(in_dim, out_dim):
        std = math.sqrt(2.0 / (in_dim + hidden))
        W1 = (rng.standard_normal((in_dim, hidden)) * std).reshape(-1).tolist()
        b1 = [0.0] * hidden
        W2 = (rng.standard_normal((hidden, out_dim)) * 0.1).reshape(-1).tolist()
        b2 = [0.0] * out_dim
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
                'in': in_dim, 'hidden': hidden, 'out': out_dim}

    return {
        'schema': SCHEMA,
        'policy': make(d['state'] + d['action'], 1),
        'value': make(d['state'], 1),
        'metadata': {'kind': 'mlp', 'dims': d, 'seed': seed},
    }
