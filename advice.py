"""Learned move selection: blend tactical search with remembered outcomes.

Three signals, in order of authority (documented so ChatGPT can audit):

  1. TACTICAL  - the minimax/shortest-path search score. DOMINANT. Never
                 overridden by memory; memory only *nudges*.
  2. OWN EXP   - what actually happened in every recorded game (self-play,
                 archive, live): per-move win rate + causal eval-delta. This
                 is the "why did this move win/lose" signal, and it is the
                 mechanism that protects against surprising play in a real
                 game: a move that historically loses its eval advantage is
                 discounted even if the static eval likes it.
  3. OPPONENT  - what THIS opponent tends to play (color-weighted). Deliberately
                 the SMALLEST influence: it only adjusts a move by the gap
                 between the opponent's predicted reply and their optimal
                 reply, and is capped. It nudges; it never dictates.

All weights are named constants at the top of this file. Lower score = better
(consistent with coach.eval_side).
"""
from __future__ import annotations

import math

import coach as c
import learning

# ---- Own-experience weights ------------------------------------------------
# Centipawn-equivalent influence of the outcome-winrate prior at full confidence.
# A move with 100% winrate across many games gets up to EXP_WIN_CP centipawns of
# preference vs one with 0%. Small on purpose: static eval differences are
# usually hundreds of centipawns, so memory nudges but rarely flips a clear call.
EXP_WIN_CP = 25.0

# Fraction of the raw eval-delta applied. eval-delta is already in centipawns
# (delta = eval_before - eval_after for the mover), so this is a straight scale.
# 0.35 means "trust remembered causal effect 35% as much as a fresh static eval."
EXP_DELTA_SCALE = 0.35

# Hard cap on the eval-delta term per move (centipawns). Prevents one weird
# memorized line from dominating.
EXP_DELTA_MAX = 60.0

# Sample-count half-saturation: how many recorded games give 50% confidence.
# Below this, memory barely speaks; above it, confidence asymptotes to 1.
EXP_CONF_K = 5.0

# ---- Opponent-tendency weights --------------------------------------------
# Influence of the opponent's predicted reply, relative to the eval gap it
# creates. 0.15 = "at most 15% of the gap between their predicted and optimal
# reply is credited." Keeps opponent modeling a small factor.
OPP_WEIGHT = 0.15

# Hard cap on the opponent term per move (centipawns).
OPP_MAX = 12.0

# Opponent sample half-saturation.
OPP_CONF_K = 5.0

# How many of the top tactical candidates we bother to look ahead against the
# opponent model (the expensive part). Bounded for live-game latency.
OPP_LOOKAHEAD_N = 4


def _conf(samples, k):
    """Sample-size confidence in [0, 1]: samples/(samples+k)."""
    if samples <= 0:
        return 0.0
    return samples / (samples + k)


def _own_exp_adj(history, move):
    """Experience adjustment (centipawns) for `move` at the current position.
    Negative = prefer (lower score is better)."""
    pos = list(history)
    adj = 0.0

    # Outcome winrate prior
    outs = learning.outcome_stats(pos)
    cell = outs.get(move)
    if cell and cell.get('total'):
        winrate = cell['winrate']  # 0..1
        conf = _conf(cell['total'], EXP_CONF_K)
        # (winrate - 0.5) * 2  maps 0..1 -> -1..1 ; negative when move tends to lose
        adj += -EXP_WIN_CP * conf * (winrate - 0.5) * 2.0

    # Causal eval-delta prior
    evals = learning.eval_delta_stats(pos)
    ecell = evals.get(move)
    if ecell and ecell.get('games'):
        conf = _conf(ecell['games'], EXP_CONF_K)
        delta = max(-EXP_DELTA_MAX, min(EXP_DELTA_MAX, ecell['avg_delta']))
        # positive delta = move helped the mover (lowered eval) -> prefer -> negative adj
        adj += -EXP_DELTA_SCALE * conf * delta

    return adj


def _terminal_eval(g, side):
    if g.winner is None:
        return g.eval_side(side)
    return -c.WIN if g.winner == side else c.WIN


def _opp_adj(history, move, side, opp_name, opp_color):
    """Small adjustment from predicting the opponent's reply to `move`.
    Positive = the opponent's predicted (tendency) reply is worse for me than
    their optimal reply, so discount this move slightly. Negative = bonus."""
    if not opp_name:
        return 0.0
    g = c.Game(list(history) + [move])
    if g.winner is not None or g.to_move == side:
        return 0.0
    opp_side = g.to_move
    insight = learning.opponent_insight(opp_name, g.history, opp_color)
    if not insight or not insight.get('known') or not insight.get('top'):
        return 0.0

    predicted = insight['top'][0][0]  # their most-likely reply
    conf = _conf(insight.get('total', 0), OPP_CONF_K)

    # Value for me of their optimal reply (worst case for me).
    best_for_me = None
    for r in g.moves(opp_side):
        g3 = g.copy()
        g3._play(r)
        v = _terminal_eval(g3, side)
        if best_for_me is None or v < best_for_me:
            best_for_me = v
    if best_for_me is None:
        return 0.0

    # Value for me of their *predicted* reply.
    try:
        g4 = g.copy()
        g4._play(predicted)
    except ValueError:
        return 0.0
    predicted_val = _terminal_eval(g4, side)

    gap = predicted_val - best_for_me  # >0 => their tendency is worse for me
    return max(-OPP_MAX, min(OPP_MAX, OPP_WEIGHT * conf * gap))


def advise(history, side=None, depth=2, seconds=5.0, engine='python',
           opp_name=None, opp_color=None, rollouts=60000, seed=None):
    """Run tactical search then re-rank by blended score.

    Returns the search result dict with two extra keys:
      'scored'  - [[blended_score, move], ...] sorted ascending (best first)
      'blend'   - per-move breakdown {move, tactical, own_exp, opponent, final}
    The tactical engine result (without blend) is preserved under 'tactical'.
    """
    hist = list(history)
    if engine == 'mcts':
        import mcts_coach
        result = mcts_coach.search(hist, side, seconds, rollouts, seed)
    else:
        result = c.search(hist, side, depth, seconds)

    tactical = result.get('scored') or []
    if not tactical:
        result['tactical'] = []
        result['blend'] = []
        return result

    side = c.Game(hist).to_move if side is None else side

    rows = []
    for rank, (score, move) in enumerate(tactical):
        own = _own_exp_adj(hist, move)
        opp = _opp_adj(hist, move, side, opp_name, opp_color) if rank < OPP_LOOKAHEAD_N else 0.0
        final = score + own + opp
        rows.append({'move': move, 'tactical': score, 'own_exp': own,
                     'opponent': opp, 'final': final})

    rows.sort(key=lambda r: r['final'])
    result['tactical'] = tactical
    result['scored'] = [[r['final'], r['move']] for r in rows]
    result['blend'] = rows
    result['engine'] = engine
    return result
