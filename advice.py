"""Tactical advice + a learned board-state prior on top.

The tactical engine (MCTS or minimax) is authoritative; the learned prior
(real win/loss from recorded games keyed by board state, not move order) is a
*close-call* signal. It never overrides a clear tactical gap, but it separates
near-equal candidates using what actually won in practice.
"""
import coach as c
import learning

# Learned-prior strength. LEARN_CP bounds how many centipawns the prior can move
# a candidate; LEARN_CONF_K is the sample-count half-saturation (how many games
# before we half-trust the winrate); LEARN_MIN_SAMPLES is the floor before the
# prior speaks at all. All deliberately small: the prior nudges, it never dictates.
LEARN_CP = 30.0
LEARN_CONF_K = 15.0
LEARN_MIN_SAMPLES = 3

# The "why" (position-swing) signal. REASON_CP bounds its influence; a move's
# avg_delta is already in ~centipawns, so this scales it down conservatively.
REASON_CP = 0.5
REASON_CONF_K = 15.0
REASON_MIN_SAMPLES = 3


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
    # Learned board-state prior: what actually won/lost from this exact position
    # across every recorded game (own live games + the top-player archive). It
    # is a *prior on close calls*, not a dictation: the tactical score still
    # dominates, and the prior only separates candidates whose tactical scores
    # are close. Confidence scales with sample count (Laplace-smoothed).
    prior = learning.learned_prior(hist)
    reasons = learning.learned_reasons(hist)
    # Normalize the tactical signal onto a shared ~centipawn-ish scale so the
    # learned prior can flip a genuine near-tie but never a confident gap.
    #   minimax: score IS centipawns.
    #   MCTS:    score = -visits; map visit share (visits / total_visits) to a
    #            0..1000 strength scale, so two moves within a few percent of
    #            each other's visits are a real close call.
    # Everything below is "lower is better" so a single ascending sort works.
    if engine == 'mcts':
        # score = -visits already sorts best-first ascending (most visits = most
        # negative). Strength is already in that order; use it directly.
        visits = {mv: -sc for sc, mv in tactical}
        total_visits = max(1, sum(visits.values()))
        # more visits -> more negative -> better; -1000..0
        tactical_strength = {mv: -1000.0 * visits[mv] / total_visits for mv in visits}
    else:
        tactical_strength = {mv: sc for sc, mv in tactical}
    rows = []
    for score, move in tactical:
        preference, evidence = 0, None
        if abs(score) < c.WIN - 100 and opp_name and time.monotonic()-start < seconds:
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
        cell = prior.get(move)
        learned = 0.0
        if cell and cell.get('total', 0) >= LEARN_MIN_SAMPLES:
            # Laplace-smoothed winrate, centred at 0.5, scaled by sample confidence.
            # Higher winrate -> NEGATIVE learned (lower = better = prefer).
            n = cell['total']
            smoothed = (cell['won'] + 1) / (n + 2)
            conf = n / (n + LEARN_CONF_K)
            learned = -LEARN_CP * conf * (smoothed - 0.5) * 2.0
        # The "why" signal: average position swing the move produced for its
        # mover, independent of who won. A move that reliably improves the
        # position (positive avg_delta) is good even if the raw winrate is
        # noisy from few samples. Negative avg_delta = the move hurt -> avoid.
        why = reasons.get(move)
        learned_why = 0.0
        if why and why.get('games', 0) >= REASON_MIN_SAMPLES:
            n = why['games']
            conf = n / (n + REASON_CONF_K)
            learned_why = -REASON_CP * conf * why['avg_delta']
        rows.append(dict(learned=learned, learned_why=learned_why, move=move, tactical=score,
                         own_exp=0, opponent=0, strength=tactical_strength[move],
                         final=tactical_strength[move] + learned + learned_why,
                         preference=preference, evidence=evidence,
                         samples=cell.get('total', 0) if cell else 0,
                         why=why.get('games', 0) if why else 0))
    rows.sort(key=lambda r: (r['final'], len(r['move']), r['move']))
    result['scored'] = [(r['final'], r['move']) for r in rows]
    result['blend'] = rows
    if result['scored'][0][1] != tactical[0][1]:
        # The original principal variation belongs to a different root.
        result['principal_variation'] = [result['scored'][0][1]]
    result['elapsed'] = time.monotonic()-start
    return result
