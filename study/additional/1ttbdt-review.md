# 1ttbdt: coach-followed loss

The user confirmed every move followed the highlighted recommendation.
Red: pawan568. Blue: steak2222. Red won after 55 plies; blue had two walls left.

## Demonstrated problem

At ply 20, blue played va2. A completed depth-two full-width search selects e5,
and a fresh seeded parallel MCTS run also prefers e5. In the actual continuation
va2, d4, e5, he4, blue's route estimate becomes 16. In the legal counterfactual
e5, d4, e4, it is 10. Advancing now lets blue get below the later barrier; another
wall placement spends that tempo. This is a concrete counterfactual, not proof
that e5 wins against every possible reply.

At ply 54, no legal move prevents red's immediate goal. This is independently
verified by generating every legal blue move and checking red's legal goal moves.
The coach now reports that fact without wasting simulations or implying its
recommended move can still save the position.

There was no saved engine-response trace for the original game, and the current
learning tables had no entry at the exact ply-20 position when checked. The
unbounded-prior defect is real, but it cannot be identified as the cause of this
particular original recommendation. Fresh searches are not a replay of the old
engine's random state. A bounded local advice trace now records actual requests,
positions, recommendations, engine type, and source-build fingerprint for future
reviews. Those logs are ignored by Git and not uploaded.

## Safeguards and validation

- Raw heuristic swings and mixed self-play outcomes no longer reorder live MCTS.
- Opponent evidence breaks exact ties; visit counts are not heuristic points.
- A Python champion requires matching engine/source and the held-out gate.
- Parallel MCTS workers share one deadline and report subprocess timeouts honestly.
- Both-zero-wall endings use exact graph solving with jumps, turns, and cycles;
  clocks are excluded, and incomplete solving falls back without an exact claim.
- Post-game review validates the board against the history, has a 12-second total
  budget, and labels partial results. Fixed the duplicate guard that suppressed it.

52 tests passed locally before release, including this tempo mistake, the final
unavoidable loss, blocked finishing jumps, unvalidated priors, and killed workers.
Live checks: opening 1.23 seconds; ply 20 about 4.06 seconds. These are measured
samples, not a promised game win rate. The earlier gated Python experiment scored
100/200 and was not promoted; it does not measure this newer build.
