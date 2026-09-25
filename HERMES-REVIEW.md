# Review of dbf0d25 and the match benchmark

## Fixed defects

- The cross-check inserted a minimax heuristic score into negative MCTS visit
  counts. The later advice sort could restore the original MCTS move while the
  explanation still claimed an override. Overrides now use the complete minimax
  ranking on one score scale, keep the searched reply, and preserve raw MCTS
  rankings separately. Opponent tie-breaking cannot contradict the override.
- The extra search previously ran after MCTS spent the whole requested budget.
  Wall-present positions now reserve part of the same deadline for the check.
  Incomplete depth-two searches cannot override. Immediate wins and unavoidable
  next-turn losses bypass it.
- Depth-two heuristic comparison was described as an exact wall search. The
  explanation and logs now identify its depth, units and lack of a win proof.
- The benchmark called raw MCTS, bypassing the live safeguards. It now calls the
  live decision path, records its source build, and avoids filling real-game
  advice logs with benchmark traffic. Ply-limited unfinished games are labeled
  unresolved, not draws. It remains a screening tool, not a promotion gate or
  evidence of an 85% human win rate.
- The loss generator accepted incomplete minimax searches and depended on MCTS
  proposing the very wall it could miss. It now requires a completed comparison,
  retains the minimax candidate, and records the MCTS proposal separately.

## Evidence and limits

Public API retrieval returned HTTP 403. Local advice logs supplied valid histories
for both 6pg0tr and 7kz1qb. Ten selected positions are saved in
study/tactical-loss-cases.json, with tests covering the complete advice path.
These are trace-derived position cases, not newly downloaded finished games.

At 6pg0tr ply 37, hh5 and hh6 have equal depth-two heuristic scores. Therefore
the claim that hh5 was uniquely winning is not established. The test preserves
hh6 when MCTS already chooses it. No terminal winning proof was obtained here.

Live checks on the corrected server: 6pg0tr ply 37 returned hh5 in 3.72 seconds;
7kz1qb ply 44 returned hg8 in 3.43 seconds. Both logged the override and matched
the current engine fingerprint. Timing samples are not percentile guarantees.

## Deployment

The corrected server runs on 8811. Overlay 1.2 requires live_protocol 5 so the
older server on 8810 cannot answer it. The Downloads userscript is updated;
Firefox/Tampermonkey still requires replacing the installed script, saving it,
and reloading Barricade. Do not run both script versions together.

Existing untracked study/match-benchmark.json and study/regressions/52s6kd.json
were left untouched. They were not generated or validated as strength evidence
by this review.
