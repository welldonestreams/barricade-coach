# Measurable improvement toward an 85% win target

85% is a target against a specified opponent field and time control, not a property
of an engine. Track outright wins separately from draws, losses and simulation
estimates. Measure results by opponent rating and color as the account climbs.
No benchmark below establishes an 85% win rate against top leaderboard players.

## 1. Freeze a trustworthy baseline

Use overlay 1.1 with the compatible backend. Save actual advice traces and the
engine build for each game. Keep the current build as a frozen opponent. Turn each
confirmed mistake into a legal-position regression case. Require zero illegal or
stale recommendations in the regression suite. Measure end-to-end latency under
training load: target p95 below five seconds, with a request cutoff below twenty.

Implemented in this patch: deadline safeguards, local advice traces, exclusion of
unvalidated mixed priors, exact solving when both wall inventories are zero,
recognition of unavoidable immediate loss, and corrected post-game review flow.
These are correctness improvements; their match-strength gain remains unmeasured.

## 2. Build an independent arena before changing learning

Run paired games with colors swapped against frozen MCTS, minimax, and several
older versions with different styles. Use fresh openings and a held-out position
set. Compare at equal live time budgets. Store engine hashes, seeds, starts,
outcomes, timeouts and latency. Keep paired results together for uncertainty
estimates. Start with 200 games for screening; use a larger, fresh confirmation
set for promotion. Select its size and acceptance rule before seeing results.

Require a confidence interval above an even score against the baseline, no
material regression against other opponents, and passing correctness/latency
checks. This proves improvement against that arena, not 85% human wins. Measure
the latter separately on a defined rating band and fixed time control, reporting
sample size and uncertainty. The existing Python promotion experiment does not
validate a new MCTS policy.

## 3. Produce better training targets

Deduplicate the harvested games and split entire games before sampling positions.
Exclude overlapping board positions across training and evaluation where possible;
hold out recent games and some players to test generalization. Tag every sample
with source, engine version, search budget, side to move and result. Never combine
repeated imports as independent evidence.

Replay legal positions from strong human games and the user's losses. Favor
decisions about pawn contact, spending the last walls, route traps and endgames;
limit repeated center-file openings. Generate random positions through legal
play, not arbitrary pawn/wall placement. Use longer offline search with multiple
seeds and adversarial replies to compare alternatives. Keep uncertainty when
searches disagree; reserve claims of proof for completed exact solving.

Store the stronger search's move distribution as a policy target and finished
game outcomes as value targets. Mark truncated games as unresolved, not losses.
Do not label a move good just because its player eventually won. A before/after
heuristic delta is a feature to test, not a causal explanation or a new teacher.

## 4. Train a compact evaluator that generalizes

Prototype a small policy/value model using pawn locations, walls, inventories,
side to move and goal direction. Its outputs are promising legal moves and a
position-outcome estimate. Mask illegal moves. Compare it with simpler feature
models on held-out data before choosing model complexity.

Integrate the candidate into MCTS to guide exploration and evaluate leaves;
preserve legality checks and tactical proofs. Benchmark the whole search, not
just prediction accuracy. A model that fits records better can still play worse.
Train offline; keep live inference small enough for the four-second search budget.

Search plus a model predicting moves and outcomes is an established approach in
[AlphaGo Zero](https://deepmind.google/blog/alphago-zero-starting-from-scratch/).
Applying that approach here is an engineering experiment, not a promise of its
results or compute requirements.

## 5. Make self-play challenging and promotion controlled

Train against a mixture of frozen opponents and current candidates, with varied
legal starting positions and both colors. Sample some failures more often while
retaining broad coverage. Hold the evaluation field apart from this training
league. Keep candidate and champion artifacts separate; background workers must
not immediately rewrite the live policy. Promote only after the arena gate passes,
and retain a rollback build. Self-play's feedback is useful when independently
tested; top-player starting positions alone do not prevent reinforcing mistakes.

## 6. Optimize speed and explanations after strength is measured

Reserve compute for live advice so training cannot starve it. Test validated
opening shortcuts and search-tree reuse keyed by the complete position. Reject
cached advice after any state mismatch. Check route-changing wall replies before
calling a race safe; a shortest-path race is not exact while usable walls remain.

Explain a move using its searched continuation: what the opponent can do, the
route change, the wall cost, and whether the result is proven or estimated.
Evaluate opponent-specific adjustments separately before letting them override
general search. Familiarity with an opponent is not evidence a weaker move works.

## Immediate work order

1. Install the corrected overlay and collect versioned advice traces.
2. Build the equal-budget MCTS arena and record the frozen baseline.
3. Generate deeper-search targets for the loss/regression corpus and held-out split.
4. Train and benchmark a compact policy/value candidate.
5. Add league self-play and promote only measured improvements.

Steps 2-5 are proposed next work, not features completed by this patch. Archive
size, worker count and simulated win rates are not substitutes for these results.
