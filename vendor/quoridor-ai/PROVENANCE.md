# Quoridor MCTS attribution

Author: Kyutae Lee (gorisanson). MIT license is preserved in LICENSE.
Source: https://github.com/gorisanson/quoridor-ai
Pinned commit: e15d6e7825e6f7c2f3a41de0ecef1af7aa290f1e
Retrieved: 2026-09-06. game.js and ai.js are copied without source changes.

The adapter calls MonteCarloTreeSearch directly, with exploration constant 0.2,
bounded batches, optional seeded randomness, and Barricade coordinate conversion.
It does not call AI.chooseNextMove or its random opening overrides. Python validates
the entire history and filters moves that concede an immediate pawn win whenever
a safe alternative exists. The copied engine uses selective wall expansion and
biased rollouts; simulated win rates are not calibrated human win probabilities.
