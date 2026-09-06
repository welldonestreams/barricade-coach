---
name: barricade-coach
description: Use the local Barricade coach for validated move advice and post-game study.
---

# Barricade coaching workflow

Use the files in the same directory as this skill. Python 3.10+ is required;
MCTS additionally needs Node.js. Read README.md for notation and score semantics.

## Live move relay

Keep confirmed history from red's first move, the user's color, and any pending
recommendation. Start with `new game, I'm red/blue`; for a midgame require full
history. Run the engine on every actual position. If a user reports an opponent
reply after our recommendation, treat that as confirmation they played it unless
they say otherwise. Apply corrections before calculating. Never add a suggestion
twice or continue after a history-validation error. A missing/illegal move needs
one concise clarification; never invent the board.

During an established live game, answer with the move only unless input is
ambiguous or invalid. Validate and search in one call; do not run long reviews
between moves. For quick tactical advice:

`python coach.py "<history>" red --seconds 5 --json`

For optional Monte Carlo advice:

`python coach.py "<history>" red --engine mcts --seconds 5 --json`

The persistent Python daemon accepts `history|red` or `history|blue` on stdin and
returns JSON including move, completed depth, time and timeout status. It stays
alive after bad input. It does not maintain game state itself; send full history.
A depth-0 or MCTS fallback is unsearched and must not be represented as best play.
Neither engine is a perfect solver or has measured leaderboard Elo.

## Review and learning

Use the supplied public game record, then replay with Game before grading.
`review_games.py` grades the user's side in their five games and both sides in the
five high-Elo examples. `grade_game` reports one-based ply numbers and gives equal
scores equal rank. Only compare grades at their reported completed depth.
Local scores are not official Barricade accuracy or win percentages.

`collect_games.py <usernames...>` retrieves public game indexes and histories with
pagination, deduplication, rate-limit backoff and resume. The cached indexes are
snapshots. Record validated coverage and errors rather than implying all records
were downloaded or all positions deeply analyzed.

Public GET `/games/<shareCode>` was verified in this review. Official analysis
POST required login; do not claim it is unauthenticated or fabricate its grades.

## Strategic judgment

Count both routes, including immediate jumps and whose turn it is. A wall costs
a move and one of ten reserves. Compare its effect on both routes and the best
reply. Inspect fences, flank escapes and defensive route-sealing walls. The
"add at least two squares" guideline is a useful question, not a legality rule:
a quiet defensive wall or parity move may still be strong. Avoid hard-coding
one player's opening or assuming a high-Elo move must be optimal.

When ahead, test the opponent's remaining wall threats before running. When the
position is decided, prefer a demonstrated finishing line. Study the user's
clock loss separately from board errors. Use REVIEW.md and the current game
records for evidence instead of old rating or performance claims.
