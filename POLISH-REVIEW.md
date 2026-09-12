# Overlay and training review

## Applied fixes

- The former overlay mixed player color with rank orientation and never reversed
  files. It guessed moves from snapshot differences, cached position queries by
  that guessed history, and accepted late responses. Predictive prefetch could
  cache the current board under a hypothetical future history.
- Version 1.0 reads the actual numbered move list and checks it against pawn
  locations, stable placed-wall IDs, and both remaining counts. The server
  independently replays every move. Any disagreement clears advice. The site’s
  canonical grid coordinates determine rotation independently of player color.
- Only advice for your side is highlighted. Replies are tied to a request,
  history, and full position, then checked against the freshly read board.
  Unconfirmed hover previews are never treated as moves. The unreliable yellow
  accusation has been removed.
- Explicit-position input now rejects out-of-bounds pawns, duplicate/crossing
  walls, impossible reserves, and blocked paths. Opponent models validate whole
  games before counting them and match actual positions, including early play.
- Legacy outcome/evaluation blending was unsafe: it treated unfinished games as
  losses, mixed self-play with live evidence, and added heuristic units to MCTS
  visits. Live advice now preserves tactical order. Validated opponent evidence
  and an independently gated champion may only break exact tactical ties.

## Verified

39 tests passed, including all 46 pre-move positions from 5s2skz and the pictured
blue-to-move position. The latter correctly recommends the legal e4 jump; d2 is
not legal there. Browser fixture checks showed the highlight following rotation
and clearing after a confirmed move. The real public replay DOM was inspected to
verify the grid, placed-wall, history, and player-card selectors.

Six live requests took 0.212–3.883 seconds. Another request with the new training
workers running took 3.898 seconds. Opening search uses a 0.35-second allowance;
normal live search uses four seconds. The overlay abandons a request after 18
seconds and displays an error instead of old advice. Search quality is still
heuristic and these timings do not guarantee performance on every machine.

A two-worker smoke experiment trained four games and evaluated two color pairs.
It scored 2/4 and correctly did not promote. A larger 100-training-game,
100-held-out-pair experiment is running in the background with two workers.
At its start the validated top-100 dataset contained 7,765 training games and
1,927 held-out games. Results go to memory/experiments and logs/training-gated.log.
No strength gain or leaderboard Elo has yet been established. Old workers’ mixed
SQLite tables cannot affect live advice.

## Installation state

The updated server is running on standard port 8810 (health live_protocol 3).
The old 8811 instance was stopped after verifying 8810. The updated userscript is
in overlay/ and copied to the user's Downloads folder; the prior Downloads copy
was backed up. Firefox is not exposed through the available browser connection,
so its installed Tampermonkey script has NOT been changed or verified. Replace
the existing script with version 1.0, save, and reload Barricade. Do not keep two
versions enabled. A green Coach label should mark the recommended destination.

The archive collector’s generated opening-book changes were left untouched.
