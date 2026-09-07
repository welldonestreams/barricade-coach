# Barricade Coach

Local Quoridor/Barricade analysis with validated move history, tactical search,
and an optional Monte Carlo engine. Advice is heuristic: neither engine is
proven perfect or benchmarked at leaderboard Elo.

## Run

Requires Python 3.10+; optional MCTS also requires Node.js on PATH. No Python or
npm packages are required. Both engines run locally without network calls.

```sh
python server.py
# Open http://127.0.0.1:8810
python coach.py "e2,e8,e3,e7,e4,e6" red explain
python coach.py "e2,e8,e3,e7,e4,e6" red --engine mcts --seconds 5 --json
python coach.py "e2,e8" --depth 4 --seconds 15
python coach_server.py
# stdin: e2,e8|red   -> one JSON response per line
```

Use the visible **Red · first** or **Blue · second** buttons to choose your color
at any time, including midgame. Switching color keeps the moves and board intact
and updates whose turn you are waiting for. Your selection survives reloads.
Enter each actual move, or paste the full comma-separated
history and click Load. Click a recommendation to record that you played it.
Illegal entries leave the confirmed board unchanged. Undo corrects an entry;
Reset starts a game. Confirmed history and your color survive reloads in the
same browser. A suggestion is cleared whenever the position changes.

During chat coaching, say **new game, I'm red** or **new game, I'm blue**. For a
midgame, provide the full history from red's first move. Report the opponent's
move and any departure from our previous suggestion. The coach needs the
actual position; it must not guess missing walls or moves.

## Notation

Red starts e1 and aims for rank 9; blue starts e9 and aims for rank 1. Red moves
first. Pawn destinations are a1 through i9. `h6` is a pawn square, not a wall.

- `hd4`: horizontal wall blocking d4-d5 and e4-e5.
- `vd5`: vertical wall blocking d5-e5 and d6-e6.
- Wall anchors range a1 through h8. Overlap and crossing are illegal.
- Each player has ten walls. Every wall must leave both players a route.
- Jump straight over an adjacent opponent when possible; diagonals around that
  opponent are allowed only when the straight jump is blocked by a wall or edge.
- Reaching the goal ends the game. No further moves are accepted.

## Engines

**Tactical search (default):** full-width minimax with alpha-beta pruning and
iterative deepening. Every legal root move and wall is eligible. Depth is actual
plies, not an ignored setting. The clock returns the last *completed* iteration
so alternatives are scored at the same depth. Depth 0 explicitly means an
unsearched legal fallback. Cached board graphs and distance maps avoid replaying
and recomputing the same geometry throughout search.

Its nonterminal score is 100 times the path-distance difference, plus a small
wall-reserve term and turn adjustment. It considers an immediate legal jump,
then uses wall-only distance; this is a heuristic, not an exact race solver.
Decisive terminal scores override it. Lower scores are preferred, but a negative
score is **not proof of a win**. Equal scores share a grade rank.

**Monte Carlo (experimental):** Kyutae Lee's MIT-licensed engine, pinned under
`vendor/quoridor-ai`, with local Node.js execution. It samples games using
selective wall expansion and biased rollouts. Our adapter omits upstream random
opening overrides, validates notation and state independently, takes immediate
wins, and filters moves allowing an immediate opponent win when a safe move exists.
It preserves the last completed simulation batch if its process exceeds the
budget. A tiny budget can return an explicitly marked fallback. Alternatives
are ordered by simulation visits; those numbers are not comparable to minimax
scores or calibrated win probabilities. `--seed` and `--rollouts` support repeatable
experiments; time-limited runs may complete different numbers of simulations.

Quick mode allows five seconds; Study allows fifteen. Actual runtime may be
shorter, or slightly exceed the budget for validation/process cleanup. MCTS uses
up to 60,000 rollouts by default. Choosing it does not imply greater playing strength.

## Shared games and study

The ten supplied game records are in `study/`. The initial dataset has 418
legal plies. Tests compare both independent rule implementations at all 428
positions, including starts and final positions. See `REVIEW.md` for findings,
limitations, and the user's concrete practice positions.

```sh
python -m unittest -v
python review_games.py
python benchmark.py --seconds 1
python collect_games.py MikeJordan RejeCted_ owen12345
```

The collector uses public profile pagination, deduplicates game codes, respects
rate limits, and resumes downloaded files. Full profile lists and raw archives
stay out of Git; local collection summaries describe coverage and failures.
Cached profile indexes are reused on subsequent runs (they are snapshots, not a
live subscription). Rename the relevant cached profile file to request a fresh index.
When collection finishes, the opening reference is rebuilt automatically.
It only supports the 9x9 rules implemented here; unsupported games are reported.

Official per-move analysis is separate: on 2026-09-06 the shared game GET worked,
but POST `/api/analyze` returned `login_required`. The local review contains our
heuristic grades, **not** Barricade's official accuracy. Old claims of perfect
agreement with the official engine were not reproducible and have been removed.

## Local server

`/api/state?h=...` validates and returns the position without a search.
`/api/move?h=...&side=red&engine=python&depth=2&seconds=5` adds advice.
`engine=mcts` selects the optional engine. Requests are bounded; only one search
runs at a time. The server binds to loopback, rejects foreign Host/Origin headers,
and does not enable cross-origin access. It is a local tool, not a public service.

## Strategy sources and attribution

- [Rules and basics](https://quoridorstrats.wordpress.com/beginners-guide-rules-and-basics/)
- [Blocade strategy guide](https://playblocade.com/blog/quoridor-strategy)
- [Kyutae Lee's MCTS implementation](https://github.com/gorisanson/quoridor-ai)
- [Netlify Quoridor AI](https://quoridor-ai.netlify.app/): its served interface connects
  to remote Ishtar/Ka engines; their server implementation was not copied.

The MCTS copyright, license, source commit and adapter differences are in
`vendor/quoridor-ai/LICENSE` and `vendor/quoridor-ai/PROVENANCE.md`.


## Firefox / Tampermonkey overlay (1.0)

Run START-COACH.cmd, then replace the existing Tampermonkey script with
`overlay/barricade-live-coach.user.js` (also served at
http://127.0.0.1:8810/barricade-live-coach.user.js). Save and reload Barricade.
Do not run the old and new scripts together. Version 1.0 appears in the panel.


The overlay independently checks the numbered move list, pawn squares, placed
wall identifiers, and the player cards' remaining counts. Unreadable or mismatched
positions receive no advice. It never reconstructs moves from polling differences
or accuses you of a move inferred from a hover preview. Switching Red/Blue changes
your side, independently of rotating the board. Steak-prefixed usernames and
configured aliases are recognized. Green marks the recommended square or wall;
it moves with scrolling/rotation and clears on a changed or uncertain position.
Replay scrubbing with a full future move list fails closed rather than guessing.

Live searches normally get 4 seconds; uncontested central openings get 0.35 seconds.
The bridge stops waiting after 18 seconds and discards stale responses. This is a
response deadline, not a promise that every overloaded machine finds strong advice.
`/api/live` checks the board/history match and legality again before responding.
Route changes and a searched reply explain the recommendation; short searches are
not proof of optimal play. Opponent data breaks exact tactical ties only, from
validated matching positions and the opponent's current color. Model collection is
asynchronous and never blocks the move query. No matching evidence is shown plainly.

## Training without automatic self-reinforcement

`python training.py --games 100 --workers 2 --evaluate 100` runs bounded parallel
experiments from randomly sampled, validated positions in the current top-100
archive. Two CPU cores are reserved where possible; Windows workers run below
normal priority. It uses a fixed tactical engine, not the live learning blend.

Games are deduplicated by normalized history and split by content hash. Held-out
games and their exact positions never contribute learned outcomes. Cutoff games
are not losses. Each candidate is evaluated from identical held-out positions in
both colors against a frozen baseline. A minimum of 100 pairs and a conservative
95% lower bound above 50% on pair wins are required for promotion. Unsuccessful
experiments remain under ignored `memory/experiments`; only a passing champion can
break exact nonterminal tactical ties in live advice. Legacy mixed self-play tables
and repeated static evaluation deltas cannot influence recommendations. More
self-play does not itself establish stronger play or leaderboard Elo.

The old selfplay.py collector remains for compatibility; its tables are not used
for live move selection. Existing older worker processes do not acquire the new
promotion behavior until restarted. Tests: `python -m unittest -v`, including Node
regressions. The local `/overlay-fixture` page exercises the screenshot position
and rotation without playing a real match.
