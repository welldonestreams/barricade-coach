---
name: barricade-coach
description: Use when coaching or analyzing Barricade/Quoridor games (barricade.gg). Live move advice, game review, or top-level study. Engine + workflow live in the user's workspace.
---

# Barricade / Quoridor Coach

Chance plays ranked Barricade (barricade.gg) as **steak2222** (~1340 elo, chasing
top elo ~2200). This skill = engine + workflows for live coaching and review.

## Engine location & usage

Workspace: `workspace/barricade-coach/` (engine `coach.py`; also mirrored in
`workspace/barricade/`).

- Best move for a position:
  `python3 coach.py "<full history csv from move 1, red first>" red|blue`
  (top moves listed; add `explain` for the reasoning on the top move).
- Fast daemon (persistent, pty): `coach_server.py` — send `"csv|side"` on stdin,
  get the top move back (~0.3 s warm).
- Grade a whole game (both colors): `grade_game(history, side)` in `coach.py`.

History is the FULL comma-separated move list starting with red's first move
(e.g. `e2,e8,e3,e7,...`). Parity decides whose turn it is.

## Notation (verified)

Pawn move = destination square; red (P1) starts e1/rank 1 and moves UP; blue
starts e9 and moves DOWN. `hXY` = horizontal wall between ranks Y–Y+1 spanning
columns X..X+1; `vXY` = vertical wall between columns X..X+1 at ranks Y..Y+1.
3-char = wall, 2-char = pawn move (so `h6` is a pawn move to h6). Walls that
would overlap, cross (same anchor letter+rank), or leave either pawn with no
path are illegal. Adjacent-pawn jumps + jump-blocked diagonal steps are legal.

## Live coaching protocol (moves only, speed-critical)

During a live game the user is on a clock — reply with ONLY the move, no
commentary. Keep history yourself; each user message is the opponent's last
move (or a "new game, I'm red/blue" header). Run the engine per move; answer in
one short token. If the user says they played a different move than advised,
update history to match reality before computing.

Latency lessons learned (hard): never run multi-step validation/explain turns
mid-game; use one engine call. 5+3 time control is too tight for LLM relay —
recommend 10+0+ or casual for coached games.

## Game review workflow

Fetch real games (no auth needed for shared games):
- `GET https://api.barricade.gg/games/<shareCode>` → record incl. historyCsv.
- `POST https://api.barricade.gg/api/analyze` with
  `{"gameId":"<id from record>"}` → per-ply engine grades (only after the game
  has been analyzed once — have the user open the /analysis link to trigger).

Review pattern: replay history with coach (legality check), grade both sides,
flag rank>3 / delta≥2 moves (ignore endgame-walk noise), narrate pawn journeys
and wall-fence structures, tie lessons to the user's recurring leaks (flank
neglect, over-walling, slow conversion, clock management).

## Strategy reference (from Blocade's engine-author guide)

- One-number rule: compare shortest paths every turn; every move lowers yours
  or raises theirs. Walls that add <2 squares to their path: don't play.
- Open center (e-file) 3 pushes; answer early opponent walls by WALKING.
- Walls work in fences: extend board-edge/other walls; lone walls get
  side-stepped for ~1 square.
- Jump = free tempo; watch parity before stepping adjacent.
- When your route is shorter and wall-proof: RUN, don't wall.
- Top players (MikeJordan ~2200) open with vd5 as 5th move as red, wall the
  flank the opponent drifts toward, and use herding + photo-finish races.

## Known user context

steak2222, US. Session 2026-09-06: 3W–2L (~1330-1350) vs 1200-1400s; losses =
left-flank neglect (a-file runs), over-walling, on-time loss, slow conversion.
Top-elo study set available: 3ma7p2, v88zty, x7zv38, 7p63jt, 5sx6pm (all
feature MikeJordan 2200).
