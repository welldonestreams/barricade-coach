# Barricade Coach

Local, zero-token Quoridor/Barricade move coach + game analyzer.

- Engine: `coach.py` — board model, legality, shortest-path ("one-number rule")
  minimax with opponent best-reply lookahead. Pure Python, stdlib only.
- Web UI: `server.py` → http://127.0.0.1:8807 — click-to-play board, instant
  local rendering, best-move + top alternatives + path lengths.
- CLI: `python3 coach.py "e2,e8,e3,e7,e4,e6" red` → top moves, or `... explain`
  for the reasoning, or grade a full game with `grade_game()`.

## Run it

```bash
python3 server.py          # then open http://127.0.0.1:8807
python3 coach.py "e2,e8,e3,e7,hd1,e5,e6,e4,e7,e3" red
```

Live-play protocol (works with any LLM or solo): track the move list from move 1
(red first). Before your move, run the engine on the history so far with your
side. Play the top move, then wait for the opponent's reply, append it, repeat.

## Notation (verified against real barricade.gg games)

- Pawn move = destination square, files a–i, ranks 1–9. Red starts e1 (rank 1,
  moves up), blue starts e9 (moves down). Red = Player 1 = first move.
- `hXY` = horizontal wall: blocks vertical crossing between ranks Y and Y+1,
  spanning columns X..X+1. (h d 4 blocks d4↔d5 and e4↔e5.)
- `vXY` = vertical wall: blocks horizontal crossing between columns X..X+1 at
  ranks Y and Y+1. (v d 5 blocks d5↔e5 and d6↔e6.)
- Walls with the same anchor (letter + rank) cross → illegal. Parallel walls
  sharing an edge → illegal. A wall must leave both pawns a path to goal.
- 2-char moves are always pawn moves (`h6` = pawn to h6). 3-char moves starting
  with h/v are walls.
- Jump rule: when adjacent to the opponent, you may jump over them; if the jump
  target is blocked, you may step diagonally to either side square of them.
  (Both appear in top-level games.)

## Engine notes

- Score = (my shortest path − opponent's shortest path), minimaxed 2 plies
  (your move → their best reply). Lower is better; ≤0 means you're winning
  the race.
- Validated: replayed ~600 real plies from barricade.gg (ranked 1200–2200)
  with 100% legality; reproduced the site engine's best move at 4/4 key
  decision points tested (ha2, vb6, hd5, hb5).
- Limitations: heuristic depth-2. Opening/quiet positions often tie at +0 —
  prefer the hd3-family wall directly in front of your pawn (blocks the jump
  over you) or keep pushing the center. Late "both pawns walking" positions
  can rank many walks equal; trust the SP readout there.

## Post-game review (official engine data, no auth needed for shared games)

```bash
curl -s https://api.barricade.gg/games/<SHARECODE>          # game record
curl -s -X POST https://api.barricade.gg/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"gameId":"<GAME_ID_FROM_RECORD>"}'                    # per-ply engine grades
```

Engine analysis only exists once the game has been analyzed (open the
`/analysis` link once). `grade_game(history, side)` in coach.py grades every
one of your moves against this coach's best.

## Files

- `coach.py` — engine + CLI (also has `explain()` and `grade_game()`)
- `coach_server.py` — stdin daemon: `"history csv|side"` → top move
- `server.py` + `ui/index.html` — local web app
- `SKILL.md` — adoption doc for Hermes/other agents
