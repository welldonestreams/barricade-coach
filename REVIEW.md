# Coach review and training notes — 2026-09-06

The original coach had rule and search errors that made some recommendations
unusable. These are corrected on `improve-coach-search`. The optional MIT-licensed
MCTS engine adds a second search method; neither engine has demonstrated
leaderboard Elo or perfect play.

## Corrected findings

| Severity | Finding | Result |
|---|---|---|
| High | Unlimited walls per player | Ten-wall inventory tracked by side; exhausted players cannot generate or place walls. |
| High | Illegal history accepted by the API | Every ply is validated; input errors identify the ply; UI commits only a validated history. |
| High | Wins/losses treated like ordinary path scores | Goal arrival ends the game and receives decisive search scores. |
| High | Requested depth greater than two was ignored | Actual recursive minimax, alpha-beta pruning, iterative deepening and completed-depth metadata. |
| High | Old Windows server could share the same port | Exclusive loopback binding, engine identity in responses and UI mismatch detection; current port is 8810. |
| Medium | Shortest paths ignored immediate jumps and wall reserves | Legal-next-step race heuristic and a small reserve term; explicitly still approximate. |
| Medium | Tied scores produced different mistake ranks | Equal scores receive equal ranks; unsearched fallbacks are not graded. |
| Medium | Obsolete clickable advice survived state changes | Advice is cleared during validation/search and on opponent turns or game end. |
| Medium | CLI without a side could index a missing argument | Argument parser, inferred turn and clean validation errors. |
| Medium | CORS allowed arbitrary websites; parallel searches unbounded | Same-origin/Host checks, history/budget limits and a single search slot. |
| Low | Wall lines offset using percentages instead of pixels | Horizontal/vertical anchors now use centered pixel offsets; visually checked. |
| Low | Documented daemon did not exist; official-analysis claims stale | Added daemon and replaced unsupported performance/authentication claims. |

The Python search considers every legal wall; it does not discard quiet defensive
placements. Its evaluation is not an exact endgame solver. It can still miss
long fence combinations and choices beyond its completed depth. MCTS explores
selectively and its biased rollouts may also miss critical lines.

## Your five games

The shared records show three wins and two losses. One loss was explicitly on
time; the other ended with the opponent reaching a1. This review grades all 90
of your decisions in those games, and all 238 decisions in the five supplied
high-Elo games: 328 local depth-two grades in total. All completed depth two.
These grades are heuristic comparisons, not official accuracy or proof that an
alternative wins.

| Game | Result for steak2222 | Concrete lesson |
|---|---|---|
| [7bj7bj](https://barricade.gg/analysis?game=7bj7bj) | Win by resignation | You spent all ten walls. Study when to secure the route versus adding more offense. |
| [52s6kd](https://barricade.gg/analysis?game=52s6kd) | Loss on time | Separate clock failure from the exposed corridor described below. |
| [bjyzdh](https://barricade.gg/analysis?game=bjyzdh) | Win by resignation | Keep this as a successful conversion example; winning does not make every earlier move optimal. |
| [81s8yr](https://barricade.gg/analysis?game=81s8yr) | Loss by goal arrival | Watch the a-file escape and your opponent's wall replies before committing to a fence. |
| [6eh4n2](https://barricade.gg/analysis?game=6eh4n2) | Win by resignation | Distinguish useful fence support from walls that merely look threatening. |

In 81s8yr, six of your nine walls did not immediately increase the opponent's
wall-only shortest path. That is a review prompt, not six proven mistakes:
a defensive or preparatory wall may be valuable without changing this number.

### Practice position 1: defend the corridor, not just the distance count

Game 52s6kd, blue to move at ply 18. You played `hg6`. Both wall-only path lengths
stayed red 8 / blue 9, but the legal reply `ve3` would stretch blue's route to 26.
That reply was a counterfactual threat, not the move actually played; the recorded
loss was on time. The tactical coach preferred `vg3`, and a separate seeded MCTS
run also preferred it. `vg3` leaves the immediate distances unchanged too, but
under the tactical coach's best reply the paths become red 8 / blue 11.

Paste this history and choose blue:

```text
e2,e8,e3,e7,e4,e6,hd3,he6,hf3,hc6,hh3,vf5,va6,hg4,vd5,hh5,hd4
```

Drill: explain why `vg3` matters even though it adds no immediate distance.
This is why the old unconditional "never wall unless it adds two squares" rule
was removed from the coaching instructions.

### Practice position 2: stop the flank before spending a quiet wall

Game 81s8yr, red to move at ply 23. The paths were red 11 / blue 9. Your `vg8`
changed neither distance. `vd5` changes blue's route from 9 to 13 without changing
red's 11, and both the tactical search and a separate seeded MCTS run preferred
it here. This is a strong study candidate, not a proven forced win.

```text
e2,e8,e3,e7,e4,e6,hd3,he6,hf3,hc6,hh3,ha6,vc4,ve4,hb5,va4,hf5,va2,vf7,hg6,hh5,hh7
```

Drill: identify the left-flank route, place `vd5`, and trace the opponent's best
reply. Do not stop at the favorable immediate distance change.

## Top-player corpus and openings

The three supplied profiles exposed these game indexes when fetched:
MikeJordan 275, RejeCted_ 760, and owen12345 1,758. Deduplication across profiles
yields 2,770 distinct share codes. This is the set exposed by the public endpoint,
not a guarantee of every game ever played. Individual game bodies are downloaded
with rate-limit backoff; `study/collection-summary.json` records final coverage
when collection finishes. Raw archives and indexes remain local and ignored by Git.

`build_book.py` builds observed opening references from validated downloaded
games, counting a move only when one of the three study players played it.
It requires three examples, covers only the first twelve plies, and shows sample
counts in the UI. The book records human choices, not optimal moves or a trained
neural network. It never overrides search. Its `source_games` field gives the
actual corpus coverage used in that build. Rebuild it after more games download.

The five supplied high-Elo examples include a MikeJordan resignation loss in
5sx6pm. There is no justification for blindly copying all moves from that set.

## AI sources

- [Kyutae Lee / gorisanson](https://github.com/gorisanson/quoridor-ai): MIT-licensed
  MCTS, pinned at e15d6e7825e6f7c2f3a41de0ecef1af7aa290f1e. Its two engine files
  are retained unchanged with the license. Our adapter supplies coordinate
  conversion, budget enforcement, optional seeds and independent tactical checks.
  It bypasses the upstream random opening overrides. No service worker, website
  analytics, installers or npm dependencies were imported.
- [Quoridor AI on Netlify](https://quoridor-ai.netlify.app/): the served application
  bundle configures remote Ishtar-v3 and Ka WebSocket engines. This is evidence
  about the frontend connection, not access to their search implementation or
  weights. No server engine code or license was available in the files inspected;
  none was copied or represented as integrated.
- [Rules and basics](https://quoridorstrats.wordpress.com/beginners-guide-rules-and-basics/)
  and [Blocade strategy](https://playblocade.com/blog/quoridor-strategy): useful
  guidance on route counting, reserves, fences and pawn timing. Treat strategic
  rules of thumb as questions to test against replies, not mandatory move filters.

Public shared-game GETs worked. The official analysis POST returned
`login_required`; the old claim of unauthenticated official grades is removed.

## Verification and practical limits

All 29 automated tests passed. The suite covers legality, inventories, overlap/crossing, path
preservation, jumps/diagonals, terminal positions, immediate wins and threats,
actual depth-three search against an unpruned oracle, timeout fallbacks, tie
ranking, CLI/daemon errors, HTTP validation/origins, exclusive port ownership and
engine dispatch. The independent Python and JavaScript rule implementations
agree on pawn positions, remaining walls, winners and *every legal move* at all
428 positions in the ten-game set (418 played plies plus ten starting positions).

Browser checks cover rejected input retaining the board, valid move entry,
history import, completed-game display, reload persistence, wall geometry and
selected-engine reporting. The current app uses port 8810 because an older
process was serving 8807; that older process was left intact.

`study/benchmark-1s.json` records four paired games from `e2,e8`, two seeds and
both colors, with one second per move: MCTS won all four. `study/benchmark.json`
records the five-second follow-up: MCTS won all four of those games too. These are small local
comparisons against this Python coach, not Barricade rating estimates. Seeds
control randomness but wall-clock budgets can still alter simulation counts.

No neural training, proof of optimal moves, official-engine accuracy comparison,
or guarantee of wins was performed. Live advice still depends on an accurate
move history and enough remaining clock time for the selected search budget.
