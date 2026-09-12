# 4wpcv0 review

The public record is a blue loss by `steak2222` against `bosu`, ending by
resignation after 52 plies. Coach build `70a7cd0752a0` logged 20 of blue's 26
turns. Sixteen logged recommendations were followed and four were not, so this
game is useful for diagnosis but is not a fully coach-guided result.

## Trace comparison

- Ply 6: coach said `vf1`; `hg1` was played. Completed depth-2 search prefers
  `vf1` by 276 heuristic points.
- Ply 26: coach said and blue played `c7`. The production cross-check stopped
  after depth 2. A completed wider depth-3 check instead prefers the defensive
  `hf3`-`hf6` wall family by 236 points over `c7`. This exposed a real search
  control bug and now triggers one wider adaptive check.
- Ply 32: coach said `hd3`; `c6` was played. Completed depth-2 search prefers
  `hd3` by 200 points.
- Ply 34: coach said and blue played `d6`. Depth 2 prefers `hd3`, while completed
  depth 4 scores `d6` and `hd3` equally. This is inconclusive and is not a strict
  repair-gate case.
- Ply 38: coach said `ve4`; `f6` was played. Completed depth-2 search prefers
  `ve4` by 588 points.
- Ply 44: coach said `g7`; `f6` was played. By then the depth-5 endgame warning
  found no saving line and labeled the recommendation as best resistance.

Blue also made untraced moves at plies 36, 40, 46, 48, 50, and 52. The offline
review flags plies 36 and 40, but no saved recommendation exists for those
positions, so they cannot honestly be attributed to the coach.

The machine-readable offline review is `study/regressions/4wpcv0.json`.
