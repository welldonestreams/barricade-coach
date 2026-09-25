# 0fsy74 screenshot review

The public record is a red win by `steak2222` against `zai0299`. At ply 53,
red had no walls and the legal pawn moves were `h5`, `i6`, and `g6`.

The deployed four-second MCTS preferred `g6`. Its selective depth-5
cross-check instead scored `h5` best and overrode MCTS by 188 heuristic points,
which produced the backwards-looking screenshot recommendation. A completed
depth-6 check tied `h5` and `i6` while still preferring both to `g6`; three
independent 15-second MCTS runs initially preferred `g6`, though one of five
fixed-seed runs still selected `h5`. The methods disagree, so the selective
heuristic does not have enough evidence to overrule MCTS.

The safeguard now keeps the deeper check for warnings and explanation when the
mover has no walls, but it cannot replace MCTS with another pawn move. This rare
endgame also gets eight independent MCTS roots for 15 seconds; five fixed-seed
production runs then returned `g6` 5/5, at roughly one million simulations each.
