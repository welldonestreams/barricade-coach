#!/usr/bin/env python3
"""
Barricade/Quoridor coach: given a move history in Barricade.gg notation,
returns the best moves for the side to move, using shortest-path math
(the 'one-number rule') with shallow minimax.
Notation (verified against barricade.gg games):
  pawn move = destination square, e.g. 'e4' (files a-i, ranks 1-9, rank1 = red's home)
  wall      = hXY (horizontal: blocks vertical crossing between ranks Y and Y+1,
              spanning columns X and X+1) or vXY (vertical: blocks horizontal
              crossing between columns X and X+1, spanning ranks Y..Y+1).
  Walls with the same anchor (letter+rank) cross -> illegal. Adjacent parallel
  walls (same orientation sharing an edge) -> illegal. Same exact slot -> illegal.
  A wall is illegal if it leaves either pawn with no path to its goal.
"""

from collections import deque
import sys

LETTERS = 'abcdefghi'
FILES = {c: i for i, c in enumerate(LETTERS)}   # a=0..h? i=8
RED, BLUE = 0, 1
# red starts bottom (rank1, row0), moves up to rank9 (row8); blue opposite.
STARTS = {RED: (4, 0), BLUE: (4, 8)}
GOALS = {RED: 8, BLUE: 0}   # row index of goal line

def letter_idx(x): return FILES[x]

# --- geometry helpers -----------------------------------------------------
# Horizontal wall hXY: separates rows rank Y-1 and rank Y (row idx Y-1 <-> Y)
#   across columns X..X+1. Valid Y in 1..8, X in a..h.
# Vertical wall vXY: separates columns X and X+1 across rows rank Y..Y+1
#   (row idx Y-1 and Y). Valid X in a..h, Y in 1..8.
# (row idx = rank-1; rank1 = row0 ... rank9 = row8)

def H_rows(h):  # h = (file_letter, rank) -> set of blocked (rowA,rowB) pairs per column handled in move fn
    pass

def wall_segments(w):
    """Return list of blocked cell-pairs (tuples of 2 cells) for wall w.
    H: horizontal wall blocks vertical adjacency for columns X,X+1 between rows Y-1 and Y.
    V: vertical wall blocks horizontal adjacency for rows Y-1,Y between columns X,X+1.
    Cells as (col_idx, row_idx)."""
    typ = w[0]
    fx, y = letter_idx(w[1]), int(w[2:])
    segs = set()
    if typ == 'h':
        for c in (fx, fx + 1):
            segs.add(((c, y - 1), (c, y)))
    else:
        for r in (y - 1, y):
            segs.add(((fx, r), (fx + 1, r)))
    return segs

def wall_anchor_line(w):
    """For crossing/overlap logic return normalized key."""
    return (w[0], w[1], int(w[2:]))

def legal_wall(ws, w, pawns):
    """ws: set of existing wall keys ('hXY'/'vXY'); w new wall; pawns both positions."""
    typ, fx, y = w[0], w[1], int(w[2:])
    fxi = letter_idx(fx)
    if not (0 <= fxi <= 7 and 1 <= y <= 8):
        return False
    if w in ws:
        return False
    # parallel-adjacent / overlapping check
    for other in ws:
        ot, ox, oy = other[0], letter_idx(other[1]), int(other[2:])
        if ot == 'h' and typ == 'h':
            if oy == y and abs(ox - fxi) <= 1:
                return False
        if ot == 'v' and typ == 'v':
            if ox == fxi and abs(oy - y) <= 1:
                return False
        if ot != typ and ox == fxi and oy == y:
            return False  # cross
    # must leave a path for both pawns (ignore other pawn as obstacle)
    for p, row_target in (pawns[0], GOALS[RED]), (pawns[1], GOALS[BLUE]):
        if not path_exists(ws | {w}, p, row_target):
            return False
    return True

def blocked(ws):
    """ws: set of wall strings -> frozenset of blocked adjacency pairs (both orders)."""
    s = set()
    for w in ws:
        for a, b in wall_segments(w):
            s.add((a, b)); s.add((b, a))
    return s

def path_exists(ws, start, goal_row, ignore=None):
    """BFS over cells; only walls block. Returns True if reachable."""
    blk = blocked(ws)
    q = deque([start]); seen = {start}
    while q:
        c, r = q.popleft()
        if r == goal_row:
            return True
        for dc, dr in ((1,0),(-1,0),(0,1),(0,-1)):
            nc, nr = c+dc, r+dr
            if 0 <= nc <= 8 and 0 <= nr <= 8 and (nc, nr) not in seen:
                if ((c, r), (nc, nr)) in blk:
                    continue
                seen.add((nc, nr)); q.append((nc, nr))
    return False

def shortest(ws, start, goal_row, blockers=()):
    """BFS step count; walls + blocker cells block. Returns None if unreachable."""
    blk = blocked(ws)
    q = deque([(start, 0)]); seen = {start}
    while q:
        (c, r), d = q.popleft()
        if r == goal_row:
            return d
        for dc, dr in ((1,0),(-1,0),(0,1),(0,-1)):
            nc, nr = c+dc, r+dr
            if 0 <= nc <= 8 and 0 <= nr <= 8 and (nc, nr) not in seen and (nc, nr) not in blockers:
                if ((c, r), (nc, nr)) in blk:
                    continue
                seen.add((nc, nr)); q.append(((nc, nr), d+1))
    return None

# --- game state -----------------------------------------------------------
class Game:
    def __init__(self, history=()):
        self.history = []
        self.walls = set()
        self.pawns = {RED: STARTS[RED], BLUE: STARTS[BLUE]}
        for mv in history:
            self.apply(mv)

    @property
    def to_move(self):
        return RED if len(self.history) % 2 == 0 else BLUE

    def apply(self, mv):
        mover = self.to_move
        if mv[0] in 'hv' and len(mv) == 3:
            self.walls.add(mv)
        else:
            # pawn move (destination square)
            col, row = letter_idx(mv[0]), int(mv[1]) - 1
            self.pawns[mover] = (col, row)
        self.history.append(mv)

    def pawn_moves(self, side):
        p = self.pawns[side]
        opp = self.pawns[1 - side]
        blk = blocked(self.walls)
        res = []
        for dc, dr in ((1,0),(-1,0),(0,1),(0,-1)):
            n = (p[0]+dc, p[1]+dr)
            if not (0 <= n[0] <= 8 and 0 <= n[1] <= 8):
                continue
            if ((p, n)) in blk:
                continue
            if n == opp:
                # jump
                b = (n[0]+dc, n[1]+dr)
                if 0 <= b[0] <= 8 and 0 <= b[1] <= 8 and b not in (opp,) and (n, b) not in blk:
                    # straight jump over
                    res.append(('jump', b))
                else:
                    # jump blocked: diagonal to either side of opp
                    for sdc, sdr in ((dr, -dc), (-dr, dc)):
                        d = (n[0]+sdc, n[1]+sdr)
                        if 0 <= d[0] <= 8 and 0 <= d[1] <= 8 and d != opp and (n, d) not in blk:
                            res.append(('jump', d))
            else:
                res.append(('step', n))
        return res

    def is_wall_legal(self, w):
        """Pure legality: in-bounds, no overlap/adjacency/cross with existing walls,
        and both pawns keep a path to their goals."""
        return legal_wall(self.walls, w, tuple(self.pawns.values()))

    def legal_walls(self, side=None):
        """All legal wall placements (full scan - 128 slots max)."""
        out = []
        for typ in ('h', 'v'):
            for fx in LETTERS[:8]:
                for y in range(1, 9):
                    w = f'{typ}{fx}{y}'
                    if w in self.walls:
                        continue
                    if legal_wall(self.walls, w, tuple(self.pawns.values())):
                        out.append(w)
        return out

    def moves(self, side):
        m = []
        for kind, cell in self.pawn_moves(side):
            m.append(f"{LETTERS[cell[0]]}{cell[1]+1}")
        m += self.legal_walls(side)
        return m

    def eval_side(self, side):
        """Score from `side` perspective: my shortest - their shortest (lower=better for me).
        If one pawn has no path -> treat as huge loss for it."""
        mine = shortest(self.walls, self.pawns[side], GOALS[side])
        theirs = shortest(self.walls, self.pawns[1-side], GOALS[1-side])
        if mine is None: mine = 99
        if theirs is None: theirs = 99
        return mine - theirs

# --- coaching search ------------------------------------------------------
def best_moves(history, side, n=3, depth=2, verbose=True):
    g = Game(history)
    assert g.to_move == side, f"history parity: {'red' if g.to_move==RED else 'blue'} to move, not {side}"
    cands = g.moves(side)
    scored = []
    for mv in cands:
        g2 = Game(history)
        g2.apply(mv)
        v = g2.eval_side(side)  # after my move, their turn: they'll minimize my score
        if depth >= 2 and not win_move(g2, side):
            # opponent best reply: they maximize my score (mySP-theirSP high = bad for me)
            best_rep = None
            for rep in g2.moves(1 - side):
                g3 = Game(g2.history)
                g3.apply(rep)
                rv = g3.eval_side(side)
                if best_rep is None or rv > best_rep[0]:
                    best_rep = (rv, rep)
            v = best_rep[0] if best_rep is not None else v
        scored.append((v, mv))
    scored.sort(key=lambda x: x[0])
    if verbose:
        pass
    return scored[:n], scored

def win_move(g2, side):
    """After a move, did that side just win or is it 1 step away on its next turn?"""
    sp = shortest(g2.walls, g2.pawns[side], GOALS[side])
    return sp == 0

def analyze(history, side):
    top, allm = best_moves(history, side, n=5)
    return top, allm

def explain(history, side, move):
    """Detailed look at one candidate: resulting paths + opponent's best reply."""
    g = Game(history)
    g.apply(move)
    me, opp = (RED, BLUE) if side == RED else (BLUE, RED)
    def sps(gg):
        return (shortest(gg.walls, gg.pawns[RED], GOALS[RED]),
                shortest(gg.walls, gg.pawns[BLUE], GOALS[BLUE]))
    after = sps(g)
    # opponent best reply (maximizes my score)
    worst = None
    for rep in g.moves(opp):
        g3 = Game(g.history)
        g3.apply(rep)
        rv = g3.eval_side(side)
        if worst is None or rv > worst[0]:
            worst = (rv, rep)
    if worst:
        g4 = Game(g.history)
        g4.apply(worst[1])
        final = sps(g4)
    names = {RED: 'red', BLUE: 'blue'}
    lines = [f"play {move}:",
             f"  red SP={after[0]}, blue SP={after[1]}",
             f"  their best reply: {worst[1]} -> red SP={final[0]}, blue SP={final[1]} (score {worst[0]:+d})"]
    return "\n".join(lines)

def grade_game(history, side, n=4):
    """Grade every ply of `side` in history using coach best-moves.
    Returns list of dicts: ply, move, rank, best, score_played, score_best, n_cand."""
    rows = []
    for i in range(len(history)):
        if i % 2 != (0 if side == RED else 1):
            continue
        state = history[:i]
        try:
            scored = best_moves(state, side, n=0)[1]  # full sorted list
        except Exception:
            rows.append(dict(ply=i, move=history[i], rank=None, best=None,
                             score_played=None, score_best=None, delta=None))
            continue
        if not scored:
            rows.append(dict(ply=i, move=history[i], rank=None, best=None,
                             score_played=None, score_best=None, delta=None))
            continue
        mv = history[i]
        best_sc, best_mv = scored[0]
        played_sc = None
        for sc, m in scored:
            if m == mv:
                played_sc = sc
                rank = scored.index((sc, m)) + 1
                break
        rows.append(dict(ply=i, move=mv, rank=rank if played_sc is not None else None,
                         best=best_mv, score_played=played_sc, score_best=best_sc,
                         delta=(played_sc - best_sc) if played_sc is not None else None))
    return rows

def replay_valid(history):
    g = Game()
    for i, mv in enumerate(history):
        side = g.to_move
        legal = g.moves(side)
        if mv not in legal:
            return False, i, mv, side
        g.apply(mv)
    return True, len(history), None, None

if __name__ == '__main__':
    hist = sys.argv[1].split(',') if len(sys.argv) > 1 and sys.argv[1] else []
    side = RED if sys.argv[2].lower() in ('red', 'p1', 'r') else BLUE if len(sys.argv) > 2 else None
    if side is None:
        # default: whoever is to move per parity
        g = Game(hist)
        side = g.to_move
    ok, n, bad, bside = replay_valid(hist)
    print(f"history replay: {'OK' if ok else f'INVALID at ply {n} move {bad} (side {bside})'}")
    top, allm = best_moves(hist, side, n=6)
    print(f"side to move: {'RED' if side == RED else 'BLUE'}")
    g = Game(hist)
    print(f"pawn SP -> red: {shortest(g.walls, g.pawns[RED], GOALS[RED])}, blue: {shortest(g.walls, g.pawns[BLUE], GOALS[BLUE])}")
    print("top moves (score = mySP-theirSP after opp best reply; lower better):")
    for v, mv in top:
        print(f"   {mv:5s}  {v:+d}")
    if len(sys.argv) > 3 and sys.argv[3] == 'explain' and top:
        print("---")
        print(explain(hist, side, top[0][1]))
