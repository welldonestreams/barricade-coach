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
import re
import time
import math
import argparse
import json
from functools import lru_cache

WIN = 100000

# Path-robustness: a lead is only as safe as the route is hard to block. A pawn
# with a single shortest path can be forced onto a long detour by one wall; a
# pawn with several distinct shortest routes is much harder to wall off. We cap
# the distinct-route count here (more than this is just "safe").
RESILIENCE_CAP = 3
FRAGILITY_CP = 18.0   # centipawn penalty per fragility unit (when the other side can still wall)
ROUTE_OPTION_CP = 12.0
WALL_RESERVE_CP = 24.0
LATE_WALL_RESERVE_CP = 12.0


def parse_history(value):
    if isinstance(value, str):
        value = value.split(',') if value.strip() else []
    return [normalize_move(m) for m in value]

def normalize_move(m):
    if not isinstance(m, str):
        raise ValueError('Move must be text')
    m = m.strip().lower()
    if not re.fullmatch(r'(?:[a-i][1-9]|[hv][a-h][1-8])', m):
        raise ValueError(f'Invalid notation: {m!r}; use e2 or hd4')
    return m

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
    if not isinstance(w, str) or not re.fullmatch(r'[hv][a-h][1-8]', w):
        return False
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
    return _blocked(frozenset(ws))

@lru_cache(maxsize=4096)
def _blocked(ws):
    """ws: set of wall strings -> frozenset of blocked adjacency pairs (both orders)."""
    s = set()
    for w in ws:
        for a, b in wall_segments(w):
            s.add((a, b)); s.add((b, a))
    return frozenset(s)

def path_exists(ws, start, goal_row, ignore=None):
    """BFS over cells; only walls block. Returns True if reachable."""
    return shortest(ws, start, goal_row) is not None

@lru_cache(maxsize=4096)
def graph(ws):
    blk = blocked(ws)
    return tuple(tuple(n[1]*9+n[0] for n in ((x+1,y),(x-1,y),(x,y+1),(x,y-1))
                       if 0 <= n[0] < 9 and 0 <= n[1] < 9 and ((x,y),n) not in blk)
                 for y in range(9) for x in range(9))

@lru_cache(maxsize=8192)
def distances(ws, goal_row):
    adj = graph(ws)
    dist = [99]*81
    q = deque(range(goal_row*9, goal_row*9+9))
    for i in q:
        dist[i] = 0
    while q:
        i = q.popleft()
        for j in adj[i]:
            if dist[j] == 99:
                dist[j] = dist[i]+1
                q.append(j)
    return tuple(dist)

def shortest(ws, start, goal_row, blockers=()):
    if not blockers:
        d = distances(frozenset(ws), goal_row)[start[1]*9+start[0]]
        return None if d == 99 else d
    return _shortest(frozenset(ws), start, goal_row, tuple(blockers))

@lru_cache(maxsize=65536)
def _shortest(ws, start, goal_row, blockers=()):
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

@lru_cache(maxsize=8192)
def path_resilience(ws, start, goal_row, cap=RESILIENCE_CAP):
    """Number of files used by a path within two steps of shortest, capped.

    A two-step allowance recognizes useful nearby escape lanes on an open board
    while excluding distant horizontal detours. This measures practical route
    breadth before the raw shortest distance worsens.
    """
    adj = graph(frozenset(ws))
    def bfs(sources):
        dist = [99]*81
        dq = deque()
        for s in sources:
            dist[s] = 0
            dq.append(s)
        while dq:
            i = dq.popleft()
            for j in adj[i]:
                if dist[j] == 99:
                    dist[j] = dist[i] + 1
                    dq.append(j)
        return dist
    to_goal = distances(frozenset(ws), goal_row)
    s0 = start[1]*9 + start[0]
    if to_goal[s0] == 99:
        return 0
    from_start = bfs([s0])
    d0 = to_goal[s0]
    files = {cell % 9 for cell in range(81)
             if to_goal[cell] != 99 and from_start[cell] + to_goal[cell] <= d0+2}
    return min(len(files), cap)

# --- game state -----------------------------------------------------------
class Game:
    def __init__(self, history=()):
        self.history = []
        self.walls = set()
        self.pawns = {RED: STARTS[RED], BLUE: STARTS[BLUE]}
        self.remaining = {RED: 10, BLUE: 10}
        for i, mv in enumerate(parse_history(history)):
            try:
                self.apply(mv)
            except ValueError as exc:
                raise ValueError(f'Ply {i + 1}: {exc}') from exc

    @property
    def to_move(self):
        return RED if len(self.history) % 2 == 0 else BLUE

    def apply(self, mv):
        mv = normalize_move(mv)
        if self.winner is not None:
            raise ValueError('Game already finished')
        if len(mv) == 3:
            valid = self.remaining[self.to_move] > 0 and self.is_wall_legal(mv)
        else:
            valid = mv in [f'{LETTERS[p[0]]}{p[1]+1}' for _, p in self.pawn_moves(self.to_move)]
        if not valid:
            raise ValueError(f'Illegal move {mv}')
        self._play(mv)

    @property
    def winner(self):
        return next((s for s in (RED, BLUE) if self.pawns[s][1] == GOALS[s]), None)

    def copy(self):
        g = object.__new__(type(self))
        g.history = self.history.copy()
        g.walls = self.walls.copy()
        g.pawns = self.pawns.copy()
        g.remaining = self.remaining.copy()
        return g

    @classmethod
    def from_position(cls, red_sq, blue_sq, walls, side, red_left=10, blue_left=10):
        """Build a Game from an explicit position instead of a move history.
        red_sq/blue_sq: 'e5' squares; walls: list of wall notations; side: RED/BLUE
        (the side to move). Used by the live overlay so a wrong/partial move
        history can never desync the board the coach reasons about."""
        if side not in (RED, BLUE):
            raise ValueError('Invalid side')
        for sq in (red_sq, blue_sq):
            if not isinstance(sq, str) or not re.fullmatch(r'[a-i][1-9]', sq):
                raise ValueError('Invalid pawn square')
        if red_sq == blue_sq:
            raise ValueError('Pawns cannot occupy the same square')
        for count in (red_left, blue_left):
            if type(count) is not int or not 0 <= count <= 10:
                raise ValueError('Wall reserves must be integers from 0 to 10')
        walls = list(walls)
        if len(set(walls)) != len(walls) or len(walls) != 20-red_left-blue_left:
            raise ValueError('Placed walls and remaining counts disagree')
        g = cls()
        g.history = ['~'] if side == BLUE else []
        g.pawns = {RED: (letter_idx(red_sq[0]), int(red_sq[1])-1),
                   BLUE: (letter_idx(blue_sq[0]), int(blue_sq[1])-1)}
        for wall in walls:
            if not isinstance(wall, str) or not re.fullmatch(r'[hv][a-h][1-8]', wall):
                raise ValueError('Invalid wall notation')
            if not g.is_wall_legal(wall):
                raise ValueError('Walls overlap, cross, or block a goal')
            g.walls.add(wall)
        if red_sq[1] == '9' and blue_sq[1] == '1':
            raise ValueError('Both players cannot have won')
        g.remaining = {RED: red_left, BLUE: blue_left}
        return g

    def _play(self, mv):
        """Internal only: apply a move already validated or generated as legal."""
        mover = self.to_move
        if mv[0] in 'hv' and len(mv) == 3:
            self.walls.add(mv)
            self.remaining[mover] -= 1
        else:
            # pawn move (destination square)
            col, row = letter_idx(mv[0]), int(mv[1]) - 1
            self.pawns[mover] = (col, row)
        self.history.append(mv)

    def pawn_moves(self, side):
        if self.winner is not None:
            return []
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
        side = self.to_move if side is None else side
        if self.winner is not None or self.remaining[side] == 0:
            return []
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
        """Heuristic centipawns; lower is better. Terminal results dominate.

        Terms, in order of weight:
          1. race: 100 * (my distance - their distance) to goal. DOMINANT.
          2. wall reserves: 8 * (their walls left - my walls left).
          3. fragility: a lead only matters if my route is hard to block. A
             pawn with a single shortest path (fragility 2) that the opponent
             can still wall off is worth less than the raw race says, so we
             discount my position by FRAGILITY_CP per missing alternative route
             and reward the same weakness in the opponent. This is what makes
             the coach spend a wall to protect its own corridor instead of
             blindly advancing when "ahead". Gated on the other side having
             walls left: with no walls in hand, nobody can be blocked further.
          4. tempo: small side-to-move term."""
        if self.winner is not None:
            return -WIN if self.winner == side else WIN
        mine = self.race_distance(side)
        theirs = self.race_distance(1-side)
        if mine is None: mine = 99
        if theirs is None: theirs = 99
        reserve_cp=WALL_RESERVE_CP if sum(self.remaining.values())>=10 else LATE_WALL_RESERVE_CP
        score = 100 * (mine - theirs) + reserve_cp * (self.remaining[1-side] - self.remaining[side])
        if self.remaining[1-side] > 0:
            my_res = path_resilience(frozenset(self.walls), self.pawns[side], GOALS[side])
            score += FRAGILITY_CP * (RESILIENCE_CAP - my_res)
            my_options = self.route_options(side)
            score += ROUTE_OPTION_CP * (RESILIENCE_CAP - my_options)
        if self.remaining[side] > 0:
            their_res = path_resilience(frozenset(self.walls), self.pawns[1-side], GOALS[1-side])
            score -= FRAGILITY_CP * (RESILIENCE_CAP - their_res)
            their_options = self.route_options(1-side)
            score -= ROUTE_OPTION_CP * (RESILIENCE_CAP - their_options)
        return score + (-50 if self.to_move == side else 50)

    def race_distance(self, side):
        """One legal pawn turn then wall-only distance: local jump heuristic."""
        return min((1 + shortest(self.walls, p, GOALS[side]) for _, p in self.pawn_moves(side)), default=99)

    def route_options(self, side, cap=RESILIENCE_CAP):
        """Count legal pawn continuations within two steps of best, capped.

        This catches a pawn entering a one-exit corridor before the raw shortest
        path becomes worse. The term is active only while the opponent can
        still place walls.
        """
        moves=self.pawn_moves(side)
        if not moves:return 0
        lengths=[1+shortest(self.walls,p,GOALS[side]) for _,p in moves]
        best=min(lengths)
        return min(cap,sum(length<=best+2 for length in lengths))

# --- coaching search ------------------------------------------------------
class SearchTimeout(Exception):
    pass

def search(history, side=None, depth=2, time_limit=5.0):
    """Full-width iterative deepening. Only complete, equally deep root scores survive."""
    if not isinstance(depth, int) or not 1 <= depth <= 12:
        raise ValueError('Depth must be an integer from 1 to 12')
    if not isinstance(time_limit, (int, float)) or not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('Time limit must be positive and finite')
    started = time.monotonic()
    deadline = started + time_limit
    g = Game(history)
    side = g.to_move if side is None else side
    if side not in (RED, BLUE) or g.to_move != side:
        raise ValueError(f'History says {"red" if g.to_move == RED else "blue"} to move')
    return _search_game(g, side, depth, time_limit, started, deadline)


def search_position(red_sq, blue_sq, walls, side, depth=2, time_limit=5.0,
                    red_left=10, blue_left=10):
    """Search from an explicit position (live overlay path) instead of history.
    side: RED/BLUE = the side to move."""
    if not isinstance(depth, int) or not 1 <= depth <= 12:
        raise ValueError('Depth must be an integer from 1 to 12')
    if not isinstance(time_limit, (int, float)) or not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('Time limit must be positive and finite')
    if side not in (RED, BLUE):
        raise ValueError('side must be red or blue')
    started = time.monotonic()
    deadline = started + time_limit
    g = Game.from_position(red_sq, blue_sq, walls, side, red_left, blue_left)
    return _search_game(g, side, depth, time_limit, started, deadline)


def _search_game(g, side, depth, time_limit, started, deadline):
    if g.winner is not None:
        return dict(scored=[], depth=0, nodes=0, elapsed=time.monotonic()-started, timed_out=False, winner=g.winner, principal_variation=[])
    nodes = 0
    transpositions = {}
    tt_hits = 0
    def state_key(state):
        """History-independent board key for exact completed-depth reuse."""
        return (state.pawns[RED], state.pawns[BLUE], tuple(sorted(state.walls)),
                state.remaining[RED], state.remaining[BLUE], state.to_move)
    def check_time():
        if time.monotonic() >= deadline:
            raise SearchTimeout
    def children(state):
        out = []
        candidates = [f'{LETTERS[p[0]]}{p[1]+1}' for _, p in state.pawn_moves(state.to_move)]
        if state.remaining[state.to_move]:
            for typ in 'hv':
                for fx in LETTERS[:8]:
                    for y in range(1, 9):
                        check_time()
                        w = f'{typ}{fx}{y}'
                        if state.is_wall_legal(w):
                            candidates.append(w)
        for mv in candidates:
            check_time()
            child = state.copy()
            child._play(mv)
            out.append((child.eval_side(side), mv, child))
        out.sort(key=lambda x: (x[0] if state.to_move == side else -x[0], len(x[1]), x[1]))
        return out
    def minimax(state, left, alpha, beta, ply):
        nonlocal nodes, tt_hits
        check_time()
        nodes += 1
        if state.winner is not None:
            return (-WIN + ply if state.winner == side else WIN - ply), []
        if left == 0:
            return state.eval_side(side), []
        key=(state_key(state),left)
        cached=transpositions.get(key)
        if cached is not None:
            tt_hits += 1
            return cached
        maximizing = state.to_move != side
        value, line = (-math.inf if maximizing else math.inf), []
        complete=True
        for _, mv, child in children(state):
            v, tail = minimax(child, left-1, alpha, beta, ply+1)
            if (v > value if maximizing else v < value):
                value, line = v, [mv] + tail
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                complete=False
                break
        # Alpha-beta cutoffs are bounds. Cache only complete exact results.
        if complete:
            transpositions[key]=(value,line)
        return value, line
    fallback = min(g.pawn_moves(side), key=lambda x: (shortest(g.walls, x[1], GOALS[side]), x[1]))[1]
    fallback = f'{LETTERS[fallback[0]]}{fallback[1]+1}'
    scored, completed, pv, timed_out = [(g.eval_side(side), fallback)], 0, [fallback], False
    try:
        roots = children(g)
        for current in range(1, depth+1):
            iteration, lines = [], {}
            for _, mv, child in roots:
                v, tail = minimax(child, current-1, -math.inf, math.inf, 1)
                iteration.append((v, mv))
                lines[mv] = [mv] + tail
            iteration.sort(key=lambda x: (x[0], len(x[1]), x[1]))
            scored, completed, pv = iteration, current, lines[iteration[0][1]]
            order = {mv: i for i, (_, mv) in enumerate(scored)}
            roots.sort(key=lambda x: order[x[1]])
    except SearchTimeout:
        timed_out = True
    return dict(scored=scored, depth=completed, nodes=nodes, tt_hits=tt_hits,
                elapsed=time.monotonic()-started,
                timed_out=timed_out, winner=None, principal_variation=pv)

def best_moves(history, side, n=3, depth=2, verbose=True, time_limit=5.0):
    scored = search(history, side, depth, time_limit)['scored']
    return scored[:n], scored

def win_move(g2, side):
    """True only when that pawn has reached its goal."""
    return g2.winner == side

def analyze(history, side):
    top, allm = best_moves(history, side, n=5)
    return top, allm

def explain(history, side, move):
    """Detailed look at one candidate: resulting paths + opponent's best reply."""
    g = Game(history)
    if g.to_move != side:
        raise ValueError('Wrong side to move')
    g.apply(move)
    if g.winner is not None:
        return f'Play {move}: reaches the goal and wins immediately.'
    me, opp = (RED, BLUE) if side == RED else (BLUE, RED)
    def sps(gg):
        return (shortest(gg.walls, gg.pawns[RED], GOALS[RED]),
                shortest(gg.walls, gg.pawns[BLUE], GOALS[BLUE]))
    after = sps(g)
    # opponent best reply (maximizes my score)
    worst = None
    for rep in g.moves(opp):
        g3 = g.copy()
        g3._play(rep)
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

def grade_game(history, side, n=4, depth=2, time_limit=5.0, total_time=None):
    """Grade every ply of `side` in history using coach best-moves.
    Returns list of dicts: ply, move, rank, best, score_played, score_best, n_cand."""
    deadline = time.monotonic()+total_time if total_time is not None else math.inf
    history = Game(history).history
    if side not in (RED, BLUE):
        raise ValueError('Side must be red (0) or blue (1)')
    rows = []
    for i in range(len(history)):
        if i % 2 != (0 if side == RED else 1):
            continue
        state = history[:i]
        remaining = deadline-time.monotonic()
        if remaining <= 0: break
        result = search(state, side, depth, min(time_limit, remaining))
        scored = result['scored']
        if result['depth'] == 0:
            rows.append(dict(ply=i+1, move=history[i], rank=None, best=None,
                             score_played=None, score_best=None, delta=None,
                             depth=0, timed_out=True))
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
                rank = 1 + sum(s < sc for s, _ in scored)
                break
        rows.append(dict(ply=i+1, move=mv, rank=rank if played_sc is not None else None,
                         best=best_mv, score_played=played_sc, score_best=best_sc,
                         depth=result['depth'], timed_out=result['timed_out'],
                         delta=(played_sc - best_sc) if played_sc is not None else None))
    return rows

def replay_valid(history):
    g = Game()
    for i, mv in enumerate(history):
        side = g.to_move
        try:
            g.apply(mv)
        except ValueError:
            return False, i, mv, side
    return True, len(history), None, None

def main():
    parser = argparse.ArgumentParser(description='Local Quoridor coach; heuristic advice, not perfect play.')
    parser.add_argument('history', nargs='?', default='')
    parser.add_argument('side', nargs='?', choices=['red', 'blue'])
    parser.add_argument('mode', nargs='?', choices=['explain'])
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--engine', choices=['python','mcts'], default='python')
    parser.add_argument('--rollouts', type=int, default=60000)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    try:
        hist = parse_history(args.history)
        side = None if args.side is None else (RED if args.side == 'red' else BLUE)
        if args.engine == 'mcts':
            import mcts_coach
            result = mcts_coach.search(hist, side, args.seconds, args.rollouts, args.seed)
        else:
            result = search(hist, side, args.depth, args.seconds)
        if args.json:
            print(json.dumps(result))
        elif result['winner'] is not None:
            print(f'Game over: {"red" if result["winner"] == RED else "blue"} won')
        else:
            if args.engine == 'mcts':
                print(f'MCTS: {result["simulations"]} simulations, {result["elapsed"]:.2f}s; score = negative visits (tactical overrides take priority)')
            else:
                print(f'Completed depth {result["depth"]}, {result["elapsed"]:.2f}s; heuristic score, lower is better')
            if result['timed_out']:
                print('Time budget reached; showing last complete iteration (depth 0 means fallback).')
            for score, move in result['scored'][:6]:
                print(f'{move:5s} {score:+d}')
            if args.mode == 'explain':
                print(explain(hist, Game(hist).to_move, result['scored'][0][1]))
    except ValueError as exc:
        parser.error(str(exc))

if __name__ == '__main__':
    main()
