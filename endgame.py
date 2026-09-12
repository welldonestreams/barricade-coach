"""Exact pawn-only goal reachability, including jumps, turns and cycles.

Applies only when BOTH players have no walls. Clocks are outside this model.
An incomplete graph search returns None, never an 'exact' partial answer.
"""
from collections import defaultdict, deque
import heapq
import time
import coach as c


def solve(game, seconds=.75):
    if game.winner is not None or any(game.remaining.values()): return None
    started=time.monotonic();deadline=started+seconds
    def key(g):
        return (g.pawns[0],g.pawns[1],g.to_move)
    root=key(game);todo=deque([root]);seen={root};edges={};parents=defaultdict(list)
    values={};distances={};heap=[]
    while todo:
        if time.monotonic()>=deadline: return None
        state=todo.popleft()
        g=game.copy();g.pawns={0:state[0],1:state[1]};g.history=['~'] if state[2] else []
        if g.winner is not None:
            values[state]=1 if g.winner==g.to_move else -1
            distances[state]=0;heapq.heappush(heap,(0,state));edges[state]=[];continue
        successors=[]
        for _,dest in g.pawn_moves(g.to_move):
            child=(dest,state[1],1) if state[2]==0 else (state[0],dest,0)
            move=c.LETTERS[dest[0]]+str(dest[1]+1)
            successors.append((move,child));parents[child].append(state)
            if child not in seen:seen.add(child);todo.append(child)
        edges[state]=successors
    remaining={s:len(e) for s,e in edges.items()};longest=defaultdict(int)
    while heap:
        if time.monotonic()>=deadline: return None
        distance,child=heapq.heappop(heap)
        for parent in parents[child]:
            if parent in values:continue
            if values[child]==-1:
                values[parent]=1;distances[parent]=distance+1
                heapq.heappush(heap,(distance+1,parent))
            else:
                remaining[parent]-=1;longest[parent]=max(longest[parent],distance+1)
                if remaining[parent]==0:
                    values[parent]=-1;distances[parent]=longest[parent]
                    heapq.heappush(heap,(longest[parent],parent))
    def ranked(state):
        out=[]
        for move,child in edges[state]:
            value=-values.get(child,0);distance=distances.get(child,0)+1
            score=(-c.WIN+distance) if value==1 else (c.WIN-distance if value==-1 else 0)
            out.append((score,move,child))
        return sorted(out)
    ordered=ranked(root);pv=[];state=root;visited=set()
    for _ in range(16):
        if state in visited or not edges[state]:break
        visited.add(state);_,move,state=ranked(state)[0];pv.append(move)
    outcome=values.get(root,0)
    return dict(engine='exact',exact=True,scored=[(s,m) for s,m,_ in ordered],
                depth=None,elapsed=time.monotonic()-started,timed_out=False,
                principal_variation=pv,winner=None,nodes=len(seen),
                outcome={1:'forced goal',0:'no forced goal',-1:'opponent forced goal'}[outcome],
                goal_plies=distances.get(root),model='both inventories empty; clocks excluded')
