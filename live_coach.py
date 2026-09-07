"""Verified live-position contract shared by the overlay and regression tests."""
import hashlib
import json
import time
import coach as c
import advice


def position(g):
    return dict(red=c.LETTERS[g.pawns[c.RED][0]]+str(g.pawns[c.RED][1]+1),
                blue=c.LETTERS[g.pawns[c.BLUE][0]]+str(g.pawns[c.BLUE][1]+1),
                walls=sorted(g.walls), side='red' if g.to_move == c.RED else 'blue',
                red_left=g.remaining[c.RED], blue_left=g.remaining[c.BLUE])


def explain(g, result):
    if not result['scored']:
        return 'Game finished.'
    move=result['scored'][0][1]
    child=g.copy(); child.apply(move)
    if child.winner is not None:
        return f'{move} reaches the goal and wins immediately.'
    side=g.to_move
    before=[c.shortest(g.walls,g.pawns[s],c.GOALS[s]) for s in (side,1-side)]
    after=[c.shortest(child.walls,child.pawns[s],c.GOALS[s]) for s in (side,1-side)]
    jump=sum(abs(child.pawns[side][i]-g.pawns[side][i]) for i in (0,1))>1
    kind='Uses one wall' if len(move)==3 else ('Jumps over the opponent' if jump else 'Moves your pawn')
    text=f'{kind}. Your route {before[0]} → {after[0]}; opponent route {before[1]} → {after[1]} (squares, excluding jumps).'
    pv=result.get('principal_variation',[])
    if len(pv)>1:
        text+=f' Searched reply: {pv[1]}.'
    if result.get('depth',0)==0:
        text+=' Legal fallback; search did not finish a depth.'
    else:
        text+=f" Compared all legal moves through depth {result['depth']}; deeper threats may remain."
    return text


def query(params):
    started=time.monotonic()
    hist=c.parse_history(params.get('h',''))
    if len(params.get('h',''))>8000:
        raise ValueError('History too long')
    g=c.Game(hist)
    actual=position(g)
    # The visible board and numbered move list must independently agree.
    for field in ('red','blue','walls','red_left','blue_left'):
        expected=actual[field]
        supplied=params.get(field)
        if field=='walls': supplied=sorted(filter(None,(supplied or '').split(',')))
        elif field.endswith('_left'):
            try: supplied=int(supplied)
            except (TypeError,ValueError): raise ValueError('Cannot read remaining walls')
        if supplied!=expected:
            raise ValueError(f'Board/history mismatch: {field}; waiting for a stable board')
    seconds=float(params.get('seconds','4'))
    if not 0<seconds<=15:
        raise ValueError('Time budget must be greater than zero and at most 15 seconds')
    # A short full-width search in the uncontested central opening, not a
    # hard-coded move. As soon as pawns approach or a wall appears, use full budget.
    early=not g.walls and all(p[0]==4 for p in g.pawns.values()) and abs(g.pawns[0][1]-g.pawns[1][1])>3
    budget=min(seconds,.35) if early else seconds
    result=advice.advise(hist,g.to_move,depth=2 if early else 4,
                         seconds=max(.001,budget-(time.monotonic()-started)),
                         opp_name=params.get('opponent') or None)
    legal=g.moves(g.to_move)
    if any(mv not in legal for _,mv in result['scored']):
        raise ValueError('Engine returned a move outside the validated legal set')
    return dict(position=actual, history=g.history, to_move=actual['side'],
                legal=legal, top=result['scored'][:5], why=explain(g,result),
                winner=g.winner, request_id=params.get('request_id'),
                search={k:result[k] for k in ('depth','elapsed','timed_out','principal_variation')},
                opponent_evidence=[r for r in result.get('blend',[]) if r.get('evidence')][:4])
