"""Verified live-position contract shared by the overlay and regression tests."""
import hashlib
import json
import time
import coach as c
import advice
import learning
from pathlib import Path

BUILD = hashlib.sha256(b"".join(Path(__file__).with_name(n).read_bytes() for n in ("coach.py", "advice.py", "mcts_coach.py", "policy_value.py", "endgame.py", "live_coach.py"))).hexdigest()[:12]

def trace(params, payload):
    """Bounded local audit trail; never sends game history to an external service."""
    try:
        folder=Path(__file__).with_name("logs");folder.mkdir(exist_ok=True)
        path=folder/"live-advice.jsonl"
        if path.exists() and path.stat().st_size>5_000_000:
            path.replace(folder/"live-advice.previous.jsonl")
        row=dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),build=BUILD,
                 game=params.get("game_id", "")[:240],history=payload["history"],
                 position=payload["position"],top=payload["top"],search=payload["search"])
        with path.open("a",encoding="utf-8") as f:f.write(json.dumps(row)+"\n")
    except OSError:
        pass


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
    if result.get('forced_loss'):
        return text+' Every legal move allows the opponent to reach the goal next turn; no saving move remains.'
    if result.get('exact'):
        text+=f" Exact pawn-only result: {result['outcome']}"
        if result.get('goal_plies') is not None: text+=f" in {result['goal_plies']} plies with optimal play"
        return text+'. Includes jumps and whose turn it is; excludes clocks.'
    if result.get('engine')=='mcts':
        sims=result.get('simulations',0)
        if sims:
            text+=f' {sims} simulated games searched end-to-end.'
        else:
            text+=' Monte Carlo search; too few simulations to be confident.'
        text+=' Rollout outcomes are estimates, not a measured win rate against players.'
        if result.get('timed_out'):
            text+=' Time limit reached; using completed simulation batches.'
        ov=result.get('tactical_override')
        if ov:
            kind='selective' if ov.get('selective') else 'full-width'
            text+=f" Cross-check found a better move: MCTS favored {ov['mcts_top']}, but a completed {kind} depth-{ov['depth']} search prefers {ov['minimax_best']} by {ov['gap']:g} heuristic points. This is not a proven win."
        return text
    pv=result.get('principal_variation',[])
    if len(pv)>1:
        text+=f' Searched reply: {pv[1]}.'
    if result.get('depth',0)==0:
        text+=' Legal fallback; search did not finish a depth.'
    else:
        text+=f" Compared all legal moves through depth {result['depth']}; deeper threats may remain."
    return text


def validated_game(params):
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
    return g


def query(params, record_trace=True):
    started=time.monotonic()
    g=validated_game(params)
    hist=g.history
    actual=position(g)
    seconds=float(params.get('seconds','4'))
    if not 0<seconds<=15:
        raise ValueError('Time budget must be greater than zero and at most 15 seconds')
    # A short full-width search in the uncontested central opening, not a
    # hard-coded move. As soon as pawns approach or a wall appears, use full budget.
    engine=(params.get('engine','mcts') or 'mcts').lower()
    if engine not in ('mcts','python'):
        engine='mcts'
    # The opening is cheap to decide; give the search the full budget otherwise.
    # MCTS needs a little time to spin up its Node subprocess, so give the
    # uncontested opening a short but meaningful search rather than a lookup.
    early=not g.walls and all(p[0]==4 for p in g.pawns.values()) and abs(g.pawns[0][1]-g.pawns[1][1])>3
    if engine=='mcts':
        budget=min(seconds, 1.2) if early else seconds
    else:
        budget=min(seconds,.35) if early else seconds
    result=None
    if not any(g.remaining.values()) and g.winner is None:
        import endgame
        result=endgame.solve(g,seconds=min(.75,budget/3))
    if result is None:
        result=advice.advise(hist,g.to_move,depth=2 if early else 4,
                             seconds=max(.001,budget-(time.monotonic()-started)),
                             opp_name=params.get('opponent') or None,
                             engine=engine)
    result['elapsed']=time.monotonic()-started
    legal=g.moves(g.to_move)
    if any(mv not in legal for _,mv in result['scored']):
        raise ValueError('Engine returned a move outside the validated legal set')
    payload=dict(position=actual, history=g.history, to_move=actual['side'],
                legal=legal, top=result['scored'][:5], why=explain(g,result),
                winner=g.winner, request_id=params.get('request_id'),
                search={k:result[k] for k in ('engine','depth','elapsed','timed_out','principal_variation','forced_loss','exact','outcome','goal_plies') if k in result},
                opponent_evidence=[r for r in result.get('blend',[]) if r.get('evidence')][:4],build=BUILD)
    payload['search'].update({k:result[k] for k in ('tactical_override','crosscheck_depth','score_units','policy_guided','policy_value_guided','policy_model','policy_model_id') if k in result})
    if record_trace:
        trace(params,payload)
    return payload
