"""Optional MIT-licensed MCTS engine, with independent Python tactical checks."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import coach as c

def search(history, side=None, time_limit=5, rollouts=60000, seed=None):
    if not isinstance(time_limit, (int,float)) or not math.isfinite(time_limit) or not 0 < time_limit <= 60:
        raise ValueError('MCTS seconds must be greater than 0 and at most 60')
    if not isinstance(rollouts, int) or not 2 <= rollouts <= 200000:
        raise ValueError('Rollouts must be an integer from 2 to 200000')
    started = time.monotonic()
    g = c.Game(history)
    side = g.to_move if side is None else side
    if side != g.to_move:
        raise ValueError('Wrong side to move')
    result = dict(engine='mcts', scored=[], depth=None, nodes=0, elapsed=0, timed_out=False,
                  winner=g.winner, principal_variation=[], candidates=[], simulations=0)
    if g.winner is not None:
        return result
    # Full-width one-reply safety filter; simulation percentages never override wins.
    safe, legal = [], g.moves(side)
    for move in legal:
        child = g.copy()
        child._play(move)
        if child.winner == side:
            result.update(scored=[(-c.WIN,move)], principal_variation=[move], tactical='immediate win', elapsed=time.monotonic()-started)
            return result
        if not any(p[1] == c.GOALS[1-side] for _, p in child.pawn_moves(1-side)):
            safe.append(move)
    node = shutil.which('node')
    if not node:
        raise ValueError('MCTS needs Node.js; select the Python engine or install Node.js')
    payload = dict(history=g.history, rollouts=rollouts, seconds=max(0.001,time_limit-(time.monotonic()-started)), seed=seed)
    flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess,'CREATE_NO_WINDOW') else 0
    try:
        completed = subprocess.run([node, str(Path(__file__).with_name('mcts_adapter.cjs'))],
            input=json.dumps(payload), text=True, capture_output=True,
            timeout=max(0.01,time_limit-(time.monotonic()-started)), creationflags=flags)
        output = completed.stdout
        if completed.returncode:
            raise ValueError(f'MCTS failed: {completed.stderr.strip()[:200]}')
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b''
        if isinstance(output, bytes):
            output = output.decode('utf-8', errors='replace')
        result['timed_out'] = True
    batches = []
    for line in output.splitlines():
        try:
            batches.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A killed process may leave one incomplete final line.
    data = batches[-1] if batches else dict(candidates=[],simulations=0)
    allowed = set(safe or legal)
    candidates = [row for row in data['candidates'] if row['move'] in allowed]
    if not candidates:
        fallback = min(allowed, key=lambda mv: _value(g,mv,side))
        result.update(scored=[(0,fallback)], principal_variation=[fallback], fallback=True)
    else:
        result.update(scored=[(-r['visits'],r['move']) for r in candidates],
                      principal_variation=[candidates[0]['move']], candidates=candidates, fallback=False)
    result.update(simulations=data['simulations'], nodes=data['simulations'], elapsed=time.monotonic()-started,
                  filtered_immediate_losses=len(legal)-len(safe))
    return result

def _value(g, move, side):
    child=g.copy()
    child._play(move)
    return child.eval_side(side), len(move), move
