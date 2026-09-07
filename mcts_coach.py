"""Optional MIT-licensed MCTS engine, with independent Python tactical checks."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import os
import coach as c

def search(history, side=None, time_limit=5, rollouts=60000, seed=None, workers=None):
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
    allowed = set(safe or legal)
    if time_limit - (time.monotonic() - started) < 0.05:
        # Budget too small to even spawn Node meaningfully; return the safe
        # tactical fallback (a legal move) rather than racing a doomed search.
        fallback = min(allowed, key=lambda mv: _value(g, mv, side))
        result.update(scored=[(0, fallback)], principal_variation=[fallback], fallback=True,
                      elapsed=time.monotonic()-started)
        return result
    node = shutil.which('node')
    if not node:
        raise ValueError('MCTS needs Node.js; select the Python engine or install Node.js')
    # Root parallelization: run N independent MCTS trees (distinct seeds) across
    # the machine's cores and merge their visit counts. The vendored engine is
    # single-threaded, so one process only used ~1 core (~10k sims/s); N processes
    # give ~Nx simulations in the same wall-clock, and the engine's own data shows
    # strength scales with total rollouts. workers is auto-sized but capped to
    # leave headroom for the server + self-play; pass an int to override.
    if workers is None:
        try:
            ncpu = os.cpu_count() or 2
        except Exception:
            ncpu = 2
        workers = max(1, min(4, ncpu - 2))
    workers = max(1, int(workers))
    seconds = max(0.001, time_limit - (time.monotonic() - started))
    flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
    adapter = str(Path(__file__).with_name('mcts_adapter.cjs'))

    def run_one(w):
        # Each worker gets the FULL rollouts cap but its own seed, so it runs flat
        # out for the whole time budget (the cap is only hit at very short budgets).
        seed_w = None if seed is None else (seed + w * 7919) & 0xFFFFFFFF
        payload = dict(history=g.history, rollouts=rollouts, seconds=seconds, seed=seed_w)
        try:
            completed = subprocess.run([node, adapter], input=json.dumps(payload),
                text=True, capture_output=True, timeout=seconds + 10.0, creationflags=flags)
            out = completed.stdout
            if completed.returncode:
                return dict(candidates=[], simulations=0, timed_out=True)
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or b''
            if isinstance(out, bytes):
                out = out.decode('utf-8', errors='replace')
        batches = []
        for line in out.splitlines():
            try:
                batches.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A killed process may leave one incomplete final line.
        data = batches[-1] if batches else dict(candidates=[], simulations=0)
        data['timed_out'] = False
        return data

    datas = [run_one(w) for w in range(workers)] if workers == 1 else list(
        _parallel_map(run_one, range(workers)))
    # Merge root visit counts across workers (sum visits per move; weight the win
    # rate by visits so it stays meaningful in the aggregate).
    merged = {}
    total_sims = 0
    any_timed_out = False
    for data in datas:
        total_sims += data.get('simulations', 0)
        any_timed_out = any_timed_out or data.get('timed_out', False)
        for row in data.get('candidates', []):
            mv = row['move']
            if mv not in merged:
                merged[mv] = [0, 0.0]
            merged[mv][0] += row.get('visits', 0)
            merged[mv][1] += row.get('rollout_win_rate', 0.0) * row.get('visits', 0)
    candidates = []
    for mv, (visits, wr_sum) in merged.items():
        if mv in allowed and visits > 0:
            candidates.append(dict(move=mv, visits=visits, rollout_win_rate=wr_sum / visits))
    candidates.sort(key=lambda r: (-r['visits'], r['move']))
    if not candidates:
        fallback = min(allowed, key=lambda mv: _value(g, mv, side))
        result.update(scored=[(0, fallback)], principal_variation=[fallback], fallback=True)
    else:
        result.update(scored=[(-r['visits'], r['move']) for r in candidates],
                      principal_variation=[candidates[0]['move']], candidates=candidates, fallback=False)
    result.update(simulations=total_sims, nodes=total_sims, elapsed=time.monotonic()-started,
                  timed_out=any_timed_out, filtered_immediate_losses=len(legal)-len(safe),
                  workers=workers)
    return result


def _parallel_map(fn, items):
    """Run fn across items concurrently (threads; each blocks on a subprocess)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        return list(ex.map(fn, items))


def _value(g, move, side):
    child=g.copy()
    child._play(move)
    return child.eval_side(side), len(move), move
