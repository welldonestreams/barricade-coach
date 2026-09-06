"""Learning memory for the Barricade coach.

Stores every completed game (position -> move -> outcome per ply) and builds
per-opponent tendency models from public game histories.  All data stays
local under memory/ (gitignored); nothing is sent anywhere.

Storage:
  memory/learning.db            SQLite (outcomes + eval-deltas + game dedup);
                                WAL mode, safe for concurrent writers
  memory/opponents/<name>.json  learned tendency model for one opponent
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import coach

ROOT = Path(__file__).resolve().parent
MEMORY = ROOT / 'memory'
GAMES = MEMORY / 'games'
OPPONENTS = MEMORY / 'opponents'
OUTCOMES = MEMORY / 'outcomes.json'   # legacy JSON (migrated on first use)
EVALS = MEMORY / 'eval-deltas.json'   # legacy JSON (migrated on first use)
DB = MEMORY / 'learning.db'

NAME_OK = re.compile(r'^[A-Za-z0-9_-]{1,32}$')


def _ensure():
    MEMORY.mkdir(parents=True, exist_ok=True)
    OPPONENTS.mkdir(parents=True, exist_ok=True)


def _db():
    _ensure()
    con = sqlite3.connect(str(DB), timeout=15.0)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=NORMAL')
    con.execute('PRAGMA busy_timeout=15000')
    return con


def _init(con):
    con.execute('''CREATE TABLE IF NOT EXISTS outcomes(
        position TEXT NOT NULL, move TEXT NOT NULL,
        won INTEGER NOT NULL DEFAULT 0, lost INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(position, move))''')
    con.execute('''CREATE TABLE IF NOT EXISTS evals(
        position TEXT NOT NULL, move TEXT NOT NULL,
        games INTEGER NOT NULL DEFAULT 0, sum_delta REAL NOT NULL DEFAULT 0,
        won INTEGER NOT NULL DEFAULT 0, lost INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(position, move))''')
    con.execute('''CREATE TABLE IF NOT EXISTS games(
        key TEXT PRIMARY KEY, played_at TEXT)''')


def _stable_key(history_csv):
    return hashlib.md5(history_csv.encode('utf-8', 'replace')).hexdigest()[:16]


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# --------------------------------------------------------------------------
# Outcome memory: position -> move -> won/lost, aggregated over all recorded games
# --------------------------------------------------------------------------

def record_game(history_csv, winner, red_name=None, blue_name=None,
                sharecode=None, opp_rating=None, source='live', seed_len=0):
    """Record a finished game into outcome memory.

    history_csv: comma ply list. winner: 'red'/'blue'/None(draw).
    seed_len: if this game was self-play continued from an imported position,
    only plies >= seed_len are recorded (the seed prefix is context, not a
    self-play outcome we should learn from).
    Returns True if newly recorded, False if a duplicate."""
    _ensure()
    try:
        game = coach.Game(coach.parse_history(history_csv))
    except ValueError as exc:
        raise ValueError(f'Game does not replay: {exc}')
    if winner not in (None, 'red', 'blue'):
        raise ValueError('winner must be red, blue or null')
    winner_side = {'red': coach.RED, 'blue': coach.BLUE}.get(winner)

    key = sharecode or _stable_key(history_csv)
    con = _db()
    _init(con)
    try:
        if con.execute('SELECT 1 FROM games WHERE key=?', (key,)).fetchone():
            return False
        con.execute('INSERT INTO games(key, played_at) VALUES(?,?)', (key, _now()))
        g = coach.Game()
        for i, mv in enumerate(game.history):
            mover = g.to_move
            if i < seed_len:
                try:
                    g.apply(mv)
                except ValueError:
                    break
                continue
            pos = ','.join(g.history)
            if mover == winner_side:
                con.execute('''INSERT INTO outcomes(position,move,won,lost) VALUES(?,?,1,0)
                               ON CONFLICT(position,move) DO UPDATE SET won=won+1''', (pos, mv))
            else:
                con.execute('''INSERT INTO outcomes(position,move,won,lost) VALUES(?,?,0,1)
                               ON CONFLICT(position,move) DO UPDATE SET lost=lost+1''', (pos, mv))
            try:
                g.apply(mv)
            except ValueError:
                break
        con.commit()
        return True
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def outcome_stats(history):
    """Return {move: {won, lost, total, winrate}} for legal moves at this
    position, from the recorded games (empty dict if unseen)."""
    pos = ','.join(history)
    con = _db()
    _init(con)
    try:
        rows = con.execute('SELECT move, won, lost FROM outcomes WHERE position=?', (pos,)).fetchall()
    finally:
        con.close()
    out = {}
    for mv, won, lost in rows:
        total = won + lost
        if total:
            out[mv] = {'won': won, 'lost': lost, 'total': total, 'winrate': won / total}
    return out


# --------------------------------------------------------------------------
# Eval-delta ("why") memory: position -> move -> avg eval swing + outcome
# --------------------------------------------------------------------------

def record_evals(history_csv, winner, seed_len=0):
    """Replay a finished game and record the causal 'why' per move.

    For each ply >= seed_len, store how much the move swung the mover's
    evaluation (delta = eval_before - eval_after; eval is lower-is-better, so
    positive delta means the move helped the mover, negative means it hurt),
    tagged by whether that mover went on to win."""
    game = coach.Game(coach.parse_history(history_csv))
    winner_side = {'red': coach.RED, 'blue': coach.BLUE}.get(winner)
    if winner not in (None, 'red', 'blue'):
        raise ValueError('winner must be red, blue or null')
    con = _db()
    _init(con)
    try:
        g = coach.Game()
        for i, mv in enumerate(game.history):
            if i < seed_len:
                try:
                    g.apply(mv)
                except ValueError:
                    break
                continue
            pos = ','.join(g.history)
            mover = g.to_move
            before = g.eval_side(mover)
            g.apply(mv)
            after = g.eval_side(mover)
            delta = before - after
            won = 1 if mover == winner_side else 0
            lost = 0 if mover == winner_side else 1
            con.execute('''INSERT INTO evals(position,move,games,sum_delta,won,lost)
                           VALUES(?,?,1,?,?,?)
                           ON CONFLICT(position,move) DO UPDATE SET
                             games=games+1, sum_delta=sum_delta+excluded.sum_delta,
                             won=won+excluded.won, lost=lost+excluded.lost''',
                        (pos, mv, delta, won, lost))
        con.commit()
        return True
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def eval_delta_stats(history):
    """For the current position, return {move: {games, avg_delta, won, lost}}."""
    pos = ','.join(history)
    con = _db()
    _init(con)
    try:
        rows = con.execute('SELECT move, games, sum_delta, won, lost FROM evals WHERE position=?',
                           (pos,)).fetchall()
    finally:
        con.close()
    out = {}
    for mv, games, sum_delta, won, lost in rows:
        if games:
            out[mv] = {'games': games, 'avg_delta': sum_delta / games, 'won': won, 'lost': lost}
    return out


def memory_stats():
    """Lightweight counts for /api/memory display."""
    con = _db()
    _init(con)
    try:
        games = con.execute('SELECT COUNT(*) FROM games').fetchone()[0]
        positions = con.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0]
    finally:
        con.close()
    return {'positions': positions, 'games': games}


# --------------------------------------------------------------------------
# Legacy JSON migration (one-time; idempotent)
# --------------------------------------------------------------------------

def migrate_legacy_json():
    """Pull any legacy outcomes.json / eval-deltas.json into SQLite. No-op if
    the JSON files are absent or already empty."""
    _ensure()
    if OUTCOMES.exists():
        try:
            data = json.loads(OUTCOMES.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            data = None
        if data and data.get('positions'):
            con = _db()
            _init(con)
            try:
                for pos, moves in data['positions'].items():
                    for mv, cell in moves.items():
                        con.execute('''INSERT INTO outcomes(position,move,won,lost) VALUES(?,?,?,?)
                                       ON CONFLICT(position,move) DO UPDATE SET
                                         won=won+excluded.won, lost=lost+excluded.lost''',
                                    (pos, mv, cell.get('won', 0), cell.get('lost', 0)))
                con.commit()
            finally:
                con.close()
        OUTCOMES.rename(OUTCOMES.with_suffix('.json.migrated'))
    if EVALS.exists():
        try:
            data = json.loads(EVALS.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            data = None
        if data and data.get('positions'):
            con = _db()
            _init(con)
            try:
                for pos, moves in data['positions'].items():
                    for mv, cell in moves.items():
                        con.execute('''INSERT INTO evals(position,move,games,sum_delta,won,lost)
                                       VALUES(?,?,?,?,?,?)
                                       ON CONFLICT(position,move) DO UPDATE SET
                                         games=games+excluded.games,
                                         sum_delta=sum_delta+excluded.sum_delta,
                                         won=won+excluded.won, lost=lost+excluded.lost''',
                                    (pos, mv, cell.get('games', 0), cell.get('sum_delta', 0.0),
                                     cell.get('won', 0), cell.get('lost', 0)))
                con.commit()
            finally:
                con.close()
        EVALS.rename(EVALS.with_suffix('.json.migrated'))
    # Rebuild the games table key from any leftover memory/games/*.json so
    # dedup counts stay consistent.
    if GAMES.exists():
        con = _db()
        _init(con)
        try:
            for path in sorted(GAMES.glob('*.json')):
                key = path.stem
                con.execute('INSERT OR IGNORE INTO games(key, played_at) VALUES(?,?)',
                            (key, _now()))
            con.commit()
        finally:
            con.close()


# --------------------------------------------------------------------------
# Opponent tendency models
# --------------------------------------------------------------------------

def _game_side(game, name):
    if game.get('player1Username') == name:
        return 0, 1
    if game.get('player2Username') == name:
        return 1, 0
    return None


def _winner_side(game):
    w = game.get('winner')
    if w in (0, 1) and not isinstance(w, bool):
        return int(w)
    try:
        return int(w) - 1
    except (TypeError, ValueError):
        return None


def build_opponent_model(name, game_records):
    if not NAME_OK.match(name):
        raise ValueError('Invalid opponent name')
    model = {
        'name': name,
        'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'games': 0, 'wins': 0,
        'rating_high': None, 'rating_low': None,
        'openings': {'red': defaultdict(lambda: defaultdict(int)),
                     'blue': defaultdict(lambda: defaultdict(int))},
        'responses': defaultdict(lambda: defaultdict(int)),
        'loss_positions': defaultdict(int),
        'avg_move_ms': None,
        'move_times': [],
    }
    for game in game_records:
        sides = _game_side(game, name)
        if sides is None:
            continue
        my_side, opp_side = sides
        winner_side = _winner_side(game)
        hist = coach.parse_history(game.get('historyCsv', ''))
        try:
            g = coach.Game()
            for i, mv in enumerate(hist):
                if g.to_move == my_side:
                    pos = ','.join(g.history)
                    if len(g.history) < 6:
                        color = 'red' if my_side == 0 else 'blue'
                        model['openings'][color][mv][len(g.history)] += 1
                    else:
                        model['responses'][pos][mv] += 1
                    if winner_side is not None and winner_side != my_side:
                        model['loss_positions'][pos] += 1
                try:
                    g.apply(mv)
                except ValueError:
                    break
        except ValueError:
            continue
        model['games'] += 1
        if winner_side is not None and winner_side == my_side:
            model['wins'] += 1
        mt = game.get('moveTimes') or []
        if mt:
            model['move_times'].extend(mt)
        rb = game.get('p1Rating' if my_side == 0 else 'p2Rating')
        if rb:
            model['rating_high'] = max(model['rating_high'] or 0, rb)
            model['rating_low'] = min(model['rating_low'] or rb, rb)
    if model['move_times']:
        model['avg_move_ms'] = sum(model['move_times']) / len(model['move_times'])
    model['openings'] = {k: {m: dict(d) for m, d in v.items()}
                         for k, v in model['openings'].items()}
    model['responses'] = {p: dict(d) for p, d in model['responses'].items()}
    model['loss_positions'] = dict(model['loss_positions'])
    del model['move_times']
    return model


def save_opponent(name, model):
    _ensure()
    (OPPONENTS / f'{name}.json').write_text(json.dumps(model), encoding='utf-8')


def load_opponent(name):
    path = OPPONENTS / f'{name}.json'
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            return None
    return None


def opponent_insight(name, history):
    model = load_opponent(name)
    if not model:
        return None
    pos = ','.join(history)
    resp = model['responses'].get(pos)
    if not resp:
        if len(history) < 6:
            return {'known': False, 'games': model['games'],
                    'wins': model['wins'], 'note': 'early game'}
        return {'known': False, 'games': model['games'], 'wins': model['wins']}
    total = sum(resp.values())
    top = sorted(resp.items(), key=lambda kv: -kv[1])[:3]
    lost_here = model['loss_positions'].get(pos, 0)
    return {'known': True, 'total': total, 'top': top, 'lost_here': lost_here,
            'games': model['games'], 'wins': model['wins']}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Build/refresh an opponent model from study/archive games')
    p.add_argument('name', help='barricade.gg username')
    p.add_argument('--src', default=str(ROOT / 'study' / 'archive'),
                   help='folder of raw game json (default study/archive)')
    a = p.parse_args()
    src = Path(a.src)
    records = []
    for path in sorted(src.glob('*.json')):
        try:
            records.append(json.loads(path.read_text(encoding='utf-8')))
        except (ValueError, OSError):
            continue
    model = build_opponent_model(a.name, records)
    save_opponent(a.name, model)
    print(f'{a.name}: {model["games"]} games, {model["wins"]} wins, '
          f'{len(model["responses"])} response positions, '
          f'rating {model["rating_low"]:.0f}-{model["rating_high"]:.0f}' if model['rating_high'] else
          f'{a.name}: {model["games"]} games, {model["wins"]} wins')
