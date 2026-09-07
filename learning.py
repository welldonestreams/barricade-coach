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
    con.execute('''CREATE TABLE IF NOT EXISTS prior(
        position TEXT NOT NULL, move TEXT NOT NULL,
        won INTEGER NOT NULL DEFAULT 0, lost INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(position, move))''')
    con.execute('''CREATE TABLE IF NOT EXISTS prior_seen(
        key TEXT PRIMARY KEY)''')
    con.execute('''CREATE TABLE IF NOT EXISTS reasons(
        position TEXT NOT NULL, move TEXT NOT NULL,
        games INTEGER NOT NULL DEFAULT 0, sum_delta REAL NOT NULL DEFAULT 0,
        won INTEGER NOT NULL DEFAULT 0, lost INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(position, move))''')
    con.execute('''CREATE TABLE IF NOT EXISTS reasons_seen(
        key TEXT PRIMARY KEY)''')


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
            state = position_key(g)
            if winner_side is None:
                g.apply(mv)
                continue
            won = 1 if mover == winner_side else 0
            lost = 0 if mover == winner_side else 1
            con.execute('''INSERT INTO outcomes(position,move,won,lost) VALUES(?,?,?,?)
                           ON CONFLICT(position,move) DO UPDATE SET won=won+excluded.won, lost=lost+excluded.lost''', (pos, mv, won, lost))
            con.execute('''INSERT INTO prior(position,move,won,lost) VALUES(?,?,?,?)
                           ON CONFLICT(position,move) DO UPDATE SET won=won+excluded.won, lost=lost+excluded.lost''', (state, mv, won, lost))
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

def learned_prior(history):
    """Board-state-keyed outcome prior for the CURRENT position.

    Returns {move: {won, lost, total, winrate}} for every move ever played from
    this exact board state (pawns + walls + reserves + side to move), across all
    recorded games regardless of move order. This is the generalization the
    legacy move-sequence key lacked: a position reached by any path aggregates
    to the same key, so real win/loss signal accumulates instead of fragmenting
    into one-sample rows."""
    g = coach.Game(history)
    state = position_key(g)
    con = _db()
    _init(con)
    try:
        rows = con.execute('SELECT move, won, lost FROM prior WHERE position=?', (state,)).fetchall()
    finally:
        con.close()
    out = {}
    for mv, won, lost in rows:
        total = won + lost
        if total:
            out[mv] = {'won': won, 'lost': lost, 'total': total, 'winrate': won / total}
    return out


def prior_stats():
    """How many board-state positions and moves the prior has learned."""
    con = _db()
    _init(con)
    try:
        positions = con.execute('SELECT COUNT(*) FROM prior').fetchone()[0]
        moves = con.execute('SELECT COUNT(*) FROM (SELECT DISTINCT move FROM prior)').fetchone()[0]
    finally:
        con.close()
    return {'positions': positions, 'moves': moves}


def learned_reasons(history):
    """Board-state-keyed eval-swing signal for the CURRENT position.

    For each move ever played from this exact board state, returns {move:
    {games, avg_delta, won, lost, winrate}} where avg_delta is the average
    heuristic swing the move produced for its mover (eval_before - eval_after;
    positive = the move improved the mover's position, negative = it hurt).
    This is the 'why' a move is good or bad, independent of who won the game:
    a strong move in a lost game still shows a positive avg_delta."""
    g = coach.Game(history)
    state = position_key(g)
    con = _db()
    _init(con)
    try:
        rows = con.execute('SELECT move, games, sum_delta, won, lost FROM reasons WHERE position=?', (state,)).fetchall()
    finally:
        con.close()
    out = {}
    for mv, games, sum_delta, won, lost in rows:
        if games:
            out[mv] = {'games': games, 'avg_delta': sum_delta / games,
                       'won': won, 'lost': lost, 'winrate': won / games}
    return out


def build_reasons_from_archive(src=None, max_games=None):
    """Replay archive games into the board-state 'reasons' (why) table.

    For every ply, records the mover's eval swing (before - after) against the
    board state, tagged by the eventual result. Idempotent via reasons_seen.
    Returns the number of games newly incorporated."""
    if src is None:
        src = ROOT / 'study' / 'archive'
    src = Path(src)
    con = _db()
    _init(con)
    imported = 0
    try:
        for path in sorted(src.glob('*.json')):
            try:
                rec = json.loads(path.read_text(encoding='utf-8-sig'))
            except (ValueError, OSError):
                continue
            try:
                game = coach.Game(coach.parse_history(rec.get('historyCsv', '')))
            except ValueError:
                continue
            hist = game.history
            if not hist:
                continue
            winner = rec.get('winner')
            winner_side = {'1': coach.RED, '2': coach.BLUE}.get(str(winner))
            key = ','.join(hist)
            if con.execute('SELECT 1 FROM reasons_seen WHERE key=?', (key,)).fetchone():
                continue
            con.execute('INSERT INTO reasons_seen(key) VALUES(?)', (key,))
            g = coach.Game()
            for mv in hist:
                mover = g.to_move
                state = position_key(g)
                if winner_side is None:
                    g.apply(mv)
                    continue
                before = g.eval_side(mover)
                g.apply(mv)
                after = g.eval_side(mover)
                delta = before - after
                won = 1 if mover == winner_side else 0
                lost = 0 if mover == winner_side else 1
                con.execute("""INSERT INTO reasons(position,move,games,sum_delta,won,lost)
                               VALUES(?,?,1,?,?,?)
                               ON CONFLICT(position,move) DO UPDATE SET
                                 games=games+1, sum_delta=sum_delta+excluded.sum_delta,
                                 won=won+excluded.won, lost=lost+excluded.lost""",
                            (state, mv, delta, won, lost))
            imported += 1
            if max_games and imported >= max_games:
                break
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return imported


def explain_why(history, move):
    """Concrete 'why' for one move: what the archive says it did, and how often.

    Returns a short human string, e.g. 'historically +12 position swing, won
    61% (N=18)' -- or None if the position/move is unseen."""
    reasons = learned_reasons(history)
    cell = reasons.get(move)
    if not cell:
        return None
    n = cell['games']
    return f"historically {cell['avg_delta']:+.0f} position swing, won {cell['winrate']*100:.0f}% (N={n})"


def build_prior_from_archive(src=None, max_games=None):
    """Replay archive games into the board-state prior.

    Reads every game in study/archive (top-player public games), replays it, and
    records each move's win/loss against its board state. Idempotent: games are
    deduped via prior_seen by their full move list, so re-runs only add new games.
    Returns the number of games newly incorporated."""
    if src is None:
        src = ROOT / 'study' / 'archive'
    src = Path(src)
    con = _db()
    _init(con)
    imported = 0
    try:
        for path in sorted(src.glob('*.json')):
            try:
                rec = json.loads(path.read_text(encoding='utf-8-sig'))
            except (ValueError, OSError):
                continue
            try:
                game = coach.Game(coach.parse_history(rec.get('historyCsv', '')))
            except ValueError:
                continue
            hist = game.history
            if not hist:
                continue
            winner = rec.get('winner')
            winner_side = {'1': coach.RED, '2': coach.BLUE}.get(str(winner))
            key = ','.join(hist)
            if con.execute('SELECT 1 FROM prior_seen WHERE key=?', (key,)).fetchone():
                continue
            con.execute('INSERT INTO prior_seen(key) VALUES(?)', (key,))
            g = coach.Game()
            for mv in hist:
                mover = g.to_move
                state = position_key(g)
                if winner_side is None:
                    g.apply(mv)
                    continue
                won = 1 if mover == winner_side else 0
                lost = 0 if mover == winner_side else 1
                con.execute("""INSERT INTO prior(position,move,won,lost) VALUES(?,?,?,?)
                               ON CONFLICT(position,move) DO UPDATE SET won=won+excluded.won, lost=lost+excluded.lost""", (state, mv, won, lost))
                try:
                    g.apply(mv)
                except ValueError:
                    break
            imported += 1
            if max_games and imported >= max_games:
                break
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return imported


def record_evals(history_csv, winner, seed_len=0):
    """Replay a finished game and record the heuristic change per move.

    For each ply >= seed_len, store how much the move swung the mover's
    evaluation (delta = eval_before - eval_after; eval is lower-is-better, so
    positive delta means the move helped the mover, negative means it hurt),
    tagged by the result; this is not a causal explanation."""
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
            state = position_key(g)
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
            con.execute('''INSERT INTO reasons(position,move,games,sum_delta,won,lost)
                           VALUES(?,?,1,?,?,?)
                           ON CONFLICT(position,move) DO UPDATE SET
                             games=games+1, sum_delta=sum_delta+excluded.sum_delta,
                             won=won+excluded.won, lost=lost+excluded.lost''',
                        (state, mv, delta, won, lost))
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

def position_key(g):
    return json.dumps([g.pawns[0], g.pawns[1], sorted(g.walls),
                       g.remaining[0], g.remaining[1], g.to_move], separators=(',', ':'))


def _game_side(game, name):
    if game.get('player1Username') == name:
        return 0, 1
    if game.get('player2Username') == name:
        return 1, 0
    return None


def _winner_side(game):
    # Public Barricade records use player numbers 1 and 2, including strings.
    value = game.get('winner')
    if isinstance(value, bool): return None
    return {'1': coach.RED, '2': coach.BLUE}.get(str(value))


def build_opponent_model(name, game_records):
    if not NAME_OK.match(name):
        raise ValueError('Invalid opponent name')
    model = {
        'name': name,
        'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'format': 3,
        'games': 0, 'wins': 0,
        'rating_high': None, 'rating_low': None,
        'openings': {'red': defaultdict(lambda: defaultdict(int)),
                     'blue': defaultdict(lambda: defaultdict(int))},
        # Color-split: the opponent's tendencies differ by which color they
        # played, so responses/loss_positions are keyed by the opponent's color.
        'responses': {'red': defaultdict(lambda: defaultdict(int)),
                      'blue': defaultdict(lambda: defaultdict(int))},
        'loss_positions': {'red': defaultdict(int), 'blue': defaultdict(int)},
        'color_games': {'red': 0, 'blue': 0},
        'avg_move_ms': None,
        'move_times': [],
    }
    seen_histories = set()
    for game in game_records:
        sides = _game_side(game, name)
        if sides is None:
            continue
        my_side, opp_side = sides
        color = 'red' if my_side == 0 else 'blue'
        winner_side = _winner_side(game)
        try:
            hist = coach.Game(coach.parse_history(game.get('historyCsv', ''))).history
            key = ','.join(hist)
            if key in seen_histories: continue
            seen_histories.add(key)
            g = coach.Game()
            for i, mv in enumerate(hist):
                if g.to_move == my_side:
                    pos = position_key(g)
                    if len(g.history) < 6:
                        model['openings'][color][mv][len(g.history)] += 1
                    model['responses'][color][pos][mv] += 1
                    if winner_side is not None and winner_side != my_side:
                        model['loss_positions'][color][pos] += 1
                try:
                    g.apply(mv)
                except ValueError:
                    break
        except ValueError:
            continue
        model['games'] += 1
        model['color_games'][color] += 1
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
    model['responses'] = {k: {p: dict(d) for p, d in v.items()}
                          for k, v in model['responses'].items()}
    model['loss_positions'] = {k: dict(v) for k, v in model['loss_positions'].items()}
    del model['move_times']
    return model


def save_opponent(name, model):
    _ensure()
    (OPPONENTS / f'{name}.json').write_text(json.dumps(model), encoding='utf-8')


def load_opponent(name):
    if not NAME_OK.fullmatch(name):
        return None
    path = OPPONENTS / f'{name}.json'
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            return None
    return None


# Weight given to the opponent's same-color games vs opposite-color games when
# blending tendencies. Same-color play is more predictive of what they'll do
# *as this color*, but the other color is not ruled out (players have transferable
# habits). Weights are deliberately soft; see advice.py for the overall blend.
OPP_SAME_COLOR_WEIGHT = 1.0
OPP_OTHER_COLOR_WEIGHT = 0.0


def opponent_insight(name, history, opp_color=None):
    """Tendencies of `name` at the current position, optionally color-weighted.

    opp_color: 'red' or 'blue' = the color the opponent is playing this game.
    When given, same-color games count 0.75 and opposite-color games 0.25
    (soft blend, never zeroing out the other color)."""
    model = load_opponent(name)
    if not model:
        return None
    pos = ','.join(history)

    # Format 1 (legacy): flat dicts {pos: {move: count}}.
    if model.get('format') == 3:
        pos = position_key(coach.Game(history))
    if model.get('format') not in (2, 3):
        resp = (model.get('responses') or {}).get(pos)
        if not resp:
            return {'known': False, 'games': model.get('games', 0),
                    'wins': model.get('wins', 0)}
        total = sum(resp.values())
        top = sorted(resp.items(), key=lambda kv: -kv[1])[:3]
        lost_here = (model.get('loss_positions') or {}).get(pos, 0)
        return {'known': True, 'total': total, 'top': top, 'lost_here': lost_here,
                'games': model.get('games', 0), 'wins': model.get('wins', 0)}

    resp = model.get('responses') or {}
    losses = model.get('loss_positions') or {}
    if opp_color in ('red', 'blue'):
        same = (resp.get(opp_color) or {}).get(pos, {})
        other_c = 'blue' if opp_color == 'red' else 'red'
        other = (resp.get(other_c) or {}).get(pos, {})
        blended = defaultdict(float)
        for mv, n in same.items():
            blended[mv] += n * OPP_SAME_COLOR_WEIGHT
        for mv, n in other.items():
            blended[mv] += n * OPP_OTHER_COLOR_WEIGHT
        lost_here = (losses.get(opp_color) or {}).get(pos, 0)
    else:
        blended = defaultdict(float)
        for color in ('red', 'blue'):
            for mv, n in (resp.get(color) or {}).get(pos, {}).items():
                blended[mv] += n
        lost_here = (losses.get('red') or {}).get(pos, 0) + (losses.get('blue') or {}).get(pos, 0)
    blended = {mv: c for mv, c in blended.items() if c > 0}

    if not blended:
        if len(history) < 6:
            return {'known': False, 'games': model['games'],
                    'wins': model['wins'], 'note': 'early game',
                    'color_games': model.get('color_games')}
        return {'known': False, 'games': model['games'], 'wins': model['wins'],
                'color_games': model.get('color_games')}
    total = sum(blended.values())
    top = sorted(blended.items(), key=lambda kv: -kv[1])[:3]
    return {'known': True, 'total': total, 'top': top, 'lost_here': lost_here,
            'games': model['games'], 'wins': model['wins'],
            'color_games': model.get('color_games')}


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
