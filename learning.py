"""Learning memory for the Barricade coach.

Stores every completed game (position -> move -> outcome per ply) and builds
per-opponent tendency models from public game histories.  All data stays
local under memory/ (gitignored); nothing is sent anywhere.

Layout:
  memory/games/<sharecode>.json   raw game record (history, result, players)
  memory/outcomes.json            position -> move -> {won, lost, games}
  memory/opponents/<name>.json    learned tendency model for one opponent
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import coach

ROOT = Path(__file__).resolve().parent
MEMORY = ROOT / 'memory'
GAMES = MEMORY / 'games'
OPPONENTS = MEMORY / 'opponents'
OUTCOMES = MEMORY / 'outcomes.json'

NAME_OK = re.compile(r'^[A-Za-z0-9_-]{1,32}$')


def _ensure():
    GAMES.mkdir(parents=True, exist_ok=True)
    OPPONENTS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Outcome memory: position -> move -> won/lost, aggregated over all recorded games
# --------------------------------------------------------------------------

def _empty_outcomes():
    return {'version': 1, 'positions': {}, 'games': 0, 'updated': None}


def load_outcomes():
    _ensure()
    if OUTCOMES.exists():
        try:
            return json.loads(OUTCOMES.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            pass
    return _empty_outcomes()


def save_outcomes(data):
    _ensure()
    data['updated'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    tmp = OUTCOMES.with_suffix('.tmp')
    tmp.write_text(json.dumps(data), encoding='utf-8')
    tmp.replace(OUTCOMES)


def record_game(history_csv, winner, red_name=None, blue_name=None,
                sharecode=None, opp_rating=None, source='live'):
    """history_csv: comma ply list. winner: 'red'/'blue'/None(draw).
    winner is from the *board*; record_game converts to 'side to move won'.
    """
    _ensure()
    try:
        game = coach.Game(coach.parse_history(history_csv))
    except ValueError as exc:
        raise ValueError(f'Game does not replay: {exc}')
    winner_side = {'red': coach.RED, 'blue': coach.BLUE}.get(winner)
    if winner not in (None, 'red', 'blue'):
        raise ValueError('winner must be red, blue or null')
    # avoid double-recording identical games
    key = sharecode or hash(history_csv) & 0xFFFFFFFF
    path = GAMES / f'{key}.json'
    if path.exists():
        return False
    record = {
        'history': game.history,
        'winner': winner,
        'red': red_name, 'blue': blue_name,
        'opp_rating': opp_rating,
        'source': source,
        'played_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'plies': len(game.history),
    }
    path.write_text(json.dumps(record), encoding='utf-8')

    data = load_outcomes()
    positions = data['positions']
    # for each ply, position before the move; the mover won iff mover==winner
    g = coach.Game()
    for i, mv in enumerate(game.history):
        mover = g.to_move
        pos = ','.join(g.history)
        outcome = 'won' if mover == winner_side else 'lost'
        cell = positions.setdefault(pos, {}).setdefault(mv, {'won': 0, 'lost': 0})
        cell[outcome] += 1
        try:
            g.apply(mv)
        except ValueError:
            break
    data['games'] += 1
    save_outcomes(data)
    return True


def outcome_stats(history):
    """Return {move: {won, lost, total, winrate}} for legal moves at this
    position, from the recorded games (empty dict if unseen)."""
    pos = ','.join(history)
    data = load_outcomes()
    raw = data['positions'].get(pos, {})
    out = {}
    for mv, cell in raw.items():
        won, lost = cell.get('won', 0), cell.get('lost', 0)
        total = won + lost
        if total:
            out[mv] = {'won': won, 'lost': lost, 'total': total,
                       'winrate': won / total}
    return out


# --------------------------------------------------------------------------
# Opponent tendency models
# --------------------------------------------------------------------------

def _game_side(game, name):
    """Return (side_index, opponent_index) for `name` in a raw archive game,
    or None if name didn't play. player1 = red (coach.RED=0), player2 = blue."""
    if game.get('player1Username') == name:
        return 0, 1
    if game.get('player2Username') == name:
        return 1, 0
    return None


def _winner_side(game):
    """winner field is '1' or '2' (player number, 1-indexed); map to side index."""
    w = game.get('winner')
    if w in (0, 1) and not isinstance(w, bool):  # tolerate int-0/1 too
        return int(w)
    try:
        return int(w) - 1
    except (TypeError, ValueError):
        return None


def _plies_for_side(history, side):
    return history[side::2]


def build_opponent_model(name, game_records):
    """game_records: list of raw barricade.gg game dicts (as collected by
    collect_games.py). Learns opening repertoire and response stats."""
    if not NAME_OK.match(name):
        raise ValueError('Invalid opponent name')
    model = {
        'name': name,
        'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'games': 0, 'wins': 0,
        'rating_high': None, 'rating_low': None,
        'openings': {'red': defaultdict(lambda: defaultdict(int)),
                     'blue': defaultdict(lambda: defaultdict(int))},
        # position -> their move -> count (their side to move at that position)
        'responses': defaultdict(lambda: defaultdict(int)),
        # positions where they moved and lost shortly after (blunder rough)
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
            # moveTimes indexed by ply over the whole game
            model['move_times'].extend(mt)
        rb = game.get('p1Rating' if my_side == 0 else 'p2Rating')
        if rb:
            model['rating_high'] = max(model['rating_high'] or 0, rb)
            model['rating_low'] = min(model['rating_low'] or rb, rb)
    if model['move_times']:
        model['avg_move_ms'] = sum(model['move_times']) / len(model['move_times'])
    # convert defaultdicts to plain
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
    """Given an opponent model and the current position, return a short
    human line: what they typically do here and whether they've lost from it."""
    model = load_opponent(name)
    if not model:
        return None
    pos = ','.join(history)
    resp = model['responses'].get(pos)
    if not resp:
        # try openings if very early
        color = 'red' if len(history) % 2 == 0 else 'blue'
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
