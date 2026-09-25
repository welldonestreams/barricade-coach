"""Shared held-out repair cases for training and candidate promotion."""
from functools import lru_cache
import json
from pathlib import Path
import coach as c

ROOT=Path(__file__).resolve().parent
CASES=ROOT/'study'/'repair-gate-cases.json'


@lru_cache(maxsize=1)
def rows():
    """Return validated held-out repair rows.

    Fail closed: a missing/unreadable gate file or a malformed gate case
    raises instead of being silently skipped. A skipped case would let a
    candidate pass the promotion gate without being tested on every required
    repair, and would quietly disable the training-side leakage filter.
    """
    if not CASES.exists():
        raise FileNotFoundError(f'repair gate file missing: {CASES}')
    try:
        data=json.loads(CASES.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError) as e:
        raise ValueError(f'unreadable repair gate file {CASES}: {e}') from e
    if not isinstance(data,list):
        raise ValueError(f'repair gate file {CASES} is not a JSON list')
    out=[]
    for i,row in enumerate(data):
        if (not isinstance(row,dict) or not isinstance(row.get('history'),list)
                or not isinstance(row.get('best'),str) or not row['best']):
            raise ValueError(f'malformed repair gate case #{i} in {CASES}: {row!r}')
        out.append(row)
    return tuple(out)


@lru_cache(maxsize=1)
def histories():
    """Canonical game histories that must never be used to train a candidate."""
    return frozenset(tuple(row['history']) for row in rows())


def position_key(history):
    """Transposition-independent board identity for leakage prevention."""
    g=c.Game(history)
    return (g.pawns[c.RED],g.pawns[c.BLUE],tuple(sorted(g.walls)),
            g.remaining[c.RED],g.remaining[c.BLUE],g.to_move)


@lru_cache(maxsize=1)
def positions():
    """Held-out board states, including equivalent move-order transpositions."""
    return frozenset(position_key(row['history']) for row in rows())
