"""Shared held-out repair cases for training and candidate promotion."""
from functools import lru_cache
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CASES=ROOT/'study'/'repair-gate-cases.json'


@lru_cache(maxsize=1)
def rows():
    """Return well-formed held-out repair rows, or an empty tuple on I/O failure."""
    try:
        data=json.loads(CASES.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError):
        return ()
    return tuple(row for row in data if isinstance(row,dict)
                 and isinstance(row.get('history'),list) and row.get('best')) if isinstance(data,list) else ()


@lru_cache(maxsize=1)
def histories():
    """Canonical game histories that must never be used to train a candidate."""
    return frozenset(tuple(row['history']) for row in rows())
