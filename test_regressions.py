#!/usr/bin/env python3
"""Regression test for loss-derived cases.

study/regressions/*.json holds critical positions from real losses where the
played move was materially worse than the best found by a long offline search.
The cases are generated with MCTS (non-deterministic), so this test re-verifies
each case with the DETERMINISTIC minimax engine and only asserts on the solid
subset minimax confirms -- keeping the test stable across runs.

Two assertions:
  1. well-formedness: both the played and best moves are legal.
  2. (confirmed subset) minimax depth-2 ranks `best` no worse than `played`,
     i.e. the blunder is real under an independent engine too.
"""
import json
import unittest
from pathlib import Path

import coach as c

ROOT = Path(__file__).resolve().parent
REGRESS = ROOT / 'study' / 'regressions'


def load_cases():
    cases = []
    if not REGRESS.exists():
        return cases
    for path in sorted(REGRESS.glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        for bl in data.get('blunders', []):
            bl['_code'] = path.stem
            cases.append(bl)
    return cases


class RegressionTests(unittest.TestCase):
    def test_regression_cases_are_well_formed(self):
        cases = load_cases()
        self.assertGreater(len(cases), 0, 'no regression cases present')
        for bl in cases:
            g = c.Game(bl['history'])
            legal = set(g.moves(g.to_move))
            self.assertIn(bl['best'], legal,
                          f"{bl['_code']} ply {bl['ply']}: best {bl['best']} illegal")
            self.assertIn(bl['played'], legal,
                          f"{bl['_code']} ply {bl['ply']}: played {bl['played']} illegal")

    def test_confirmed_blunders_beaten_by_minimax(self):
        confirmed = 0
        for bl in load_cases():
            g = c.Game(bl['history'])
            r = c.search(bl['history'], g.to_move, depth=2, time_limit=3)
            scored = dict((m, s) for s, m in r['scored'])
            best_sc = scored.get(bl['best'])
            played_sc = scored.get(bl['played'])
            if best_sc is None or played_sc is None:
                continue  # depth-2 didn't surface both; not assertable
            confirmed += 1
            self.assertLessEqual(best_sc, played_sc,
                f"{bl['_code']} ply {bl['ply']}: best {bl['best']} scored {best_sc} "
                f"but played {bl['played']} scored {played_sc}")
        self.assertGreater(confirmed, 0, 'no case was confirmable by minimax')


if __name__ == '__main__':
    unittest.main()
