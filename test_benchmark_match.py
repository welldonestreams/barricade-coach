import unittest

import benchmark_match as bm


class BenchmarkTests(unittest.TestCase):
    def test_summary_separates_coach_color_and_reports_uncertainty(self):
        rows=[
            dict(opponent='depth=3',coach_side=0,winner=0,coach_won=True,pair_id='a'),
            dict(opponent='depth=3',coach_side=0,winner=1,coach_won=False,pair_id='b'),
            dict(opponent='depth=3',coach_side=1,winner=1,coach_won=True,pair_id='a'),
            dict(opponent='depth=3',coach_side=1,winner=None,coach_won=False,pair_id='b'),
        ]
        summary=bm.summarize(rows)
        self.assertEqual(summary['by_coach_color']['red']['wins'],1)
        self.assertEqual(summary['by_coach_color']['blue']['unresolved'],1)
        self.assertEqual(summary['by_opponent']['depth=3']['games'],4)
        self.assertEqual(len(summary['by_coach_color']['red']['winrate_95']),2)
        self.assertEqual(summary['paired_color']['both_won'],1)
        self.assertEqual(summary['paired_color']['unresolved'],1)

    def test_seed_is_stable_and_each_game_gets_a_trace(self):
        calls=[]
        def fake_play(opener,side,spec,seconds,seed,opening_seconds):
            calls.append(seed)
            return side,3,list(opener)+['e3'],[dict(move='e3')]
        from unittest.mock import patch
        with patch.object(bm,'play',side_effect=fake_play):
            first=bm.run(['depth=2'],1,2,[['e2','e8']],99)
            first_calls=list(calls);calls.clear()
            second=bm.run(['depth=2'],1,2,[['e2','e8']],99)
        self.assertEqual(first_calls,calls)
        self.assertEqual([r['seed'] for r in first],[r['seed'] for r in second])
        self.assertTrue(all(r['decisions'] for r in first))


if __name__=='__main__':unittest.main()
