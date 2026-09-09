"""Exercise the final advice path, not just the override helper."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import advice
import benchmark_match
import coach as c
import live_coach
import mcts_coach
import regress_losses


CASES = json.loads(Path(__file__).with_name('study').joinpath('tactical-loss-cases.json').read_text())
RECENT = json.loads(Path(__file__).with_name('study').joinpath('recent-loss-cases.json').read_text())


def mcts(move, **extra):
    return dict(scored=[(-60000, move)], principal_variation=[move],
                fallback=False, depth=None, simulations=60000, **extra)


class TacticalCrosscheckTests(unittest.TestCase):
    def test_recent_live_losses_are_overridden(self):
        self.assertEqual({row['code'] for row in RECENT},{'qm1cd6','xc2vpz'})
        for case in RECENT:
            with self.subTest(code=case['code'],ply=case['ply']):
                with patch.object(mcts_coach,'search',return_value=mcts(case['mcts_move'])):
                    result=advice.advise(case['history'],engine='mcts',seconds=4)
                self.assertEqual(result['scored'][0][1],case['best'])
                self.assertGreater(result['tactical_override']['gap'],80)
                self.assertTrue(result['tactical_override']['selective'])

    def test_real_loss_overrides_survive_final_sort(self):
        self.assertEqual({r['code'] for r in CASES}, {'6pg0tr', '7kz1qb'})
        for case in CASES:
            with self.subTest(code=case['code'], ply=case['ply']):
                with patch.object(mcts_coach, 'search', return_value=mcts(case['mcts_move'])):
                    result = advice.advise(case['history'], engine='mcts', seconds=4)
                self.assertEqual(result['scored'][0][1], case['best'])
                self.assertEqual(result['principal_variation'][0], case['best'])
                self.assertEqual(result['tactical_override']['minimax_best'], case['best'])
                self.assertTrue(all(score > -c.WIN for score, _ in result['scored']))
                self.assertFalse(result['tactical_override']['proven'])

    def test_equal_wall_alternative_is_not_overridden(self):
        case=CASES[0]
        with patch.object(mcts_coach, 'search', return_value=mcts('hh6')):
            result=advice.advise(case['history'], engine='mcts', seconds=4)
        self.assertEqual(result['scored'][0][1], 'hh6')
        self.assertNotIn('tactical_override', result)

    def test_crosscheck_uses_only_remaining_total_budget(self):
        clock=[0.0]; budgets=[]
        def sample(*args,**kwargs):
            budgets.append(args[2]); clock[0]+=args[2]
            return mcts(CASES[0]['mcts_move'])
        def tactical(*args, **kwargs):
            budgets.append(kwargs['time_limit']); clock[0]+=kwargs['time_limit']
            return dict(scored=[(0, 'hh5')], depth=1)
        with patch.object(advice.time, 'monotonic', side_effect=lambda:clock[0]), \
             patch.object(mcts_coach, 'search', side_effect=sample), \
             patch.object(c, 'search', side_effect=tactical):
            result=advice.advise(CASES[0]['history'], engine='mcts', seconds=4)
        self.assertAlmostEqual(sum(budgets), 4)
        self.assertLessEqual(result['elapsed'], 4)
        self.assertNotIn('tactical_override', result)

    def test_proven_result_does_not_run_crosscheck(self):
        for extra in (dict(forced_loss=True), dict(tactical='immediate win')):
            with patch.object(mcts_coach, 'search', return_value=mcts('hh5', **extra)), \
                 patch.object(c, 'candidate_search', side_effect=AssertionError('already proven')):
                advice.advise(CASES[0]['history'], engine='mcts', seconds=4)

    def test_expired_budget_and_exact_threshold_do_not_override(self):
        result=mcts('d3')
        with patch.object(c, 'candidate_search', side_effect=AssertionError('expired')):
            self.assertIs(advice._tactical_crosscheck([], 0, result, 0), result)
        mm=dict(scored=[(0,'hh5'),(advice.TACTICAL_GAP_CP,'d3')],
                depth=2,principal_variation=['hh5'])
        with patch.object(c, 'candidate_search', return_value=mm):
            self.assertNotIn('tactical_override', advice._tactical_crosscheck([],0,result,1))

    def test_explanation_does_not_claim_exact_wall_solution(self):
        case=CASES[0]
        with patch.object(mcts_coach, 'search', return_value=mcts(case['mcts_move'])):
            result=advice.advise(case['history'], engine='mcts', seconds=4)
        text=live_coach.explain(c.Game(case['history']), result)
        self.assertIn('not a proven win', text)
        self.assertNotIn('exact wall search', text)

    def test_benchmark_calls_live_decision_path(self):
        with patch.object(live_coach, 'query', return_value=dict(top=[(0,'e2')])) as query:
            move,_=benchmark_match.coach_move([], c.RED, 4)
        self.assertEqual(move,'e2')
        self.assertEqual(query.call_args.args[0]['red_left'],10)
        self.assertEqual(query.call_args.args[0]['seconds'],'4')
        self.assertFalse(query.call_args.kwargs['record_trace'])

    def test_loss_generator_rejects_incomplete_search(self):
        with patch.object(regress_losses,'long_search',return_value=('d1',{})), \
             patch.object(c,'search',return_value=dict(depth=1,scored=[(0,'d1'),(200,'e2')])):
            self.assertEqual(regress_losses.build_regressions('e2','blue','red',1), [])

    def test_loss_generator_keeps_wall_search_candidate_when_mcts_misses(self):
        with patch.object(regress_losses,'long_search',return_value=('e2',{})), \
             patch.object(c,'search',return_value=dict(depth=2,scored=[(0,'d1'),(200,'e2')])), \
             patch.object(regress_losses,'opponent_reply',return_value=('e8',{})):
            rows=regress_losses.build_regressions('e2','blue','red',1)
        self.assertEqual(rows[0]['best'],'d1')
        self.assertEqual(rows[0]['mcts_proposal'],'e2')
        self.assertFalse(rows[0]['proven'])


if __name__ == '__main__': unittest.main()
