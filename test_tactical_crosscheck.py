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

    def test_selective_check_cannot_replace_an_mcts_wall(self):
        hist='e2,e8,e3,e7,e4,e6,hd3,hg3,e5,he6,d6,hc6,he5,ve4,hb3,ha6,c6,vf5,vc5,va2'.split(',')
        mm=dict(scored=[(0,'hc4'),(300,'vb4')],depth=3,
                principal_variation=['hc4'],selective=True,beam=8)
        with patch.object(c,'candidate_search',return_value=mm):
            result=advice._tactical_crosscheck(hist,c.Game(hist).to_move,
                                                mcts('vb4'),2)
        self.assertEqual(result['scored'][0][1],'vb4')
        self.assertNotIn('tactical_override',result)

    def test_selective_check_can_replace_a_wasted_wall_with_pawn_tempo(self):
        case=next(row for row in json.loads(Path('study/repair-gate-cases.json').read_text())
                  if row.get('code')=='52s6kd' and row.get('ply')==16)
        mm=dict(scored=[(0,'e5'),(300,'vb5')],depth=3,
                principal_variation=['e5'],selective=True,beam=8)
        with patch.object(c,'candidate_search',return_value=mm):
            result=advice._tactical_crosscheck(case['history'],
                                                c.Game(case['history']).to_move,
                                                mcts('vb5'),2)
        self.assertEqual(result['scored'][0][1],'e5')
        self.assertEqual(result['tactical_override']['mcts_top'],'vb5')

    def test_mcts_gets_full_budget_before_bounded_tactical_check(self):
        budgets=[]
        def sample(*args,**kwargs):
            budgets.append(args[2])
            return mcts(CASES[0]['mcts_move'])
        def tactical(*args, **kwargs):
            budgets.append(kwargs['time_limit'])
            return dict(scored=[(0, 'hh5')], depth=1)
        with patch.object(mcts_coach, 'search', side_effect=sample), \
             patch.object(c, 'candidate_search', side_effect=tactical), \
             patch.object(c, 'search', side_effect=tactical):
            result=advice.advise(CASES[0]['history'], engine='mcts', seconds=4)
        self.assertEqual(len(budgets),2)
        self.assertTrue(3.8 <= budgets[0] < 4)
        self.assertEqual(budgets[1],3)
        self.assertLessEqual(result['elapsed'], 7.1)
        self.assertNotIn('tactical_override', result)

    def test_proven_result_is_not_overridden(self):
        for extra in (dict(forced_loss=True), dict(tactical='immediate win')):
            with patch.object(mcts_coach, 'search', return_value=mcts('hh5', **extra)), \
                 patch.object(c, 'candidate_search', return_value=dict(
                     scored=[(0,'d3'),(200,'hh5')],depth=2,principal_variation=['d3'])):
                result=advice.advise(CASES[0]['history'], engine='mcts', seconds=4)
            self.assertEqual(result['scored'][0][1],'hh5')
            self.assertNotIn('tactical_override',result)

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

    def test_deep_losing_endgame_is_presented_as_best_resistance(self):
        hist='e2,e8,e3,e7,e4,e6,e5,e4,he3,he5,hb5,vd5,vc6,d4,hc3,hg5,ha3,he4,vd3,vh4,hh2,vb4,f5,hh8,g5,hf8,h5,hd8,vg3,hb8,h4,d5,h3,d6,i3,d7,i4,e7,i5,f7,i6,g7,h6,h7,h8'.split(',')
        mm=dict(scored=[(862,'g7'),(974,'g8')],depth=5,
                principal_variation=['g7','vg6','g8','f8','h8'])
        with patch.object(mcts_coach,'search',return_value=mcts('g7')), \
             patch.object(c,'candidate_search',return_value=mm):
            result=advice.advise(hist,engine='mcts',seconds=4)
        self.assertEqual(result['scored'][0][1],'g7')
        self.assertIn('best resistance',result['position_warning'])
        self.assertIn('looks badly losing',live_coach.explain(c.Game(hist),result))

    def test_no_wall_pawn_shuffle_cannot_override_mcts(self):
        hist='e2,e8,e3,e7,e4,e6,he3,d6,hc3,d5,ha3,hd5,vd4,hf5,hb5,vg4,vc5,vg2,vf2,d4,vc7,c4,f4,b4,g4,a4,g3,a5,g2,hh8,ha6,a6,hb8,b6,g1,c6,h1,c7,h2,hg3,h3,c8,i3,b8,i4,a8,h4,a9,h5,b9,h6,hg6'.split(',')
        with patch.object(mcts_coach,'search',return_value=mcts('g6')):
            result=advice.advise(hist,engine='mcts',seconds=4)
        self.assertEqual(result['crosscheck_depth'],5)
        self.assertEqual(result['scored'][0][1],'g6')
        self.assertNotIn('tactical_override',result)

    def test_no_wall_endgame_gets_stable_extended_mcts_budget(self):
        hist='e2,e8,e3,e7,e4,e6,he3,d6,hc3,d5,ha3,hd5,vd4,hf5,hb5,vg4,vc5,vg2,vf2,d4,vc7,c4,f4,b4,g4,a4,g3,a5,g2,hh8,ha6,a6,hb8,b6,g1,c6,h1,c7,h2,hg3,h3,c8,i3,b8,i4,a8,h4,a9,h5,b9,h6,hg6'.split(',')
        seen=[]
        def sample(*args,**kwargs):
            seen.append((args[2],args[3],kwargs.get('workers')))
            return mcts('g6')
        with patch.object(mcts_coach,'search',side_effect=sample):
            result=advice.advise(hist,engine='mcts',seconds=4,seed=7)
        self.assertEqual(seen,[(15.0,200000,8)])
        self.assertTrue(result['extended_endgame_search'])
        self.assertEqual(result['mcts_budget'],15.0)

    def test_ambiguous_pawn_stop_expands_for_delayed_wall(self):
        hist='he8,d9,hc8,he1,f1,hg1,g1,vh1,f1,vf8,e1,c9,d1,b9,d2,b8,va6,ha8,d3,hg7,e3,hh8,f3,b7,g3'.split(',')
        with patch.object(mcts_coach,'search',return_value=mcts('c7')):
            result=advice.advise(hist,engine='mcts',seconds=4)
        self.assertTrue(result.get('crosscheck_expanded'))
        self.assertIn(result['scored'][0][1],{'hf3','hf4','hf5','hf6'})
        self.assertEqual(result['tactical_override']['mcts_top'],'c7')

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
