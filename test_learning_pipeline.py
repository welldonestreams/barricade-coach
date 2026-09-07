import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import arena
import coach as c
import mcts_coach
import policy_value as pv
import teacher_data
import train_policy
import measure_real_games


class PolicyValueTests(unittest.TestCase):
    def test_route_flexibility_and_wall_quality_are_model_features(self):
        g=c.Game()
        self.assertEqual(c.path_resilience(frozenset(),g.pawns[c.RED],c.GOALS[c.RED]),3)
        self.assertEqual(g.route_options(c.RED),3)
        state=pv.state_features(g)
        self.assertIn('my_resilience:3',state)
        self.assertIn('my_route_options:3',state)
        advanced=c.Game('e2,e8,e3,e7,e4,e6')
        self.assertIn('wall_no_immediate_gain',pv.action_features(advanced,'ha4'))
        self.assertIn('wall_net:1',pv.action_features(advanced,'hd3'))

    def test_teacher_search_reports_transposition_reuse_metric(self):
        data=json.loads((Path(__file__).with_name('study')/'additional'/'1ttbdt.json').read_text())
        history=c.parse_history(data['historyCsv'])[:53]
        result=c.search(history,c.Game(history).to_move,depth=2,time_limit=3)
        self.assertGreaterEqual(result['depth'],1)
        self.assertIn('tt_hits',result)
        self.assertGreaterEqual(result['tt_hits'],0)

    def test_model_trains_preferred_legal_move_and_value(self):
        row=dict(history=[],split='train',game_hash='g',winner=c.RED,policy={'e2':1.0})
        model=train_policy.train([row]*20,epochs=3,rate=.08,value_rate=.04,seed=1)
        priors=pv.policy_priors(model,c.Game())
        self.assertEqual(max(priors,key=priors.get),'e2')
        self.assertGreater(pv.value(model,c.Game()),0)
        self.assertAlmostEqual(sum(priors.values()),1)

    def test_graded_teacher_value_trains_without_winner(self):
        # A graded search value must train the value head even when no eventual
        # winner label exists (the core weakness: outcome-only labels are weak).
        row=dict(history=[],split='train',game_hash='g',policy={'e2':1.0},value=-0.8)
        model=train_policy.train([row]*20,epochs=3,rate=.08,value_rate=.04,seed=1)
        self.assertLess(pv.value(model,c.Game()),0)  # learns "side to move is losing"


    def test_color_normalization_points_both_sides_toward_rank_two(self):
        self.assertEqual(pv.normalize_move('e8',c.BLUE),'e2')
        self.assertEqual(pv.normalize_move('hh7',c.BLUE),'hh2')

    def test_player_holdout_is_stable_and_excluded_from_training(self):
        name=next('player'+str(i) for i in range(1000) if teacher_data.heldout_player('player'+str(i)))
        self.assertTrue(teacher_data.heldout_player(name.upper()))
        rows=[dict(history=[],split='train',game_hash='held',player_holdout=True,
                   winner=0,policy={'e2':1}),
              dict(history=['e2','e8'],split='train',game_hash='train',player_holdout=False,
                   winner=0,policy={'e3':1})]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rows.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            loaded=list(train_policy.examples([path]))
        self.assertEqual([r['game_hash'] for r in loaded],['train'])

    def test_unpromoted_or_wrong_code_model_is_never_live(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(pv,'CHAMPION',Path(tmp)/'champion.json'):
            model=pv.new_model();model['report']={'promoted':False}
            pv.save(model,pv.CHAMPION);self.assertIsNone(pv.load_champion())
            model['report']={'promoted':True,'arena_pairs':100,'score_lower_95':.6,
                             'policy_code_sha256':'wrong'}
            pv.save(model,pv.CHAMPION);self.assertIsNone(pv.load_champion())

    def test_guided_adapter_uses_legal_soft_priors(self):
        g=c.Game();model=pv.new_model();model['policy']['ab:a:e2']=8
        priors=pv.policy_priors(model,g)
        result=mcts_coach.search([],time_limit=2,rollouts=128,seed=4,workers=1,
                                 root_priors=priors)
        self.assertTrue(result['policy_guided'])
        self.assertIn(result['scored'][0][1],g.moves(g.to_move))
        self.assertEqual(set(priors),set(g.moves(g.to_move)))

    def test_search_priors_use_policy_and_value(self):
        g=c.Game();model=pv.new_model();model['policy']['ab:a:e2']=2
        priors=pv.search_priors(model,g)
        self.assertEqual(max(priors,key=priors.get),'e2')
        self.assertAlmostEqual(sum(priors.values()),1)

    def test_teacher_requires_completed_requested_depth(self):
        fixture=Path(__file__).with_name('study')/'additional'/'1ttbdt.json'
        with patch.object(c,'search',return_value=dict(depth=1,scored=[(0,'e2')])):
            self.assertIsNone(teacher_data.record(fixture,__import__('random').Random(1),8,8,2,1))

    def test_arena_confidence_needs_real_pairs(self):
        self.assertEqual(arena.lower95([1]),0)
        self.assertGreater(arena.lower95([1]*100),.5)
        self.assertLessEqual(arena.lower95([.5]*100),.5)
        self.assertEqual(measure_real_games.interval(0,0),[None,None])
        self.assertLess(measure_real_games.interval(85,100)[0],.85)


if __name__=='__main__':unittest.main()
