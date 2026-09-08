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
        # signed "who is ahead" signals for wall economy
        self.assertTrue(any(x.startswith('progress_lead:') for x in state))
        self.assertTrue(any(x.startswith('wall_lead:') for x in state))
        advanced=c.Game('e2,e8,e3,e7,e4,e6')
        self.assertIn('wall_no_immediate_gain',pv.action_features(advanced,'ha4'))
        self.assertIn('wall_net:1',pv.action_features(advanced,'hd3'))
        # wall placement quality: every wall move carries an opponent-resilience
        # change feature (does the wall cut the opponent's route flexibility?)
        wall=c.Game('e2,e8,e3,e7,e4,e6,he3,hd4,f4,hf4')
        af=pv.action_features(wall,'vc5')
        self.assertTrue(any(x.startswith('wall_opp_resilience_change:') for x in af))

    def test_teacher_keeps_depth2_wall_positions(self):
        # Wall-heavy positions often can't finish depth-3 in budget. A completed
        # depth-2 search is still a usable graded label and must not be dropped.
        fixture=Path(__file__).with_name('study')/'additional'/'1ttbdt.json'
        with patch.object(c,'search',return_value=dict(depth=2,scored=[(0,'e2')],nodes=10,
                                                      tt_hits=0,elapsed=0.1)):
            row=teacher_data.record(fixture,__import__('random').Random(1),8,8,3,1)
        self.assertIsNotNone(row)
        self.assertEqual(row['teacher']['depth'],2)


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

    def test_loss_regressions_become_weighted_independent_examples(self):
        path=Path(__file__).with_name('study')/'recent-loss-cases.json'
        rows=list(train_policy.examples([path]))
        self.assertEqual({row['game_hash'].split(':')[1] for row in rows},{'qm1cd6','xc2vpz'})
        self.assertTrue(all(row['weight']==8 and row['source']=='independent-loss-regression'
                            for row in rows))


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

    def test_arena_ignores_non_row_json_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mixed.jsonl'
            path.write_text('"not a row"\n'+json.dumps(dict(history=[],player_holdout=True)),encoding='utf-8')
            self.assertEqual(arena.holdout_positions([path],1,1,player_only=True),[[]])

    def test_arena_runs_complete_color_swapped_pairs_in_parallel(self):
        fake=dict(points=1.0,winner=c.RED,plies=10,illegal=0,candidate_latencies=[.1])
        with patch.object(arena,'play',return_value=fake):
            report=arena.evaluate({},[[],['e2','e8']],.01,1,workers=2)
        self.assertEqual(report['arena_pairs'],2)
        self.assertEqual(report['games'],4)
        self.assertEqual(len(report['records']),4)

    def test_repair_gate_is_disjoint_from_training(self):
        # The promotion gate's repair positions must be held out from training,
        # otherwise a candidate can pass by memorizing the 8x-weighted training
        # examples (the exact positions the gate would then re-test).
        root=Path(__file__).with_name('study')
        train_keys=set()
        for fn in ('tactical-loss-cases.json','recent-loss-cases.json'):
            for case in json.loads((root/fn).read_text(encoding='utf-8')):
                train_keys.add((case['code'],case['ply']))
        gate_cases=json.loads((root/'repair-gate-cases.json').read_text(encoding='utf-8'))
        self.assertGreater(len(gate_cases),0,'gate file missing or empty')
        for case in gate_cases:
            key=(case['code'],case['ply'])
            self.assertNotIn(key,train_keys,
                f'gate case {key} also appears in the training set; gate would test memorization')
            # and the expected move must be legal on its position
            self.assertIn(case['best'],c.Game(case['history']).moves(c.Game(case['history']).to_move))

    def test_repair_gate_histories_are_filtered_from_every_training_source(self):
        gate_case=json.loads((Path(__file__).with_name('study')/'repair-gate-cases.json').read_text())[0]
        normal=dict(history=['e2'],split='train',game_hash='normal',policy={'e8':1})
        leaked=dict(history=gate_case['history'],split='train',game_hash='leaked',
                    policy={gate_case['best']:1})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'teacher.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in (leaked,normal)),encoding='utf-8')
            loaded=list(train_policy.examples([path]))
        self.assertEqual([row['game_hash'] for row in loaded],['normal'])

    def test_arena_holdout_excludes_repair_gate_histories(self):
        gate_case=json.loads((Path(__file__).with_name('study')/'repair-gate-cases.json').read_text())[0]
        rows=[dict(history=gate_case['history'],player_holdout=True),
              dict(history=['e2'],player_holdout=True)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'teacher.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows),encoding='utf-8')
            self.assertEqual(arena.holdout_positions([path],1,1,player_only=True),[['e2']])

    def test_arena_regression_default_is_held_out_file(self):
        # arena's default --regression must point at the held-out gate file,
        # NOT the training loss files. (Guards the fix against drift.)
        self.assertEqual(arena.GATE_CASES.name,'repair-gate-cases.json')
        self.assertTrue(arena.GATE_CASES.exists())
        self.assertNotIn(str(arena.GATE_CASES),
                         ['tactical-loss-cases.json','recent-loss-cases.json'])



if __name__=='__main__':unittest.main()
