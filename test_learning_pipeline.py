import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import arena
import coach as c
import mcts_coach
import policy_value as pv
import repair_gate
import teacher_data
import train_nn
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
        self.assertTrue(all(row['weight']==16 and row['source']=='independent-loss-regression'
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
        fake=dict(points=1.0,winner=c.RED,plies=10,illegal=0,decision_failures=0,candidate_latencies=[.1])
        with patch.object(arena,'play',return_value=fake):
            report=arena.evaluate({},[[],['e2','e8']],.01,1,workers=2)
        self.assertEqual(report['arena_pairs'],2)
        self.assertEqual(report['games'],4)
        self.assertEqual(len(report['records']),4)

    def test_arena_timeout_is_a_recorded_failed_decision(self):
        with patch.object(arena.advice,'advise',side_effect=c.SearchTimeout):
            move,elapsed,result=arena.decision(c.Game(),.01,1,None)
        self.assertIn(move,c.Game().moves(c.RED))
        self.assertGreaterEqual(elapsed,0)
        self.assertTrue(result['arena_timeout'])

    def test_repair_gate_is_disjoint_from_training(self):
        # The promotion gate's repair positions must be held out from training,
        # otherwise a candidate can pass by memorizing the 8x-weighted training
        # examples (the exact positions the gate would then re-test).
        root=Path(__file__).with_name('study')
        train_keys=set()
        for fn in ('tactical-loss-cases.json','recent-loss-cases.json',
                   'training-loss-cases.json','pawn-loss-cases.json'):
            for case in json.loads((root/fn).read_text(encoding='utf-8')):
                train_keys.add((case['code'],case['ply']))
        gate_cases=json.loads((root/'repair-gate-cases.json').read_text(encoding='utf-8'))
        self.assertGreater(len(gate_cases),0,'gate file missing or empty')
        for case in gate_cases:
            key=(case['code'],case['ply'])
            self.assertNotIn(key,train_keys,
                f'gate case {key} also appears in the training set; gate would test memorization')
            # and the expected move must be legal on its position
            legal=c.Game(case['history']).moves(c.Game(case['history']).to_move)
            self.assertIn(case['best'],legal)
            self.assertIn(case['best'],case['acceptable'])
            self.assertTrue(set(case['acceptable']).issubset(legal))

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

    def test_repair_gate_failure_skips_expensive_arena(self):
        self.assertTrue(arena.repair_gate_passes([dict(correct=True)]))
        self.assertFalse(arena.repair_gate_passes([dict(correct=True),dict(correct=False)]))
        self.assertFalse(arena.repair_gate_passes([]))

    def test_arena_repair_gate_fails_closed(self):
        # A candidate must never pass the gate with a case silently skipped:
        # malformed cases, illegal expected moves, and missing case files all
        # raise instead of dropping the case from the count.
        model=pv.new_model()
        with tempfile.TemporaryDirectory() as tmp:
            good=Path(tmp)/'good.json'
            good.write_text(json.dumps([dict(code='x',ply=1,history=['e2'],best='e8')]),encoding='utf-8')
            rows=arena.repair_screen(model,[good])
            self.assertEqual(len(rows),1)
            bad=Path(tmp)/'bad.json'
            bad.write_text(json.dumps([dict(code='x',ply=1,history=['e2'],best='zz9')]),encoding='utf-8')
            with self.assertRaises(ValueError):
                arena.repair_screen(model,[bad])
            ugly=Path(tmp)/'ugly.json'
            ugly.write_text(json.dumps([dict(nope=True)]),encoding='utf-8')
            with self.assertRaises(ValueError):
                arena.repair_screen(model,[ugly])
            with self.assertRaises(FileNotFoundError):
                arena.repair_screen(model,[Path(tmp)/'missing.json'])

    def test_repair_screen_thresholds_are_frozen_predicates(self):
        # acceptable in top-10 OR >=5% prior mass passes the raw screen;
        # deeper ranks with negligible mass fail it.
        legal_moves = [m for m in c.Game(['e2']).moves(c.BLUE)]
        fake = {m: 0.001 for m in legal_moves}
        fake['e8'] = 0.9
        case = dict(code='x', ply=2, history=['e2'], best='e8', acceptable=['e8'])
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pv, 'search_priors', return_value=fake):
            path = Path(tmp)/'c.json'
            path.write_text(json.dumps([case]), encoding='utf-8')
            rows = arena.repair_screen(pv.new_model(), [path])
            self.assertTrue(rows[0]['passed'])
            self.assertEqual(rows[0]['raw_rank'], 1)
        # expected at rank 40 with 0.001 mass: fails the frozen screen
        fake2 = {m: 0.001 for m in legal_moves}
        fake2['e8'] = 0.0
        fake2[legal_moves[0]] = 0.9
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pv, 'search_priors', return_value=fake2):
            path = Path(tmp)/'c.json'
            path.write_text(json.dumps([case]), encoding='utf-8')
            rows = arena.repair_screen(pv.new_model(), [path])
            self.assertFalse(rows[0]['passed'])

    def test_live_repair_check_records_and_parity(self):
        # Patch the production path: guided acceptable on all seeds beats an
        # unguided baseline that misses one; parity + records are checked.
        legal = [m for m in c.Game(['e2']).moves(c.BLUE)]
        acc = ['e8']
        seen=[]
        def fake_advise(hist, side, seconds, seed, policy_model, **kw):
            seen.append(policy_model)
            ok = policy_model is not False or seed != arena.REPAIR_SEEDS[-1]
            return dict(scored=[(-10, 'e8' if ok else legal[0])], simulations=500,
                        elapsed=0.01, fallback=False, timed_out=True,
                        policy_model_id='m1' if policy_model else None)
        case = dict(code='x', ply=2, history=['e2'], best='e8', acceptable=acc)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(arena.advice, 'advise', side_effect=fake_advise):
            path = Path(tmp)/'c.json'
            path.write_text(json.dumps([case]), encoding='utf-8')
            rows = arena.live_repair_check(pv.new_model(), [path],
                                           seconds=0.1, seeds=arena.REPAIR_SEEDS,
                                           workers=2)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['correct'])
        self.assertEqual(rows[0]['guided_ok'], 5)
        self.assertEqual(rows[0]['unguided_ok'], 4)
        self.assertTrue(rows[0]['parity_ok'])
        self.assertIn(False,seen)
        self.assertNotIn(None,seen)
        run = rows[0]['runs'][0]
        self.assertEqual(run['guided']['move'], 'e8')
        self.assertIn('simulations', run['guided'])
        self.assertIn('elapsed', run['guided'])
        # SearchTimeout inside advise => failed run, never a crash
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(arena.advice, 'advise', side_effect=c.SearchTimeout):
            path = Path(tmp)/'c.json'
            path.write_text(json.dumps([case]), encoding='utf-8')
            rows = arena.live_repair_check(pv.new_model(), [path],
                                           seconds=0.1, seeds=arena.REPAIR_SEEDS[:2],
                                           workers=2)
        self.assertFalse(rows[0]['correct'])
        self.assertEqual(rows[0]['guided_ok'], 0)

    def test_production_root_priors_keep_an_exploration_floor(self):
        import mcts_coach
        legal=['e2','d1','f1','ha1']
        raw={'e2':0.999,'ha1':0.001,'illegal':99}
        shaped=mcts_coach.tempered_root_priors(raw,legal)
        self.assertEqual(set(shaped),set(legal))
        self.assertAlmostEqual(sum(shaped.values()),1.0,places=9)
        floor=mcts_coach.ROOT_UNIFORM_MIX/len(legal)
        self.assertTrue(all(value>=floor for value in shaped.values()))
        self.assertGreater(shaped['ha1'],raw['ha1'])

    def test_wide_tactical_root_covers_known_sharp_walls(self):
        cases=json.loads((Path(__file__).with_name('study')/
                          'repair-gate-cases.json').read_text(encoding='utf-8'))
        wanted={('52s6kd',18):'vg3',('zjj0bn',27):'vg1',
                ('zjj0bn',29):'hg3'}
        found=0
        for case in cases:
            key=(case.get('code'),case.get('ply'))
            if key not in wanted:
                continue
            game=c.Game(case['history'])
            self.assertIn(wanted[key],c.relevant_walls(game,slack=4))
            found+=1
        self.assertEqual(found,len(wanted))

    def test_nearby_search_coverage_cases_stay_in_wide_root(self):
        path=Path(__file__).with_name('study')/'search-coverage-cases.json'
        rows=json.loads(path.read_text(encoding='utf-8'))
        self.assertGreaterEqual(len(rows),4)
        for row in rows:
            game=c.Game(row['history'])
            legal=set(game.moves(game.to_move))
            wide=set(c.relevant_walls(game,slack=4))
            self.assertTrue(set(row['must_consider']).issubset(legal))
            self.assertTrue(set(row['must_consider']).issubset(wide))

    def test_wide_root_narrows_only_after_complete_coverage_pass(self):
        case=next(row for row in json.loads((Path(__file__).with_name('study')/
                         'repair-gate-cases.json').read_text())
                  if row['code']=='benchmark-blue-2673606728')
        game=c.Game(case['history'])
        result=c.candidate_search(case['history'],game.to_move,depth=3,
                                  time_limit=8,beam=8,
                                  root_moves=['e2','hg4'],wide_root=True)
        self.assertGreaterEqual(result['depth'],2)
        self.assertGreater(result['initial_root_candidates'],
                           result['root_candidates'])
        self.assertTrue(result['staged_root_narrowing'])
        self.assertIn('e2',[move for _,move in result['scored']])
        self.assertIn('hg4',[move for _,move in result['scored']])

    def test_complete_depth2_pawn_margin_is_not_lost_to_deeper_tie(self):
        case=next(row for row in json.loads((Path(__file__).with_name('study')/
                       'repair-gate-cases.json').read_text())
                  if row['code']=='52s6kd' and row['ply']==16)
        result=c.candidate_search(case['history'],c.BLUE,depth=3,time_limit=8,
                    beam=8,root_moves=['e5','vb5'],wide_root=True,
                    preserve_depth2_pawn_margin=75)
        self.assertEqual(result['depth'],2)
        self.assertEqual(result['scored'][0][1],'e5')
        self.assertTrue(result['stopped_on_decisive_pawn'])

    def test_repair_gate_rows_fail_closed_on_bad_file(self):
        # rows()/histories() silently returning () on a broken gate file would
        # disable the training-side leakage filter. They must raise instead.
        with tempfile.TemporaryDirectory() as tmp,patch.object(repair_gate,'CASES',Path(tmp)/'cases.json'):
            repair_gate.rows.cache_clear();repair_gate.histories.cache_clear()
            repair_gate.CASES.write_text(json.dumps([dict(code='ok',history=['e2'],best='e8')]),encoding='utf-8')
            self.assertEqual(len(repair_gate.rows()),1)
            repair_gate.rows.cache_clear()
            repair_gate.CASES.write_text(json.dumps([dict(no='history')]),encoding='utf-8')
            with self.assertRaises(ValueError):
                repair_gate.rows()
            repair_gate.rows.cache_clear()
            repair_gate.CASES.write_text('not json',encoding='utf-8')
            with self.assertRaises(ValueError):
                repair_gate.rows()
            repair_gate.rows.cache_clear()
            repair_gate.CASES.unlink()
            with self.assertRaises(FileNotFoundError):
                repair_gate.rows()
            repair_gate.rows.cache_clear()

    def test_arena_regression_default_is_held_out_file(self):
        # arena's default --regression must point at the held-out gate file,
        # NOT the training loss files. (Guards the fix against drift.)
        self.assertEqual(arena.GATE_CASES.name,'repair-gate-cases.json')
        self.assertTrue(arena.GATE_CASES.exists())
        self.assertNotIn(str(arena.GATE_CASES),
                         ['tactical-loss-cases.json','recent-loss-cases.json'])

    def test_neural_schema2_round_trips_and_dispatches(self):
        import nn_model
        g=c.Game();m=nn_model.new_model(seed=42)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'m.json'
            pv.save(m,path);m2=pv.load(path)
        self.assertEqual(pv.model_id(m),pv.model_id(m2))
        self.assertAlmostEqual(pv.value(m,g),pv.value(m2,g),places=9)
        pr=pv.search_priors(m,g)
        self.assertAlmostEqual(sum(pr.values()),1.0,places=6)
        self.assertTrue(all(v>=0 for v in pr.values()))
        # schema-1 path still intact
        self.assertIsNotNone(pv.model_id(pv.new_model()))
        self.assertEqual(len(pv.code_hash()),64)

    def test_neural_action_features_include_wall_tempo_and_geometry(self):
        import nn_model
        g=c.Game('e2,e8,e3,e7,e4,e6'.split(','))
        wall=nn_model.action_features(g,'hd3')
        pawn=nn_model.action_features(g,'e5')
        self.assertEqual(len(wall),len(pawn))
        self.assertEqual(len(wall),11)
        self.assertTrue(all(isinstance(v,float) for v in wall))

    def test_neural_loader_rejects_stale_or_malformed_dimensions(self):
        import nn_model
        model=nn_model.new_model(seed=5)
        model['metadata']['dims']['action']-=1
        with self.assertRaises(ValueError):nn_model.validate(model)
        model=nn_model.new_model(seed=5)
        model['policy']['W1'].pop()
        with self.assertRaises(ValueError):nn_model.validate(model)

    def _write_rows(self, tmp, name, rows):
        path=Path(tmp)/name
        path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
        return path

    def test_loss_rows_are_soft_and_merged_by_canonical_state(self):
        # A mined-loss row and a teacher row reaching the SAME board state
        # must merge into one example with an averaged target -- never two
        # competing one-hot gradients.
        hist=['e2','e8','e3','e7','e4','e6']
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp)/'corpus.jsonl'
            l=Path(tmp)/'losses.jsonl'
            t.write_text(json.dumps(dict(game_hash='t1',split='train',player_holdout=False,
                history=hist,value=0.5,policy={'hd3':0.6,'e5':0.4},
                teacher=dict(depth=3)))+'\n',encoding='utf-8')
            l.write_text(json.dumps(dict(game_hash='l1',split='train',player_holdout=False,
                history=hist,value=-0.2,policy={'hd3':0.9,'vf5':0.1},
                teacher=dict(depth=2)))+'\n',encoding='utf-8')
            rows=list(train_nn.load_rows([t,l]))
            sources={r['source'] for r in rows}
            self.assertEqual(sources,{'teacher','loss'})
            ex=train_nn.build_examples(rows)
            self.assertEqual(len(ex),1,'canonical duplicates must merge')
            self.assertEqual(ex[0]['kind'],'loss','hard supervision must retain stratified sampling')
            self.assertIn('hd3',ex[0]['target'])
            self.assertGreater(ex[0]['weight'],1.0,'merged rows keep combined weight')

    def test_deeper_label_with_same_source_hash_refines_shallow_target(self):
        hist=['e2','e8','e3','e7','e4','e6']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'loss-deep.jsonl'
            common=dict(game_hash='mined:x:7',split='train',player_holdout=False,
                        source='loss',history=hist,value=0.0)
            rows=[{**common,'policy':{'e5':0.8,'hd3':0.2},
                   'teacher':{'depth':2}},
                  {**common,'policy':{'hd3':0.9,'e5':0.1},
                   'teacher':{'depth':4},'weight':5.0}]
            path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n',
                            encoding='utf-8')
            loaded=list(train_nn.load_rows([path]))
            self.assertEqual(len(loaded),2)
            examples=train_nn.build_examples(loaded)
            self.assertEqual(len(examples),1)
            self.assertGreater(examples[0]['target']['hd3'],
                               examples[0]['target']['e5'])

    def test_loss_weight_scales_with_search_margin(self):
        # Sharp mined targets (large stable margin => top mass high) train at
        # 3x; flat ones at 2x.
        with tempfile.TemporaryDirectory() as tmp:
            l=Path(tmp)/'losses.jsonl'
            l.write_text('\n'.join(json.dumps(r) for r in [
                dict(game_hash='sharp',split='train',player_holdout=False,
                     history=[],value=0.3,policy={'e2':1.0}),
                # red to move after e2,e8: e3/d2/f2 are all legal, so the
                # uniform target stays flat (top mass 1/3) after legal-filter.
                dict(game_hash='flat',split='train',player_holdout=False,
                     history=['e2','e8'],value=-0.1,
                     policy={m:1/20 for m in
                             ('e3','d2','f2','e4','d3','f3','c2','g2','e1','d1',
                              'f1','c1','g1','ha2','hc2','hg2','ve2','vd2','vf2',
                              'vc2','vg2')}),
            ])+'\n',encoding='utf-8')
            rows=list(train_nn.load_rows([l]))
            ex=train_nn.build_examples(rows)
            self.assertEqual(len(ex),2)
            sharp=[e for e in ex if max(e['target'].values())>=0.99]
            flat=[e for e in ex if max(e['target'].values())<0.99]
            self.assertEqual(len(sharp),1)
            self.assertEqual(len(flat),1)
            self.assertEqual(sharp[0]['weight'],3.0)
            self.assertEqual(flat[0]['weight'],2.0)

    def test_train_stratifies_when_both_pools_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp)/'corpus.jsonl'
            l=Path(tmp)/'losses.jsonl'
            t.write_text('\n'.join(json.dumps(dict(game_hash=f't{i}',split='train',
                player_holdout=False,history=[],value=0.0,policy={'e2':1.0})) for i in range(40))+'\n',encoding='utf-8')
            l.write_text('\n'.join(json.dumps(dict(game_hash=f'l{i}',split='train',
                player_holdout=False,history=['e2'],value=0.0,policy={'e8':1.0})) for i in range(8))+'\n',encoding='utf-8')
            rows=list(train_nn.load_rows([t,l]))
            ex=train_nn.build_examples(rows)
            import nn_model
            m=nn_model.new_model(seed=7)
            m=train_nn.train(m,ex,epochs=2,batch=16,seed=7,verbose=False)
            self.assertTrue(m['metadata'].get('stratified'))
            self.assertEqual(m['metadata'].get('teacher_frac'),0.7)

    def test_train_accepts_loss_only_corpus(self):
        rows=[dict(history=[],policy={'e2':1.0},value=.2,source='loss')]
        examples=train_nn.build_examples(rows)
        import nn_model
        model=train_nn.train(nn_model.new_model(seed=8),examples,epochs=1,
                             batch=4,seed=8,verbose=False)
        self.assertEqual(model['metadata']['examples'],1)



if __name__=='__main__':unittest.main()
