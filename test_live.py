import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import coach as c
import advice
import learning
import live_coach
import training

ROOT=Path(__file__).resolve().parent


class LiveTests(unittest.TestCase):
    def test_screenshot_position_rejects_d2(self):
        g=c.Game('e2,e8,e3,e7,e4,e6,e5'.split(','))
        self.assertEqual(g.to_move,c.BLUE)
        self.assertNotIn('d2',g.moves(c.BLUE))
        self.assertIn('e4',g.moves(c.BLUE))

    def test_all_46_positions_match_history_and_legal_search(self):
        data=json.loads((ROOT/'study/additional/5s2skz.json').read_text(encoding='utf-8'))
        g=c.Game()
        for mv in c.parse_history(data['historyCsv']):
            p=live_coach.position(g)
            result=c.search_position(p['red'],p['blue'],p['walls'],g.to_move,
                                     depth=1,time_limit=.15,red_left=p['red_left'],blue_left=p['blue_left'])
            self.assertTrue(all(m in g.moves(g.to_move) for _,m in result['scored']))
            g.apply(mv)
        self.assertEqual(len(g.history),46)
        self.assertEqual(g.remaining[c.BLUE],0)

    def test_bad_position_never_reaches_search(self):
        invalid=[('e0','e9',[],0,10,10),('e1','e1',[],0,10,10),
                 ('e1','e9',[],0,-1,10),('e1','e9',['he4'],0,10,10),
                 ('e1','e9',['he4','ve4'],0,9,9),('e1','e9',[],8,10,10)]
        for args in invalid:
            with self.assertRaises(ValueError): c.Game.from_position(*args)

    def test_live_refuses_mismatched_board_and_reserves(self):
        p=live_coach.position(c.Game())
        params={**p,'walls':'','h':''}
        with patch.object(advice,'advise') as search:
            for field,value in [('red','d2'),('red_left',9),('walls','he4')]:
                with self.assertRaises(ValueError): live_coach.query({**params,field:value})
            search.assert_not_called()

    def test_live_response_tied_to_request_and_explains(self):
        g=c.Game('e2,e8,e3,e7,e4,e6,e5'.split(','));p=live_coach.position(g)
        d=live_coach.query({**p,'walls':'','h':','.join(g.history),'seconds':'.3','request_id':'abc'})
        self.assertEqual(d['request_id'],'abc')
        self.assertEqual(d['to_move'],'blue')
        self.assertNotIn('d2',[m for _,m in d['top']])
        self.assertIn('route',d['why'])

    def test_opponent_model_validates_whole_game_and_matches_position(self):
        good=dict(player1Username='Test',player2Username='Other',winner='1',historyCsv='e2,e8,e3')
        bad={**good,'historyCsv':'e2,e8,d9'}
        model=learning.build_opponent_model('Test',[good,bad])
        self.assertEqual(model['games'],1)
        with patch.object(learning,'load_opponent',return_value=model):
            ins=learning.opponent_insight('Test',[],'red')
        self.assertEqual(ins['top'],[('e2',1.0)])

    def test_legacy_selfplay_does_not_influence_advice(self):
        with patch.object(learning,'outcome_stats',side_effect=AssertionError('legacy read')),patch.object(learning,'eval_delta_stats',side_effect=AssertionError('circular read')):
            result=advice.advise([],seconds=.1)
        self.assertIn(result['scored'][0][1],c.Game().moves(c.RED))

    def test_draws_not_recorded_as_losses(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(learning,'DB',Path(tmp)/'test.db'):
            learning.record_game('e2,e8',None)
            self.assertEqual(learning.outcome_stats([]),{})

    def test_holdout_and_promotion(self):
        hist=['e2','e8']
        self.assertEqual(training.split_for(hist),training.split_for(list(hist)))
        self.assertFalse(training.promotion_gate([2]*20)[0])
        self.assertFalse(training.promotion_gate([1]*100)[0])
        self.assertTrue(training.promotion_gate([2]*100)[0])
        matches=[dict(winner=None,rows=[('x','e2',0)]),dict(winner=0,rows=[('held','e2',0),('train','e2',0)])]
        self.assertEqual(training.aggregate(matches,{'held'}),{'train':{'e2':[1,1]}})

    def test_overlay_regressions(self):
        subprocess.run(['node','overlay/test-overlay.cjs'],cwd=ROOT,check=True,capture_output=True)


if __name__=='__main__':unittest.main()
