import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import advice
import coach as c
import endgame
import learning
import live_coach
import mcts_coach


class RankingTests(unittest.TestCase):
    def test_unvalidated_swings_cannot_reverse_700_to_300_visits(self):
        result=dict(scored=[(-700,'e2'),(-300,'d1')],simulations=1000,depth=None,
                    principal_variation=['e2'],elapsed=0,timed_out=False)
        with patch.object(mcts_coach,'search',return_value=result), \
             patch.object(learning,'learned_prior',side_effect=AssertionError('ungated prior')), \
             patch.object(learning,'learned_reasons',side_effect=AssertionError('self-reinforcing heuristic')):
            actual=advice.advise([],engine='mcts',seconds=1)
        self.assertEqual(actual['scored'],[(-700,'e2'),(-300,'d1')])

    def test_opponent_evidence_can_break_an_exact_tie(self):
        result=dict(scored=[(0,'d1'),(0,'e2'),(100,'f1')],depth=2,
                    principal_variation=['d1','e8'],elapsed=0,timed_out=False)
        def insight(name,hist,color):
            return dict(known=True,top=[('e8',10)])
        with patch.object(c,'search',return_value=result),patch.object(advice,'approved_prior',return_value={}), \
             patch.object(learning,'opponent_insight',side_effect=insight):
            actual=advice.advise([],opp_name='Test')
        self.assertEqual(actual['scored'][0],(0,'e2'))
        self.assertEqual(actual['scored'][-1],(100,'f1'))
        self.assertEqual(actual['principal_variation'],['e2'])

    def test_champion_requires_gate_and_correct_engine(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(learning,'MEMORY',Path(tmp)):
            (Path(tmp)/'champion.json').write_text(json.dumps({'report':{'promoted':True},'positions':{}}))
            self.assertEqual(advice.approved_prior(c.Game(),'python'),{})
            self.assertEqual(advice.approved_prior(c.Game(),'mcts'),{})

    def test_explanation_does_not_query_training_database(self):
        result=dict(scored=[(-100,'e2')],engine='mcts',simulations=1000,timed_out=True)
        with patch.object(learning,'explain_why',side_effect=AssertionError('blocking/circular lookup')):
            explanation=live_coach.explain(c.Game(),result)
        self.assertIn('estimates',explanation)
        self.assertIn('Time limit reached',explanation)

    def test_draw_evaluations_are_not_losses(self):
        with patch.object(learning,'_db',side_effect=AssertionError('draw must not write')):
            self.assertFalse(learning.record_evals('e2,e8',None))


class DeadlineTests(unittest.TestCase):
    def test_partial_batches_survive_timeout_and_timeout_is_reported(self):
        timeouts=[]
        batch=json.dumps(dict(simulations=128,candidates=[dict(move='e2',visits=120,rollout_win_rate=.5)]))+'\n'
        def expired(args,**kwargs):
            timeouts.append(kwargs['timeout'])
            raise subprocess.TimeoutExpired(args,kwargs['timeout'],output=batch.encode())
        with patch.object(mcts_coach.subprocess,'run',side_effect=expired),patch.object(mcts_coach.shutil,'which',return_value='node'):
            r=mcts_coach.search([],time_limit=1,workers=4)
        self.assertEqual(len(timeouts),4)
        self.assertTrue(all(0<t<=1 for t in timeouts))
        self.assertTrue(r['timed_out'])
        self.assertEqual(r['simulations'],512)
        self.assertEqual(r['scored'],[(-480,'e2')])


class EndgameTests(unittest.TestCase):
    def game(self,red,blue):
        g=c.Game();g.remaining={0:0,1:0};g.pawns={0:red,1:blue};return g

    def test_finishing_jump_and_turn_are_exact(self):
        g=self.game((4,6),(4,7))
        result=endgame.solve(g,2)
        self.assertTrue(result['exact'])
        self.assertEqual(result['scored'][0],(-c.WIN+1,'e9'))
        self.assertEqual(result['goal_plies'],1)
        self.assertEqual(result['principal_variation'],['e9'])

    def test_forced_loss_and_all_children_against_two_ply_oracle(self):
        g=self.game((4,3),(4,1));r=endgame.solve(g,2)
        self.assertEqual(r['outcome'],'opponent forced goal')
        self.assertEqual(r['goal_plies'],2)
        for score,mv in r['scored']:
            child=g.copy();child.apply(mv)
            self.assertTrue(any(dest[1]==0 for _,dest in child.pawn_moves(c.BLUE)))
            self.assertEqual(score,c.WIN-2)

    def test_wall_threat_or_timeout_cannot_be_called_exact(self):
        self.assertIsNone(endgame.solve(c.Game()))
        g=self.game((4,3),(4,4));g.remaining[1]=1
        self.assertIsNone(endgame.solve(g))
        g.remaining[1]=0
        self.assertIsNone(endgame.solve(g,0))

    def test_blocked_jump_uses_legal_diagonal(self):
        g=self.game((4,6),(4,7));g.walls={'he8'}
        r=endgame.solve(g,2)
        self.assertNotIn('e9',[m for _,m in r['scored']])
        self.assertIn('d8',[m for _,m in r['scored']])
        self.assertEqual({m for _,m in r['scored']},set(g.moves(g.to_move)))


class ReportedLossTests(unittest.TestCase):
    def test_move_twenty_preserves_tempo_before_he4(self):
        path=Path(__file__).resolve().parent/'study/additional/1ttbdt.json'
        hist=json.loads(path.read_text())['historyCsv'].split(',')
        g=c.Game(hist[:19])
        result=c.search(g.history,side=c.BLUE,depth=2,time_limit=10)
        self.assertEqual(result['depth'],2)
        self.assertEqual(result['scored'][0][1],'e5')
        played=g.copy()
        for mv in ['va2','d4','e5','he4']:played.apply(mv)
        alternative=g.copy()
        for mv in ['e5','d4','e4']:alternative.apply(mv)
        self.assertEqual(played.race_distance(c.BLUE),16)
        self.assertEqual(alternative.race_distance(c.BLUE),10)

    def test_final_position_has_no_saving_move_and_needs_no_rollouts(self):
        path=Path(__file__).resolve().parent/'study/additional/1ttbdt.json'
        hist=json.loads(path.read_text())['historyCsv'].split(',')[:53]
        with patch.object(mcts_coach.subprocess,'run',side_effect=AssertionError('already proven')):
            result=mcts_coach.search(hist)
        self.assertTrue(result['forced_loss'])
        self.assertEqual(result['simulations'],0)
        self.assertIn('no saving move',live_coach.explain(c.Game(hist),result))

    def test_review_total_deadline_returns_partial_without_false_grades(self):
        with patch.object(c,'search',side_effect=AssertionError('deadline exhausted')):
            rows=c.grade_game(['e2','e8'],c.RED,total_time=0)
        self.assertEqual(rows,[])


if __name__=='__main__':unittest.main()
