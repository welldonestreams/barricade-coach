import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import tempfile
import io
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
import coach as c
import coach_server
import server
import mcts_coach
import collect_games

class CollectorTests(unittest.TestCase):
    def test_pagination_deduplication_and_resume(self):
        data={code:json.loads((ROOT/f'study/{code}.json').read_text(encoding='utf-8-sig')) for code in ('7bj7bj','81s8yr')}
        def get(path):
            if '/games?page=1' in path:
                return dict(games=[dict(shareCode='7bj7bj',result='win')],pagination=dict(hasMore=True,totalGames=2))
            if '/games?page=2' in path:
                return dict(games=[dict(shareCode='7bj7bj',result='win'),dict(shareCode='81s8yr',result='loss')],pagination=dict(hasMore=False,totalGames=2))
            return data[path.rsplit('/',1)[-1]]
        with tempfile.TemporaryDirectory() as tmp, patch.object(collect_games,'ROOT',Path(tmp)), patch.object(collect_games.time,'sleep'), patch('builtins.print'):
            with patch.object(collect_games,'get',side_effect=get): collect_games.collect(['Test'])
            summary=json.loads((Path(tmp)/'collection-summary.json').read_text())
            self.assertEqual(summary['unique_games'],2); self.assertEqual(summary['validated'],2)
            self.assertTrue(summary['complete'])
            with patch.object(collect_games,'get') as network: collect_games.collect(['Test'])
            network.assert_not_called()

    def test_rate_limit_backoff(self):
        error=HTTPError('https://api.barricade.gg/test',429,'rate limited',{'Retry-After':'120'},io.BytesIO())
        with patch.object(collect_games,'urlopen',side_effect=[error,io.BytesIO(b'{"ok":true}')]), patch.object(collect_games.time,'sleep') as sleep, patch('builtins.print'):
            self.assertEqual(collect_games.get('/test'),{'ok':True})
        sleep.assert_called_once_with(120)
        error.close()

ROOT = Path(__file__).resolve().parent
RACE = 'e2,e8,e3,e7,e4,e6,e5,e4,e6,e3,e7,e2,e8'.split(',')

class RulesTests(unittest.TestCase):
    def test_notation_and_atomic_rejection(self):
        g=c.Game(' E2 , E8 ')
        self.assertEqual(g.history,['e2','e8'])
        for move in ('','e0','e10','hi4','va9','x4','e7','ve0','hv1'):
            before=(g.history.copy(),g.pawns.copy(),g.walls.copy(),g.remaining.copy())
            with self.subTest(move=move), self.assertRaises(ValueError): g.apply(move)
            self.assertEqual(before,(g.history,g.pawns,g.walls,g.remaining))
        with self.assertRaises(ValueError): c.Game('e2,,e8')

    def test_h_file_pawn_not_wall(self):
        g=c.Game('f1,e8,g1,e7,h1')
        self.assertEqual(g.pawns[c.RED],(7,0))
        self.assertEqual(g.remaining[c.RED],10)

    def test_wall_inventory(self):
        data=json.loads((ROOT/'study/7bj7bj.json').read_text(encoding='utf-8-sig'))
        g=c.Game(data['historyCsv'].split(',')[:38])
        self.assertEqual(g.remaining[c.RED],0)
        self.assertEqual(g.legal_walls(c.RED),[])
        w=next(w for w in g.legal_walls(c.BLUE))
        with self.assertRaises(ValueError): g.apply(w)

    def test_cross_overlap_touch(self):
        g=c.Game(['hd4'])
        for w in ('vd4','hc4','he4','hd4','vi1','zz1'):
            self.assertFalse(g.is_wall_legal(w),w)
        self.assertTrue(g.is_wall_legal('hf4'))
        self.assertTrue(c.Game(['vd4']).is_wall_legal('vd6'))

    def test_must_leave_both_paths(self):
        walls={'ha1','hc1','he1','hg1'}
        self.assertTrue(c.path_exists(walls,c.STARTS[c.RED],8))
        self.assertFalse(c.legal_wall(walls,'vh1',tuple(c.STARTS.values())))

    def test_jump_and_blocked_diagonals(self):
        g=c.Game('e2,e8,e3,e7,e4,e6,e5')
        moves={p for _,p in g.pawn_moves(c.BLUE)}
        self.assertIn((4,3),moves)
        self.assertNotIn((3,4),moves)
        g.walls.add('hd4')
        moves={p for _,p in g.pawn_moves(c.BLUE)}
        self.assertNotIn((4,3),moves)
        self.assertTrue({(3,4),(5,4)} <= moves)

    def test_edge_jump(self):
        g=c.Game(); g.pawns={c.RED:(8,4),c.BLUE:(7,4)}
        moves={p for _,p in g.pawn_moves(c.BLUE)}
        self.assertTrue({(8,3),(8,5)} <= moves)

    def test_game_over(self):
        g=c.Game(RACE+['d2','e9'])
        self.assertEqual(g.winner,c.RED)
        self.assertEqual(g.moves(c.BLUE),[])
        self.assertEqual(g.eval_side(c.RED),-c.WIN)
        with self.assertRaises(ValueError): g.apply('d1')
        self.assertEqual(c.search(g.history)['scored'],[])
        self.assertIn('wins immediately',c.explain(RACE+['d2'],c.RED,'e9'))

    def test_all_shared_games(self):
        total=0
        for path in ROOT.glob('study/??????.json'):
            data=json.loads(path.read_text(encoding='utf-8-sig'))
            g=c.Game(data['historyCsv'])
            total+=len(g.history)
            for s in (c.RED,c.BLUE): self.assertEqual(g.remaining[s],data[f'p{s+1}RemainingBarricades'])
        self.assertEqual(total,418)

class SearchTests(unittest.TestCase):
    def test_opening(self):
        r=c.search([],depth=2,time_limit=15)
        self.assertEqual(r['depth'],2)
        self.assertEqual(r['scored'][0][1],'e2')
        self.assertEqual(len(r['scored']),131)

    def test_immediate_win_priority(self):
        r=c.search(RACE,c.BLUE,depth=2,time_limit=10)
        self.assertEqual(r['scored'][0],(-c.WIN+1,'e1'))

    def test_block_immediate_loss(self):
        g=c.Game(); g.pawns={c.RED:(4,3),c.BLUE:(4,1)}
        with patch.object(c,'Game',return_value=g): r=c.search([],c.RED,depth=2,time_limit=15)
        chosen=g.copy(); chosen.apply(r['scored'][0][1])
        self.assertFalse(any(p[1]==0 for _,p in chosen.pawn_moves(c.BLUE)))

    def test_depth_three_against_unpruned_oracle(self):
        g=c.Game(); g.pawns={c.RED:(3,6),c.BLUE:(5,2)}; g.remaining={c.RED:0,c.BLUE:0}
        def oracle(state,left,ply):
            if state.winner is not None: return -c.WIN+ply if state.winner==c.RED else c.WIN-ply
            if not left: return state.eval_side(c.RED)
            scores=[]
            for mv in state.moves(state.to_move):
                child=state.copy(); child.apply(mv); scores.append(oracle(child,left-1,ply+1))
            return (min if state.to_move==c.RED else max)(scores)
        with patch.object(c,'Game',return_value=g): r=c.search([],c.RED,depth=3,time_limit=5)
        self.assertEqual(r['depth'],3)
        for score,mv in r['scored']:
            child=g.copy(); child.apply(mv)
            self.assertEqual(score,oracle(child,2,1),mv)

    def test_timeout_and_wrong_side(self):
        r=c.search([],time_limit=0.000001)
        self.assertTrue(r['timed_out']); self.assertEqual(r['depth'],0)
        c.Game().apply(r['scored'][0][1])
        with self.assertRaises(ValueError): c.search([],c.BLUE)
        for seconds in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): c.search([],time_limit=seconds)
        for depth in (0,13,2.5):
            with self.assertRaises(ValueError): c.search([],depth=depth)

    def test_cli_defaults_and_bad_history(self):
        r=subprocess.run([sys.executable,str(ROOT/'coach.py'),'--depth','1','--json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0); self.assertEqual(json.loads(r.stdout)['scored'][0][1],'e2')
        r=subprocess.run([sys.executable,str(ROOT/'coach.py'),'e9'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0); self.assertNotIn('Traceback',r.stderr)

    def test_daemon_error_does_not_poison_next_position(self):
        self.assertIn('error',coach_server.handle('e9|red'))
        self.assertEqual(coach_server.handle('|red')['move'],'e2')

    def test_tied_grades(self):
        result=dict(scored=[(0,'d1'),(0,'e2'),(100,'f1')],depth=2,timed_out=False)
        with patch.object(c,'search',return_value=result): rows=c.grade_game(['e2'],c.RED)
        self.assertEqual(rows[0]['rank'],1)
        with self.assertRaises(ValueError): c.grade_game(['e9'],c.RED)

    def test_unsearched_fallback_is_not_a_grade(self):
        result=dict(scored=[(50,'e2')],depth=0,timed_out=True)
        with patch.object(c,'search',return_value=result): row=c.grade_game(['e2'],c.RED)[0]
        self.assertIsNone(row['rank']); self.assertIsNone(row['delta']); self.assertIsNone(row['best'])

class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd=server.LocalServer(('127.0.0.1',0),server.Handler)
        cls.thread=threading.Thread(target=cls.httpd.serve_forever,daemon=True); cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.httpd.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown(); cls.httpd.server_close(); cls.thread.join()
    def fetch(self,path,headers=None):
        try:
            with urlopen(Request(self.base+path,headers=headers or {})) as r: return r.status,json.load(r)
        except HTTPError as e:
            with e: return e.code,json.load(e)
    def test_state_and_illegal_history(self):
        status,d=self.fetch('/api/state?h=e2,e8')
        self.assertEqual(status,200); self.assertEqual(d['remaining'],{'red':10,'blue':10})
        self.assertEqual(d['pawns'],{'red':'e2','blue':'e8'}); self.assertIsNone(d['search'])
        for path in ('/api/move?h=e9','/api/move?side=banana','/api/move?side=blue','/api/move?seconds=nan','/api/move?engine=bad'):
            self.assertEqual(self.fetch(path)[0],400,path)
    def test_local_origin_and_busy(self):
        self.assertEqual(self.fetch('/api/state',{'Origin':'https://example.com'})[0],403)
        self.assertEqual(self.fetch('/api/state',{'Host':'evil.example'})[0],403)
        server.SEARCH_LOCK.acquire()
        try: self.assertEqual(self.fetch('/api/move')[0],429)
        finally: server.SEARCH_LOCK.release()
        self.assertEqual(self.fetch('/api/move?depth=1')[0],200)
    def test_terminal_state(self):
        status,d=self.fetch('/api/move?h='+','.join(RACE+['d2','e9']))
        self.assertEqual(status,200); self.assertEqual(d['winner'],'red'); self.assertEqual(d['top'],[])

    def test_selected_engine_is_dispatched(self):
        result=dict(scored=[(-9,'e2')],engine='mcts',depth=None,elapsed=0.01,simulations=128)
        with patch.object(mcts_coach,'search',return_value=result) as engine:
            status,d=self.fetch('/api/move?engine=mcts&seconds=1')
        self.assertEqual(status,200); self.assertEqual(d['search']['engine'],'mcts')
        engine.assert_called_once_with([],c.RED,1.0)
        status,d=self.fetch('/api/move?engine=python&depth=1')
        self.assertEqual(d['search']['engine'],'python')

    def test_health_and_exclusive_port(self):
        self.assertEqual(self.fetch('/api/health')[1]['protocol'],2)
        with self.assertRaises(OSError):
            other=server.LocalServer(self.httpd.server_address,server.Handler)
            other.server_close()

@unittest.skipUnless(shutil.which('node'),'Node.js optional MCTS dependency')
class MctsTests(unittest.TestCase):
    def test_independent_rules_agree_all_shared_positions(self):
        histories=[]
        for path in ROOT.glob('study/??????.json'):
            h=json.loads(path.read_text(encoding='utf-8-sig'))['historyCsv'].split(',')
            histories.extend(h[:i] for i in range(len(h)+1))
        r=subprocess.run([shutil.which('node'),str(ROOT/'mcts_adapter.cjs')],input=json.dumps(dict(inspect=True,histories=histories)),text=True,capture_output=True,check=True)
        data=json.loads(r.stdout)
        for h,d in zip(histories,data,strict=True):
            g=c.Game(h)
            self.assertEqual(d['pawns'],[list(g.pawns[s]) for s in (c.RED,c.BLUE)],h)
            self.assertEqual(d['remaining'],list(g.remaining.values()),h)
            self.assertEqual(d['winner'],g.winner,h)
            self.assertEqual(d['legal'],sorted(g.moves(g.to_move)),h)
    def test_mcts_seed_legal_and_terminal(self):
        h='e2,e8,e3,e7,e4,e6'
        # workers=1 pins the single-tree path this test is about: a fixed seed
        # must reproduce the exact rollout count and move ranking.
        a=mcts_coach.search(h,rollouts=256,time_limit=5,seed=10,workers=1)
        b=mcts_coach.search(h,rollouts=256,time_limit=5,seed=10,workers=1)
        self.assertEqual(a['scored'],b['scored']); self.assertEqual(a['simulations'],256)
        for _,mv in a['scored']: c.Game(h).apply(mv)
        self.assertEqual(mcts_coach.search(RACE)['scored'][0][1],'e1')
        self.assertEqual(mcts_coach.search(RACE+['d2','e9'])['scored'],[])
    def test_mcts_tiny_budget_safe_fallback(self):
        r=mcts_coach.search([],time_limit=0.001,seed=1)
        self.assertTrue(r['fallback']); c.Game().apply(r['scored'][0][1])
    def test_mcts_missing_node_and_invalid_budget(self):
        with patch.object(shutil,'which',return_value=None), self.assertRaises(ValueError): mcts_coach.search([])
        with self.assertRaises(ValueError): mcts_coach.search([],rollouts=0)
        with self.assertRaises(ValueError): mcts_coach.search([],time_limit=float('nan'))

if __name__=='__main__': unittest.main()
