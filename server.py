#!/usr/bin/env python3
"""barricade-coach local web server.
Usage: python3 server.py  ->  http://127.0.0.1:8810
"""
import json
import os
import sys
import threading
import socket
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coach as c
try:
    import learning
    LEARNING_OK = True
except Exception:
    LEARNING_OK = False

ROOT = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(ROOT, 'ui', 'index.html')
SEARCH_LOCK = threading.BoundedSemaphore(1)

class LocalServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can let two processes serve the same loopback port.
    allow_reuse_address = os.name != 'nt'
    def server_bind(self):
        if os.name == 'nt':
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

@lru_cache(maxsize=1)
def read_book(modified):
    with open(os.path.join(ROOT,'study','opening-book.json'),encoding='utf-8') as f:
        return json.load(f)

def opening_reference(history):
    path=os.path.join(ROOT,'study','opening-book.json')
    try:
        book=read_book(os.stat(path).st_mtime_ns)
        return dict(source_games=book['source_games'],moves=book['positions'].get(','.join(history),[]))
    except (OSError,ValueError,KeyError,TypeError):
        return dict(source_games=0,moves=[])

def parse_move(h):
    if len(h) > 8000:
        raise ValueError('History too long (maximum 8000 characters)')
    return c.parse_history(h)

def json_or(text):
    try:
        return json.loads(text)
    except ValueError:
        return None

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in allowed:
            self._send(403, {'error': 'Local access only'})
            return
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://{h}' for h in allowed}:
            self._send(403, {'error': 'Same-origin access only'})
            return
        u = urlparse(self.path)
        if u.path == '/api/health':
            self._send(200, {'service': 'barricade-coach', 'protocol': 2})
            return
        if u.path == '/api/legal':
            q = parse_qs(u.query)
            try:
                hist = parse_move(q.get('h', [''])[0])
                g = c.Game(hist)
                walls_left = {'red': g.remaining[c.RED], 'blue': g.remaining[c.BLUE]}
                if g.winner is not None:
                    self._send(200, {'to_move': 'red' if g.to_move == c.RED else 'blue',
                                     'legal_pawn': [], 'legal_walls': [], 'walls_left': walls_left,
                                     'winner': ('red' if g.winner == c.RED else 'blue')})
                    return
                pawn_dests = []
                for _, p in g.pawn_moves(g.to_move):
                    pawn_dests.append(c.LETTERS[p[0]] + str(p[1] + 1))
                self._send(200, {
                    'to_move': 'red' if g.to_move == c.RED else 'blue',
                    'pawn_origin': c.LETTERS[g.pawns[g.to_move][0]] + str(g.pawns[g.to_move][1] + 1),
                    'legal_pawn': sorted(set(pawn_dests)),
                    'legal_walls': g.legal_walls(),
                    'walls_left': walls_left,
                    'winner': None,
                })
            except (ValueError, TypeError) as e:
                self._send(400, {'error': str(e), 'legal_history': False})
            return
        if u.path in ('/api/move', '/api/state'):
            q = parse_qs(u.query)
            try:
                hist = parse_move(q.get('h', [''])[0])
                side_s = q.get('side', [''])[0].lower()
                if side_s not in ('', 'red', 'blue'):
                    raise ValueError('Side must be red or blue')
                side = {'red': c.RED, 'blue': c.BLUE}.get(side_s)
                g = c.Game(hist)
                to_move = g.to_move
                if side is None:
                    side = to_move
                sp = {
                    'red': c.shortest(g.walls, g.pawns[c.RED], c.GOALS[c.RED]),
                    'blue': c.shortest(g.walls, g.pawns[c.BLUE], c.GOALS[c.BLUE]),
                }
                result = None
                if side != to_move:
                    raise ValueError('Requested side is not on move')
                if u.path == '/api/move':
                    depth = int(q.get('depth', ['2'])[0])
                    seconds = float(q.get('seconds', ['5'])[0])
                    if not 1 <= depth <= 6 or not 0 < seconds <= 15:
                        raise ValueError('Use depth 1-6 and seconds greater than 0, at most 15')
                    if not SEARCH_LOCK.acquire(blocking=False):
                        self._send(429, {'error': 'Coach is busy; try again shortly'})
                        return
                    try:
                        engine = q.get('engine', ['python'])[0]
                        if engine == 'mcts':
                            import mcts_coach
                            result = mcts_coach.search(hist, side, seconds)
                        elif engine == 'python':
                            result = c.search(hist, side, depth, seconds)
                            result['engine'] = 'python'
                        else:
                            raise ValueError('Engine must be python or mcts')
                    finally:
                        SEARCH_LOCK.release()
                scored = result['scored'] if result else []
                top = [[s, m] for s, m in scored[:5]]
                self._send(200, {
                    'to_move': 'red' if to_move == c.RED else 'blue',
                    'requested_side': 'red' if side == c.RED else 'blue',
                    'top': top,
                    'history': g.history,
                    'opening_reference': opening_reference(g.history),
                    'winner': None if g.winner is None else ('red' if g.winner == c.RED else 'blue'),
                    'remaining': {'red': g.remaining[c.RED], 'blue': g.remaining[c.BLUE]},
                    'search': None if result is None else {k: v for k, v in result.items() if k not in ('scored', 'winner')},
                    'sp': sp,
                    'walls': sorted(g.walls),
                    'pawns': {'red': c.LETTERS[g.pawns[c.RED][0]] + str(g.pawns[c.RED][1] + 1),
                              'blue': c.LETTERS[g.pawns[c.BLUE][0]] + str(g.pawns[c.BLUE][1] + 1)},
                    'legal_history': True,
                })
            except (ValueError, TypeError) as e:
                self._send(400, {'error': str(e), 'legal_history': False})
            return
        if u.path == '/api/opponent':
            q = parse_qs(u.query)
            if not LEARNING_OK:
                self._send(501, {'error': 'learning module unavailable'})
                return
            name = q.get('name', [''])[0]
            if not name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in name):
                self._send(400, {'error': 'Provide ?name=<username>'})
                return
            try:
                hist = parse_move(q.get('h', [''])[0])
                model = learning.load_opponent(name)
                if not model:
                    self._send(404, {'error': f'No model for {name} yet - collect their games first', 'name': name})
                    return
                insight = learning.opponent_insight(name, hist)
                stats = learning.outcome_stats(hist)
                self._send(200, {'name': name, 'model': {
                    'games': model.get('games'), 'wins': model.get('wins'),
                    'rating_low': model.get('rating_low'), 'rating_high': model.get('rating_high'),
                    'avg_move_ms': model.get('avg_move_ms'),
                }, 'insight': insight, 'outcome_stats': stats})
            except Exception as e:
                self._send(400, {'error': str(e)})
            return
        if u.path == '/api/memory':
            q = parse_qs(u.query)
            if not LEARNING_OK:
                self._send(501, {'error': 'learning module unavailable'})
                return
            try:
                hist = parse_move(q.get('h', [''])[0])
                self._send(200, {'positions': len(learning.load_outcomes().get('positions', {})),
                                 'games': learning.load_outcomes().get('games', 0),
                                 'stats': learning.outcome_stats(hist)})
            except Exception as e:
                self._send(400, {'error': str(e)})
            return
        if u.path == '/api/record':
            q = parse_qs(u.query)
            if not LEARNING_OK:
                self._send(501, {'error': 'learning module unavailable'})
                return
            try:
                h = q.get('h', [''])[0]
                winner = q.get('winner', [''])[0] or None
                red = q.get('red', [''])[0] or None
                blue = q.get('blue', [''])[0] or None
                opp = q.get('opponent', [''])[0] or None
                sharecode = q.get('share', [''])[0] or None
                # map opponent name to red/blue side based on parity of plies? caller
                # supplies red/blue directly; opponent is whichever side is not 'me'
                saved = learning.record_game(h, winner, red_name=red, blue_name=blue,
                                             sharecode=sharecode)
                self._send(200, {'recorded': saved, 'winner': winner, 'opponent': opp})
            except Exception as e:
                self._send(400, {'error': str(e)})
            return
        if u.path == '/' or u.path == '/index.html':
            with open(UI, 'rb') as f:
                self._send(200, f.read(), 'text/html')
            return
        self._send(404, {'error': 'not found'})

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8810
    try:
        httpd = LocalServer(('127.0.0.1', port), Handler)
    except OSError as exc:
        sys.exit(f'Cannot open port {port}: {exc}. Stop the older coach or run python server.py with another port, e.g. 8810.')
    print(f'barricade-coach on http://127.0.0.1:{port}', flush=True)
    httpd.serve_forever()
