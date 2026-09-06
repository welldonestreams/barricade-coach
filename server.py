#!/usr/bin/env python3
"""barricade-coach local web server.
Usage: python3 server.py  ->  http://127.0.0.1:8807
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coach as c

ROOT = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(ROOT, 'ui', 'index.html')

def parse_move(h):
    return [m for m in (h or '').split(',') if m]

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/api/move':
            q = parse_qs(u.query)
            hist = parse_move(q.get('h', [''])[0])
            side_s = q.get('side', [''])[0].lower()
            side = c.RED if side_s.startswith('r') else (c.BLUE if side_s.startswith('b') else None)
            try:
                g = c.Game(hist)
                to_move = g.to_move
                if side is None:
                    side = to_move
                sp = {
                    'red': c.shortest(g.walls, g.pawns[c.RED], c.GOALS[c.RED]),
                    'blue': c.shortest(g.walls, g.pawns[c.BLUE], c.GOALS[c.BLUE]),
                }
                scored = c.best_moves(hist, side, n=0)[1]
                top = [[s, m] for s, m in scored[:5]]
                self._send(200, {
                    'to_move': 'red' if to_move == c.RED else 'blue',
                    'requested_side': 'red' if side == c.RED else 'blue',
                    'top': top,
                    'sp': sp,
                    'walls': sorted(g.walls),
                    'pawns': {'red': c.LETTERS[g.pawns[c.RED][0]] + str(g.pawns[c.RED][1] + 1),
                              'blue': c.LETTERS[g.pawns[c.BLUE][0]] + str(g.pawns[c.BLUE][1] + 1)},
                    'legal_history': True,
                })
            except Exception as e:
                self._send(400, {'error': str(e), 'legal_history': False})
            return
        if u.path == '/' or u.path == '/index.html':
            with open(UI, 'rb') as f:
                self._send(200, f.read(), 'text/html')
            return
        self._send(404, {'error': 'not found'})

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8807
    print(f'barricade-coach on http://127.0.0.1:{port}')
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
