#!/usr/bin/env python3
"""Turn losses into regression cases.

For a finished game, replay it and, at every ply the USER moved, run a much
longer offline search than the live coach gets. Where the played move is
materially worse than the best found (a blunder), record a regression case:
the exact position, the played move, the best move + alternatives, and the
opponent's strongest reply to the best move.

These cases (a) document what the coach should have done and (b) let a test
assert the coach still finds a good move at those critical positions after
future changes.

Usage:
  python regress_losses.py --code 81s8yr
  python regress_losses.py --history "e2,e8,..." --winner blue --side red
"""
import argparse
import json
import sys
import time
from pathlib import Path

import coach as c
import mcts_coach

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'study'
REGRESS = STUDY / 'regressions'

USER_PREFIXES = ('steak',)


def is_user(name):
    return bool(name) and name.casefold().startswith(USER_PREFIXES)


def locate(code):
    for rel in (f'study/archive/{code}.json', f'study/additional/{code}.json', f'study/{code}.json'):
        p = ROOT / rel
        if p.exists():
            return p
    return None


def long_search(hist, side, seconds):
    """Best move via a long MCTS search; fall back to minimax depth 4 if MCTS fails."""
    try:
        r = mcts_coach.search(hist, side, time_limit=seconds)
        if r.get('scored'):
            return r['scored'][0][1], r
    except Exception:
        pass
    r = c.search(hist, side, 4, seconds)
    return r['scored'][0][1], r


def opponent_reply(hist, best_move, seconds):
    """The opponent's strongest reply to the best move."""
    g = c.Game(list(hist) + [best_move])
    if g.winner is not None:
        return None, 'game over'
    r = c.search(g.history, g.to_move, 3, seconds)
    return r['scored'][0][1], r


def build_regressions(history, winner, user_side, seconds, margin=30.0):
    """Record completed minimax disagreements and the separate MCTS proposal.

    Scores are heuristic points. A disagreement is not a proof of a lost game.
    """
    history = c.Game(c.parse_history(history)).history
    user_side = {'red': c.RED, 'blue': c.BLUE}[user_side]
    cases = []
    g = c.Game()
    for i, mv in enumerate(history):
        side = g.to_move
        if side == user_side:
            mcts_proposal, _ = long_search(g.history, side, seconds)
            # Compare only scores from the same completed depth.
            mm = c.search(g.history, side, depth=2, time_limit=min(4, seconds))
            if mm.get('depth',0)<2:
                g.apply(mv)
                continue
            # Include minimax's own candidate even when MCTS misses a wall.
            best_move = mm['scored'][0][1]
            mm_scores = {m: sc for sc, m in mm['scored']}
            played_sc = mm_scores.get(mv)
            best_sc = mm_scores.get(best_move)
            if played_sc is not None and best_sc is not None and played_sc - best_sc > margin:
                reply, _ = opponent_reply(g.history, best_move, seconds)
                cases.append(dict(
                    ply=i + 1, history=g.history[:], position=live_position(g),
                    completed_depth=mm['depth'], score_units='heuristic points', proven=False,
                    mcts_proposal=mcts_proposal,
                    played=mv, best=best_move, delta=round(played_sc - best_sc, 1),
                    alternatives=[m for _, m in mm['scored'][:4]],
                    opponent_reply=reply))
        try:
            g.apply(mv)
        except ValueError:
            break
    return cases


def live_position(g):
    return dict(red=c.LETTERS[g.pawns[c.RED][0]] + str(g.pawns[c.RED][1] + 1),
                blue=c.LETTERS[g.pawns[c.BLUE][0]] + str(g.pawns[c.BLUE][1] + 1),
                walls=sorted(g.walls),
                red_left=g.remaining[c.RED], blue_left=g.remaining[c.BLUE],
                to_move='red' if g.to_move == c.RED else 'blue')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code', help='share code of a game in study/archive (or study)')
    ap.add_argument('--history', help='comma ply list')
    ap.add_argument('--winner', choices=('red', 'blue'), help='who won')
    ap.add_argument('--side', choices=('red', 'blue'), help='which side is the user (default: auto-detect steak accounts)')
    ap.add_argument('--seconds', type=float, default=12.0, help='offline search budget per ply')
    ap.add_argument('--margin', type=float, default=30.0, help='score gap that counts as a blunder')
    args = ap.parse_args()

    if args.code:
        path = locate(args.code)
        if not path:
            print(f'game {args.code} not found locally', file=sys.stderr)
            sys.exit(1)
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        history = data['historyCsv']
        w = data.get('winner')
        winner = {'1': 'red', '2': 'blue', 1: 'red', 2: 'blue'}.get(w if not isinstance(w, bool) else None) or args.winner
        names = [data.get('player1Username'), data.get('player2Username')]
        if args.side:
            user_side = args.side
        elif is_user(names[0]):
            user_side = 'red'
        elif is_user(names[1]):
            user_side = 'blue'
        else:
            # Not a steak game; grade the losing side.
            user_side = 'blue' if winner == 'red' else 'red'
        code = args.code
    elif args.history and args.winner and args.side:
        history = args.history
        winner = args.winner
        user_side = args.side
        code = 'adhoc'
    else:
        ap.error('need --code, or --history + --winner + --side')

    cases = build_regressions(history, winner, user_side, args.seconds, args.margin)
    REGRESS.mkdir(parents=True, exist_ok=True)
    out = dict(code=code, winner=winner, user_side=user_side,
               seconds=args.seconds, margin=args.margin, blunders=cases,
               built=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (REGRESS / f'{code}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'{code}: {len(cases)} blunder(s) at margin>{args.margin}', flush=True)
    for cse in cases:
        print(f"  ply {cse['ply']}: played {cse['played']} -> best {cse['best']} "
              f"(delta {cse['delta']}), opp reply {cse['opponent_reply']}", flush=True)


if __name__ == '__main__':
    main()
