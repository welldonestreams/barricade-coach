#!/usr/bin/env python3
"""Mine defensive-wall training cases from the archive.

Extends study/training-loss-cases.json with new {code, ply, history, best,
played, delta} rows. Replays candidate games and, at plies where the eventual
LOSER is to move and still holds walls, runs a completed depth-2 minimax.
Where the best move is a WALL and the played move was >margin worse, the
position is recorded as a defensive-wall training case (16x-weighted in the
NN loader). The 300-case corpus was mined this way from 95 games (see the repo
history / logs/mine-cases.log); this script reproduces the method at scale.

Guards:
- repair-gate games (81s8yr/52s6kd/zjj0bn/tzr1w5) are never scanned, and any
  position whose history matches a held-out gate history is skipped, so the
  promotion gate stays disjoint from training data.
- Already-covered (code, ply) rows and exact duplicate histories are skipped.
- Idempotent / resumable: safe to re-run; appends only new rows.

Usage:
  python mine_walls.py --games 900 --max-new 1500 --workers 8   # wall-best cases
  python mine_walls.py --kind pawn --max-new 1200 --margin 50 \
      --out study/pawn-loss-cases.json \
      --skip-source study/training-loss-cases.json              # pawn-best cases
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import coach as c

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'study' / 'training-loss-cases.json'
NL = '\r\n'  # this repo's study files are CRLF on the PC working copy


def load_state():
    existing = []
    if OUT.exists():
        existing = json.loads(OUT.read_text(encoding='utf-8-sig'))
    keys = {(x['code'], x['ply']) for x in existing}
    hists = {tuple(x['history']) for x in existing}
    return existing, keys, hists


def gate_codes_and_histories():
    try:
        import repair_gate
        rows = repair_gate.rows()
        return {r['code'] for r in rows}, {tuple(r['history']) for r in rows}
    except (ImportError, ValueError, FileNotFoundError):
        return set(), set()


def mine_game(task):
    path, skip_codes, skip_hists, margin, min_ply, search_s, max_rows, kind = task
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        hist = c.Game(c.parse_history(data.get('historyCsv', ''))).history
    except Exception:
        return [], path.stem
    stem = path.stem
    if stem in skip_codes or len(hist) < min_ply + 3:
        return [], stem
    if sum(1 for m in hist if len(m) == 3) < 4:
        return [], stem
    winner = {'1': c.RED, '2': c.BLUE, 1: c.RED, 2: c.BLUE}.get(data.get('winner'))
    if winner is None:
        return [], stem
    red_walls = sum(1 for i, m in enumerate(hist) if len(m) == 3 and i % 2 == 0)
    blue_walls = sum(1 for i, m in enumerate(hist) if len(m) == 3 and i % 2 == 1)
    loser = 1 - winner
    loser_left = (10 - red_walls) if loser == c.RED else (10 - blue_walls)
    if loser_left < 1:
        return [], stem
    rows = []
    g = c.Game()
    for i, mv in enumerate(hist):
        side = g.to_move
        if (i >= min_ply and side == loser and g.remaining[side] >= 1
                and len(g.walls) >= 3):
            try:
                r = c.search(g.history, side, 2, search_s)
            except c.SearchTimeout:
                pass
            else:
                if r.get('depth', 0) >= 2 and r.get('scored'):
                    best = r['scored'][0][1]
                    best_ok = (len(best) == 3 if kind == 'wall'
                               else len(best) == 2 if kind == 'pawn' else True)
                    if best_ok:
                        sc = {m: s for s, m in r['scored']}
                        ps = sc.get(mv)
                        bs = sc.get(best)
                        if ps is not None and bs is not None and ps - bs > margin:
                            key = tuple(g.history)
                            if key not in skip_hists:
                                skip_hists.add(key)
                                rows.append(dict(code=stem, ply=i + 1,
                                                 history=g.history[:], best=best,
                                                 played=mv,
                                                 delta=round(ps - bs, 1)))
                                if len(rows) >= max_rows:
                                    break
        try:
            g.apply(mv)
        except ValueError:
            break
    return rows, stem


def append_cases(cases):
    if not OUT.exists():
        OUT.write_text('[\n]', encoding='utf-8')
    raw = OUT.read_text(encoding='utf-8-sig')
    nl = '\r\n' if '\r\n' in raw else '\n'
    tail = raw.rstrip()
    if not tail.endswith(']'):
        raise ValueError('output file does not end with ]')
    head = tail[:-1].rstrip()
    sep = ',' + nl if head.endswith('}') else ''
    chunks = [head, sep]
    for i, case in enumerate(cases):
        if i:
            chunks.append(',' + nl)
        body = json.dumps(case, indent=2)
        chunks.append(nl.join('  ' + line for line in body.split('\n')))
    chunks.append(nl + ']' + nl)
    OUT.write_text(''.join(chunks), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--games', type=int, default=900)
    ap.add_argument('--max-new', type=int, default=1500)
    ap.add_argument('--margin', type=float, default=25.0)
    ap.add_argument('--search-s', type=float, default=3.0)
    ap.add_argument('--min-ply', type=int, default=10)
    ap.add_argument('--max-per-game', type=int, default=12)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260908)
    ap.add_argument('--kind', choices=('wall', 'pawn', 'any'), default='wall',
                    help='best-move type to record')
    ap.add_argument('--skip-source', default=None,
                    help='another cases JSON whose (code,ply)/histories are skipped')
    ap.add_argument('--out', default=None, help='override output file (smoke tests)')
    args = ap.parse_args()
    if args.out:
        global OUT
        OUT = Path(args.out)
    existing, keys, hists = load_state()
    skip_codes, gate_hists = gate_codes_and_histories()
    skip_codes |= {x['code'] for x in existing}
    skip_hists = set(hists) | gate_hists
    if args.skip_source:
        src = json.loads(Path(args.skip_source).read_text(encoding='utf-8-sig'))
        skip_codes |= {x['code'] for x in src if isinstance(x, dict)}
        for x in src:
            if isinstance(x, dict) and isinstance(x.get('history'), list):
                skip_hists.add(tuple(x['history']))
                keys.add((x.get('code'), x.get('ply')))
        print(json.dumps(dict(skip_source=args.skip_source, skipped=len(src))), flush=True)
    files = sorted((ROOT / 'study' / 'archive').glob('*.json'))
    rng = random.Random(args.seed)
    rng.shuffle(files)
    print(json.dumps(dict(existing=len(existing), candidate_files=len(files),
                          gate_codes=len(skip_codes))), flush=True)
    found = added = examined = 0
    started = time.monotonic()
    new_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        it = (pool.submit(mine_game,
                          (str(p), skip_codes, skip_hists, args.margin,
                           args.min_ply, args.search_s, args.max_per_game,
                           args.kind))
              for p in files)
        import concurrent.futures as cf
        pending = set()
        for fut in it:
            pending.add(fut)
            if len(pending) >= args.workers * 2:
                done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for d in done:
                    rows, stem = d.result()
                    examined += 1
                    found += len(rows)
                    for row in rows:
                        key = (row['code'], row['ply'])
                        if key not in keys:
                            keys.add(key)
                            new_rows.append(row)
                    if len(new_rows) >= args.max_new:
                        break
                if len(new_rows) >= args.max_new or examined >= args.games:
                    for d in pending:
                        d.cancel()
                    break
            if len(new_rows) >= args.max_new or examined >= args.games:
                break
        else:
            done, _ = cf.wait(pending)
            for d in done:
                rows, _ = d.result()
                examined += 1
                found += len(rows)
                for row in rows:
                    key = (row['code'], row['ply'])
                    if key not in keys:
                        keys.add(key)
                        new_rows.append(row)
    new_rows.sort(key=lambda r: (r['code'], r['ply']))
    # cross-worker dedupe by position history (workers each held their own copy)
    seen_hists = set(hists)
    final = []
    for row in new_rows:
        h = tuple(row['history'])
        if h in seen_hists:
            continue
        seen_hists.add(h)
        final.append(row)
    new_rows = final
    if new_rows:
        append_cases(new_rows)
    total = len(existing) + len(new_rows)
    print(json.dumps(dict(scanned=examined, cases=found, added=len(new_rows),
                          total=total,
                          seconds=round(time.monotonic() - started, 1))), flush=True)


if __name__ == '__main__':
    main()
