#!/usr/bin/env python3
"""Download public profile game histories, with pagination, deduplication and resume.

No credentials or official analysis requests. Archives stay out of Git by default.
"""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import coach

BASE = 'https://api.barricade.gg'
ROOT = Path(__file__).resolve().parent / 'study'

def get(path):
    request = Request(BASE + path, headers={'User-Agent': 'BarricadeCoach/1.0 (public-game study)', 'Accept': 'application/json'})
    for attempt in range(6):
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            retry = exc.headers.get('Retry-After', '')
            delay = max(60 * (attempt+1), int(retry) if retry.isdigit() else 0)
            print(f'Public API rate limit; pausing {delay}s before retry', flush=True)
            time.sleep(delay)

def collect(usernames):
    archive = ROOT / 'archive'
    profiles = ROOT / 'profiles'
    archive.mkdir(parents=True, exist_ok=True)
    profiles.mkdir(parents=True, exist_ok=True)
    all_codes, summary = set(), {}
    for name in usernames:
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Use a plain profile username')
        cached = profiles / f'{name}.json'
        if cached.exists():
            games = json.loads(cached.read_text(encoding='utf-8'))
            all_codes.update(g['shareCode'] for g in games)
            summary[name] = dict(games=len(games), cached=True, wins=sum(g['result']=='win' for g in games))
            print(name, summary[name], flush=True)
            continue
        page, games, seen = 1, [], set()
        while True:
            data = get(f'/api/users/{quote(name, safe="")}/games?page={page}&limit=100')
            batch = data['games']
            fresh = [g for g in batch if g['shareCode'] not in seen]
            games.extend(fresh)
            seen.update(g['shareCode'] for g in fresh)
            pagination = data['pagination']
            if not pagination['hasMore']:
                break
            if not fresh:
                raise ValueError(f'{name}: pagination stopped making progress at page {page}')
            page += 1
            time.sleep(0.1)
        (profiles / f'{name}.json').write_text(json.dumps(games, indent=2), encoding='utf-8')
        all_codes.update(seen)
        summary[name] = dict(games=len(games), reported_total=pagination['totalGames'], pages=page,
                             wins=sum(g['result']=='win' for g in games))
        print(name, summary[name], flush=True)
    failures, validated, plies = [], 0, 0
    for i, code in enumerate(sorted(all_codes)):
        try:
            if not code.isalnum():
                raise ValueError('Invalid share code')
            path = archive / f'{code}.json'
            data = None
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding='utf-8'))
                except ValueError:
                    pass  # Retry an incomplete download from an interrupted old run.
            if data is None:
                data = get('/games/' + code)
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
                temporary.replace(path)
                time.sleep(1.1)
            if data['boardSize'] != 9:
                raise ValueError(f'Unsupported board size {data["boardSize"]}')
            game = coach.Game(data['historyCsv'])
            for side in (coach.RED, coach.BLUE):
                if game.remaining[side] != data[f'p{side+1}RemainingBarricades']:
                    raise ValueError('Wall inventory mismatch')
            validated += 1
            plies += len(game.history)
        except HTTPError as exc:
            failures.append(dict(code=code, error=str(exc)))
            if exc.code in (401,403,429):
                print(f'Stopping download: HTTP {exc.code}. Run again later to resume.', flush=True)
                break
        except Exception as exc:
            failures.append(dict(code=code, error=str(exc)))
        if (i+1) % 50 == 0:
            print(f'{i+1}/{len(all_codes)} downloaded; {validated} valid', flush=True)
    report = dict(profiles=summary, unique_games=len(all_codes), validated=validated, plies=plies,
                  failures=failures, fetched_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    report['downloaded'] = sum((archive/f'{code}.json').exists() for code in all_codes)
    report['complete'] = report['downloaded'] == len(all_codes)
    (ROOT/'collection-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('usernames', nargs='+')
    collect(parser.parse_args().usernames)
    from build_book import build
    build()
