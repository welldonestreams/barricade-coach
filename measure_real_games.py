#!/usr/bin/env python3
"""Measure games that were demonstrably played from versioned coach advice.

A result enters the coach win-rate sample only when every turn for the selected
account has one unambiguous advice row, all rows use one engine build, and the
played move equals the logged top recommendation. Partial or diverged games are
reported separately so ordinary games cannot inflate or depress the coach rate.
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import coach as c

ROOT = Path(__file__).resolve().parent
CODE = re.compile(r'^[A-Za-z0-9]{6}$')


def normalize_game_id(value):
    """Return a share code from a bare code, game path, or analysis URL."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if CODE.fullmatch(value):
        return value
    for pattern in (r'/game/([A-Za-z0-9]{6})(?:$|[/?#&])',
                    r'[?&]game=([A-Za-z0-9]{6})(?:$|[&#])'):
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


def bucket(rating):
    if not isinstance(rating, (int, float)):
        return 'unknown'
    lo = int(rating) // 100 * 100
    return f'{lo}-{lo + 99}'


def interval(wins, games):
    if not games:
        return [None, None]
    z = 1.96
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def selected(name, accounts, prefix):
    folded = (name or '').casefold()
    return folded in accounts or (prefix and folded.startswith(prefix.casefold()))


def top_move(row):
    top = row.get('top')
    if not isinstance(top, list) or not top:
        return None
    item = top[0]
    if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
        return item[1]
    if isinstance(item, dict) and isinstance(item.get('move'), str):
        return item['move']
    return None


def load_advice(log_dir):
    rows = defaultdict(list)
    invalid = 0
    for log in sorted(Path(log_dir).glob('live-advice*.jsonl')):
        for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                invalid += 1
                continue
            code = normalize_game_id(row.get('game'))
            history = row.get('history')
            move = top_move(row)
            if not code or not isinstance(history, list) or not move:
                continue
            rows[code].append(row)
    return rows, invalid


def game_codes_from_profiles(profile_dir, accounts, prefix):
    codes = set()
    for path in Path(profile_dir).glob('*.json'):
        if not selected(path.stem, accounts, prefix):
            continue
        try:
            games = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        for game in games if isinstance(games, list) else []:
            code = normalize_game_id(game.get('shareCode')) if isinstance(game, dict) else None
            if code:
                codes.add(code)
    return codes


def locate_game(code, root=ROOT):
    for relative in (f'study/archive/{code}.json', f'study/additional/{code}.json', f'study/{code}.json'):
        path = Path(root) / relative
        if path.exists():
            return path
    return None


def audit_game(data, advice_rows, accounts, prefix, code=None):
    code = code or normalize_game_id(data.get('shareCode'))
    names = [data.get('player1Username', ''), data.get('player2Username', '')]
    sides = [index for index, name in enumerate(names) if selected(name, accounts, prefix)]
    if len(sides) != 1:
        return None
    side = sides[0]
    try:
        history = c.Game(c.parse_history(data.get('historyCsv', ''))).history
    except (TypeError, ValueError):
        return dict(code=code, status='invalid_game', account=names[side])

    by_history = defaultdict(list)
    for row in advice_rows:
        by_history[tuple(row.get('history', []))].append(row)

    expected = [ply for ply in range(len(history)) if ply % 2 == side]
    covered = followed = 0
    builds = set()
    misses, divergences, ambiguities = [], [], []
    matched = []
    expected_side = 'red' if side == 0 else 'blue'
    for ply in expected:
        candidates = [row for row in by_history.get(tuple(history[:ply]), [])
                      if (row.get('side') or row.get('position', {}).get('side'))
                      in (None, expected_side)]
        choices = {(str(row.get('build', 'unknown')), top_move(row)) for row in candidates}
        if not choices:
            misses.append(ply + 1)
            continue
        if len(choices) != 1:
            ambiguities.append(dict(ply=ply + 1, choices=sorted([list(x) for x in choices])))
            continue
        build, recommendation = next(iter(choices))
        covered += 1
        builds.add(build)
        actual = history[ply]
        did_follow = actual == recommendation
        followed += int(did_follow)
        matched.append(dict(ply=ply + 1, build=build, recommended=recommendation,
                            played=actual, followed=did_follow))
        if not did_follow:
            divergences.append(dict(ply=ply + 1, recommended=recommendation, played=actual))

    full_coverage = covered == len(expected) and not ambiguities
    one_build = len(builds) == 1
    fully_followed = full_coverage and followed == len(expected) and one_build
    winner = {'1': 0, '2': 1, 1: 0, 2: 1}.get(data.get('winner'))
    opponent_rating = data.get('p2Rating' if side == 0 else 'p1Rating')
    return dict(
        code=code, account=names[side], opponent=names[1 - side],
        color=expected_side, opponent_rating=opponent_rating,
        result='win' if winner == side else ('loss' if winner is not None else 'unresolved'),
        user_turns=len(expected), advised_turns=covered, followed_turns=followed,
        coverage=covered / len(expected) if expected else 0,
        follow_rate=followed / covered if covered else None,
        full_coverage=full_coverage, fully_followed=fully_followed,
        build=next(iter(builds)) if one_build else None,
        status=('fully_followed' if fully_followed else
                'mixed_builds' if full_coverage and not one_build else
                'diverged' if divergences else
                'ambiguous' if ambiguities else 'partial'),
        missing_plies=misses, ambiguous_plies=ambiguities,
        divergences=divergences, matched=matched)


def measure(accounts=('steak2222',), prefix='steak', root=ROOT, log_dir=None):
    root = Path(root)
    account_set = {name.casefold() for name in accounts if name}
    advice, invalid_log_rows = load_advice(log_dir or root / 'logs')
    codes = set(advice)
    codes.update(game_codes_from_profiles(root / 'study' / 'profiles', account_set, prefix))
    audits, missing_archives = [], []
    for code in sorted(codes):
        path = locate_game(code, root)
        if not path:
            if code in advice:
                missing_archives.append(code)
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            continue
        audit = audit_game(data, advice.get(code, []), account_set, prefix, code)
        if audit:
            audits.append(audit)

    stats = defaultdict(lambda: [0, 0, 0])
    for game in audits:
        if not game.get('fully_followed') or game['result'] == 'unresolved':
            continue
        key = (game['build'], bucket(game['opponent_rating']), game['color'])
        stats[key][2] += 1
        if game['result'] == 'win':
            stats[key][0] += 1
        else:
            stats[key][1] += 1
    rows = []
    for (build, band, color), (wins, losses, games) in sorted(stats.items()):
        confidence = interval(wins, games)
        rows.append(dict(build=build, opponent_rating=band, color=color, games=games,
                         wins=wins, losses=losses, winrate=wins / games,
                         winrate_95=confidence, target_85_supported=confidence[0] >= .85))
    counts = defaultdict(int)
    for audit in audits:
        counts[audit['status']] += 1
    verified_losses = [dict(code=g['code'], build=g['build'], color=g['color'],
                            opponent=g['opponent'], opponent_rating=g['opponent_rating'])
                       for g in audits if g.get('fully_followed') and g['result'] == 'loss']
    return dict(accounts=sorted(account_set), prefix=prefix, rows=rows,
                counts=dict(sorted(counts.items())), games=audits,
                verified_loss_queue=verified_losses,
                missing_archives=missing_archives, invalid_log_rows=invalid_log_rows,
                methodology=('Win-rate rows include only completed games with one advice row for every '
                             'user turn, one engine build, and every played move equal to the logged top move.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounts', default='steak2222', help='comma-separated exact usernames')
    parser.add_argument('--prefix', default='steak', help='also select usernames with this prefix; blank disables')
    parser.add_argument('--output', help='optional JSON report path')
    parser.add_argument('--queue-losses', help='optional path for fully verified loss review queue')
    args = parser.parse_args()
    report = measure(tuple(x.strip() for x in args.accounts.split(',') if x.strip()), args.prefix)
    rendered = json.dumps(report, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding='utf-8')
    if args.queue_losses:
        path = Path(args.queue_losses)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(cases=report['verified_loss_queue']), indent=2), encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
