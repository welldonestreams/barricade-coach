#!/usr/bin/env python3
"""Refresh selected profiles, audit coach usage, and review verified losses.

Loss reviews are kept separate from both training input and the held-out repair
gate. A human can promote a deeply confirmed position into exactly one of those
sets later, preserving the no-feedback-loop invariant.
"""
import argparse
import json
import time
from pathlib import Path

import collect_games
import measure_real_games
import regress_losses

ROOT = Path(__file__).resolve().parent


def review_loss(item, seconds, margin):
    path = measure_real_games.locate_game(item['code'])
    if not path:
        return dict(**item, review='missing archive')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    cases = regress_losses.build_regressions(
        data['historyCsv'], 'blue' if item['color'] == 'red' else 'red',
        item['color'], seconds, margin)
    report = dict(code=item['code'], source='verified_coach_trace',
                  coach_build=item['build'], user_side=item['color'],
                  seconds=seconds, margin=margin, blunders=cases,
                  status='candidate_only_not_training_or_gate',
                  built=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    destination = ROOT / 'study' / 'regressions' / f"verified-{item['code']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return dict(**item, review=str(destination.relative_to(ROOT)), candidates=len(cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounts', default='steak2222', help='comma-separated exact usernames')
    parser.add_argument('--prefix', default='steak', help='also select usernames with this prefix')
    parser.add_argument('--seconds', type=float, default=12,
                        help='independent offline search budget per user turn in a verified loss')
    parser.add_argument('--margin', type=float, default=30,
                        help='minimum completed depth-2 score gap to save as a candidate')
    parser.add_argument('--skip-loss-review', action='store_true')
    args = parser.parse_args()
    accounts = tuple(x.strip() for x in args.accounts.split(',') if x.strip())
    if not accounts:
        parser.error('at least one exact account is required for profile refresh')

    collect_games.collect(accounts, refresh_profiles=True)
    report = measure_real_games.measure(accounts, args.prefix)
    reviewed = []
    if not args.skip_loss_review:
        for item in report['verified_loss_queue']:
            reviewed.append(review_loss(item, args.seconds, args.margin))
    report['reviewed_losses'] = reviewed
    report['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    measurement = ROOT / 'study' / 'real-game-measurement.json'
    queue = ROOT / 'study' / 'regression-queue.json'
    measurement.write_text(json.dumps(report, indent=2), encoding='utf-8')
    queue.write_text(json.dumps(dict(cases=report['verified_loss_queue'], reviewed=reviewed), indent=2),
                     encoding='utf-8')
    print(json.dumps(dict(counts=report['counts'], rows=report['rows'],
                               verified_losses=len(report['verified_loss_queue']),
                               reviewed_losses=reviewed,
                               report=str(measurement)), indent=2))


if __name__ == '__main__':
    main()
