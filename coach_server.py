#!/usr/bin/env python3
"""Persistent stdin coach: history CSV|red or blue -> one JSON result per line."""
import json
import sys
import coach

def handle(line):
    try:
        history, side = line.strip().rsplit('|', 1)
        if side not in ('red', 'blue'):
            raise ValueError('Side must be red or blue')
        result = coach.search(history, coach.RED if side == 'red' else coach.BLUE)
        return dict(move=result['scored'][0][1] if result['scored'] else None,
                    **{k: v for k, v in result.items() if k != 'scored'})
    except ValueError as exc:
        return {'error': str(exc)}

if __name__ == '__main__':
    for line in sys.stdin:
        print(json.dumps(handle(line)), flush=True)
