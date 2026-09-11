#!/usr/bin/env python3
"""Run one unattended, fail-closed Barricade improvement cycle.

No candidate can affect live advice unless arena.py promotes it after the raw
screen, production repair/parity check, and color-swapped paired arena. Raw
self-play is generated before candidate training and is frozen for the run, so
the candidate never trains on its own unvalidated output.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import policy_value

ROOT=Path(__file__).resolve().parent
NIGHTLY=ROOT/'memory'/'nightly'
TRAINING_ARTIFACTS=(
    'archive-teacher.jsonl',
    'late-teacher.jsonl',
    'frozen-raw-selfplay.jsonl',
    'verified-loss-teacher.jsonl',
)


def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def low_priority():
    if os.name=='nt':
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
        except Exception:pass


def alive(pid):
    try:os.kill(int(pid),0);return True
    except (OSError,ValueError,TypeError):return False


def prior_training_inputs(current_dir, limit=3):
    """Reuse only independent corpora from completed earlier runs.

    Candidate models, promotion reports, and post-promotion league output are
    deliberately excluded.  This makes improvement cumulative without feeding
    an unvalidated model's own choices back into its successor.
    """
    completed=[]
    for status_path in NIGHTLY.glob('*/status.json'):
        if status_path.parent.resolve()==current_dir.resolve():continue
        try:status=json.loads(status_path.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):continue
        if status.get('state')!='complete':continue
        paths=[status_path.parent/name for name in TRAINING_ARTIFACTS]
        paths=[path for path in paths if path.is_file() and path.stat().st_size]
        if paths:completed.append((status.get('finished',''),status_path.parent,paths))
    completed.sort(key=lambda row:(row[0],str(row[1])))
    selected=completed[-max(0,limit):] if limit else []
    return [path for _,_,paths in selected for path in paths]


class Run:
    def __init__(self,run_dir):
        self.dir=run_dir;self.dir.mkdir(parents=True,exist_ok=True)
        self.status_path=self.dir/'status.json';self.status=dict(
            state='running',pid=os.getpid(),started=utc(),run_dir=str(self.dir),
            champion_before=None,champion_after=None,current=None,steps=[])
        model=policy_value.load_champion()
        self.status['champion_before']=policy_value.model_id(model) if model else None
        self.write()
    def write(self):
        temp=self.status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.status,indent=2),encoding='utf-8');temp.replace(self.status_path)
        latest=NIGHTLY/'latest.json';tmp=latest.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.status,indent=2),encoding='utf-8');tmp.replace(latest)
    def step(self,name,args,optional=False):
        marker=self.dir/(name+'.done.json')
        if marker.exists():
            saved=json.loads(marker.read_text(encoding='utf-8'));self.status['steps'].append(saved)
            self.status['current']=None;self.write();print(f'= {name} already complete',flush=True);return True
        row=dict(name=name,state='running',started=utc(),command=[str(x) for x in args])
        self.status['current']=row;self.write();print('+',name,' '.join(map(str,args)),flush=True)
        log=self.dir/(name+'.log');started=time.monotonic()
        with log.open('a',encoding='utf-8') as out:
            completed=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,
                                     stdout=out,stderr=subprocess.STDOUT)
        row.update(state='complete' if completed.returncode==0 else 'failed',
                   finished=utc(),seconds=round(time.monotonic()-started,2),
                   returncode=completed.returncode,log=str(log))
        self.status['steps'].append(row);self.status['current']=None;self.write()
        if completed.returncode==0:marker.write_text(json.dumps(row,indent=2),encoding='utf-8')
        if completed.returncode and not optional:raise RuntimeError(f'{name} failed; see {log}')
        return completed.returncode==0


def rel(path):return str(Path(path).resolve().relative_to(ROOT))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--accounts',default='steak2222,steak222');ap.add_argument('--prefix',default='steak')
    ap.add_argument('--teacher-positions',type=int,default=5000)
    ap.add_argument('--selfplay-games',type=int,default=300)
    ap.add_argument('--late-positions',type=int,default=500)
    ap.add_argument('--history-runs',type=int,default=3,
                    help='completed independent nightly corpora to retain')
    ap.add_argument('--seed',type=int,
                    help='master seed (generated and recorded when omitted)')
    ap.add_argument('--run-dir');args=ap.parse_args()
    if args.history_runs<0:ap.error('--history-runs must be non-negative')
    seed=args.seed if args.seed is not None else time.time_ns()%2_000_000_000
    low_priority();NIGHTLY.mkdir(parents=True,exist_ok=True)
    lock=NIGHTLY/'nightly.lock'
    if lock.exists():
        try:old=json.loads(lock.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):old={}
        if alive(old.get('pid')):raise SystemExit(f'nightly run already active as pid {old.get("pid")}')
        lock.unlink(missing_ok=True)
    lock.write_text(json.dumps(dict(pid=os.getpid(),started=utc())),encoding='utf-8')
    stamp=time.strftime('%Y%m%d-%H%M%S',time.localtime())
    run_dir=Path(args.run_dir).resolve() if args.run_dir else NIGHTLY/stamp
    run=Run(run_dir)
    run.status['seed']=seed;run.status['history_runs']=args.history_runs;run.write()
    teacher=run_dir/'archive-teacher.jsonl';late=run_dir/'late-teacher.jsonl'
    baseline=run_dir/'frozen-raw-selfplay.jsonl';verified=run_dir/'verified-loss-teacher.jsonl'
    candidate=run_dir/'candidate'/'model.json';candidate.parent.mkdir(parents=True,exist_ok=True)
    try:
        common=['--accounts',args.accounts,'--prefix',args.prefix,'--seconds','30','--margin','30']
        run.step('import-start',['update_real_games.py',*common],optional=True)
        run.step('frozen-selfplay',['league_selfplay.py','--games',args.selfplay_games,
                 '--seconds','.18','--workers','4','--raw-only','--no-production-safety','--seed',seed,
                 '--output',rel(baseline)])
        run.step('archive-teacher',['teacher_data.py','--positions',args.teacher_positions,
                 '--depth','3','--seconds','12','--workers','8','--balance-colors','--top-players-only',
                 '--seed',seed+1,'--output',rel(teacher)])
        run.step('late-teacher',['late_teacher.py','--positions',args.late_positions,
                 '--scan-games','30000','--depth','6','--min-depth','5','--seconds','20',
                 '--workers','8','--seed',seed+2,'--output',rel(late)])
        run.step('import-pretrain',['update_real_games.py',*common],optional=True)
        run.step('verified-loss-teacher',['verified_loss_teacher.py','--reports',
                 'study/regressions/verified-*.json','--output',rel(verified),
                 '--depth','4','--min-depth','3','--seconds','30','--workers','8'])
        history_inputs=prior_training_inputs(run_dir,args.history_runs)
        inputs=[teacher,late,baseline,verified,*history_inputs,
                ROOT/'study/loss-teacher-walls.jsonl',ROOT/'study/loss-teacher-pawns.jsonl',
                ROOT/'study/deep-wall-teacher.jsonl']
        run.status['training_inputs']=[str(path) for path in inputs]
        run.status['prior_training_inputs']=[str(path) for path in history_inputs]
        run.write()
        run.step('train-candidate',['train_nn.py',*[str(p) for p in inputs],
                 '--epochs','20','--hidden','512','--teacher-frac','.65',
                 '--feature-workers','8','--seed',seed+3,'--output',str(candidate)])
        run.step('promotion-gate',['arena.py',str(candidate),str(teacher),'--pairs','100',
                 '--seconds','1','--workers','2','--repair-workers','1','--player-holdout-only',
                 '--measure','--promote'])
        report=json.loads((candidate.parent/'arena-report.json').read_text(encoding='utf-8'))
        if report.get('promoted'):
            run.step('post-promotion-league',['league_selfplay.py','--games','200','--seconds','.18',
                     '--workers','4','--no-production-safety','--seed',seed+4,
                     '--models',str(candidate),'--output',rel(run_dir/'promoted-league.jsonl')])
        run.step('import-final',['update_real_games.py',*common],optional=True)
        champion=policy_value.load_champion();run.status.update(
            state='complete',finished=utc(),current=None,promoted=bool(report.get('promoted')),
            champion_after=policy_value.model_id(champion) if champion else None,
            arena_report=str(candidate.parent/'arena-report.json'))
        run.write();print(json.dumps(run.status,indent=2),flush=True)
    except Exception as error:
        run.status.update(state='failed',finished=utc(),current=None,error=str(error),
                          traceback=traceback.format_exc());run.write();raise
    finally:
        try:
            current=json.loads(lock.read_text(encoding='utf-8'))
            if current.get('pid')==os.getpid():lock.unlink(missing_ok=True)
        except (OSError,json.JSONDecodeError):pass


if __name__=='__main__':main()
