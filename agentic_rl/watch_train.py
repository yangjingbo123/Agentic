#!/usr/bin/env python3
"""
Real-time training progress monitor.
Usage: python watch_train.py [progress.json path]
       python watch_train.py  # auto-detects latest run under checkpoints/
"""
import json
import os
import sys
import time
import glob

def find_latest_progress():
    pattern = "checkpoints/*/*/progress.json"
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def bar(step, total, width=40):
    filled = int(width * step / total) if total else 0
    return f"[{'#' * filled}{'.' * (width - filled)}]"

def display(path):
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        print(f"  waiting for {path} ...")
        return

    step  = d.get('step', 0)
    total = d.get('total_steps', 0)
    pct   = d.get('percent', 0)
    acc   = d.get('acc', 0)
    rew   = d.get('reward', 0)
    gen   = d.get('gen_time_s', 0)
    upd   = d.get('update_time_s', 0)
    ela   = d.get('elapsed_min', 0)
    eta   = d.get('eta_min', 0)

    os.system('clear')
    print("=" * 60)
    print(f"  Training Progress Monitor")
    print(f"  {path}")
    print("=" * 60)
    print(f"  Step    : {step} / {total}  ({pct}%)")
    print(f"  {bar(step, total)}")
    print(f"  Acc     : {acc:.4f}")
    print(f"  Reward  : {rew:.4f}")
    print(f"  Gen     : {gen:.1f}s/step   Update: {upd:.1f}s/step")
    print(f"  Elapsed : {ela:.1f} min    ETA: {eta:.1f} min")
    print("=" * 60)
    print("  Ctrl+C to exit")

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_latest_progress()
        if path is None:
            print("No progress.json found under checkpoints/. Pass path as argument.")
            print("Usage: python watch_train.py checkpoints/<project>/<run>/progress.json")
            sys.exit(1)
        print(f"Auto-detected: {path}")
        time.sleep(1)

    try:
        while True:
            display(path)
            # refresh auto-detect every iteration so it picks up new runs
            if len(sys.argv) == 1:
                latest = find_latest_progress()
                if latest and latest != path:
                    path = latest
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")

if __name__ == '__main__':
    main()
