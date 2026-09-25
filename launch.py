"""One single-shape job per requested GPU, with visible logs and failure status."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from ssr.geometry import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs', required=True, help='JSON list of {name, args: [reconstruct.py CLI arguments]}')
    p.add_argument('--gpus', required=True, nargs='+', type=int)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError('Duplicate GPU IDs')
    jobs = json.loads(Path(args.jobs).read_text())
    names = [job['name'] for job in jobs]
    if len(set(names)) != len(names) or any(Path(name).name != name or name in ('.', '..') for name in names):
        raise ValueError('Job names must be unique directory names')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    pending = queue.Queue()
    for job in jobs:
        if '--output' in job['args'] or any(x.startswith('--output=') for x in job['args']):
            raise ValueError('Job output is assigned by launcher')
        pending.put(job)
    def lane(gpu):
        receipts = []
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                break
            command = [sys.executable, str(Path(__file__).parent / 'reconstruct.py'),
                       '--output', str(root / job['name']), *job['args']]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4')
            with (root / (job['name'] + '.log')).open('w') as log:
                code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt = {'job': job['name'], 'gpu': gpu, 'returncode': code, 'command': command}
            write_json(root / (job['name'] + '.receipt.json'), receipt)
            receipts.append(receipt)
        return receipts
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        receipts = [item for lane_results in pool.map(lane, args.gpus) for item in lane_results]
    failed = any(row['returncode'] for row in receipts)
    write_json(root / 'status.json', {'status': 'failed' if failed else 'complete', 'jobs': receipts})
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
