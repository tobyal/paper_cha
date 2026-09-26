"""One single-shape job per requested GPU, with visible logs and failure status."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import hashlib
import zipfile
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
    project = Path(__file__).resolve().parent
    source_files = subprocess.check_output(['git', '-C', str(project), 'ls-files', '--cached', '--others',
                                             '--exclude-standard', '-z']).decode().split('\0')
    with zipfile.ZipFile(root / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in source_files:
            if name and (project / name).is_file():
                archive.write(project / name, name)
    source_hash = hashlib.sha256((root / 'source_snapshot.zip').read_bytes()).hexdigest()
    write_json(root / 'source.json', {'snapshot_sha256': source_hash,
        'git_base': subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD']).decode().strip(),
        'working_tree_changes': subprocess.check_output(['git', '-C', str(project), 'status', '--short']).decode()})
    write_json(root / 'status.json', {'status': 'running', 'pid': os.getpid(), 'gpus': args.gpus,
                                     'total_jobs': len(jobs)})
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
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                       PAPER_CHA_SOURCE_SHA256=source_hash)
            with (root / (job['name'] + '.log')).open('w') as log:
                code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                evaluation_code = None
                if code == 0 and job.get('case'):
                    evaluation_code = subprocess.call([sys.executable, str(Path(__file__).parent / 'scripts/analyze_run.py'),
                        '--case', job['case'], '--run', str(root / job['name'])], env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt = {'job': job['name'], 'gpu': gpu, 'returncode': code, 'command': command,
                       'evaluation_returncode': evaluation_code, 'definition': job}
            write_json(root / (job['name'] + '.receipt.json'), receipt)
            receipts.append(receipt)
        return receipts
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        receipts = [item for lane_results in pool.map(lane, args.gpus) for item in lane_results]
    failed = any(row['returncode'] or row['evaluation_returncode'] for row in receipts)
    summary_code = None
    if any(row['evaluation_returncode'] == 0 for row in receipts):
        summary_script = ('summarize_diagnostics.py' if all(job.get('family') == 'adaptation_diagnostics' for job in jobs)
                          else 'summarize_evidence.py')
        summary_code = subprocess.call([sys.executable, str(Path(__file__).parent / 'scripts' / summary_script),
                                        '--root', str(root)])
        failed = failed or summary_code != 0
    write_json(root / 'status.json', {'status': 'failed' if failed else 'complete', 'jobs': receipts,
                                     'summary_returncode': summary_code})
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
