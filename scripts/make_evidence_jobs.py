"""Generate the first 1024-point, fixed-step evidence batch without GT arguments."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--support-budget', type=int, default=2048)
    p.add_argument('--queries', type=int, default=512)
    p.add_argument('--mesh-resolution', type=int, default=128)
    args = p.parse_args()
    case = json.loads(Path(args.case).read_text())
    snapshots = sorted(set([max(1, args.steps // 4), max(1, args.steps // 2),
                            max(1, args.steps * 3 // 4), args.steps]))
    shared = ['--input', case['input'], '--steps', str(args.steps), '--minutes', '0',
              '--queries', str(args.queries), '--support-budget', str(args.support_budget),
              '--require-support-budget', '--field-max-levels', '6', '--seed', '21',
              '--mesh-resolution', str(args.mesh_resolution),
              '--snapshot-steps', '0', *map(str, snapshots[:-1])]
    jobs = []
    # Start the coupling comparison together on three GPUs.
    for mode in ('frozen', 'oneway', 'joint', 'raw'):
        jobs.append({'name': f'{case["name"]}_{mode}', 'args': [*shared, '--mode', mode],
                     'case': str(Path(args.case).resolve()), 'family': 'coupling' if mode != 'raw' else 'source'})
    for method, source in case['sources'].items():
        jobs.append({'name': f'{case["name"]}_{method}',
                     'args': [*shared, '--mode', 'external', '--external-support', source['path']],
                     'case': str(Path(args.case).resolve()), 'family': 'source', 'support_source': source})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(jobs, indent=2) + '\n')
    print(f'Prepared {len(jobs)} jobs, {case["observations"]} input points, {args.steps} steps')


if __name__ == '__main__':
    main()
