"""Create exactly three diagnostic runs by inheriting a completed Frozen protocol."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', required=True)
    p.add_argument('--case-name', default='airplane_1')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.baseline_root).resolve()
    baselines = {mode: str(root / f'{args.case_name}_{mode}') for mode in ('frozen', 'joint')}
    for path in baselines.values():
        status = json.loads((Path(path) / 'status.json').read_text())
        assert status['status'] == 'complete', 'Baseline is not complete'
    receipt = json.loads((root / f'{args.case_name}_frozen.receipt.json').read_text())
    original = receipt['definition']
    config = json.loads((Path(baselines['frozen']) / 'config.json').read_text())
    steps = config['args']['steps']
    assert config['observations'] == 1024
    jobs = []
    for mode in ('delayed', 'decoder_only', 'residual'):
        options = original['args'].copy()
        options[options.index('--mode') + 1] = mode
        options += ['--initial-reference', str(Path(baselines['frozen']) / 'step_000000.pt')]
        if mode == 'delayed':
            options += ['--warmup-steps', str(steps // 2)]
        jobs.append({'name': f'{args.case_name}_{mode}', 'args': options, 'case': original['case'],
                     'diagnostic_baselines': baselines, 'family': 'adaptation_diagnostics'})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(jobs, indent=2) + '\n')
    print(f'Created three {steps}-step diagnostics; warm-up={steps // 2}')


if __name__ == '__main__':
    main()
