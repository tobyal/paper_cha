"""Evaluate saved supports and meshes offline; no GT enters reconstruction."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssr.geometry import load_points, write_json
from ssr.metrics import surface_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--run', required=True)
    args = p.parse_args()
    case = json.loads(Path(args.case).read_text())
    run = Path(args.run)
    observations = load_points(case['input'])
    with np.load(case['evaluation_cache']) as data:
        gt, scale, regions = data['gt'], float(data['scale']), data['regions']
    consumed = load_points(run / 'input.npz')
    assert np.array_equal(consumed, observations), 'Evaluation case/input mismatch'
    initial = load_points(run / 'support_initial.ply')
    rows = []
    for path in sorted(run.glob('*.ply')):
        if not path.name.startswith(('support_', 'mesh_')):
            continue
        kind, tag = path.stem.split('_', 1)
        points = load_points(path, 100000, seed=22, strict_count=False)
        metrics = surface_metrics(points, gt, observations, scale, regions)
        snapshot = run / f'snapshot_{tag}.json'
        state = json.loads(snapshot.read_text()) if snapshot.exists() else {'step': 0, 'seconds': 0}
        metrics.update(kind=kind, tag=tag, step=state['step'], seconds=state['seconds'])
        if kind == 'support' and len(points) == len(initial):
            metrics['normalized_displacement'] = float(np.linalg.norm(points - initial, axis=1).mean() / scale)
            metrics['normalized_displacement_p95'] = float(np.quantile(np.linalg.norm(points - initial, axis=1), .95) / scale)
        if state.get('adaptation_audit') is not None:
            metrics['adaptation_audit'] = state['adaptation_audit']
        write_json(run / 'metrics' / f'{path.stem}.json', metrics)
        rows.append(metrics)
    write_json(run / 'evaluation.json', {'case': case['name'], 'rows': rows,
        'status': json.loads((run / 'status.json').read_text()),
        'config': json.loads((run / 'config.json').read_text()),
        'gt_used_for_training': False})
    print(f'Evaluated {len(rows)} artifacts in {run}', flush=True)


if __name__ == '__main__':
    main()
