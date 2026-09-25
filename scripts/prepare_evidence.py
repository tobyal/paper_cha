"""Prepare fixed 1024-point evidence cases; optionally reuse audited baseline supports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssr.geometry import load_points, normalize, save_points, write_json
from ssr.metrics import observation_regions


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--name', default='airplane_1')
    p.add_argument('--points', type=int, default=1024)
    p.add_argument('--seed', type=int, default=21)
    p.add_argument('--reuse-time-support', type=Path)
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    if args.reuse_time_support:
        previous = args.reuse_time_support.resolve()
        protocol = json.loads((previous / 'protocol.json').read_text())
        assert digest(args.mesh) == protocol['raw_sha256'], 'Baseline GT shape mismatch'
        input_file = previous / 'inputs' / str(args.points) / 'input.npy'
        assert digest(input_file) == protocol['inputs'][str(args.points)]['sha256']
        observations = np.load(input_file)
        gt = np.load(previous / 'evaluation_only/gt.npy')[:100000]
        frame = {'name': 'historical baseline canonical frame', 'raw_to_frame_center': protocol['center'],
                 'raw_to_frame_scale': protocol['scale']}
        for method in ('ntps', 'bsdf'):
            folder = previous / 'models' / f'{method}_{args.points}'
            status = json.loads((folder / 'status.json').read_text())
            assert status['status'] == 'complete'
            consumed = folder / 'input_consumed.npy'
            assert digest(consumed) == digest(input_file), 'Different baseline observations'
            original = folder / '30min' / ('support_native.npy' if method == 'bsdf' else 'support.npy')
            support = np.load(original)
            target = out / f'{method}_30min_support.npy'
            np.save(target, support)
            sources[method] = {'path': str(target), 'original_path': str(original),
                'sha256': digest(original), 'input_sha256': digest(consumed), 'count': len(support),
                'status': status, 'snapshot': json.loads((folder / '30min/metadata.json').read_text()),
                'implementation': 'local PyTorch NTPS port; official TF parity unverified' if method == 'ntps'
                                  else 'local BSDF with chunked gradient recomputation',
                'reuse': True, 'support_semantics': 'generated support before concatenating observations'}
        input_provenance = {'path': str(input_file), 'sha256': digest(input_file)}
    else:
        observations = load_points(args.mesh, args.points, args.seed)
        gt = load_points(args.mesh, 100000, args.seed + 100)
        frame = {'name': 'original mesh coordinates'}
        input_provenance = {'mesh_sha256': digest(args.mesh), 'sampling_seed': args.seed}
    assert observations.shape == (args.points, 3)
    _, center, scale = normalize(observations)
    distance, regions = observation_regions(gt, observations)
    np.save(out / 'input.npy', observations.astype(np.float32))
    save_points(out / 'input.ply', observations)
    np.savez(out / 'evaluation_only.npz', gt=gt, observation_distance=distance, regions=regions,
             scale=scale, center=center)
    case = {'name': args.name, 'input': str(out / 'input.npy'), 'observations': args.points,
            'source_mesh': str(Path(args.mesh).resolve()), 'input_provenance': input_provenance,
            'evaluation_cache': str(out / 'evaluation_only.npz'), 'frame': frame, 'sources': sources,
            'evaluation_scale': scale, 'input_sha256': digest(out / 'input.npy')}
    write_json(out / 'case.json', case)
    print(json.dumps(case, indent=2))


if __name__ == '__main__':
    main()
