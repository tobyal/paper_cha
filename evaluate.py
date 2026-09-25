"""Evaluation only: GT is never supplied to reconstruction by this command."""
import argparse
import numpy as np
from ssr.geometry import load_points, write_json
from ssr.metrics import surface_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction', required=True)
    p.add_argument('--gt', required=True)
    p.add_argument('--observations', required=True, help='Saved input.ply from the reconstruction run')
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=100000)
    args = p.parse_args()
    pred = load_points(args.prediction, args.samples, strict_count=False)
    gt = load_points(args.gt, args.samples, seed=22, strict_count=False)
    obs = load_points(args.observations)
    # Pointclouds with fewer than --samples are valid support sets, not upsampled.
    scale = float(np.linalg.norm(obs - (obs.max(0) + obs.min(0)) / 2, axis=1).max())
    metrics = surface_metrics(pred, gt, obs, scale)
    metrics.update(distance_units='input world coordinates',
                   normal_note='No fabricated normals for explicit support points')
    write_json(args.output, metrics)


if __name__ == '__main__':
    main()
