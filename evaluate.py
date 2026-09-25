"""Evaluation only: GT is never supplied to reconstruction by this command."""
import argparse
import numpy as np
from scipy.spatial import cKDTree
from ssr.geometry import load_points, write_json


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
    acc = cKDTree(gt).query(pred, workers=4)[0]
    cov = cKDTree(pred).query(gt, workers=4)[0]
    gap = cKDTree(obs).query(gt, workers=4)[0]
    scale = float(np.linalg.norm(obs - (obs.max(0) + obs.min(0)) / 2, axis=1).max())
    selected = gap >= np.quantile(gap, .75)
    write_json(args.output, {'accuracy': float(acc.mean()), 'coverage': float(cov.mean()),
                            'cd_l1': float((acc.mean() + cov.mean()) / 2),
                            'sampled_hausdorff': float(max(acc.max(), cov.max())),
                            'gap_coverage': float(cov[selected].mean()),
                            'normalized_cd_l1': float((acc.mean() + cov.mean()) / (2 * scale)),
                            'prediction_count': len(pred), 'gt_count': len(gt),
                            'distance_units': 'input world coordinates', 'surface_metric': 'sample based',
                            'normal_consistency': None, 'normal_note': 'No fabricated normals for explicit support points'})


if __name__ == '__main__':
    main()
