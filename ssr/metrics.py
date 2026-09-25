"""Offline surface metrics on a shared GT sample and observation-distance bins."""
import numpy as np
from scipy.spatial import cKDTree


def observation_regions(gt, observations):
    distance = cKDTree(observations).query(gt, workers=4)[0]
    # Stable equal-size rank bins also handle tied distances without empty bins.
    regions = np.empty(len(gt), dtype=np.int8)
    for index, ids in enumerate(np.array_split(np.argsort(distance, kind='stable'), 4)):
        regions[ids] = index
    return distance, regions


def surface_metrics(prediction, gt, observations, scale, regions=None):
    if not len(prediction) or not len(gt) or scale <= 0:
        raise ValueError('Nonempty surfaces and positive evaluation scale required')
    acc = cKDTree(gt).query(prediction, workers=4)[0]
    cov = cKDTree(prediction).query(gt, workers=4)[0]
    distance, default_regions = observation_regions(gt, observations)
    regions = default_regions if regions is None else regions
    bins = []
    for i in range(4):
        selected = regions == i
        bins.append({'quantile': [25 * i, 25 * (i + 1)], 'gt_count': int(selected.sum()),
                     'observation_distance_mean': float(distance[selected].mean()),
                     'coverage': float(cov[selected].mean()),
                     'normalized_coverage': float(cov[selected].mean() / scale)})
    cd = (acc.mean() + cov.mean()) / 2
    result = {'accuracy': float(acc.mean()), 'coverage': float(cov.mean()), 'cd_l1': float(cd),
              'sampled_hausdorff': float(max(acc.max(), cov.max())),
              'gap_coverage': bins[3]['coverage'], 'normalized_cd_l1': float(cd / scale),
              'normalized_accuracy': float(acc.mean() / scale),
              'normalized_coverage': float(cov.mean() / scale),
              'normalized_gap_coverage': bins[3]['normalized_coverage'],
              'normalized_sampled_hausdorff': float(max(acc.max(), cov.max()) / scale),
              'observation_distance_bins': bins, 'prediction_count': len(prediction), 'gt_count': len(gt),
              'coverage_convention': 'mean GT-to-prediction distance; lower is better',
              'surface_metric': 'sample based', 'normal_consistency': None,
              'evaluation_scale': float(scale)}
    result['f_scores'] = {}
    for threshold in (.005, .01, .02):
        precision = float((acc <= threshold * scale).mean())
        recall = float((cov <= threshold * scale).mean())
        result['f_scores'][str(threshold)] = {'normalized_threshold': threshold,
            'precision': precision, 'recall': recall,
            'f_score': 2 * precision * recall / (precision + recall) if precision + recall else 0.}
    return result
