"""Build evidence tables and standalone scientific plots from completed runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssr.geometry import load_points, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    reports = [(path.parent.name, json.loads(path.read_text())) for path in sorted(root.glob('*/evaluation.json'))]
    if not reports:
        raise RuntimeError('No evaluated runs yet')
    rows = []
    for name, report in reports:
        for row in report['rows']:
            rows.append({'run': name, 'case': report['case'], 'mode': report['config']['args']['mode'],
                         'kind': row['kind'], 'tag': row['tag'], 'step': row['step'],
                         **{key: row.get(key) for key in ('normalized_cd_l1', 'normalized_accuracy',
                             'normalized_coverage', 'normalized_gap_coverage', 'normalized_sampled_hausdorff',
                             'normalized_displacement')},
                         'f_score_001': row['f_scores']['0.01']['f_score'], 'count': row['prediction_count']})
    with (root / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for case in sorted({report['case'] for _, report in reports}):
        members = [(name, report) for name, report in reports if report['case'] == case]
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for name, report in members:
            label = name.removeprefix(case + '_')
            for kind, ax in [('support', axes[0, 0]), ('mesh', axes[0, 1])]:
                curve = sorted([row for row in report['rows'] if row['kind'] == kind and row['tag'] != 'initial'],
                               key=lambda r: r['step'])
                ax.plot([r['step'] for r in curve], [r['normalized_cd_l1'] for r in curve], marker='o', label=label)
            curve = sorted([row for row in report['rows'] if row['kind'] == 'support' and row['tag'] != 'initial'],
                           key=lambda r: r['step'])
            axes[1, 0].plot([r['step'] for r in curve], [r['normalized_displacement'] for r in curve],
                            marker='o', label=label)
            final = next(row for row in report['rows'] if row['kind'] == 'support' and row['tag'] == 'final')
            axes[1, 1].plot([1, 2, 3, 4], [b['normalized_coverage'] for b in final['observation_distance_bins']],
                            marker='o', label=label)
        titles = ['Support CD-L1 (lower is better)', 'Mesh CD-L1 (lower is better)',
                  'Support displacement from initialization', 'Final support: distance-to-observation analysis']
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title)
            ax.set_xlabel('Step')
            ax.set_ylabel('Distance / observation scale')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        axes[1, 1].set_xlabel('Observation-distance quartile: near -> far')
        axes[1, 1].set_xticks([1, 2, 3, 4], ['0-25%', '25-50%', '50-75%', '75-100%'])
        for suffix in ('png', 'pdf'):
            figure.savefig(root / f'{case}_evidence.{suffix}', dpi=180)
        plt.close(figure)

        # Fixed projections, axis limits and point sizes across all support sets.
        receipt = json.loads((root / f'{members[0][0]}.receipt.json').read_text())
        case_data = json.loads(Path(receipt['definition']['case']).read_text())
        with np.load(case_data['evaluation_cache']) as data:
            gt = data['gt']
        items = [('GT samples', gt[::max(1, len(gt) // 5000)]),
                 ('Input 1024', load_points(case_data['input']))]
        for name, report in members:
            label = name.removeprefix(case + '_')
            if label != 'raw':
                items.append((label, load_points(root / name / 'support_final.ply')))
        axis_order = np.argsort(np.ptp(gt, axis=0))[::-1]
        figure, axes = plt.subplots(2, len(items), figsize=(3 * len(items), 6), squeeze=False,
                                    constrained_layout=True)
        for column, (label, points) in enumerate(items):
            for row, vertical in enumerate(axis_order[1:]):
                horizontal = axis_order[0]
                ax = axes[row, column]
                ax.scatter(points[:, horizontal], points[:, vertical], s=1.2, alpha=.65,
                           linewidths=0, rasterized=True)
                extent = max(np.ptp(gt[:, horizontal]), np.ptp(gt[:, vertical])) * .55
                middle = (gt.min(0) + gt.max(0)) / 2
                ax.set_xlim(middle[horizontal] - extent, middle[horizontal] + extent)
                ax.set_ylim(middle[vertical] - extent, middle[vertical] + extent)
                ax.set_aspect('equal')
                ax.set_title(label if row == 0 else f'{"xyz"[horizontal]} / {"xyz"[vertical]} view')
                ax.set_xticks([])
                ax.set_yticks([])
        figure.savefig(root / f'{case}_supports.png', dpi=160)
        plt.close(figure)

        paired = []
        for name, report in members:
            support = next((r for r in report['rows'] if r['kind'] == 'support' and r['tag'] == 'final'), None)
            mesh = next((r for r in report['rows'] if r['kind'] == 'mesh' and r['tag'] == 'final'), None)
            if support and mesh:
                paired.append({'run': name, 'support_cd': support['normalized_cd_l1'], 'mesh_cd': mesh['normalized_cd_l1']})
        correlation = None
        if len(paired) >= 3 and np.std([r['support_cd'] for r in paired]) > 0 and np.std([r['mesh_cd'] for r in paired]) > 0:
            correlation = float(np.corrcoef([r['support_cd'] for r in paired], [r['mesh_cd'] for r in paired])[0, 1])
        summaries.append({'case': case, 'support_mesh_pairs': paired, 'pearson_descriptive_only': correlation})
    write_json(root / 'summary.json', {'runs': len(reports), 'cases': summaries,
        'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'interpretation': 'Pilot measurements; correlation is descriptive, not a causal proof'})
    text = ['# 1024 点证据链实验', '',
            '完整先验指 Encoding + 中间表示与查询网络 + Decode。场为原三平面多尺度后端，无 GSHE。', '',
            '以下距离均除以输入观测的固定归一化尺度，数值越小越好；F-score 越大越好。']
    groups = [('中间支持质量', lambda r: r['kind'] == 'support' and r['tag'] == 'final'),
              ('固定支持来源：共同后端网格', lambda r: r['kind'] == 'mesh' and r['tag'] == 'final'
               and r['mode'] in ('raw', 'external', 'frozen')),
              ('支持—场反馈：Frozen / One-way / Joint', lambda r: r['tag'] == 'final'
               and r['mode'] in ('frozen', 'oneway', 'joint'))]
    for title, include in groups:
        text += ['', f'## {title}', '',
                 '| Run | Object | CD-L1 | Coverage | GapCoverage | F-score@0.01 |',
                 '|---|---|---:|---:|---:|---:|']
        for row in rows:
            if include(row):
                text.append(f"| {row['run']} | {row['kind']} | {row['normalized_cd_l1']:.6f} | "
                            f"{row['normalized_coverage']:.6f} | {row['normalized_gap_coverage']:.6f} | {row['f_score_001']:.4f} |")
    text += ['', '共同后端比较使用 Raw / NTPS / BSDF / Frozen；反馈比较使用 Frozen / One-way / Joint。',
             'Raw 保持 1024 个观测，其他支持按配置统一预算。NTPS 为本地 PyTorch 移植版，未验证官方 TF 等价性。',
             '历史 baseline 支持来自 30 分钟优化，复用来源、输入哈希、坐标系与实现信息保存在任务记录。',
             '本次运行比较固定步数的共同后端；不是三种原生方法在相同时间内的最终网格排名。', '']
    (root / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    print(f'Summarized {len(reports)} runs in {root}', flush=True)


if __name__ == '__main__':
    main()
