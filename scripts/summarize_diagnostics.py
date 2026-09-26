"""Report three adaptation diagnostics against the shared, completed baselines."""
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

GROUPS = {'A_delayed': ['frozen', 'joint', 'delayed'],
          'B_decoder': ['frozen', 'joint', 'decoder_only'],
          'C_residual': ['frozen', 'joint', 'decoder_only', 'residual']}
LABELS = {'frozen': 'Frozen', 'joint': 'Full Joint', 'delayed': 'Delayed Joint',
          'decoder_only': 'Decoder-only', 'residual': 'Residual Support'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    receipts = [json.loads(p.read_text()) for p in sorted(root.glob('*.receipt.json'))]
    source = next(r['definition'] for r in receipts if r['definition'].get('diagnostic_baselines'))
    paths = {mode: Path(path) for mode, path in source['diagnostic_baselines'].items()}
    for receipt in receipts:
        path = root / receipt['job']
        if (path / 'evaluation.json').exists():
            report = json.loads((path / 'evaluation.json').read_text())
            paths[report['config']['args']['mode']] = path
    reports = {mode: json.loads((path / 'evaluation.json').read_text()) for mode, path in paths.items()}
    common_keys = ['input_sha256', 'support_count', 'field_progressive_levels', 'loss_weights']
    baseline = reports['frozen']['config']
    for mode, report in reports.items():
        for key in common_keys:
            assert report['config'][key] == baseline[key], (mode, key)
        for key in ('steps', 'queries', 'seed', 'mesh_resolution', 'field_lr', 'support_lr'):
            assert report['config']['args'][key] == baseline['args'][key], (mode, key)
        assert report['status']['steps'] == baseline['args']['steps']
        initial = load_points(paths[mode] / 'support_initial.ply')
        for row in report['rows']:
            if row['kind'] == 'support' and 'normalized_displacement_p95' not in row:
                points = load_points(paths[mode] / f'support_{row["tag"]}.ply')
                row['normalized_displacement_p95'] = float(np.quantile(np.linalg.norm(points - initial, axis=1), .95)
                                                           / report['config']['scale'])
    columns = ['mode', 'kind', 'tag', 'step', 'normalized_cd_l1', 'normalized_accuracy',
               'normalized_coverage', 'normalized_gap_coverage', 'normalized_displacement',
               'normalized_displacement_p95']
    rows = [{**row, 'mode': mode} for mode, report in reports.items() for row in report['rows']]
    with (root / 'diagnostic_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 1024 点适应诊断结果', '',
             '三个新增配置共享上一轮 Frozen / Full Joint 参照。所有误差均按同一观测尺度归一化，越小越好。', '',
             'Decoder-only 仅更新 skip_mlp + disp_mlp；Residual Support 冻结完整网络，只优化选定支持点的坐标残差。', '',
             '位移正则复用原 prior 项，权重为 0.2。未进行额外学习率扫描或选择 GT 最佳中间步。']
    final = {}
    for mode, report in reports.items():
        final[mode] = {row['kind']: row for row in report['rows'] if row['tag'] == 'final'}
    for group, modes in GROUPS.items():
        lines += ['', f'## {group}', '', '| Setting | Support CD | Support GapCoverage | Mesh CD | Mesh GapCoverage |',
                  '|---|---:|---:|---:|---:|']
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for mode in modes:
            if mode not in reports:
                lines.append(f'| {LABELS[mode]} | missing | missing | missing | missing |')
                continue
            support = final[mode].get('support', {})
            mesh = final[mode].get('mesh', {})
            fmt = lambda obj, key: f'{obj[key]:.6f}' if key in obj else 'no mesh'
            lines.append(f'| {LABELS[mode]} | {fmt(support,"normalized_cd_l1")} | {fmt(support,"normalized_gap_coverage")} | '
                         f'{fmt(mesh,"normalized_cd_l1")} | {fmt(mesh,"normalized_gap_coverage")} |')
            support_curve = sorted([r for r in reports[mode]['rows'] if r['kind'] == 'support' and r['tag'] != 'initial'],
                                   key=lambda r: r['step'])
            mesh_curve = sorted([r for r in reports[mode]['rows'] if r['kind'] == 'mesh' and r['step'] > 0],
                                key=lambda r: r['step'])
            for ax, curve, metric in [(axes[0, 0], support_curve, 'normalized_gap_coverage'),
                                      (axes[0, 1], mesh_curve, 'normalized_cd_l1'),
                                      (axes[1, 0], mesh_curve, 'normalized_gap_coverage'),
                                      (axes[1, 1], support_curve, 'normalized_displacement')]:
                ax.plot([r['step'] for r in curve], [r[metric] for r in curve], marker='o', label=LABELS[mode])
        titles = ['Support GapCoverage', 'Mesh CD-L1 (trained checkpoints)',
                  'Mesh GapCoverage (trained checkpoints)', 'Mean support displacement']
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title)
            ax.set_xlabel('Step')
            ax.set_ylabel('Distance / observation scale')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
            if group == 'A_delayed' and 'delayed' in reports:
                ax.axvline(reports['delayed']['config']['args']['warmup_steps'], color='gray', linestyle='--', alpha=.6)
        for suffix in ('png', 'pdf'):
            figure.savefig(root / f'{group}.{suffix}', dpi=180)
        plt.close(figure)
        lines += ['', f'![{group}]({group}.png)']
    audits = {mode: [r for r in report['rows'] if r['kind'] == 'support' and r.get('adaptation_audit')]
              for mode, report in reports.items() if mode in ('delayed', 'decoder_only', 'residual')}
    missing = [mode for mode in LABELS if mode not in reports]
    write_json(root / 'diagnostic_summary.json', {
        'status': 'complete' if not missing else 'incomplete', 'missing': missing, 'final': final,
        'audits': audits, 'shared_baseline_paths': source['diagnostic_baselines'],
        'initialization_checks': {m: r['config'].get('initialization_check') for m, r in reports.items()},
        'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    lines += ['', '## 更新范围核对', '', '| Setting | Frozen state max change | Trainable network parameters | Residual parameters |',
              '|---|---:|---:|---:|']
    for mode in audits:
        audit = final[mode]['support']['adaptation_audit']
        lines.append(f'| {LABELS[mode]} | {audit["frozen_state_max_change"]:.3g} | '
                     f'{audit["trainable_network_parameters"]} | {audit["residual_parameters"]} |')
    lines += ['', '## 支持位移', '', '| Setting | Mean displacement | P95 displacement |', '|---|---:|---:|']
    for mode, result in final.items():
        support = result['support']
        lines.append(f'| {LABELS[mode]} | {support["normalized_displacement"]:.6f} | '
                     f'{support["normalized_displacement_p95"]:.6f} |')
    lines += ['', 'A 同时改变支持更新的启动时刻与次数；C 的坐标参数与网络参数不同，相同学习率不代表相同位移幅度。',
              '结论限于当前目标、预算和单个 airplane。若网格改善但支持覆盖退化，应分开陈述，不能统一称为支持变好。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Diagnostic report generated for {len(reports)} configurations', flush=True)


if __name__ == '__main__':
    main()
