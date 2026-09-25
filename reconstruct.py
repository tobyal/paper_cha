"""Frozen, one-way and joint support-field optimization, plus fixed-source controls."""
import argparse
import hashlib
import os
import time
from pathlib import Path
import numpy as np
import torch
from ssr.geometry import load_points, normalize, patch_layout, fps_indices, save_points, write_json
from ssr.support import SurfaceSupportLearner
from ssr.implicit_reconstruction import MultiScaleTriPlaneSDF, reconstruction_loss, total_loss, DEFAULT_WEIGHTS
from ssr.export import export_mesh


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--observations', type=int, default=None, help='Required when constructing sparse input from a mesh')
    p.add_argument('--mode', choices=['joint', 'oneway', 'frozen', 'raw', 'external'], default='joint')
    p.add_argument('--external-support', help='NTPS/BSDF support in the same world coordinates as the input')
    p.add_argument('--initialization', choices=['pretrained', 'random'], default='pretrained')
    p.add_argument('--checkpoint')
    p.add_argument('--no-deform', action='store_true')
    p.add_argument('--decoder', choices=['query', 'direct'], default='query')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--minutes', type=float, default=30., help='Optimization wall limit; 0 uses only the step budget')
    p.add_argument('--queries', type=int, default=512)
    p.add_argument('--support-budget', type=int, default=4096)
    p.add_argument('--require-support-budget', action='store_true', help='Reject insufficient support (Raw is exempt)')
    p.add_argument('--field-max-levels', type=int, default=None, help='Fix progressive encoding limit across support sources')
    p.add_argument('--field-lr', type=float, default=1e-3)
    p.add_argument('--support-lr', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=21)
    p.add_argument('--device', default='cuda')
    p.add_argument('--mesh-resolution', type=int, default=128)
    p.add_argument('--snapshot-steps', type=int, nargs='*', default=[100, 500, 1000, 2000])
    p.add_argument('--early-stop', action='store_true', help='Stop on training-loss plateau after at least 500 updates')
    p.add_argument('--normals', help='Optional observed normals [N,3] .npy aligned with ALL input points; disallows subsampling')
    return p


def main(args=None):
    args = parser().parse_args(args)
    if args.steps < 1 or args.minutes < 0 or args.queries < 8 or args.support_budget < 16:
        raise ValueError('Invalid optimization budget')
    if args.mode == 'external' and not args.external_support:
        raise ValueError('External mode requires --external-support')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite an existing run: {out}')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    source = load_points(args.input, args.observations, args.seed)
    normalized, center, scale = normalize(source)
    observations = torch.tensor(normalized, device=args.device)
    normals = None
    if args.normals:
        if args.observations is not None:
            raise ValueError('Normals require preselected observations; omit --observations')
        normals = torch.tensor(np.load(args.normals), dtype=torch.float32, device=args.device)
        if normals.shape != observations.shape:
            raise ValueError('Normal/observation count mismatch')
    save_points(out / 'input.ply', source)
    np.savez(out / 'input.npz', p=source, center=center, scale=scale)
    with torch.no_grad():
        distance = torch.cdist(observations, observations)
        distance.fill_diagonal_(float('inf'))
        spacing = float(distance.min(1).values.median())
    learner, layout, selected, reference = None, None, None, None
    adaptive = args.mode in ('joint', 'oneway')
    if args.mode in ('joint', 'oneway', 'frozen'):
        learner = SurfaceSupportLearner(args.initialization, args.checkpoint, not args.no_deform, args.decoder).to(args.device)
        layout = patch_layout(observations)
        with torch.no_grad():
            initial = learner(layout)
            selected = fps_indices(initial['points'], args.support_budget)
            reference = initial['points'][selected].clone()
        if args.mode == 'frozen':
            learner.requires_grad_(False)
        provenance = learner.provenance
        raw_count = len(initial['points'])
    else:
        array = source if args.mode == 'raw' else load_points(args.external_support)
        fixed = torch.tensor((array - center) / scale, device=args.device)
        selected = fps_indices(fixed, args.support_budget)
        reference = fixed[selected].clone()
        raw_count = len(fixed)
        provenance = {'initialization': 'none', 'source': args.input if args.mode == 'raw' else args.external_support}
    unique_count = len(torch.unique(reference, dim=0))
    if args.require_support_budget and args.mode != 'raw' and unique_count != args.support_budget:
        raise ValueError(f'Expected {args.support_budget} distinct supports, got {unique_count}')
    # Reset before creating the common implicit decoder for all ablations.
    torch.manual_seed(args.seed + 1)
    field = MultiScaleTriPlaneSDF(point_size=len(reference), max_levels=args.field_max_levels).to(args.device)
    groups = [{'params': field.parameters(), 'lr': args.field_lr}]
    if learner is not None and adaptive:
        groups.append({'params': learner.parameters(), 'lr': args.support_lr})
    optimizer = torch.optim.Adam(groups)
    metadata = {'args': vars(args), 'provenance': provenance, 'loss_weights': DEFAULT_WEIGHTS,
                'source_snapshot_sha256': os.environ.get('PAPER_CHA_SOURCE_SHA256'),
                'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                'observations': len(source), 'support_count': len(reference), 'raw_support_count': raw_count,
                'unique_support_count_initial': unique_count,
                'support_network_trainable': adaptive, 'field_to_support_feedback': args.mode == 'joint',
                'prior_scope': 'complete encoding + local representation + query sampling + decoding network',
                'gradient_clipping': 'each optimizer parameter group independently, norm 10',
                'center': center.tolist(), 'scale': scale, 'spacing': spacing,
                'gt_used_for_reconstruction': False, 'field': 'P3D-Mesh multiscale hash tri-plane + 3D hash-grid, without GSHE',
                'field_fusion': 'xy + yz + xz + grid',
                'support_role': 'dynamic pull targets and zero-level constraints; no support feature input',
                'field_version': 2, 'field_progressive_levels': field.plane_encoding.max_levels}
    write_json(out / 'config.json', metadata)
    save_points(out / 'support_initial.ply', reference * scale + torch.as_tensor(center, device=args.device))

    def predict():
        if adaptive:
            prediction = learner(layout)
            prediction['points'] = prediction['points'][selected]
            return prediction
        return {'points': reference, 'regularizer': reference.new_zeros(())}

    def snapshot(step, final=False):
        with torch.no_grad():
            support = predict()['points'].detach()
        tag = 'final' if final else f'step_{step:06d}'
        save_points(out / f'support_{tag}.ply', support * scale + torch.as_tensor(center, device=support.device))
        result = export_mesh(field, out / f'mesh_{tag}.ply', center, scale, args.mesh_resolution)
        write_json(out / f'snapshot_{tag}.json', {
            'step': step, 'seconds': time.monotonic() - start, 'mesh': result,
            'support_displacement_normalized': float((support - reference).norm(dim=-1).mean()),
            'support_displacement_world': float((support - reference).norm(dim=-1).mean()) * scale,
            'support_count': len(support), 'point_correspondence': 'fixed initial FPS indices'})
        torch.save({'field': field.state_dict(), 'support_network': learner.state_dict() if learner else None,
                    'support': support.cpu(), 'observations': observations.cpu(), 'selected_indices': selected.cpu(),
                    'normalization': {'center': center, 'scale': scale}, 'step': step, 'field_encoding_step': field.step,
                    'optimizer': optimizer.state_dict(), 'config': metadata}, out / f'{tag}.pt')
        return result

    start = time.monotonic()
    stop = 'step_budget'
    ema, best, stale = None, float('inf'), 0
    final_step = 0
    try:
        if 0 in args.snapshot_steps:
            snapshot(0)
        with (out / 'training.jsonl').open('w', encoding='utf-8') as log:
            for step in range(1, args.steps + 1):
                if args.minutes and time.monotonic() - start >= args.minutes * 60:
                    stop = 'time_budget'
                    break
                optimizer.zero_grad(set_to_none=True)
                field.set_step(step - 1)
                prediction = predict()
                # Random initialization has no learned geometric prior to preserve.
                prior = reference if args.initialization == 'pretrained' and adaptive else None
                losses = reconstruction_loss(field, prediction, observations, prior, args.queries, spacing, normals,
                                             field_feedback=args.mode == 'joint')
                loss = total_loss(losses)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite objective')
                loss.backward()
                parameters = [p for group in optimizer.param_groups for p in group['params']]
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError('Nonfinite gradient')
                # Shared clipping would let support gradients rescale field updates.
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], 10.)
                optimizer.step()
                final_step = step
                value = float(loss.detach())
                ema = value if ema is None else .95 * ema + .05 * value
                row = {'step': step, 'seconds': time.monotonic() - start, 'loss': value, 'ema': ema,
                       **{name: float(v.detach()) for name, v in losses.items()}}
                log.write(__import__('json').dumps(row) + '\n')
                log.flush()
                if step == 1 or step % 25 == 0:
                    print(row, flush=True)
                    write_json(out / 'status.json', {'status': 'running', **row})
                if step in args.snapshot_steps:
                    snapshot(step)
                if args.early_stop and step % 50 == 0 and step >= 500:
                    if ema < best * .997:
                        best, stale = ema, 0
                    else:
                        stale += 1
                    if stale >= 10:
                        stop = 'training_loss_plateau'
                        break
        optimization_seconds = time.monotonic() - start
        mesh = snapshot(final_step, final=True)
        write_json(out / 'status.json', {'status': 'complete', 'steps': final_step, 'stop': stop,
                                        'optimization_seconds': optimization_seconds, 'total_seconds': time.monotonic() - start,
                                        'mesh': mesh})
    except Exception as error:
        write_json(out / 'status.json', {'status': 'failed', 'step': final_step, 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
