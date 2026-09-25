"""Compare Method reorganization against the committed pre-refactor implementation.

Run from any directory; the reference is loaded from this repository's Git
history, independently of the current Section 3.2--3.5 helpers.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import types
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssr.geometry import patch_layout, write_json
from ssr.support import SurfaceSupportLearner
from ssr.implicit_reconstruction import MultiScaleTriPlaneSDF, reconstruction_loss, total_loss


def baseline_module(revision, name):
    source = subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', f'{revision}:ssr/{name}.py'], text=True)
    module = types.ModuleType(f'ssr._before_{name}')
    module.__file__ = str(ROOT / 'ssr' / f'{name}.py')
    module.__package__ = 'ssr'
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def compare(a, b):
    assert a.shape == b.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), float((a - b).abs().max())
    return float((a - b).abs().max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='3c8c9f4bce4ff2e754cee3d8fa9c1f5a08479b87')
    parser.add_argument('--output', default='verification_method_refactor.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(21)
    old_support = baseline_module(args.reference, 'support')
    old_objective = baseline_module(args.reference, 'objective')
    points = torch.randn(100, 3, device='cuda')
    points = points / points.norm(dim=-1, keepdim=True) * .65
    layout = patch_layout(points, patch_size=64, overlap=1)
    variants = {}
    for name, options in [('query', {}), ('rigid', {'deform': False}),
                          ('direct', {'decoder': 'direct'}),
                          ('random', {'initialization': 'random'})]:
        old = old_support.SurfaceSupportLearner(**options).cuda().train()
        new = SurfaceSupportLearner(**options).cuda().train()
        new.load_state_dict(old.state_dict(), strict=True)
        assert list(old.state_dict()) == list(new.state_dict())
        assert list(dict(old.named_parameters())) == list(dict(new.named_parameters()))
        a, b = old(layout), new(layout)
        assert a.keys() == b.keys()
        output_errors = {key: compare(a[key], b[key]) for key in a}
        (a['points'].square().mean() + .001 * a['regularizer']).backward()
        (b['points'].square().mean() + .001 * b['regularizer']).backward()
        gradient_errors = {}
        for (key, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
            assert (p.grad is None) == (q.grad is None), key
            if p.grad is not None:
                gradient_errors[key] = compare(p.grad, q.grad)
        variants[name] = {'state_items': len(new.state_dict()),
                          'output_errors': output_errors,
                          'gradient_max_error': max(gradient_errors.values()),
                          'compared_gradient_tensors': len(gradient_errors)}
        print(name, variants[name], flush=True)
        del old, new, a, b

    # The old import paths remain aliases of the canonical Section 3.5 API.
    from ssr.field import MultiScaleTriPlaneSDF as CompatibleField
    from ssr.objective import reconstruction_loss as compatible_loss
    assert CompatibleField is MultiScaleTriPlaneSDF
    assert compatible_loss is reconstruction_loss
    field = MultiScaleTriPlaneSDF(4096).cuda()
    support = points.repeat_interleave(4, 0) + torch.randn(400, 3, device='cuda') * .02
    loss_results = []
    for objective, aggregate in [(old_objective.reconstruction_loss, old_objective.total_loss),
                                 (reconstruction_loss, total_loss)]:
        live = support.detach().clone().requires_grad_(True)
        prediction = {'points': live, 'patch_points': live.reshape(1, -1, 3),
                      'regularizer': live.new_tensor(.1)}
        torch.manual_seed(27)
        losses = objective(field, prediction, points, support, queries=32, spacing=.1,
                           normals=points / points.norm(dim=-1, keepdim=True))
        gradient, = torch.autograd.grad(aggregate(losses), live)
        loss_results.append(({key: value.detach() for key, value in losses.items()}, gradient))
    loss_errors = {key: compare(loss_results[0][0][key], loss_results[1][0][key])
                   for key in loss_results[0][0]}
    result = {'status': 'passed', 'reference_commit': args.reference, 'variants': variants,
              'loss_errors': loss_errors,
              'loss_support_gradient_error': compare(loss_results[0][1], loss_results[1][1]),
              'compatibility_imports': True,
              'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted((ROOT / 'ssr').glob('*.py'))}}
    write_json(args.output, result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
