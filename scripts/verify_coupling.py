"""Verify that one-way isolates field feedback while updating the complete prior."""
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssr.geometry import patch_layout, write_json
from ssr.support import SurfaceSupportLearner
from ssr.implicit_reconstruction import MultiScaleTriPlaneSDF, reconstruction_loss, total_loss


def main():
    torch.set_num_threads(2)
    torch.manual_seed(21)
    observations = torch.randn(100, 3, device='cuda')
    observations = observations / observations.norm(dim=-1, keepdim=True) * .65
    layout = patch_layout(observations, patch_size=64, overlap=1)
    learner = SurfaceSupportLearner().cuda()
    field = MultiScaleTriPlaneSDF(2048, max_levels=6).cuda()
    parameters = [p for p in learner.parameters() if p.requires_grad]
    result = {}
    values = []
    for feedback in (False, True):
        prediction = learner(layout)
        reference = prediction['points'].detach().clone()
        torch.manual_seed(25)
        losses = reconstruction_loss(field, prediction, observations, reference,
            queries=32, spacing=.1, field_feedback=feedback)
        values.append({k: float(v.detach()) for k, v in losses.items()})
        per_term = {}
        for name in ('pull', 'surface'):
            gradients = torch.autograd.grad(losses[name], parameters, retain_graph=True, allow_unused=True)
            magnitude = sum(float(g.abs().sum()) for g in gradients if g is not None)
            assert (magnitude > 0) == feedback, (name, feedback, magnitude)
            field_gradient, = torch.autograd.grad(losses[name], field.lin8.weight_v, retain_graph=True)
            assert torch.isfinite(field_gradient).all() and field_gradient.abs().sum() > 0
            per_term[name] = {'support_gradient_l1': magnitude,
                              'field_gradient_l1': float(field_gradient.abs().sum())}
        learner.zero_grad(set_to_none=True)
        field.zero_grad(set_to_none=True)
        total_loss(losses).backward()
        groups = {'encoding': learner.backbone.encoder, 'representation': learner.backbone.decoder.rem,
                  'queries': learner.backbone.decoder.kgm, 'decoding': learner.backbone.decoder.disp_mlp}
        gradients = {name: sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
                     for name, module in groups.items()}
        assert all(v > 0 for v in gradients.values()), gradients
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
        result['joint' if feedback else 'oneway'] = {'field_terms': per_term, 'whole_prior_gradient_l1': gradients}
    assert values[0] == values[1], 'Detaching feedback changed forward loss values'
    frozen = SurfaceSupportLearner().cuda().requires_grad_(False)
    with torch.no_grad():
        before = frozen(layout)['points'].clone()
    optimizer = torch.optim.Adam(field.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    field(observations).abs().mean().backward()
    optimizer.step()
    assert torch.equal(before, frozen(layout)['points'])
    result.update(status='passed', same_forward_losses=True, frozen_support_unchanged=True)
    write_json(ROOT / 'verification_coupling.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
