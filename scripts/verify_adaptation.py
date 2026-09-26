"""Exercise delayed feedback, decoder boundaries and residual patch gradients on CUDA."""
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssr.adaptation import SupportAdaptation
from ssr.support import SurfaceSupportLearner
from ssr.geometry import patch_layout, fps_indices, write_json
from ssr.implicit_reconstruction import MultiScaleTriPlaneSDF, reconstruction_loss, total_loss


def main():
    torch.set_num_threads(2)
    torch.manual_seed(21)
    points = torch.randn(100, 3, device='cuda') * .2
    layout = patch_layout(points, patch_size=64, overlap=1)
    results = {}
    for mode in ('delayed', 'decoder_only', 'residual'):
        learner = SurfaceSupportLearner().cuda()
        with torch.no_grad():
            initial = learner(layout)
            selected = fps_indices(initial['points'], 128)
        controller = SupportAdaptation(learner, layout, initial, selected, mode, warmup_steps=2)
        assert torch.equal(controller.predict()['points'], controller.reference)
        field = MultiScaleTriPlaneSDF(2048, max_levels=6).cuda()
        optimizer = torch.optim.Adam([{'params': field.parameters(), 'lr': 1e-3},
                                      {'params': controller.parameters(), 'lr': 1e-5}])
        rows = []
        for step in range(1, 5):
            active = controller.begin_step(step)
            optimizer.zero_grad(set_to_none=True)
            field.set_step(step - 1)
            prediction = controller.predict()
            if controller.delta is not None:
                assert torch.equal(prediction['points'], prediction['patch_points'].reshape(-1, 3)[selected])
                patch_gradient, = torch.autograd.grad(prediction['patch_points'].sum(), controller.delta, retain_graph=True)
                assert torch.equal(patch_gradient, torch.ones_like(controller.delta))
            losses = reconstruction_loss(field, prediction, points, controller.reference if active else None,
                                          queries=32, spacing=.1, field_feedback=active)
            if mode == 'residual':
                assert torch.allclose(losses['prior'], controller.delta.square().sum(-1).mean())
            total_loss(losses).backward()
            support_grad = sum(float(p.grad.abs().sum()) for p in controller.parameters() if p.grad is not None)
            assert (support_grad > 0) == active
            for group in optimizer.param_groups:
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in group['params'])
                torch.nn.utils.clip_grad_norm_(group['params'], 10.)
            optimizer.step()
            assert int(optimizer.state[field.lin8.weight_v]['step']) == step, 'Field Adam state reset'
            assert not learner.backbone.decoder.rem.kp_pos.requires_grad
            assert not learner.backbone.decoder.kgm.query_pos.requires_grad
            if not active:
                assert torch.equal(controller.predict()['points'], controller.reference)
            rows.append({'step': step, 'active': active, 'support_gradient_l1': support_grad})
        audit = controller.audit()
        assert audit['frozen_state_max_change'] == 0, audit
        if mode == 'decoder_only':
            names = [name for name, p in learner.named_parameters() if p.requires_grad]
            assert all(name.startswith(('backbone.decoder.skip_mlp.', 'backbone.decoder.disp_mlp.')) for name in names)
            assert audit['network_state_max_change'] > 0
            assert all(value == 0 for value in audit['representation_max_changes'].values())
        elif mode == 'residual':
            assert audit['network_state_max_change'] == 0
            assert controller.delta.abs().max() > 0
        else:
            assert [row['active'] for row in rows] == [False, False, True, True]
        results[mode] = {'steps': rows, 'audit': audit}
        del learner, field, optimizer, controller, initial, prediction, losses
    write_json(ROOT / 'verification_adaptation.json', {'status': 'passed', 'results': results})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
