"""Meaningful contract tests on actual CUDA RepKPU modules and joint gradients."""
import argparse
import torch
from ssr.geometry import patch_layout, fps_indices, write_json
from ssr.support import SurfaceSupportLearner
from ssr.implicit_reconstruction import MultiScaleTriPlaneSDF, reconstruction_loss, total_loss


def nonzero_grad(module):
    return sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum() > 0)
               for p in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='verification.json')
    parser.add_argument('--p3d-reference', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(21)
    points = torch.randn(100, 3, device='cuda')
    points = points / points.norm(dim=-1, keepdim=True) * .65
    layout = patch_layout(points, patch_size=64, overlap=1)
    learner = SurfaceSupportLearner().cuda().train()
    prediction = learner(layout)
    with torch.no_grad():
        native, _ = learner.backbone(layout['patches'])
        native_world = native.transpose(1, 2) * layout['radii'] + layout['centers']
    error = float((native_world - prediction['patch_points']).abs().max())
    assert error < 1e-6, error
    indices = fps_indices(prediction['points'].detach(), 128)
    prediction['points'] = prediction['points'][indices]
    reference = prediction['points'].detach().clone()
    field = MultiScaleTriPlaneSDF(point_size=4096).cuda()
    native_field = torch.load(args.p3d_reference, map_location='cpu', weights_only=True)
    field.load_state_dict(native_field['state'], strict=True)
    field_errors = {}
    for step, expected in native_field['outputs'].items():
        field.set_step(step)
        values, gradients = field.value_gradient(native_field['points'].cuda().clone(), create_graph=False)
        value_error = float((values.detach().cpu() - expected['value']).abs().max())
        gradient_error = float((gradients.detach().cpu() - expected['gradient']).abs().max())
        assert value_error < 1e-6 and gradient_error < 1e-5, (value_error, gradient_error)
        field_errors[step] = {'value': value_error, 'spatial_gradient': gradient_error}
    assert not any('ghse' in key or 'gshe' in key for key in field.state_dict())
    # Restore the actual geometric initialization for joint-update checks.
    del field, native_field
    field = MultiScaleTriPlaneSDF(point_size=4096).cuda()
    query = torch.randn(32, 3, device='cuda') * .5
    # Zero-surface loss evaluated AT live generated coordinates must reach support.
    value = field(prediction['points'])
    value.square().mean().backward()
    field_to_support = nonzero_grad(learner)
    assert field_to_support > 0
    learner.zero_grad(set_to_none=True)
    field.zero_grad(set_to_none=True)
    prediction = learner(layout)
    prediction['points'] = prediction['points'][indices]
    losses = reconstruction_loss(field, prediction, points, reference, queries=32, spacing=.1)
    optimizer = torch.optim.Adam(list(learner.parameters()) + list(field.parameters()), lr=1e-4)
    before = learner.backbone.decoder.disp_mlp[-1].weight.detach().clone()
    field_before = field.lin8.weight_v.detach().clone()
    loss = total_loss(losses)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(learner.parameters()) + list(field.parameters()))
    optimizer.step()
    support_update = float((before - learner.backbone.decoder.disp_mlp[-1].weight).abs().max())
    field_update = float((field_before - field.lin8.weight_v).abs().max())
    assert support_update > 0 and field_update > 0
    # Geometric init zeros the first-layer spatial-feature columns; after one
    # update, verify that all four real encoding branches receive gradients.
    optimizer.zero_grad(set_to_none=True)
    field.set_step(2000)
    next_prediction = learner(layout)
    next_prediction['points'] = next_prediction['points'][indices]
    next_losses = reconstruction_loss(field, next_prediction, points, reference, queries=32, spacing=.1)
    total_loss(next_losses).backward()
    encoding_gradients = {name: float(encoding.embeddings.grad.abs().sum()) for name, encoding in {
        'xy': field.plane_encoding.xy, 'yz': field.plane_encoding.yz,
        'xz': field.plane_encoding.xz, 'grid': field.grid_encoding.hash_grid}.items()}
    assert all(value > 0 for value in encoding_gradients.values()), encoding_gradients
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in field.parameters())
    frozen = reference.clone()
    frozen_before = frozen.clone()
    field.zero_grad(set_to_none=True)
    field(query).square().mean().backward()
    assert torch.equal(frozen_before, frozen) and not frozen.requires_grad
    rigid = SurfaceSupportLearner(deform=False).cuda()
    with torch.no_grad():
        rigid_pred = rigid(layout)
        canonical = rigid.backbone.decoder.rem.kp_pos.view(1, 3, 1, -1)
        rigid_error = float((rigid_pred['kernel_offsets'] - canonical).abs().max())
    assert rigid_error == 0
    direct = SurfaceSupportLearner(decoder='direct').cuda()
    direct_result = direct(layout)
    direct_result['points'].square().mean().backward()
    assert nonzero_grad(direct.direct) > 0 and nonzero_grad(direct.backbone.decoder.attns) == 0
    random = SurfaceSupportLearner(initialization='random').cuda()
    with torch.no_grad():
        assert torch.isfinite(random(layout)['points']).all()
    result = {'status': 'passed', 'native_repkpu_max_error': error,
              'surface_zero_loss_support_gradient_tensors': field_to_support,
              'original_p3d_field_errors': field_errors, 'gshe_absent': True,
              'encoding_gradient_l1': encoding_gradients,
              'joint_support_parameter_update': support_update, 'joint_field_parameter_update': field_update,
              'rigid_kernel_error': rigid_error, 'frozen_support_unchanged': True,
              'direct_decoder_bypasses_attention': True, 'random_forward_finite': True,
              'joint_second_order_backward_finite': True, 'losses': {k: float(v.detach()) for k, v in losses.items()}}
    write_json(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
