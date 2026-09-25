"""Method 3.5: support-guided implicit surface reconstruction.

Original P3D-Mesh multiscale hash tri-plane/grid SDF, without GSHE fusion.

Support is a differentiable training target, not an additional field input.
"""
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from .geometry import directed_distance, nearest_indices
from .p3d_sdf import SDFNetwork


class MultiScaleTriPlaneSDF(SDFNetwork):
    def __init__(self, point_size, max_levels=None):
        config = SimpleNamespace(d_in=3, d_out=1, d_hidden=256, n_layers=8,
                                 skip_in=[4], multires=8, geometric_init=True,
                                 weight_norm=True, inside_outside=False)
        super().__init__(point_size, config)
        if max_levels is not None:
            if not 1 <= max_levels <= 16:
                raise ValueError('Field max_levels must be between 1 and 16')
            self.plane_encoding.max_levels = max_levels
            self.grid_encoding.max_levels = max_levels
        self.step = 0

    def set_step(self, step):
        self.step = int(step)

    def forward(self, x, step=None):
        return super().forward(x, self.step if step is None else step)

    def value_gradient(self, x, create_graph=True):
        x = x.requires_grad_(True)
        value = self(x)
        gradient = torch.autograd.grad(value.sum(), x, create_graph=create_graph, retain_graph=create_graph)[0]
        return value, gradient


DEFAULT_WEIGHTS = {'pull': 1., 'surface': 0.3, 'observation_sdf': 0.3, 'observation_cover': 1.,
                   'eikonal': 0.1, 'prior': 0.2, 'repulsion': 0.05, 'kernel': 0.001, 'normal': 0.1}


def reconstruction_loss(field, prediction, observations, reference, queries=512, spacing=0.03,
                        normals=None, field_feedback=True):
    """One-way disables only field-to-support feedback, not support adaptation.

    Query locations are detached in both modes. Observation coverage, prior,
    repulsion and representation regularization always retain live support.
    """
    support = prediction['points']
    field_support = support if field_feedback else support.detach()
    count = min(queries, len(support))
    selected = torch.randperm(len(support), device=support.device)[:count]
    selected_support = field_support[selected]
    near_count = queries * 3 // 4
    centers = torch.cat([observations, support.detach()], 0)
    near = centers[torch.randint(len(centers), (near_count,), device=support.device)]
    near = near + torch.randn_like(near) * max(spacing, 0.015)
    uniform = torch.rand(queries - near_count, 3, device=support.device) * 2.4 - 1.2
    x = torch.cat([near, uniform], 0).detach().requires_grad_(True)
    values, gradients = field.value_gradient(x)
    pulled = x - values * F.normalize(gradients, dim=-1, eps=1e-8)
    target = field_support[nearest_indices(x, field_support)[:, 0]]
    # Live target coordinates allow pull loss to adapt the support learner.
    losses = {'pull': ((pulled - target).square().sum(-1) + 1e-12).sqrt().mean(),
              'surface': field(selected_support).abs().mean(),
              'observation_sdf': field(observations).abs().mean(),
              # One-way: observations must be represented, while missing-region
              # support is not forced to collapse back onto the sparse samples.
              'observation_cover': directed_distance(observations, support),
              'eikonal': (gradients.norm(dim=-1) - 1).square().mean(),
              'prior': (support - reference).square().sum(-1).mean() if reference is not None else support.new_zeros(()),
              'kernel': torch.as_tensor(prediction['regularizer'], device=support.device),
              'normal': support.new_zeros(())}
    if normals is not None:
        _, observation_grad = field.value_gradient(observations.detach().clone())
        losses['normal'] = (1 - F.cosine_similarity(observation_grad, normals, dim=-1).abs()).mean()
    patches = prediction.get('patch_points')
    if patches is not None:
        groups = patches.reshape(-1, 4, 3)
        distances = ((groups[:, :, None] - groups[:, None, :]).square().sum(-1) + 1e-12).sqrt()
        mask = ~torch.eye(4, dtype=torch.bool, device=support.device)
        losses['repulsion'] = (F.relu(0.25 * spacing - distances[:, mask]) / max(spacing, 1e-5)).square().mean()
    else:
        losses['repulsion'] = support.new_zeros(())
    return losses


def total_loss(losses, weights=None):
    weights = DEFAULT_WEIGHTS if weights is None else weights
    return sum(weights[key] * value for key, value in losses.items())
