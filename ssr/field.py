"""Original P3D-Mesh multiscale hash tri-plane/grid SDF, without GSHE fusion.

Support is a differentiable training target, not an additional field input.
"""
from types import SimpleNamespace
from .p3d_sdf import SDFNetwork


class MultiScaleTriPlaneSDF(SDFNetwork):
    def __init__(self, point_size):
        config = SimpleNamespace(d_in=3, d_out=1, d_hidden=256, n_layers=8,
                                 skip_in=[4], multires=8, geometric_init=True,
                                 weight_norm=True, inside_outside=False)
        super().__init__(point_size, config)
        self.step = 0

    def set_step(self, step):
        self.step = int(step)

    def forward(self, x, step=None):
        return super().forward(x, self.step if step is None else step)

    def value_gradient(self, x, create_graph=True):
        import torch
        x = x.requires_grad_(True)
        value = self(x)
        gradient = torch.autograd.grad(value.sum(), x, create_graph=create_graph, retain_graph=create_graph)[0]
        return value, gradient
