"""Three diagnostics of support adaptation, preserving the pretrained prior."""
import torch
from .representation import build_local_representation
from .query_sampling import sample_surface_queries


DIAGNOSTIC_MODES = ('delayed', 'decoder_only', 'residual')


class SupportAdaptation:
    def __init__(self, learner, layout, initial, selected, mode, warmup_steps=3000):
        if mode not in DIAGNOSTIC_MODES:
            raise ValueError(mode)
        self.learner, self.layout, self.selected = learner, layout, selected
        self.mode, self.warmup_steps = mode, warmup_steps
        self.initial = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in initial.items()}
        self.reference = self.initial['points'][selected].clone()
        self.original_trainable = [name for name, p in learner.named_parameters() if p.requires_grad]
        self.original_state = {k: v.detach().clone() for k, v in learner.state_dict().items()}
        learner.requires_grad_(False)
        self.delta = None
        self.active = mode != 'delayed'
        if mode == 'decoder_only':
            # The native decoder object also owns REM/KGM/attention: keep these frozen.
            learner.backbone.decoder.skip_mlp.requires_grad_(True)
            learner.backbone.decoder.disp_mlp.requires_grad_(True)
        elif mode == 'residual':
            self.delta = torch.nn.Parameter(torch.zeros_like(self.reference))
        self.initial_features = self._features() if mode == 'decoder_only' else None

    def parameters(self):
        if self.delta is not None:
            return [self.delta]
        if self.mode == 'delayed':
            # Adam may hold frozen parameters; their state starts at first update.
            # Restore only originally trainable parameters, never fixed kernel/query positions.
            named = dict(self.learner.named_parameters())
            return [named[name] for name in self.original_trainable]
        return [p for p in self.learner.parameters() if p.requires_grad]

    def begin_step(self, step):
        if self.mode == 'delayed' and not self.active and step > self.warmup_steps:
            named = dict(self.learner.named_parameters())
            for name in self.original_trainable:
                named[name].requires_grad_(True)
            self.active = True
        return self.active

    def predict(self):
        if not self.active:
            return {'points': self.reference, 'regularizer': self.reference.new_zeros(())}
        if self.delta is not None:
            # Scatter the SAME residual into the patch layout used by the existing
            # repulsion objective; unselected support locations remain fixed.
            patches = self.initial['patch_points']
            updated = patches.reshape(-1, 3).index_add(0, self.selected, self.delta)
            return {**self.initial, 'points': self.reference + self.delta,
                    'patch_points': updated.reshape_as(patches)}
        result = self.learner(self.layout)
        result['points'] = result['points'][self.selected]
        return result

    @torch.no_grad()
    def _features(self):
        rep = build_local_representation(self.learner.backbone, self.layout['patches'])
        queries = sample_surface_queries(self.learner.backbone.decoder, rep)
        return {'anchor': rep.anchor_features, 'kernel_offsets': rep.kernel_offsets,
                'kernel_features': rep.kernel_features, 'query_features': queries.features}

    @torch.no_grad()
    def audit(self):
        trainable = {name for name, p in self.learner.named_parameters() if p.requires_grad}
        changes = {name: float((value - self.original_state[name]).abs().max())
                   for name, value in self.learner.state_dict().items()}
        result = {'active': self.active, 'mode': self.mode,
                  'trainable_network_parameters': sum(p.numel() for p in self.learner.parameters() if p.requires_grad),
                  'residual_parameters': 0 if self.delta is None else self.delta.numel(),
                  'frozen_state_max_change': max((v for k, v in changes.items() if k not in trainable), default=0.),
                  'network_state_max_change': max(changes.values(), default=0.)}
        if self.initial_features is not None:
            result['representation_max_changes'] = {
                k: float((v - self.initial_features[k]).abs().max()) for k, v in self._features().items()}
        return result
