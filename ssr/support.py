"""Real RepKPU components; no placeholder encoders or renamed MLP kernels."""
import hashlib
import sys
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/repkpu'))
from models.repkpu import RepKPU
from cfgs.upsampling.pugan_args import parse_pugan_args


def native_config():
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        return parse_pugan_args()
    finally:
        sys.argv = saved


class SurfaceSupportLearner(nn.Module):
    def __init__(self, initialization='pretrained', checkpoint=None, deform=True, decoder='query'):
        super().__init__()
        self.backbone = RepKPU(native_config())
        self.decoder_kind = decoder
        self.deform = deform
        self.provenance = {'initialization': initialization, 'deformation': deform, 'decoder': decoder}
        if initialization == 'pretrained':
            checkpoint = Path(checkpoint or ROOT / 'weights/repkpu_pugan.pth')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            # Native official state or a checkpoint produced by train_prior.py.
            state = state.get('backbone', state)
            self.backbone.load_state_dict(state, strict=True)
            self.provenance.update(checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), state_items=len(state))
        if not deform:
            # Keep the official relative-coordinate calculation and receptive radius.
            # Switching REM.is_deform=False directly changes that path in upstream.
            self.backbone.decoder.rem.deform_conv.register_forward_hook(lambda module, args, out: torch.zeros_like(out))
        if decoder == 'direct':
            self.direct = nn.Sequential(nn.Conv1d(64, 128, 1), nn.ReLU(), nn.Conv1d(128, 12, 1))
            self.provenance['new_random_head'] = True
        # Single-shape optimization uses fixed pretrained BN statistics. Affine
        # parameters and all other backbone weights still receive gradients.
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, layout):
        pos = layout['patches']
        batch, _, anchors = pos.shape
        feat = self.backbone.encoder(pos)
        decoder = self.backbone.decoder
        local, kernels, regularizer = decoder.rem(pos, feat)
        if self.decoder_kind == 'query':
            queries = decoder.kgm(pos, local)
            q, k = decoder.projector(queries[1]), decoder.projector(kernels[1])
            for attention in decoder.attns:
                q = attention(q, k)
            decoded = decoder.skip_mlp(torch.cat([
                q.reshape(batch, decoder.disp_dim, -1),
                feat.repeat_interleave(decoder.r, dim=-1)], dim=1))
            offsets = torch.tanh(decoder.disp_mlp(decoded))
        else:
            offsets = torch.tanh(self.direct(local).view(batch, 3, decoder.r, anchors).permute(0, 1, 3, 2).reshape(batch, 3, -1))
        support = pos.repeat_interleave(decoder.r, dim=-1) + offsets
        patch_support = support.transpose(1, 2) * layout['radii'] + layout['centers']
        return {'points': patch_support.reshape(-1, 3), 'patch_points': patch_support,
                'regularizer': regularizer, 'kernel_offsets': kernels[0],
                'anchor_features': feat, 'kernel_features': kernels[1]}
