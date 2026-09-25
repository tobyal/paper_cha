"""Method 3.1 orchestrator: representation -> queries -> explicit support.

Sections 3.2--3.4 live in representation, query_sampling and support_mapping.
The native backbone owns all weights so existing checkpoints remain compatible.
"""
import hashlib
import sys
from pathlib import Path
import torch
from torch import nn
from .representation import build_local_representation
from .query_sampling import sample_surface_queries
from .support_mapping import decode_surface_support

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
        # 3.2: anchor encoding and adaptive local surface representation.
        representation = build_local_representation(self.backbone, layout['patches'])
        decoder = self.backbone.decoder
        # 3.3: read surface evidence (bypassed only by the direct ablation).
        queries = (sample_surface_queries(decoder, representation)
                   if self.decoder_kind == 'query' else None)
        # 3.4: decode evidence into support coordinates, retaining gradients.
        return decode_surface_support(decoder, representation, queries, layout,
                                      getattr(self, 'direct', None))
