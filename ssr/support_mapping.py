"""Method 3.4: decode feature evidence into explicit 3D surface support.

'Inverse mapping' describes the representation-to-geometry direction, not a
mathematically invertible or bijective network.
"""
import torch
from .representation import LocalSurfaceRepresentation


def decode_surface_support(decoder, representation: LocalSurfaceRepresentation,
                           queries, layout, direct_head=None):
    """Decode offsets, add anchors, and undo each patch's normalization.

    Result coordinates remain in the normalized whole-shape frame used by the
    implicit field. The reconstruction entry point restores world coordinates.
    """
    pos = representation.anchors
    batch, _, anchors = pos.shape
    if queries is not None:
        decoded = decoder.skip_mlp(torch.cat([
            queries.features.reshape(batch, decoder.disp_dim, -1),
            representation.anchor_features.repeat_interleave(decoder.r, dim=-1)], dim=1))
        offsets = torch.tanh(decoder.disp_mlp(decoded))
    else:
        # Existing direct-decoder ablation deliberately skips Section 3.3.
        offsets = torch.tanh(direct_head(representation.local_features)
                             .view(batch, 3, decoder.r, anchors)
                             .permute(0, 1, 3, 2).reshape(batch, 3, -1))
    support = pos.repeat_interleave(decoder.r, dim=-1) + offsets
    patch_support = support.transpose(1, 2) * layout['radii'] + layout['centers']
    return {'points': patch_support.reshape(-1, 3), 'patch_points': patch_support,
            'regularizer': representation.regularizer,
            'kernel_offsets': representation.kernel_offsets,
            'anchor_features': representation.anchor_features,
            'kernel_features': representation.kernel_features}
