"""Method 3.2: construct anchor-based local geometric representations."""
from dataclasses import dataclass
from torch import Tensor


@dataclass
class LocalSurfaceRepresentation:
    """Geometry and features in normalized patch coordinates (B patches, N anchors).

    This is a spatially anchored learned representation, not a coordinate-free
    latent manifold. Tensors retain their original computation graphs.
    """
    anchors: Tensor             # [B, 3, N]
    anchor_features: Tensor     # [B, 64, N]
    local_features: Tensor      # [B, 64, N]
    kernel_offsets: Tensor      # [B, 3, N, M], relative to each anchor
    kernel_features: Tensor     # [B, 128, N, M]
    regularizer: Tensor


def build_local_representation(backbone, anchors):
    """3.2.1 geometry encoding, followed by 3.2.2 adaptive local bases.

    Parameters remain owned by the native backbone, preserving checkpoint keys
    and optimizer parameter order. No new trainable modules are introduced.
    """
    features = backbone.encoder(anchors)
    local, kernels, regularizer = backbone.decoder.rem(anchors, features)
    return LocalSurfaceRepresentation(
        anchors, features, local, kernels[0], kernels[1], regularizer)
