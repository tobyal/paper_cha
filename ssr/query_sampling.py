"""Method 3.3: read local surface representations with surface queries."""
from dataclasses import dataclass
from torch import Tensor
from .representation import LocalSurfaceRepresentation


@dataclass
class SurfaceQueries:
    """Fixed local query offsets and their learned, attended representations."""
    offsets: Tensor             # [3, r], shared local query positions
    features: Tensor            # [B, 128, N, r], r=4 in the native checkpoint


def sample_surface_queries(decoder, representation: LocalSurfaceRepresentation):
    """KGM initializes queries; attention reads the adaptive local bases.

    'Sampling' denotes extracting surface evidence at finite query slots. The
    native implementation retains spatial query offsets; it does not sample an
    arbitrary continuous latent distribution.
    """
    queries = decoder.kgm(representation.anchors, representation.local_features)
    q = decoder.projector(queries[1])
    k = decoder.projector(representation.kernel_features)
    for attention in decoder.attns:
        q = attention(q, k)
    return SurfaceQueries(queries[0], q)
