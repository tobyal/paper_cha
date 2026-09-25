from pathlib import Path
import numpy as np
import torch
import trimesh
import mcubes
from .geometry import write_json


@torch.no_grad()
def export_mesh(field, path, center, scale, resolution=128):
    path = Path(path)
    axis = torch.linspace(-1.2, 1.2, resolution, device=next(field.parameters()).device)
    values = []
    # Slabs bound memory; evaluate the field learned with dynamic support targets.
    for x in axis:
        yy, zz = torch.meshgrid(axis, axis, indexing='ij')
        xyz = torch.stack([torch.full_like(yy, x), yy, zz], -1).reshape(-1, 3)
        values.append(torch.cat([field(block) for block in xyz.split(2048)]).cpu().numpy())
    grid = np.asarray(values, dtype=np.float32).reshape(resolution, resolution, resolution)
    if not (grid.min() < 0 < grid.max()):
        write_json(path.with_suffix('.json'), {'status': 'no_zero_crossing', 'min': float(grid.min()), 'max': float(grid.max())})
        return {'status': 'no_zero_crossing'}
    vertices, faces = mcubes.marching_cubes(grid, 0.)
    vertices = (vertices / (resolution - 1) * 2.4 - 1.2) * scale + np.asarray(center)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.export(path)
    return {'status': 'exported', 'vertices': len(vertices), 'faces': len(faces), 'path': str(path)}
