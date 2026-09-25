from pathlib import Path
import json
import numpy as np
import torch
import trimesh


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def load_points(path, count=None, seed=21, key='p', strict_count=True):
    path = Path(path)
    rng = np.random.default_rng(seed)
    if path.suffix == '.npz':
        with np.load(path) as data:
            points = data[key].copy()
    elif path.suffix == '.npy':
        points = np.load(path)
    elif path.suffix in ('.xyz', '.txt'):
        points = np.loadtxt(path)[:, :3]
    else:
        shape = trimesh.load(path, process=False)
        if isinstance(shape, trimesh.Scene):
            shape = trimesh.util.concatenate(tuple(shape.geometry.values()))
        if isinstance(shape, trimesh.Trimesh):
            if count is None:
                raise ValueError('Mesh input requires an explicit sample count')
            # Mesh is used ONLY to construct sparse observations, not train targets.
            triangles = shape.triangles[rng.choice(len(shape.faces), count, p=shape.area_faces / shape.area)]
            uv = rng.random((count, 2))
            uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
            points = triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0]) + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
        else:
            points = np.asarray(shape.vertices)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite [N,3] points')
    if count is not None and len(points) > count:
        points = points[rng.choice(len(points), count, replace=False)]
    if strict_count and count is not None and len(points) < count:
        raise ValueError('Requested more observations than available; no duplicate padding')
    return points


def normalize(points):
    center = (points.max(0) + points.min(0)) / 2
    scale = float(np.linalg.norm(points - center, axis=1).max())
    if scale < 1e-8:
        raise ValueError('Degenerate observations')
    return (points - center) / scale, center, scale


@torch.no_grad()
def fps_indices(points, count):
    count = min(count, len(points))
    selected = torch.empty(count, dtype=torch.long, device=points.device)
    distance = torch.full((len(points),), float('inf'), device=points.device)
    index = ((points - points.mean(0)) ** 2).sum(1).argmax()
    for i in range(count):
        selected[i] = index
        distance = torch.minimum(distance, ((points - points[index]) ** 2).sum(1))
        distance[selected[:i + 1]] = -1
        index = distance.argmax()
    return selected


@torch.no_grad()
def nearest_indices(query, reference, k=1, chunk=512):
    return torch.cat([torch.cdist(block, reference).topk(min(k, len(reference)), largest=False).indices
                      for block in query.split(chunk)], 0)


def directed_distance(query, reference):
    indices = nearest_indices(query, reference)[:, 0]
    return ((query - reference[indices]).square().sum(-1) + 1e-12).sqrt().mean()


def patch_layout(points, patch_size=256, overlap=3):
    if len(points) < 16:
        raise ValueError('RepKPU encoder requires at least 16 observations')
    size = min(patch_size, len(points))
    seeds = fps_indices(points, max(1, int(np.ceil(len(points) / size * overlap))))
    indices = nearest_indices(points[seeds], points, size)
    patches = points[indices]
    centers = patches.mean(1, keepdim=True)
    radii = (patches - centers).norm(dim=-1).amax(1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    return {'patches': ((patches - centers) / radii).transpose(1, 2).contiguous(),
            'centers': centers, 'radii': radii, 'indices': indices}


def save_points(path, points):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    trimesh.PointCloud(points).export(path)
