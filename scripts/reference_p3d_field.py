"""Independent-process reference: original P3D SDF with baseline GSHE bypass."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--p3d-root', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
root = Path(args.p3d_root).resolve()
args.output = str(Path(args.output).resolve())
sys.path.insert(0, str(root))
os.chdir(root)
from models.gsdf.networks.gsdf_net import SDFNetwork

torch.set_num_threads(2)
torch.manual_seed(52)
cfg = SimpleNamespace(d_in=3, d_out=1, d_hidden=256, n_layers=8, skip_in=[4], multires=8,
                      geometric_init=True, weight_norm=True, inside_outside=False)
network = SDFNetwork(4096, cfg).cuda()
# A literal sum replaces the GSHE operator; everything else remains original.
class SumFusion(torch.nn.Module):
    def forward(self, xy, yz, xz, grid, use_ggpr=True):
        return xy + yz + xz + grid
network.ghse = SumFusion()
with torch.no_grad():
    # Activate encoded channels for a meaningful comparison beyond geometric init.
    network.lin0.weight_v[:, 3:].normal_(0, .02)
points = torch.rand(24, 3, device='cuda') * 1.6 - .8
outputs = {}
for step in [0, 2000, 7000]:
    x = points.detach().clone().requires_grad_(True)
    value = network(x, step)
    gradient = torch.autograd.grad(value.sum(), x)[0]
    outputs[step] = {'value': value.detach().cpu(), 'gradient': gradient.detach().cpu()}
torch.save({'state': {k: v.cpu() for k, v in network.state_dict().items()},
            'points': points.cpu(), 'outputs': outputs}, args.output)
print('Original P3D field reference saved:', args.output)
