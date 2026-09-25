import torch
import torch.nn as nn
import numpy as np
from .p3d_backend.embedder import get_embedder
from .p3d_backend.multiscale import Hash_triplane, Hash_grid

class SDFNetwork(nn.Module):
     def __init__(self,
                 point_size,
                 config):
          super(SDFNetwork, self).__init__()

          dims = [config.d_in] + [config.d_hidden for _ in range(config.n_layers)] + [config.d_out]

          self.embed_fn_fine = None
          if config.multires > 0:
               embed_fn, input_ch = get_embedder(config.multires, input_dims=config.d_in)
               self.embed_fn_fine = embed_fn
               dims[0] = input_ch

   
          self.grid_encoding = Hash_grid(point_size=point_size, use_pro=True)
          self.plane_encoding = Hash_triplane(point_size=point_size, multires=config.multires, use_pro=True)

          


          dims[0] += 16*2
          self.num_layers = len(dims)
          self.skip_in = config.skip_in

          for l in range(0, self.num_layers - 1):
               if l + 1 in self.skip_in:
                    out_dim = dims[l + 1] - dims[0]
               else:
                    out_dim = dims[l + 1]

               lin = nn.Linear(dims[l], out_dim)

               if config.geometric_init:
                    if l == self.num_layers - 2:
                         if not config.inside_outside:
                              torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                              torch.nn.init.constant_(lin.bias, -0.5)
                         else:
                              torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                              torch.nn.init.constant_(lin.bias, 0.5)
                    elif config.multires > 0 and l == 0:
                         torch.nn.init.constant_(lin.bias, 0.0)
                         torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                         torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    elif config.multires > 0 and l in self.skip_in:
                         torch.nn.init.constant_(lin.bias, 0.0)
                         torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                         torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                    else:
                         torch.nn.init.constant_(lin.bias, 0.0)
                         torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

               if config.weight_norm:
                    lin = nn.utils.weight_norm(lin)
               setattr(self, "lin" + str(l), lin)
               
          self.activation = nn.ReLU()

     def forward(self, inputs, step):
          feature = 0.

          if self.embed_fn_fine is not None:
               inputs = self.embed_fn_fine(inputs)

          xy_feat, yz_feat, xz_feat = self.plane_encoding(inputs[..., :3], step)
          grid_feat = self.grid_encoding(inputs[..., :3], step)

          fused_feature = xy_feat + yz_feat + xz_feat + grid_feat
          feature += fused_feature

          inputs = torch.cat((inputs, feature), dim=-1)

          x = inputs
          for l in range(0, self.num_layers - 1):
               lin = getattr(self, "lin" + str(l))
               if l in self.skip_in:
                    x = torch.cat([x, inputs], 1) / np.sqrt(2)

               x = lin(x)
               if l < self.num_layers - 2:
                    x = self.activation(x)

          return x

     def sdf(self, x, step):
          return self.forward(x, step)

     def gradient(self, x, step):
          x.requires_grad_(True)
          y = self.sdf(x,step)
          d_output = torch.ones_like(y, requires_grad=False, device=y.device)
          gradients = torch.autograd.grad(
               outputs=y,
               inputs=x,
               grad_outputs=d_output,
               create_graph=True,
               retain_graph=True,
               only_inputs=True)[0]
          return gradients, y


