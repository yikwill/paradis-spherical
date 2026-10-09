"""DISCO adapter for PARADIS's south-to-north, periodic longitude grid."""
import math

import torch
from torch import nn
from torch_harmonics import DiscreteContinuousConvS2
from torch_harmonics.grid import EquiangularGrid


def pole_average(x):
    """A scalar field has one value at each actual pole."""
    return torch.cat([x[..., :1, :].mean(-1, keepdim=True).expand_as(x[..., :1, :]),
                      x[..., 1:-1, :],
                      x[..., -1:, :].mean(-1, keepdim=True).expand_as(x[..., -1:, :])], dim=-2)


class SphericalConv(nn.Module):
    def __init__(self, in_channels, out_channels, mesh_size, kernel_size, groups=1, bias=True):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError("Spatial DISCO kernels require an odd width of at least three")
        grid = EquiangularGrid(*mesh_size)
        self.conv = DiscreteContinuousConvS2(
            grid, grid, in_channels, out_channels, kernel_shape=(kernel_size, 2 * kernel_size + 2),
            basis_type="piecewise linear", basis_norm_mode="nodal", groups=groups, bias=bias,
            theta_cutoff=kernel_size * math.pi / (mesh_size[0] - 1) / 2,
            optimized_kernel=True)
        self.cpu_depthwise = in_channels == out_channels == groups
        if self.cpu_depthwise:
            # Register the op before torch.compile traces the forward method.
            from model import disco_cpu
        # Sort once by output latitude for the direct depthwise CPU contraction.
        order = torch.argsort(self.conv.psi_row_idx, stable=True)
        counts = torch.bincount(self.conv.psi_row_idx, minlength=mesh_size[0])
        self.register_buffer("rowptr", torch.cat([counts.new_zeros(1), counts.cumsum(0)]), persistent=False)
        self.register_buffer("kernel", self.conv.psi_ker_idx[order].contiguous(), persistent=False)
        self.register_buffer("column", self.conv.psi_col_idx[order].contiguous(), persistent=False)
        self.register_buffer("values", self.conv.psi_vals[order].contiguous(), persistent=False)

    def forward(self, x):
        # torch-harmonics uses increasing colatitude (north to south).
        x = pole_average(x).flip(-2).contiguous()
        if self.cpu_depthwise and x.device.type == "cpu" and x.dtype in (torch.float32, torch.float64):
            from model.disco_cpu import depthwise
            output = depthwise(x, self.conv, self.rowptr, self.kernel, self.column, self.values)
            if self.conv.bias is not None:
                output = output + self.conv.bias[None, :, None, None]
        else:
            output = self.conv(x)
        return pole_average(output.flip(-2))


def spatial_conv(in_channels, out_channels, mesh_size, kernel_size, groups=1, bias=True, backend="disco"):
    if backend == "disco":
        return SphericalConv(in_channels, out_channels, mesh_size, kernel_size, groups, bias)
    if backend == "conv2d":
        return nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups, bias=bias)
    raise ValueError(f"Unknown spatial convolution: {backend}")
