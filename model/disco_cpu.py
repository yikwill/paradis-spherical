"""Compile and register the CPU depthwise contraction and its exact first derivative."""
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

load(name="paradis_disco_cpu_v2", sources=[str(Path(__file__).with_suffix(".cpp"))],
     extra_cflags=["-O3", "-march=native", "-ffast-math", "-funroll-loops", "-fopenmp"],
     extra_ldflags=["-fopenmp"], is_python_module=False)


@torch.library.register_fake("paradis_disco::forward")
def _forward_fake(x, w, ptr, kernel, column, values):
    return torch.empty_like(x)


@torch.library.register_fake("paradis_disco::backward")
def _backward_fake(grad, x, w, ptr, kernel, column, values):
    return torch.empty_like(x), torch.empty_like(w)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad):
    dx, dw = torch.ops.paradis_disco.backward(grad.contiguous(), *ctx.saved_tensors)
    return dx, dw, None, None, None, None


torch.library.register_autograd("paradis_disco::forward", _backward, setup_context=_setup)


def depthwise(x, conv, ptr, kernel, column, values):
    return torch.ops.paradis_disco.forward(x, conv.weight, ptr, kernel, column, values.to(x.dtype))
