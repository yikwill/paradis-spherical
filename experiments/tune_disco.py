"""Measure supported DISCO fusion before retaining an optimization."""
import json
from pathlib import Path
import time

import torch
from model.spherical import SphericalConv


def main():
    torch.set_num_threads(4)
    rows = []
    for shape, channels, kernel in [((121,240),64,7), ((61,120),64,5)]:
        model = SphericalConv(channels, channels, shape, kernel, groups=channels)
        model.cpu_depthwise = False
        x = torch.randn(1, channels, *shape, requires_grad=True)
        expected = model(x)
        probe = torch.randn_like(expected)
        gradients = torch.autograd.grad((expected*probe).sum(), (x, model.conv.weight))
        for fused in (False, True):
            model.conv.fused = fused
            output = model(x)
            actual = torch.autograd.grad((output*probe).sum(), (x, model.conv.weight))
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
            for a,b in zip(actual, gradients):
                torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-4)
            samples=[]
            for _ in range(6):
                model.zero_grad(set_to_none=True)
                x.grad=None
                start=time.perf_counter()
                model(x).square().mean().backward()
                samples.append(time.perf_counter()-start)
            row=dict(shape=shape,channels=channels,kernel=kernel,fused=fused,seconds=samples)
            rows.append(row)
            print(row,flush=True)
    Path('report/results/disco-tuning.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__ == '__main__':
    main()
