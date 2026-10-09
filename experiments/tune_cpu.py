"""Validate and time the direct CPU depthwise contraction against torch-harmonics."""
import json
from pathlib import Path
import time

import torch
from model.spherical import SphericalConv
import model.disco_cpu


def main():
    torch.set_num_threads(4)
    results=[]
    for dtype, shape, channels, kernel in [(torch.float64,(17,32),2,3), (torch.float32,(121,240),64,7), (torch.float32,(61,120),64,5)]:
        layer=SphericalConv(channels,channels,shape,kernel,groups=channels).to(dtype=dtype)
        layer.cpu_depthwise=False
        x=torch.randn(1,channels,*shape,dtype=dtype,requires_grad=True)
        probe=torch.randn_like(x)
        expected=layer(x)
        gradients=torch.autograd.grad((expected*probe).sum(), (x,layer.conv.weight))
        for direct in (False,True):
            layer.cpu_depthwise=direct
            actual=layer(x)
            grads=torch.autograd.grad((actual*probe).sum(), (x,layer.conv.weight))
            tol=1e-10 if dtype==torch.float64 else 2e-4
            torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
            for a,b in zip(grads,gradients):
                torch.testing.assert_close(a,b,rtol=tol,atol=tol*10)
            times=[]
            for _ in range(6):
                layer.zero_grad(set_to_none=True)
                x.grad=None
                start=time.perf_counter()
                layer(x).square().mean().backward()
                times.append(time.perf_counter()-start)
            row=dict(shape=shape,channels=channels,kernel=kernel,dtype=str(dtype),direct=direct,seconds=times,
                     output_error=float((expected-actual).abs().max().detach()),
                     gradient_error=max(float((a-b).abs().max()) for a,b in zip(grads,gradients)))
            print(row,flush=True)
            results.append(row)
    Path('report/results/cpu-tuning.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
