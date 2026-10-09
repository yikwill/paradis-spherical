"""Compare the first direct CPU kernel to the folded kernel, interleaved and seeded."""
import json
from pathlib import Path
import random
import statistics
import subprocess
from tempfile import TemporaryDirectory
import time

import torch
from torch.utils.cpp_extension import load

from model.spherical import SphericalConv


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    old_source = subprocess.check_output(['git','show','58f5fbf:model/disco_cpu.cpp'],text=True)
    with TemporaryDirectory() as directory:
        source = Path(directory)/'reference.cpp'
        source.write_text(old_source.replace('paradis_disco','paradis_disco_v1'))
        load(name='paradis_disco_v1_reference',sources=[str(source)],
             extra_cflags=['-O3','-march=native','-ffast-math','-funroll-loops','-fopenmp'],
             extra_ldflags=['-fopenmp'],is_python_module=False)
    results=[]
    for height,channels,kernel,dtype in [(17,3,3,torch.float64),(61,32,5,torch.float32),(121,64,7,torch.float32)]:
        width=2*(height-1)
        model=SphericalConv(channels,channels,(height,width),kernel,groups=channels).to(dtype=dtype)
        old_order=torch.argsort(model.conv.psi_row_idx,stable=True)
        old_geometry=(model.rowptr,model.conv.psi_ker_idx[old_order].contiguous(),
                      model.conv.psi_col_idx[old_order].contiguous(),model.conv.psi_vals[old_order].contiguous())
        new_geometry=(model.rowptr,model.kernel,model.column,model.values)
        x=torch.randn(2 if height==17 else 1,channels,height,width,dtype=dtype)
        probe=torch.randn_like(x)
        weight=model.conv.weight.detach()
        operators=[(torch.ops.paradis_disco_v1,old_geometry),(torch.ops.paradis_disco,new_geometry)]
        outputs=[]
        for op,geometry in operators:
            outputs.append((op.forward(x,weight,*geometry),*op.backward(probe,x,weight,*geometry)))
        tolerance=(1e-10,1e-10) if dtype==torch.float64 else (2e-4,2e-3)
        for old,new in zip(*outputs):
            torch.testing.assert_close(old,new,rtol=tolerance[0],atol=tolerance[1])
        samples=[[],[]]
        rng=random.Random(42)
        for repeat in range(20):
            order=[0,1]
            rng.shuffle(order)
            for index in order:
                op,geometry=operators[index]
                start=time.perf_counter()
                op.forward(x,weight,*geometry)
                op.backward(probe,x,weight,*geometry)
                elapsed=time.perf_counter()-start
                if repeat>=3:
                    samples[index].append(elapsed)
        row=dict(shape=list(x.shape),kernel=kernel,dtype=str(dtype),seed=42,
                 old_and_folded_seconds=samples,
                 output_inputgrad_weightgrad_errors=[(old-new).abs().max().item() for old,new in zip(*outputs)],
                 median_speedup=statistics.median(samples[0])/statistics.median(samples[1]))
        results.append(row)
        print(json.dumps(row),flush=True)
    Path('report/results/grouped-cpu-final.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
