"""Check eager/compiled agreement and record compilation overhead and timings."""
import json
from pathlib import Path
import time

import torch
from experiments.common import compact_config, synthetic_data
from trainer import ParadisTask
import model.disco_cpu


def main():
    torch.set_num_threads(4)
    results=[]
    for backend in ('conv2d','disco'):
        cfg=compact_config(backend,64,32)
        dm,batch=synthetic_data(cfg,121,240,full_features=True)
        task=ParadisTask(dm,cfg)
        x=torch.cat([batch[0][:,0],batch[2][:,0].permute(0,3,1,2),batch[3][:,0].permute(0,3,1,2)],1)
        eager=task(x)
        task.loss_fn(eager,batch[1][:,0]).backward()
        gradients=[p.grad.clone() for p in task.parameters()]
        for compiled in (False,True):
            if compiled:
                task.model.compile()
            start=time.perf_counter()
            task.zero_grad(set_to_none=True)
            actual=task(x)
            task.loss_fn(actual,batch[1][:,0]).backward()
            warmup=time.perf_counter()-start
            torch.testing.assert_close(actual,eager,rtol=1e-3,atol=1e-4)
            for p,g in zip(task.parameters(),gradients):
                torch.testing.assert_close(p.grad,g,rtol=2e-3,atol=2e-5)
            samples=[]
            for _ in range(5):
                task.zero_grad(set_to_none=True)
                start=time.perf_counter()
                task.loss_fn(task(x),batch[1][:,0]).backward()
                samples.append(time.perf_counter()-start)
            row=dict(backend=backend,compiled=compiled,warmup_seconds=warmup,seconds=samples,
                     output_max_error=float((actual-eager).abs().max().detach()))
            results.append(row)
            print(row,flush=True)
            Path('report/results/compile-tuning.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
