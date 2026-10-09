"""Measure channels-last storage on full weather-shaped tensors."""
import json
from pathlib import Path
import time

import torch

from experiments.common import compact_config, synthetic_data
from trainer import ParadisTask


def main():
    torch.set_num_threads(4)
    results=[]
    for backend in ('conv2d','disco'):
        cfg=compact_config(backend,64,32)
        dm,batch=synthetic_data(cfg,121,240,full_features=True)
        model=ParadisTask(dm,cfg)
        x=torch.cat([batch[0][:,0],batch[2][:,0].permute(0,3,1,2),batch[3][:,0].permute(0,3,1,2)],1)
        with torch.no_grad():
            reference=model(x)
        for layout in (torch.contiguous_format,torch.channels_last):
            model.to(memory_format=layout)
            x=x.contiguous(memory_format=layout)
            with torch.no_grad():
                torch.testing.assert_close(model(x),reference,rtol=2e-4,atol=2e-5)
            samples=[]
            for _ in range(5):
                model.zero_grad(set_to_none=True)
                start=time.perf_counter()
                model.loss_fn(model(x),batch[1][:,0]).backward()
                samples.append(time.perf_counter()-start)
            row=dict(backend=backend,layout=str(layout),seconds=samples)
            results.append(row)
            print(row,flush=True)
    Path('report/results/layout-tuning.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
