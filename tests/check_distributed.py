"""Two-rank CPU training smoke check, launched by torchrun."""
import os
from pathlib import Path

import torch.distributed as dist

from data.datamodule import Era5DataModule
from tests.check_data import config, tiny_store
from trainer import ParadisTask, fit
from utils.system import setup_system


def main():
    dist.init_process_group("gloo")
    root=Path(os.environ["PARADIS_TEST_DATA"])
    if dist.get_rank()==0:
        tiny_store(root)
    dist.barrier()
    cfg=config(root)
    cfg.model.spatial_conv="disco"
    cfg.training.dataset.start_date="2016-12-30"
    cfg.training.dataset.end_date="2016-12-31"
    cfg.training.validation_dataset.start_date="2017-01-01"
    cfg.training.validation_dataset.end_date="2017-01-02"
    cfg.training.max_steps=3
    cfg.training.scheduler.wsd.warmup=1
    cfg.training.accumulate_grad_batches=2
    cfg.training.validation_dataset.validation_every_n_steps=1
    cfg.training.log_dir=str(root/'logs')
    cfg.compute.threads=1
    for steps,detach in ((2,1),(3,2)):
        cfg.model.forecast_steps=steps
        cfg.training.optimizer.detach_gradient_every=detach
        cfg.training.experiment_name=f'ddp-{steps}-{detach}'
        setup_system(cfg)
        data=Era5DataModule(cfg)
        data.setup('fit')
        progress=fit(ParadisTask(data,cfg),data,cfg)
        assert progress['step']==3
        dist.barrier()
        if dist.get_rank()==0:
            assert (root/'logs'/cfg.training.experiment_name/'last.pt').exists()
            print(f'PASS: two-rank CPU DISCO, {steps} steps/truncation {detach}, accumulation, validation, checkpoint')
    dist.destroy_process_group()


if __name__=='__main__':
    main()
