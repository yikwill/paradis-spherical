"""Train PARADIS with PyTorch, optionally launched with torchrun."""
import logging
import os

import hydra
import torch.distributed as dist

from data.datamodule import Era5DataModule
from trainer import ParadisTask, fit
from utils.system import setup_system


@hydra.main(version_base=None, config_path="config/", config_name="paradis_settings")
def main(cfg):
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl" if cfg.compute.accelerator in ("gpu", "cuda") else "gloo")
    setup_system(cfg)
    data = Era5DataModule(cfg)
    data.setup("fit")
    fit(ParadisTask(data, cfg), data, cfg)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
