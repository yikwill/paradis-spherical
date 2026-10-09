"""Runtime setup without a training framework."""
import os
import random

import numpy as np
import torch


def setup_system(cfg):
    seed = cfg.init.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    torch.set_num_threads(cfg.compute.get("threads", min(8, len(os.sched_getaffinity(0)))))
    if cfg.compute.accelerator in ("gpu", "cuda"):
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.set_float32_matmul_precision("high")
    if not cfg.forecast.enable and cfg.training.max_steps < 0 and cfg.training.max_epochs < 0:
        raise ValueError("Set training.max_steps or training.max_epochs")
