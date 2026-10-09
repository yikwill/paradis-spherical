"""Run in the separate reference environment, against an unchanged checkout.

Usage: PYTHONPATH=/path/to/upstream python tests/capture_upstream.py output.pt
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Import the upstream trainer BEFORE adding this project's experiment helpers.
from trainer import LitParadis
sys.path.append(str(Path(__file__).resolve().parents[1]))
from experiments.common import compact_config, synthetic_data

torch.set_num_threads(2)
cfg = compact_config(static_channels=128)
cfg.model.forecast_steps = 2
cfg.training.accumulate_grad_batches = 2
dm, batch = synthetic_data(cfg)
torch.manual_seed(42)
task = LitParadis(dm, cfg)
task.trainer = SimpleNamespace(estimated_stepping_batches=20, is_last_batch=False)
initial = {k: v.detach().clone() for k, v in task.model.state_dict().items()}
configured = task.configure_optimizers()
optimizer = configured["optimizer"]
scheduler = configured["lr_scheduler"]["scheduler"]
task.optimizers = lambda: optimizer
task.lr_schedulers = lambda: scheduler
task.manual_backward = lambda loss: loss.backward()
task.log = lambda *a, **kw: None
inputs = torch.cat([batch[0][:, 0], batch[2][:, 0].permute(0, 3, 1, 2),
                    batch[3][:, 0].permute(0, 3, 1, 2)], dim=1)
prediction = task(inputs).detach()
loss = task.training_step(batch, 0)
grads = {k: p.grad.detach().clone() for k, p in task.model.named_parameters()}
task.training_step(batch, 1)
updated = task.model.state_dict()
torch.save(dict(cfg=cfg, initial=initial, prediction=prediction, loss=loss,
                grads=grads, updated=updated, lr=scheduler.get_last_lr()), sys.argv[1])
print("Captured unchanged upstream output, two-step rollout loss, gradients, and accumulated AdamW update")
