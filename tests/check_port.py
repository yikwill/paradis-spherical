"""Compare native training to a fixture captured from the pinned upstream trainer.

python -m tests.check_port /tmp/paradis-upstream-port.pt /path/to/upstream
"""
import importlib.util
import json
from pathlib import Path
import sys

import torch
from experiments.common import synthetic_data
from trainer import ParadisTask, configure_optimizers


def reference_model(path, dm, cfg):
    # Load the unchanged model and its absolute imports under their original names.
    names = ["model.blocks", "model.advection", "model.paradis"]
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(name, Path(path) / (name.replace(".", "/") + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules["model.paradis"].Paradis(dm, cfg, dm.dataset.lat_rad_grid, dm.dataset.lon_rad_grid)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def main():
    torch.set_num_threads(2)
    saved = torch.load(sys.argv[1], weights_only=False)
    cfg = saved["cfg"]
    dm, batch = synthetic_data(cfg)
    model = reference_model(sys.argv[2], dm, cfg)
    model.load_state_dict(saved["initial"])
    task = ParadisTask(dm, cfg, model=model)
    optimizer, scheduler = configure_optimizers(model, cfg, 20)
    inputs = torch.cat([batch[0][:, 0], batch[2][:, 0].permute(0, 3, 1, 2),
                        batch[3][:, 0].permute(0, 3, 1, 2)], dim=1)
    prediction = task(inputs).detach()
    torch.testing.assert_close(prediction, saved["prediction"], rtol=0, atol=0)
    optimizer.zero_grad()
    loss = task.rollout(batch, backward=lambda x: x.backward(), accumulation=2)
    torch.testing.assert_close(loss, saved["loss"], rtol=0, atol=0)
    max_grad_error = 0.0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, saved["grads"][name], rtol=1e-6, atol=1e-7)
        max_grad_error = max(max_grad_error, (parameter.grad - saved["grads"][name]).abs().max().item())
    task.rollout(batch, backward=lambda x: x.backward(), accumulation=2)
    optimizer.step()
    scheduler.step()
    max_update_error = 0.0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved["updated"][name], rtol=1e-6, atol=1e-7)
        max_update_error = max(max_update_error, (value - saved["updated"][name]).abs().max().item())
    assert scheduler.get_last_lr() == saved["lr"]
    result = dict(output_max_abs_error=(prediction - saved["prediction"]).abs().max().item(),
                  loss_abs_error=(loss - saved["loss"]).abs().item(),
                  gradient_max_abs_error=max_grad_error, update_max_abs_error=max_update_error,
                  forecast_steps=2, accumulated_batches=2, status="passed")
    Path("report/results/port.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
