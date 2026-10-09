"""DISCO reference, gradient, geometry, full-model, and resume checks."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch_harmonics import DiscreteContinuousConvS2

from experiments.common import compact_config, synthetic_data
from model.spherical import SphericalConv, pole_average
from trainer import (ParadisTask, configure_optimizers, save_checkpoint,
                     load_checkpoint, rng_state)


def check_disco():
    adapter = SphericalConv(2, 2, (17, 32), 3, groups=2).double()
    conv = adapter.conv
    reference = DiscreteContinuousConvS2(conv.grid_in, conv.grid_out, 2, 2,
                                        kernel_shape=(3, 8), groups=2,
                                        theta_cutoff=conv.theta_cutoff, optimized_kernel=False).double()
    reference.load_state_dict(conv.state_dict())
    x = torch.randn(1, 2, 17, 32, dtype=torch.float64, requires_grad=True)
    expected = pole_average(reference(pole_average(x).flip(-2)).flip(-2))
    actual = adapter(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    v = torch.randn_like(actual)
    a_grad = torch.autograd.grad((actual*v).sum(), (x, conv.weight, conv.bias))
    r_grad = torch.autograd.grad((expected*v).sum(), (x, reference.weight, reference.bias))
    for a, b in zip(a_grad, r_grad):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(adapter(x.roll(5, -1)), actual.roll(5, -1), rtol=1e-6, atol=1e-7)
    assert actual[..., 0, :].std(-1).max() < 1e-12
    assert actual[..., -1, :].std(-1).max() < 1e-12
    assert conv.kernel_size == 9
    assert abs(float(conv.grid_in.quad_weights.sum()) - 4 * torch.pi) < 1e-12
    small = SphericalConv(1, 1, (9, 16), 3, groups=1).double()
    probe = torch.randn(1, 1, 9, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(small, (probe,), fast_mode=True)
    return (actual-expected).abs().max().item()


def check_resume(backend):
    cfg = compact_config(backend)
    cfg.model.forecast_steps = 2
    cfg.training.accumulate_grad_batches = 2
    dm, batch = synthetic_data(cfg)
    torch.manual_seed(42)
    task = ParadisTask(dm, cfg)
    spatial = [m for m in task.modules() if isinstance(m, SphericalConv)]
    if backend == "disco":
        assert len(spatial) == 9  # 3 static plus velocity/advection/diffusion for each block
        assert all(m.kernel_size == (1, 1) for m in task.modules() if isinstance(m, nn.Conv2d))
    assert not any(name.endswith((".U", ".V", ".A")) for name, _ in task.named_parameters())
    optimizer, scheduler = configure_optimizers(task.model, cfg, 20)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    def update(model, opt, sched):
        opt.zero_grad(set_to_none=True)
        for _ in range(2):
            loss = model.rollout(batch, backward=lambda x: x.backward(), accumulation=2)
            assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
        sched.step()
    update(task, optimizer, scheduler)
    with TemporaryDirectory() as directory:
        path = Path(directory)/"resume.pt"
        save_checkpoint(path, task, optimizer, scheduler, scaler, {"step": 1})
        expected_random = torch.rand(5)
        update(task, optimizer, scheduler)
        resumed = ParadisTask(dm, cfg)
        opt, sched = configure_optimizers(resumed.model, cfg, 20)
        progress = load_checkpoint(path, resumed, opt, sched, scaler)
        assert progress["step"] == 1
        torch.testing.assert_close(torch.rand(5), expected_random, rtol=0, atol=0)
        update(resumed, opt, sched)
        for a, b in zip(task.parameters(), resumed.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        bad = copy.deepcopy(cfg)
        bad.model.spatial_conv = "conv2d" if backend == "disco" else "disco"
        resumed.cfg = bad
        try:
            load_checkpoint(path, resumed)
        except ValueError:
            pass
        else:
            raise AssertionError("Incompatible convolution backend was accepted")
    return sum(p.numel() for p in task.parameters())


def check_static_reuse(backend):
    """Compare cached rollout to recomputing static features at every step."""
    for detach in (1, 2, 3):
        cfg = compact_config(backend)
        cfg.model.forecast_steps = 3
        cfg.training.optimizer.detach_gradient_every = detach
        dm, batch = synthetic_data(cfg)
        cached = ParadisTask(dm, cfg)
        reference = copy.deepcopy(cached)
        actual = cached.rollout(batch, backward=lambda loss: loss.backward())
        inputs, targets, forcings, constants = batch
        fields = inputs[:, 0]
        constants = constants[:, 0].permute(0, 3, 1, 2)
        chunk = 0
        losses = []
        for step in range(3):
            output = reference(torch.cat([fields, forcings[:, step].permute(0, 3, 1, 2), constants], 1))
            loss = reference.loss_fn(output, targets[:, step])
            losses.append(loss.detach())
            chunk = chunk + loss / 3
            fields = reference._autoregression_next_input(fields, output)
            if (step + 1) % detach == 0 or step == 2:
                chunk.backward()
                fields = fields.detach()
                chunk = 0
        torch.testing.assert_close(actual, torch.stack(losses).mean(), rtol=1e-6, atol=1e-7)
        for a, b in zip(cached.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=1e-6)


def check_model_longitude(backend):
    cfg = compact_config(backend)
    dm, batch = synthetic_data(cfg)
    torch.manual_seed(42)
    task = ParadisTask(dm, cfg).eval()
    fields = pole_average(torch.cat([batch[0][:, 0], batch[2][:, 0].permute(0, 3, 1, 2),
                                     batch[3][:, 0].permute(0, 3, 1, 2)], 1))
    # Coarsening preserves shifts that are multiples of its stride.
    shift = cfg.model.coarsening_factor
    with torch.no_grad():
        expected = task(fields)
        actual = task(fields.roll(shift, -1))
    torch.testing.assert_close(actual, expected.roll(shift, -1), rtol=2e-5, atol=1e-6)
    return dict(shift=shift, max_abs_error=(actual - expected.roll(shift, -1)).abs().max().item(),
                # The inherited pooling does not preserve scalar poles globally.
                pole_longitude_std=[expected[..., row, :].std(-1).max().item() for row in (0, -1)])


def main():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    for backend in ("conv2d", "disco"):
        check_static_reuse(backend)
    result = dict(disco_reference_max_abs_error=check_disco(),
                  parameters={name: check_resume(name) for name in ("conv2d", "disco")},
                  full_model_longitude={name: check_model_longitude(name) for name in ("conv2d", "disco")},
                  checkpoint_resume="bitwise identical", static_reuse="loss and gradients match for truncation 1/2/3", status="passed")
    Path("report/results/model-checks.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
