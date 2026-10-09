"""Pilot and paired partial weather runs with a predeclared common update count."""
import argparse
import json
from pathlib import Path
import time

from omegaconf import OmegaConf
import torch

from data.datamodule import Era5DataModule
from experiments.common import compact_config
from trainer import ParadisTask, configure_optimizers, fit
from utils.system import setup_system


def weather_config(root, backend, steps, threads):
    cfg = compact_config(backend, channels=64, static_channels=32)
    cfg.dataset.root_dir = root
    cfg.training.dataset.start_date = "2016-01-01"
    cfg.training.dataset.end_date = "2016-12-31T18:00:00"
    cfg.training.validation_dataset.start_date = "2017-01-01"
    cfg.training.validation_dataset.end_date = "2017-01-31T18:00:00"
    cfg.training.dataset.preload = True
    cfg.training.validation_dataset.preload = True
    cfg.training.max_steps = steps
    cfg.training.scheduler.wsd.warmup = min(20, max(1, steps // 10))
    cfg.training.scheduler.wsd.decay = .2
    cfg.training.log_every_n_steps = max(1, steps // 20)
    cfg.training.validation_dataset.validation_every_n_steps = max(1, steps // 10)
    cfg.training.validation_dataset.validation_batches = None
    cfg.training.log_dir = "logs/weather"
    cfg.training.experiment_name = backend
    cfg.compute.threads = threads
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/yikwill/paradis-data")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--backend", choices=["conv2d", "disco", "both"], default="both")
    args = parser.parse_args()
    cfg = weather_config(args.root, "conv2d", args.steps, args.threads)
    setup_system(cfg)
    start = time.perf_counter()
    data = Era5DataModule(cfg)
    data.setup("fit")
    print(f"Loaded {len(data.dataset)} train / {len(data.val_dataset)} validation windows in {time.perf_counter()-start:.1f}s", flush=True)
    backends = ("conv2d", "disco") if args.backend == "both" else (args.backend,)
    pilots = {}
    for backend in backends:
        cfg = weather_config(args.root, backend, args.steps, args.threads)
        cfg.compute.compile = args.compile
        setup_system(cfg)
        task = ParadisTask(data, cfg)
        print(f"{backend}: {sum(p.numel() for p in task.parameters())} parameters", flush=True)
        if args.pilot:
            optimizer, _ = configure_optimizers(task.model, cfg, args.steps)
            batch = next(iter(data.train_dataloader()))
            times = []
            for index in range(5):
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = task.rollout(batch, backward=lambda x: x.backward())
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite pilot loss")
                optimizer.step()
                times.append(time.perf_counter()-start)
                print(backend, index, times[-1], loss.item(), flush=True)
            pilots[backend] = dict(step_seconds=times, loss=float(loss), parameters=sum(p.numel() for p in task.parameters()))
        else:
            OmegaConf.save(cfg, Path("report/results") / f"weather-{backend}.yaml", resolve=True)
            start = time.perf_counter()
            progress = fit(task, data, cfg)
            progress["fit_wall_seconds"] = time.perf_counter()-start
            Path(f"report/results/weather-{backend}-progress.json").write_text(json.dumps(progress, indent=2)+"\n")
        del task
    if args.pilot:
        Path("report/results/weather-pilot.json").write_text(json.dumps(pilots, indent=2)+"\n")


if __name__ == "__main__":
    main()
