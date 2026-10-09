"""Matched CPU latency measurements, optionally with an operator profile."""
import argparse
import gc
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch

from experiments.common import compact_config, synthetic_data
from tests.check_port import reference_model
from trainer import ParadisTask, configure_optimizers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlat", type=int, default=121)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--upstream", default="/home/yikwill/paradis-upstream")
    parser.add_argument("--output", default="report/results/benchmark.json")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    models, optimizers = {}, {}
    inputs = targets = None
    construction = {}
    for name in ("upstream", "conv2d", "disco"):
        cfg = compact_config("conv2d" if name == "upstream" else name, args.channels, 128)
        cfg.model.static_hidden_size = 64
        cfg.model.bias_channels = 8 if name == "upstream" else 0
        dm, batch = synthetic_data(cfg, args.nlat, 2*(args.nlat-1), full_features=True)
        torch.manual_seed(42)
        start = time.perf_counter()
        model = reference_model(args.upstream, dm, cfg) if name == "upstream" else None
        task = ParadisTask(dm, cfg, model=model)
        if args.compile and name != "upstream":
            task.model.compile()
        construction[name] = time.perf_counter()-start
        models[name] = task
        optimizers[name] = configure_optimizers(task.model, cfg, 20)[0]
        inputs = torch.cat([batch[0][:, 0], batch[2][:, 0].permute(0, 3, 1, 2),
                            batch[3][:, 0].permute(0, 3, 1, 2)], dim=1)
        targets = batch[1][:, 0]
    def run(name, workload):
        task = models[name]
        if workload == "inference":
            task.eval()
            with torch.inference_mode():
                task(inputs)
        else:
            task.train()
            optimizers[name].zero_grad(set_to_none=True)
            task.loss_fn(task(inputs), targets).backward()
            optimizers[name].step()
    samples = {workload: {name: [] for name in models} for workload in ("inference", "training")}
    rng = np.random.default_rng(42)
    warmup = {}
    for workload in samples:
        warmup[workload] = {}
        for name in models:
            start = time.perf_counter()
            for _ in range(2):
                run(name, workload)
            warmup[workload][name] = time.perf_counter()-start
        gc.collect()
        for repeat in range(args.repeats):
            for name in rng.permutation(list(models)):
                start = time.perf_counter()
                run(name, workload)
                elapsed = time.perf_counter()-start
                samples[workload][name].append(elapsed)
                print(workload, name, repeat, f"{elapsed:.4f}s", flush=True)
    summary = {}
    for workload, values in samples.items():
        summary[workload] = {name: float(np.median(v)) for name,v in values.items()}
        for denominator in ("upstream", "conv2d"):
            a, b = np.array(values["disco"]), np.array(values[denominator])
            draws = rng.integers(0, len(a), size=(10000, len(a)))
            ratios = np.median(a[draws], axis=1)/np.median(b[draws], axis=1)
            ci = np.quantile(ratios, [.025, .95, .975]).tolist()
            summary[workload][f"disco_over_{denominator}"] = dict(median_ratio=float(np.median(a)/np.median(b)),
                                                                bootstrap_95_interval=[ci[0],ci[2]],
                                                                one_sided_95_upper=ci[1], no_slowdown=ci[1] <= 1)
    result = dict(arguments=vars(args), torch=torch.__version__, platform=platform.platform(),
                  parameters={name: sum(p.numel() for p in task.parameters()) for name,task in models.items()},
                  construction_seconds=construction, two_warmup_calls_seconds=warmup,
                  samples_seconds=samples, summary=summary,
                  combined_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  note="All variants use identical synthetic inputs, grid, latent and static widths; upstream retains learned spatial biases. Training includes AdamW.")
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    if args.profile:
        for name in models:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
                run(name, "training")
            table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
            Path(args.output).with_name(f"profile-{name}.txt").write_text("\n".join(line.rstrip() for line in table.splitlines())+"\n")


if __name__ == "__main__":
    main()
