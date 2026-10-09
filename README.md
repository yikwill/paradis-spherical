# PARADIS spherical reproduction

A native PyTorch fork of [PARADIS](https://github.com/Wx-Alliance-Alliance-Meteo/paradis_model),
pinned to `9a6f4b7d0dadfb80918ff8929bd92b6a5da2bcc5`. The reference paper is
[Pereira et al., Learning to Advect](https://arxiv.org/html/2601.21151v1).
This is a compact CPU reproduction and architecture experiment, not the paper's
full weather training run. Numerical findings and limitations are in
[the report](report/report.pdf), with [LaTeX source](report/report.tex) and
[raw results](report/results).

The fork replaces Lightning/TensorBoard with native PyTorch training and JSONL
logs. Every spatial convolution can use `model.spatial_conv=disco` (the default)
or `conv2d`; pointwise convolutions remain unchanged. Learned spatial bias maps
are removed. The only explicit coordinate input is unnormalized `sin_latitude`;
physical constants and time/solar forcings remain.

DISCO uses the pinned NVIDIA torch-harmonics implementation, piecewise-linear
basis shape `(k, 2*k+2)`, and support radius `k * latitude_spacing / 2`. Its adapter
handles latitude orientation, periodic longitude, and scalar pole averaging.
A CPU depthwise operator contracts the same library quadrature stencil without
materializing the basis-expanded activation. Other operators and devices use
torch-harmonics. The CPU extension has output, first-gradient, and finite-difference
checks; higher-order derivatives are not implemented.

## Environment

Linux x86-64, micromamba, and a C++ compiler are required. The committed CPU lock
uses Python 3.11 and PyTorch 2.14.1+cpu. From the repository root:

```bash
export PATH="$HOME/.local/bin:$PATH"
bash environment/setup.sh
micromamba activate paradis-spherical
```

If shell activation is unavailable, prefix commands with
`micromamba run -r "$HOME/micromamba" -n paradis-spherical`.
The activation hook selects the environment's C++ runtime, needed on this host.
The first DISCO call builds a small C++ extension in PyTorch's cache. Build flags
use the current CPU architecture; rebuild the extension when moving machines.
`environment/conda-linux-64.lock`, `environment/pip-cpu.lock`, and the pinned
harmonics revision in `environment/setup.sh` specify the environment. The existing
environment was exercised; a second clean installation was not used for validation.
Neither Lightning nor TensorBoard is a project dependency.

GPU execution requires a separately compatible CUDA PyTorch/harmonics build.
The recorded experiments are CPU-only; CUDA performance and parity are unmeasured.

## Data and paired weather experiment

The preparation command downloads full 2016 and January 2017 from the real
WeatherBench 2 ERA5 121x240 equiangular-with-poles dataset. It writes losslessly
compressed yearly stores, physical constants, and statistics fitted on 2016 only.
Allow roughly 12 GB of disk and at least 40 GB available RAM for preloaded training.
The preload allocator enforces `dataset.preload_max_gb` (64 by default).

```bash
python -m experiments.prepare_weather /path/to/paradis-data
python -m experiments.weather --root /path/to/paradis-data --steps 2000 --threads 4 --compile
python -m experiments.evaluate_weather --root /path/to/paradis-data
```

Both variants use the same seed (42), data order, AdamW settings, schedule, and
2,000 updates. They have two processor blocks, 64 latent, 32 transported, and 32
static channels. The 97 output features retain the upstream variable/pressure-level
selection. All inputs and targets stay inside their respective date splits.
Logs and checkpoints are written to `logs/weather/{conv2d,disco}`. Existing run
logs are protected from accidental reuse; use a new experiment name or explicitly
resume through `train.py`. Repeating the paired script requires a fresh log location
(move completed runs first).

For general training, use Hydra overrides or a saved experiment configuration:

```bash
python train.py --config-path report/results --config-name weather-disco \
  dataset.root_dir=/path/to/paradis-data training.experiment_name=disco-repeat
```

To resume exactly, retain the same training configuration and log directory and add
`init.restart=true init.checkpoint_path=/path/to/last.pt`. Checkpoints include model,
optimizer, scheduler, random-generator states, feature/grid/statistics metadata, and
training progress. Incompatible architecture, normalization, schedule, or distributed
world size is rejected. `torchrun --standalone --nproc-per-node=2 train.py ...`
launches native distributed training. The two-rank CPU path is checked; GPU DDP is
not validated here. AdamW is the supported optimizer.

The inherited default config describes a much larger GPU model. Use the compact
experiment configuration for these CPU results. Native validation supports the
configured aggregate loss; the old Lightning diagnostic callbacks are removed.

## Forecasting

Use a native checkpoint and its saved configuration. Forecast horizons can change;
feature order, model architecture, grid, and normalization must match.

```bash
python forecast.py --config logs/weather/disco/config.yaml \
  --checkpoint-path logs/weather/disco/best.pt --root-dir /path/to/paradis-data \
  --output-file /path/to/new-forecast.zarr --forecast-steps 3 \
  --start-date 2017-01-01T06:00:00 --end-date 2017-01-01T12:00:00 \
  --sampling-interval 6h --num-devices 1 --num-workers 0 --flush-every-n-steps 1
```

Forecast output is incremental Zarr with physical units and geographic wind
components. The output path must be new. Input history and the requested forcing
horizon must exist in the dataset. Native forecasting currently writes on one device.

## Tracer and performance experiments

```bash
python -m experiments.tracer --model paradis --steps 500 --seconds 7200
python -m experiments.tracer --model unet-residual --steps 500 --threads 4 --seconds 7200
python -m experiments.tracer --model paradis --steps 400 --continuation-steps 300 --seconds 5400
python -m experiments.tracer --model unet-residual --steps 400 --continuation-steps 300 --threads 4 --seconds 7200
python -m experiments.tracer --model paradis --steps 1000 --identity-steps 300 --continuation-steps 600 --threads 4 --seconds 10800
python -m experiments.tracer --model unet-residual --steps 1000 --identity-steps 300 --continuation-steps 600 --threads 4 --seconds 10800
```

All tracer runs use the full 721x1440 grid. The main experiment uses one fixed 68-degree
northeast displacement; continuation is a separately labelled diagnostic. The full
compact PARADIS replaces the paper's tiny NSL example as requested. The main U-Net follows Figure 19's normalized residual blocks (46,481 parameters).
ChannelNorm, unit residual scale, 3x3 kernels, average pooling, and bilinear upsampling
remain assumptions; the paper reports 45,909 parameters and its toy source is unavailable.
Earlier plain U-Net runs (30,601 parameters, `--model unet`) remain as auxiliary diagnostics. Gaussian width, center, and
optimizer settings are explicit in the script and report. Fields/checkpoints default
to `/home/yikwill/paradis-data/tracer`; override `--output` on another machine.

Create an unchanged reference worktree, then benchmark when other experiments
are idle. Affinity is optional and must match the local CPU topology.

```bash
git worktree add --detach ../paradis-upstream 9a6f4b7d0dadfb80918ff8929bd92b6a5da2bcc5
python -m experiments.benchmark --upstream ../paradis-upstream --compile --repeats 20 --profile
python -m experiments.benchmark_data --upstream ../paradis-upstream --root /path/to/paradis-data
python -m experiments.memory upstream --upstream ../paradis-upstream --compile
python -m experiments.memory conv2d --compile
python -m experiments.memory disco --compile
```

The latency harness compares the pinned upstream model with two optimized models
on identical full-feature synthetic tensors and matched compact widths. Upstream
retains spatial biases; the input width is held fixed. Training timings include
forward, loss, backward, and AdamW in a common native harness. They do not measure
the full Lightning training framework. Compilation warmup and construction are
reported separately. The strict gate uses the one-sided 95% bootstrap upper bound
of the DISCO/upstream median latency ratio, for both inference and training.
Memory measurements use separate processes and include Python/libraries/construction.

## Regression checks

```bash
python -m tests.check_data
python -m tests.check_loss
python -m tests.check_weather_metrics
python -m tests.check_model
python -m tests.check_fit
PARADIS_TEST_DATA=/tmp/paradis-ddp-check python -m torch.distributed.run --standalone --nproc-per-node=2 -m tests.check_distributed
python -m tests.check_forecast /path/to/paradis-data
python -m experiments.tune_compile
```

The port can also be compared to a fixture captured from the unchanged upstream
Lightning trainer. Keep Lightning in a separate reference environment, run the
capture from outside this repository with `PYTHONPATH` pointing to the worktree,
then run `python -m tests.check_port /path/to/fixture.pt ../paradis-upstream` here.
See `tests/capture_upstream.py`. This isolates numerical orchestration parity from
intentional architecture and loss-weight corrections.

The regressions cover temporal split isolation, single/multiple input frames,
normalization/cache immutability, all 97 named loss weights, DISCO reference values
and gradients, longitude shifts/poles, static-feature reuse with truncation, exact
checkpoint continuation, native distributed training, and real-data forecast writing.
Two upstream bugs are deliberately corrected: temporal windows crossing split
boundaries, and pressure weighting applied by channel position after common-first
feature reordering.

## Attribution

The original PARADIS source and history are retained under the [Apache 2.0 license](LICENSE).
The architecture follows the upstream authors' implementation. torch-harmonics is
an NVIDIA project with its own license. Upstream acknowledgements include
[Advection Augmented Convolutional Neural Networks](https://arxiv.org/abs/2406.19253)
and [Karina](https://arxiv.org/abs/2403.10555) for geocyclic padding.
