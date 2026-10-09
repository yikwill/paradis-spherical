# Reproduction work log

The original agreed plan is in `experiments/plan.json`. Setup and report generation
are outside the 12-hour CPU experiment budget; four GPU-hours remain reserved.

## Established

- Full upstream history cloned, pinned at `9a6f4b7d0dadfb80918ff8929bd92b6a5da2bcc5`.
- Private GitHub repository: https://github.com/yikwill/paradis-spherical.
- Unchanged reference worktree: `/home/yikwill/paradis-upstream`.
- Micromamba: `/home/yikwill/.local/bin/micromamba`; root `/home/yikwill/micromamba`;
  environment `paradis-spherical`. CPU PyTorch 2.14.1, torch-harmonics source
  `6e7648d4435109c8a4ae747a9cd279b6ae3a30ba`, compiled with OpenMP/native CPU flags.
- Activation hooks select the environment's libstdc++; without activation the
  host's older libstdc++ can break NumPy after importing PyTorch.
- Reference-only Lightning venv: `/home/yikwill/paradis-reference-env`.
  Neither Lightning nor TensorBoard is installed in the project environment.
- Prepared real ERA5: `/home/yikwill/paradis-data/{2016,2017,constants,stats}`.
  Full 2016 and January 2017, 121x240, 97 output features. Training-only stats.
  Preparation completed in about 230 seconds. Data not committed.

## Implemented and checked

- Native PyTorch training, AdamW gate learning-rate multiplier, original WSD,
  autoregression, truncation, fixed accumulation scaling, validation, clipping,
  early stopping, atomic checkpoints and RNG resume; native forecasting entrypoint.
- Port fixture captured using unchanged Lightning trainer; outputs, two-step loss,
  gradients, accumulated AdamW update matched exactly (`report/results/port.json`).
- Dataset split leak reproduced against upstream: Dec 31 origins read Jan 1 targets.
  Fixed complete-window split isolation; one-frame shape bug fixed; constants ordered
  as configured; sin(latitude) generated unnormalized; forcings precomputed; cached
  sample path bypasses xarray. `tests/check_data.py` passes.
- All spatial convolutions use selectable conv2d/DISCO, including static encoder;
  GlobalBias removed; only sin(latitude) remains as explicit coordinate input.
- `tests/check_model.py`: DISCO/reference gradients, longitude shifts, pole rows,
  quadrature, finite-difference input gradient, both full models, bitwise checkpoint
  resume, incompatible checkpoint rejection. Passed.
- `tests/check_fit.py`: actual fit loop interruption/resume within epoch and across
  epoch boundary is bitwise identical. Passed.
- `tests/check_distributed.py`: torchrun two-rank CPU DISCO accumulation/validation/
  checkpoint smoke passed. Logs in `/tmp/paradis-ddp.log`.
- Additional upstream bug reproduced: common-first channel ordering assigned
  pressure factors to surface winds. Fixed loss weights by feature name and added
  `tests/check_loss.py` (all 97 features pass). This is an intentional correctness
  change separate from the numerically identical orchestration port.

## Optimization and active experiments

- Initial reduced-feature profiling found DISCO basis-expanded intermediates and
  copies dominant. `report/results/pilot-benchmark.json` uses only 5 output features,
  so must NOT compare its absolute timings to later full-feature benchmarks.
- Direct depthwise C++ CPU contraction (`model/disco_cpu.*`) uses unchanged
  torch-harmonics stencil/weights without K-expanded activations. Enabled by default
  for CPU fp32/fp64 depthwise convolutions; other paths use torch-harmonics.
- Kernel tests showed ~3.3x large-layer and ~2x small-layer training speedups after
  OpenMP and vectorization flags. Before OpenMP it was slower. Raw measurements in
  `report/results/cpu-tuning.json` and `/tmp/paradis-cpu-tuning-openmp.log`.
- Supported torch-harmonics fusion and channels-last layout were tested. No
  universal benefit, neither enabled. Geometry caching remains torch-harmonics'
  existing cache. Static encoder may be reused within rollout truncation chunks and
  across a forecast via explicit hidden_static (avoids stale automatic caches).
- Full-feature benchmark uses original/static widths 128 latent, 64 hidden for all
  variants; 64 dynamic latent, 32 advected, two blocks, identical input tensors.
  Upstream keeps its spatial biases. `benchmark-direct-cpu.json` currently fails
  the strict no-slowdown gate; timings had concurrent workload noise. Final benchmark
  should be rerun after long experiments are finished, with no competing own jobs.
- Real weather pilot (before named-weight bug fix): 1462 train / 122 val windows,
  111638 parameters in each variant, conv2d ~0.68 s/update, library DISCO ~1.29 s.
  Planned paired count: 2000 updates, same seed/order/AdamW/schedule. Not launched yet.
- Active fixed-target full-resolution tracer runs: PARADIS (tool session 22045,
  log `/tmp/paradis-tracer.log`, cores16-23, 500 steps/max7200 seconds); U-Net (session
  9860, `/tmp/paradis-unet.log`, cores24-31, 500 steps/max3600 seconds).
  Tracer assumptions: sigma5deg, amplitude1, initial(0,0), great-circle rotation68deg
  bearing45deg. U-Net is a reconstructed depth3/topwidth4 architecture, not recovered
  paper source; report actual parameters. Current script also allows separately
  labelled angular continuation if fixed-target optimization fails.
- Active compilation test: session23606, log `/tmp/paradis-compile.log`, cores8-11.
  It checks gradients/outputs before retaining compilation. Results written incrementally
  to `report/results/compile-tuning.json`.
- Setup and experiments so far began ~2026-10-09 00:33 UTC; long tracer runs near
  00:50 UTC. Use saved per-experiment elapsed times to audit budget.

## Remaining

- Finish compilation choice, launch paired weather training with fixed common count.
- Inspect tracer results, pursue a clearly labelled continuation run if needed within
  the four-hour tracer allocation. Do not fabricate successful transport.
- Verify native forecast writes a real Zarr output and checkpoint horizon changes work.
- Test static-cache output/gradient equivalence (beyond resume tests), truncation,
  compile behavior, final required checks. Add data-loading timings/memory measurements.
- Final isolated interleaved benchmarks, correct source/config provenance and budget.
- Write reproducible README, report/report.tex and report/report.pdf with actual figures
  and generated tables, inspect PDF. Report failed targets and unavailable GPU honestly.
- Commit/push authorized private repository and verify visibility; update paradis-setup
  status. Preserve original license and attribution. Do not commit datasets/checkpoints.
