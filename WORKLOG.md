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

## Checkpoint 2026-10-09 01:15 UTC

- Core implementation committed as `58f5fbf`. Local commits use William Yik's
  GitHub identity with the GitHub noreply address; no global Git setting changed.
- Compilation output/gradient checks passed for both backends. Weather training
  now running sequentially for conv2d then DISCO, exactly 2000 updates each,
  four threads on CPUs8-11, compiled. Session77601; `/tmp/paradis-weather.log`.
  Actual data preload took163.1seconds under concurrent load; first100updatesdone.
- Final model checks include static-cache gradients for truncation1/2/3 and pass.
  Final real-ERA5 forecasting check passes after checkpoint metadata changes.
- Upstream fixture recaptured with original bias_channels and original constant
  names, using `/tmp/paradis-upstream-port-final.pt`. The native port comparison
  still gives exactly zero output/loss/gradient/update errors. Logs `/tmp/paradis-port-final.log`.
- Data-loader pilot passes matching-time dynamic/forcing equality exactly. Original
  loader requires original constants ending in latitude/longitude, so its complete
  original constant configuration is loaded from the reference worktree. Constants
  are deliberately excluded from equality checks. Pilot setup37.85s/10.90s and
  sample0.2241s/0.1878s upstream/optimized; repeat idle for final measurement.
  Raw pilot `report/results/data-benchmark-pilot.json`.
- Added `experiments/memory.py` for separate-process RSS; run all three variants
  with `--compile` after own training jobs finish. Final model benchmark also needs
  `--compile --repeats20 --profile` in isolation.
- Added final-format README, report/report.tex, and report/make_figures.py.
  These are drafts pending final numbers/figures, compilation and visual inspection.
- Added two identity-warmup tracer diagnostics after early continuation approached
  zero predictions. Same random initialization and objective: identity300updates,
  ramp600updates, fixed100updates, fourthreads. PARADIS session48116 onCPUs12-15,
  log `/tmp/paradis-tracer-warm.log`, max10800seconds; U-Net session79583 onCPUs0-3,
  log `/tmp/paradis-unet-warm.log`, max5400seconds. U-Net identityfit improves by
  step50; PARADIS only~step15 currently. These must remain separately labelled.
- Original active runs remain: fixed PARADIS22045/fixed U-Net9860/continuation
  PARADIS19819/continuation U-Net76441. Eightthread PARADIS continuation has slow
  ~25second updates; it may hit5400second limit before target68degrees. Report
  completed steps/actualangle, never pretend all protocols reached same endpoint.
- `paradis-setup/README.md` updated to ongoing status; historical handoff retained.
- `pip check` passes. Compileall in the project Python3.11 passes; host system
  python3 is too old for existing walrus syntax and is not a supported environment.

## Checkpoint 2026-10-09 01:38 UTC

- Read the actual v1 paper PDF after HTML extraction omitted Figure19 internals.
  Figure19 shows ConvAct + normalized two-convolution residual blocks. Added a
  closer U-Net (`--model unet-residual`), 46,481 parameters vs paper45,909. Assumed
  upstream ChannelNorm, fixed residual scale1, 3x3 convs, averagepool2, bilinearup.
  The prior plain U-Net30,601parameters is now auxiliary only. Forward/backward
  shape and finite-gradient smoke passed. Main report figure uses residual U-Net.
- Added three residual U-Net runs, all4threads: fixed500/max7200sec onCPUs24-27
  session91318/log `/tmp/paradis-unet-residual.log`; ramp300/400total/max7200sec
  onCPUs40-43 session74836/log `/tmp/paradis-unet-residual-continuation.log`;
  identity300+ramp600+fixed100/max10800sec onCPUs28-31 session21030/log
  `/tmp/paradis-unet-residual-warm.log`. Started~01:25UTC.
- Plain fixed U-Net completed500updates: relativeMSE0.99235,26.36min; plain ramp
  completed400updates with almostidentical error20.93min. They approximate the
  optimal constant field, not transport. Plain identity/ramp stillrunning.
- Full PARADIS identitywarmup begins to improve around150updates (~0.00177MSE),
  so keep running. Main fixedPARADIS~step250 currently~0.001779MSE.
- Weatherconv2d~step700, validation0.13204at600. DISCO has NOT started yet.
- Added `experiments/evaluate_weather.py`: final-checkpoint physical sixhour RMSE
  forz500/T2m pluspersistence; run afterbothweatherresults. Its small unit/latitude/
  ordering regression `tests/check_weather_metrics.py` passes.
- `report/make_figures.py` now builds residual-UNet vsPARADIS main/diagnostic maps,
  all9triallearningcurves/tables including lasttrainingangle, physicalRMSEtable,
  RSS/data-loadingnumbers, and optimalconstanttracerbaseline. Report/READMEupdated.
- Final stage controller now session6238, script `/tmp/paradis-final-stage.py`, log
  `/tmp/paradis-final-measurements.log`. Waits for TWOweatherprogress+NINEtracerJSONs,
  asserts2000weatherupdates, runs isolated20repeatcompiledbenchmark/profiles, data
  benchmark, threefreshmemoryprocesses, physicalevaluation, figures+latexmk. It MUST
  only run when our training is idle. It does not push or inspect PDF. Can interrupt
  waiting controller with write_stdin Ctrl-C (confirmed works withoutTTY).
- Git origin SSH failed on host /etc/ssh config ownership. Changed repo-local origin
  to HTTPS. `git -c credential.helper='!gh auth git-credential' ls-remote origin`
  succeeds. No pushes yet; use samehelperforfinalpush. No globalSSH/Git edits.

### Additional CPU optimization prototype (not adopted yet)

- TH stencil has ~2.85--3.06basis contributions peruniqueinputgridpoint. Folding
  coefficients for equal(row,column) canavoidrepeatvectorloads/dotproducts.
- Candidate C++ source `/tmp/paradis_disco_grouped.cpp`; original native source is
  STILL model/disco_cpu.cpp (commit58f5fbf). Candidate namespace is
  `paradis_disco_grouped` toavoidcolliding with liveprocesses. Sorted geometry order
  is argsort(psi_row_idx*(H*W)+psi_col_idx,stable=True); rowptrcountsunchanged.
- Trial `/tmp/paradis-tune-grouped.py` firstsequentialtimingswerenoisy(smallslower).
  Followup `/tmp/paradis-tune-grouped-interleaved.py` randomized20blocks, excluded
  first3: small61x120C32k5 median1.337x; large121x240C64k7 median1.733x faster.
  Float64value/graderrors~1e-15; fp32output/inputgradmax~3.6e-6; weightgradidentical.
  Rawresults `report/results/grouped-cpu-{tuning,interleaved}.json`.
- DO NOT update main `model/disco_cpu.py/.cpp` yet: the running weather process
  imported old SphericalConv code but lazily imports disco_cpu only when the SECOND
  backend constructs. Wait until `/tmp/paradis-weather.log` prints
  `disco: 111638 parameters` (and fit starts), soallactivejobs have loaded oldmodule.
  Then safely adopt candidate by replacing namespace backto`paradis_disco`, changing
  load extension cache name to`paradis_disco_cpu_v2` (neveroverwrite liveold.so), and
  changing SphericalConv geometry sort to(row,column). Active jobs willretainold
  loadedcode/library. Finalbench/evaluationwilluserevisedkernel. Explain revision
  difference inreport. Re-run model/fit/compilechecks afteradoptionbeforebenchmark.
- Need a permanent reproducible groupedtuningdriver afteradoption: retrieve oldC++
  from `git show58f5fbf:model/disco_cpu.cpp`, replace itsnamespace with uniquev1name,
  compile inTemporaryDirectory, compare oldrow-sortedvsnewcolumn-sorted kernel at
  identicalweights; no permanent duplicateC++source needed.
- Consider finalphysicalevaluation also computing aggregatevalidationloss and
  comparing against loggedfinalscore to confirm newkernel/compiled-vs-eager effects
  on trained checkpoints are negligible. Do not claim trainingusednewkernel.

## Checkpoint 2026-10-09 01:55 UTC

- Committed residual baseline/evaluation as `7a7b107`. Native model/trainer code is
  still identical to58f5fbf; foldedkernel notyetadopted. Wait untilweatherDISCOhas
  constructed before changinglazyCPUmodulefiles (see priorcheckpoint).
- Extended tests/check_distributed.py to autoregressive2/truncate1 and3/truncate2,
  with accumulation2, validation/checkpointing overepochs. Both2ranktests PASS;
  log `/tmp/paradis-ddp-ar.log`, session57716complete. No trainer changes needed.
- evaluate_weather nowalso re-evaluates aggregatevalidationloss and assertsclose
  tologgedfinalscore (rtol1e-5/atol1e-7), toverifynewkernel/eagermodeoncheckpoints.
  tests/check_weather_metrics stillpasses (session12739complete).
- Added `experiments/tune_grouped.py` (NOT RUN YET): permanentseeded interleaved
  old-vs-foldedkernel test. It retrievesoldC++from58f5fbf, compilesunderseparatenamespace,
  thenusescurrentmodelkernel. RunonlyAFTERadoption; otherwiseitwouldcompareoldtoold.
- Finalstagecontroller is NOW session1927, `/tmp/paradis-final-stage.py`, same log.
  It additionally waits for `report/results/folded-kernel-checks.json` (mustcreate
  onlyafterfoldedadoptionandmodel/fit/compilechecksPASS), and runs permanentgrouped
  tuning FIRST inisolation, thenbenchmark/data/RSS/quality/figures/latexasbefore.
- Benchmark nowrecords Gitrevision, CPUaffinity, and SHA256ofmodel/trainer/loss/driver
  sources. This edit iscurrentlyuncommitted. Report alsoincludeswarmupdiagnosticmap,
  explicitlylistsPARADIStracer+sinlat vsU-Nettraceronly, and retainsallprotocols.
- Weatherconv2d1400updates, val0.12125. PARADISidentitywarmup~291updates withMSE
  7.2e-7, abouttoenterangularramp. ResidualUNetwarmup~211updatesMSE2.7e-5.

## Automatic staging queued, 2026-10-09 02:05 UTC

- `/tmp/paradis-adopt-folded.py` is now running as session94249, log
  `/tmp/paradis-adopt-folded.log`, CPUs4-7. It waits for the first DISCOweather
  metricsrow (not justconstruction), verifiesnativefilesunchanged, then ATOMICALLY
  installs candidateC++underoriginalopnamespace, changesextensioncacheto_v2, and
  sortsSphericalConvstencilsby(row,column). DO NOT manuallyeditthesethreefiles while
  thisguardedscriptwaits; itwillabortifchanged. AllactiveDISCOprocesseswillalready
  havev1loaded; theircode/librariesremainstable.
- Afteradoption itrenamescompile-tuning.json tocompile-tuning-v1.json, runs
  check_model/check_fit/tune_compile/check_forecast/check_port into
  `/tmp/paradis-folded-check-{0..4}.log`, andonlyonALLsuccesswrites
  report/results/folded-kernel-checks.json. Ifitfails, inspect/fixbeforeresumingfinal
  stage. No commit/pushisautomatic. Finalstage session1927waits forthisflag+allruns.
- Reportdraftnowexplainssource-revisiondistinction andkernelcoefficientfolding;
  generatorwillproducekernel-table.texfromfinalisolatedtune_groupedresults.
- Weatherconv2d1700updates (~02:02UTC), nextbackendprobably~02:12UTC. TracerPARADIS
  warmupisnowramping andkeepsMSE~2.5e-5 at8degrees. ResidualU-Netwarmup~291updates.

## Folded kernel verified, 2026-10-09 02:20 UTC

- The guarded adoption completed after DISCO weather training had loaded version
  one. All five checks passed: operator/model/resume, fit interruption, compiled
  outputs and gradients, real-data forecasting, and the unchanged upstream port.
  `report/results/folded-kernel-checks.json` records source hashes and check times.
- A separate full-model longitude regression passes for coarsening-aligned shifts
  in both backends (maximum absolute error below 2e-7). DISCO operators enforce
  scalar pole rows, but inherited local pooling does not preserve this property
  throughout the full model. The report now states this limitation explicitly.
- Conv2d weather completed 2,000 updates, final validation loss 0.11407607.
  DISCO is running with the original loaded kernel. Final checkpoint evaluation
  will verify that the folded implementation reproduces the logged scores.
- Fixed-target PARADIS completed 500 updates with relative MSE 0.93464; the plain
  U-Net identity/ramp run and residual U-Net ramp run ended near the constant
  baseline. Identity-first PARADIS is still tracking its ramp at about 20 degrees.
- The report generator now enumerates the nine intended runs, avoiding accidental
  interpretation of its generated tracer-baselines.json as a training history.
  Full figure/PDF generation remains queued after all runs and isolated timing.

## Forecast compilation regression, 2026-10-09 03:01 UTC

- Forecasting ignored `compute.compile` even when the checkpoint configuration
  enabled it. The CLI now uses the same whole-model/modules choices as training.
- Expanded `tests.check_forecast` to invoke the actual argument parser and main
  function with compilation enabled, assert that Dynamo captures graphs, and
  compare every exported physical field against eager execution. It also retains
  the horizon, incremental-output, lead-time and overwrite checks.
- Running this regression with `forecast.py` from `9a235ec` fails specifically
  with "Saved compilation setting was ignored". The current implementation passes;
  log: `/tmp/paradis-forecast-compile-regression.log`. Compiled/eager physical-field
  tolerances are rtol=5e-5 and atol=5e-4. No training model or kernel code changed.
- The two-rank autoregressive DISCO test also passes with the folded kernel;
  log: `/tmp/paradis-ddp-folded-final.log`.
- Seven of nine tracer runs are complete. The no-warmup PARADIS ramp hit its time
  cap after 204 updates at 46.24 degrees; final-target relative MSE is 1.09117.
  Identity-first PARADIS remains accurate around 50 degrees. Weather losses at
  1,000 updates are 0.126824 (DISCO) and 0.126639 (Conv2d).
