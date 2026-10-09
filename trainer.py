"""Native PyTorch autoregressive training, validation, and resumable checkpoints."""
import contextlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.paradis import Paradis
from utils.loss import ParadisLoss


def build_adamw_param_groups(model, lr, weight_decay, gate_lr_mult=10.0):
    gates, base = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (gates if name.split(".")[0] in {"alpha_adv", "alpha_diff", "alpha_reac"}
             else base).append(parameter)
    groups = [dict(params=base, lr=lr, weight_decay=weight_decay)]
    if gates:
        groups.append(dict(params=gates, lr=lr * gate_lr_mult, weight_decay=0.0))
    return groups


def configure_optimizers(model, cfg, total_steps):
    settings = cfg.training
    opt = settings.optimizer
    if opt.name != "adamw":
        raise ValueError("This reproduction supports the native AdamW optimizer")
    parameters = (build_adamw_param_groups(model, opt.lr, opt.weight_decay)
                  if opt.get("custom_adamw", True) else model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=opt.lr, weight_decay=opt.weight_decay,
                                  betas=(opt.beta1, opt.beta2))
    schedules = settings.scheduler
    if sum(s.enabled for s in schedules.values()) != 1:
        raise ValueError("Exactly one learning-rate scheduler must be enabled")
    if schedules.one_cycle.enabled:
        s = schedules.one_cycle
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, total_steps=total_steps, max_lr=opt.lr,
            pct_start=s.warmup_pct_start, div_factor=s.lr_div_factor,
            final_div_factor=s.lr_final_div, anneal_strategy="cos")
    elif schedules.reduce_lr.enabled:
        s = schedules.reduce_lr
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=s.factor, patience=s.patience, threshold=s.threshold,
            threshold_mode=s.threshold_mode, min_lr=s.min_lr)
    else:
        s = schedules.wsd
        warmup = s.warmup if s.warmup >= 1 else s.warmup * total_steps
        decay = s.decay if s.decay >= 1 else s.decay * total_steps
        if warmup < 0 or decay < 0 or warmup + decay > total_steps:
            raise ValueError("Warmup and decay must fit inside the training budget")
        def multiplier(step):
            if step < warmup:
                return (step + 1) / warmup
            if step <= total_steps - decay:
                return 1.0
            return max(0, (total_steps - step) / decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


class ParadisTask(nn.Module):
    def __init__(self, datamodule, cfg, model=None):
        super().__init__()
        self.cfg = cfg
        self.datamodule = datamodule
        self.model = model if model is not None else Paradis(
            datamodule, cfg, datamodule.dataset.lat_rad_grid, datamodule.dataset.lon_rad_grid)
        self.num_common_features = datamodule.num_common_features
        self.n_inputs = cfg.dataset.n_time_inputs
        weights = []
        for name in datamodule.output_name_order:
            variable = re.sub(r"_h\d+$", "", name)
            source = (cfg.training.variable_loss_weights.atmospheric if "_h" in name
                      else cfg.training.variable_loss_weights.surface)
            weights.append(source[variable])
        kwargs = dict(
            lat_grid=datamodule.lat, pressure_levels=torch.tensor(cfg.features.pressure_levels),
            num_features=datamodule.num_out_features,
            num_surface_vars=len(cfg.features.output.surface),
            var_loss_weights=torch.tensor(weights), output_name_order=datamodule.output_name_order,
            delta_loss=cfg.training.loss_function.delta_loss,
            apply_latitude_weights=cfg.training.loss_function.lat_weights)
        self.loss_fn = ParadisLoss(loss_function=cfg.training.loss_function.type, **kwargs)
        validation_loss = cfg.training.loss_function.get("validation_loss")
        self.val_loss_fn = (ParadisLoss(loss_function=validation_loss, **kwargs)
                            if validation_loss else self.loss_fn)

    def forward(self, fields, hidden_static=None):
        if hidden_static is None:
            return self.model(fields)
        return self.model(fields, hidden_static=hidden_static)

    def _autoregression_next_input(self, inputs, outputs):
        common = outputs[:, :self.num_common_features]
        if self.n_inputs == 1:
            return common
        return torch.cat([inputs[:, self.num_common_features:self.num_common_features * self.n_inputs],
                          common], dim=1)

    def rollout(self, batch, backward=None, accumulation=1, validation=False):
        """Backpropagate each truncation chunk with the upstream loss scale."""
        inputs, targets, forcings, constants = batch
        inputs = inputs[:, 0]
        constants = constants[:, 0].permute(0, 3, 1, 2)
        forcings = forcings.permute(0, 1, 4, 2, 3)
        steps = targets.shape[1]
        detach_every = self.cfg.training.optimizer.get("detach_gradient_every") or steps
        criterion = self.val_loss_fn if validation else self.loss_fn
        total = inputs.new_zeros(())
        chunk = 0
        hidden_static = None
        for step in range(steps):
            if steps > 1 and hasattr(self.model, "encode_static") and step % detach_every == 0:
                hidden_static = self.model.encode_static(constants)
            fields = torch.cat([inputs, forcings[:, step], constants], dim=1)
            outputs = self(fields, hidden_static=hidden_static)
            loss = criterion(outputs, targets[:, step])
            total = total + loss.detach()
            chunk = chunk + loss / (steps * accumulation)
            inputs = self._autoregression_next_input(fields, outputs)
            if (step + 1) % detach_every == 0 or step + 1 == steps:
                if backward is not None:
                    backward(chunk)
                inputs = inputs.detach()
                chunk = 0
        return total / steps


def bare_model(task):
    return task.model.module if isinstance(task.model, DistributedDataParallel) else task.model


def checkpoint_metadata(task):
    cfg, dm = task.cfg, task.datamodule
    architecture = OmegaConf.to_container(cfg.model, resolve=True)
    for name in ("forecast_steps", "val_forecast_steps"):
        architecture.pop(name, None)
    statistics = {}
    for name in ("input_mean", "input_std", "output_mean", "output_std", "q_min", "q_max", "toa_rad_mean", "toa_rad_std"):
        if hasattr(dm.dataset, name):
            value = getattr(dm.dataset, name)
            statistics[name] = value.tolist() if isinstance(value, torch.Tensor) else float(value)
    return dict(model=architecture,
                features=OmegaConf.to_container(cfg.features, resolve=True),
                normalization=OmegaConf.to_container(cfg.normalization),
                statistics=statistics,
                n_inputs=cfg.dataset.n_time_inputs, prediction_delta=cfg.dataset.prediction_delta,
                latitude=dm.lat.tolist(), longitude=dm.lon.tolist())


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, task, optimizer, scheduler, scaler, progress):
    states = [rng_state()]
    if dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, rng_state())
        if dist.get_rank() != 0:
            return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {k.replace("_orig_mod.", ""): v for k, v in bare_model(task).state_dict().items()}
    state = dict(model=weights, optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=states,
                 progress=progress, metadata=checkpoint_metadata(task),
                 config=OmegaConf.to_container(task.cfg, resolve=True))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        torch.save(state, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def load_checkpoint(path, task, optimizer=None, scheduler=None, scaler=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["metadata"] != checkpoint_metadata(task):
        raise ValueError("Checkpoint model, features, normalization, or grid is incompatible")
    if optimizer is not None:
        old, new = state["config"], OmegaConf.to_container(task.cfg, resolve=True)
        for section, keys in {"training": ("optimizer", "scheduler", "max_steps", "max_epochs", "accumulate_grad_batches", "dataset", "validation_dataset", "loss_function", "variable_loss_weights"),
                              "dataset": ("sampling_interval", "time_resolution"),
                              "compute": ("batch_size", "use_amp", "threads", "compile", "gradient_checkpointing"), "init": ("seed",),
                              "model": ("forecast_steps",)}.items():
            if any(old[section][key] != new[section][key] for key in keys):
                raise ValueError(f"Resume configuration changed in {section}")
        world = dist.get_world_size() if dist.is_initialized() else 1
        if len(state["rng"]) != world:
            raise ValueError("Exact resume requires the same distributed world size")
    bare_model(task).load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"][dist.get_rank() if dist.is_initialized() else 0])
    return state["progress"]


def move_batch(batch, device):
    return tuple(x.to(device, non_blocking=device.type == "cuda") for x in batch)


@torch.no_grad()
def validate(task, loader, device, amp=False, limit=None):
    task.eval()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss = task.rollout(move_batch(batch, device), validation=True)
        total[0] += loss.double() * batch[0].shape[0]
        total[1] += batch[0].shape[0]
    if dist.is_initialized():
        dist.all_reduce(total)
    task.train()
    return (total[0] / total[1]).item() if total[1] else None


def fit(task, datamodule, cfg):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cfg.compute.accelerator in ("gpu", "cuda") else torch.device("cpu")
    task.to(device)
    amp = cfg.compute.use_amp
    if amp and device.type != "cuda":
        raise ValueError("Mixed precision is supported on CUDA; use compute.use_amp=false on CPU")
    accumulation = cfg.training.accumulate_grad_batches
    if accumulation < 1:
        raise ValueError("accumulate_grad_batches must be positive")
    seed = cfg.init.seed or 0
    sampler = DistributedSampler(datamodule.dataset, num_replicas=world, rank=rank, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(datamodule.dataset, batch_size=cfg.compute.batch_size, sampler=sampler,
                        num_workers=cfg.compute.num_workers, generator=generator,
                        pin_memory=device.type == "cuda", persistent_workers=cfg.compute.num_workers > 0)
    val_loader = DataLoader(datamodule.val_dataset, batch_size=cfg.compute.batch_size,
                           sampler=range(rank, len(datamodule.val_dataset), world),
                           num_workers=cfg.compute.val_num_workers, generator=generator,
                           pin_memory=device.type == "cuda")
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    if not updates_per_epoch:
        raise ValueError("Training split has no complete temporal windows")
    total_steps = cfg.training.max_steps
    if total_steps < 0:
        total_steps = cfg.training.max_epochs * updates_per_epoch
    optimizer, scheduler = configure_optimizers(task.model, cfg, total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    progress = dict(step=0, epoch=0, batch=0, best_loss=float("inf"), bad_validations=0, elapsed_seconds=0.0)
    if cfg.init.checkpoint_path:
        if cfg.init.restart:
            progress = load_checkpoint(cfg.init.checkpoint_path, task, optimizer, scheduler, scaler)
        else:
            load_checkpoint(cfg.init.checkpoint_path, task)
    if cfg.compute.compile is True:
        task.model.compile()
    elif cfg.compute.compile == "modules":
        task.model._compile()
    if world > 1:
        task.model = DistributedDataParallel(task.model, device_ids=[device.index] if device.type == "cuda" else None,
                                             broadcast_buffers=False)
    log_dir = Path(cfg.training.log_dir) / (cfg.training.experiment_name or "run")
    if (log_dir / "metrics.jsonl").exists() and not cfg.init.restart:
        raise FileExistsError(f"Training logs already exist: {log_dir}; choose a new experiment_name")
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, log_dir / "config.yaml", resolve=True)
        source_root = Path(__file__).resolve().parent
        provenance = dict(
            source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip(),
            source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source_root, text=True).strip()),
            torch=torch.__version__, threads=torch.get_num_threads(), world_size=world,
            cpu_affinity=sorted(os.sched_getaffinity(0)), parameters=sum(p.numel() for p in task.parameters()))
        (log_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    previous_elapsed = progress["elapsed_seconds"]
    started = time.perf_counter()
    every_val = cfg.training.validation_dataset.validation_every_n_steps
    limit_val = cfg.training.validation_dataset.validation_batches
    plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    if plateau and limit_val == 0:
        raise ValueError("ReduceLROnPlateau requires validation")
    task.train()
    while progress["step"] < total_steps:
        if cfg.training.max_epochs > 0 and progress["epoch"] >= cfg.training.max_epochs:
            break
        sampler.set_epoch(progress["epoch"])
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for index, batch in enumerate(loader):
            if index < progress["batch"]:
                continue
            update = (index + 1) % accumulation == 0 or index + 1 == len(loader)
            sync = task.model.no_sync() if world > 1 and not update else contextlib.nullcontext()
            with sync, torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = task.rollout(move_batch(batch, device), backward=lambda x: scaler.scale(x).backward(), accumulation=accumulation)
            accumulated += loss.item()
            if not update:
                continue
            scaler.unscale_(optimizer)
            if cfg.training.gradient_clip_val is not None:
                nn.utils.clip_grad_norm_(task.parameters(), cfg.training.gradient_clip_val, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not plateau:
                scheduler.step()
            progress["step"] += 1
            progress["batch"] = index + 1
            progress["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
            row = dict(step=progress["step"], epoch=progress["epoch"], train_loss=accumulated / min(accumulation, (index % accumulation) + 1),
                       lr=optimizer.param_groups[0]["lr"], elapsed_seconds=progress["elapsed_seconds"])
            accumulated = 0.0
            due = (every_val and progress["step"] % every_val == 0) or (not every_val and index + 1 == len(loader)) or progress["step"] == total_steps
            if due and limit_val != 0:
                value = validate(task, val_loader, device, amp, limit_val)
                row["val_loss"] = value
                if value is not None:
                    if plateau:
                        scheduler.step(value)
                    improved = value < progress["best_loss"]
                    progress["bad_validations"] = 0 if improved else progress["bad_validations"] + 1
                    if improved:
                        progress["best_loss"] = value
                        if cfg.training.checkpointing.enabled:
                            save_checkpoint(log_dir / "best.pt", task, optimizer, scheduler, scaler, progress)
            if rank == 0:
                with (log_dir / "metrics.jsonl").open("a") as output:
                    output.write(json.dumps(row) + "\n")
                if due or progress["step"] % cfg.training.log_every_n_steps == 0:
                    print(json.dumps(row), flush=True)
            if cfg.training.checkpointing.enabled and (due or progress["step"] % cfg.training.log_every_n_steps == 0):
                save_checkpoint(log_dir / "last.pt", task, optimizer, scheduler, scaler, progress)
            if progress["step"] >= total_steps or (cfg.training.early_stopping.enabled and progress["bad_validations"] >= cfg.training.early_stopping.patience):
                return progress
        progress["epoch"] += 1
        progress["batch"] = 0
    return progress
