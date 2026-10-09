"""Small, deterministic configurations shared by checks and experiments."""

from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def compact_config(backend="conv2d", channels=8, static_channels=8):
    cfg = OmegaConf.load(ROOT / "config/paradis_settings.yaml")
    cfg.model.latent_size = channels
    cfg.model.velocity_vectors = channels // 2
    cfg.model.num_layers = 2
    cfg.model.spatial_conv = backend
    cfg.model.static_latent_size = static_channels
    cfg.model.static_hidden_size = static_channels
    for key in ("velocity_net", "reaction", "output_proj"):
        cfg.model.physblock[key].hidden_dim = channels
    cfg.compute.compile = False
    cfg.compute.use_amp = False
    cfg.compute.accelerator = "cpu"
    cfg.compute.num_workers = 0
    cfg.compute.val_num_workers = 0
    cfg.training.reports.enable = False
    cfg.training.reports.features = []
    cfg.training.max_steps = 20
    cfg.training.scheduler.wsd.warmup = 2
    cfg.training.scheduler.wsd.decay = 0.2
    return cfg


def synthetic_data(cfg, nlat=17, nlon=32, full_features=False):
    """Metadata and a batch using the real dataset's channel ordering."""
    if not full_features:
        cfg.features.pressure_levels = [500, 850]
        cfg.features.input.atmospheric = ["geopotential", "temperature"]
        cfg.features.output.atmospheric = ["geopotential", "temperature"]
        cfg.features.input.surface = ["2m_temperature"]
        cfg.features.output.surface = ["2m_temperature"]
        cfg.features.input.forcings = ["sin_time_of_day"]
    names = [f"{v}_h{p}" for v in cfg.features.output.atmospheric
             for p in cfg.features.pressure_levels] + list(cfg.features.output.surface)
    in_names = [f"{v}_h{p}" for v in cfg.features.input.atmospheric
                for p in cfg.features.pressure_levels] + list(cfg.features.input.surface)
    common = [n for n in names if n in in_names]
    names = common + [n for n in names if n not in common]
    in_names = common + [n for n in in_names if n not in common]
    n_forcings = len(cfg.features.input.forcings) * cfg.dataset.n_time_inputs
    lat = torch.linspace(-90, 90, nlat, dtype=torch.float64)
    lon = torch.arange(nlon, dtype=torch.float64) * (360 / nlon)
    grids = torch.meshgrid(lat.float().deg2rad(), lon.float().deg2rad(), indexing="ij")
    ds = SimpleNamespace(
        num_in_dyn_features=len(in_names) * cfg.dataset.n_time_inputs + n_forcings,
        num_in_static_features=len(cfg.features.input.constants),
        lat_rad_grid=grids[0], lon_rad_grid=grids[1],
        dyn_input_features=in_names * cfg.dataset.n_time_inputs,
    )
    dm = SimpleNamespace(dataset=ds, num_common_features=len(common),
                         num_out_features=len(names), output_name_order=names,
                         lat=lat, lon=lon)
    generator = torch.Generator().manual_seed(1234)
    def rand(*shape):
        return torch.randn(*shape, generator=generator)
    batch = (rand(1, 1, len(in_names) * cfg.dataset.n_time_inputs, nlat, nlon),
             rand(1, cfg.model.forecast_steps, len(names), nlat, nlon),
             rand(1, cfg.model.forecast_steps, nlat, nlon, n_forcings),
             rand(1, 1, nlat, nlon, ds.num_in_static_features))
    return dm, batch
