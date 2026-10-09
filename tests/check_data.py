"""Runnable temporal-window, normalization, constants, and cache regression checks."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import xarray as xr

from data.era5_dataset import ERA5Dataset
from experiments.common import compact_config


def tiny_store(root):
    times = np.arange(np.datetime64("2016-12-30T00"), np.datetime64("2017-01-03T00"), np.timedelta64(6, "h"))
    lat, lon = np.linspace(-90, 90, 9), np.arange(16) * 22.5
    values = np.broadcast_to(np.arange(len(times), dtype=np.float32)[:, None, None, None], (len(times), 1, 9, 16)).copy()
    dataset = xr.Dataset({"data": (("time", "features", "latitude", "longitude"), values)},
                         coords=dict(time=times, features=["2m_temperature"], latitude=lat, longitude=lon))
    for year in (2016, 2017):
        dataset.sel(time=str(year)).to_zarr(root / str(year), zarr_format=2)
    stats = xr.Dataset({key: ("features", [value]) for key, value in {"mean": 4., "std": 2., "min": 0., "max": 7.}.items()},
                       coords={"features": ["2m_temperature"]})
    stats.attrs.update(toa_radiation_std=1., toa_radiation_mean=0.)
    stats.to_zarr(root / "stats", zarr_format=2)
    constants = xr.Dataset({"land_sea_mask": (("latitude", "longitude"), np.full((9, 16), 0.5, dtype=np.float32))},
                           coords={"latitude": lat, "longitude": lon})
    constants.to_zarr(root / "constants", zarr_format=2)
    return times


def config(root):
    cfg = compact_config()
    cfg.dataset.root_dir = str(root)
    cfg.features.input.atmospheric = []
    cfg.features.output.atmospheric = []
    cfg.features.input.surface = ["2m_temperature"]
    cfg.features.output.surface = ["2m_temperature"]
    cfg.features.input.constants = ["sin_latitude", "land_sea_mask"]
    cfg.features.input.forcings = ["sin_time_of_day"]
    return cfg


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        times = tiny_store(root)
        cfg = config(root)
        for n_inputs in (1, 2):
            cfg.dataset.n_time_inputs = n_inputs
            kwargs = dict(root_dir=str(root), start_date="2016-12-30", end_date="2016-12-31",
                          forecast_steps=2, cfg=cfg, time_interval="6h")
            disk = ERA5Dataset(**kwargs)
            cached = ERA5Dataset(**kwargs, preload=True)
            assert len(disk) == 8 - n_inputs - 2 + 1
            for index in (0, len(disk) - 1):
                a, b = disk[index], cached[index]
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
                x, y, forcing, constants = b
                assert x.shape == (1, n_inputs, 9, 16)
                torch.testing.assert_close(x[0, :, 0, 0], (torch.arange(index, index+n_inputs).float()-4)/2)
                torch.testing.assert_close(y[:, 0, 0, 0], (torch.arange(index+n_inputs, index+n_inputs+2).float()-4)/2)
                assert forcing.shape == (2, 9, 16, n_inputs)
                torch.testing.assert_close(constants[0, :, 0, 0], torch.linspace(-90, 90, 9).deg2rad().sin())
                assert (constants[..., 1] == .5).all()
                for first, second in zip(b, cached[index]):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
            assert disk.time[-1] + 2 * np.timedelta64(6, "h") < np.datetime64("2017-01-01")
            validation = ERA5Dataset(str(root), "2017-01-01", "2017-01-02", 2, cfg=cfg)
            assert validation.time[0] - (n_inputs-1)*np.timedelta64(6, "h") >= np.datetime64("2017-01-01")
        cfg.dataset.prediction_delta = "12h"
        try:
            ERA5Dataset(**kwargs)
        except ValueError as error:
            assert "prediction_delta" in str(error)
        else:
            raise AssertionError("Inconsistent autoregressive history was accepted")
    print("PASS: input order, normalization, single/multiple inputs, split isolation, constants, cache immutability, invalid intervals")


if __name__ == "__main__":
    main()
