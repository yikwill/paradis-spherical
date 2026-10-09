"""Download the agreed ERA5 subset and fit normalization on 2016 only.

python -m experiments.prepare_weather /home/yikwill/paradis-data
"""
import json
from pathlib import Path
import sys
import time

import dask
import numpy as np
from numcodecs import Blosc
from omegaconf import OmegaConf
import xarray as xr

from experiments.common import ROOT
from scripts.preprocess_dataset import compute_cartesian_wind
from data.forcings.toa_radiation import toa_radiation_stats

SOURCE = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(ROOT / "config/paradis_settings.yaml")
    remote = xr.open_zarr(SOURCE, consolidated=True, storage_options={"token": "anon"})
    static_names = ["geopotential_at_surface", "land_sea_mask", "slope_of_sub_gridscale_orography", "standard_deviation_of_orography"]
    raw_names = ["geopotential", "temperature", "specific_humidity", "vertical_velocity",
                 "u_component_of_wind", "v_component_of_wind", "10m_u_component_of_wind",
                 "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure", "total_precipitation_6hr"]
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    start = time.perf_counter()
    with dask.config.set(scheduler="threads", num_workers=8):
        if not (root / "constants" / ".zmetadata").exists():
            constants = remote[static_names].sortby("latitude").transpose("latitude", "longitude").load()
            for name in static_names:
                constants[name] = constants[name].astype("float32")
                constants[name].encoding = {}
                constants[name].attrs.update(mean=float(constants[name].mean()), std=float(constants[name].std()))
            constants.to_zarr(root / "constants", mode="w", zarr_format=2, consolidated=True)
        names = [f"{v}_h{p}" for v in cfg.features.output.atmospheric for p in cfg.features.pressure_levels] + list(cfg.features.output.surface)
        for year, end in [(2016, "2016-12-31T18:00:00"), (2017, "2017-01-31T18:00:00")]:
            destination = root / str(year)
            marker = root / f"{year}.complete"
            if marker.exists():
                continue
            subset = remote[raw_names].sel(time=slice(f"{year}-01-01", end), level=list(cfg.features.pressure_levels)).sortby("latitude")
            fields = compute_cartesian_wind(subset)
            arrays = []
            for name in names:
                if "_h" in name:
                    variable, level = name.rsplit("_h", 1)
                    arrays.append(fields[variable].sel(level=int(level), drop=True))
                else:
                    arrays.append(fields[name])
            stacked = xr.concat(arrays, dim=xr.IndexVariable("features", names)).transpose("time", "features", "latitude", "longitude").astype("float32")
            stacked.name = "data"
            stacked.encoding = {}
            stacked = stacked.chunk(dict(time=8, features=len(names), latitude=121, longitude=240))
            print(f"Downloading and transforming {year}: {stacked.shape}", flush=True)
            stacked.to_dataset().to_zarr(destination, mode="w", zarr_format=2, consolidated=True,
                                        encoding={"data": {"compressor": compressor}})
            marker.write_text("complete\n")
            print(f"Completed {year} after {time.perf_counter()-start:.1f}s", flush=True)
        if not (root / "stats" / ".zmetadata").exists():
            train = xr.open_zarr(root / "2016", consolidated=True)
            print("Computing training-only statistics", flush=True)
            reductions = {name: getattr(train.data, name)(dim=("time", "latitude", "longitude"), skipna=False)
                          for name in ("mean", "std", "min", "max")}
            stats = xr.Dataset(reductions).compute().astype("float32")
            if not all(np.isfinite(stats[k]).all() for k in stats) or not bool((stats["std"] > 0).all()):
                raise ValueError("Training data or its normalization contains invalid values")
            mean, std = toa_radiation_stats(train.time.values, train.latitude.values, train.longitude.values)
            stats.attrs.update(toa_radiation_mean=float(mean), toa_radiation_std=float(std),
                               normalization_start="2016-01-01", normalization_end_exclusive="2017-01-01")
            stats.to_zarr(root / "stats", mode="w", zarr_format=2, consolidated=True)
    manifest = dict(source=SOURCE, train=["2016-01-01", "2017-01-01"], validation=["2017-01-01", "2017-02-01"],
                    grid=[121, 240], features=names, normalization="training data only", compression="lossless zstd",
                    preparation_seconds=time.perf_counter()-start)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "report/results/data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
