"""ERA5 dataset handling"""

import os
import re

import dask
import numpy
import torch
import xarray
from omegaconf import DictConfig

from data.forcings import time_forcings, toa_radiation
from utils.normalization import (
    normalize_humidity,
    normalize_precipitation,
    normalize_standard,
)


def _preprocess_era5_store(ds: xarray.Dataset) -> xarray.Dataset:
    coord_updates = {}

    if "latitude" in ds.coords:
        coord_updates["latitude"] = ds.latitude.astype("float64").round(6)

    if "longitude" in ds.coords:
        coord_updates["longitude"] = ds.longitude.astype("float64").round(6)

    if coord_updates:
        ds = ds.assign_coords(**coord_updates)

    for coord in ("features", "latitude", "longitude"):
        if coord in ds.coords and coord in ds.dims and ds.sizes.get(coord, 0) > 1:
            ds = ds.sortby(coord)

    return ds

class ERA5Dataset(torch.utils.data.Dataset):
    """Prepare and process ERA5 dataset for Pytorch."""

    def __init__(
        self,
        root_dir: str,
        start_date: str,
        end_date: str,
        forecast_steps: int,
        dtype: torch.dtype = torch.float32,
        preload: bool = False,  # Whether to preload the dataset
        cfg: DictConfig = DictConfig({}),
        time_interval: str = None,
        prediction_stage=False,
    ) -> None:

        self.cfg = cfg
        features_cfg = cfg.features
        self.preload = preload
        self.eps = 1e-12
        self.root_dir = root_dir
        self.forecast_steps = forecast_steps
        self.dtype = dtype
        self.forcing_inputs = features_cfg.input.forcings
        self.concat_input = cfg.dataset.n_time_inputs > 1
        self.n_time_inputs = cfg.dataset.n_time_inputs if self.concat_input else 1
        self.custom_normalization = not cfg.normalization.standard
        self.prediction_stage = prediction_stage

        # Lazy open this dataset
        ds = xarray.open_mfdataset(
            os.path.join(root_dir, "*[0-9]"),
            chunks={},
            engine="zarr",
            preprocess=_preprocess_era5_store,
            join="exact",
        )

        if ds.latitude.values[0] > ds.latitude.values[-1]:
            ds = ds.sortby("latitude")

        if ds.longitude.values[0] > ds.longitude.values[-1]:
            ds = ds.sortby("longitude")

        # Add stats to data array
        ds_stats = xarray.open_dataset(
            os.path.join(self.root_dir, "stats"), engine="zarr"
        )

        # Store them in main dataset for easier processing
        ds["mean"] = ds_stats["mean"]
        ds["std"] = ds_stats["std"]
        ds["max"] = ds_stats["max"]

        ds["min"] = ds_stats["min"]
        ds.attrs["toa_radiation_std"] = ds_stats.attrs["toa_radiation_std"]
        ds.attrs["toa_radiation_mean"] = ds_stats.attrs["toa_radiation_mean"]

        # Split bounds apply to every input and target, not only initialization times.
        if "T" not in start_date:
            start_date += "T00:00:00"
        if end_date is not None and "T" not in end_date:
            end_date += "T23:59:59"
        self.time_resolution = time_resolution = int(cfg.dataset.time_resolution[:-1])
        interval_hours = int((time_interval or cfg.dataset.time_resolution)[:-1])
        if time_resolution <= 0 or interval_hours < time_resolution or interval_hours % time_resolution:
            raise ValueError("Sampling interval must be a positive multiple of the native time step")
        if cfg.dataset.prediction_delta != cfg.dataset.time_resolution:
            raise ValueError("Autoregressive history requires prediction_delta=time_resolution")
        self.interval_steps = interval_hours // time_resolution
        self.prediction_shift = 0
        step = numpy.timedelta64(time_resolution, "h")
        if prediction_stage:
            origins = ds.sel(time=slice(start_date, end_date, self.interval_steps)).time.values
            first = numpy.datetime64(start_date) - (self.n_time_inputs - 1) * step
            last = origins[-1] + (forecast_steps - 1) * step if len(origins) else first
            ds = ds.sel(time=slice(first, last))
            self.length = len(origins)
            self.time = origins
        else:
            ds = ds.sel(time=slice(start_date, end_date))
            count = max(0, ds.sizes["time"] - self.n_time_inputs - forecast_steps + 1)
            self.length = len(range(0, count, self.interval_steps))
            indices = numpy.arange(0, count, self.interval_steps) + self.n_time_inputs - 1
            self.time = ds.time.values[indices]
        if not self.length:
            raise ValueError("Date range contains no complete temporal windows")
        if not numpy.all(numpy.diff(ds.time.values) == step):
            raise ValueError("Dataset timestamps must be contiguous at the configured resolution")
        if prediction_stage and ds.time.values[0] != first:
            raise ValueError("Forecast initialization is missing its input history")
        self.ds_loader = ds.sel(time=self.time)

        # Extract latitude and longitude to build the graph
        self.lat = torch.from_numpy(ds.latitude.values.copy())
        self.lon = torch.from_numpy(ds.longitude.values.copy())
        self.lat_size = len(self.lat)
        self.lon_size = len(self.lon)
        self.pressure_levels = features_cfg.pressure_levels

        # Store the size of the grid (lat * lon)
        self.grid_size = ds.latitude.size * ds.longitude.size

        # Setup input and output features based on config
        input_atmospheric = [
            variable + f"_h{level}"
            for variable in features_cfg.input.atmospheric
            for level in features_cfg.pressure_levels
        ]

        output_atmospheric = [
            variable + f"_h{level}"
            for variable in features_cfg.output.atmospheric
            for level in features_cfg.pressure_levels
        ]

        # Update feature counts
        common_features = list(
            filter(
                lambda x: x in input_atmospheric + features_cfg["input"]["surface"],
                output_atmospheric + features_cfg["output"]["surface"],
            )
        )

        self.num_common_features = len(common_features)

        # Constant input variables
        ds_constants = xarray.open_dataset(
            os.path.join(root_dir, "constants"), engine="zarr"
        ).compute()  # Definitely preload constants

        # Convert lat/lon to radians
        lat_rad = torch.deg2rad(self.lat).to(self.dtype)
        lon_rad = torch.deg2rad(self.lon).to(self.dtype)
        self.lat_rad_grid, self.lon_rad_grid = torch.meshgrid(
            lat_rad, lon_rad, indexing="ij"
        )

        if len(lat_rad) != len(ds_constants.latitude.values):
            ds_constants = ds_constants.sel(latitude=ds.latitude, longitude=ds.longitude)

        if ds_constants.latitude.values[0] > ds_constants.latitude.values[-1]:
            ds_constants = ds_constants.sortby("latitude")

        if ds_constants.longitude.values[0] > ds_constants.longitude.values[-1]:
            ds_constants = ds_constants.sortby("longitude")

        # Use zscore to normalize the following variables
        normalize_const_vars = {
            "geopotential_at_surface",
            "slope_of_sub_gridscale_orography",
            "standard_deviation_of_orography",
        }

        constants = []
        forbidden = {"lon_spacing", "cos_latitude", "cos_longitude", "sin_longitude", "latitude", "longitude"}
        if forbidden.intersection(features_cfg.input.constants) or "sin_latitude" not in features_cfg.input.constants:
            raise ValueError("Use sin_latitude as the sole explicit coordinate input")
        for var in features_cfg.input.constants:
            if var == "sin_latitude":
                array = torch.sin(self.lat_rad_grid)
            else:
                array = torch.from_numpy(ds_constants[var].values).to(self.dtype)
                if var in normalize_const_vars:
                    array = (array - ds_constants[var].attrs["mean"]) / ds_constants[var].attrs["std"]
            constants.append(array)
        self.constant_data = torch.stack(constants, dim=-1).unsqueeze(0)

        # Store these for access in forecaster
        self.ds_constants = ds_constants

        # Order them so that common features are placed first
        input_features = input_atmospheric + list(features_cfg.input.surface)
        output_features = output_atmospheric + list(features_cfg.output.surface)

        common_features = [f for f in output_features if f in input_features]

        output_only_features = [f for f in output_features if f not in input_features]
        input_only_features = [f for f in input_features if f not in output_features]

        self.dyn_input_features = common_features + input_only_features
        self.dyn_output_features = common_features + output_only_features

        # Store the number of dynamic features without concatenation
        self.num_dyn_inputs_single = len(self.dyn_input_features)

        # Inputs must exist because they are required by the model.
        ds_input = ds.sel(features=self.dyn_input_features)

        ds_output = ds.sel(features=self.dyn_output_features)

        self.ds_loader = self.ds_loader.sel(features=self.dyn_input_features)
        self.ds_loader = self.ds_loader.transpose("time", "latitude", "longitude", "features")

        # Pre-select the features in the right order
        if self.preload:
            cells = ds.sizes["time"] * self.lat_size * self.lon_size
            # Include two copies for Dask materialization plus cached forcings.
            estimated = cells * 4 * (2 * (len(input_features) + len(output_features))
                                     + len(self.forcing_inputs) * self.n_time_inputs)
            if estimated > cfg.dataset.get("preload_max_gb", 64) * 1024**3:
                raise ValueError("Preloading exceeds dataset.preload_max_gb; disable preload or raise the explicit budget")
            ds_input = ds_input.compute()
            ds_output = ds_output.compute()

        # Concatenate dynamic input features as many times as needed
        if self.concat_input:
            self.dyn_input_features *= self.n_time_inputs

        # Fetch data
        self.ds_input = ds_input["data"].transpose("time", "latitude", "longitude", "features")
        self.ds_output = ds_output["data"].transpose("time", "latitude", "longitude", "features")

        # Get the indices to apply custom normalizations
        self._prepare_normalization(ds_input, ds_output)

        # Calculate the final number of input and output features after preparation
        # Number of dynamic inputs
        self.num_in_dyn_features = (
            len(self.dyn_input_features) + len(self.forcing_inputs) * self.n_time_inputs
        )

        # Number of static features
        self.num_in_static_features = self.constant_data.shape[-1]

        # Number of total inputs
        self.num_in_features = self.num_in_dyn_features + self.num_in_static_features

        # Number of total outputs
        self.num_out_features = len(self.dyn_output_features)

        # Cache forcing values once; normalization is independent of the sample.
        forcing_steps = ds.sizes["time"] - self.n_time_inputs + 1
        self.forcing_data = self._compute_forcings(ds_input, forcing_steps) if self.preload else None
        if self.preload:
            # Already in memory: leave xarray and Dask out of __getitem__.
            self._input_array = numpy.asarray(self.ds_input.data)
            self._output_array = numpy.asarray(self.ds_output.data)


        # Store the mean and standard deviation of quantities to be reported
        if not cfg.forecast.enable and cfg.training.reports.enable:
            report_features = ds_output.sel(features=cfg.training.reports.features)
            self.report_stats = {
                "mean": report_features["mean"].values,
                "std": report_features["std"].values,
            }


    def __len__(self):
        # Do not yield a value for the last time in the dataset since there
        # is no future data
        return self.length

    def __getitem__(self, ind: int):
        if ind < 0 or ind >= self.length:
            raise IndexError(ind)
        offset = ind * self.interval_steps
        stop_input = offset + self.n_time_inputs
        stop_output = stop_input + self.forecast_steps
        if self.preload:
            x = self._input_array[offset:stop_input]
            y = None if self.prediction_stage else self._output_array[stop_input:stop_output]
        else:
            x = self.ds_input.data[offset:stop_input]
            y = None if self.prediction_stage else self.ds_output.data[stop_input:stop_output]
            x, y = dask.compute(x, y, scheduler="synchronous")
        # Copy only the requested window: normalization must not mutate the cache.
        x = torch.tensor(x, dtype=self.dtype).permute(1, 2, 0, 3)
        x = x.reshape(1, self.lat_size, self.lon_size, -1)
        y = None if y is None else torch.tensor(y, dtype=self.dtype)
        self._apply_normalization(x, y)
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.forcing_data is None:
            times = self.ds_input.isel(time=slice(offset, stop_input + self.forecast_steps - 1))
            forcings = self._compute_forcings(times, self.forecast_steps)
        else:
            forcings = self.forcing_data[offset:offset + self.forecast_steps]
        if self.prediction_stage:
            return ind, x, forcings, self.constant_data
        return x, y.permute(0, 3, 1, 2).contiguous(), forcings, self.constant_data

    def _prepare_normalization(
        self, ds_input: xarray.Dataset, ds_output: xarray.Dataset
    ) -> None:
        """
        Prepare indices and statistics for normalization in a vectorized fashion.

        This method identifies indices for specific types of features
        (e.g., precipitation, humidity, and others) for both input and output
        datasets, converts them into PyTorch tensors, and retrieves
        mean and standard deviation values for z-score normalization.

        Parameters:
            ds_input: xarray.Dataset
                Input dataset containing mean and standard deviation values.
            ds_output: xarray.Dataset
                Output dataset containing mean and standard deviation values.
        """

        # Initialize lists to store indices for each feature type
        self.norm_precip_in = []
        self.norm_humidity_in = []
        self.norm_zscore_in = []

        self.norm_precip_out = []
        self.norm_humidity_out = []
        self.norm_zscore_out = []

        # Process dynamic input features
        for i, feature in enumerate(self.dyn_input_features):
            feature_name = re.sub(
                r"_h\d+$", "", feature
            )  # Remove height suffix (e.g., "_h10")
            if feature_name == "total_precipitation_6hr" and self.custom_normalization:
                self.norm_precip_in.append(i)
            elif feature_name == "specific_humidity" and self.custom_normalization:
                self.norm_humidity_in.append(i)
            else:
                self.norm_zscore_in.append(i)

        # Process dynamic output features
        for i, feature in enumerate(self.dyn_output_features):
            feature_name = re.sub(
                r"_h\d+$", "", feature
            )  # Remove height suffix (e.g., "_h10")
            if feature_name == "total_precipitation_6hr" and self.custom_normalization:
                self.norm_precip_out.append(i)
            elif feature_name == "specific_humidity" and self.custom_normalization:
                self.norm_humidity_out.append(i)
            else:
                self.norm_zscore_out.append(i)

        # Convert lists of indices to PyTorch tensors for efficient indexing
        if self.custom_normalization:
            self.norm_precip_in = torch.tensor(self.norm_precip_in, dtype=torch.long)
            self.norm_precip_out = torch.tensor(self.norm_precip_out, dtype=torch.long)
            self.norm_humidity_in = torch.tensor(
                self.norm_humidity_in, dtype=torch.long
            )
            self.norm_humidity_out = torch.tensor(
                self.norm_humidity_out, dtype=torch.long
            )
        self.norm_zscore_in = torch.tensor(self.norm_zscore_in, dtype=torch.long)
        self.norm_zscore_out = torch.tensor(self.norm_zscore_out, dtype=torch.long)

        # Retrieve mean and standard deviation values for z-score normalization
        self.input_mean = torch.tensor(ds_input["mean"].data, dtype=self.dtype)
        self.input_std = torch.tensor(ds_input["std"].data, dtype=self.dtype)
        self.input_max = torch.tensor(ds_input["max"].data, dtype=self.dtype)
        self.input_min = torch.tensor(ds_input["min"].data, dtype=self.dtype)

        self.output_mean = torch.tensor(ds_output["mean"].data, dtype=self.dtype)
        self.output_std = torch.tensor(ds_output["std"].data, dtype=self.dtype)
        self.output_max = torch.tensor(ds_output["max"].data, dtype=self.dtype)
        self.output_min = torch.tensor(ds_output["min"].data, dtype=self.dtype)

        # Keep only statistics of variables that require standard normalization
        self.input_mean = self.input_mean[
            self.norm_zscore_in % self.num_dyn_inputs_single
        ]
        self.input_std = self.input_std[
            self.norm_zscore_in % self.num_dyn_inputs_single
        ]
        self.output_mean = self.output_mean[self.norm_zscore_out]
        self.output_std = self.output_std[self.norm_zscore_out]

        # Prepare variables required in custom normalization
        if self.custom_normalization:

            if len(self.norm_humidity_in) > 0:
                # Maximum and minimum specific humidity in dataset
                self.q_max = torch.max(
                    self.input_max[self.norm_humidity_in % self.num_dyn_inputs_single]
                ).detach()
                self.q_min = torch.min(
                    self.input_min[self.norm_humidity_in % self.num_dyn_inputs_single]
                ).detach()
                if self.q_min < self.eps:
                    self.q_min = torch.tensor(self.eps).detach()
            else:
                self.q_max = torch.tensor(0.0).detach()
                self.q_min = torch.tensor(self.eps).detach()

        # Extract the toa_radiation mean and std
        self.toa_rad_std = ds_input.attrs["toa_radiation_std"]
        self.toa_rad_mean = ds_input.attrs["toa_radiation_mean"]


    def _apply_normalization(
        self, input_data: torch.Tensor, output_data: torch.Tensor = None
    ) -> None:

        # Apply custom normalizations to input
        if self.custom_normalization:
            input_data[..., self.norm_precip_in] = normalize_precipitation(
                input_data[..., self.norm_precip_in]
            )
            input_data[..., self.norm_humidity_in] = normalize_humidity(
                input_data[..., self.norm_humidity_in], self.q_min, self.q_max, self.eps
            )

            if output_data is not None:
                # Apply custom normalizations to output
                output_data[..., self.norm_precip_out] = normalize_precipitation(
                    output_data[..., self.norm_precip_out]
                )
                output_data[..., self.norm_humidity_out] = normalize_humidity(
                    output_data[..., self.norm_humidity_out],
                    self.q_min,
                    self.q_max,
                    self.eps,
                )

        # Apply standard normalizations to input and output
        input_data[..., self.norm_zscore_in] = normalize_standard(
            input_data[..., self.norm_zscore_in],
            self.input_mean,
            self.input_std,
        )

        if output_data is not None:
            output_data[..., self.norm_zscore_out] = normalize_standard(
                output_data[..., self.norm_zscore_out],
                self.output_mean,
                self.output_std,
            )


    def _compute_forcings(self, input_data: xarray.Dataset, steps: int) -> torch.Tensor:
        """Computes forcing parameters based in input_data array"""

        forcings_time_ds = time_forcings(input_data["time"].values)

        forcings = []
        for var in self.forcing_inputs:
            if var == "toa_incident_solar_radiation":
                toa_rad = toa_radiation(
                    input_data["time"].values,
                    self.lat.cpu().numpy(),
                    self.lon.cpu().numpy(),
                )

                toa_rad = torch.tensor(
                    (toa_rad - self.toa_rad_mean) / self.toa_rad_std,
                    dtype=self.dtype,
                )
                toa_rad = toa_rad.unfold(0, self.n_time_inputs, 1)

                forcings.append(toa_rad)
            else:
                # Get the time forcings
                if var in forcings_time_ds:
                    var_ds = forcings_time_ds[var]
                    var_forcing = torch.tensor(var_ds.data, dtype=self.dtype)
                    var_forcing = (
                        var_forcing.unfold(0, self.n_time_inputs, 1)
                        .view(steps, 1, 1, self.n_time_inputs)
                        .expand(steps, self.lat_size, self.lon_size, self.n_time_inputs)
                    )

                    forcings.append(var_forcing)

        if len(forcings) > 0:
            return torch.cat(forcings, dim=-1)
        return torch.empty(steps, self.lat_size, self.lon_size, 0)
