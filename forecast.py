import argparse
import logging
from omegaconf import OmegaConf
import torch

from trainer import ParadisTask, load_checkpoint, move_batch
from data.datamodule import Era5DataModule
from utils.system import setup_system
from utils.file_output import ZarrForecastWriter

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="Run forecasts with a trained model.")

    parser.add_argument("--config", help="Path to config YAML", required=True)
    parser.add_argument(
        "--checkpoint-path", help="Path to model checkpoint", required=True
    )
    parser.add_argument("--output-file", help="Output Zarr path", required=True)

    parser.add_argument("--root-dir", default=None, help="Override root dir")
    parser.add_argument(
        "--forecast-steps", type=int, default=40, help="Autoregressive forecast steps"
    )
    parser.add_argument(
        "--sampling-interval",
        type=str,
        default="36h",
        help='Dataset sampling interval, e.g. "36h"',
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Forecast start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Forecast end date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Prediction batch size"
    )
    parser.add_argument(
        "--num-devices", type=int, help="Number of devices", required=True
    )
    parser.add_argument(
        "--flush-every-n-steps", type=int, help="Write a forecast every n steps to reduce CPU memory usage", required=True,
    )

    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of dataloader workers"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.forecast.enable = True

    cfg.init.checkpoint_path = args.checkpoint_path
    cfg.forecast.output_file = args.output_file

    if args.root_dir is not None:
        cfg.dataset.root_dir = args.root_dir

    cfg.model.forecast_steps = args.forecast_steps
    cfg.dataset.sampling_interval = args.sampling_interval

    cfg.forecast.start_date = args.start_date
    cfg.forecast.end_date = args.end_date
    cfg.forecast.write_every_n = args.flush_every_n_steps

    cfg.compute.batch_size = args.batch_size
    cfg.compute.num_devices = args.num_devices

    cfg.compute.use_amp = False
    cfg.compute.num_workers = args.num_workers

    # Only supporting single node for now
    cfg.compute.num_nodes = 1

    # Restart continues the training from the current step
    cfg.init.restart = True

    datamodule = Era5DataModule(cfg)
    datamodule.setup(stage="predict")

    if args.num_devices != 1:
        raise ValueError("Forecast file writing currently supports one device")
    setup_system(cfg)
    device = torch.device("cuda" if cfg.compute.accelerator in ("gpu", "cuda") else "cpu")
    model = ParadisTask(datamodule, cfg).to(device).eval()
    load_checkpoint(cfg.init.checkpoint_path, model)
    writer = ZarrForecastWriter(cfg, datamodule.dataset)
    writer.init_store(datamodule.dataset)
    with torch.inference_mode():
        for batch in datamodule.predict_dataloader():
            predict_batch(model, move_batch(batch, device), writer)



def predict_batch(self, batch, writer):
    sample_indices, input_data, forcings, constant_data = batch

    dataset = self.datamodule.dataset

    num_forecast_steps = self.cfg.model.forecast_steps
    output_frequency = self.cfg.forecast.output_frequency
    write_every_n = self.cfg.forecast.get("write_every_n", num_forecast_steps)

    from utils.postprocessing import (
        denormalize_datasets,
        convert_cartesian_to_spherical_winds,
    )

    output_features = list(dataset.dyn_output_features)
    constants = constant_data[:, :1].permute(0, 1, 4, 2, 3)
    hidden_static = self.model.encode_static(constants[:, 0])

    chunk_buffer = []
    chunk_start_idx = None
    stored_step_idx = 0

    for step in range(num_forecast_steps):
        forcings_step = forcings[:, step].unsqueeze(1).permute(0, 1, 4, 2, 3)

        model_input = torch.cat(
            [input_data, forcings_step, constants], dim=2
        ).squeeze(1)
        output_data = self(model_input, hidden_static=hidden_static)

        input_data = self._autoregression_next_input(
            model_input, output_data
        ).unsqueeze(1)

        if step % output_frequency == 0:
            if chunk_start_idx is None:
                chunk_start_idx = stored_step_idx

            chunk_buffer.append(output_data.detach())
            stored_step_idx += 1

            if len(chunk_buffer) == write_every_n:
                chunk_tensor = torch.stack(chunk_buffer, dim=1).cpu()
                denormalize_datasets(None, chunk_tensor, dataset)

                chunk_np = chunk_tensor.numpy()
                convert_cartesian_to_spherical_winds(
                    dataset.lat, dataset.lon, self.cfg, chunk_np, output_features
                )

                writer.write_forecast_chunk(
                    forecast=chunk_np,
                    sample_indices=(
                        None
                        if sample_indices is None
                        else sample_indices.cpu().numpy()
                    ),
                    start_idx=chunk_start_idx,
                    dataset=dataset,
                )

                del chunk_tensor, chunk_np
                chunk_buffer.clear()
                chunk_start_idx = None

        del output_data

    if chunk_buffer:
        chunk_tensor = torch.stack(chunk_buffer, dim=1).cpu()
        denormalize_datasets(None, chunk_tensor, dataset)

        chunk_np = chunk_tensor.numpy()
        convert_cartesian_to_spherical_winds(
            dataset.lat, dataset.lon, self.cfg, chunk_np, output_features
        )

        writer.write_forecast_chunk(
            forecast=chunk_np,
            sample_indices=(
                None if sample_indices is None else sample_indices.cpu().numpy()
            ),
            start_idx=chunk_start_idx,
            dataset=dataset,
        )

    return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
