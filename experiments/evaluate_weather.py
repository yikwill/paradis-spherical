"""Six-hour physical RMSE for two interpretable fields and a persistence baseline."""
import argparse
import json
import math
from pathlib import Path

from omegaconf import OmegaConf
import torch

from data.datamodule import Era5DataModule
from trainer import ParadisTask, load_checkpoint
from utils.system import setup_system


@torch.inference_mode()
def evaluate(task, loader, feature_indices, scales, weights):
    squared = torch.zeros(2, len(feature_indices), dtype=torch.float64)
    count = 0
    aggregate = 0.0
    for inputs, targets, forcings, constants in loader:
        inputs = inputs[:, 0]
        fields = torch.cat([inputs, forcings[:, 0].permute(0, 3, 1, 2),
                            constants[:, 0].permute(0, 3, 1, 2)], dim=1)
        output = task(fields)
        aggregate += task.val_loss_fn(output, targets[:, 0]).item()*len(inputs)
        prediction = output[:, feature_indices]
        target = targets[:, 0, feature_indices]
        last_frame = inputs[:, (task.n_inputs-1)*task.num_common_features:task.n_inputs*task.num_common_features]
        persistence = last_frame[:, feature_indices]
        for index, value in enumerate((prediction, persistence)):
            error = (value-target).double()*scales[None, :, None, None]
            squared[index] += (error.square()*weights[None, None, :, None]).mean((-2,-1)).sum(0)
        count += len(inputs)
    return (squared/count).sqrt().tolist(), count, aggregate/count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/yikwill/paradis-data')
    parser.add_argument('--checkpoint', choices=['last', 'best'], default='last')
    args = parser.parse_args()
    result = dict(checkpoint=args.checkpoint, lead_hours=6, fields=['geopotential_h500', '2m_temperature'],
                  units=['m2 s-2', 'K'], scores={})
    data = None
    for backend in ('conv2d', 'disco'):
        directory = Path('logs/weather')/backend
        cfg = OmegaConf.load(directory/'config.yaml')
        cfg.dataset.root_dir = args.root
        cfg.compute.threads = 4
        setup_system(cfg)
        if data is None:
            # Only load January for this evaluation; statistics still come from 2016.
            cfg.training.dataset.start_date = cfg.training.validation_dataset.start_date
            cfg.training.dataset.end_date = cfg.training.validation_dataset.end_date
            data = Era5DataModule(cfg)
            data.setup('fit')
        task = ParadisTask(data, cfg).eval()
        load_checkpoint(directory/f'{args.checkpoint}.pt', task)
        assert cfg.normalization.standard
        indices = [data.output_name_order.index(name) for name in result['fields']]
        assert max(indices) < task.num_common_features
        scales = data.val_dataset.output_std[indices].double()
        weights = data.lat.deg2rad().cos().clamp_min(0)
        weights /= weights.mean()
        score, count, aggregate = evaluate(task, data.val_dataloader(), indices, scales, weights)
        metrics = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        logged = [row['val_loss'] for row in metrics if 'val_loss' in row]
        reference = logged[-1] if args.checkpoint == 'last' else min(logged)
        result['scores'][backend] = dict(rmse=score[0], persistence_rmse=score[1], windows=count,
                                         aggregate_loss=aggregate, logged_validation_loss=reference,
                                         aggregate_abs_difference=abs(aggregate-reference))
        print(backend, result['scores'][backend], flush=True)
        assert math.isclose(aggregate, reference, rel_tol=1e-5, abs_tol=1e-7), 'Checkpoint re-evaluation differs from logged validation'
    assert result['scores']['conv2d']['persistence_rmse'] == result['scores']['disco']['persistence_rmse']
    Path('report/results/weather-physical-rmse.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
