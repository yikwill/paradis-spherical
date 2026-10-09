"""Compare real, preloaded ERA5 sample loading at matching initialization times."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from data.era5_dataset import ERA5Dataset
from experiments.weather import weather_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/yikwill/paradis-data')
    parser.add_argument('--upstream', default='/home/yikwill/paradis-upstream')
    args = parser.parse_args()
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('reference_dataset', Path(args.upstream)/'data/era5_dataset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = weather_config(args.root, 'conv2d', 20, 4)
    result = {}
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ('2017', 'stats', 'constants'):
            (root/name).symlink_to(Path(args.root)/name, target_is_directory=True)
        datasets = {}
        for name, cls in [('upstream', module.ERA5Dataset), ('optimized', ERA5Dataset)]:
            settings = copy.deepcopy(cfg)
            if name == 'upstream':
                settings.features.input.constants = OmegaConf.load(Path(args.upstream)/'config/paradis_settings.yaml').features.input.constants
            start = time.perf_counter()
            datasets[name] = cls(str(root), '2017-01-01T06:00:00', '2017-01-31T12:00:00', 1,
                                 cfg=settings, preload=True, time_interval='6h')
            result[name] = dict(setup_seconds=time.perf_counter()-start, sample_seconds=[])
        old, new = datasets.values()
        indices = np.random.default_rng(42).integers(3, len(new)-3, 40)
        max_error = 0.0
        for index in indices:
            old_index = int(np.flatnonzero(old.time == new.time[index])[0])
            samples = {}
            for name, at in [('upstream', old_index), ('optimized', int(index))]:
                start = time.perf_counter()
                samples[name] = datasets[name][at]
                result[name]['sample_seconds'].append(time.perf_counter()-start)
            # The revised coordinate input is deliberately different; dynamic fields
            # and temporal/solar forcings must agree at matching timestamps.
            for a, b in zip(samples['upstream'][:3], samples['optimized'][:3]):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
                max_error = max(max_error, (a-b).abs().max().item())
        for row in result.values():
            row['median_sample_seconds'] = float(np.median(row['sample_seconds'][2:]))
    result['dynamic_and_forcing_max_abs_error'] = max_error
    result['combined_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['note'] = 'Preloaded January 2017, 121x240, full features, four CPU threads; first two samples excluded from medians. Setup includes disk reads; filesystem cache is uncontrolled. Constants differ by design.'
    Path('report/results/data-benchmark.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
