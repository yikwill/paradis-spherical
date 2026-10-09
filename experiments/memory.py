"""Fresh-process peak resident memory for one matched benchmark variant."""
import argparse
import json
from pathlib import Path
import resource

import torch

from experiments.common import compact_config, synthetic_data
from tests.check_port import reference_model
from trainer import ParadisTask, configure_optimizers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=['upstream', 'conv2d', 'disco'])
    parser.add_argument('--upstream', default='/home/yikwill/paradis-upstream')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    imported_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cfg = compact_config('conv2d' if args.variant == 'upstream' else args.variant, 64, 128)
    cfg.model.static_hidden_size = 64
    cfg.model.bias_channels = 8 if args.variant == 'upstream' else 0
    dm, batch = synthetic_data(cfg, 121, 240, full_features=True)
    torch.manual_seed(42)
    reference = reference_model(args.upstream, dm, cfg) if args.variant == 'upstream' else None
    task = ParadisTask(dm, cfg, model=reference)
    if args.compile and args.variant != 'upstream':
        task.model.compile()
    optimizer, _ = configure_optimizers(task.model, cfg, 20)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        task.rollout(batch, backward=lambda loss: loss.backward())
        optimizer.step()
    result = dict(variant=args.variant, imported_peak_rss_kib=imported_rss,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  compiled=args.compile and args.variant != 'upstream',
                  note='Fresh process, three training updates; total resident peak includes Python, libraries, model construction and compilation. Four CPU threads.')
    Path(f'report/results/memory-{args.variant}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
