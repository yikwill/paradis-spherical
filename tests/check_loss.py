"""Loss weights must follow feature names even after common-first ordering."""
import torch
from experiments.common import compact_config, synthetic_data
from trainer import ParadisTask


def main():
    cfg=compact_config()
    dm,_=synthetic_data(cfg,full_features=True)
    criterion=ParadisTask(dm,cfg).loss_fn
    for i,name in enumerate(dm.output_name_order):
        if '_h' in name:
            variable,level=name.rsplit('_h',1)
            expected=cfg.training.variable_loss_weights.atmospheric[variable]*max(.2,int(level)/1000)
        else:
            expected=cfg.training.variable_loss_weights.surface[name]
        torch.testing.assert_close(criterion.feature_weights[i],torch.tensor(expected))
    print('PASS: all 97 pressure and variable weights follow the reordered feature names')


if __name__=='__main__':
    main()
