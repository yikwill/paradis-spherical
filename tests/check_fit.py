"""Check the complete training loop, including an interruption inside an epoch."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory

from omegaconf import OmegaConf
import torch

from data.datamodule import Era5DataModule
from tests.check_data import config, tiny_store
import trainer
from utils.system import setup_system


class Interrupted(Exception):
    pass


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        tiny_store(root)
        cfg = config(root)
        cfg.training.dataset.start_date = "2016-12-30"
        cfg.training.dataset.end_date = "2016-12-31"
        cfg.training.validation_dataset.start_date = "2017-01-01"
        cfg.training.validation_dataset.end_date = "2017-01-02"
        cfg.training.max_steps = 6
        cfg.training.accumulate_grad_batches = 2
        cfg.training.log_dir = str(root / "logs")
        cfg.training.experiment_name = "full"
        cfg.training.validation_dataset.validation_every_n_steps = 1
        cfg.compute.threads = 2
        data = Era5DataModule(cfg)
        data.setup("fit")
        setup_system(cfg)
        full = trainer.ParadisTask(data, cfg)
        trainer.fit(full, data, cfg)
        restarted_cfg = copy.deepcopy(cfg)
        restarted_cfg.training.experiment_name = "resumed"
        setup_system(restarted_cfg)
        interrupted = trainer.ParadisTask(data, restarted_cfg)
        save = trainer.save_checkpoint
        def stop(path, *args):
            save(path, *args)
            if Path(path).name == "last.pt" and args[-1]["step"] == 2:
                raise Interrupted()
        trainer.save_checkpoint = stop
        try:
            trainer.fit(interrupted, data, restarted_cfg)
        except Interrupted:
            pass
        finally:
            trainer.save_checkpoint = save
        restarted_cfg.init.checkpoint_path = str(root / "logs/resumed/last.pt")
        restarted_cfg.init.restart = True
        resumed = trainer.ParadisTask(data, restarted_cfg)
        progress = trainer.fit(resumed, data, restarted_cfg)
        assert progress["step"] == 6
        for first, second in zip(full.parameters(), resumed.parameters()):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
    print("PASS: complete training loop matches after interruption, checkpoint reload, and epoch transition")


if __name__ == "__main__":
    main()
