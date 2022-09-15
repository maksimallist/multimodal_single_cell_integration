import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Callable, Union

import numpy
import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter


def is_json_serializable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


class Watcher:
    config = {}

    def __init__(self, *args, **kwargs):
        self.log(*args, **kwargs)

    @staticmethod
    def _create_log_struct(struct: Dict, name: str) -> Dict:
        if name not in struct:
            struct[name] = {}
        return struct[name]

    def log(self, *args, **kwargs):
        if len(args) != 0:
            log_structure = self.config
            for arg in args:
                log_structure = self._create_log_struct(log_structure, arg)
                for att, val in kwargs.items():
                    if is_json_serializable(val):
                        log_structure[att] = val
                    setattr(self, att, val)
        else:
            for att, val in kwargs.items():
                if is_json_serializable(val):
                    self.config[att] = val
                setattr(self, att, val)

    def rlog(self, *args, **kwargs):
        self.log(*args, **kwargs)
        values = tuple([v for v in kwargs.values()])
        if len(values) == 1:
            return values[0]
        elif len(values) > 1:
            return values
        else:
            return None

    def save_conf(self, save_path: Path):
        save_path = save_path.joinpath('exp_config.json')
        with save_path.open('w') as outfile:
            json.dump(self.config, outfile, indent=4)


class ExpWatcher(Watcher):
    checkpoints_folder = None
    _tensorboard_logs = None

    def __init__(self, exp_name: str, root: Path, *args, **kwargs):
        super(ExpWatcher, self).__init__(exp_name=exp_name, root=str(root), *args, **kwargs)
        date = datetime.now().strftime("%d.%m.%Y-%H.%M")
        self.log(exp_date=date)
        self.exp_root = root.joinpath(exp_name + '-' + date)
        self.writer = None
        self._experiments_preparation()

        # checkpoint callback
        self.counter = 0
        self.verbose = None
        self.mode = None
        self.best_met = None
        self.check_freq = None

    def _experiments_preparation(self):
        self.exp_root.mkdir(parents=True, exist_ok=False)
        # make folder for checkpoints
        self.checkpoints_folder = self.exp_root.joinpath('checkpoints')
        self.checkpoints_folder.mkdir()

        self.best_model_folder = self.checkpoints_folder.joinpath('best_model')
        self.best_model_folder.mkdir()
        self.best_model_save_path = str(self.best_model_folder.joinpath('best_model.pt'))

        # make folder for tb_logs
        self._tensorboard_logs = self.exp_root.joinpath(f"tb_logs")
        self._tensorboard_logs.mkdir()
        self.writer = SummaryWriter(log_dir=str(self._tensorboard_logs))

    def save_model(self, train_step: int, trainable_rule: Module, name: str):
        model_path = self.checkpoints_folder.joinpath("train_step_" + str(train_step))
        model_path.mkdir()
        torch.save(trainable_rule.state_dict(), str(model_path.joinpath(f"{name}.pth")))

    def add_model_checkpoint_callback(self,
                                      mode: str = 'min',
                                      check_freq: int = 5,
                                      verbose: int = 0):
        assert verbose in [0, 1], ValueError(f"The argument 'verbose' can only take values from the list: [0, 1], "
                                             f"but {verbose} was found.")
        assert mode in ['min', 'max'], ValueError(f"The argument 'mode' can only take values from the list: "
                                                  f"['min', 'max'], but '{mode}' was found.")

        self.log(mode=mode)
        if mode == 'max':
            self.log(best_met=-numpy.inf)
        else:
            self.log(best_met=numpy.inf)

        self.log(verbose=verbose)
        self.log(check_freq=check_freq)

    def is_model_better(self,
                        monitor: float,
                        step: int,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer):
        if self.mode == 'min':
            if monitor < self.best_met:
                self.counter += 1
        else:
            if monitor > self.best_met:
                self.counter += 1

        if self.counter == self.check_freq:
            self.best_met = monitor
            if self.verbose == 1:
                print(f"[ Saving best model weights for step: {step} |  Best met: {self.best_met} ]")

            self.counter = 0
            torch.save({'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       self.best_model_save_path)

    def save_config(self):
        super(ExpWatcher, self).save_conf(self.exp_root)

    def add_scalar(self, *args, **kwargs):
        self.writer.add_scalar(*args, **kwargs)
