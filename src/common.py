import importlib

import numpy as np
import torch


def get_cls_by_name(name: str) -> type:
    """Get class by its name and module path.
    Args:
        name (str): e.g., transformers:T5ForConditionalGeneration, modeling_t5:my_class
    Returns:
        type: found class for `name`
    """
    module_name, cls_name = name.split(':')
    return getattr(importlib.import_module(module_name), cls_name)


def get_bite_pos_emb(digit: int, pad_len: int, pad_side: str = 'right') -> np.array:
    sep = ','
    byte_string = np.base_repr(digit, base=2)
    byte_string = sep.join([byte_string[i] for i in range(len(byte_string))])
    byte_array = np.fromstring(byte_string, dtype=int, sep=sep)
    if byte_array.shape[0] < pad_len:
        if pad_side == 'right':
            byte_array = np.pad(byte_array, (0, pad_len - byte_array.shape[0]), 'constant', constant_values=0)
        elif pad_side == 'left':
            byte_array = np.pad(byte_array, (pad_len - byte_array.shape[0], 0), 'constant', constant_values=0)
        else:
            raise ValueError(f"The argument 'pad_side' can only take values from a list: ['right', 'left'], "
                             f"but {pad_side} was found.")

    return byte_array


def pearson_corr_loss(predict: torch.Tensor, target: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    predict = predict - torch.mean(predict, dim=1).unsqueeze(1)
    target = target - torch.mean(target, dim=1).unsqueeze(1)
    loss_tensor = -torch.sum(predict * target, dim=1) / (target.shape[-1] - 1)  # minus because we want gradient ascend

    if normalize:
        s1 = torch.sqrt(torch.sum(predict * predict, dim=1) / (predict.shape[-1] - 1))
        s2 = torch.sqrt(torch.sum(target * target, dim=1) / (target.shape[-1] - 1))
        loss_tensor = loss_tensor / s1 / s2

    return loss_tensor


def add_dimension(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.unsqueeze(tensor, dim=dim)


def kfold_split(dataset, folds: int):
    fold_sizes = [len(dataset) // folds] * (folds - 1) + [len(dataset) // folds + len(dataset) % folds]
    ds_folds = torch.utils.data.random_split(dataset, fold_sizes, generator=torch.Generator().manual_seed(42))
    for fold in range(folds):
        yield torch.utils.data.ConcatDataset(ds_folds[:fold] + ds_folds[fold + 1:]), ds_folds[fold]


def train_val_split(dataset, val_volume: float = 0.2):
    train_examples = len(dataset)
    val_num = int(train_examples * val_volume)
    folds = torch.utils.data.random_split(dataset, [train_examples - val_num, val_num], generator=torch.Generator())
    return folds[0], folds[1]


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.min_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss) -> None:
        if loss > self.min_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
