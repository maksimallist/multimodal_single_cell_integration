from typing import Callable, Optional, Tuple, Union

import h5py
import hdf5plugin  # не удалять!
import torch
from torch.utils.data import Dataset


class FlowDataset(Dataset):
    features_name: str = "block0_values"
    col_name: str = 'block0_items'
    cell_id_name: str = "axis1"

    def __init__(self,
                 features_file: str,
                 targets_file: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 device: Optional[Union[str, torch.device]] = None):
        self.features, self.features_names, self.features_shape = self.get_hdf5_flow(features_file)
        if targets_file:
            self.targets, self.targets_names, self.targets_shape = self.get_hdf5_flow(targets_file)
            assert self.targets_shape[0] == self.features_shape[0], \
                AssertionError(f"Длины файлов фичей и таргетов не совпадают; "
                               f"features_file: {self.features_shape[0]}, targets_file: {self.targets_shape[0]};")
        else:
            self.targets = None

        self.device = device
        self.transform = transform
        self.target_transform = target_transform

    def get_hdf5_flow(self, file_path: str):
        file_flow = h5py.File(file_path, 'r')
        col_names = list(file_flow[list(file_flow.keys())[0]])
        assert self.features_name in col_names
        assert self.cell_id_name in col_names

        lines, features_shape = file_flow[list(file_flow.keys())[0]][self.features_name].shape
        data_flow = file_flow[list(file_flow.keys())[0]]

        return data_flow, col_names, (lines, features_shape)

    def __len__(self):
        return len(self.features['axis1'])

    def __getitem__(self, item: int) -> Union[Tuple[str, torch.Tensor], Tuple[str, torch.Tensor, torch.Tensor]]:
        x_cell_id = self.features[self.cell_id_name][item].decode("utf-8")
        if self.targets is not None:
            y_cell_id = self.targets[self.cell_id_name][item].decode("utf-8")
            assert x_cell_id == y_cell_id, AssertionError("Несовпадение порядка элементов в списке фичей и таргетов.")

            x = torch.from_numpy(self.features[self.features_name][item])
            y = torch.from_numpy(self.targets[self.features_name][item])

            if self.transform:
                x = self.transform(x)
            if self.target_transform:
                y = self.target_transform(y)
            if self.device:
                x, y = x.to(self.device), y.to(self.device)

            return x_cell_id, x, y
        else:
            x = torch.from_numpy(self.features[self.features_name][item])

            if self.transform:
                x = self.transform(x)
            if self.device:
                x = x.to(self.device)

            return x_cell_id, x
