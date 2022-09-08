from typing import Callable, Optional, Tuple, Union

import h5py
import hdf5plugin  # не удалять!
import pandas as pd
import torch
from torch.utils.data import Dataset


class FlowDataset(Dataset):
    _reserve_name: str = "block0_values"

    def __init__(self,
                 features_file: str,
                 targets_file: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 device: Optional[Union[str, torch.device]] = None):
        self.features, self.features_names, self.features_shape = self.get_hdf5_flow(features_file)
        if targets_file:
            self.targets, self.targets_names, self.targets_shape = self.get_hdf5_flow(targets_file)
            assert self.targets_shape[0] == self.features_shape[0]
        else:
            self.targets = targets_file

        self.device = device
        self.transform = transform
        self.target_transform = target_transform

    def get_hdf5_flow(self, file_path: str):
        file_flow = h5py.File(file_path, 'r')
        col_names = list(file_flow[list(file_flow.keys())[0]])
        assert self._reserve_name in col_names

        lines, features_shape = file_flow[list(file_flow.keys())[0]]["block0_values"].shape
        data_flow = file_flow[list(file_flow.keys())[0]]

        return data_flow, col_names, (lines, features_shape)

    def __len__(self):
        return len(self.features['axis1'])

    def __getitem__(self, item: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.targets is not None:
            x = torch.from_numpy(self.features['block0_values'][item])
            y = torch.from_numpy(self.targets['block0_values'][item])

            if self.transform:
                x = self.transform(x)
            if self.target_transform:
                y = self.target_transform(y)
            if self.device:
                x, y = x.to(self.device), y.to(self.device)

            return x, y
        else:
            x = torch.from_numpy(self.features['block0_values'][item])

            if self.transform:
                x = self.transform(x)
            if self.device:
                x = x.to(self.device)

            return x


class MyDataset(Dataset):
    def __init__(self,
                 features_file: str,
                 start_pos: int,
                 load_pos: int,
                 targets_file: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 device=None):
        self.features = pd.read_hdf(features_file, start=start_pos, stop=load_pos)
        if targets_file:
            self.targets = pd.read_hdf(targets_file, start=start_pos, stop=load_pos)
            assert self.features.shape[0] == self.targets.shape[0]
        else:
            self.targets = targets_file

        self.cell_ids = list(self.features.index)
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, item: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        cell_id = self.cell_ids[item]
        if self.targets is not None:
            x = torch.from_numpy(self.features.loc[cell_id].values)
            y = torch.from_numpy(self.targets.loc[cell_id].values)

            if self.transform:
                x = self.transform(x)
            if self.target_transform:
                y = self.target_transform(y)
            if self.device:
                x, y = x.to(self.device), y.to(self.device)

            return x, y
        else:
            x = torch.from_numpy(self.features.loc[cell_id].values)

            if self.transform:
                x = self.transform(x)
            if self.device:
                x = x.to(self.device)

            return x
