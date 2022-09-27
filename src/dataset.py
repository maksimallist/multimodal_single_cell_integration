from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
# не удалять! import hdf5plugin !
import hdf5plugin
import pandas as pd
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


class SCCDataset(Dataset):
    reserved_names: List[str] = ['train_multi_inputs', 'train_multi_targets', 'train_cite_inputs', 'train_cite_targets',
                                 'test_multi_inputs', 'test_cite_inputs']

    meta_names: List[str] = ['day', 'donor', 'cell_type', 'technology']
    meta_keys: List[str] = ['cell_id', 'day', 'donor', 'cell_type', 'technology']
    meta_unique_vals: Dict = {}

    pos_name: str = 'position'
    index_name: str = 'cell_id'
    target_name: str = 'gene_id'

    features_name: str = "block0_values"
    cell_id_name: str = "axis1"
    col_name: str = 'axis0'

    def __init__(self,
                 meta_file: str,
                 features_file: str,
                 targets_file: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.transform = transform
        self.target_transform = target_transform

        self.metadata = self.read_metadata(meta_file)
        self.features, self.features_shape = self.get_hdf5_flow(features_file)

        if targets_file:
            self.targets, self.targets_shape = self.get_hdf5_flow(targets_file)
            assert self.targets_shape[0] == self.features_shape[0], \
                AssertionError(f"Длины файлов фичей и таргетов не совпадают; "
                               f"features_file: {self.features_shape[0]}, targets_file: {self.targets_shape[0]};")
        else:
            self.targets = None

    def read_metadata(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=self.index_name)
        for key in self.meta_names:
            self.meta_unique_vals[key] = list(df[key].unique())

        return df

    def get_hdf5_flow(self, file_path: str):
        file_flow = h5py.File(file_path, 'r')

        file_keys = list(file_flow.keys())
        assert len(file_keys) == 1, AssertionError(f"Incorrect file format, '{file_path}' file have more than one "
                                                   f"group: {file_keys}.")

        file_name = file_keys[0]
        assert file_name in self.reserved_names, AssertionError(f"Incorrect file format, group name must be in "
                                                                f"{self.reserved_names}, but {file_name} was found.")

        datasets_names = list(file_flow[file_name])
        assert self.features_name in datasets_names, AssertionError(f"Incorrect file format, dataset name "
                                                                    f"{self.features_name} was not found in hdf5 file "
                                                                    f"datasets list.")
        assert self.cell_id_name in datasets_names, AssertionError(f"Incorrect file format, dataset name "
                                                                   f"{self.cell_id_name} was not found in hdf5 file "
                                                                   f"datasets list.")
        assert self.col_name in datasets_names, AssertionError(f"Incorrect file format, dataset name {self.col_name} "
                                                               f"was not found in hdf5 file datasets list.")

        lines, features_shape = file_flow[file_name][self.features_name].shape

        return file_flow[file_name], (lines, features_shape)

    def __len__(self):
        return len(self.features[self.cell_id_name])

    def __getitem__(self, item: int) -> Union[Tuple[torch.Tensor, Dict[str, str]],
                                              Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]]:
        cell_id = self.features[self.cell_id_name][item].decode("utf-8")
        meta_data = {key: self.metadata[key][cell_id] for key in self.meta_names}
        meta_data[self.pos_name] = self.features[self.col_name][item].decode("utf-8")
        meta_data[self.index_name] = cell_id

        x = self.features[self.features_name][item]
        if self.transform:
            x = self.transform(x)

        if self.targets is not None:
            meta_data[self.target_name] = self.targets[self.col_name][item].decode("utf-8")
            y = self.targets[self.features_name][item]
            if self.target_transform:
                y = self.target_transform(y)

            return x, y, meta_data
        else:
            return x, meta_data
