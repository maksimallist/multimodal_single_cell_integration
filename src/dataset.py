from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
# не удалять! import hdf5plugin !
import hdf5plugin
import numpy as np
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


class FSCCDataset(Dataset):
    file_types = ['inputs', 'targets']
    h5_reserved_names: List[str] = ['train_multi_inputs', 'train_multi_targets', 'train_cite_inputs',
                                    'train_cite_targets', 'test_multi_inputs', 'test_cite_inputs']

    dataflows = {'cite': {'train': {'inputs': None, 'targets': None},
                          'test': {'inputs': None}},
                 'multi': {'train': {'inputs': None, 'targets': None},
                           'test': {'inputs': None}}}

    metadata = None
    meta_unique_vals: Dict = {}
    metadata_file: str = 'metadata.csv'
    meta_transform_names: List[str] = ['day', 'donor', 'cell_type']
    meta_names: List[str] = ['day', 'donor', 'cell_type', 'technology']
    meta_keys: List[str] = ['cell_id', 'day', 'donor', 'cell_type', 'technology']

    col_name: str = 'axis0'
    pos_name: str = 'position'
    index_name: str = 'cell_id'
    cell_id_name: str = "axis1"
    target_name: str = 'gene_id'
    features_name: str = "block0_values"

    def __init__(self,
                 dataset_path: Union[str, Path],
                 task: str, mode: str,
                 meta_transform: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.task = task
        self.mode = mode
        self.data_ids = None
        self.data_shapes = None
        self.dataset_path = dataset_path

        self.transform = transform
        self.target_transform = target_transform
        self.meta_transform = meta_transform
        # init dataset
        self._read_task_dataset(dataset_path)

    def _read_metadata(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=self.index_name)
        for key in self.meta_names:
            self.meta_unique_vals[key] = list(df[key].unique())

        return df

    def _transform_metalabels(self, meta_dict: Dict, cell_id: str) -> Dict:
        if self.meta_transform:
            if self.meta_transform == 'index':
                for key in self.meta_transform_names:
                    meta_dict[key] = self.meta_unique_vals[key].index(self.metadata[key][cell_id])
            elif self.meta_transform == 'one_hot':
                for key in self.meta_transform_names:
                    one_hot_vector = np.zeros((len(self.meta_unique_vals[key]),))
                    one_hot_vector[self.meta_unique_vals[key].index(self.metadata[key][cell_id])] = 1
                    meta_dict[key] = one_hot_vector
            else:
                raise ValueError(f"The argument 'meta_transform' can only take values from a list "
                                 f"['index', 'one_hot', None], but '{self.meta_transform}' was found.")
        else:
            meta_dict = {key: self.metadata[key][cell_id] for key in self.meta_names}

        return meta_dict

    def _get_task_flow(self, folder_path: Path, mode: str, task: str, file_type: str) -> None:
        file_name = '_'.join([mode, task, file_type])
        print(f"[ Reading {file_name}.h5 file ... ]")
        f_path = str(folder_path.joinpath(f"{file_name}.h5").absolute())
        flow, feature_shape = self.get_hdf5_flow(f_path)
        # write data in structure
        self.dataflows[task][mode][file_type] = flow
        self.data_shapes[task][mode][file_type] = feature_shape
        print(f"[ Reading {file_name}.h5 file is complete. ]")

    def _read_task_dataset(self, folder_path: Union[str, Path]) -> None:
        self.data_shapes = {self.task: {self.mode: {s: None for s in self.file_types}}}

        if isinstance(folder_path, str):
            folder_path = Path(folder_path)
        # read metadata file
        self.metadata = self._read_metadata(str(folder_path.joinpath(self.metadata_file)))
        # read all h5 files
        if self.mode == 'train':
            for file_type in self.file_types:
                self._get_task_flow(folder_path, self.mode, self.task, file_type)
        elif self.mode == 'test':
            self._get_task_flow(folder_path, self.mode, self.task, self.file_types[0])
        else:
            raise ValueError(f"Argument 'mode' can only take values from a list: ['train', 'test'], "
                             f"but {self.mode} was found.")

        self.data_ids = self._set_data_ids()

    def _set_data_ids(self):
        feature_flow = self.dataflows[self.task][self.mode]['inputs']
        return [x.decode("utf-8") for x in feature_flow[self.cell_id_name]]

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, item: int) -> Dict:
        cell_id = self.data_ids[item]
        features = self.dataflows[self.task][self.mode]['inputs']
        # , self.pos_name: features[self.col_name][item].decode("utf-8")
        meta_data = {self.index_name: cell_id}
        meta_data = self._transform_metalabels(meta_data, cell_id)

        x = features[self.features_name][item]
        if self.transform:
            x = self.transform(x)

        meta_data[self.file_types[0]] = x

        if self.dataflows[self.task][self.mode].get('targets'):
            targets = self.dataflows[self.task][self.mode]['targets']
            meta_data[self.target_name] = targets[self.cell_id_name][item].decode("utf-8")
            y = targets[self.features_name][item]

            if self.target_transform:
                y = self.target_transform(y)

            meta_data[self.file_types[1]] = y

            return meta_data
        else:
            return meta_data

    def get_hdf5_flow(self, file_path: str):
        file_flow = h5py.File(file_path, 'r')

        file_keys = list(file_flow.keys())
        assert len(file_keys) == 1, AssertionError(f"Incorrect file format, '{file_path}' file have more than one "
                                                   f"group: {file_keys}.")

        file_name = file_keys[0]
        assert file_name in self.h5_reserved_names, \
            AssertionError(f"Incorrect file format, group name must be in {self.h5_reserved_names}, "
                           f"but {file_name} was found.")

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

    def reindex_dataset(self,
                        day: Optional[Union[int, List[int]]] = None,
                        donor: Optional[Union[int, List[int]]] = None,
                        cell_type: Optional[Union[str, List[str]]] = None) -> None:
        conditions = []
        if (day is not None) and isinstance(day, int):
            conditions.append((self.metadata['day'] == day))
        elif (day is not None) and isinstance(day, list):
            conditions.append((self.metadata['day'].isin(day)))

        if (donor is not None) and isinstance(donor, int):
            conditions.append((self.metadata['donor'] == donor))
        elif (donor is not None) and isinstance(donor, list):
            conditions.append((self.metadata['donor'].isin(donor)))

        if (cell_type is not None) and isinstance(cell_type, int):
            conditions.append((self.metadata['cell_type'] == cell_type))
        elif (cell_type is not None) and isinstance(cell_type, list):
            conditions.append((self.metadata['cell_type'].isin(cell_type)))

        if len(conditions) > 0:
            feature_flow = self.dataflows[self.task][self.mode]['inputs']
            ids = {x.decode("utf-8") for x in feature_flow[self.cell_id_name]}

            final_cond = conditions[0]
            if len(conditions) > 1:
                for cond in conditions[1:]:
                    final_cond &= cond

            cond_index = set(self.metadata[final_cond].index)
            self.data_ids = list(cond_index & ids)

    def rebase(self, task: Optional[str] = None, mode: Optional[str] = None):
        if task is not None:
            self.task = task
        if mode is not None:
            self.mode = mode

        self._read_task_dataset(self.dataset_path)
        self.data_ids = self._set_data_ids()

    def set_length(self, length: int) -> None:
        self.data_ids = self.data_ids[:length]
