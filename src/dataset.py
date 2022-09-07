from typing import Optional, Callable

import pandas as pd
from torch.utils.data import Dataset


# todo: переписать, или дописать класс, который работал бы с датасетом в поточном режиме
class MyDataset(Dataset):
    def __init__(self,
                 features_file: str,
                 start_pos: int,
                 load_pos: int,
                 targets_file: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.features = pd.read_hdf(features_file, start=start_pos, stop=load_pos)
        if targets_file:
            self.targets = pd.read_hdf(targets_file, start=start_pos, stop=load_pos)
            assert self.features.shape[0] == self.targets.shape[0]
        else:
            self.targets = targets_file

        self.cell_ids = list(self.features.index)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, item):
        cell_id = self.cell_ids[item]
        if self.targets is not None:
            x = self.features.loc[cell_id].values
            y = self.targets.loc[cell_id].values

            if self.transform:
                x = self.transform(x)

            if self.target_transform:
                y = self.target_transform(y)

            return x, y
        else:
            x = self.features.loc[cell_id].values
            if self.transform:
                x = self.transform(x)
            return x
