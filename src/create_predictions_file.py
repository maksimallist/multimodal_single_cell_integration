import gc
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.dataset import FlowDataset


def transform_cite_predictions(cell_id_list: List[str], target_names: List[str], pred_tensor: np.array) -> pd.DataFrame:
    global_id, global_target_names, global_pred = [], [], []
    for id_name, vector in zip(cell_id_list, pred_tensor):
        vector = vector.flatten()
        global_id.extend([id_name] * vector.shape[0])
        global_target_names.extend(target_names)
        global_pred.extend(vector.tolist())

    data = {"cell_id": global_id, "gene_id": global_target_names, "target": global_pred}
    df = pd.DataFrame.from_dict(data)

    return df.set_index(['cell_id', 'gene_id'])


def get_targets_names(file_path: str):
    tmp_data = FlowDataset(file_path)
    target_names = []
    for x in tmp_data.features[tmp_data.col_name]:
        target_names.append(x.decode("utf-8"))

    del [tmp_data]
    gc.collect()

    return target_names


if __name__ == '__main__':
    # determine dataset folder and get column names
    data_path = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/dataset')
    submissions_path = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/submissions/')

    cite_target_names = get_targets_names(str(data_path.joinpath('train_cite_targets.h5')))
    mutiome_target_names = get_targets_names(str(data_path.joinpath('train_multi_targets.h5')))

    # determine experiments folder
    exp_root = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/experiments/')
    # Load mutiome predictions
    # Make cite predictions
    print(f"[ Read cite predictions ... ]")
    # -----------------------------------------------------------------------------------------------------------------
    test_pred = np.load(str(exp_root.joinpath('check_fix_c-15.09.2022-14.12', 'cite_eval.npy')))
    ids = np.load(str(exp_root.joinpath('check_fix_c-15.09.2022-14.12', 'cite_ids.npy'))).tolist()
    cite_eval = transform_cite_predictions(ids, cite_target_names, test_pred)
    # cite_eval.to_csv("/home/mks/PycharmProjects/multimodal_single_cell_integration/submissions/cite_eval.csv")

    del [test_pred, ids]  # , cite_eval
    gc.collect()
    print(f"[ Reading cite predictions complete. ]")

    # -----------------------------------------------------------------------------------------------------------------
    print(f"[ Read mutiome predictions ... ]")

    test_pred = np.load(str(exp_root.joinpath('check_fix_m-14.09.2022-20.16', 'multi_eval.npy')))
    ids = np.load(str(exp_root.joinpath('check_fix_m-14.09.2022-20.16', 'multi_ids.npy'))).tolist()
    mutiome_eval = transform_cite_predictions(ids, mutiome_target_names, test_pred)
    # mutiome_eval.to_csv(str(submissions_path.joinpath('mutiome_eval.csv')))

    del [test_pred, ids]  #, mutiome_eval
    gc.collect()
    print(f"[ Reading mutiome predictions complete. ]")
    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    print(f"[ Concat tasks predictions ... ]")
    concat_pred = pd.concat([cite_eval, mutiome_eval], axis=1)
    del [cite_eval, mutiome_eval]
    gc.collect()
    print(f"[ Concatenation of tasks predictions is complete. ]")
    # concat_pred.to_csv(str(submissions_path.joinpath("pred_sub.csv")))

    # # Read and write submission file
    # -----------------------------------------------------------------------------------------------------------------
    print(f"[ Read evaluation metafile ... ]")
    evaluation_ids = pd.read_csv(str(data_path.joinpath('evaluation_ids.csv')), index_col=['cell_id', 'gene_id'])
    print(f"[ Reading evaluation metafile complete. ]")
    print(f"[ Run creation of submission file ... ]")
    submission = pd.concat([evaluation_ids, concat_pred], axis=1)
    assert not submission.isna().any()
    del [evaluation_ids, concat_pred]
    gc.collect()

    # delete ['cell_id', 'gene_id'] columns
    submission = submission.reset_index()
    submission = submission.set_index('row_id')
    del [submission['cell_id'], submission['gene_id']]
    gc.collect()
    # write submission file
    submission.to_csv(str(submissions_path.joinpath("test_submission.csv")))
    print(f"[ Submission file is done and saved. ]")
