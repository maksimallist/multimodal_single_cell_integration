import gc
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FlowDataset
from src.model import EncoderConfig, MultiModel, CiteModel


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

    del tmp_data

    return target_names


if __name__ == '__main__':
    # determine dataset folder and get column names
    data_path = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/dataset')
    cite_target_names = get_targets_names(str(data_path.joinpath('train_cite_targets.h5')))
    mutiome_target_names = get_targets_names(str(data_path.joinpath('train_multi_targets.h5')))

    # determine experiments folder
    exp_root = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/experiments/')
    device = torch.device('cuda:0')

    # Make mutiome predictions
    # -----------------------------------------------------------------------------------------------------------------
    # load json file with info about experiment
    with open(str(exp_root.joinpath('check_fix_m-14.09.2022-13.01', 'exp_config.json')), 'r') as conf_file:
        exp_conf = json.load(conf_file)

    # create the object of class MultiModel
    encoder_conf = EncoderConfig(filters=exp_conf['model']['encoder_filters'],
                                 kernels=exp_conf['model']['encoder_kernels'])
    multi_model = MultiModel(encoder_conf,
                             enc_out=exp_conf['model']['encoder_out'],
                             dec_in=exp_conf['model']['decoder_in'])

    # load weights of the model
    weights_path = exp_root.joinpath('check_fix_m-14.09.2022-13.01', 'checkpoints', 'best_model', 'best_model.pt')
    multi_model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    multi_model.to(device)
    multi_model.eval()

    # load test data
    test_features = FlowDataset(str(data_path.joinpath('test_multi_inputs.h5')))
    loader = DataLoader(test_features, batch_size=10, shuffle=False)

    # start predictions on test set
    test_pred, ids = [], []
    with tqdm(loader, desc='Batch', disable=False) as progress:
        for i, (cell_ids, features) in enumerate(progress):
            ids.extend(cell_ids)
            features = features[:, None, :].to(device)
            p = multi_model(features).detach()
            p = torch.flatten(p, start_dim=1)
            p = p.cpu().numpy()
            test_pred.append(p)

    test_pred = np.vstack(test_pred)
    np.save("/home/mks/PycharmProjects/multimodal_single_cell_integration/submissions/mutiome_eval.npy", test_pred)
    # mutiome_eval = transform_cite_predictions(ids, mutiome_target_names, test_pred)
    # mutiome_eval.to_csv("/home/mks/PycharmProjects/multimodal_single_cell_integration/submissions/mutiome_eval.csv")

    del multi_model, test_pred, ids, loader, test_features  # mutiome_eval
    gc.collect()
    # -----------------------------------------------------------------------------------------------------------------

    # Make mutiome predictions
    # -----------------------------------------------------------------------------------------------------------------
    # load json file with info about experiment
    with open(str(exp_root.joinpath('check_fix_c-14.09.2022-12.23', 'exp_config.json')), 'r') as conf_file:
        exp_conf = json.load(conf_file)

    # create the object of class MultiModel
    encoder_conf = EncoderConfig(filters=exp_conf['model']['encoder_filters'],
                                 kernels=exp_conf['model']['encoder_kernels'])
    cite_model = CiteModel(encoder_conf, enc_out=exp_conf['model']['encoder_out'])

    # load weights of the model
    weights_path = exp_root.joinpath('check_fix_c-14.09.2022-12.23', 'checkpoints', 'best_model', 'best_model.pt')
    cite_model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    cite_model.to(device)
    cite_model.eval()

    # load test data
    test_features = FlowDataset(str(data_path.joinpath('test_cite_inputs.h5')))
    loader = DataLoader(test_features, batch_size=8, shuffle=False)

    # start predictions on test set
    test_pred, ids = [], []
    with tqdm(loader, desc='Batch', disable=False) as progress:
        for i, (cell_ids, features) in enumerate(progress):
            ids.extend(cell_ids)
            features = features[:, None, :].to(device)
            p = cite_model(features).detach()
            p = torch.flatten(p, start_dim=1)
            p = p.cpu().numpy()
            test_pred.append(p)

    test_pred = np.vstack(test_pred)
    np.save("/home/mks/PycharmProjects/multimodal_single_cell_integration/submissions/cite_eval.npy", test_pred)
    # cite_eval = transform_cite_predictions(ids, cite_target_names, test_pred)
    # cite_eval.to_csv("/home/mks/PycharmProjects/multimodal_single_cell_integration/submissions/cite_eval.csv")

    del cite_model, test_pred, ids, loader, test_features  # cite_eval
    gc.collect()
    # -----------------------------------------------------------------------------------------------------------------
    # concat_pred = pd.concat([cite_eval, mutiome_eval], axis=1)
    # del cite_eval, mutiome_eval
    # gc.collect()
    #
    # # Read and write submission file
    # -----------------------------------------------------------------------------------------------------------------
    # evaluation_ids = pd.read_csv(str(data_path.joinpath('evaluation_ids.csv')), index_col=['cell_id', 'gene_id'])
    # submission = pd.concat([evaluation_ids, concat_pred], axis=1)
    # assert not submission.isna().any()
    # del evaluation_ids, concat_pred
    # gc.collect()
    #
    # # delete ['cell_id', 'gene_id'] columns
    # submission = submission.reset_index()
    # submission = submission.set_index('row_id')
    # del submission['cell_id'], submission['gene_id']
    # gc.collect()
    #
    # # write submission file
    # submission.to_csv("/home/mks/PycharmProjects/multimodal_single_cell_integration/submissions/submission.csv")
