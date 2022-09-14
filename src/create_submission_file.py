from pathlib import Path
import json

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


from src.model import EncoderConfig, MultiModel
from src.dataset import FlowDataset


# def add_dimension(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
#     return torch.unsqueeze(tensor, dim=dim)


if __name__ == '__main__':
    # determine the folders
    exp_path = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/experiments/test')
    data_path = Path('/home/mks/PycharmProjects/multimodal_single_cell_integration/dataset')

    # load json file with info about experiment
    with open(str(exp_path.joinpath('fold_1-13.09.2022-17.36', 'exp_config.json')), 'r') as conf_file:
        exp_conf = json.load(conf_file)

    # create the object of class MultiModel
    encoder_conf = EncoderConfig(filters=exp_conf['model']['encoder_filters'],
                                 kernels=exp_conf['model']['encoder_kernels'])
    multi_model = MultiModel(encoder_conf,
                             enc_out=exp_conf['model']['encoder_out'],
                             dec_in=exp_conf['model']['decoder_in'])

    # load weights of the model
    weights_path = exp_path.joinpath('fold_1-13.09.2022-17.36', 'checkpoints', 'best_model', 'best_model.pt')
    multi_model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    multi_model.eval()

    # determine the device on which the calculations will be carried out
    device = torch.device('cuda:0')
    multi_model.to(device)

    # load test data
    test_features = FlowDataset(str(data_path.joinpath('test_multi_inputs.h5')))
    loader = DataLoader(test_features, batch_size=2, shuffle=False)

    # start predictions on test set
    test_pred = []
    with tqdm(loader, miniters=100, desc='Batch', disable=False) as progress:
        for i, features in enumerate(progress):
            if i <= 1000:
                features = features[:, None, :].to(device)
                test_pred.append(multi_model(features).detach().cpu().numpy())
            else:
                break

    test_pred = np.stack(test_pred, axis=0)
    # -----------------------------------------------------------------------------------------------------------------
    # test_pred = np.zeros((len(Xt), 140), dtype=np.float32)
    # for fold in range(N_SPLITS):
    #     print(f"Predicting with fold {fold}")
    #     model = load_model(f"/kaggle/temp/model_{fold}",
    #                        custom_objects={'negative_correlation_loss': negative_correlation_loss})
    #     test_pred += model.predict(Xt)
    # -----------------------------------------------------------------------------------------------------------------

    submission = pd.read_csv(str(data_path.joinpath('sample_submission.csv')), index_col='row_id', squeeze=True)
    submission.iloc[:len(test_pred.ravel())] = test_pred.ravel()
    # assert not submission.isna().any()
    submission.to_csv(str(exp_path.joinpath('submission.csv')))
