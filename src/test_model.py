from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import EncoderConfig, Model

input_dim = 228942
target_dim = 23418
enc_out = 448
dec_in = 484
batch_size = 2
channels = 1

if __name__ == '__main__':
    # data paths
    file_name = 'train_multi_inputs.h5'
    data_path = Path(__file__).parent.joinpath('dataset', file_name).absolute()

    # read part of data
    start, stop = 0, batch_size
    x_data = pd.read_hdf(str(data_path), start=start, stop=stop).values
    print(x_data.shape, type(x_data))
    x_data = x_data[:, None, :]

    # model
    device = 'cuda'
    encoder_conf = EncoderConfig(filters=(1, 8, 8, 32, 128, 512, 128, 32, 8, 1),
                                 kernels=(15, 5, 5, 5, 5, 5, 5, 5, 3))

    model = Model(encoder_conf, enc_out=enc_out, dec_in=dec_in).to(device)
    p_num = 0
    for p in model.parameters():
        p_num += np.prod(np.array(p.shape))

    print(f"Number of trainable parameters in model: {p_num};")

    # try forward model
    tensor = torch.from_numpy(x_data).to(device)
    out = model(tensor).detach().cpu().numpy()
    print(f"Model output shape: {out.shape};")
