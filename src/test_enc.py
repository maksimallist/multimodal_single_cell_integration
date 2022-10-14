import torch

from cell_encoder import MostSimpleCellEncoder

if __name__ == '__main__':
    encoder = MostSimpleCellEncoder(feature_len=1000,
                                    emb_dim=16,
                                    bin_len=9,
                                    emb_normalization=True,
                                    val_emb_mode='sum')

    x = torch.rand(2, 2000)
    y = encoder(x)
