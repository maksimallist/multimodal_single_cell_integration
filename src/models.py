from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from src.nn_blocks import CiteClsHead, Encoder, Decoder, SimpleDecoder


# -------------------------------------------- Multiome models --------------------------------------------------------

@dataclass
class MultiModelConf:
    # encoder atts
    filters: tuple = (1, 8, 8, 32, 128, 512, 128, 32, 8, 1)
    kernels: tuple = (15, 5, 5, 5, 5, 5, 5, 5, 3)
    enc_out: int = 448
    dec_in: int = 484
    add_bottleneck: bool = False
    # decoder atts
    conv1_in_filters: int = 1
    conv1_out_filters: int = 16
    conv1_kernel_size: int = 14
    conv1_stride: int = 3
    conv1_padding: int = 0
    conv1_output_padding: int = 0

    conv2_in_filters: int = 16
    conv2_out_filters: int = 32
    conv2_kernel_size: int = 4
    conv2_stride: int = 2
    conv2_padding: int = 1
    conv2_output_padding: int = 1

    conv3_in_filters: int = 32
    conv3_out_filters: int = 16
    conv3_kernel_size: int = 4
    conv3_stride: int = 4
    conv3_padding: int = 1
    conv3_output_padding: int = 1

    conv4_in_filters: int = 16
    conv4_out_filters: int = 1
    conv4_kernel_size: int = 5
    conv4_stride: int = 2
    conv4_padding: int = 0
    conv4_output_padding: int = 1


class MultiModel(nn.Module):
    def __init__(self, conf: MultiModelConf):
        super().__init__()
        self.encoder = Encoder(conf)
        self.decoder = Decoder(conf)

        # bottleneck
        self.bottleneck = nn.Linear(in_features=conf.enc_out, out_features=conf.dec_in)
        self.add_layer = conf.add_bottleneck
        if conf.add_bottleneck:
            self.bottleneck2 = nn.Linear(in_features=conf.dec_in, out_features=conf.dec_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(x)
        h = functional.relu(self.bottleneck(enc_out))
        if self.add_layer:
            h = functional.relu(self.bottleneck2(h))

        logits = self.decoder(h)

        return functional.silu(logits)


class MultiModelSD(nn.Module):
    def __init__(self, conf: MultiModelConf):
        super().__init__()
        self.encoder = Encoder(conf)
        self.decoder = SimpleDecoder(conf)

        # bottleneck
        self.bottleneck = nn.Linear(in_features=conf.enc_out, out_features=conf.dec_in)
        self.add_layer = conf.add_bottleneck
        if conf.add_bottleneck:
            self.bottleneck2 = nn.Linear(in_features=conf.dec_in, out_features=conf.dec_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(x)
        h = functional.relu(self.bottleneck(enc_out))
        if self.add_layer:
            h = functional.relu(self.bottleneck2(h))

        logits = self.decoder(h)

        return functional.silu(logits)


# ------------------------------------------------- Cite models -------------------------------------------------------
@dataclass
class CiteModelConf:
    filters: tuple = (1, 32, 64, 128, 64, 32, 1)
    kernels: tuple = (15, 5, 5, 5, 5, 3)
    enc_out: int = 345


class CiteModel(nn.Module):
    def __init__(self, conf: CiteModelConf):
        super().__init__()
        self.encoder = Encoder(conf)
        self.decoder = CiteClsHead(input_dim=conf.enc_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(x)
        enc_out = torch.squeeze(enc_out)
        logits = self.decoder(enc_out)

        return logits
