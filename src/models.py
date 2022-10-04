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
    conv1 = {"in_filters": 1, "out_filters": 16, "kernel_size": 14, "stride": 3, "padding": 0, "output_padding": 0}
    conv2 = {"in_filters": 16, "out_filters": 32, "kernel_size": 4, "stride": 2, "padding": 1, "output_padding": 1}
    conv3 = {"in_filters": 32, "out_filters": 16, "kernel_size": 4, "stride": 4, "padding": 1, "output_padding": 1}
    conv4 = {"in_filters": 16, "out_filters": 1, "kernel_size": 5, "stride": 2, "padding": 0, "output_padding": 1}


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
