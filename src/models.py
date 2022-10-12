from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
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


# ------------------------------------------------- Burtsev model -----------------------------------------------------


@dataclass
class BurtsevEncoderConf:
    input_shape: tuple
    activation: Callable
    input_channels = [(1, 32), (64, 128), (128, 128), (128, 256), (256, 128)]
    out_channels = [(32, 64), (128, 128), (128, 128), (256, 256), (128, 128)]
    kernels = [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3)]
    pooling = [2, 2, 2, 2, 2]


class BEncBlock(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int, int],
                 activation: Callable,
                 pool: int,
                 in_c: Tuple[int, int],
                 out_c: Tuple[int, int],
                 kernels: Tuple[int, int]):
        super().__init__()
        self.activation = activation

        self.conv1 = nn.Conv2d(in_c[0], out_c[0], kernels[0])
        out_h, out_w = self.calc_out_shape(input_shape, kernels[0])
        self.norm1 = nn.LayerNorm([out_c[0], out_h, out_w])

        self.conv2 = nn.Conv2d(in_c[1], out_c[1], kernels[1])
        out_h, out_w = self.calc_out_shape((out_h, out_w), kernels[1])
        self.norm2 = nn.LayerNorm([out_c[1], out_h, out_w])

        self.pool = nn.AvgPool2d(pool)
        self.block_out_shape = (out_h // pool, out_w // pool)

    @staticmethod
    def calc_out_shape(in_shape: Tuple[int, int], kernel_size: int) -> Tuple[int, int]:
        return in_shape[0] - (kernel_size // 2) * 2, in_shape[1] - (kernel_size // 2) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv1(x)
        o = self.norm1(o)
        o = self.activation(o)

        o = self.conv2(o)
        o = self.norm2(o)
        o = self.activation(o)

        out = self.pool(o)

        return out


class BurtsevModelEncoder(nn.Module):
    def __init__(self, conf: BurtsevEncoderConf):
        super().__init__()
        self.activation = conf.activation
        self.conv_block1 = BEncBlock(input_shape=conf.input_shape,
                                     activation=conf.activation,
                                     pool=conf.pooling[0],
                                     in_c=conf.input_channels[0],
                                     out_c=conf.out_channels[0],
                                     kernels=conf.kernels[0])
        self.conv_block2 = BEncBlock(input_shape=self.conv_block1.block_out_shape,
                                     activation=conf.activation,
                                     pool=conf.pooling[1],
                                     in_c=conf.input_channels[1],
                                     out_c=conf.out_channels[1],
                                     kernels=conf.kernels[1])
        self.conv_block3 = BEncBlock(input_shape=self.conv_block2.block_out_shape,
                                     activation=conf.activation,
                                     pool=conf.pooling[2],
                                     in_c=conf.input_channels[2],
                                     out_c=conf.out_channels[2],
                                     kernels=conf.kernels[2])
        self.conv_block4 = BEncBlock(input_shape=self.conv_block3.block_out_shape,
                                     activation=conf.activation,
                                     pool=conf.pooling[3],
                                     in_c=conf.input_channels[3],
                                     out_c=conf.out_channels[3],
                                     kernels=conf.kernels[3])
        self.conv_block5 = BEncBlock(input_shape=self.conv_block4.block_out_shape,
                                     activation=conf.activation,
                                     pool=conf.pooling[4],
                                     in_c=conf.input_channels[4],
                                     out_c=conf.out_channels[4],
                                     kernels=conf.kernels[4])

        self.conv_last1 = nn.Conv2d(conf.input_channels[4][-1], conf.out_channels[4][-1], conf.kernels[4][-1])
        last_h, last_w = self.conv_block5.calc_out_shape(self.conv_block5.block_out_shape, conf.kernels[4][-1])
        self.norm_last1 = nn.LayerNorm([conf.out_channels[4][-1], last_h, last_w])

        self.conv_last2 = nn.Conv2d(conf.input_channels[4][-1], conf.out_channels[4][-1], conf.kernels[4][-1])
        last_h, last_w = self.conv_block5.calc_out_shape((last_h, last_w), conf.kernels[4][-1])
        self.norm_last2 = nn.LayerNorm([conf.out_channels[4][-1], last_h, last_w])

    def forward(self, x: torch.Tensor):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        x = self.conv_last1(x)
        x = self.norm_last1(x)
        x = self.activation(x)

        x = self.conv_last2(x)
        x = self.norm_last2(x)
        x = self.activation(x)

        return x


@dataclass
class BurtsevDecoderConf:
    input_shape: tuple
    activation: Callable
    input_channels = [(128, 128), (128, 128), (128, 64), (64, 32), (32, 16)]
    out_channels = [(128, 128), (128, 128), (64, 64), (32, 32), (16, 1)]
    kernels = [(3, 3), (3, 3), (5, 5), (5, 5), (7, 9)]
    pooling = [2, 2, 2, 2, 2]


class BDecBlock(nn.Module):
    def __init__(self,
                 scale: int,
                 activation: Callable,
                 input_shape: Tuple[int, int],
                 in_c: Tuple[int, int],
                 out_c: Tuple[int, int],
                 kernels: Tuple[int, int]):
        super().__init__()
        self.activation = activation
        self.dec_upsample1 = nn.Upsample(scale_factor=scale, mode='nearest')

        self.dec_conv1 = nn.ConvTranspose2d(in_c[0], out_c[0], kernels[0])
        out_h, out_w = self.calc_out_shape((input_shape[0] * scale, input_shape[1] * scale), kernels[0])
        self.norm1 = nn.LayerNorm([out_c[0], out_h, out_w])

        self.dec_conv2 = nn.ConvTranspose2d(in_c[1], out_c[1], kernels[1])
        out_h, out_w = self.calc_out_shape((out_h, out_w), kernels[1])
        self.norm2 = nn.LayerNorm([out_c[1], out_h, out_w])

        self.block_out_shape = (out_h, out_w)

    @staticmethod
    def calc_out_shape(in_shape: Tuple[int, int], kernel_size: int) -> Tuple[int, int]:
        return in_shape[0] + (kernel_size // 2) * 2, in_shape[1] + (kernel_size // 2) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.dec_upsample1(x)

        o = self.dec_conv1(o)
        o = self.norm1(o)
        o = self.activation(o)

        o = self.dec_conv2(o)
        o = self.norm2(o)
        out = self.activation(o)

        return out


class BurtsevModelDecoder(nn.Module):
    def __init__(self, conf: BurtsevDecoderConf):
        super().__init__()
        self.activation = conf.activation
        self.deconv1 = nn.ConvTranspose2d(conf.input_channels[0][0], conf.out_channels[0][0], 5)
        input_shape = (conf.input_shape[0] + 4, conf.input_shape[1] + 4)
        self.conv_block1 = BDecBlock(scale=conf.pooling[0],
                                     input_shape=input_shape,
                                     activation=conf.activation,
                                     in_c=conf.input_channels[0],
                                     out_c=conf.out_channels[0],
                                     kernels=conf.kernels[0])
        self.conv_block2 = BDecBlock(scale=conf.pooling[1],
                                     input_shape=self.conv_block1.block_out_shape,
                                     activation=conf.activation,
                                     in_c=conf.input_channels[1],
                                     out_c=conf.out_channels[1],
                                     kernels=conf.kernels[1])
        self.conv_block3 = BDecBlock(scale=conf.pooling[2],
                                     input_shape=self.conv_block2.block_out_shape,
                                     activation=conf.activation,
                                     in_c=conf.input_channels[2],
                                     out_c=conf.out_channels[2],
                                     kernels=conf.kernels[2])
        self.conv_block4 = BDecBlock(scale=conf.pooling[3],
                                     input_shape=self.conv_block3.block_out_shape,
                                     activation=conf.activation,
                                     in_c=conf.input_channels[3],
                                     out_c=conf.out_channels[3],
                                     kernels=conf.kernels[3])
        self.conv_block5 = BDecBlock(scale=conf.pooling[4],
                                     input_shape=self.conv_block4.block_out_shape,
                                     activation=conf.activation,
                                     in_c=conf.input_channels[4],
                                     out_c=conf.out_channels[4],
                                     kernels=conf.kernels[4])

    def forward(self, x: torch.Tensor):
        x = self.activation(self.deconv1(x))
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x


class BurtsevAutoEncoder(nn.Module):
    def __init__(self,
                 encoder_config: BurtsevEncoderConf,
                 decoder_config: BurtsevDecoderConf,
                 bottle_neck: int,
                 activation: Optional[Callable] = functional.leaky_relu):
        super(BurtsevAutoEncoder, self).__init__()
        self.activation = activation
        self.encoder = BurtsevModelEncoder(encoder_config)  # input_shape=[bs, 1, 478, 478] | out_shape=[bs, 128, 6, 6]
        self.decoder = BurtsevModelDecoder(decoder_config)  # input_shape=[bs, 128, 6, 6] | out_shape=[bs, 1, 478, 478]

        self.encoder_neck_shape = (128, 6, 6)  # [C, H, W]
        self.enc_flat_shape = int(np.prod(self.encoder_neck_shape))

        self.enc_neck = nn.Linear(self.enc_flat_shape, bottle_neck)
        self.dec_neck = nn.Linear(bottle_neck, self.enc_flat_shape)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x = self.encoder(tensor)
        x = x.flatten(start_dim=1)
        bottle_neck = self.activation(self.enc_neck(x))
        dec_in = self.activation(self.dec_neck(bottle_neck))
        dec_in = torch.reshape(dec_in, (dec_in.shape[0], *self.encoder_neck_shape))
        out = self.decoder(dec_in)

        return out

    def get_bottleneck(self, tensor: torch.Tensor) -> torch.Tensor:
        x = self.encoder(tensor)
        x = x.flatten(start_dim=1)

        return self.activation(self.enc_neck(x))


@dataclass
class BurtsevForwardModelConf:
    input_shape: Tuple[int, int, int]
    activation: Callable
    scale = 2
    in_channels = [1, 4, 8, 16, 32, 64, 64, 128, 256, 256, 256, 256, 128, 128, 64, 64, 64, 32, 16, 8, 4]
    out_channels = [4, 8, 16, 32, 64, 64, 128, 256, 256, 256, 256, 128, 128, 64, 64, 64, 32, 16, 8, 4, 1]
    kernels = [7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3]
    before_pool = {7: 1}
    after_pool = {7: 6, 5: 9, 3: 5}


class BurtsevForwardModel(nn.Module):
    def __init__(self, conf: BurtsevForwardModelConf):
        super(BurtsevForwardModel, self).__init__()
        self.activation = conf.activation
        self.conv0 = nn.Conv2d(conf.in_channels[0], conf.out_channels[0], conf.kernels[0])
        self.pool = nn.AvgPool2d(conf.scale)
        # tower
        self.conv1 = nn.Conv2d(conf.in_channels[1], conf.out_channels[1], conf.kernels[1])
        self.conv2 = nn.Conv2d(conf.in_channels[2], conf.out_channels[2], conf.kernels[2])
        self.norm1 = nn.BatchNorm2d(conf.out_channels[2])
        self.conv3 = nn.Conv2d(conf.in_channels[3], conf.out_channels[3], conf.kernels[3])
        self.conv4 = nn.Conv2d(conf.in_channels[4], conf.out_channels[4], conf.kernels[4])
        self.norm2 = nn.BatchNorm2d(conf.out_channels[4])
        self.conv5 = nn.Conv2d(conf.in_channels[5], conf.out_channels[5], conf.kernels[5])
        self.conv6 = nn.Conv2d(conf.in_channels[6], conf.out_channels[6], conf.kernels[6])
        self.norm3 = nn.BatchNorm2d(conf.out_channels[6])
        self.conv7 = nn.Conv2d(conf.in_channels[7], conf.out_channels[7], conf.kernels[7])
        self.conv8 = nn.Conv2d(conf.in_channels[8], conf.out_channels[8], conf.kernels[8])
        self.norm4 = nn.BatchNorm2d(conf.out_channels[8])
        self.conv9 = nn.Conv2d(conf.in_channels[9], conf.out_channels[9], conf.kernels[9])
        self.conv10 = nn.Conv2d(conf.in_channels[10], conf.out_channels[10], conf.kernels[10])
        self.norm5 = nn.BatchNorm2d(conf.out_channels[10])
        self.conv11 = nn.Conv2d(conf.in_channels[11], conf.out_channels[11], conf.kernels[11])
        self.conv12 = nn.Conv2d(conf.in_channels[12], conf.out_channels[12], conf.kernels[12])
        self.norm6 = nn.BatchNorm2d(conf.out_channels[12])
        self.conv13 = nn.Conv2d(conf.in_channels[13], conf.out_channels[13], conf.kernels[13])
        self.conv14 = nn.Conv2d(conf.in_channels[14], conf.out_channels[14], conf.kernels[14])
        self.norm7 = nn.BatchNorm2d(conf.out_channels[14])
        self.conv15 = nn.Conv2d(conf.in_channels[15], conf.out_channels[15], conf.kernels[15])
        self.conv16 = nn.Conv2d(conf.in_channels[16], conf.out_channels[16], conf.kernels[16])
        self.norm8 = nn.BatchNorm2d(conf.out_channels[16])
        self.conv17 = nn.Conv2d(conf.in_channels[17], conf.out_channels[17], conf.kernels[17])
        self.conv18 = nn.Conv2d(conf.in_channels[18], conf.out_channels[18], conf.kernels[18])
        self.norm9 = nn.BatchNorm2d(conf.out_channels[18])
        self.conv19 = nn.Conv2d(conf.in_channels[19], conf.out_channels[19], conf.kernels[19])
        self.conv20 = nn.Conv2d(conf.in_channels[20], conf.out_channels[20], conf.kernels[20])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv0(tensor))
        x = self.pool(x)
        # tower
        x = self.activation(self.conv1(x))
        x = self.activation(self.norm1(self.conv2(x)))
        x = self.activation(self.conv3(x))
        x = self.activation(self.norm2(self.conv4(x)))
        x = self.activation(self.conv5(x))
        x = self.activation(self.norm3(self.conv6(x)))
        x = self.activation(self.conv7(x))
        x = self.activation(self.norm4(self.conv8(x)))
        x = self.activation(self.conv9(x))
        x = self.activation(self.norm5(self.conv10(x)))
        x = self.activation(self.conv11(x))
        x = self.activation(self.norm6(self.conv12(x)))
        x = self.activation(self.conv13(x))
        x = self.activation(self.norm7(self.conv14(x)))
        x = self.activation(self.conv15(x))
        x = self.activation(self.norm8(self.conv16(x)))
        x = self.activation(self.conv17(x))
        x = self.activation(self.norm9(self.conv18(x)))
        x = self.activation(self.conv19(x))
        out = self.activation(self.conv20(x))

        return out
