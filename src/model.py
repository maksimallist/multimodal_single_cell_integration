from dataclasses import dataclass
from typing import Tuple, Callable

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional


class Residual(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class AttentionPool(nn.Module):
    def __init__(self, dim: int, pool_size: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p=pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = functional.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = functional.pad(mask, (0, remainder), value=True)

            x = self.pool_fn(x)
            logits = self.to_attn_logits(x)

            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)
        else:
            x = self.pool_fn(x)
            logits = self.to_attn_logits(x)

        attn = logits.softmax(dim=-1)

        return (x * attn).sum(dim=-1)


class EncoderBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 dim_out: int,
                 kernel_size: int,
                 use_batchnorm: bool = True,
                 gelu_weight: float = 1.702):
        super().__init__()
        self.gelu_weight = gelu_weight
        self.use_batchnorm = use_batchnorm

        # first block
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, padding=kernel_size // 2)
        if use_batchnorm:
            self.batch_norm = nn.BatchNorm1d(dim_out)

        # second block
        self.resid_batch_norm = nn.BatchNorm1d(dim_out)
        self.resid_conv = nn.Conv1d(dim_out, dim_out, 1)

        # third
        self.residual = Residual(self.residual_ops)
        self.att = AttentionPool(dim_out, pool_size=2)

    def residual_ops(self, tensor: torch.Tensor) -> torch.Tensor:
        x = self.resid_batch_norm(tensor)
        x = torch.sigmoid(self.gelu_weight * x) * x
        x = self.resid_conv(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.batch_norm(x)
        x = torch.sigmoid(self.gelu_weight * x) * x
        x = self.residual(x)
        out = self.att(x)

        return out


@dataclass
class EncoderConfig:
    model_type = "Encoder"

    def __init__(self, filters: Tuple, kernels: Tuple):
        super().__init__()
        self.filters = filters
        self.kernels = kernels


class Encoder(nn.Module):
    def __init__(self, config: dataclass):
        super().__init__()
        # create conv tower
        conv_layers = []
        for i, (dim_in, dim_out, ks) in enumerate(zip(config.filters[:-1], config.filters[1:], config.kernels)):
            if i == 0:
                conv_layers.append(EncoderBlock(dim_in, dim_out, kernel_size=ks, use_batchnorm=False))
            else:
                conv_layers.append(EncoderBlock(dim_in, dim_out, kernel_size=ks))

        self.conv_tower = nn.Sequential(*conv_layers)

    def forward(self, input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        return self.conv_tower(input_tensor)  # 448 size


class SimpleDecoderBlock(nn.Module):
    def __init__(self,
                 in_filters: int,
                 out_filters: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 output_padding: int,
                 use_batchnorm: bool = True,
                 gelu_weight: float = 1.702):
        super().__init__()
        self.gelu_weight = gelu_weight
        self.use_batchnorm = use_batchnorm

        # first block
        self.conv = nn.ConvTranspose1d(in_filters, out_filters, kernel_size, stride, padding, output_padding)
        if use_batchnorm:
            self.batch_norm = nn.BatchNorm1d(out_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.batch_norm(x)
        out = torch.sigmoid(self.gelu_weight * x) * x

        return out


class ResAttDecoderBlock(nn.Module):
    def __init__(self,
                 in_filters: int,
                 out_filters: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 output_padding: int,
                 use_batchnorm: bool = True,
                 gelu_weight: float = 1.702):
        super().__init__()
        self.gelu_weight = gelu_weight
        self.use_batchnorm = use_batchnorm

        # first block
        self.conv = nn.ConvTranspose1d(in_filters, out_filters, kernel_size, stride, padding, output_padding)
        if use_batchnorm:
            self.batch_norm = nn.BatchNorm1d(out_filters)

        # second block
        self.resid_conv = nn.Conv1d(out_filters, out_filters, 1)
        self.resid_batch_norm = nn.BatchNorm1d(out_filters)

        # third
        self.residual = Residual(self.residual_ops)
        self.att = AttentionPool(out_filters, pool_size=1)

    def residual_ops(self, tensor: torch.Tensor) -> torch.Tensor:
        x = self.resid_batch_norm(tensor)
        x = torch.sigmoid(self.gelu_weight * x) * x
        x = self.resid_conv(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.batch_norm(x)
        x = torch.sigmoid(self.gelu_weight * x) * x
        x = self.residual(x)
        out = self.att(x)

        return out


# todo: add dataclass with config
class SimpleDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # create conv tower | input_shape = (bs, c=1, h=484)
        self.conv1 = SimpleDecoderBlock(1, 16, kernel_size=14, stride=3, padding=0, output_padding=0)
        self.conv2 = SimpleDecoderBlock(16, 32, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.conv3 = SimpleDecoderBlock(32, 16, kernel_size=4, stride=4, padding=1, output_padding=1)
        self.conv4 = SimpleDecoderBlock(16, 1, kernel_size=5, stride=2, padding=0, output_padding=1)

    def forward(self, input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # create conv tower | input_shape = (bs, c=1, h=484)
        self.conv1 = ResAttDecoderBlock(1, 16, kernel_size=14, stride=3, padding=0, output_padding=0)
        self.conv2 = ResAttDecoderBlock(16, 32, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.conv3 = ResAttDecoderBlock(32, 16, kernel_size=4, stride=4, padding=1, output_padding=1)
        self.conv4 = ResAttDecoderBlock(16, 1, kernel_size=5, stride=2, padding=0, output_padding=1)

    def forward(self, input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


class MultiModel(nn.Module):
    def __init__(self, enc_conf, enc_out: int, dec_in: int, add_bottleneck: bool = False):
        super().__init__()
        self.encoder = Encoder(enc_conf)
        self.decoder = Decoder()

        # bottleneck
        self.bottleneck = nn.Linear(in_features=enc_out, out_features=dec_in)
        self.add_layer = add_bottleneck
        if add_bottleneck:
            self.bottleneck2 = nn.Linear(in_features=dec_in, out_features=dec_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(x)
        h = functional.relu(self.bottleneck(enc_out))
        if self.add_layer:
            h = functional.relu(self.bottleneck2(h))

        logits = self.decoder(h)

        return functional.silu(logits)


class MultiModelSD(nn.Module):
    def __init__(self, enc_conf, enc_out: int, dec_in: int, add_bottleneck: bool = False):
        super().__init__()
        self.encoder = Encoder(enc_conf)
        self.decoder = SimpleDecoder()

        # bottleneck
        self.bottleneck = nn.Linear(in_features=enc_out, out_features=dec_in)
        self.add_layer = add_bottleneck
        if add_bottleneck:
            self.bottleneck2 = nn.Linear(in_features=dec_in, out_features=dec_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(x)
        h = functional.relu(self.bottleneck(enc_out))
        if self.add_layer:
            h = functional.relu(self.bottleneck2(h))

        logits = self.decoder(h)

        return functional.silu(logits)


class CiteClsHead(nn.Module):
    def __init__(self, input_dim: int):  # 345
        super(CiteClsHead, self).__init__()
        self.activation = functional.silu

        self.l1 = nn.Linear(in_features=input_dim, out_features=300, bias=True)
        self.dropout_1 = nn.Dropout(0.4)

        self.l2 = nn.Linear(in_features=300, out_features=256, bias=True)
        self.dropout_2 = nn.Dropout(0.4)

        self.l3 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.dropout_3 = nn.Dropout(0.4)

        self.l4 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.dropout_4 = nn.Dropout(0.4)

        self.dropout_5 = nn.Dropout(0.4)

        self.l5 = nn.Linear(in_features=940, out_features=140, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t1 = self.activation(self.l1(x))
        dt1 = self.dropout_1(t1)

        t2 = self.activation(self.l2(dt1))
        dt2 = self.dropout_2(t2)

        t3 = self.activation(self.l3(dt2))
        dt3 = self.dropout_3(t3)

        t4 = self.activation(self.l4(dt3))
        dt4 = self.dropout_4(t4)

        tc = torch.concat([dt1, dt2, dt3, dt4], dim=1)
        dtf = self.dropout_5(tc)

        logits = self.activation(self.l5(dtf))

        return logits


class CiteModel(nn.Module):
    def __init__(self, enc_conf, enc_out: int):
        super().__init__()
        self.encoder = Encoder(enc_conf)
        self.decoder = CiteClsHead(input_dim=enc_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(x)
        enc_out = torch.squeeze(enc_out)
        logits = self.decoder(enc_out)

        return logits
