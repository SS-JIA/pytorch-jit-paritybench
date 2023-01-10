import sys
_module = sys.modules[__name__]
del sys
dataset = _module
dataset_stub = _module
evaluate = _module
conv1d = _module
decoder = _module
encoder = _module
group_norm = _module
mask_tcn = _module
spectrogram = _module
tasnet = _module
train = _module
data_generator = _module
logger = _module
loss = _module
ranger = _module
sgdr_learning_rate = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


from functools import partial


from random import shuffle


import torch


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data.dataloader import default_collate


import pandas as pd


from pandas.io.json import json_normalize


import torch.nn as nn


import torch.nn.functional as F


import re


import time


import random


from torch.utils.data import DataLoader


import math


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


import itertools as it


class Conv1dGenerated(nn.Module):
    """
    1D convolution with a kernel generated by a linear transformation of the instrument embedding
    """

    def __init__(self, E_1, E_2, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        """
        Arguments:
            E_1 {int} -- Dimension of the instrument embedding
            E_2 {int} -- Dimension of the instrument embedding bottleneck
            in_channels {int} -- Number of channels of the input
            out_channels {int} -- Number of channels of the output
            kernel_size {int} -- Kernel size of the convolution

        Keyword Arguments:
            stride {int} -- Stride of the convolution (default: {1})
            padding {int} -- Padding of the convolution (default: {0})
            dilation {int} -- Dilation of the convolution (default: {1})
            groups {int} -- Number of groups of the convolution (default: {1})
            bias {bool} -- Whether to use bias in the convolution (default: {False})
        """
        super(Conv1dGenerated, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bottleneck = nn.Linear(E_1, E_2) if E_1 is not None else nn.Parameter(torch.randn((4, E_2)), requires_grad=True)
        self.kernel = nn.Linear(E_2, out_channels * in_channels // groups * kernel_size)
        self.bias = nn.Linear(E_2, self.out_channels) if bias else None

    def forward(self, instrument, x):
        """
        Arguments:
            instrument {torch.tensor} -- Instrument embedding of shape (4, E_1)
            x {torch.tensor} -- Input of the convolution of shape (B, 4, C, T)

        Returns:
            torch.tensor -- Output of the convolution of shape (B, 4, C', T)
        """
        batch_size = x.shape[0]
        instrument = self.bottleneck(instrument)
        kernel = self.kernel(instrument).view(4 * self.out_channels, self.in_channels // self.groups, self.kernel_size)
        x = x.view(batch_size, 4 * self.in_channels, -1)
        x = F.conv1d(x, kernel, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=4 * self.groups)
        x = x.view(batch_size, 4, self.out_channels, -1)
        if self.bias:
            x += self.bias(instrument).view(1, 4, self.out_channels, 1)
        return x


class Conv1dStatic(nn.Module):
    """
    1D convolution with an independent kernel for each instrument
    """

    def __init__(self, _, __, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        """
        Arguments:
            in_channels {int} -- Number of channels of the input
            out_channels {int} -- Number of channels of the output
            kernel_size {int} -- Kernel size of the convolution

        Keyword Arguments:
            stride {int} -- Stride of the convolution (default: {1})
            padding {int} -- Padding of the convolution (default: {0})
            dilation {int} -- Dilation of the convolution (default: {1})
            groups {int} -- Number of groups of the convolution (default: {1})
            bias {bool} -- Whether to use bias in the convolution (default: {False})
        """
        super(Conv1dStatic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(4 * in_channels, 4 * out_channels, kernel_size, stride, padding, dilation, 4 * groups, bias)

    def forward(self, _, x):
        """
        Arguments:
            _ {None} -- unused argument (for compatibility with Conv1dGenerated)
            x {torch.tensor} -- Input of the convolution of shape (B, 4, C, T)

        Returns:
            torch.tensor -- Output of the convolution of shape (B, 4, C', T)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, 4 * self.in_channels, -1)
        x = self.conv(x)
        x = x.view(batch_size, 4, self.out_channels, -1)
        return x


class Decoder(nn.Module):
    """
    Decodes the latent representation back to waveforms
    """

    def __init__(self, N, kernel_size, stride, layers):
        """
        Arguments:
            N {int} -- Dimension of the input latent representation
            kernel_size {int} -- Base convolutional kernel size
            stride {int} -- Stride of the transposed covolutions
            layers {int} -- Number of parallel convolutions with different kernel sizes
        """
        super(Decoder, self).__init__()
        self.filter_widths = [(N // 2 ** (l + 1)) for l in range(layers)]
        total_input_width = np.array(self.filter_widths).sum()
        self.bottleneck = nn.Sequential(nn.ConvTranspose1d(N, total_input_width, kernel_size=1, stride=1, bias=False), nn.ReLU())
        self.filters = nn.ModuleList([])
        for l in range(layers):
            n = N // 2 ** (l + 1)
            k = kernel_size * 2 ** l
            self.filters.append(nn.ConvTranspose1d(n, 1, kernel_size=k, stride=stride, bias=False, padding=(k - stride) // 2))

    def forward(self, x):
        """
        Arguments:
            x {torch.tensor} -- Latent representation of the four instrument with shape (B*4, N, T')

        Returns:
            torch.tensor -- Signal of the four instruments with shape (B*4, 1, T)
        """
        x = self.bottleneck(x)
        output = 0.0
        x = x.split(self.filter_widths, dim=1)
        for i in range(len(x)):
            output += self.filters[i](x[i])
        return output


class Spectrogram(nn.Module):
    """
    Calculate the mel spectrogram as an additional input for the encoder
    """

    def __init__(self, n_fft, hop, mels, sr):
        """
        Arguments:
            n_fft {int} -- The number fo frequency bins
            hop {int} -- Hop size (stride)
            mels {int} -- The number of mel filters
            sr {int} -- Sampling rate of the signal
        """
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.mels = mels
        self.sr = sr
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        stft_size = n_fft // 2 + 1
        self.mel_transform = nn.Conv1d(stft_size, mels, kernel_size=1, stride=1, padding=0, bias=True)
        self.mean = nn.Parameter(torch.empty(1, stft_size, 1), requires_grad=False)
        self.std = nn.Parameter(torch.empty(1, stft_size, 1), requires_grad=False)
        self.affine_bias = nn.Parameter(torch.zeros(1, stft_size, 1), requires_grad=True)
        self.affine_scale = nn.Parameter(torch.ones(1, stft_size, 1), requires_grad=True)

    def forward(self, audio_signal, target_length=None):
        """
        Arguments:
            audio_signal {torch.tensor} -- input tensor of shape (B, 1, T)

        Keyword Arguments:
            target_length {int, None} -- Optional argument for interpolating the time dimension of the result to $target_length (default: {None})

        Returns:
            torch.tensor -- mel spectrogram of shape (B, mels, T')
        """
        mag = self.calculate_mag(audio_signal, db_conversion=True)
        mag = (mag - self.mean) / self.std
        mag = mag * self.affine_scale + self.affine_bias
        mag = self.mel_transform(mag)
        if target_length is not None:
            mag = F.interpolate(mag, size=target_length, mode='linear', align_corners=True)
        return mag

    def calculate_mag(self, signal, db_conversion=True):
        """
        Calculate the dB magnitude of the STFT of the input signal

        Arguments:
            audio_signal {torch.tensor} -- input tensor of shape (B, 1, T)

        Keyword Arguments:
            db_conversion {bool} -- True if the method should logaritmically transform the result to dB (default: {True})

        Returns:
            torch.tensor -- output tensor of shape (B, N', T')
        """
        signal = signal.view(-1, signal.shape[-1])
        stft = torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, normalized=False, onesided=True, pad_mode='reflect')
        mag = (stft ** 2).sum(-1)
        if db_conversion:
            mag = torch.log10(mag + 1e-08)
        return mag

    def compute_stats(self, dataset, portion):
        """
        Calculate the mean and std statistics of the dataset

        Arguments:
            dataset {MusicDataset} -- MusicDataset class
            portion {int from {0,1,2}} -- Used to select data with only one value of the sampling rate
        """
        with torch.no_grad():
            specgrams = []
            samples = 5000
            for i_batch, (mix, _, _) in enumerate(dataset):
                mix = mix[portion]
                spec = self.calculate_mag(mix, db_conversion=True)
                specgrams.append(spec)
                if (i_batch + 1) * mix.shape[0] > samples:
                    break
            specgrams = torch.cat(specgrams, 0)
            self.mean.data = specgrams.mean(dim=(0, 2), keepdim=True)
            self.std.data = specgrams.std(dim=(0, 2), keepdim=True)
        None


class Encoder(nn.Module):
    """
    Encodes the waveforms into the latent representation
    """

    def __init__(self, N, kernel_size, stride, layers, num_mels, sampling_rate):
        """
        Arguments:
            N {int} -- Dimension of the output latent representation
            kernel_size {int} -- Base convolutional kernel size
            stride {int} -- Stride of the convolutions
            layers {int} -- Number of parallel convolutions with different kernel sizes
            num_mels {int} -- Number of mel filters in the mel spectrogram
            sampling_rate {int} -- Sampling rate of the input
        """
        super(Encoder, self).__init__()
        K = sampling_rate // 8000
        self.spectrogram = Spectrogram(n_fft=1024 * K, hop=256 * K, mels=num_mels, sr=sampling_rate)
        self.filters = nn.ModuleList([])
        filter_width = num_mels
        for l in range(layers):
            n = N // 4
            k = kernel_size * 2 ** l
            self.filters.append(nn.Conv1d(1, n, kernel_size=k, stride=stride, bias=False, padding=(k - stride) // 2))
            filter_width += n
        self.nonlinearity = nn.ReLU()
        self.bottleneck = nn.Sequential(nn.Conv1d(filter_width, N, kernel_size=1, stride=1, bias=False), nn.ReLU(), nn.Conv1d(N, N, kernel_size=1, stride=1, bias=False))

    def forward(self, signal):
        """
        Arguments:
            signal {torch.tensor} -- mixed signal of shape (B, 1, T)

        Returns:
            torch.tensor -- latent representation of shape (B, N, T)
        """
        convoluted_x = []
        for filter in self.filters:
            x = filter(signal).unsqueeze(-2)
            convoluted_x.append(x)
        x = torch.cat(convoluted_x, dim=-2)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = self.nonlinearity(x)
        spectrogram = self.spectrogram(signal, x.shape[-1])
        x = torch.cat([x, spectrogram], dim=1)
        return self.bottleneck(x)


class GroupNormGenerated(nn.Module):
    """
    Group normalization layer with scale and bias factor created with a linear transformation of the instrument embedding
    """

    def __init__(self, E_1, E_2, num_groups, num_channels, eps=1e-08):
        """
        Arguments:
            E_1 {int} -- Dimension of the instrument embedding
            E_2 {int} -- Dimension of the instrument embedding bottleneck
            num_groups {int} -- Number of normalized groups
            num_channels {int} -- Number of channels

        Keyword Arguments:
            eps {int} -- Constant for numerical stability (default: {1e-8})
        """
        super(GroupNormGenerated, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.bottleneck = nn.Linear(E_1, E_2)
        self.affine = nn.Linear(E_2, num_channels + num_channels)

    def forward(self, instrument, x):
        """
        Arguments:
            instrument {torch.tensor} -- Instrument embedding of shape (4, E_1)
            x {torch.tensor} -- Input of the groupnorm of shape (B, 4, C, T)

        Returns:
            torch.tensor -- Output of the groupnorm of shape (B, 4, C, T)
        """
        batch_size = x.shape[0]
        instrument = self.bottleneck(instrument)
        affine = self.affine(instrument)
        scale = affine[:, :self.num_channels].contiguous().view(-1)
        bias = affine[:, self.num_channels:].contiguous().view(-1)
        x = x.view(batch_size, 4 * self.num_channels, -1)
        x = F.group_norm(x, 4 * self.num_groups, scale, bias, self.eps)
        x = x.view(batch_size, 4, self.num_channels, -1)
        return x


class GroupNormStatic(nn.Module):
    """
    Group normalization layer with an independent scale and bias factor for each instrument
    """

    def __init__(self, _, __, num_groups, num_channels, eps=1e-08):
        """
        Arguments:
            num_groups {int} -- Number of normalized groups
            num_channels {int} -- Number of channels

        Keyword Arguments:
            eps {int} -- Constant for numerical stability (default: {1e-8})
        """
        super(GroupNormStatic, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.group_norm = nn.GroupNorm(4 * num_groups, 4 * num_channels, eps)

    def forward(self, _, x):
        """
        Arguments:
            _ {None} -- unused argument (for compatibility with GroupNormGenerated)
            x {torch.tensor} -- Input of the groupnorm of shape (B, 4, C, T)

        Returns:
            torch.tensor -- Output of the groupnorm of shape (B, 4, C, T)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, 4 * self.num_channels, -1)
        x = self.group_norm(x)
        x = x.view(batch_size, 4, self.num_channels, -1)
        return x


def Conv1dWrapper(generated, *args, **kwargs):
    """
    Wrapper around the convolutional layer generated by an instrument embedding, and standard static convolution

    Arguments:
        generated {bool} -- True if you want to use the generated convolution
        *args -- Positional arguments passed to the __init__ function of the chosen module
        **kwargs -- Keyword arguments passed to the __init__ function of the chosen module

    Returns:
        nn.Module
    """
    if generated:
        return Conv1dGenerated(*args, **kwargs)
    else:
        return Conv1dStatic(*args, **kwargs)


def GroupNormWrapper(generated, *args, **kwargs):
    """
    Wrapper around the group normalization layer generated by an instrument embedding, and standard group normalization

    Arguments:
        generated {bool} -- True if you want to use the generated groupnorm
        *args -- Positional arguments passed to the __init__ function of the chosen module
        **kwargs -- Keyword arguments passed to the __init__ function of the chosen module

    Returns:
        nn.Module
    """
    if generated:
        return GroupNormGenerated(*args)
    else:
        return GroupNormStatic(*args)


class TCNLayer(nn.Module):
    """
    One layer of the dilated temporal convolution with bottleneck
    """

    def __init__(self, generated, E_1, E_2, B, H, Sc, kernel, residual_bias, padding, dilation=1):
        """
        Arguments:
            generated {bool} -- True if you want to use the generated weights
            E_1 {int} -- Dimension of the instrument embedding
            E_2 {int} -- Dimension of the instrument embedding bottleneck
            B {int} -- Dimension of the bottleneck convolution
            H {int} -- Hidden dimension
            Sc {int} -- Skip-connection dimension
            kernel {int} -- Kernel size of the dilated convolution
            residual_bias {bool} -- True if you want to apply bias to the residual and skip connections
            padding {int} -- Padding of the dilated convolution

        Keyword Arguments:
            dilation {int} -- Dilation of the dilated convolution (default: {1})
        """
        super(TCNLayer, self).__init__()
        self.norm_1 = GroupNormWrapper(generated, E_1, E_2, 8, H, eps=1e-08)
        self.prelu_1 = nn.PReLU()
        self.conv1d = Conv1dWrapper(generated, E_1, E_2, B, H, 1, bias=False)
        self.norm_2 = GroupNormWrapper(generated, E_1, E_2, 8, H, eps=1e-08)
        self.prelu_2 = nn.PReLU()
        self.dconv1d = Conv1dWrapper(generated, E_1, E_2, H, H, kernel, dilation=dilation, groups=H, padding=padding, bias=False)
        self.res_out = Conv1dWrapper(generated, E_1, E_2, H, B, 1, bias=residual_bias)
        self.skip_out = Conv1dWrapper(generated, E_1, E_2, H, Sc, 1, bias=residual_bias)

    def forward(self, instrument, x):
        """
        Arguments:
            instrument {torch.tensor} -- Instrument embedding of shape (4, E_1)
            x {torch.tensor} -- Input of the module of shape (B, 4, B, T)

        Returns:
            (torch.tensor, torch.tensor) -- Output of the module of shape [(B, 4, B, T), (B, 4, Sc, T)]
        """
        x = self.norm_1(instrument, self.prelu_1(self.conv1d(instrument, x)))
        x = self.norm_2(instrument, self.prelu_2(self.dconv1d(instrument, x)))
        residual = self.res_out(instrument, x)
        skip = self.skip_out(instrument, x)
        return residual, skip


class MaskingModule(nn.Module):
    """
    Creates a [0,1] mask of the four instruments on the latent matrix
    """

    def __init__(self, generated, E_1, E_2, N, B, H, layer, stack, kernel=3, residual_bias=False, partial_input=False):
        """
        Arguments:
            generated {bool} -- True if you want to use the generated weights
            E_1 {int} -- Dimension of the instrument embedding
            E_2 {int} -- Dimension of the instrument embedding bottleneck
            N {int} -- Dimension of the latent matrix
            B {int} -- Dimension of the bottleneck convolution
            H {int} -- Hidden dimension
            layer {[type]} -- Number of temporal convolution layers in a stack
            stack {[type]} -- Number of stacks

        Keyword Arguments:
            kernel {int} -- Kernel size of the dilated convolution (default: {3})
            residual_bias {bool} -- True if you want to apply bias to the residual and skip connections (default: {False})
            partial_input {bool} -- True if the module expects input from the preceding masking module (default: {False})
        """
        super(MaskingModule, self).__init__()
        self.N = N
        self.in_N = N + N // 2 if partial_input else N
        self.norm_1 = GroupNormWrapper(generated, E_1, E_2, 8, self.in_N, eps=1e-08)
        self.prelu_1 = nn.PReLU()
        self.in_conv = Conv1dWrapper(generated, E_1, E_2, self.in_N, B, 1, bias=False)
        self.norm_2 = GroupNormWrapper(generated, E_1, E_2, 8, B, eps=1e-08)
        self.prelu_2 = nn.PReLU()
        self.tcn = nn.ModuleList([TCNLayer(generated, E_1, E_2, B, H, B, kernel, residual_bias, dilation=2 ** i, padding=2 ** i) for _ in range(stack) for i in range(layer)])
        self.norm_3 = GroupNormWrapper(generated, E_1, E_2, 8, B, eps=1e-08)
        self.prelu_3 = nn.PReLU()
        self.mask_output = Conv1dWrapper(generated, E_1, E_2, B, N, 1, bias=False)
        self.norm_4 = GroupNormWrapper(generated, E_1, E_2, 8, N, eps=1e-08)
        self.prelu_4 = nn.PReLU()

    def forward(self, instrument, x, partial_input=None):
        """
        Arguments:
            instrument {torch.tensor} -- Instrument embedding of shape (4, E_1)
            x {torch.tensor} -- Latent representation of the mix of shape (B, 4, N, T) (expanded in the 2nd dimension)
            partial_input {torch.tensor, None} -- Optional input from the preceding masking module of shape (B, 4, N/2, T)

        Returns:
            torch.tensor -- [0,1] mask of shape (B, 4, N, T)
        """
        if partial_input is not None:
            x = torch.cat([x, partial_input], 2)
        x = self.in_conv(instrument, self.norm_1(instrument, self.prelu_1(x)))
        x = self.norm_2(instrument, self.prelu_2(x))
        skip_connection = 0.0
        for layer in self.tcn:
            residual, skip = layer(instrument, x)
            x = x + residual
            skip_connection = skip_connection + skip
        mask = self.mask_output(instrument, self.norm_3(instrument, self.prelu_3(skip_connection)))
        mask = self.norm_4(instrument, self.prelu_4(mask))
        return F.softmax(mask, dim=1)


def dissimilarity_loss(latents, mask):
    """
    Minimize the similarity between the different instrument latent representations

    Arguments:
        latents {torch.tensor} -- latent matrix from the encoder of shape: (B, 1, T', N)
        mask {torch.tensor} -- boolean mask: True when the signal is 0.0; shape (B, 4)

    Returns:
        torch.tensor -- shape: ()
    """
    a_i = 0, 0, 0, 1, 1, 2
    b_i = 1, 2, 3, 2, 3, 3
    a = latents[a_i, :, :, :]
    b = latents[b_i, :, :, :]
    count = (mask[:, a_i] * mask[:, b_i]).sum() + 1e-08
    sim = F.cosine_similarity(a.abs(), b.abs(), dim=-1)
    sim = sim.sum(dim=(0, 1)) / count
    return sim.mean()


def sdr_objective(estimation, origin, mask=None):
    """
    Scale-invariant signal-to-noise ratio (SI-SNR) loss

    Arguments:
        estimation {torch.tensor} -- separated signal of shape: (B, 4, 1, T)
        origin {torch.tensor} -- ground-truth separated signal of shape (B, 4, 1, T)

    Keyword Arguments:
        mask {torch.tensor, None} -- boolean mask: True when $origin is 0.0; shape (B, 4, 1) (default: {None})

    Returns:
        torch.tensor -- SI-SNR loss of shape: (4)
    """
    origin_power = torch.pow(origin, 2).sum(dim=-1, keepdim=True) + 1e-08
    scale = torch.sum(origin * estimation, dim=-1, keepdim=True) / origin_power
    est_true = scale * origin
    est_res = estimation - est_true
    true_power = torch.pow(est_true, 2).sum(dim=-1).clamp(min=1e-08)
    res_power = torch.pow(est_res, 2).sum(dim=-1).clamp(min=1e-08)
    sdr = 10 * (torch.log10(true_power) - torch.log10(res_power))
    if mask is not None:
        sdr = (sdr * mask).sum(dim=(0, -1)) / mask.sum(dim=(0, -1)).clamp(min=1e-08)
    else:
        sdr = sdr.mean(dim=(0, -1))
    return sdr


def similarity_loss(latents, mask):
    """
    Maximize the similarity between the same instrument latent representations

    Arguments:
        latents {torch.tensor} -- latent matrix from the encoder of shape: (B, 1, T', N)
        mask {torch.tensor} -- boolean mask: True when the signal is 0.0; shape (B, 4)

    Returns:
        torch.tensor -- shape: ()
    """
    a = latents
    b = torch.roll(latents, 1, dims=1)
    count = (mask * torch.roll(mask, 1, dims=0)).sum().clamp(min=1e-08)
    sim = F.cosine_similarity(a, b, dim=-1)
    sim = sim.sum(dim=(0, 1)) / count
    return sim.mean()


def calculate_loss(estimated_separation, true_separation, mask, true_latents, estimated_mix, true_mix, args):
    """
    The loss function, the sum of 4 different partial losses

    Arguments:
        estimated_separation {torch.tensor} -- separated signal of shape: (B, 4, 1, T)
        true_separation {torch.tensor} -- ground-truth separated signal of shape (B, 4, 1, T)
        mask {torch.tensor} -- boolean mask: True when $true_separation is 0.0; shape (B, 4, 1)
        true_latents {torch.tensor} -- latent matrix from the encoder of shape: (B, 1, T', N)
        estimated_mix {torch.tensor} -- estimated reconstruction of the mix, shape: (B, 1, T)
        true_mix {torch.tensor} -- ground-truth mixed signal, shape: (B, 1, T)
        args {dict} -- argparse hyperparameters

    Returns:
        (torch.tensor, torch.tensor) -- shape: [(), (7)]
    """
    stats = torch.zeros(7)
    sdr = sdr_objective(estimated_separation, true_separation, mask)
    stats[:4] = sdr
    total_loss = -sdr.sum()
    reconstruction_sdr = sdr_objective(estimated_mix, true_mix).mean() if args.reconstruction_loss_weight > 0 else 0.0
    stats[4] = reconstruction_sdr
    total_loss += -args.reconstruction_loss_weight * reconstruction_sdr
    if args.similarity_loss_weight > 0.0 or args.dissimilarity_loss_weight > 0.0:
        mask = mask.squeeze(-1)
        true_latents = true_latents * mask.unsqueeze(-1).unsqueeze(-1)
        true_latents = true_latents.transpose(0, 1)
    dissimilarity = dissimilarity_loss(true_latents, mask) if args.dissimilarity_loss_weight > 0.0 else 0.0
    stats[5] = dissimilarity
    total_loss += args.dissimilarity_loss_weight * dissimilarity
    similarity = similarity_loss(true_latents, mask) if args.similarity_loss_weight > 0.0 else 0.0
    stats[6] = similarity
    total_loss += -args.similarity_loss_weight * similarity
    return total_loss, stats


class TasNet(nn.Module):
    """
    One stage of encoder->mask->decoder for a single sampling rate
    """

    def __init__(self, independent_params, N, L, W, B, H, sr, partial_input, args):
        """
        Arguments:
            independent_params {bool} -- False if you want to use the generated weights
            N {int} -- Dimension of the latent matrix
            L {int} -- Dimension of the latent representation
            W {int} -- Kernel size of the en/decoder transfomation
            B {int} -- Dimension of the bottleneck convolution in the masking subnetwork
            H {int} -- Hidden dimension of the masking subnetwork
            sr {int} -- Sampling rate of the processed signal
            partial_input {bool} -- True if the module should expect input from preceding stage
            args {dict} -- Other argparse hyperparameters
        """
        super(TasNet, self).__init__()
        assert sr * 4 % L == 0
        self.N = N
        self.stride = W
        self.out_channels = 1
        self.C = 4
        self.encoder = Encoder(self.N, L, W, args.filters, args.num_mels, sr)
        self.decoder = Decoder(self.N, L, W, args.filters)
        self.dropout = nn.Dropout2d(args.dropout)
        self.mask = MaskingModule(not independent_params, args.E_1, args.E_2, N, B, H, args.layers, args.stack, args.kernel, args.residual_bias, partial_input=partial_input)
        self.instrument_embedding = nn.Embedding(self.C, args.E_1) if not independent_params and args.E_1 is not None else None
        self.args = args

    def forward(self, input_mix, separated_inputs, mask, partial_input=None):
        """
        Forward pass for training; returns the loss and hidden state to be passed to the next stage

        Arguments:
            input_mix {torch.tensor} -- Mixed signal of shape (B, 1, T)
            separated_inputs {torch.tensor} -- Ground truth separated mixed signal of shape (B, 4, 1, T)
            mask {torch.tensor} -- Boolean mask: True when $separated_inputs is 0.0; shape: (B, 4, 1)

        Keyword Arguments:
            partial_input {torch.tensor, None} -- Optional input from the preceding masking module of shape (B, 4, N/2, T') (default: {None})

        Returns:
            (torch.tensor, torch.tensor, torch.tensor) -- (
                the total loss of shape (),
                list of statistics with partial losses and metrics of shape (7),
                partial input to be passed to the next stage of shape (B, 4, N, T')
            )
        """
        batch_size = input_mix.shape[0]
        mix_latent = self.encoder(input_mix)
        mix_latents = mix_latent.unsqueeze(1)
        mix_latents = mix_latents.expand(-1, self.C, -1, -1).contiguous()
        if self.args.similarity_loss_weight > 0.0 or self.args.dissimilarity_loss_weight > 0.0:
            separated_gold_latents = self.encoder(separated_inputs.view(self.C * batch_size, input_mix.shape[1], -1))
            separated_gold_latents = separated_gold_latents.view(batch_size, self.C, self.N, -1).permute(0, 1, 3, 2).contiguous()
        else:
            separated_gold_latents = None
        instruments = torch.arange(0, self.C, device=mix_latent.device)
        if self.instrument_embedding is not None:
            instruments = self.instrument_embedding(instruments)
        mask_input = self.dropout(mix_latents.view(batch_size * self.C, self.N, -1).unsqueeze(-1)).squeeze(-1).view(batch_size, self.C, self.N, -1)
        masks = self.mask(instruments, mask_input, partial_input)
        separated_latents = mix_latents * masks
        decoder_input = separated_latents.view(batch_size * self.C, self.N, -1)
        output_signal = self.decoder(decoder_input)
        output_signal = output_signal.view(batch_size, self.C, self.out_channels, -1)
        if self.args.reconstruction_loss_weight > 0:
            reconstruction = self.decoder(mix_latent)
        else:
            reconstruction = None
        loss, stats = calculate_loss(output_signal, separated_inputs, mask, separated_gold_latents, reconstruction, input_mix, self.args)
        return loss, stats, separated_latents

    def inference(self, x, partial_input=None):
        """
        Forward pass for inference; returns the separated signal and hidden state to be passed to the next stage

        Arguments:
            x {torch.tensor} -- mixed signal of shape (1, 1, T)

        Keyword Arguments:
            partial_input {torch.tensor, None} -- Optional input from the preceding masking module of shape (B, 4, N/2, T') (default: {None})

        Returns:
            (torch.tensor, torch.tensor) -- (
                separated signal of shape (1, 4, 1, T),
                hidden state to be passed to the next stage of shape (1, 4, N, T')
            )
        """
        x = self.encoder(x)
        x = x.expand(self.C, -1, -1).unsqueeze_(0)
        if partial_input is not None:
            mask_input = torch.cat([x, partial_input], 2)
        else:
            mask_input = x
        del partial_input
        instruments = torch.arange(0, self.C, device=x.device)
        if self.instrument_embedding is not None:
            instruments = self.instrument_embedding(instruments)
        masks = self.mask(instruments, mask_input)
        del mask_input
        x = x * masks
        del masks
        x.squeeze_(0)
        hidden = x
        x = self.decoder(x)
        return x.unsqueeze_(0), hidden.unsqueeze_(0)


class MultiTasNet(nn.Module):
    """
    Multiple stages of Tasnet stacked sequentially
    """

    def __init__(self, args):
        """
        Arguments:
            args {dict} -- Other argparse hyperparameters
        """
        super(MultiTasNet, self).__init__()
        self.args = args
        self.W = args.W
        self.base_sr = args.sampling_rate
        self.stages_num = args.stages_num
        self.stages = nn.ModuleList([])
        for stage_i in range(self.stages_num):
            m = 2 ** stage_i
            stage = TasNet(args.independent_params, m * args.N, m * args.L, m * args.W, args.B, args.H, m * args.sampling_rate, partial_input=stage_i != 0, args=args)
            self.stages.append(stage)

    def forward(self, input_mixes, separated_inputs, masks):
        """
        Forward pass for training

        Arguments:
            input_mixes {[torch.tensor]} -- List of mixed signals for all stages of shape (B, 1, T)
            separated_inputs {[torch.tensor]} -- List of ground truth separated mixed signal of shape (B, 4, 1, T)
            masks {[torch.tensor]} -- List of boolean mask: True when $separated_inputs is 0.0; shape: (B, 4, 1)

        Returns:
            (torch.tensor, torch.tensor) -- (
                the total loss of shape (1),
                list of statistics with partial losses and metrics (15)
            )
        """
        assert len(input_mixes) == self.stages_num
        assert len(separated_inputs) == self.stages_num
        assert len(masks) == self.stages_num
        loss, stats, hidden = None, None, None
        for i, stage in enumerate(self.stages):
            _loss, _stats, hidden = stage(input_mixes[i], separated_inputs[i], masks[i], hidden)
            loss = _loss if loss is None else loss + _loss
            stats = _stats if stats is None else torch.cat([stats[:i * 4], _stats], 0)
        stats.unsqueeze_(0)
        loss.unsqueeze_(0)
        return loss, stats

    def inference(self, input_audio, n_chunks=4):
        """
        Forward pass for inference; returns the separated signal

        Arguments:
            input_audio {torch.tensor} -- List of mixed signals for all stages of shape (B, 1, T)

        Keyword Arguments:
            n_chunks {int} -- Divide the $input_audio to chunks to trade speed for memory (default: {4})

        Returns:
            torch.tensor -- Separated signal of shape (1, 4, 1, T)
        """
        assert len(input_audio) == self.stages_num
        chunks = [int(input_audio[0].shape[-1] / n_chunks * c + 0.5) for c in range(n_chunks)]
        chunks.append(input_audio[0].shape[-1])
        chunk_intervals = [(max(0, chunks[n] - self.base_sr * 8), min(chunks[n + 1] + self.base_sr * 8, input_audio[0].shape[-1])) for n in range(n_chunks)]
        chunk_intervals = [((s, e - (e - s) % self.W) if s == 0 else (s + (e - s) % self.W, e)) for s, e in chunk_intervals]
        full_outputs = None
        for c in range(n_chunks):
            outputs, hidden = [], None
            for i, stage in enumerate(self.stages):
                m = 2 ** i
                output, hidden = stage.inference(input_audio[i][:, :, m * chunk_intervals[c][0]:m * chunk_intervals[c][1]], hidden)
                output = output[:, :, :, m * (chunks[c] - chunk_intervals[c][0]):output.shape[-1] - m * (chunk_intervals[c][1] - chunks[c + 1])]
                outputs.append(output)
            del hidden
            if full_outputs is None:
                full_outputs = outputs
            else:
                full_outputs = [torch.cat([f, o], -1) for f, o in zip(full_outputs, outputs)]
        return full_outputs

    def compute_stats(self, train_loader):
        """
        Calculate the mean and std statistics of the dataset for the spectrogram modules

        Arguments:
            train_loader {MusicDataset}
        """
        for i, stage in enumerate(self.stages):
            stage.encoder.spectrogram.compute_stats(train_loader, i)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv1dStatic,
     lambda: ([], {'_': 4, '__': 4, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Decoder,
     lambda: ([], {'N': 4, 'kernel_size': 4, 'stride': 1, 'layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GroupNormStatic,
     lambda: ([], {'_': 4, '__': 4, 'num_groups': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_pfnet_research_meta_tasnet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

