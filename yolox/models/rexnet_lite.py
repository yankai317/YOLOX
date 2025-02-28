"""
ReXNet_lite
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
from math import ceil


def _make_divisible(channel_size, divisor=None, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if not divisor:
        return channel_size

    if min_value is None:
        min_value = divisor
    new_channel_size = max(min_value, int(channel_size + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channel_size < 0.9 * channel_size:
        new_channel_size += divisor
    return new_channel_size


def _add_conv(out, in_channels, channels, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=True, bn_momentum=0.1, bn_eps=1e-5):
    out.append(nn.Conv2d(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels, momentum=bn_momentum, eps=bn_eps))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.SiLU(inplace=True))


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, channels, t, kernel_size=3, stride=1,
                 bn_momentum=0.1, bn_eps=1e-5,
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.conv_shortcut = None
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels
        out = []
        if t != 1:
            dw_channels = in_channels * t
            _add_conv(out, in_channels=in_channels, channels=dw_channels, bn_momentum=bn_momentum,
                      bn_eps=bn_eps)
        else:
            dw_channels = in_channels

        _add_conv(out, in_channels=dw_channels, channels=dw_channels * 1, kernel=kernel_size, stride=stride,
                  pad=(kernel_size // 2),
                  num_group=dw_channels, bn_momentum=bn_momentum, bn_eps=bn_eps)

        _add_conv(out, in_channels=dw_channels, channels=channels, active=False, bn_momentum=bn_momentum,
                  bn_eps=bn_eps)

        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)

        if self.use_shortcut:
            out[:, 0:self.in_channels] += x
        return out


class ReXNetV1_lite(nn.Module):
    def __init__(self, fix_head_stem=False, divisible_value=8,
                 input_ch=16, final_ch=164, multiplier=1.0, classes=1000,
                 dropout_ratio=0.2,
                 bn_momentum=0.1,
                 bn_eps=1e-5, kernel_conf='333333'):
        super(ReXNetV1_lite, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        kernel_sizes = [int(element) for element in kernel_conf]

        strides = sum([[element] + [1] * (layers[idx] - 1) for idx, element in enumerate(strides)], [])
        ts = [1] * layers[0] + [6] * sum(layers[1:])
        kernel_sizes = sum([[element] * layers[idx] for idx, element in enumerate(kernel_sizes)], [])
        self.num_convblocks = sum(layers[:])
        self.out_channel = []
        features = []
        inplanes = input_ch / multiplier if multiplier < 1.0 else input_ch
        first_channel = 32 / multiplier if multiplier < 1.0 or fix_head_stem else 32
        first_channel = _make_divisible(int(round(first_channel * multiplier)), divisible_value)

        in_channels_group = []
        channels_group = []

        _add_conv(features, 3, first_channel, kernel=3, stride=2, pad=1,
                  bn_momentum=bn_momentum, bn_eps=bn_eps)

        for i in range(self.num_convblocks):
            inplanes_divisible = _make_divisible(int(round(inplanes * multiplier)), divisible_value)
            if i == 0:
                in_channels_group.append(first_channel)
                channels_group.append(inplanes_divisible)
            else:
                in_channels_group.append(inplanes_divisible)
                inplanes += final_ch / (self.num_convblocks - 1 * 1.0)
                inplanes_divisible = _make_divisible(int(round(inplanes * multiplier)), divisible_value)
                channels_group.append(inplanes_divisible)

        for block_idx, (in_c, c, t, k, s) in enumerate(
                zip(in_channels_group, channels_group, ts, kernel_sizes, strides)):
            # print(block_idx, c, s)
            if block_idx +1 == 5 or block_idx +1 == 11:
                self.out_channel.append(c)
            features.append(LinearBottleneck(in_channels=in_c,
                                             channels=c,
                                             t=t,
                                             kernel_size=k,
                                             stride=s,
                                             bn_momentum=bn_momentum,
                                             bn_eps=bn_eps))

        # pen_channels = int(1280 * multiplier) if multiplier > 1 and not fix_head_stem else 1280
        # _add_conv(features, c, pen_channels, bn_momentum=bn_momentum, bn_eps=bn_eps)

        self.features = nn.Sequential(*features)
        self.out_channel.append(c)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.output = nn.Sequential(
        #     nn.Conv2d(pen_channels, 1024, 1, bias=True),
        #     nn.BatchNorm2d(1024, momentum=bn_momentum, eps=bn_eps),
        #     nn.ReLU6(inplace=True),
        #     nn.Dropout(dropout_ratio),
        #     nn.Conv2d(1024, classes, 1, bias=True))

    def forward(self, x):
        outs = {}
        linear_idx = 0
        for m in self.features:
            x = m(x)
            if isinstance(m, LinearBottleneck):
                linear_idx += 1
                if linear_idx == 3:
                    outs['dark2'] = x
                elif linear_idx == 5:
                    outs['dark3'] = x
                elif linear_idx ==11:
                    outs['dark4'] = x
        outs['dark5'] = x
        return outs

if __name__ == '__main__':
    model = ReXNetV1_lite(multiplier=1.5)
    out = model(torch.randn(2, 3, 224, 224))
    loss = out.sum()
    loss.backward()
    print('Checked a single forward/backward iteration')

