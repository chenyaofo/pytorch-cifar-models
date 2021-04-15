'''
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/shufflenetv2.py

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import sys
import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional


cifar10_pretrained_weight_urls = {
    'shufflenetv2_x0_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar10_shufflenetv2_x0_5-1308b4e9.pt',
    'shufflenetv2_x1_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar10_shufflenetv2_x1_0-98807be3.pt',
    'shufflenetv2_x1_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar10_shufflenetv2_x1_5-296694dd.pt',
    'shufflenetv2_x2_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar10_shufflenetv2_x2_0-ec31611c.pt',
}

cifar100_pretrained_weight_urls = {
    'shufflenetv2_x0_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar100_shufflenetv2_x0_5-1977720f.pt',
    'shufflenetv2_x1_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar100_shufflenetv2_x1_0-9ae22beb.pt',
    'shufflenetv2_x1_5': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar100_shufflenetv2_x1_5-e2c85ad8.pt',
    'shufflenetv2_x2_0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar100_shufflenetv2_x2_0-e7e584cd.pt',
}


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int
    ) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int,
        o: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        stages_repeats: List[int],
        stages_out_channels: List[int],
        num_classes: int = 1000,
        inverted_residual: Callable[..., nn.Module] = InvertedResidual
    ) -> None:
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),  # NOTE: change stride 2 -> 1 for CIFAR10/100
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) NOTE: remove this maxpool layer for CIFAR10/100

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        # x = self.maxpool(x) NOTE: remove this maxpool layer for CIFAR10/100
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _shufflenet_v2(
    arch: str,
    stages_repeats: List[int],
    stages_out_channels: List[int],
    model_urls: Dict[str, str],
    progress: bool = True,
    pretrained: bool = False,
    **kwargs: Any
) -> ShuffleNetV2:
    model = ShuffleNetV2(stages_repeats=stages_repeats, stages_out_channels=stages_out_channels, ** kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def cifar10_shufflenetv2_x0_5(*args, **kwargs) -> ShuffleNetV2: pass
def cifar10_shufflenetv2_x1_0(*args, **kwargs) -> ShuffleNetV2: pass
def cifar10_shufflenetv2_x1_5(*args, **kwargs) -> ShuffleNetV2: pass
def cifar10_shufflenetv2_x2_0(*args, **kwargs) -> ShuffleNetV2: pass


def cifar100_shufflenetv2_x0_5(*args, **kwargs) -> ShuffleNetV2: pass
def cifar100_shufflenetv2_x1_0(*args, **kwargs) -> ShuffleNetV2: pass
def cifar100_shufflenetv2_x1_5(*args, **kwargs) -> ShuffleNetV2: pass
def cifar100_shufflenetv2_x2_0(*args, **kwargs) -> ShuffleNetV2: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for stages_repeats, stages_out_channels, model_name in \
        zip([[4, 8, 4]]*4,
            [[24, 48, 96, 192, 1024], [24, 116, 232, 464, 1024], [24, 176, 352, 704, 1024], [24, 244, 488, 976, 2048]],
            ["shufflenetv2_x0_5", "shufflenetv2_x1_0", "shufflenetv2_x1_5", "shufflenetv2_x2_0"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        setattr(
            thismodule,
            method_name,
            partial(_shufflenet_v2,
                    arch=model_name,
                    stages_repeats=stages_repeats,
                    stages_out_channels=stages_out_channels,
                    model_urls=model_urls,
                    num_classes=num_classes)
        )
