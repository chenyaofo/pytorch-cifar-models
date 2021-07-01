'''
Modified from https://raw.githubusercontent.com/DingXiaoH/RepVGG/main/repvgg.py

MIT License

Copyright (c) 2020 DingXiaoH

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import sys
import copy
import torch

import torch.nn as nn
import numpy as np

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from functools import partial
from typing import Union, List, Dict, Any, cast

cifar10_pretrained_weight_urls = {
    'repvgg_a0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar10_repvgg_a0-ef08a50e.pt',
    'repvgg_a1': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar10_repvgg_a1-38d2431b.pt',
    'repvgg_a2': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar10_repvgg_a2-09488915.pt',
}

cifar100_pretrained_weight_urls = {
    'repvgg_a0': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar100_repvgg_a0-2df1edd0.pt',
    'repvgg_a1': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar100_repvgg_a1-c06b21a7.pt',
    'repvgg_a2': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar100_repvgg_a2-8e71b1f8.pt',
}


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)


#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
#   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.


    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')


class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes,
                                  kernel_size=3, stride=1, padding=1, deploy=self.deploy)  # NOTE: change stride 2 -> 1 for CIFAR10/100
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=1)  # NOTE: change stride 2 -> 1 for CIFAR10/100
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def convert_to_inference_model(self, do_copy=False):
        model = copy.deepcopy(self) if do_copy else self
        for module in model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def _repvgg(arch: str, num_blocks: List[int], width_multiplier: List[float],
            model_urls: Dict[str, str],
            pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RepVGG:
    model = RepVGG(num_blocks=num_blocks, width_multiplier=width_multiplier,
                   override_groups_map=None, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def cifar10_repvgg_a0(*args, **kwargs) -> RepVGG: pass
def cifar10_repvgg_a1(*args, **kwargs) -> RepVGG: pass
def cifar10_repvgg_a2(*args, **kwargs) -> RepVGG: pass


def cifar100_repvgg_a0(*args, **kwargs) -> RepVGG: pass
def cifar100_repvgg_a1(*args, **kwargs) -> RepVGG: pass
def cifar100_repvgg_a2(*args, **kwargs) -> RepVGG: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for num_blocks, width_multiplier, model_name in\
            zip([[2, 4, 14, 1], [2, 4, 14, 1], [2, 4, 14, 1]],
                [[0.75, 0.75, 0.75, 2.5], [1, 1, 1, 2.5], [1.5, 1.5, 1.5, 2.75]],
                ["repvgg_a0", "repvgg_a1", "repvgg_a2"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        setattr(
            thismodule,
            method_name,
            partial(_repvgg,
                    arch=model_name,
                    num_blocks=num_blocks,
                    width_multiplier=width_multiplier,
                    model_urls=model_urls,
                    num_classes=num_classes)
        )
