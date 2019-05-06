'''
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
Deep Residual Learning for Image Recognition.
In CVPR, 2016.
'''

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['CifarResNet', 'cifar_resnet20', 'cifar_resnet32', 'cifar_resnet44', 'cifar_resnet56']

pretrained_settings = {
    "cifar10": {
        'resnet20': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar10-resnet20-30abc31d.pth',
        'resnet32': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar10-resnet32-e96f90cf.pth',
        'resnet44': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar10-resnet44-f2c66da5.pth',
        'resnet56': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar10-resnet56-f5939a66.pth',
        'num_classes': 10
    },
    "cifar100": {
        'resnet20': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet20-8412cc70.pth',
        'resnet32': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet32-6568a0a0.pth',
        'resnet44': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet44-20aaa8cf.pth',
        'resnet56': 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/cifar100-resnet56-2f147f26.pth',
        'num_classes': 100
    }

}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def cifar_resnet20(pretrained=None, **kwargs):
    if pretrained is None:
        model = CifarResNet(BasicBlock, [3, 3, 3], **kwargs)
    else:
        model = CifarResNet(BasicBlock, [3, 3, 3], num_classes=pretrained_settings[pretrained]['num_classes'])
        model.load_state_dict(model_zoo.load_url(pretrained_settings[pretrained]['resnet20']))
    return model


def cifar_resnet32(pretrained=None, **kwargs):
    if pretrained is None:
        model = CifarResNet(BasicBlock, [5, 5, 5], **kwargs)
    else:
        model = CifarResNet(BasicBlock, [5, 5, 5], num_classes=pretrained_settings[pretrained]['num_classes'])
        model.load_state_dict(model_zoo.load_url(pretrained_settings[pretrained]['resnet32']))
    return model


def cifar_resnet44(pretrained=None, **kwargs):
    if pretrained is None:
        model = CifarResNet(BasicBlock, [7, 7, 7], **kwargs)
    else:
        model = CifarResNet(BasicBlock, [7, 7, 7], num_classes=pretrained_settings[pretrained]['num_classes'])
        model.load_state_dict(model_zoo.load_url(pretrained_settings[pretrained]['resnet44']))
    return model


def cifar_resnet56(pretrained=None, **kwargs):
    if pretrained is None:
        model = CifarResNet(BasicBlock, [9, 9, 9], **kwargs)
    else:
        model = CifarResNet(BasicBlock, [9, 9, 9], num_classes=pretrained_settings[pretrained]['num_classes'])
        model.load_state_dict(model_zoo.load_url(pretrained_settings[pretrained]['resnet56']))
    return model
