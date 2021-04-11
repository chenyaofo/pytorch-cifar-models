# Pytorch CIFAR Models

## Introduction

The goal of this project is to provide some neural network examples and a simple training codebase for begginners.

## Get Started with Google Colab <a href="https://colab.research.google.com/github/chenyaofo/pytorch-cifar-models/blob/master/colab/start_on_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Train Models**: Open the notebook to train the models from scratch on CIFAR10/100.
It will takes several hours depend on the complexity of the model and the allocated GPU type.

**Test Models**: Open the notebook to measure the validation accuracy on CIFAR10/100 with pretrained models.
It will only take about few seconds.

## Use Models with Pytorch Hub

You can simply use the pretrained models in your project with `torch.hub` API.
It will automatically load the code and the pretrained weights from GitHub.

``` python
import torch
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
```

To list all available model entry, you can run:

```python
import torch
from pprint import pprint
pprint(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))
```


## Model Zoo

### CIFAR-10

|  Model   |  Top-1 Acc.(%) | Top-5 Acc.(%) | #Params.(M) | #MAdds(M) |                    |
|----------|----------------|---------------|-------------|-----------|--------------------|
| resnet20 | 92.60 | 99.81 | 0.27 | 40.81 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/resnet20/default.log) \| tensorboard
| resnet32 | 93.53 | 99.77 | 0.47 | 69.12 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/resnet32/default.log) \| tensorboard
| resnet44 | 94.01 | 99.77 | 0.66 | 97.44 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/resnet44/default.log) \| tensorboard
| resnet56 | 94.37 | 99.83 | 0.86 | 125.75 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/resnet56/default.log) \| tensorboard
| vgg11_bn | 92.79 | 99.72 | 9.76 | 153.29 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/vgg11_bn/default.log) \| tensorboard
| vgg13_bn | 94.00 | 99.77 | 9.94 | 228.79 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/vgg13_bn/default.log) \| tensorboard
| vgg16_bn | 94.16 | 99.71 | 15.25 | 313.73 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/vgg16_bn/default.log) \| tensorboard
| vgg19_bn | 93.91 | 99.64 | 20.57 | 398.66 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/vgg19_bn/default.log) \| tensorboard

### CIFAR-100

|  Model   |  Top-1 Acc.(%) | Top-5 Acc.(%) | #Params.(M) | #MAdds(M) |                    |
|----------|----------------|---------------|-------------|-----------|--------------------|
| resnet20 | 68.83 | 91.01 | 0.28 | 40.82 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/resnet20/default.log) \| tensorboard
| resnet32 | 70.16 | 90.89 | 0.47 | 69.13 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/resnet32/default.log) \| tensorboard
| resnet44 | 71.63 | 91.58 | 0.67 | 97.44 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/resnet44/default.log) \| tensorboard
| resnet56 | 72.63 | 91.94 | 0.86 | 125.75 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/resnet56/default.log) \| tensorboard
| vgg11_bn | 70.78 | 88.87 | 9.80 | 153.34 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/vgg11_bn/default.log) \| tensorboard
| vgg13_bn | 74.63 | 91.09 | 9.99 | 228.84 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/vgg13_bn/default.log) \| tensorboard
| vgg16_bn | 74.00 | 90.56 | 15.30 | 313.77 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/vgg16_bn/default.log) \| tensorboard
| vgg19_bn | 73.87 | 90.13 | 20.61 | 398.71 | [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/vgg19_bn/default.log) \| tensorboard

