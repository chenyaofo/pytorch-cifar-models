# PyTorch CIFAR Models

## Introduction

The goal of this project is to provide some neural network examples and a simple training codebase for begginners.

## Get Started with Google Colab <a href="https://colab.research.google.com/github/chenyaofo/pytorch-cifar-models/blob/master/colab/start_on_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Train Models**: Open the notebook to train the models from scratch on CIFAR10/100.
It will takes several hours depend on the complexity of the model and the allocated GPU type.

**Test Models**: Open the notebook to measure the validation accuracy on CIFAR10/100 with pretrained models.
It will only take about few seconds.

## Use Models with Pytorch Hub

You can simply use the pretrained models in your project with `torch.hub` API.
It will automatically load the code and the pretrained weights from GitHub
(If you cannot directly access GitHub, please check [this issue](https://github.com/chenyaofo/pytorch-cifar-models/issues/14) for solution).

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
| resnet20 | 92.60 | 99.81 | 0.27 | 40.81 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/resnet20/default.log)
| resnet32 | 93.53 | 99.77 | 0.47 | 69.12 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/resnet32/default.log)
| resnet44 | 94.01 | 99.77 | 0.66 | 97.44 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/resnet44/default.log)
| resnet56 | 94.37 | 99.83 | 0.86 | 125.75 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/resnet56/default.log)
| vgg11_bn | 92.79 | 99.72 | 9.76 | 153.29 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg11_bn-eaeebf42.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/vgg11_bn/default.log)
| vgg13_bn | 94.00 | 99.77 | 9.94 | 228.79 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg13_bn-c01e4a43.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/vgg13_bn/default.log)
| vgg16_bn | 94.16 | 99.71 | 15.25 | 313.73 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/vgg16_bn/default.log)
| vgg19_bn | 93.91 | 99.64 | 20.57 | 398.66 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg19_bn-57191229.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/vgg19_bn/default.log)
| mobilenetv2_x0_5 | 92.88 | 99.86 | 0.70 | 27.97 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x0_5-ca14ced9.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/mobilenetv2_x0_5/default.log)
| mobilenetv2_x0_75 | 93.72 | 99.79 | 1.37 | 59.31 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x0_75-a53c314e.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/mobilenetv2_x0_75/default.log)
| mobilenetv2_x1_0 | 93.79 | 99.73 | 2.24 | 87.98 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x1_0-fe6a5b48.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/mobilenetv2_x1_0/default.log)
| mobilenetv2_x1_4 | 94.22 | 99.80 | 4.33 | 170.07 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x1_4-3bbbd6e2.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/mobilenetv2_x1_4/default.log)
| shufflenetv2_x0_5 | 90.13 | 99.70 | 0.35 | 10.90 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar10_shufflenetv2_x0_5-1308b4e9.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/shufflenetv2_x0_5/default.log)
| shufflenetv2_x1_0 | 92.98 | 99.73 | 1.26 | 45.00 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar10_shufflenetv2_x1_0-98807be3.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/shufflenetv2_x1_0/default.log)
| shufflenetv2_x1_5 | 93.55 | 99.77 | 2.49 | 94.26 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar10_shufflenetv2_x1_5-296694dd.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/shufflenetv2_x1_5/default.log)
| shufflenetv2_x2_0 | 93.81 | 99.79 | 5.37 | 187.81 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar10_shufflenetv2_x2_0-ec31611c.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/shufflenetv2_x2_0/default.log)
| repvgg_a0 | 94.39 | 99.82 | 7.84 | 489.08 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar10_repvgg_a0-ef08a50e.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/repvgg_a0/default.log)
| repvgg_a1 | 94.89 | 99.83 | 12.82 | 851.33 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar10_repvgg_a1-38d2431b.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/repvgg_a1/default.log)
| repvgg_a2 | 94.98 | 99.82 | 26.82 | 1850.10 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar10_repvgg_a2-09488915.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar10/repvgg_a2/default.log)

### CIFAR-100

|  Model   |  Top-1 Acc.(%) | Top-5 Acc.(%) | #Params.(M) | #MAdds(M) |                    |
|----------|----------------|---------------|-------------|-----------|--------------------|
| resnet20 | 68.83 | 91.01 | 0.28 | 40.82 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/resnet20/default.log)
| resnet32 | 70.16 | 90.89 | 0.47 | 69.13 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet32-84213ce6.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/resnet32/default.log)
| resnet44 | 71.63 | 91.58 | 0.67 | 97.44 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet44-ffe32858.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/resnet44/default.log)
| resnet56 | 72.63 | 91.94 | 0.86 | 125.75 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/resnet56/default.log)
| vgg11_bn | 70.78 | 88.87 | 9.80 | 153.34 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg11_bn-57d0759e.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/vgg11_bn/default.log)
| vgg13_bn | 74.63 | 91.09 | 9.99 | 228.84 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg13_bn-5ebe5778.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/vgg13_bn/default.log)
| vgg16_bn | 74.00 | 90.56 | 15.30 | 313.77 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg16_bn-7d8c4031.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/vgg16_bn/default.log)
| vgg19_bn | 73.87 | 90.13 | 20.61 | 398.71 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg19_bn-b98f7bd7.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/vgg19_bn/default.log)
| mobilenetv2_x0_5 | 70.88 | 91.72 | 0.82 | 28.08 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x0_5-9f915757.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/mobilenetv2_x0_5/default.log)
| mobilenetv2_x0_75 | 73.61 | 92.61 | 1.48 | 59.43 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x0_75-d7891e60.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/mobilenetv2_x0_75/default.log)
| mobilenetv2_x1_0 | 74.20 | 92.82 | 2.35 | 88.09 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x1_0-1311f9ff.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/mobilenetv2_x1_0/default.log)
| mobilenetv2_x1_4 | 75.98 | 93.44 | 4.50 | 170.23 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x1_4-8a269f5e.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/mobilenetv2_x1_4/default.log)
| shufflenetv2_x0_5 | 67.82 | 89.93 | 0.44 | 10.99 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar100_shufflenetv2_x0_5-1977720f.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/shufflenetv2_x0_5/default.log)
| shufflenetv2_x1_0 | 72.39 | 91.46 | 1.36 | 45.09 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar100_shufflenetv2_x1_0-9ae22beb.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/shufflenetv2_x1_0/default.log)
| shufflenetv2_x1_5 | 73.91 | 92.13 | 2.58 | 94.35 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar100_shufflenetv2_x1_5-e2c85ad8.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/shufflenetv2_x1_5/default.log)
| shufflenetv2_x2_0 | 75.35 | 92.62 | 5.55 | 188.00 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/shufflenetv2/cifar100_shufflenetv2_x2_0-e7e584cd.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/shufflenetv2_x2_0/default.log)
| repvgg_a0 | 75.22 | 92.93 | 7.96 | 489.19 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar100_repvgg_a0-2df1edd0.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/repvgg_a0/default.log)
| repvgg_a1 | 76.12 | 92.71 | 12.94 | 851.44 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar100_repvgg_a1-c06b21a7.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/repvgg_a1/default.log)
| repvgg_a2 | 77.18 | 93.51 | 26.94 | 1850.22 | [model](https://github.com/chenyaofo/pytorch-cifar-models/releases/download/repvgg/cifar100_repvgg_a2-8e71b1f8.pt) \| [log](https://cdn.jsdelivr.net/gh/chenyaofo/pytorch-cifar-models@logs/logs/cifar100/repvgg_a2/default.log)

