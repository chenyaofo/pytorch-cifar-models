# Pytorch CIFAR Models

## Introduction

The goal of this project is to provide some neural network examples and a simple training codebase for begginners.

## Get Started with Google Colab (TODO)

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

|  Model   |  Top-1 Acc.(%) | Top-5 Acc.(%) | #Params.(M) | MAdds(M) |                    |
|----------|----------------|---------------|-------------|----------|--------------------|
| Resnet20 | 91.65          | 99.68         | 0.27        | 40.81    | log \| tensorboard |
| Resnet34 | 92.81          | 99.72         | 0.46        | 69.12    | log \| tensorboard |
| Resnet44 | 93.24          | 99.75         | 0.66        | 97.44    | log \| tensorboard |
| Resnet56 | 93.69          | 99.68         | 0.85        | 125.75   | log \| tensorboard |

### CIFAR-100

|  Model   |  Top-1 Acc.(%) | Top-5 Acc.(%) | #Params.(M) | MAdds(M) |                    |
|----------|----------------|---------------|-------------|----------|--------------------|
| Resnet20 | 66.61          | 89.95         | 0.27        | 40.81    | log \| tensorboard |
| Resnet34 | 68.74          | 90.23         | 0.46        | 69.12    | log \| tensorboard |
| Resnet44 | 69.49          | 90.39         | 0.66        | 97.44    | log \| tensorboard |
| Resnet56 | 70.79          | 91.10         | 0.85        | 125.75   | log \| tensorboard |



