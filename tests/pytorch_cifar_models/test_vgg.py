import pytest

import torch
import pytorch_cifar_models.vgg as vgg


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
@pytest.mark.parametrize("model_name", ["vgg11", "vgg13", "vgg16", "vgg19"])
@pytest.mark.parametrize("bn", ["", "_bn"])
def test_resnet(dataset, model_name, bn):
    num_classes = 10 if dataset == "cifar10" else 100
    model = getattr(vgg, f"{dataset}_{model_name}{bn}")()
    x = torch.empty((1, 3, 32, 32))
    y = model(x)
    assert y.shape == (1, num_classes)
