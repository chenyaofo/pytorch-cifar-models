import pytest

import torch
import pytorch_cifar_models.resnet as resnet


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
@pytest.mark.parametrize("model_name", ["resnet20", "resnet32", "resnet44", "resnet56"])
def test_resnet(dataset, model_name):
    num_classes = 10 if dataset == "cifar10" else 100
    model = getattr(resnet, f"{dataset}_{model_name}")()
    x = torch.empty((1, 3, 32, 32))
    y = model(x)
    assert y.shape == (1, num_classes)
