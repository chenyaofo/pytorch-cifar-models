import pytest

import torch
import pytorch_cifar_models.repvgg as repvgg


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
@pytest.mark.parametrize("model_name", ["repvgg_a0", "repvgg_a1", "repvgg_a2"])
def test_resnet(dataset, model_name):
    num_classes = 10 if dataset == "cifar10" else 100
    model = getattr(repvgg, f"{dataset}_{model_name}")()
    x = torch.empty((1, 3, 32, 32))
    y = model(x)
    assert y.shape == (1, num_classes)
