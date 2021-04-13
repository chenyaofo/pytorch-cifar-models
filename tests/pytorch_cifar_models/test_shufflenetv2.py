import pytest

import torch
import pytorch_cifar_models.shufflenetv2 as shufflenetv2


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
@pytest.mark.parametrize("model_name", ["shufflenetv2_x0_5", "shufflenetv2_x1_0", "shufflenetv2_x1_5", "shufflenetv2_x2_0"])
def test_resnet(dataset, model_name):
    num_classes = 10 if dataset == "cifar10" else 100
    model = getattr(shufflenetv2, f"{dataset}_{model_name}")()
    x = torch.empty((1, 3, 32, 32))
    y = model(x)
    assert y.shape == (1, num_classes)
