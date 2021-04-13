import pytest

import torch
import pytorch_cifar_models.mobilenetv2 as mobilenetv2


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
@pytest.mark.parametrize("model_name", ["mobilenetv2_x0_5", "mobilenetv2_x0_75", "mobilenetv2_x1_0", "mobilenetv2_x1_4"])
def test_resnet(dataset, model_name):
    num_classes = 10 if dataset == "cifar10" else 100
    model = getattr(mobilenetv2, f"{dataset}_{model_name}")()
    x = torch.empty((1, 3, 32, 32))
    y = model(x)
    assert y.shape == (1, num_classes)