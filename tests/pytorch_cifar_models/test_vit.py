import pytest

import torch
import pytorch_cifar_models.vit as vit


@pytest.mark.parametrize("dataset", ["cifar10", "cifar100"])
@pytest.mark.parametrize("model_name", ["vit_b16", "vit_b32", "vit_l16", "vit_l32", "vit_h14"])
def test_resnet(dataset, model_name):
    num_classes = 10 if dataset == "cifar10" else 100
    model = getattr(vit, f"{dataset}_{model_name}")()
    x = torch.empty((1, 3, 224, 224))
    y = model(x)
    assert y.shape == (1, num_classes)
