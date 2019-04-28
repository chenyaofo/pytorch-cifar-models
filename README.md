# CIFAR-pretrained-models

## Accuracy in the Validation Set

The validation is performed with the original view of the image(size=32x32).

**Note**: the FLOPs only counts the conv and linear layer.

| Model    | Acc@1 | Acc@5 | #param. | FLOPs |
|----------|-------|-------|---------|-------|
| Resnet20 |       |       |         |       |
| Resnet32 |       |       |         |       |
| Resnet56 |       |       |         |       |

## Pretrained Models

All the pretrained models are avaliable in the [release](https://github.com/chenyaofo/CIFAR-pretrained-models/releases).

## Implementation Details

The models are trained and exported with Pytorch(1.0.1.post2) and torchvision(0.2.1).

The training data augumentation follow [1],
```
torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.49139968, 0.48215827, 0.44653124],
        std=[0.24703233, 0.24348505, 0.26158768],
    ),
])
```

All the models are trained with a mini batch size of 256 and the following optimizer,
```
torch.optim.SGD(..., lr=0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
```
the following scheduler,
```
torch.optim.lr_scheduler.MultiStepLR(..., milestones=[100,150], gamma=0.1)
```
the total training epochs is 200.


## Reference

[1]. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

## Acknowledgement

Thanks for the computer vision community and github open source community.
