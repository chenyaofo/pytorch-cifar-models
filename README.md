# CIFAR-pretrained-models

## Accuracy in the Validation Set

The validation is performed with the original view of the image(size=32x32).

**Note**: the FLOPs and the number of parameters only counts the conv and linear layer.

| Model         | Acc@1(C10) | Acc@5(C10) | Acc@1(C100) | Acc@5(C100) | #param. | FLOPs |
|---------------|------------|------------|-------------|-------------|---------|-------|
| Resnet20[[1]] |    91.65        |    99.68        |     66.61        |    89.95         |   0.27M      |   40.81M    |
| Resnet32[[1]] |    92.81        |     99.72       |     68.74        |    90.23         |   0.46M      |  69.12M     |
| Resnet44[[1]] |    93.24        |      99.75      |     69.49        |    90.39         |   0.66M      |   97.44M    |
| Resnet56[[1]] |     93.69       |    99.68        |     70.79        |    91.10         |   0.85M      |   125.75M    |


## Pretrained Models

All the pretrained models are avaliable in the [release](https://github.com/chenyaofo/CIFAR-pretrained-models/releases).

## Implementation Details

The models are trained and exported with [PyTorch(1.1.0)](https://github.com/pytorch/pytorch/releases/tag/v1.1.0)
and 
[torchvision(0.2.2)](https://github.com/pytorch/vision/releases/tag/v0.2.2).

The training data augumentation follow [[1]],
```
torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], # mean=[0.5071, 0.4865, 0.4409] for cifar100
        std=[0.2023, 0.1994, 0.2010], # std=[0.2009, 0.1984, 0.2023] for cifar100
    ),
])
```

All the models are trained with a mini batch size of 256 and the following optimizer,
```
torch.optim.SGD(lr=0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
```
the following scheduler,
```
ctx.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=200,eta_min=0.001)
```
the total training epochs is 200.


## Reference

1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep residual learning for image recognition. In *CVPR*, 2016.

[1]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

2. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Identity Mappings in Deep Residual Networks. In *ECCV*, 2016.

[2]: https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38

## Acknowledgement

Thanks for the computer vision community and github open source community.
