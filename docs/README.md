
# README

## Benchmark Model


|       arch       |  dataset |  top1  |  top5  |
|:----------------:|:--------:|:------:|:------:|
|    MobileNetv2   | CIFAR100 | 80.620 | 95.820 |
|     ResNet18     | CIFAR100 | 80.540 | 96.040 |
|     ResNet50     | CIFAR100 | 83.540 | 96.820 |
|     ResNet152    | CIFAR100 | 85.490 | 97.590 |
| ResNeXt101_32x8d | CIFAR100 | 85.720 | 97.650 |

## Distillation

|    arch_s   |     arch_t    |  dataset | lambda |  top1  |  top5  |
|:-----------:|:-------------:|:--------:|:------:|:------:|:------:|
| MobileNetV2 |    ResNet50   | CIFAR100 |   7.0  | 83.370 | 96.810 |
| MobileNetV2 |    ResNet152   | CIFAR100 |   8.0  | 84.530 | 97.470 |
| MobileNetV2 |    ResNext_32x8d   | CIFAR100 |  6.0  | 84.520 | 97.470 |
|   ResNet18  |    ResNet50   | CIFAR100 |  10.0  | 83.130 | 96.350 |
|   ResNet50  |   ResNet152   | CIFAR100 |   6.0  | 86.240 | 97.610 |
|   ResNet50  | ResNeXt_32x8d | CIFAR100 |   6.0  | 86.220 | 97.490 |

## See

* [mobilenet](mobilenet.md)

* [resnet](resnet.md)