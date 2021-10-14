<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.md">🇺🇸</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/KnowledgeReview.git"><img align="center" src="./imgs/KnowledgeReview.png"></a></div>

<p align="center">
  «KnowledgeReview»复现了论文<a title="" href="https://arxiv.org/abs/2104.09044">Distilling Knowledge via Knowledge Review</a>
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

* 解析：[Distilling Knowledge via Knowledge Review](https://blog.zhujian.life/posts/8da15989.html)

|     arch_s    |  top1  |  top5  |     arch_t    |  top1  |  top5  |  dataset | lambda |  top1  |  top5  |
|:-----------:|:------:|:------:|:-------------:|:------:|:------:|:--------:|:------:|:------:|:------:|
| MobileNetv2 | 80.620 | 95.820 |    ResNet50   | 83.540 | 96.820 | CIFAR100 |  7.0  | 83.370 | 96.810 |
| MobileNetv2 | 80.620 | 95.820 |    ResNet152   | 85.490 | 97.590 | CIFAR100 |  8.0  | 84.530 | 97.470 |
| MobileNetv2 | 80.620 | 95.820 |    ResNeXt_32x8d   | 85.720 | 97.650 | CIFAR100 |  6.0  | 84.520 | 97.470 |
|   ResNet18  | 80.540 | 96.040 |    ResNet50   | 83.540 | 96.820 | CIFAR100 |   10.0  | 83.130 | 96.350 |
|   ResNet50  | 83.540 | 96.820 |   ResNet152   | 85.490 | 97.590 | CIFAR100 |   6.0  | 86.240 | 97.610 |
|   ResNet50  | 83.540 | 96.820 | ResNeXt_32x8d | 85.720 | 97.650 | CIFAR100 |   6.0  | 86.220 | 97.490 |

更多内容参见[docs](./docs/README.md)

## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [安装](#安装)
- [用法](#用法)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

和之前的知识迁移算法不同，`RFD`使用了跨阶段的教师特征来训练学生特征。同时，它还设计了一个新的残差学习框架用于简化学生特征转换操作，以及设计了`ABF`（基于融合注意力）模块和`HCL`（分层内容损失）函数来辅助特征蒸馏训练。

当前实现基于[ ZJCV/overhaul ](https://github.com/ZJCV/overhaul)和[ dvlab-research/ReviewKD](https://github.com/dvlab-research/ReviewKD)。

## 安装

```
$ pip install -r requirements.txt
```

## 用法

* 训练

```angular2html
$ CUDA_VISIBLE_DEVICES=0 python tools/train.py -cfg=configs/rfd/resnet/rfd_6_0_r152_pret_r50_c100_224_e100_sgd_mslr.yaml
```

* 测试

```angular2html
$ CUDA_VISIBLE_DEVICES=0 python tools/test.py -cfg=configs/rfd/resnet/rfd_6_0_r152_pret_r50_c100_224_e100_sgd_mslr.yaml
```

## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 致谢

```
@misc{chen2021distilling,
      title={Distilling Knowledge via Knowledge Review}, 
      author={Pengguang Chen and Shu Liu and Hengshuang Zhao and Jiaya Jia},
      year={2021},
      eprint={2104.09044},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/ZJCV/KnowledgeReview/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2021 zjykzj