<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/KnowledgeReview.git"><img align="center" src="./imgs/KnowledgeReview.png"></a></div>

<p align="center">
  «KnowledgeReview» re-implements the paper <a title="" href="https://arxiv.org/abs/2104.09044">Distilling Knowledge via Knowledge Review</a>
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

|     arch_s    |  top1  |  top5  |     arch_t    |  top1  |  top5  |  dataset | lambda |  top1  |  top5  |
|:-----------:|:------:|:------:|:-------------:|:------:|:------:|:--------:|:------:|:------:|:------:|
| MobileNetv2 | 80.620 | 95.820 |    ResNet50   | 83.540 | 96.820 | CIFAR100 |  7.0  | 83.370 | 96.810 |
| MobileNetv2 | 80.620 | 95.820 |    ResNet152   | 85.490 | 97.590 | CIFAR100 |  8.0  | 84.530 | 97.470 |
| MobileNetv2 | 80.620 | 95.820 |    ResNeXt_32x8d   | 85.720 | 97.650 | CIFAR100 |  6.0  | 84.520 | 97.470 |
|   ResNet18  | 80.540 | 96.040 |    ResNet50   | 83.540 | 96.820 | CIFAR100 |   10.0  | 83.130 | 96.350 |
|   ResNet50  | 83.540 | 96.820 |   ResNet152   | 85.490 | 97.590 | CIFAR100 |   6.0  | 86.240 | 97.610 |
|   ResNet50  | 83.540 | 96.820 | ResNeXt_32x8d | 85.720 | 97.650 | CIFAR100 |   6.0  | 86.220 | 97.490 |

more see [docs](./docs/README.md)

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

Unlike overhaul and other knowledge transfer algorithms，knowledge review use cross stage's teacher features to train student features. Meanwhile, it designed a new residual learning framework to simplify student transfer operation, and use ABF (attention based fusion) and HCL (hierarchical context loss) to improve feature distillation.

Current project implementation is based on [ ZJCV/overhaul ](https://github.com/ZJCV/overhaul) and [ dvlab-research/ReviewKD](https://github.com/dvlab-research/ReviewKD).

## Installation

```
$ pip install -r requirements.txt
```

## Usage

* Train

```angular2html
$ CUDA_VISIBLE_DEVICES=0 python tools/train.py -cfg=configs/resnet/ofd_2_0_r50_pret_r18_c100_224_e100_sgd_mslr.yaml
```

* Test

```angular2html
$ CUDA_VISIBLE_DEVICES=0 python tools/test.py -cfg=configs/resnet/ofd_2_0_r50_pret_r18_c100_224_e100_sgd_mslr.yaml
```

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

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

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/ZJCV/KnowledgeReview/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2021 zjykzj