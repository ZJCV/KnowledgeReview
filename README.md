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

...

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