# MagcDrive

[![arXiv](https://img.shields.io/badge/arXiv-2310.02601-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2310.02601) [![arXiv](https://img.shields.io/badge/Web-MagicDrive-blue.svg?style=plastic)](https://gaoruiyuan.com/magicdrive/)


This repository contains the implementation of the paper 

> MagicDrive: Street View Generation with Diverse 3D Geometry Control <br>
> [Ruiyuan Gao](https://gaoruiyuan.com/)<sup>1\*</sup>, [Kai Chen](https://kaichen1998.github.io/)<sup>2\*</sup>, [Enze Xie](https://xieenze.github.io/)<sup>3^</sup>, [Lanqing Hong](https://scholar.google.com.sg/citations?user=2p7x6OUAAAAJ&hl=en)<sup>3</sup>, [Zhenguo Li](https://scholar.google.com/citations?user=XboZC1AAAAAJ&hl=en)<sup>3</sup>, [Dit-Yan Yeung](https://sites.google.com/view/dyyeung)<sup>2</sup>, [Qiang Xu](https://cure-lab.github.io/)<sup>1^</sup><br>
> <sup>1</sup>CUHK <sup>2</sup>HKUST <sup>3</sup>Huawei Noah's Ark Lab <br>
> <sup>\*</sup>Equal Contribution <sup>^</sup>Corresponding Authors

Recent advancements in diffusion models have significantly enhanced the data synthesis with 2D control. Yet, precise 3D control in street view generation, crucial for 3D perception tasks, remains elusive. Specifically, utilizing Bird’s-Eye View (BEV) as the primary condition often leads to challenges in geometry control (e.g., height), affecting the representation of object shapes, occlusion patterns, and road surface elevations, all of which are essential to perception data synthesis, especially for 3D object detection tasks. In this paper, we introduce MAGICDRIVE, a novel street view generation framework offering diverse 3D geometry controls, including camera poses, road maps, and 3D bounding boxes, together with textual descriptions, achieved through tailored encoding strategies. Besides, our design incorporates a cross-view attention module, ensuring consistency across multiple camera views. With MAGICDRIVE, we achieve high-fidelity street-view synthesis that captures nuanced 3D geometry and various scene descriptions, enhancing tasks like BEV segmentation and 3D object detection.

## Method

In MagicDrive, we employ two strategies (cross-attention and additive encoder branch) to inject text prompt, camera pose, object boxes, and road maps as conditions for generation. We also propose cross-view attention module for multiview consistency.

![image-20231011165634648](./assets/overview.png)

## Street-view Generation with MagicDrive

Coming soon.

## Train MagicDrive

Coming soon.

## Quantitative Results

Compare MagicDrive with other methods for generation quality:

![main_results](./assets/main_results.png)

Training support with images generated from MagicDrive:

![trainability](./assets/trainability.png)

More results can be found in the main paper.

## Qualitative Results

More results can be found in the main paper.

![editings](./assets/editings.png)

## Cite Us

```bibtex
@article{gao2023magicdrive,
  title={MagicDrive: Street View Generation with Diverse 3D Geometry Control},
  author={Gao, Ruiyuan and Chen, Kai and Xie, Enze and Hong, Lanqing and Li, Zhenguo and Yeung, Dit-Yan and Xu, Qiang},
  journal={arXiv preprint arXiv:2310.02601},
  year={2023}
}
```

