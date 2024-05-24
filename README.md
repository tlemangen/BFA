# BFA

This repository contains the PyTorch code for the paper:

[Improving the transferability of adversarial examples through black-box feature attacks](https://www.sciencedirect.com/science/article/pii/S0925231224006349?via%3Dihub) (Neurocomputing 2024)

Maoyuan Wang, Jinwei Wang, Bin Ma, Xiangyang Luo.

## Datasets

The size of images is set to 299x299.

Find the class folders in a dataset structured as follows::

### Caltech-256

```text
directory/
├── train
│   ├── 001.xxx
│   ├── 002.xxx
│   ├── ...
│   └── 256.xxx
│       └── yyy.jpg
└── test
    ├── 001.xxx
    ├── 002.xxx
    ├── ...
    └── 256.xxx
        └── yyy.jpg
```

normalization:

```python
import torchvision.transforms as T

caltech256_transform = T.Compose([
    T.ToTensor(),
    T.Resize(340),
    T.CenterCrop(299),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
```

### NIPS2017

```text
directory/
├── 0.png
├── 1.png
├── ...
└── 999.jpg
```

normalization:

```python
import torchvision.transforms as T

nips2017_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### ImageNette

```text
directory/
├── train
│   ├── nxxx
│   ├── nxxx
│   ├── ...
│   └── nxxx
│       └── yyy.JPEG
└── val
    ├── nxxx
    ├── nxxx
    ├── ...
    └── nxxx
        └── yyy.JPEG
```

normalization:

```python
import torchvision.transforms as T

imagenette_transform = T.Compose([
    T.ToTensor(),
    T.Resize(int(299 * 1.1)),
    T.CenterCrop(299),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Pre-trained Models

### Caltech-256

| Models              | Packages | Accuracy (%) | Weights                                                                                                     |
|---------------------|----------|--------------|-------------------------------------------------------------------------------------------------------------|
| Inception-v3        | Pytorch  | 85.7         | [pytorch_inception_v3_caltech256.pth](resources%2Fstate_dict%2Fpytorch_inception_v3_caltech256.pth)         |
| Inception-v4        | timm     | 77.8         | [timm_inception_v4_caltech256.pth](resources%2Fstate_dict%2Ftimm_inception_v4_caltech256.pth)               |
| Inception-ResNet-v2 | timm     | 87.9         | [timm_inception_resnet_v2_caltech256.pth](resources%2Fstate_dict%2Ftimm_inception_resnet_v2_caltech256.pth) |
| ResNet-50           | Pytorch  | 84.2         | [pytorch_resnet_50_caltech256.pth](resources%2Fstate_dict%2Fpytorch_resnet_50_caltech256.pth)               |
| ResNet-152          | Pytorch  | 83.4         | [pytorch_resnet_152_caltech256.pth](resources%2Fstate_dict%2Fpytorch_resnet_152_caltech256.pth)             |
| VGG-16              | Pytorch  | 78.1         | [pytorch_vgg_16_caltech256.pth](resources%2Fstate_dict%2Fpytorch_vgg_16_caltech256.pth)                     |
| VGG-19              | Pytorch  | 77.1         | [pytorch_vgg_19_caltech256.pth](resources%2Fstate_dict%2Fpytorch_vgg_19_caltech256.pth)                     |
| DenseNet-121        | Pytorch  | 84.6         | [pytorch_densenet_121_caltech256.pth](resources%2Fstate_dict%2Fpytorch_densenet_121_caltech256.pth)         |
| DenseNet-169        | Pytorch  | 86.3         | [pytorch_densenet_169_caltech256.pth](resources%2Fstate_dict%2Fpytorch_densenet_169_caltech256.pth)         |

### NIPS2017

| Models                                        | Packages | Accuracy (%) | Weights                                  |
|-----------------------------------------------|----------|--------------|------------------------------------------|
| Inception-v3 (Inc-v3-p)                       | Pytorch  | 95.3         | inception_v3_google-0cc3c7bd.pth         |
| Inception-v3 (Inc-v3-t)                       | timm     | 95.3         | inception_v3_google-1a9a5a14.pth         |
| Inception-v4 (Inc-v4-t)                       | timm     | 94.7         | inceptionv4-8e4777a0.pth                 |
| Inception-ResNet-v2 (IncRes-v2-t)             | timm     | 97.2         | inception_resnet_v2-940b1cd6.pth         |
| ResNet-50 (Res-50-p)                          | Pytorch  | 91.8         | resnet50-0676ba61.pth                    |
| ResNet-50 (Res-50-t)                          | timm     | 94.6         | resnet50_a1_0-14fe96d1.pth               |
| ResNet-152 (Res-152-p)                        | Pytorch  | 93.6         | resnet152-394f9c45.pth                   |
| ResNet-152 (Res-152-t)                        | timm     | 95.4         | resnet152_a1h-dc400468.pth               |
| VGG-16 (Vgg-16-p)                             | Pytorch  | 85.5         | vgg16-397923af.pth                       |
| VGG-16 (Vgg-16-t)                             | timm     | 84.2         | vgg16-397923af.pth                       |
| VGG-19 (Vgg-19-p)                             | Pytorch  | 87.5         | vgg19-dcbb9e9d.pth                       |
| VGG-19 (Vgg-19-t)                             | timm     | 85.2         | vgg19-dcbb9e9d.pth                       |
| DenseNet-121 (Den-121-p)                      | Pytorch  | 92.3         | densenet121-a639ec97.pth                 |
| DenseNet-121 (Den-121-t)                      | timm     | 93.2         | densenet121_ra-50efcf5c.pth              |
| DenseNet-169 (Den-169-p)                      | Pytorch  | 94.1         | densenet169-b2777c0a.pth                 |
| DenseNet-169 (Den-169-t)                      | timm     | 94.1         | densenet169-b2777c0a.pth                 |
| Adv-Inception-v3 (Adv-Inc-v3-t)               | timm     | 86.9         | adv_inception_v3-9e27bd63.pth            |
| Ens-Adv-Inception-ResNet-v2 (Ens-IncRes-v2-t) | timm     | 94.5         | ens_adv_inception_resnet_v2-2592a550.pth |

### ImageNette

| Models              | Packages | Accuracy (%) | Weights                                                                                                     |
|---------------------|----------|--------------|-------------------------------------------------------------------------------------------------------------|
| Inception-v3        | Pytorch  | 99.6         | [pytorch_inception_v3_imagenette.pth](resources%2Fstate_dict%2Fpytorch_inception_v3_imagenette.pth)         |
| Inception-v4        | timm     | 99.6         | [timm_inception_v4_imagenette.pth](resources%2Fstate_dict%2Ftimm_inception_v4_imagenette.pth)               |
| Inception-ResNet-v2 | timm     | 99.7         | [timm_inception_resnet_v2_imagenette.pth](resources%2Fstate_dict%2Ftimm_inception_resnet_v2_imagenette.pth) |
| ResNet-50           | Pytorch  | 99.4         | [pytorch_resnet_50_imagenette.pth](resources%2Fstate_dict%2Fpytorch_resnet_50_imagenette.pth)               |
| ResNet-152          | Pytorch  | 99.7         | [pytorch_resnet_152_imagenette.pth](resources%2Fstate_dict%2Fpytorch_resnet_152_imagenette.pth)             |
| VGG-16              | Pytorch  | 99.0         | [pytorch_vgg_16_imagenette.pth](resources%2Fstate_dict%2Fpytorch_vgg_16_imagenette.pth)                     |
| VGG-19              | Pytorch  | 98.7         | [pytorch_vgg_19_imagenette.pth](resources%2Fstate_dict%2Fpytorch_vgg_19_imagenette.pth)                     |
| DenseNet-121        | Pytorch  | 98.9         | [pytorch_densenet_121_imagenette.pth](resources%2Fstate_dict%2Fpytorch_densenet_121_imagenette.pth)         |
| DenseNet-169        | Pytorch  | 99.2         | [pytorch_densenet_169_imagenette.pth](resources%2Fstate_dict%2Fpytorch_densenet_169_imagenette.pth)         |

## BFA

> `python eval.py --ds=nips2017 --model=inception_v3 --pkg=pytorch --bs=32 --attack=BFA`

## Citation

If you find the idea or code useful for your research, please consider citing our paper:

```text
@article{wang2024improving,
  title={Improving the transferability of adversarial examples through black-box feature attacks},
  author={Wang, Maoyuan and Wang, Jinwei and Ma, Bin and Luo, Xiangyang},
  journal={Neurocomputing},
  pages={127863},
  year={2024},
  publisher={Elsevier}
}
```
