import torch
import torchvision.transforms as T
from torch import Tensor


def imagenet_normalize(x: Tensor) -> Tensor:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(x)


def imagenet_denormalize(x: Tensor) -> Tensor:
    denormalize = T.Normalize(mean=[-2.1179, -2.0357, -1.8044], std=[4.3668, 4.4643, 4.4444])
    return denormalize(x)


def caltech256_normalize(x: Tensor) -> Tensor:
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return normalize(x)


def caltech256_denormalize(x):
    denormalize = T.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])
    return denormalize(x)


def imagenet_eps_normalize(eps: float, device: str) -> Tensor:
    normalize = T.Normalize(mean=[0., 0., 0.], std=[0.229, 0.224, 0.225])
    eps = normalize(torch.tensor(eps, device=device).expand((1, 3, 1, 1)))
    return eps


def caltech256_eps_normalize(eps: float, device: str) -> Tensor:
    normalize = T.Normalize(mean=[0., 0., 0.], std=[0.5, 0.5, 0.5])
    eps = normalize(torch.tensor(eps, device=device).expand((1, 3, 1, 1)))
    return eps
