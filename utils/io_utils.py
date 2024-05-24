import csv
import os
from typing import Callable, Optional, Dict

import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch import Tensor


def imshow(images: Tensor, de_norm: Optional[Callable] = None, nrow: Optional[int] = None) -> None:
    images = images.clone().detach()
    if de_norm:
        images = de_norm(images)
    if nrow:
        img = torchvision.utils.make_grid(images, nrow=nrow)
    else:
        img = torchvision.utils.make_grid(images)
    img_np = img.detach().cpu().numpy()
    plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.axis('off')
    plt.show()


def write_csv(csv_pathfile: str, **content) -> None:
    with open(csv_pathfile, 'a+', newline='') as f:
        writer = csv.writer(f)
        for key, value in content.items():
            print(key, value)
            value.insert(0, key)
            writer.writerow(value)
