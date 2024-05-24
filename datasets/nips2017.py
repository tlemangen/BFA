import os
from typing import Optional, Callable, Tuple, Any

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class NIPS2017(Dataset):

    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        self.root = root
        self.targets = torch.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'nips2017_targets.pt'))
        self.transform = transform

    def __getitem__(self, item: int) -> Tuple[Any, Any]:
        filename = str(item) + '.png'
        target = self.targets[item].type(torch.LongTensor)  # [0, 999]
        filepath = os.path.join(self.root, filename)
        img = Image.open(filepath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.targets.shape[0]


if __name__ == '__main__':
    # transforms
    test_transforms = T.Compose([
        T.ToTensor(),
    ])
    nips_2017 = NIPS2017(f'D:/datasets/NIPS2017_adversarial_competition/dev_dataset/my_images', test_transforms)
    test_data_loader = DataLoader(nips_2017, batch_size=64, shuffle=True)
    for idx, (images, labels) in enumerate(test_data_loader):
        print(idx, images.shape)
