from typing import Optional, Callable, Tuple, Any

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import VisionDataset
import torchvision.transforms as T


class ImageNette(VisionDataset):

    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        super().__init__(root)
        self.dataset = datasets.ImageFolder(root=root, transform=transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def __call__(self, *args, **kwargs):
        return self.dataset


if __name__ == '__main__':
    # transforms
    test_transforms = T.Compose([
        T.ToTensor(),
        T.Resize(340),
        T.CenterCrop(299),
    ])
    imagenette = ImageNette(f'D:/datasets/imagenette2/val', test_transforms)
    test_data_loader = DataLoader(imagenette, batch_size=64, shuffle=True)
    for idx, (images, labels) in enumerate(test_data_loader):
        print(idx, images.shape)
