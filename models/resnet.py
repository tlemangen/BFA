from typing import Tuple, Optional

import timm
import torch
import torchvision
from torch import nn, Tensor

from configs import *


class ResNet(nn.Module):

    def __init__(self, package: str, version: str, dataset: str, pretrained: bool, map_location: Optional[str]) -> None:
        super(ResNet, self).__init__()

        if dataset == 'caltech256':
            NUM_CLASSES = CALTECH256_NUM_CLASSES
        elif dataset == 'nips2017':
            NUM_CLASSES = NIPS2017_NUM_CLASSES
        elif dataset == 'imagenette':
            NUM_CLASSES = IMAGENETTE_NUM_CLASSES
        else:
            raise NotImplementedError(f"The parameter dataset={dataset} is not implemented.")

        if package == 'pytorch':
            if version == '50':
                self.model = torchvision.models.resnet50(pretrained=True)
            elif version == '152':
                self.model = torchvision.models.resnet152(pretrained=True)
            else:
                raise NotImplementedError(f"The parameter version={version} for package={package} is not implemented.")
            if NUM_CLASSES != IMAGENET_NUM_CLASSES:
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, NUM_CLASSES)
        elif package == 'timm':
            if version in ['50', '152']:
                if dataset in ['nips2017']:
                    self.model = timm.create_model(f'resnet{version}', pretrained=True)
                else:
                    raise NotImplementedError(f'version="{version}" and package="{package}" for dataset="{dataset}" '
                                              f'will be joined soon. Please switch to other values.')
            else:
                raise NotImplementedError(f"The parameter version={version} for package={package} is not implemented.")
        else:
            raise NotImplementedError(f"The parameter package={package} is not implemented.")

        if pretrained:
            if dataset in ['caltech256', 'imagenette']:
                self.model.load_state_dict(torch.load(os.path.join(
                    PROJECT_ROOT_DIR,
                    f'resources/state_dict/{package}_resnet_{version}_{dataset}.pth'),
                    map_location=torch.device(map_location)))
            elif dataset in ['nips2017']:
                pass
            else:
                raise NotImplementedError(f"The parameter dataset={dataset} is not implemented.")

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        return self.model(x)
