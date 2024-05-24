from torch import nn

from .inception import Inception
from .inception_resnet import InceptionResNet
from .resnet import ResNet
from .vgg import VGG
from .densenet import DenseNet


def models_factory(name: str, package: str, dataset: str, pretrained: bool, map_location: str) -> nn.Module:
    model_name_list = name.split('_')
    model_name = "_".join(model_name_list[:-1])
    model_version = model_name_list[-1]

    if model_name == 'inception':
        model = Inception(package, model_version, dataset, pretrained, map_location)
    elif model_name == 'inception_resnet':
        model = InceptionResNet(package, model_version, dataset, pretrained, map_location)
    elif model_name == 'resnet':
        model = ResNet(package, model_version, dataset, pretrained, map_location)
    elif model_name == 'vgg':
        model = VGG(package, model_version, dataset, pretrained, map_location)
    elif model_name == 'densenet':
        model = DenseNet(package, model_version, dataset, pretrained, map_location)
    else:
        raise NotImplementedError(f"The parameter model_name={model_name} is not implemented.")

    return model
