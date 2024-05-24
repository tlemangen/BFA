import os

import torchvision.transforms as T

PROJECT_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

IMAGENET_NUM_CLASSES = 1000
CALTECH256_NUM_CLASSES = 256
NIPS2017_NUM_CLASSES = 1000
IMAGENETTE_NUM_CLASSES = 10

caltech256_transform = T.Compose([
    T.ToTensor(),
    T.Resize(340),
    T.CenterCrop(299),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

nips2017_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

imagenette_transform = T.Compose([
    T.ToTensor(),
    T.Resize(int(299 * 1.1)),
    T.CenterCrop(299),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

configs = {
    "datasets": {
        "caltech256": {
            "train": {
                "root": "D:/datasets/caltech256/train",
                "transform": caltech256_transform
            },
            "test": {
                "root": "D:/datasets/caltech256/test",
                "transform": caltech256_transform
            },
        },
        "nips2017": {
            "train": {
                "root": None,
                "transform": None
            },
            "test": {
                "root": "D:/datasets/NIPS2017_adversarial_competition/dev_dataset/my_images",
                "transform": nips2017_transform
            },
        },
        "imagenette": {
            "train": {
                "root": "D:/datasets/imagenette2/train",
                "transform": imagenette_transform
            },
            "test": {
                "root": "D:/datasets/imagenette2/val",
                "transform": imagenette_transform
            },
        }
    },
    "models": {

    },
    "attacks": {
        "bfa": {
            "eps": 16 / 255,
            "steps": 10,
            "decay": 1.0,
            "eta": 28,
            "ensemble_number": 30,
            "layer_name": {
                "pytorch": {
                    "inception_v3": "model.Mixed_5b",
                    "resnet_152": "model.layer2.7",
                    "vgg_16": "model.features.15",
                },
                "timm": {
                    "inception_v3": "model.Mixed_5b",
                    "inception_v4": "model.features.9",  # without ablation study
                    "inception_resnet_v2": "model.conv2d_4a",
                    "resnet_152": "model.layer2.7",
                    "vgg_16": "model.features.15",
                }
            },
        },
    }
}
