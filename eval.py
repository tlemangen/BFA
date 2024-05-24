import argparse
import math
import pdb
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import attacks
from datasets import datasets_factory

from configs import *
from models import models_factory
from utils import imagenet_denormalize, caltech256_denormalize, imshow

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1234, type=int, help='random seed')
parser.add_argument('--device', default=0, type=int, help='GPU device Id')
parser.add_argument('--ds', default='caltech256', type=str, choices=["caltech256", "nips2017", "imagenette"],
                    help='dataset name')
parser.add_argument('--model', default='inception_v3', type=str,
                    choices=["inception_v3", "inception_v4", "inception_resnet_v2", "resnet_50", "resnet_152", 'vgg_16',
                             'vgg_19', 'densenet_121', 'densenet_169', "inception_v3ADV", "inception_resnet_v2ENS"],
                    help='surrogate model')
parser.add_argument('--pkg', default='pytorch', type=str, choices=["pytorch", "timm"], help='surrogate model package')
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument('--max_num', default=1000, type=int, help='max number of validated images')
parser.add_argument('--attack', default='BFA', type=str, help="attacker's class name")
args = parser.parse_args()

device = f'cuda:{args.device}'
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.ds in ["nips2017", "imagenette"]:
    models_pool = {
        'inception_v3': models_factory('inception_v3', 'timm', 'nips2017', True, device).eval().to(device),
        'inception_v4': models_factory('inception_v4', 'timm', 'nips2017', True, device).eval().to(device),
        'inception_resnet_v2': models_factory('inception_resnet_v2', 'timm', 'nips2017', True, device).eval().to(
            device),
        'resnet_50': models_factory('resnet_50', 'timm', 'nips2017', True, device).eval().to(device),
        'resnet_152': models_factory('resnet_152', 'timm', 'nips2017', True, device).eval().to(device),
        'vgg_16': models_factory('vgg_16', 'timm', 'nips2017', True, device).eval().to(device),
        'vgg_19': models_factory('vgg_19', 'timm', 'nips2017', True, device).eval().to(device),
        'densenet_121': models_factory('densenet_121', 'timm', 'nips2017', True, device).eval().to(device),
        'densenet_169': models_factory('densenet_169', 'timm', 'nips2017', True, device).eval().to(device),
        'inception_v3ADV': models_factory('inception_v3ADV', 'timm', 'nips2017', True, device).eval().to(device),
        'inception_resnet_v2ENS': models_factory('inception_resnet_v2ENS', 'timm', 'nips2017', True, device).eval().to(
            device),
    }
elif args.ds in ["caltech256"]:
    models_pool = {
        'inception_v3': models_factory('inception_v3', 'pytorch', 'caltech256', True, device).eval().to(device),
        'inception_v4': models_factory('inception_v4', 'timm', 'caltech256', True, device).eval().to(device),
        'inception_resnet_v2': models_factory('inception_resnet_v2', 'timm', 'caltech256', True, device).eval().to(
            device),
        'resnet_50': models_factory('resnet_50', 'pytorch', 'caltech256', True, device).eval().to(device),
        'resnet_152': models_factory('resnet_152', 'pytorch', 'caltech256', True, device).eval().to(device),
        'vgg_16': models_factory('vgg_16', 'pytorch', 'caltech256', True, device).eval().to(device),
        'vgg_19': models_factory('vgg_19', 'pytorch', 'caltech256', True, device).eval().to(device),
        'densenet_121': models_factory('densenet_121', 'pytorch', 'caltech256', True, device).eval().to(device),
        'densenet_169': models_factory('densenet_169', 'pytorch', 'caltech256', True, device).eval().to(device),
    }
else:
    raise NotImplementedError(f"{args.ds} is not implemented.")
surrogate_model = models_pool[args.model]


# for name, module in surrogate_model.named_modules():
#     print(name)
#     # print(name, "---", module)
# pdb.set_trace()


@torch.no_grad()
def main():
    images_root = configs['datasets'][args.ds]['test']['root']
    transform = configs['datasets'][args.ds]['test']['transform']
    dataset = datasets_factory(name=args.ds, root=images_root, transform=transform)
    if len(dataset) <= math.ceil(args.max_num / args.bs) * args.bs:
        data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=False)
        max_num = len(dataset)
        max_iter = len(data_loader)
    else:
        data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)
        max_num = math.ceil(args.max_num / args.bs) * args.bs
        max_iter = math.ceil(args.max_num / args.bs)

    attacker = attacks.attackers_factory(args.attack, args.ds, args.model, surrogate_model, args.pkg, device)

    acc = np.zeros((len(models_pool),))
    asr1 = np.zeros((len(models_pool),))
    asr2 = np.zeros((len(models_pool),))
    for idx, (images, labels) in enumerate(data_loader):
        if idx >= max_iter:
            break
        print(f'Batch: [{idx + 1}/{max_iter}]')
        images = images.to(device)
        labels = labels.to(device)

        if attacker.attack in ['GAP', 'CDA', 'LTAP', 'BIA']:
            adv = attacker(images)
        else:
            with torch.enable_grad():
                adv = attacker(images, labels)

        # print(torch.min(imagenet_denormalize(adv) - imagenet_denormalize(images)),
        #       torch.max(imagenet_denormalize(adv) - imagenet_denormalize(images)))
        # imshow(adv, imagenet_denormalize)
        # print(torch.min(caltech256_denormalize(adv) - caltech256_denormalize(images)),
        #       torch.max(caltech256_denormalize(adv) - caltech256_denormalize(images)))
        # imshow(adv, caltech256_denormalize)

        for tm_idx, target_model in enumerate(models_pool.values()):
            logits = target_model(images)
            logits_adv = target_model(adv)
            predict_labels = logits.argmax(dim=1)
            predict_adv_labels = logits_adv.argmax(dim=1)

            acc[tm_idx] += np.sum(np.array(predict_labels.detach().cpu() == labels.detach().cpu()).astype(int))
            asr1[tm_idx] += np.sum(np.array(predict_adv_labels.detach().cpu() != labels.detach().cpu()).astype(int))
            asr2[tm_idx] += np.sum((np.array(predict_labels.detach().cpu() == labels.detach().cpu()) & (
                np.array(predict_adv_labels.detach().cpu() != labels.detach().cpu()))).astype(int))

    asr1 /= max_num
    asr2 /= acc
    asr1 = np.around(asr1 * 100., 1)
    asr2 = np.around(asr2 * 100., 1)

    print(f'Dataset: {args.ds}, surrogate model: {args.model}')
    print("| Models | " + " | ".join(list(models_pool.keys())) + " |")
    print("| ASR1   | " + " | ".join(list(map(str, asr1))) + " |")
    print("| ASR2   | " + " | ".join(list(map(str, asr2))) + " |")


if __name__ == '__main__':
    main()  # python eval.py --ds=caltech256 --model=inception_v3 --pkg=pytorch --bs=32
