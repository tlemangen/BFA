import torch
from torch import nn

from .attack import Attack

from .bfa import BFA

from configs import *


def attackers_factory(class_name: str, dataset: str, surrogate_model_name: str, surrogate_model: nn.Module,
                      package: str, device: str):
    args = []
    attacker_configs = configs['attacks'][class_name.lower()]
    for key in attacker_configs.keys():
        if key in ['layer_name', 'layer_names']:
            value = attacker_configs[key][package][surrogate_model_name]
        elif key in ['state_dict']:
            continue
        else:
            value = attacker_configs[key]
        if type(value) is str:
            value = f'"{value}"'
        args.append(f'{key}={value}')
    args = ", ".join(args)
    attacker_str = f'{class_name}(dataset, surrogate_model, {args}, device="{device}")'
    print(f'Run: {attacker_str}')
    attacker = eval(attacker_str)
    if attacker.attack in ['GAP', 'CDA', 'LTAP', 'BIA']:
        attacker.generator.load_state_dict(torch.load(attacker_configs['state_dict']))
        attacker.generator.eval()
    return attacker
