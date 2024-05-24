"""

"""
import torch
from torch import nn, Tensor

from attacks import Attack

from utils import imagenet_normalize, imagenet_eps_normalize, caltech256_eps_normalize, caltech256_normalize, \
    caltech256_denormalize, imagenet_denormalize
import torch.nn.functional as F

from configs import *


class BFA(Attack):

    def __init__(self, dataset: str, model: nn.Module, eps: float, steps: int, decay: float, eta: float,
                 ensemble_number: int, layer_name: str, device: str) -> None:
        super().__init__("BFA", model, device)
        if dataset in ['caltech256']:
            self.eps_normalize = caltech256_eps_normalize
            self.normalize = caltech256_normalize
            self.denormalize = caltech256_denormalize
            self.num_classes = CALTECH256_NUM_CLASSES
        elif dataset in ['nips2017', 'imagenette']:
            self.eps_normalize = imagenet_eps_normalize
            self.normalize = imagenet_normalize
            self.denormalize = imagenet_denormalize
            if dataset == 'nips2017':
                self.num_classes = NIPS2017_NUM_CLASSES
            elif dataset == 'imagenette':
                self.num_classes = IMAGENETTE_NUM_CLASSES
            else:
                raise NotImplementedError(f"The parameter dataset={dataset} is not implemented.")
        else:
            raise NotImplementedError(f"The parameter dataset={dataset} is not implemented.")
        self.eps = self.eps_normalize(eps, device)
        self.steps = steps
        self.alpha = self.eps / self.steps
        self.decay = decay
        self.eta = eta
        self.ensemble_number = ensemble_number
        self.layer_name = layer_name
        self.feature_maps = None
        self.register_hook()
        self.loss_fn_ce = nn.CrossEntropyLoss()

    def hook(self, module: nn.Module, input: Tensor, output: Tensor) -> None:
        self.feature_maps = output
        return None

    def register_hook(self) -> None:
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook=self.hook)

    def get_maskgrad(self, images: Tensor, labels: Tensor) -> Tensor:
        images = images.clone().detach()
        images.requires_grad = True
        logits = self.model(images)
        loss = self.loss_fn_ce(logits, labels)
        maskgrad = torch.autograd.grad(loss, images)[0]
        maskgrad /= torch.sqrt(torch.sum(torch.square(maskgrad), dim=(1, 2, 3), keepdim=True))
        return maskgrad.detach()

    def get_aggregate_gradient(self, images: Tensor, labels: Tensor) -> Tensor:
        _ = self.model(images)
        images_denorm = self.denormalize(images)
        images_masked = images.clone().detach()
        aggregate_grad = torch.zeros_like(self.feature_maps)
        targets = F.one_hot(labels.type(torch.int64), self.num_classes).float().to(self.device)
        for _ in range(self.ensemble_number):
            g = self.get_maskgrad(images_masked, labels)
            images_masked = self.normalize(images_denorm + self.eta * g)
            logits = self.model(images_masked)
            loss = torch.sum(logits * targets, dim=1).mean()
            # loss = self.loss_fn_ce(logits, labels)
            aggregate_grad += torch.autograd.grad(loss, self.feature_maps)[0]
        aggregate_grad /= -torch.sqrt(torch.sum(torch.square(aggregate_grad), dim=(1, 2, 3), keepdim=True))
        # aggregate_grad /= torch.sqrt(torch.sum(torch.square(aggregate_grad), dim=(1, 2, 3), keepdim=True))
        return aggregate_grad

    def bfa_loss_function(self, aggregate_grad: Tensor, x: Tensor) -> Tensor:
        _ = self.model(x)
        fia_loss = torch.mean(torch.sum(aggregate_grad * self.feature_maps, dim=(1, 2, 3)))
        return fia_loss

    def forward(self, images: Tensor, labels: Tensor) -> Tensor:
        box_min = self.normalize(torch.zeros_like(images))
        box_max = self.normalize(torch.ones_like(images))
        box_min = torch.clamp(images - self.eps, min=box_min)
        box_max = torch.clamp(images + self.eps, max=box_max)
        adv = images.clone().detach()
        g = torch.zeros_like(adv)
        aggregate_grad = self.get_aggregate_gradient(images, labels)
        for _ in range(self.steps):
            adv.requires_grad = True
            fia_loss = self.bfa_loss_function(aggregate_grad, adv)
            grad = torch.autograd.grad(fia_loss, adv)[0]
            g = self.decay * g + grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            adv = torch.clamp(adv + self.alpha * torch.sign(g), min=box_min, max=box_max).detach()
        return adv
