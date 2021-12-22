from pytorch_lightning.utilities.types import STEP_OUTPUT
from . import SEMI_TRAINER_REGISTRY
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Tuple, Callable, Any
from DeSSL.scheduler import SchedulerBase
from torch import nn
from copy import deepcopy
from .SemiBase import SemiBase
import torch

__all__ = ['VariationalAutoEncoder']

EPS = 1e-7


def _loss(output, target, mu, logvar):
    B, C, H, W = list(target.shape)
    loss = F.binary_cross_entropy(
        output, target.view(B, -1), reduction='none').sum(dim=1)
    loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return loss


@ SEMI_TRAINER_REGISTRY.register
class VariationalAutoEncoder(SemiBase):
    r"""
    Reproduced Code based on `Semi-Supervised Learning with Deep Generative Models <https://arxiv.org/abs/1406.5298>`_.
    Args:
        encoder_model: An inference model to classify.
        vae_model: An vae model to reconstruct images as an auxiliary task.
        optimizer: The optimizer of trainer.
        loss_f: The classfication loss of trainer.
    """

    def __init__(self,
                 train_and_val_loader: Tuple[DataLoader, DataLoader],
                 model_dict: dict,
                 optimizer: dict,
                 lr_schedular: dict):
        super().__init__(train_and_val_loader, optimizer, lr_schedular)
        self.model = nn.ModuleDict(model_dict)

    def forward(self, path_name, input):
        return self.model[path_name](input)

    def training_step(self, batch, batch_idx):
        (l_data, target), (u_data, _) = batch
        # ===== working for labeled data =====
        l_output = self.model['encoder'](l_data)
        C = l_output.size(1)
        loss = F.cross_entropy(l_output, target) * 600
        target = F.one_hot(target, C)
        l_output, mu, logvar = self.model['vae'](l_data, target)
        loss += _loss(l_output, l_data, mu, logvar).mean()
        B = u_data.size(0)
        # ===== working for unlabeled data =====
        u_prob = F.softmax(self.model['encoder'](u_data), dim=1)
        loss -=  \
            (u_prob * torch.log(u_prob + EPS)).sum(dim=1).mean()
        for i in range(C):
            u_target = torch.zeros_like(u_prob)
            u_target[:, i] = 1
            u_output, mu, logvar = self.model['vae'](u_data, u_target)
            loss += (u_prob[:, i] * _loss(u_output, u_data, mu, logvar)).mean()
        return super().training_step(loss)

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.model['encoder'](input)
        loss_val = F.cross_entropy(output, target)
        super().validation_step(loss_val, output, target)
