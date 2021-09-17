from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from . import SEMI_TRAINER_REGISTRY
from . import SemiBase


@SEMI_TRAINER_REGISTRY.register
class Ladder(SemiBase):
    r'''
    Reproduced trainer based on `Semi-Supervised Learning with Ladder Networks <https://arxiv.org/abs/1507.02672>`_.

    Args:
        model: The backbone model of trainer.
        lam_list: The order list of regularization coefficient, which describes how important to recovery signals.  Corresponding to :math:`\lambda^i` in the original paper.
    '''

    def __init__(self,
                 train_and_val_loader: Tuple[DataLoader, DataLoader],
                 optimizer: dict,
                 lr_schedular: dict,
                 model: nn.Module,
                 lam_list: List[float]):
        super().__init__(train_and_val_loader, optimizer, lr_schedular)
        self.model = model
        self.lam_list = lam_list
        # self.save_hyperparameters()

    def forward(self, path_name, *input):
        return self.model(path_name, *input)

    def training_step(self, batch, batch_idx):
        (label_data, target), (unlabel_data, _) = batch
        self('clear', label_data, unlabel_data)
        output = self('noise', label_data, unlabel_data)
        self('decoder')
        loss = self.model.get_loss_d(self.lam_list)
        loss_train = loss + F.nll_loss(torch.log(output), target)
        return super().training_step(loss_train)

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self('clear', input)
        loss_val = F.nll_loss(torch.log(output), target)
        super().validation_step(loss_val, output, target)
