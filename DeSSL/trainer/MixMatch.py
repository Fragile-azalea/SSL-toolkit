from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.functional import Tensor
from . import SEMI_TRAINER_REGISTRY
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Any
from DeSSL.scheduler import SchedulerBase
from torch import nn, cat, no_grad, zeros, randperm
from torch import float as float_tensor
from .SemiBase import SemiBase
from torch.distributions.beta import Beta

__all__ = ['MixMatch']


@SEMI_TRAINER_REGISTRY.register
class MixMatch(SemiBase):
    r'''
    Reproduced trainer based on `MixMatch: A Holistic Approach to Semi-Supervised Learning <https://arxiv.org/abs/1905.02249>`_.
    Args:
        model: The backbone model of trainer.
        optimizer: The optimizer of trainer.
        loss_f: The classfication loss of trainer.
        temperature: The temperature of sharpen function. Corresponding to ``T`` in the original paper.
        beta: The hyperparameter of beta function. Corresponding to :math:`\alpha` in the original paper.
        consistency_weight: The consistency schedule of trainer. Corresponding to :math:`\lambda_\mathcal{u}` in the original paper.
        dataset_type: The type of dataset. Choose ``mix`` or ``split`` corresponding to dataset type.
    '''

    def __init__(self,
                 train_and_val_loader: Tuple[DataLoader, DataLoader],
                 optimizer: dict,
                 lr_schedular: dict,
                 model: nn.Module,
                 temperature: float,
                 beta: float,
                 consistency_weight: SchedulerBase):
        super().__init__(train_and_val_loader, optimizer, lr_schedular)
        self.temperature = temperature
        self.beta = Beta(beta, beta)
        self.model = model
        self.consistency_weight = consistency_weight

    def generate_mixmatch(self, label_data, label_target, augment_unlabel_list):
        num_list = len(augment_unlabel_list)
        batch_size = label_data.size(0)
        fake_target = sum([nn.Softmax(dim=1)(self.model(unlabel))
                           for unlabel in augment_unlabel_list]) / num_list
        fake_target = fake_target ** self.temperature
        fake_target = fake_target / fake_target.sum(dim=1, keepdim=True)
        fake_target = fake_target.repeat(num_list, 1)

        all_data = cat((label_data, *augment_unlabel_list), dim=0)
        '''
        change one-hot label_target to (batch_size, num_classes)
        combine one-hot label_target and fake_target
        '''
        all_target = cat((zeros((batch_size, fake_target.size(1)), device=label_target.get_device(
        )).scatter_(1, label_target.view(-1, 1), 1), fake_target), dim=0)

        rand_data = all_data[randperm(all_data.size(0))]
        rand_target = all_target[randperm(all_target.size(0))]

        alpha = self.beta.sample((all_target.size(0),)).to(
            label_target.get_device())
        alpha = alpha.max(1 - alpha).view(-1, 1, 1, 1)
        mix_data = alpha * all_data + (1 - alpha) * rand_data
        alpha = alpha.view(-1, 1)
        mix_target = alpha * all_target + (1 - alpha) * rand_target
        return mix_data[:batch_size], mix_data[batch_size:].split(batch_size), mix_target[:batch_size], mix_target[batch_size:].split(batch_size)

    def training_step(self, batch, batch_idx):
        (label_input, label_target), (augment_unlabel_list, _) = batch
        num_list = len(augment_unlabel_list)
        with no_grad():
            mix_input, mix_unlabel_list, mix_target, mix_fake_target_list = self.generate_mixmatch(
                label_input, label_target, augment_unlabel_list)
        output = nn.LogSoftmax(dim=1)(self.model(mix_input))
        loss_train = -(output * mix_target).sum(dim=1).mean()
        for mix_unlabel, mix_fake_target in zip(mix_unlabel_list, mix_fake_target_list):
            fake_output = self.model(mix_unlabel)
            loss_train += self.consistency_weight() * nn.MSELoss()(fake_output,
                                                                   mix_fake_target) / num_list
        return super().training_step(loss_train)

    def on_train_batch_end(self,
                           outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int) -> None:
        self.consistency_weight.step()

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input)
        loss_val = F.cross_entropy(output, target)
        super().validation_step(loss_val, output, target)
