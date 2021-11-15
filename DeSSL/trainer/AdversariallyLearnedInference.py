from pytorch_lightning.utilities.types import STEP_OUTPUT
from . import SEMI_TRAINER_REGISTRY
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Tuple, Callable, Any
from DeSSL.scheduler import SchedulerBase
from torch import nn, cat, randn_like, ones_like, zeros_like
from torch import float as float_tensor
from .SemiBase import SemiBase

__all__ = ['InterpolationConsistency']


@SEMI_TRAINER_REGISTRY.register
class AdversariallyLearnedInference(SemiBase):
    r'''
    Reproduced trainer based on `Adversarially Learned Inference <https://arxiv.org/pdf/1606.00704>`_.
    Args:
        model_dict: The ``generator_x``, ``generator_z``, ``discriminator_x``, ``discriminator_z``, and ``discriminator_x_z`` model of trainer.
        optimizer: The optimizer of trainer. 
        loss_f: The classfication loss of trainer.
        consistency_weight: The consistency schedule of trainer. 
    '''

    def __init__(self,
                 train_and_val_loader: Tuple[DataLoader, DataLoader],
                 optimizer: dict,
                 lr_schedular: dict,
                 model_dict: nn.ModuleDict,
                 consistency_weight: SchedulerBase):
        super().__init__(train_and_val_loader, optimizer, lr_schedular)
        self.model = model_dict
        self.consistency_weight = consistency_weight

    def training_step(self, batch, batch_idx, optimizer_idx):
        (label_input, label_target), (unlabel_input, _) = batch
        if optimizer_idx == 1:
            z = self.model['generator_z'](label_input).detach()
            classification_output, real_output = self.model['discriminator_x_z'](
                self.model['discriminator_x'](input), self.model['discriminator_z'](z))
            lossD = F.cross_entropy(classification_output, label_target)
            fake_z = randn_like(z)
            fake_x = self.model['generator_x'](fake_z)
            _, fake_output = self.model['discriminator_x_z'](self.model['discriminator_x'](
                fake_x.detach()), self.model['discriminator_z'](fake_z))
            output = cat((real_output, fake_output), dim=0)
            label = cat((ones_like(label_target, dtype=float_tensor), zeros_like(
                label_target, dtype=float_tensor)), dim=0).unsqueeze(1)
            lossD += self.consistency_weight() * nn.BCEWithLogitsLoss()(output, label)
            return {'lossD': lossD}

    def train_iteration(self, data: Tuple[Tensor, Tensor]) -> None:
        input, target = data
        z = self.model['generator_z'](input).detach()
        classification_output, real_output = self.model['discriminator_x_z'](
            self.model['discriminator_x'](input), self.model['discriminator_z'](z))
        lossD = self.loss_f(classification_output, target)

        fake_z = randn_like(z)
        fake_x = self.model['generator_x'](fake_z)
        _, fake_output = self.model['discriminator_x_z'](self.model['discriminator_x'](
            fake_x.detach()), self.model['discriminator_z'](fake_z))

        output = cat((real_output, fake_output), dim=0)
        label = cat((ones_like(target, dtype=float_tensor), zeros_like(
            target, dtype=float_tensor)), dim=0).unsqueeze(1)
        lossD += self.consistency_weight() * nn.BCEWithLogitsLoss()(output, label)
        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()
        self.reporter.add('lossD', lossD.detach_())

        _, real_output = self.model['discriminator_x_z'](
            self.model['discriminator_x'](input), self.model['discriminator_z'](z))
        _, fake_output = self.model['discriminator_x_z'](
            self.model['discriminator_x'](fake_x), self.model['discriminator_z'](fake_z))
        output = cat((real_output, fake_output), dim=0)
        label = cat((zeros_like(target, dtype=float_tensor), ones_like(
            target, dtype=float_tensor)), dim=0).unsqueeze(1)
        lossG = self.consistency_weight() * nn.BCEWithLogitsLoss()(output, label)
        self.optimizerG.zero_grad()
        lossG.backward()
        self.optimizerG.step()
        self.reporter.add('lossG', lossG.detach_())

    def supervised_iteration(self, data: Tuple[Tensor, Tensor]):
        input, target = data
        # According to paper, only discriminator is helpful to supervise task
        z = self.model['generator_z'](input).detach()
        output, _ = self.model['discriminator_x_z'](
            self.model['discriminator_x'](input), self.model['discriminator_z'](z))
        loss = self.loss_f(output, target)
        self.optimizerD.zero_grad()
        loss.backward()
        self.optimizerD.step()
        self.reporter.add('supervised_loss', loss.detach_())

    def unsupervised_iteration(self, data: Tuple[Tensor, Tensor]):
        input, target = data
        z = self.model['generator_z'](input)
        _, real_output = self.model['discriminator_x_z'](
            self.model['discriminator_x'](input), self.model['discriminator_z'](z.detach()))
        fake_z = randn_like(z)
        fake_x = self.model['generator_x'](fake_z)
        _, fake_output = self.model['discriminator_x_z'](self.model['discriminator_x'](
            fake_x.detach()), self.model['discriminator_z'](fake_z))
        output = cat((real_output, fake_output), dim=0)
        label = cat((ones_like(target, dtype=float_tensor), zeros_like(
            target, dtype=float_tensor)), dim=0).unsqueeze(1)
        lossD = self.consistency_weight * nn.BCEWithLogitsLoss()(output, label)
        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()
        # print(lossD.detach())
        self.reporter.add('lossD', lossD.detach_())

        _, real_output = self.model['discriminator_x_z'](
            self.model['discriminator_x'](input), self.model['discriminator_z'](z))
        _, fake_output = self.model['discriminator_x_z'](
            self.model['discriminator_x'](fake_x), self.model['discriminator_z'](fake_z))
        output = cat((real_output, fake_output), dim=0)
        label = cat((zeros_like(target, dtype=float_tensor), ones_like(
            target, dtype=float_tensor)), dim=0).unsqueeze(1)
        lossG = self.consistency_weight * nn.BCEWithLogitsLoss()(output, label)
        self.optimizerG.zero_grad()
        lossG.backward()
        self.optimizerG.step()
        self.reporter.add('lossG', lossG.detach_())

    def test_iteration(self, data: Tuple[Tensor, Tensor]):
        input, target = data
        z = self.model['generator_z'](input)
        output, _ = self.model['discriminator_x_z'](
            self.model['discriminator_x'](input), self.model['discriminator_z'](z))
        loss = self.loss_f(output, target)
        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")
        self.reporter.add('accuracy', accuracy(output, target))
        self.reporter.add('loss', loss.detach_())
        if self._report_topk is not None:
            for top_k in self._report_topk:
                self.reporter.add(
                    f'accuracy@{top_k}', accuracy(output, target, top_k))

    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> None:
        if self.is_train:
            self.train_iteration(data)
            self.consistency_weight.step()
        else:
            self.test_iteration(data)

    def state_dict(self
                   ) -> Mapping[str, Any]:

        return {'model': self.accessible_model.state_dict(),
                'optim': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'use_sync_bn': self._use_sync_bn
                }

    def load_state_dict(self,
                        state_dict: Mapping[str, Any]
                        ) -> None:
        self.accessible_model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optim'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.scheduler.last_epoch = state_dict['epoch']
        self._epoch = state_dict['epoch']
        self._use_sync_bn = state_dict['use_sync_bn']

    def data_preprocess(self,
                        data: Tuple[Tensor, ...]
                        ) -> (Tuple[Tensor, ...], int):
        if isinstance(data[0], list):
            data = (data[0][0], data[0][1], data[1])
        return TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking), data[0].size(0)
