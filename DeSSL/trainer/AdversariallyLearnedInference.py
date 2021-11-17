from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.functional import Tensor
from . import SEMI_TRAINER_REGISTRY
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Any
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
                 optimizerD: dict,
                 optimizerG: dict,
                 lr_schedular: dict,
                 model_dict: nn.ModuleDict,
                 consistency_weight: SchedulerBase):
        super().__init__(train_and_val_loader, None, lr_schedular)
        self.model = model_dict
        self._optimizerD = optimizerD
        self._optimizerG = optimizerG
        self.consistency_weight = consistency_weight

    def discriminator_update(self, input: Tensor, z: Tensor, d_target: ones_like | zeros_like, class_target: Optional[Tensor] = None):
        input = input.detach()
        z = z.detach()
        class_output, d_output = self.model['discriminator_x_z'](
            self.model['discriminator_x'](input), self.model['discriminator_z'](z))
        d_target = d_target(d_output)
        lossD = self.consistency_weight() * nn.BCEWithLogitsLoss()(d_output, d_target)
        if d_target:
            lossD += F.cross_entropy(class_output, class_target)
        return lossD

    def generator_update(self, input: Tensor, z: Tensor, d_target: ones_like | zeros_like, class_target: Optional[Tensor] = None):
        _, d_output = self.model['discriminator_x_z'](
            self.model['discriminator_x'](input), self.model['discriminator_z'](z))
        d_target = d_target(d_output)
        lossG = self.consistency_weight() * nn.BCEWithLogitsLoss()(d_output, d_target)
        return lossG

    def training_step(self, batch, batch_idx, optimizer_idx):
        (label_input, label_target), (unlabel_input, _) = batch
        label_z = self.model['generator_z'](label_input)
        fake_label_z = randn_like(label_z)
        fake_label_x = self.model['generator_x'](fake_label_z)
        unlabel_z = self.model['generator_z'](unlabel_input)
        fake_unlabel_z = randn_like(unlabel_z)
        fake_unlabel_x = self.model['generator_x'](fake_unlabel_z)
        if optimizer_idx == 0:
            lossD = (self.discriminator_update(label_input, label_z, ones_like, label_target) +
                     self.discriminator_update(fake_label_x, fake_label_z, zeros_like)) / 2.
            lossD += (self.discriminator_update(unlabel_input, unlabel_z, ones_like) +
                      self.discriminator_update(fake_unlabel_x, fake_unlabel_z, zeros_like)) / 2.
            self.log('train/lossD', lossD, on_step=True,
                     on_epoch=True, logger=True)
            return lossD
        if optimizer_idx == 1:
            lossG = (self.generator_update(label_input, label_z, zeros_like) +
                     self.generator_update(fake_label_x, fake_label_z, ones_like)) / 2.
            lossG += (self.generator_update(unlabel_input, unlabel_z, zeros_like) +
                      self.generator_update(fake_unlabel_x, fake_unlabel_z, ones_like)) / 2.
            self.log('train/lossG', lossG, on_step=True,
                     on_epoch=True, logger=True)
            return lossG

    def on_train_batch_end(self,
                           outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int) -> None:
        self.consistency_weight.step()

    def validation_step(self, batch, batch_idx):
        input, target = batch
        z = self.model['generator_z'](input)
        class_out, _ = self.model['discriminator_x_z'](
            self.model['discriminator_x'](input), self.model['discriminator_z'](z))
        loss_val = F.cross_entropy(class_out, target)
        super().validation_step(loss_val, class_out, target)

    def configure_optimizers(self):
        optimizer_fn = self._optimizerD['optimizer']
        kwargs = {}
        for k, v in self._optimizerD.items():
            if k != 'optimizer':
                kwargs[k] = v
        optimizerD = optimizer_fn(self.model['discriminator_x'].parameters(
        ) + self.model['discriminator_z'].parameters() + self.model['discriminator_x_z'].parameters(), **kwargs)
        optimizer_fn = self._optimizerG['optimizer']
        kwargs = {}
        for k, v in self._optimizerG.items():
            if k != 'optimizer':
                kwargs[k] = v
        optimizerG = optimizer_fn(self.model['generator_x'].parameters(
        ) + self.model['generator_z'].parameters(),  **kwargs)
        lr_scheduler_fn = self._lr_schedular['lr_scheduler']
        kwargs = {}
        for k, v in self._lr_schedular.items():
            if k != 'lr_scheduler':
                kwargs[k] = v
        lrD_scheduler = lr_scheduler_fn(optimizerD, **kwargs)
        lrG_scheduler = lr_scheduler_fn(optimizerG, **kwargs)
        return [optimizerD, optimizerG], [lrD_scheduler, lrG_scheduler]
