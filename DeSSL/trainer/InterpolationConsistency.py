from pytorch_lightning.utilities.types import STEP_OUTPUT
from . import SEMI_TRAINER_REGISTRY
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Tuple, Callable, Any
from DeSSL.scheduler import SchedulerBase
from torch import nn
from torch.distributions.beta import Beta
from copy import deepcopy
from DeSSL.transforms import mixup_for_one_hot
from .SemiBase import SemiBase

__all__ = ['InterpolationConsistency']


def _update_teacher(student, teacher, alpha):
    for teacher_param, stduent_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(alpha).add_(
            stduent_param.data, alpha=(1 - alpha))


@SEMI_TRAINER_REGISTRY.register
class InterpolationConsistency(SemiBase):
    r"""
    Reproduced trainer based on `Interpolation Consistency Training for Semi-Supervised Learning <https://arxiv.org/abs/1903.03825>`_.
    Args:
        model: The backbone model of trainer.
        optimizer: The optimizer of trainer. 
        loss_f: The classfication loss of trainer.
        consistency_weight: The consistency schedule of trainer. Corresponding to :math:`w(t)` in the original paper.
        alpha: The EMA schedule of trainer. Corresponding to :math:`\alpha` in the original paper.
        beta: The hyperparameter of beta function. Corresponding to :math:`\alpha` in the original paper.
    """

    def __init__(self,
                 train_and_val_loader: Tuple[DataLoader, DataLoader],
                 optimizer: dict,
                 lr_schedular: dict,
                 model: nn.Module,
                 consistency_weight: SchedulerBase,
                 alpha: SchedulerBase,
                 beta: float):
        super().__init__(train_and_val_loader, optimizer, lr_schedular)
        teacher = deepcopy(model)
        self.model = nn.ModuleDict({'teacher': teacher, 'student': model})
        self.consistency_weight = consistency_weight
        self.alpha = alpha
        self.beta = Beta(beta, beta)

    def forward(self, path_name, input):
        return self.model[path_name](input)

    def training_step(self, batch, batch_idx):
        (label_input, label_target), (unlabel_input, _) = batch
        label_output = self.model['student'](label_input)
        loss_train = F.cross_entropy(label_output, label_target)

        teacher_output = nn.Softmax(dim=1)(
            self.model['teacher'](unlabel_input)).detach()

        mix_input, mix_target = mixup_for_one_hot(
            unlabel_input, teacher_output, self.beta.sample())
        mix_output = self.model['student'](mix_input)
        loss_train += self.consistency_weight() * nn.MSELoss()(mix_output, mix_target)
        return super().training_step(loss_train)

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.model['student'](input)
        loss_val = F.cross_entropy(output, target)
        super().validation_step(loss_val, output, target)

    def on_train_batch_end(self,
                           outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int) -> None:
        _update_teacher(self.model['student'],
                        self.model['teacher'], self.alpha())
        self.consistency_weight.step()
        self.alpha.step()

    def configure_optimizers(self):
        optimizer_fn = self._optimizer['optimizer']
        kwargs = {}
        for k, v in self._optimizer.items():
            if k != 'optimizer':
                kwargs[k] = v
        optimizer = optimizer_fn(self.model['student'].parameters(), **kwargs)
        lr_scheduler_fn = self._lr_schedular['lr_scheduler']
        kwargs = {}
        for k, v in self._lr_schedular.items():
            if k != 'lr_scheduler':
                kwargs[k] = v
        lr_scheduler = lr_scheduler_fn(optimizer, **kwargs)
        return [optimizer], [lr_scheduler]
