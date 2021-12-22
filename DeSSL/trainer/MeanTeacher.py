from pytorch_lightning.utilities.types import STEP_OUTPUT
from . import SEMI_TRAINER_REGISTRY
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Tuple, Callable, Any
from DeSSL.scheduler import SchedulerBase
from torch import nn
from copy import deepcopy
from .SemiBase import SemiBase

__all__ = ['MeanTeacher']


def _update_teacher(student, teacher, alpha):
    for teacher_param, stduent_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(alpha).add_(
            stduent_param.data, alpha=(1 - alpha))


@SEMI_TRAINER_REGISTRY.register
class MeanTeacher(SemiBase):
    r"""
    Reproduced trainer based on `Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results <https://arxiv.org/abs/1703.01780>`_.

    Args:
        model: The backbone model of trainer.
        consistency_weight: The consistency schedule of trainer. Corresponding to ``consistency cost coefficient`` in the original paper.
        alpha: The EMA schedule of trainer. Corresponding to :math:`\alpha` in the original paper.
    """

    def __init__(self,
                 train_and_val_loader: Tuple[DataLoader, DataLoader],
                 optimizer: dict,
                 lr_schedular: dict,
                 model: nn.Module,
                 consistency_weight: SchedulerBase,
                 alpha: SchedulerBase):
        super().__init__(train_and_val_loader, optimizer, lr_schedular)
        teacher = deepcopy(model)
        self.model = nn.ModuleDict({'teacher': teacher, 'student': model})
        self.consistency_weight = consistency_weight
        self.alpha = alpha

    def forward(self, path_name, input):
        return self.model[path_name](input)

    def training_step(self, batch, batch_idx):
        (label_input, label_target), ((stduent_input, teacher_input), _) = batch
        stduent_output = nn.LogSoftmax(dim=1)(
            self.model['student'](stduent_input))
        teacher_output = nn.Softmax(dim=1)(
            self.model['teacher'](teacher_input)).detach()
        label_output = self.model['student'](label_input)
        loss_train = self.consistency_weight(
        ) * nn.KLDivLoss(reduction='batchmean')(stduent_output, teacher_output)
        loss_train += F.cross_entropy(label_output, label_target)
        return super().training_step(loss_train)

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.model['teacher'](input)
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
