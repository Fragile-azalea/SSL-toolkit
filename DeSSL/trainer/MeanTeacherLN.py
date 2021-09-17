from pytorch_lightning.utilities.types import STEP_OUTPUT
from . import SEMI_TRAINER_REGISTRY
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from typing import Tuple, Callable, Any
from torch import nn
from copy import deepcopy
from . import SEMI_TRAINER_REGISTRY

__all__ = ['MeanTeacher']


def _update_teacher(student, teacher, alpha):
    for teacher_param, stduent_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(alpha).add_(
            stduent_param.data, alpha=(1 - alpha))


@SEMI_TRAINER_REGISTRY.register
class MeanTeacher(LightningModule):
    def __init__(self,
                 train_and_val_loader: Tuple[DataLoader, DataLoader],
                 optimizer: dict,
                 lr_schedular: dict,
                 model: nn.Module,
                 consistency_weight: Callable,
                 alpha: Callable):
        super().__init__()
        self._loader = train_and_val_loader
        self._optimizer = optimizer
        self._lr_schedular = lr_schedular
        teacher = deepcopy(model)
        self.model = nn.ModuleDict({'teacher': teacher, 'student': model})
        self.consistency_weight = consistency_weight
        self.alpha = alpha
        # self.save_hyperparameters()

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
        self.log('train/loss', loss_train, on_step=True,
                 on_epoch=True, logger=True)
        return loss_train

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.model['teacher'](input)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('val_loss', loss_val, on_epoch=True)
        self.log('val_acc1', acc1, prog_bar=True, on_epoch=True)
        self.log('val_acc5', acc5, on_epoch=True)

    @staticmethod
    def __accuracy(output, target, topk=(1, )):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0,
                                                                keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer_fn = self._optimizer['optimizer']
        kwargs = {}
        for k, v in self._optimizer.items():
            if k != 'optimizer':
                kwargs[k] = v
        optimizer = optimizer_fn(self.parameters(), **kwargs)
        lr_scheduler_fn = self._lr_schedular['lr_scheduler']
        kwargs = {}
        for k, v in self._lr_schedular.items():
            if k != 'lr_scheduler':
                kwargs[k] = v
        lr_scheduler = lr_scheduler_fn(optimizer, **kwargs)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self._loader[0]

    def val_dataloader(self):
        return self._loader[1]

    def on_train_batch_end(self,
                           outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int) -> None:
        _update_teacher(self.model['student'],
                        self.model['teacher'], self.alpha())
        self.consistency_weight.step()
        self.alpha.step()
