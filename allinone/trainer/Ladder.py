from homura.metrics import accuracy
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from homura.reporters import _ReporterBase
from torch.optim import Optimizer
from torch import Tensor, nn
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple
from homura.trainers import TrainerBase
import torch
from . import SEMI_TRAINER_REGISTRY, unroll
from homura.utils.containers import TensorTuple

__all__ = ['Ladder']


@ SEMI_TRAINER_REGISTRY.register
class Ladder(TrainerBase):
    r'''
    Reproduced trainer based on `Semi-Supervised Learning with Ladder Networks <https://arxiv.org/abs/1507.02672>`_.

    Args:
        model: The backbone model of trainer.
        optimizer: The optimizer of trainer.
        loss_f: The classfication loss of trainer.
        lam_list: The order list of regularization coefficient, which describes how important to recovery signals.  Corresponding to :math:`\lambda^i` in the original paper.
    '''

    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 lam_list: List[float],
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        super(Ladder, self).__init__(
            model,
            optimizer,
            loss_f,
            reporters=reporters,
            scheduler=scheduler,
            verb=verb,
            use_cudnn_benchmark=use_cudnn_benchmark,
            **kwargs)
        self.lam_list = lam_list
        if report_accuracy_topk is not None and not isinstance(
                report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def data_preprocess(
            self, data: Tuple[Tuple[Tensor, ...], ...]) -> (Tuple[Tensor, ...], int):
        data = unroll(data)
        return TensorTuple(data).to(
            self.device, non_blocking=self._cuda_nonblocking), data[0].size(0)

    def iteration(self, data: Tuple[Tensor, ...]) -> None:
        if self.is_train:
            label_data, target, unlabel_data, _ = data
            # clean path
            self.model('clear', label_data, unlabel_data)
            # noise path
            output = self.model('noise', label_data, unlabel_data)
            # decoder path
            self.model('decoder')
            loss = self.model.get_loss_d(self.lam_list)
            loss += self.loss_f(torch.log(output), target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            input, target = data
            output = self.model('clear', input)
            loss = self.loss_f(torch.log(output), target)
            self.reporter.add('accuracy', accuracy(output, target))

        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")

        self.reporter.add('loss', loss.detach_())
        if self._report_topk is not None:
            for top_k in self._report_topk:
                self.reporter.add(
                    f'accuracy@{top_k}', accuracy(output, target, top_k))

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
