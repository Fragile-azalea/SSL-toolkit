from homura.trainers import TrainerBase
from typing import List, Callable, Optional, Tuple, Mapping, Any
from torch import nn, Tensor
from torch.optim import Optimizer
from homura.reporters import _ReporterBase
from copy import deepcopy
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from homura.utils.containers import TensorTuple
from homura.metrics import accuracy
from . import SEMI_TRAINER_REGISTRY, unroll

__all__ = ['MeanTeacher']


def _update_teacher(student, teacher, alpha):
    for teacher_param, stduent_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.add_((1 - alpha) * stduent_param.data, alpha=alpha)


class MeanTeacherTrainer(TrainerBase):
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 consistency_weight: float,
                 alpha: float,
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        teacher = deepcopy(model)
        model = {'student': model, 'teacher': teacher}
        super(MeanTeacherTrainer, self).__init__(model, optimizer, loss_f, reporters=reporters,
                                                 scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        self.consistency_weight = consistency_weight
        self.alpha = alpha
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def data_preprocess(self,
                        data: Tuple[Tensor, ...]
                        ) -> (Tuple[Tensor, ...], int):
        if isinstance(data[0], list):
            data = (data[0][0], data[0][1], data[1])
        return TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking), data[0].size(0)

    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> None:
        if len(data) == 3:
            stduent_input, teacher_input, target = data
        else:
            input, target = data

        if self.is_train:
            stduent_output = nn.LogSoftmax(dim=1)(
                self.model['student'](stduent_input))
            teacher_output = nn.Softmax(dim=1)(
                self.model['teacher'](teacher_input))
            loss = self.consistency_weight * \
                nn.KLDivLoss(reduction='batchmean')(
                    stduent_output, teacher_output.detach())
            if self.is_supervised:
                loss += nn.NLLLoss()(stduent_output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _update_teacher(
                self.model['student'], self.model['teacher'], self.alpha, self.epoch)
        else:
            teacher_output = self.model['teacher'](input)
            loss = self.loss_f(teacher_output, target)

        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")
        self.reporter.add('accuracy', accuracy(teacher_output, target))
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


class MeanTeacherTrainerV2(TrainerBase):
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 consistency_weight: float,
                 alpha: float,
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        teacher = deepcopy(model)
        model = {'student': model, 'teacher': teacher}
        super(MeanTeacherTrainerV2, self).__init__(model, optimizer, loss_f, reporters=reporters,
                                                   scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        self.consistency_weight = consistency_weight
        self.alpha = alpha
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def data_preprocess(self,
                        data: Tuple[Tensor, ...]
                        ) -> (Tuple[Tensor, ...], int):
        if isinstance(data[0], list):
            data = (data[0][0], data[0][1], data[1])
        return TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking), data[0].size(0)

    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> None:
        if len(data) == 3:
            stduent_input, teacher_input, target = data
        else:
            input, target = data

        if self.is_train:
            stduent_output = nn.LogSoftmax(dim=1)(
                self.model['student'](stduent_input))
            teacher_output = nn.Softmax(dim=1)(
                self.model['teacher'](teacher_input))
            loss = self.consistency_weight * \
                nn.KLDivLoss(reduction='batchmean')(
                    stduent_output, teacher_output.detach())
            loss += nn.NLLLoss()(stduent_output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _update_teacher(
                self.model['student'], self.model['teacher'], self.alpha, self.epoch)
        else:
            teacher_output = self.model['teacher'](input)
            loss = self.loss_f(teacher_output, target)

        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")
        self.reporter.add('accuracy', accuracy(teacher_output, target))
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


@SEMI_TRAINER_REGISTRY.register
class MeanTeacher(TrainerBase):
    r'''
    Reproduced trainer based on `Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results <https://arxiv.org/abs/1703.01780>`_.

    Args:
        model: The backbone model of trainer.
        optimizer: The optimizer of trainer. 
        loss_f: The classfication loss of trainer.
        consistency_weight: The consistency schedule of trainer. Corresponding to ``consistency cost coefficient`` in the original paper.
        alpha: The EMA schedule of trainer. Corresponding to :math:`\alpha` in the original paper.
        dataset_type: The type of dataset. Choose ``mix`` or ``split`` corresponding to dataset type.
    '''

    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 consistency_weight: Callable,
                 alpha: Callable,
                 dataset_type: str,
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        teacher = deepcopy(model)
        model = {'student': model, 'teacher': teacher}
        super(MeanTeacher, self).__init__(model, optimizer, loss_f, reporters=reporters,
                                          scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        self.consistency_weight = consistency_weight
        self.alpha = alpha
        assert dataset_type == 'mix' or dataset_type == 'split'
        self.dataset_type = dataset_type
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def data_preprocess(self,
                        data: Tuple[Tensor, ...]
                        ) -> (Tuple[Tensor, ...], int):
        r'''
        The labeled data and unlabeled data are combined if they were split previously.
        '''
        data = unroll(data)
        return TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking), data[0].size(0)

    def _mix_iteration(self, data):
        stduent_input, teacher_input, target = data
        stduent_output = nn.LogSoftmax(dim=1)(
            self.model['student'](stduent_input))
        teacher_output = nn.Softmax(dim=1)(
            self.model['teacher'](teacher_input)).detach()
        loss = self.consistency_weight(
        ) * nn.KLDivLoss(reduction='batchmean')(stduent_output, teacher_output)
        loss += nn.NLLLoss()(stduent_output, target)
        return loss

    def _split_iteration(self, data):
        label_input, label_target, stduent_input, teacher_input, _ = data
        stduent_output = nn.LogSoftmax(dim=1)(
            self.model['student'](stduent_input))
        teacher_output = nn.Softmax(dim=1)(
            self.model['teacher'](teacher_input)).detach()
        label_output = self.model['student'](label_input)
        loss = self.consistency_weight(
        ) * nn.KLDivLoss(reduction='batchmean')(stduent_output, teacher_output)
        loss += self.loss_f(label_output, label_target)
        return loss

    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> None:
        if self._is_train:
            if self.dataset_type == 'mix':
                loss = self._mix_iteration(data)
            else:
                loss = self._split_iteration(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _update_teacher(self.model['student'],
                            self.model['teacher'], self.alpha())
            self.reporter.add('loss', loss.detach_())
            self.consistency_weight.step()
            self.alpha.step()
        else:
            input, target = data
            teacher_output = self.model['teacher'](input)
            loss = self.loss_f(teacher_output, target)
            self.reporter.add('accuracy', accuracy(teacher_output, target))
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
