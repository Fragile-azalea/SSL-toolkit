from homura.trainers import TrainerBase
from typing import List, Callable, Optional, Tuple, Mapping, Any
from torch import nn, Tensor, cat
from torch.optim import Optimizer
from homura.reporters import _ReporterBase
from copy import deepcopy
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torchvision.transforms import RandomResizedCrop, Compose, ToPILImage, ToTensor, Normalize
from homura.utils.containers import TensorTuple
from homura.metrics import accuracy
from torch.distributions.beta import Beta
from homura.vision.transforms.mixup import mixup
from . import SEMI_TRAINER_REGISTRY


def _update_teacher(student, teacher, alpha):
    for teacher_param, stduent_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.add_((1 - alpha) * stduent_param.data, alpha=alpha)


class InterpolationConsistencyTrainer(TrainerBase):
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 consistency_weight: float,
                 alpha: float,
                 beta: float,
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        teacher = deepcopy(model)
        model = {'student': model, 'teacher': teacher}
        super(InterpolationConsistencyTrainer, self).__init__(model, optimizer, loss_f, reporters=reporters,
                                                              scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        self.consistency_weight = consistency_weight
        self.alpha = alpha
        self.beta = Beta(beta, beta)
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> None:
        input, target = data
        if self.is_train:
            if self.is_supervised:
                output = self.model['student'](input)
                loss = self.loss_f(output, target)
            else:
                teacher_output = self.model['teacher'](input).detach()
                input, teacher_target = mixup(
                    input, teacher_output, self.beta.sample())
                output = self.model['student'](input)
                loss = self.consistency_weight * nn.MSELoss()(output, teacher_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _update_teacher(
                self.model['student'], self.model['teacher'], self.alpha, self.epoch)
        else:
            output = self.model['student'](input)
            loss = self.loss_f(output, target)

        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")
        # print(output.shape, target.shape)
        self.reporter.add('accuracy', accuracy(output, target))
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


class InterpolationConsistencyTrainerV2(TrainerBase):
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 consistency_weight: float,
                 alpha: float,
                 beta: float,
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        teacher = deepcopy(model)
        model = {'student': model, 'teacher': teacher}
        super(InterpolationConsistencyTrainerV2, self).__init__(model, optimizer, loss_f, reporters=reporters,
                                                                scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        self.consistency_weight = consistency_weight
        self.alpha = alpha
        self.beta = Beta(beta, beta)
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> None:
        input, target = data
        unlabel = input[target == -100]
        if self.is_train:
            output = self.model['student'](input)
            loss = self.loss_f(output, target)
            teacher_output = self.model['teacher'](unlabel).detach()
            unlabel, teacher_target = mixup(
                unlabel, teacher_output, self.beta.sample())
            student_output = self.model['student'](unlabel)
            loss += self.consistency_weight * nn.MSELoss()(student_output, teacher_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _update_teacher(
                self.model['student'], self.model['teacher'], self.alpha, self.epoch)
        else:
            output = self.model['student'](input)
            loss = self.loss_f(output, target)

        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")
        # print(output.shape, target.shape)
        self.reporter.add('accuracy', accuracy(output, target))
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
class InterpolationConsistency(TrainerBase):
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 consistency_weight: Callable,
                 alpha: Callable,
                 beta: float,
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        teacher = deepcopy(model)
        model = {'student': model, 'teacher': teacher}
        super(InterpolationConsistency, self).__init__(model, optimizer, loss_f, reporters=reporters,
                                                       scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        self.consistency_weight = consistency_weight
        self.alpha = alpha
        self.beta = Beta(beta, beta)
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def data_preprocess(self, data: Tuple[Tensor, ...]) -> (Tuple[Tensor, ...], int):
        if isinstance(data, tuple):
            data = (cat((data[0][0], data[1][0]), dim=0),
                    cat((data[0][1], data[1][1]), dim=0))
        return TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking), data[0].size(0)

    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> None:
        input, target = data
        if self.is_train:
            unlabel = input[target == -100]
            output = self.model['student'](input)
            loss = self.loss_f(output, target)
            teacher_output = self.model['teacher'](unlabel).detach()
            unlabel, teacher_target = mixup(
                unlabel, teacher_output, self.beta.sample())
            student_output = self.model['student'](unlabel)
            loss += self.consistency_weight() * nn.MSELoss()(student_output, teacher_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _update_teacher(
                self.model['student'], self.model['teacher'], self.alpha())
            self.consistency_weight.step()
            self.alpha.step()
        else:
            output = self.model['student'](input)
            loss = self.loss_f(output, target)
            self.reporter.add('accuracy', accuracy(output, target))
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
