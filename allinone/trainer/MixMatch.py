from homura.trainers import TrainerBase
from typing import List, Callable, Optional, Tuple, Mapping, Any
from torch import nn, Tensor, zeros, cat, randperm, no_grad
from torch.optim import Optimizer
from homura.reporters import _ReporterBase
from copy import deepcopy
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torchvision.transforms import RandomResizedCrop, Compose, ToPILImage, ToTensor, Normalize
from homura.utils.containers import TensorTuple
from homura.metrics import accuracy
from torch.distributions.beta import Beta
from torchvision import transforms as tf
from . import SEMI_TRAINER_REGISTRY, unroll


class MixMatchTrainer(TrainerBase):
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 temperature: float,
                 beta: float,
                 consistency_weight: float,
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        super(MixMatchTrainer, self).__init__(model, optimizer, loss_f, reporters=reporters,
                                              scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        self.temperature = temperature
        self.beta = Beta(beta, beta)
        self.consistency_weight = consistency_weight
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def data_preprocess(self,
                        data: Tuple[Tensor, ...]
                        ) -> (Tuple[Tensor, ...], int):
        if isinstance(data[0], list):
            data = (*data[0], *data[1][0], data[1][1])
        return TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking), data[0].size(0)

    def generate_mixmatch(self, label_data, label_target, augment_unlabel_list):
        num_list = len(augment_unlabel_list)
        batch_size = label_data.size(0)
        fake_target = sum([nn.Softmax(dim=1)(self.model(unlabel).detach())
                           for unlabel in augment_unlabel_list]) / num_list
        fake_target = fake_target ** self.temperature
        fake_target = fake_target / fake_target.sum(dim=1, keepdim=True)
        fake_target = fake_target.repeat(num_list, 1)

        all_data = cat((label_data, cat(augment_unlabel_list, dim=0)), dim=0)
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

    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> None:
        input = data[0]
        target = data[1]
        if self.is_train:
            unlabel_data = data[2]
            augment_unlabel_list = data[3:-1]
            num_list = len(augment_unlabel_list)
            mix_input, mix_unlabel_list, mix_target, mix_fake_target_list = self.generate_mixmatch(
                input, target, augment_unlabel_list)
            output = nn.LogSoftmax(dim=1)(self.model(mix_input))
            loss = -(output * mix_target).sum(dim=1).mean()
            for mix_unlabel, mix_fake_target in zip(mix_unlabel_list, mix_fake_target_list):
                fake_output = self.model(mix_unlabel)
                loss += self.consistency_weight * \
                    nn.MSELoss()(fake_output, mix_fake_target) / num_list
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            output = self.model(input)
            loss = self.loss_f(output, target)

        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")
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
class MixMatch(TrainerBase):
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 temperature: float,
                 beta: float,
                 consistency_weight: Callable,
                 dataset_type: str,
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        super(MixMatch, self).__init__(model, optimizer, loss_f, reporters=reporters,
                                       scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        assert dataset_type == 'split'
        self.temperature = temperature
        self.beta = Beta(beta, beta)
        self.consistency_weight = consistency_weight
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def data_preprocess(self,
                        data: Tuple[Tensor, ...]
                        ) -> (Tuple[Tensor, ...], int):
        data = unroll(data)
        return TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking), data[0].size(0)

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

    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> None:
        input, target = data[:2]
        if self.is_train:
            augment_unlabel_list = data[2:-1]
            num_list = len(augment_unlabel_list)
            with no_grad():
                mix_input, mix_unlabel_list, mix_target, mix_fake_target_list = self.generate_mixmatch(
                    input, target, augment_unlabel_list)
            output = nn.LogSoftmax(dim=1)(self.model(mix_input))
            loss = -(output * mix_target).sum(dim=1).mean()
            for mix_unlabel, mix_fake_target in zip(mix_unlabel_list, mix_fake_target_list):
                fake_output = self.model(mix_unlabel)
                loss += self.consistency_weight() * nn.MSELoss()(fake_output,
                                                                 mix_fake_target) / num_list
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.consistency_weight.step()
        else:
            output = self.model(input)
            loss = self.loss_f(output, target)

        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")
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
