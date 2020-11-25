from homura.trainers import TrainerBase
from typing import List, Callable, Optional, Tuple, Mapping, Any
from torch import nn, Tensor, cat, randn_like, ones_like, zeros_like
from torch import float as float_tensor
from torch.optim import Optimizer
from homura.reporters import _ReporterBase
from copy import deepcopy
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from homura.utils.containers import TensorTuple
from homura.metrics import accuracy
from torch.distributions.beta import Beta
from homura.vision.transforms.mixup import mixup
from . import SEMI_TRAINER_REGISTRY


class AdversariallyLearnedInferenceTrainer(TrainerBase):
    def __init__(self,
                 model_dict: nn.ModuleDict,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 consistency_weight: float,
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        super(AdversariallyLearnedInferenceTrainer, self).__init__(model_dict, optimizer, loss_f,
                                                                   reporters=reporters, scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        self.consistency_weight = consistency_weight
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def set_optimizer(self):
        paramsD = list(self.model['discriminator_x'].parameters(
        )) + list(self.model['discriminator_z'].parameters()) + list(self.model['discriminator_x_z'].parameters())
        self.optimizerD = self.optimizer(paramsD)
        paramsG = list(self.model['generator_x'].parameters(
        )) + list(self.model['generator_z'].parameters())
        self.optimizerG = self.optimizer(paramsG)

    def supervised_iteration(self, data: Tuple[Tensor, Tensor]):
        input, target = data
        # Accrding to paper, only discriminator is helpful to supervise task
        z = self.model['generator_z'](input).detach()
        output, _ = self.model['discriminator_x_z'](
            self.model['discriminator_x'](input), self.model['discriminator_z'](z))
        loss = self.loss_f(output, target)
        self.optimizerD.zero_grad()
        loss.backward()
        self.optimizerD.step()
        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")
        self.reporter.add('accuracy', accuracy(output, target))
        self.reporter.add('loss', loss.detach_())
        if self._report_topk is not None:
            for top_k in self._report_topk:
                self.reporter.add(
                    f'accuracy@{top_k}', accuracy(output, target, top_k))

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
        # print(lossG.detach())
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
            if self.is_supervised:
                self.supervised_iteration(data)
            else:
                self.unsupervised_iteration(data)
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


class AdversariallyLearnedInferenceTrainerV2(TrainerBase):
    def __init__(self,
                 model_dict: nn.ModuleDict,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 consistency_weight: float,
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        super(AdversariallyLearnedInferenceTrainerV2, self).__init__(model_dict, optimizer, loss_f,
                                                                     reporters=reporters, scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        self.consistency_weight = consistency_weight
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def set_optimizer(self):
        paramsD = list(self.model['discriminator_x'].parameters(
        )) + list(self.model['discriminator_z'].parameters()) + list(self.model['discriminator_x_z'].parameters())
        self.optimizerD = self.optimizer(paramsD)
        paramsG = list(self.model['generator_x'].parameters(
        )) + list(self.model['generator_z'].parameters())
        self.optimizerG = self.optimizer(paramsG)

    def supervised_iteration(self, data: Tuple[Tensor, Tensor]):
        input, target = data
        # According to paper, only discriminator is helpful to supervise task
        z = self.model['generator_z'](input).detach()
        output, _ = self.model['discriminator_x_z'](
            self.model['discriminator_x'](input), self.model['discriminator_z'](z))
        loss = self.loss_f(output, target)
        self.optimizerD.zero_grad()
        self.optimizerG.zero_grad()
        loss.backward()
        self.optimizerD.step()
        self.optimizerG.step()
        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")
        self.reporter.add('accuracy', accuracy(output, target))
        self.reporter.add('loss', loss.detach_())
        if self._report_topk is not None:
            for top_k in self._report_topk:
                self.reporter.add(
                    f'accuracy@{top_k}', accuracy(output, target, top_k))

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
        # print(lossG.detach())
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
            input, target = data
            label_data = (input[target != - 100], target[target != - 100])
            unlabel_data = (input[target == - 100], target[target == - 100])
            self.supervised_iteration(label_data)
            self.unsupervised_iteration(unlabel_data)
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


@SEMI_TRAINER_REGISTRY.register
class AdversariallyLearnedInference(TrainerBase):
    def __init__(self,
                 model_dict: nn.ModuleDict,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 consistency_weight: Callable,
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        super(AdversariallyLearnedInference, self).__init__(model_dict, optimizer, loss_f,
                                                            reporters=reporters, scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        self.consistency_weight = consistency_weight
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def set_optimizer(self):
        paramsD = list(self.model['discriminator_x'].parameters(
        )) + list(self.model['discriminator_z'].parameters()) + list(self.model['discriminator_x_z'].parameters())
        self.optimizerD = self.optimizer(paramsD)
        paramsG = list(self.model['generator_x'].parameters(
        )) + list(self.model['generator_z'].parameters())
        self.optimizerG = self.optimizer(paramsG)

    def data_preprocess(self, data: Tuple[Tensor, ...]) -> (Tuple[Tensor, ...], int):
        if isinstance(data, tuple):
            data = (cat((data[0][0], data[1][0]), dim=0),
                    cat((data[0][1], data[1][1]), dim=0))
        return TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking), data[0].size(0)

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


if __name__ == "__main__":
    from torch import randn
    generator_z = Generator_z()
    x = randn([32, 3, 32, 32])
    z = generator_z(x)
    discriminator_x = Discriminator_x()
    discriminator_z = Discriminator_z()
    discriminator_x_z = Discriminator_x_z(10)
    print(discriminator_x_z(discriminator_x(x), discriminator_z(z)))
    # print(discriminator_z(z).shape)
