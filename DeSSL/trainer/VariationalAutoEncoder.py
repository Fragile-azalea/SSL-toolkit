from homura.metrics import accuracy
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from homura.reporters import _ReporterBase
from torch.optim import Optimizer
from torch import Tensor, nn, cat
import torch
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from homura.trainers import TrainerBase
import torch
from . import SEMI_TRAINER_REGISTRY
from homura.utils.containers import TensorTuple
from torch.nn import functional as F
from .utils import unroll
from torchvision.utils import make_grid

__all__ = ['VariationalAutoEncoder']


def _loss(output, target, mu, logvar, loss_f):
    B, C, H, W = list(target.shape)
    loss = loss_f(output, target.view(B, -1), reduction='none').sum(dim=1)
    loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return loss


# @ SEMI_TRAINER_REGISTRY.register
# class VariationalAutoEncoder(TrainerBase):
#     r'''
#     Reproduced Code based on `Semi-Supervised Learning with Deep Generative Models <https://arxiv.org/abs/1406.5298>`_.

#     Args:
#         encoder_model: An inference model to classify.
#         vae_model: An vae model to reconstruct images as an auxiliary task.
#         optimizer: The optimizer of trainer.
#         loss_f: The classfication loss of trainer.
#     '''

#     def __init__(self,
#                  encoder_model: nn.Module,
#                  vae_model: nn.Module,
#                  optimizer: Optimizer,
#                  loss_f: Callable,
#                  *,
#                  reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
#                  scheduler: Optional[Scheduler] = None,
#                  verb=True,
#                  use_cudnn_benchmark=True,
#                  report_accuracy_topk: Optional[int or List[int]] = None,
#                  **kwargs):
#         model = {'encoder': encoder_model, 'vae': vae_model}
#         super(VariationalAutoEncoder, self).__init__(model, optimizer, loss_f, reporters=reporters,
#                                                      scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
#         if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
#             report_accuracy_topk = [report_accuracy_topk]
#         self._report_topk = report_accuracy_topk

#     def data_preprocess(self, data: Tuple[Tensor, ...]) -> (Tuple[Tensor, ...], int):
#         r'''
#         The labeled data and unlabeled data are combined if they were split previously.
#         '''
#         data = unroll(data)
#         return TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking), data[0].size(0)

#     def iteration(self, data: Tuple[Tensor, Tensor]) -> None:
#         if self.is_train:
#             label_data, target, unlabel_data, _ = data
#             input = cat((label_data, unlabel_data), dim=0)
#             target = cat((target, _), dim=0)
#             output_target_logit = self.model['encoder'](input)
#             loss = F.cross_entropy(output_target_logit, target)
#             log_output_target = F.log_softmax(output_target_logit, dim=1)
#             sample_target = F.gumbel_softmax(log_output_target, tau=1)
#             B, C = list(sample_target.shape)
#             tmp_target = target.where(
#                 target >= 0, output_target_logit.size(1) * torch.ones_like(target))
#             sample_and_label_target = torch.zeros(
#                 (B, C+1), device=tmp_target.get_device()).scatter_(1, tmp_target.view(-1, 1), 1)
#             sample_and_label_target = sample_and_label_target[:, :-1]
#             sample_and_label_target[target < 0] = sample_target[target < 0]
#             # for idx, classes in enumerate(target):
#             #     if classes >= 0:
#             #         sample_and_label_target[idx, classes] = 1.
#             #     else:
#             #         sample_and_label_target[idx, :] = sample_target[idx,:]
#             output, mu, logvar = self.model['vae'](
#                 input, sample_and_label_target)
#             # loss += 0.001 * _loss(output, input, mu, logvar, self.loss_f)
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#         else:
#             input, target = data
#             output_target_logit = self.model['encoder'](input)
#             loss = F.cross_entropy(output_target_logit, target)
#             self.reporter.add('accuracy', accuracy(
#                 output_target_logit, target))
#         if self._is_debug and torch.isnan(loss):
#             self.logger.warning("loss is NaN")

#         self.reporter.add('loss', loss.detach_())

#     def state_dict(self
#                    ) -> Mapping[str, Any]:

#         return {'model': self.accessible_model.state_dict(),
#                 'optim': self.optimizer.state_dict(),
#                 'scheduler': self.scheduler.state_dict(),
#                 'epoch': self.epoch,
#                 'use_sync_bn': self._use_sync_bn
#                 }

#     def load_state_dict(self,
#                         state_dict: Mapping[str, Any]
#                         ) -> None:
#         self.accessible_model.load_state_dict(state_dict['model'])
#         self.optimizer.load_state_dict(state_dict['optim'])
#         self.scheduler.load_state_dict(state_dict['scheduler'])
#         self.scheduler.last_epoch = state_dict['epoch']
#         self._epoch = state_dict['epoch']
#         self._use_sync_bn = state_dict['use_sync_bn']

@ SEMI_TRAINER_REGISTRY.register
class VariationalAutoEncoder(TrainerBase):
    r'''
    Reproduced Code based on `Semi-Supervised Learning with Deep Generative Models <https://arxiv.org/abs/1406.5298>`_.

    Args:
        encoder_model: An inference model to classify.
        vae_model: An vae model to reconstruct images as an auxiliary task.
        optimizer: The optimizer of trainer.
        loss_f: The classfication loss of trainer.
    '''

    def __init__(self,
                 encoder_model: nn.Module,
                 vae_model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):
        model = {'encoder': encoder_model, 'vae': vae_model}
        super(VariationalAutoEncoder, self).__init__(model, optimizer, loss_f, reporters=reporters,
                                                     scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def data_preprocess(self, data: Tuple[Tensor, ...]) -> (Tuple[Tensor, ...], int):
        r'''
        The labeled data and unlabeled data are combined if they were split previously.
        '''
        data = unroll(data)
        return TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking), data[0].size(0)

    def iteration(self, data: Tuple[Tensor, Tensor]) -> None:
        if self.is_train:
            l_data, target, u_data, _ = data
            # ===== working for labeled data =====
            l_output = self.model['encoder'](l_data)
            C = l_output.size(1)
            loss = F.cross_entropy(l_output, target) * 600
            target = F.one_hot(target, C)
            l_output, mu, logvar = self.model['vae'](l_data, target)
            loss += _loss(l_output, l_data, mu,
                          logvar, self.loss_f).mean()
            B = u_data.size(0)
            # ===== working for unlabeled data =====
            u_prob = F.softmax(self.model['encoder'](u_data), dim=1)
            loss -=  \
                (u_prob * torch.log(u_prob + 1e-7)).sum(dim=1).mean()
            for i in range(C):
                u_target = torch.zeros_like(u_prob)
                u_target[:, i] = 1
                u_output, mu, logvar = self.model['vae'](u_data, u_target)
                loss += (u_prob[:, i] * _loss(u_output, u_data,
                                              mu, logvar, self.loss_f)).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            input, target = data
            output_target_logit = self.model['encoder'](input)
            loss = F.cross_entropy(output_target_logit, target)
            self.reporter.add('accuracy', accuracy(
                output_target_logit, target))
        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")

        self.reporter.add('loss', loss.detach_())

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


if __name__ == "__main__":
    import torch
    a = nn.Sequential(
        nn.Conv2d(3, 4, 3),
        nn.BatchNorm2d(4, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(4, 5, 3, padding=1),
        nn.BatchNorm2d(5, affine=False))
    a.to('cuda')
    a, b, c = convert_net_to_ladder_network(
        a, [a[1], a[4]], [5, 1], [nn.ConvTranspose2d(5, 4, 1)])
    a.train()
    a(torch.randn(32, 3,  6, 6, device='cuda'))
    enable_noise(b)
    a(torch.randn(32, 3,  6, 6, device='cuda'))
    c(b[-1].noise_h)
    print(get_loss_d(c, [1.0, 1.0]))
