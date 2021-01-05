from homura.metrics import accuracy
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from homura.reporters import _ReporterBase
from torch.optim import Optimizer
from torch import Tensor, nn, cat
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from homura.trainers import TrainerBase
import torch
from . import SEMI_TRAINER_REGISTRY
from homura.utils.containers import TensorTuple

__all__ = ['Ladder']


class _MemoryBatchNorm2d(nn.Module):
    def __init__(self, module, sigma, is_last):
        super(_MemoryBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(
            module.num_features, module.eps, module.momentum, False, module.track_running_stats)
        self.bn.running_mean = module.running_mean
        self.bn.running_var = module.running_var
        self.bn.num_batches_tracked = module.num_batches_tracked
        self.affine = module.affine
        self.is_noise = False
        self.sigma = sigma
        self.is_last = is_last
        if self.affine:
            with torch.no_grad():
                self.weight = module.weight
                self.bias = module.bias

    def forward(self, input):
        if self.training:
            if self.is_noise:
                self.batch_mean = torch.mean(input, dim=[0, 2, 3], keepdim=True)
                self.batch_std = torch.std(input, dim=[0, 2, 3], keepdim=True)
                input = input - self.batch_mean
                input = input / (self.batch_std + 1e-7)
                output = input + torch.randn_like(input) * self.sigma
                self.noise_z = output
            else:
                output = self.bn(input)
                self.z = output
            if self.affine:
                output = self.weight.view(
                    1, -1, 1, 1).expand_as(output) * output
                output = self.bias.view(
                    1, -1, 1, 1).expand_as(output) + output
            if self.is_last and self.is_noise:
                self.noise_h = output
        else:
            output = self.bn(input)
            if self.affine:
                output = self.weight.view(
                    1, -1, 1, 1).expand_as(output) * output
                output = self.bias.view(
                    1, -1, 1, 1).expand_as(output) + output
        return output


class _Decoder(nn.Module):
    def __init__(self, mbn):
        super(_Decoder, self).__init__()
        self.mbn = mbn
        num_features = mbn.bn.num_features
        self.a0 = nn.Conv2d(num_features, num_features, 1,
                            groups=num_features, bias=False)
        self.a1 = nn.Conv2d(num_features, num_features, 1, groups=num_features)
        self.a2 = nn.Conv2d(num_features, num_features, 1, groups=num_features)
        self.a3 = nn.Conv2d(num_features, num_features, 1,
                            groups=num_features, bias=False)
        self.a4 = nn.Conv2d(num_features, num_features, 1, groups=num_features)
        self.a5 = nn.Conv2d(num_features, num_features, 1, groups=num_features)

    def forward(self, input):
        # print(input.shape, self.mbn.z.shape)
        batch_mean = torch.mean(input, dim=[0, 2, 3], keepdim=True)
        batch_std = torch.std(input, dim=[0, 2, 3], keepdim=True)
        input = input - batch_mean
        input = input / (batch_std + 1e-7)
        mu = self.a0(torch.sigmoid(self.a1(input))) + self.a2(input)
        sigma = self.a3(torch.sigmoid(self.a4(input))) + self.a5(input)
        self.denoise_z = (self.mbn.noise_z - mu) * sigma + mu
        return self.denoise_z

    def get_loss_d(self, lam):
#         tmp = self.mbn.bn(self.denoise_z)
#         tmp = self.denoise_z - self.mbn.batch_mean.view(1, -1, 1, 1)
#         tmp /= self.mbn.batch_std.view(1, -1, 1, 1) + 1e-7
        denoise_z = self.denoise_z - self.mbn.bn.running_mean.view(1, -1, 1, 1)
        denoise_z /= (self.mbn.bn.running_var.view(1, -1, 1, 1) + 1e-7) ** 0.5
        denoise_z = denoise_z.flatten(start_dim=1)
        clean_z = self.mbn.z.flatten(start_dim=1)
        lam /= self.denoise_z.shape[1]
        return torch.mean(lam * torch.norm(denoise_z - clean_z, p=2, dim=1))


def _convert_bn_to_memory_bn(module, bn_list, sigma_list):
    module_output = module
    if module in bn_list:
        idx = bn_list.index(module)
        is_last = idx + 1 == len(bn_list)
        module_output = _MemoryBatchNorm2d(module, sigma_list[idx], is_last)
    for name, child in module.named_children():
        module_output.add_module(
            name, _convert_bn_to_memory_bn(child, bn_list, sigma_list))
    del module
    return module_output


def _build_decoder(mbn_list, v_list):
    reverse_list = mbn_list[::-1]
    decoder_list = [_Decoder(reverse_list[0])]
    for mbn, v in zip(reverse_list[1:], v_list[::-1]):
        decoder_list.append(v)
        decoder_list.append(_Decoder(mbn))
    decoder = nn.Sequential(*decoder_list)
    return decoder


def _convert_net_to_ladder_network(backbone, bn_list, sigma_list, v_list):

    assert len(bn_list) == len(sigma_list)
    for sigma in sigma_list:
        assert sigma > 0
    for bn in bn_list:
        assert isinstance(bn, nn.BatchNorm2d)
    backbone = _convert_bn_to_memory_bn(backbone, bn_list, sigma_list)
    mbn_list = list(filter(lambda module: isinstance(
        module, _MemoryBatchNorm2d), backbone.modules()))
    decoder = _build_decoder(mbn_list, v_list)
    return backbone, mbn_list, decoder


def _enable_noise(mbn_list):
    for module in mbn_list:
        module.is_noise = True


def _disable_noise(mbn_list):
    for module in mbn_list:
        module.is_noise = False


def _get_loss_d(decoder, lam_list):
    idx = -1
    for module in decoder:
        if isinstance(module, _Decoder):
            loss = module.get_loss_d(
                lam_list[idx]) + (0. if idx == -1 else loss)
            idx -= 1
    return loss


class LadderTrainer(TrainerBase):
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 bn_list: List[nn.BatchNorm2d],
                 sigma_list: List[float],
                 v_list: List[nn.Module],
                 lam_list: List[float],
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):

        backbone, mbn_list, decoder = _convert_net_to_ladder_network(
            model, bn_list, sigma_list, v_list)
        model = {'backbone': backbone, 'decoder': decoder}
        self.mbn_list = mbn_list
        self.lam_list = lam_list

        super(LadderTrainer, self).__init__(model, optimizer, loss_f, reporters=reporters,
                                            scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)

        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def iteration(self,
                  data: Tuple[Tensor, Tensor]
                  ) -> None:
        input, target = data
        _disable_noise(self.mbn_list)
        output = self.model['backbone'](input)
        if self.is_train:
            _enable_noise(self.mbn_list)
            output = self.model['backbone'](input)
            self.model['decoder'](self.mbn_list[-1].noise_h)
            loss = _get_loss_d(self.model['decoder'], self.lam_list)
            if self.is_supervised:
                loss += self.loss_f(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
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


@ SEMI_TRAINER_REGISTRY.register
class Ladder(TrainerBase):
    r'''
    Reproduced trainer based on `Semi-Supervised Learning with Ladder Networks <https://arxiv.org/abs/1507.02672>`_.

    Args:
        model: The backbone model of trainer.
        optimizer: The optimizer of trainer. 
        loss_f: The classfication loss of trainer.
        bn_list: The order list of nn.BatchNorm2d, which remembers statistics for recovering signal. 
        sigma_list: The order list of positive float, which means the :math:`\sigma` of gaussian noise.
        v_list: The order list of nn.Module, which transfroms the input signal. Corresponding to :math:`V^i` in the original paper.
        lam_list: The order list of regularization coefficient, which describes how important to recovery signals.  Corresponding to :math:`\lambda^i` in the original paper.
    '''

    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 loss_f: Callable,
                 bn_list: List[nn.BatchNorm2d],
                 sigma_list: List[float],
                 v_list: List[nn.Module],
                 lam_list: List[float],
                 *,
                 reporters: Optional[_ReporterBase or List[_ReporterBase]] = None,
                 scheduler: Optional[Scheduler] = None,
                 verb=True,
                 use_cudnn_benchmark=True,
                 report_accuracy_topk: Optional[int or List[int]] = None,
                 **kwargs):

        backbone, mbn_list, decoder = _convert_net_to_ladder_network(
            model, bn_list, sigma_list, v_list)
        model = {'backbone': backbone, 'decoder': decoder}
        self.mbn_list = mbn_list
        self.lam_list = lam_list

        super(Ladder, self).__init__(model, optimizer, loss_f, reporters=reporters,
                                     scheduler=scheduler, verb=verb, use_cudnn_benchmark=use_cudnn_benchmark, **kwargs)
        if report_accuracy_topk is not None and not isinstance(report_accuracy_topk, Iterable):
            report_accuracy_topk = [report_accuracy_topk]
        self._report_topk = report_accuracy_topk

    def data_preprocess(self, data: Tuple[Tensor, ...]) -> (Tuple[Tensor, ...], int):
        r'''
        The labeled data and unlabeled data are combined if they were split previously.
        '''
        if isinstance(data, tuple):
            data = (cat((data[0][0], data[1][0]), dim=0),
                    cat((data[0][1], data[1][1]), dim=0))
        return TensorTuple(data).to(self.device, non_blocking=self._cuda_nonblocking), data[0].size(0)

    def iteration(self, data: Tuple[Tensor, Tensor]) -> None:
        input, target = data
        _disable_noise(self.mbn_list)
        output = self.model['backbone'](input)
        if self.is_train:
            _enable_noise(self.mbn_list)
            output = self.model['backbone'](input)
            self.model['decoder'](self.mbn_list[-1].noise_h)
            loss = _get_loss_d(self.model['decoder'], self.lam_list)
            loss += self.loss_f(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            loss = self.loss_f(output, target)
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
