import torch
from torch import nn
from typing import Optional, Tuple
from . import TRANSFORM_REGISTRY

__all__ = ['mixup_for_one_hot', 'mixup_for_integer', 'MixLoss']


def mixup(input: torch.Tensor,
          gamma: float,
          indices: torch.Tensor
          ) -> torch.Tensor:
    """ mixup: Beyond Empirical Risk Minimization

    """

    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


@TRANSFORM_REGISTRY.register
def mixup_for_one_hot(input: torch.Tensor,
                      target: torch.Tensor,
                      gamma: float,
                      indices: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    mixup: Beyond Empirical Risk Minimization
    """
    if input.device != target.device:
        raise RuntimeError("Device mismatch!")
    if indices is None:
        indices = torch.randperm(input.size(
            0), device=input.device, dtype=torch.long)
    return mixup(input, gamma, indices), mixup(target, gamma, indices)


@TRANSFORM_REGISTRY.register
def mixup_for_integer(input: torch.Tensor,
                      target: torch.Tensor,
                      gamma: float,
                      indices: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    mixup_for_integer: Beyond Empirical Risk Minimization
    """
    if input.device != target.device:
        raise RuntimeError("Device mismatch!")
    if indices is None:
        indices = torch.randperm(input.size(
            0), device=input.device, dtype=torch.long)
    return mixup(input, gamma, indices), target, target[indices]


class MixLoss(nn.Module):
    def __init__(self):
        super(MixLoss, self).__init__()
        self.criterion = nn.functional.cross_entropy

    def forward(self, mix_input, mix_target):
        log_prob = nn.functional.log_softmax(mix_input)
        return (log_prob * mix_target).sum(dim=1).mean().neg()

    def forward(self, mix_input, gamma, target, perm_target):
        return gamma * self.criterion(mix_input, target) + (1. - gamma) * self.critertion(mix_input, perm_target)
