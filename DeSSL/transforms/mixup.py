import torch
from torch import nn
from typing import Optional, Tuple, Union
from . import TRANSFORM_REGISTRY


__all__ = ['mixup_for_one_hot', 'mixup_for_integer',
           'OneHotMixLoss', 'IntegerMixLoss', 'one_hot_mix_loss']


def mixup(input: torch.Tensor,
          gamma: Union[float, torch.Tensor],
          indices: torch.Tensor
          ) -> torch.Tensor:

    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    if isinstance(gamma, torch.Tensor):
        shape = [-1, ] + [1, ] * (len(input.shape) - 1)
        gamma = gamma.view(shape)

    perm_input = input[indices]
    return input * gamma + perm_input * (1 - gamma)


@TRANSFORM_REGISTRY.register
def mixup_for_one_hot(input: torch.Tensor,
                      target: torch.Tensor,
                      gamma: Union[float, torch.Tensor],
                      indices: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    `Mixup <https://arxiv.org/abs/1710.09412>`_ Augmentation for Tensor.

    Args:
        input: input tensor of data. 
        target: one-hot label of data.
        gamma: The interpolation coefficient.
        indices: indices[i] denotes the mixup target of data i. If ``indices=None``, then mixup generates indices by torch.perm. 

    Example:
        >>> from allinone.transforms import mixup_for_one_hot
        >>> input = torch.randn([256, 3, 64, 64])
        >>> target = torch.rand([256, 10])
        >>> gamma = 0.995
        >>> mix_input, mix_target = mixup_for_one_hot(input, target, gamma)

    Returns:
        Mixed tensor and mixed targets.
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
    `Mixup <https://arxiv.org/abs/1710.09412>`_ Augmentation for Tensor.

    Args:
        input: input tensor of data. 
        target: integer label of data.
        gamma: The interpolation coefficient.
        indices: indices[i] denotes the mixup target of data i. If ``indices=None``, then mixup generates indices by torch.perm. 

    Example:
        >>> from allinone.transforms import mixup_for_integer
        >>> input = torch.randn([256, 3, 64, 64])
        >>> target = torch.rand([256, 10]).argmax(dim=-1)
        >>> gamma = 0.995
        >>> mix_input, target, perm_target = mixup_for_integer(input, target, gamma)

    Returns:
        Mixed tensor, targets and perm_targets.
    """
    if input.device != target.device:
        raise RuntimeError("Device mismatch!")
    if indices is None:
        indices = torch.randperm(input.size(
            0), device=input.device, dtype=torch.long)
    return mixup(input, gamma, indices), target, target[indices]


class OneHotMixLoss(nn.Module):
    r"""
    Creates a criterion that measures the cross entropy loss between each element in
    the input and **one-hot** target.
    """

    def __init__(self):
        super(OneHotMixLoss, self).__init__()
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, mix_input: torch.Tensor, mix_target: torch.Tensor) -> torch.Tensor:
        r"""
            Args:
                mix_input: The input tensor.
                mix_target: The target tensor.

            Returns:
                The cross entropy loss.
        """
        log_prob = self.logSoftmax(mix_input)
        return (log_prob * mix_target).sum(dim=1).mean().neg()


def one_hot_mix_loss(mix_input, mix_target, is_logit=True):
    if is_logit:
        log_prob = nn.functional.log_softmax(mix_input, dim=-1)
    else:
        log_prob = mix_input.log()
    return (log_prob * mix_target).sum(dim=1).mean().neg()


class IntegerMixLoss(nn.Module):
    r"""
    Creates a criterion that measures the cross entropy loss between each element in
    the input and **integer** target.
    """

    def __init__(self):
        super(IntegerMixLoss, self).__init__()
        self.crl = nn.CrossEntropyLoss()

    def forward(self, mix_input: torch.Tensor, gamma: float, target: torch.LongTensor, perm_target: torch.LongTensor):
        r"""
            Args:
                mix_input: The input tensor.
                gamma: The interpolation coefficient.
                target: The target tensor.
                perm_target: The perm_target tensor.

            Returns:
                The cross entropy loss.
        """
        return gamma * self.crl(mix_input, target) + (1. - gamma) * self.crl(mix_input, perm_target)
