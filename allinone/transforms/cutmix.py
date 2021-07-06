import numpy as np
import torch
from typing import Optional, Tuple
from . import TRANSFORM_REGISTRY, mixup


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(input: torch.Tensor, gamma: float, indices: torch.Tensor):
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), gamma)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    gamma = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                 (input.size()[-1] * input.size()[-2]))
    return input, gamma


@TRANSFORM_REGISTRY.register
def cutmix_for_one_hot(input: torch.Tensor,
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
    mix_input, gamma = cutmix(input, gamma, indices)
    return mix_input, mixup(target, gamma, indices)


@TRANSFORM_REGISTRY.register
def cutmix_for_integer(input: torch.Tensor,
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
    mix_input, gamma = cutmix(input, gamma, indices)
    return mix_input, gamma, target, target[indices]
