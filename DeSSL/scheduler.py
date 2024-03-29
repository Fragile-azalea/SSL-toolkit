from typing import Callable
import math
from dataclasses import dataclass
from . import Registry
SCHEDULER_REGISTRY = Registry('scheduler')

__all__ = [
    'SCHEDULER_REGISTRY',
    'SchedulerBase',
    'Linear',
    'Identity',
    'GaussianKernel',
    'Lambda',
]


class SchedulerBase:
    """
    The scheduler works for providing dyniamcal (or static) hyperparmeters. (e.g., learning rate) 
    """

    def step(self) -> None:
        """
            Update Scheduler.
        """
        self.epoch += 1


@SCHEDULER_REGISTRY.register
@dataclass
class Linear(SchedulerBase):
    """
    The Linear scheduler.

    Args:
        initital: The initial value.
        speed: The difference of value following update of the scheduler.
        epoch: The number of step executed.

    Returns:
        initial + speed * epoch
    """
    initial: float = 0.
    speed: float = 0.
    epoch: int = 0

    def __call__(self):
        return self.initial + self.speed * self.epoch


@SCHEDULER_REGISTRY.register
@dataclass
class Identity(SchedulerBase):
    """
    The static identity cheduler.

    Args:
        value: The initial value.
        epoch: The number of step executed.

    Returns:
        value
    """
    value: float = 0
    epoch: int = 0

    def __call__(self):
        return self.value


@SCHEDULER_REGISTRY.register
@dataclass
class GaussianKernel(SchedulerBase):
    r"""
    The gaussian kernel scheduler.

    Args:
        mu: The :math:`\mu` of gaussian kernel.
        sigma: The :math:`\sigma` of gaussian kernel.
        epoch: The number of step executed.

    Returns:
        :math:`exp \left( {\frac{(\text{epoch} - \mu) ^ 2}{\sigma}} \right)`
    """
    mu: float = 0.
    sigma: float = 1.
    epoch: int = 0

    def __call__(self):
        return math.e ** ((self.epoch - self.mu) ** 2 / self.sigma)


@SCHEDULER_REGISTRY.register
@dataclass
class Lambda(SchedulerBase):
    r"""
    The Lambda scheduler.

    Args:
        Lamb: The Lambda function.
        epoch: The number of step executed.

    Returns:
        lamb(epoch)
    """
    lamb: Callable
    epoch: int = 0

    def __call__(self):
        return self.lamb(self.epoch)
