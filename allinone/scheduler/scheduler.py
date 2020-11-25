from . import SCHEDULER_REGISTRY
from dataclasses import dataclass
import math
from typing import Callable


class SchedulerBase:
    def step(self):
        self.epoch += 1


@SCHEDULER_REGISTRY.register
@dataclass
class Linear(SchedulerBase):
    initial: float = 0.
    speed: float = 0.
    epoch: int = 0

    def __call__(self):
        return self.initial + self.speed * self.epoch


@SCHEDULER_REGISTRY.register
@dataclass
class Identity(SchedulerBase):
    value: float = 0
    epoch: int = 0

    def __call__(self):
        return self.value


@SCHEDULER_REGISTRY.register
@dataclass
class GaussianKernel(SchedulerBase):
    mu: float = 0.
    sigma: float = 1.
    epoch: int = 0

    def __call__(self):
        return math.e ** ((self.epoch - self.mu) ** 2 / self.sigma)


@SCHEDULER_REGISTRY.register
@dataclass
class Lambda(SchedulerBase):
    lamb: Callable
    epoch: int = 0

    def __call__(self):
        return self.lamb(self.epoch)
