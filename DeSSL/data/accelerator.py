from . import ACCELERATOR_REGISTRY
import os
import torch
from torchvision.datasets import MNIST, FashionMNIST
from inspect import signature
from typing import Type

__all__ = [
    'accelerator',
    'accelerated_mnist',
    'accelerated_fashionmnist',
]


def accelerator(device: torch.device,
                mnist: Type[MNIST],
                mean: float = 0.5,
                std: float = 1.) -> Type[MNIST]:
    r'''
    An accelerator for MNIST-like dataset.

    Note:
        The accelerator will transmit all data to a device(e.g. GPU) in initialization. As a result, no subprocess use for data loading. 
        ``num_workers=0`` is necessary to dataloader.

    Example:
        >>> from DeSSL.data import accelerator
        >>> from torchvision.datasets import MNIST
        >>> FastMNIST = accelerator(torch.device('cuda:0'), MNIST, 0.1307, 0.3081)

    Args:
        device: the device on which a dataset will be allocated.
        mnist: a MNIST-like dataset.
        mean: The mean value of dataset.
        std: The std value of dataset.

    Returns:
        An AcceleratedMNIST class.
    '''
    accelerated_mnist = type('Accelerated' + mnist.__name__, (mnist,), {})

    def __init__(self, *args, **kwargs):
        super(accelerated_mnist, self).__init__(*args, **kwargs)
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(mean).div_(std)
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, mnist.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, mnist.__name__, 'processed')

    accelerated_mnist.__init__ = __init__
    accelerated_mnist.__getitem__ = __getitem__
    accelerated_mnist.raw_folder = raw_folder
    accelerated_mnist.processed_folder = processed_folder

    return accelerated_mnist


def accelerated_mnist(*args, **kwargs) -> Type[MNIST]:
    r'''
    The partial function is an initialization of AcceleratedMNIST which has ``mnist=MNIST``, ``mean=0.1307``, ``std=0.3081`` supplied.

    Example:
        >>> from DeSSL.data import accelerated_mnist
        >>> from torchvision.datasets import MNIST
        >>> FastMNIST = accelerated_mnist(torch.device('cuda:0'))
    or:

        >>> from DeSSL.data import ACCELERATOR_REGISTRY
        >>> FastMNIST = ACCELERATOR_REGISTRY('mnist')(torch.device('cuda:0'))
    '''
    name = list(signature(accelerator).parameters.keys())

    kwargs = {**dict(zip(name, args)),
              **kwargs,
              'mnist': MNIST,
              'mean': 0.1307,
              'std': 0.3081,
              }
    return accelerator(**kwargs)


def accelerated_fashionmnist(*args, **kwargs) -> Type[MNIST]:
    r'''
    The partial function is an initialization of AcceleratedMNIST which has ``mnist=FashionMNIST``, ``mean=0.286``, ``std=0.352`` supplied.
    '''
    name = list(signature(accelerator).parameters.keys())

    kwargs = {**dict(zip(name, args)),
              **kwargs,
              'mnist': FashionMNIST,
              'mean': 0.286,
              'std': 0.352,
              }
    return accelerator(**kwargs)


ACCELERATOR_REGISTRY.register_from_dict({
    'mnist': accelerated_mnist,
    'fashionmnist': accelerated_fashionmnist,
})
