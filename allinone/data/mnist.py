import os
import torch
from torchvision.datasets import MNIST

__all__ = [
    'accelerator',
    'accelerated_mnist',
]


def accelerator(device: torch.device,
                mnist: MNIST,
                mean: float = 0.5,
                std: float = 1.) -> MNIST:
    r'''
    An accelerator for MNIST-like dataset.

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


def accelerated_mnist(*args, **kwargs) -> MNIST:
    name = ['device',
            'mnist',
            'mean',
            'std']

    kwargs = {**dict(zip(name, args)),
              **kwargs,
              'mnist': MNIST,
              'mean': 0.1307,
              'std': 0.3081,
              }
    return accelerator(**kwargs)
