import os
import gzip
import numpy as np
import torch
from torchvision.datasets import MNIST
from functools import partial
from torch.utils.data import Dataset
from torchvision import transforms as tf
from .dataset import SemiDataset

__all__ = [
    'semi_mnist',
    'semi_10_mnist',
    'semi_50_mnist',
    'semi_mnist_32x32',
    'semi_10_mnist_32x32',
    'semi_50_mnist_32x32',
    'semi_mnist_gz',
    'semi_10_mnist_gz',
    'semi_50_mnist_gz',
    'semi_mnist_gz_32x32',
    'semi_10_mnist_gz_32x32',
    'semi_50_mnist_gz_32x32',
    'SEMI_MNIST',
    'accelerator',
]


device = torch.device('cuda')


def accelerator(device: torch.device,
                mnist: MNIST,
                mean: float = 0.5,
                std: float = 1.) -> MNIST:
    accelerated_mnist = type('Accelerated' + mnist.__name__, (mnist,), {})

    def __init__(self, *args, **kwargs):
        super(accelerated_mnist, self).__init__(*args, **kwargs)
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(mean).div_(std)
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
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


class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        print(args, kwargs)
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)
        print('mnist line 40 is delete!')
        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


# semi_mnist = partial(SemiDataset,
#                      dataset=MNIST,
#                      num_classes=10,
#                      label_transform=tf.Compose(
#                          [tf.ToTensor(), ]),
#                      unlabel_transform=tf.Compose(
#                          [tf.ToTensor(), ]),
#                      test_transform=tf.Compose(
#                          [tf.ToTensor(), ]),
#                      )

semi_mnist = partial(SemiDataset,
                     dataset=FastMNIST,
                     num_classes=10,
                     )

semi_10_mnist = partial(semi_mnist, num_labels_per_class=10)
semi_50_mnist = partial(semi_mnist, num_labels_per_class=50)
semi_mnist_32x32 = partial(SemiDataset,
                           dataset=MNIST,
                           num_classes=10,
                           label_transform=tf.Compose(
                               [tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
                           unlabel_transform=tf.Compose(
                               [tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
                           test_transform=tf.Compose(
                               [tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
                           )
semi_10_mnist_32x32 = partial(semi_mnist_32x32, num_labels_per_class=10)
semi_50_mnist_32x32 = partial(semi_mnist_32x32, num_labels_per_class=50)


def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)


class DealDataset(Dataset):
    def __init__(
            self,
            folder,
            train=True,
            transform=None,
            label_transform=None,
            download=False):
        assert download == False, 'Waiting for developing.'
        if train:
            data_name, label_name = "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
        else:
            data_name, label_name = "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
        (train_set, train_labels) = load_data(folder, data_name, label_name)
        self.train_set = train_set
        self.targets = train_labels
        self.transform = transform

    def __getitem__(self, index):

        img, target = self.train_set[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)


semi_mnist_gz = partial(SemiDataset,
                        dataset=DealDataset,
                        num_classes=10,
                        label_transform=tf.Compose(
                            [tf.ToPILImage(), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
                        unlabel_transform=tf.Compose(
                            [tf.ToPILImage(), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
                        test_transform=tf.Compose(
                            [tf.ToPILImage(), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
                        )
semi_10_mnist_gz = partial(semi_mnist_gz, num_labels_per_class=10)
semi_50_mnist_gz = partial(semi_mnist_gz, num_labels_per_class=50)
semi_mnist_gz_32x32 = partial(
    SemiDataset, dataset=DealDataset, num_classes=10, label_transform=tf.Compose(
        [
            tf.ToPILImage(), tf.Resize(
                (32, 32)), tf.ToTensor(), tf.Normalize(
                    (0.1307,), (0.3081,))]), unlabel_transform=tf.Compose(
                        [
                            tf.ToPILImage(), tf.Resize(
                                (32, 32)), tf.ToTensor(), tf.Normalize(
                                    (0.1307,), (0.3081,))]), test_transform=tf.Compose(
                                        [
                                            tf.ToPILImage(), tf.Resize(
                                                (32, 32)), tf.ToTensor(), tf.Normalize(
                                                    (0.1307,), (0.3081,))]), )
semi_10_mnist_gz_32x32 = partial(semi_mnist_gz_32x32, num_labels_per_class=10)
semi_50_mnist_gz_32x32 = partial(semi_mnist_gz_32x32, num_labels_per_class=50)


SEMI_MNIST = {
    'semi_mnist': semi_mnist,
    'semi_10_mnist': semi_10_mnist,
    'semi_50_mnist': semi_50_mnist,
    'semi_mnist_32x32': semi_mnist_32x32,
    'semi_10_mnist_32x32': semi_10_mnist_32x32,
    'semi_50_mnist_32x32': semi_50_mnist_32x32,
    'semi_mnist_gz': semi_mnist_gz,
    'semi_10_mnist_gz': semi_10_mnist_gz,
    'semi_50_mnist_gz': semi_50_mnist_gz,
    'semi_mnist_gz_32x32': semi_mnist_gz_32x32,
    'semi_10_mnist_gz_32x32': semi_10_mnist_gz_32x32,
    'semi_50_mnist_gz_32x32': semi_50_mnist_gz_32x32,

}
