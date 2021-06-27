from typing import Callable, Optional, Iterable, Tuple
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import CIFAR10, SVHN
from functools import partial
from torchvision import transforms as tf
import numpy as np

__all__ = ['SemiDataset', 'SemiDataLoader', 'semi_svhn']


def get_label_list(dataset: Dataset) -> (np.array):
    label_list = []
    for _, label in dataset:
        label_list.append(label)
    label_list = np.array(label_list)
    return label_list


def get_label_and_unlabel_indices(
        label_list: np.array,
        num_labels_per_class: int,
        num_classes: int) -> Tuple[list, list]:
    label_indices = []
    unlabel_indices = []
    for i in range(num_classes):
        indices = np.where(label_list == i)[0]
        np.random.shuffle(indices)
        label_indices.extend(indices[:num_labels_per_class])
        unlabel_indices.extend(indices[num_labels_per_class:])
    return label_indices, unlabel_indices


class SemiDataLoader:
    def __init__(self,
                 label_loader: DataLoader,
                 unlabel_loader: DataLoader,
                 num_iteration: int):
        self.label = label_loader
        self.unlabel = unlabel_loader
        self.num_iteration = num_iteration

    def __iter__(self):
        label_iter = iter(self.label)
        unlabel_iter = iter(self.unlabel)
        count = 0
        while count < self.num_iteration:
            count += 1
            try:
                label = label_iter.next()
            except BaseException:
                label_iter = iter(self.label)
                label = label_iter.next()

            try:
                unlabel = unlabel_iter.next()
            except BaseException:
                unlabel_iter = iter(self.unlabel)
                unlabel = unlabel_iter.next()

            yield label, unlabel

    def __len__(self):
        return self.num_iteration


class SemiDataset:
    r'''
    A class representing a semi-supervised dataset.

    Args:
        root: The root directory where the dataset exists or will be saved.
        num_labels_per_class: The number of each class.
        dataset: An instance class representing a `Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_.
        num_classes: The number of class.
        label_transform: A function/transform that takes in a labeled image and returns a transformed version. E.g, `transforms.RandomCrop <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomCrop>`_.
        unlabel_transform: A function/transform that takes in a unlabeled image and returns a transformed version. E.g, `transforms.RandomCrop <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomCrop>`_.
        test_transform: A function/transform that takes in a test image and returns a transformed version. E.g, `transforms.RandomCrop <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomCrop>`_.
        download:  If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.

    Returns:
        A semi-supervised dataset.
    '''

    def __init__(self,
                 root: str,
                 num_labels_per_class: int,
                 dataset: Dataset,
                 num_classes: int,
                 label_transform: Optional[Callable] = None,
                 unlabel_transform: Optional[Callable] = None,
                 test_transform: Optional[Callable] = None,
                 download: bool = False):
        super(SemiDataset, self).__init__()
        base_dataset = dataset(
            root, transform=label_transform, download=download)
        label_list = get_label_list(base_dataset)
        label_indices, unlabel_indices = get_label_and_unlabel_indices(
            label_list, num_labels_per_class, num_classes)
        self.length = len(label_indices) + len(unlabel_indices)
        self.label_dataset = Subset(base_dataset, label_indices)

        unlabel_base_dataset = dataset(root, transform=unlabel_transform)
        unlabel_base_dataset.targets = [-100] * self.length
#         self.unlabel_dataset = Subset(unlabel_base_dataset, unlabel_indices)
        self.unlabel_dataset = unlabel_base_dataset

        self.test_dataset = dataset(
            root, train=False, transform=test_transform, download=download)
        self.num_classes = num_classes

    def get_dataloader(self,
                       label_batch_size: int,
                       unlabel_batch_size: Optional[int] = None,
                       test_batch_size: Optional[int] = None,
                       num_iteration: Optional[int] = None,
                       shuffle: bool = True,
                       num_workers: int = 0,
                       pin_memory: bool = True,
                       drop_last: bool = False,
                       return_num_classes: bool = True):
        r'''
        Get Dataloader.

        Args:
            label_batch_size: The batch size of labeled data. 
            unlabel_batch_size: The batch size of unlabeled data. If None, use label_batch_size instead.
            test_batch_size: The batch size of testing data. If None, use label_batch_size + unlabel_batch_size instead.
        '''
        shared_dict = {'num_workers': num_workers,
                       'pin_memory': pin_memory,
                       'drop_last': drop_last
                       }
        if unlabel_batch_size is None:
            unlabel_batch_size = label_batch_size
        if test_batch_size is None:
            test_batch_size = label_batch_size + unlabel_batch_size
        label_loader = DataLoader(
            self.label_dataset,
            label_batch_size,
            shuffle=shuffle,
            **shared_dict)
        unlabel_loader = DataLoader(
            self.unlabel_dataset,
            unlabel_batch_size,
            shuffle=shuffle,
            **shared_dict)
        test_loader = DataLoader(self.test_dataset,
                                 test_batch_size, **shared_dict)
        if num_iteration is None:
            num_iteration = self.length // label_batch_size
            if not drop_last and self.length % label_batch_size != 0:
                num_iteration += 1
        train_loader = SemiDataLoader(
            label_loader, unlabel_loader, num_iteration)
        ret = [train_loader, test_loader]
        if return_num_classes:
            ret.append(self.num_classes)
        return ret

    __call__ = get_dataloader


semi_svhn = partial(SemiDataset,
                    dataset=SVHN,
                    num_classes=10,
                    label_transform=tf.Compose(
                        [tf.ToTensor(), tf.Normalize((0.4390, 0.4443, 0.4692), (0.1189, 0.1222, 0.1049))]),
                    unlabel_transform=tf.Compose(
                        [tf.ToTensor(), tf.Normalize((0.4390, 0.4443, 0.4692), (0.1189, 0.1222, 0.1049))]),
                    test_transform=tf.Compose(
                        [tf.ToTensor(), tf.Normalize((0.4390, 0.4443, 0.4692), (0.1189, 0.1222, 0.1049))]),
                    )
