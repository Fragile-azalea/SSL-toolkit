from . import SEMI_DATASET_REGISTRY
from homura.vision.data import VisionSet
from torch.utils.data import DataLoader, Sampler
from dataclasses import dataclass
from torch import randperm, ones_like
from random import shuffle
from itertools import chain
from typing import Callable

__all__ = ['Mix', 'Split']


@SEMI_DATASET_REGISTRY.register
@dataclass
class Mix(VisionSet):
    '''
    The Datasets that the labeled and unlabeled data mix.
    The code based on `VisionSet <https://github.com/moskomule/homura/blob/master/homura/vision/data/datasets.py>`_ rewrites the get_dataset method.

    Note:
        The transform of labeled and unlabeled data should be the same.

    Args:
        semi_size: The size of unlabeled data.

    Example:
        >>> mnist = Mix(MNIST, args.dataset, 10, [tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))], semi_size=args.semi_size)
        >>> label_and_unlabel_loader, test_loader, num_classes = mnist(args.batch_size, num_workers=args.num_workers, return_num_classes=True)

    '''
    semi_size: int = 0

    def get_dataset(self, *args, **kwargs):
        '''
        Get Dataset and then drop some label in train_set out, being labeled and unlabeled datasets.

        Return:
            (train_set, test_set) 
        '''
        dataset_list = super(Mix, self).get_dataset(*args, **kwargs)
        if len(dataset_list) == 3:
            train_set, test_set, num_classes = dataset_list
        else:
            train_set, test_set = dataset_list
        indices = randperm(len(train_set.targets))[:self.semi_size]
        assert len(train_set.targets) >= self.semi_size, 'len(train_set):%d < semi_size:%d' % (
            len(train_set.targets), self.semi_size)
        for idx in indices:
            train_set.targets[idx] = -100
        ret = [train_set, test_set]
        if len(dataset_list) == 3:
            ret.append(num_classes)
        return ret


@SEMI_DATASET_REGISTRY.register
@dataclass
class Split(VisionSet):
    '''
    The Datasets that the labeled and unlabeled data mix.
    The code based on `VisionSet <https://github.com/moskomule/homura/blob/master/homura/vision/data/datasets.py>`_ rewrites the get_dataset and get_dataloader method.

    Args:
        semi_size: The size of unlabeled data.
        unlabel_transform: The transform for unlabel data.

    Note:
        The length of unlabel_dataloader is :math:`\infty`.
        If unlabel_transform is None, the transform of labeled and unlabeled data will be the same.

    Example:
        >>> mnist = Split(MNIST, args.dataset, 10, [tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))], semi_size=50000)
        >>> train_loader, test_loader, inf_unlabel_loader, num_classes = mnist(args.batch_size, num_workers=args.num_workers, return_num_classes=True)

    '''
    semi_size: int = 0
    unlabel_transform: Callable = None

    def get_dataset(self, *args, **kwargs):
        '''
        The valid dataset is transformed into the unlabeled datasets.

        Return:
            (train_set, test_set, unlabel_set)
        '''
        args = (*args[0:2], self.semi_size, *args[3:])
        train_set, test_set, val_set = super(
            Split, self).get_dataset(*args, **kwargs)
        if self.unlabel_transform is not None:
            val_set.transform = self.unlabel_transform
        else:
            val_set.transform = train_set.transform
        for idx in range(len(val_set.targets)):
            val_set.targets[idx] = -100
        return train_set, test_set, val_set

    def get_dataloader(self, *args, unlabel_batch_size: int = 0, **kwargs):
        '''
        Get loaders of the training, testing, and unlabeled datasets.

        Args:
            unlabel_batch_size: The batch size of unlabel loaders. If :code:`unlabel_batch_size` is zero, the size of the unlabeled loader will be the same as the size of the training loader.

        Return:
            (train_loader, test_loader, inf_unlabel_loader, Optional[num_classes])
        '''
        dataLoader_list = super(Split, self).get_dataloader(*args, **kwargs)
        if len(dataLoader_list) == 4:
            train_dataloader, test_dataloader, val_dataloader, num_classes = dataLoader_list
        else:
            train_dataloader, test_dataloader, val_dataloader = dataLoader_list

        if unlabel_batch_size > 0:
            unlabel_dataloader = DataLoader(val_dataloader.dataset, unlabel_batch_size, sampler=val_dataloader.sampler, drop_last=val_dataloader.drop_last,
                                            num_workers=val_dataloader.num_workers, pin_memory=val_dataloader.pin_memory, collate_fn=val_dataloader.collate_fn)
        else:
            unlabel_dataloader = DataLoader(val_dataloader.dataset, train_dataloader.batch_size, sampler=val_dataloader.sampler, drop_last=val_dataloader.drop_last,
                                            num_workers=val_dataloader.num_workers, pin_memory=val_dataloader.pin_memory, collate_fn=val_dataloader.collate_fn)
        del val_dataloader
        ret = [train_dataloader, test_dataloader, unlabel_dataloader]
        print('no inf now!!!')
        if len(dataLoader_list) == 4:
            ret.append(num_classes)
        return ret

    __call__ = get_dataloader


class SemiBatchSampler(Sampler):
    def __init__(self,
                 length: int,
                 unlabel_indices: list,
                 batch_size: int,
                 unlabel_batch_size: int):
        super(SemiBatchSampler, self).__init__()
        assert unlabel_batch_size <= batch_size
        assert length >= unlabel_indices
        self.label_indices = sorted(set(range(length)) - set(unlabel_indices))
        self.label_batch_size = batch_size - unlabel_batch_size
        self.unlabel_indices = unlabel_indices
        self.unlabel_batch_size = unlabel_batch_size

    def __iter__(self):
        shuffle(self.label_indices)

        def infinite_unlabel(unlabel_indices):
            while True:
                yield shuffle(unlabel_indices)

        label_iter = [iter(self.label_indices)] * self.label_batch_size
        unlabel_iter = [
            iter(chain(infinite_unlabel(unlabel_indices)))] * self.unlabel_batch_size
        return (label + unlabel for (label, unlabel) in zip(*label_iter, *unlabel_iter))

    def __len__(self):
        return len(self.label_indices) // self.label_batch_size
