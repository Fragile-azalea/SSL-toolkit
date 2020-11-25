from . import SEMI_DATASET_REGISTRY
from homura.vision.data import VisionSet
from torch.utils.data import DataLoader, Sampler
from dataclasses import dataclass
from torch import randperm, ones_like
from random import shuffle
from itertools import chain
from typing import Callable


@SEMI_DATASET_REGISTRY.register
@dataclass
class Mix(VisionSet):
    semi_size: int = 0

    def get_dataset(self, *args, **kwargs):
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
    semi_size: int = 0
    unlabel_transform: Callable = None

    def get_dataset(self, *args, **kwargs):
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

    def get_dataloader(self, *args, unlabel_batch_size=0, **kwargs):
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
