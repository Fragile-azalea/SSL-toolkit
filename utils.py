from homura.vision.data import VisionSet
from torch.utils.data import DataLoader, Sampler
from dataclasses import dataclass
from torch import randperm
from random import shuffle
from itertools import chain


@dataclass
class SemiVisionSet(VisionSet):
    semi_size: int = 0

    def get_dataset(self, *args, **kwargs):
        train_set, test_set, _ = super(
            SemiVisionSet, self).get_dataset(*args, **kwargs)
        indices = randperm(len(train_set.targets))[:self.semi_size]
        assert len(train_set.targets) >= self.semi_size, 'len(train_set):%d < semi_size:%d' % (
            len(train_set.targets), self.semi_size)
        for idx in indices:
            train_set.targets[idx] = -100
        return train_set, test_set, _


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


def get_dataloader(self, *args, **kwargs):
    train_dataloader, test_dataloader, val_dataloader, num_classes = self._get_dataloader(
        *args, **kwargs)
    unlabel_dataloader = DataLoader(val_dataloader.dataset, train_dataloader.batch_size, sampler=val_dataloader.sampler, drop_last=val_dataloader.drop_last,
                                    num_workers=val_dataloader.num_workers, pin_memory=val_dataloader.pin_memory, collate_fn=val_dataloader.collate_fn)
    del val_dataloader
    return train_dataloader, test_dataloader, unlabel_dataloader, num_classes


def get_dataset(self, *args, **kwargs):
    train_set, test_set, val_set = self._get_dataset(*args, **kwargs)
    if hasattr(self, 'unlabel_transform'):
        val_set.transform = self.unlabel_transform
    else:
        val_set.transform = train_set.transform
    return train_set, test_set, val_set


def change_val_to_unlabel():
    setattr(VisionSet, '_get_dataset', VisionSet.get_dataset)
    setattr(VisionSet, 'get_dataset', get_dataset)
    setattr(VisionSet, '_get_dataloader', VisionSet.get_dataloader)
    setattr(VisionSet, 'get_dataloader', get_dataloader)
    setattr(VisionSet, '__call__', get_dataloader)
