import logging
from typing import Callable, Optional, Iterable
from torch.utils.data import Dataset, Subset, DataLoader
from homura.reporters import TQDMReporter, TensorboardReporter
import numpy as np
import hydra
from allinone.data.mnist import DealDataset
from torchvision import transforms as tf
from allinone import MODEL_REGISTRY
from homura.trainers import SupervisedTrainer
from homura.optim import SGD
from torch.nn import functional as F
from managpu import GpuManager
GpuManager().set_by_memory(1)

logger = logging.getLogger(__name__)


def get_label_list(dataset: Dataset) -> (np.array):
    label_list = []
    for _, label in dataset:
        label_list.append(label)
    label_list = np.array(label_list)
    return label_list


def get_label_indices(
        label_list: np.array,
        num_labels_per_class: int,
        num_classes: int) -> list:
    label_indices = []
    for i in range(num_classes):
        indices = np.where(label_list == i)[0]
        np.random.shuffle(indices)
        label_indices.extend(indices[:num_labels_per_class])
    return label_indices


class PartDataLoader:
    def __init__(self,
                 label_loader: DataLoader,
                 num_iteration: int):
        self.label = label_loader
        self.num_iteration = num_iteration

    def __iter__(self):
        label_iter = iter(self.label)
        count = 0
        while count < self.num_iteration:
            count += 1
            try:
                label = label_iter.next()
            except BaseException:
                label_iter = iter(self.label)
                label = label_iter.next()
            yield label

    def __len__(self):
        return self.num_iteration


class PartDataset:
    def __init__(self,
                 root: str,
                 num_labels_per_class: int,
                 dataset: Dataset,
                 num_classes: int,
                 label_transform: Optional[Callable] = None,
                 test_transform: Optional[Callable] = None,
                 download: bool = False):
        super(PartDataset, self).__init__()
        base_dataset = dataset(
            root, transform=label_transform, download=download)
        label_list = get_label_list(base_dataset)
        label_indices = get_label_indices(
            label_list, num_labels_per_class, num_classes)
        self.length = len(label_list)
        self.label_dataset = Subset(base_dataset, label_indices)
        self.test_dataset = dataset(
            root, train=False, transform=test_transform, download=download)
        self.num_classes = num_classes

    def get_dataloader(self,
                       label_batch_size: int,
                       test_batch_size: Optional[int] = None,
                       num_iteration: Optional[int] = None,
                       shuffle: bool = True,
                       num_workers: int = 0,
                       pin_memory: bool = True,
                       drop_last: bool = False,
                       return_num_classes: bool = True):
        shared_dict = {'num_workers': num_workers,
                       'pin_memory': pin_memory,
                       'drop_last': drop_last
                       }
        if test_batch_size is None:
            test_batch_size = label_batch_size
        label_loader = DataLoader(
            self.label_dataset,
            label_batch_size,
            shuffle=shuffle,
            **shared_dict)
        test_loader = DataLoader(self.test_dataset,
                                 test_batch_size, **shared_dict)
        if num_iteration is None:
            num_iteration = self.length // label_batch_size
            if not drop_last and self.length % label_batch_size != 0:
                num_iteration += 1
        train_loader = PartDataLoader(
            label_loader, num_iteration)
        ret = [train_loader, test_loader]
        if return_num_classes:
            ret.append(self.num_classes)
        return ret

    __call__ = get_dataloader


@hydra.main(config_path="config", config_name='base.yml')
def main(args):
    logger.info(args)
    mnist = PartDataset(
        root=args.root,
        num_labels_per_class=10,
        dataset=DealDataset,
        num_classes=10,
        label_transform=tf.Compose(
            [tf.ToPILImage(), tf.Resize(
                (32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
        test_transform=tf.Compose(
            [tf.ToPILImage(), tf.Resize(
                (32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
    )
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers, return_num_classes=True)
    lenet = MODEL_REGISTRY(args.model)(num_classes=num_classes)

    trainer = SupervisedTrainer(lenet,
                                SGD(lr=args.lr_256 * args.batch_size / 256,
                                    momentum=0.9),
                                F.cross_entropy,
                                reporters=[TQDMReporter()])
    for _ in trainer.epoch_range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)

    logger.info(f"Max Accuracy={max(trainer.history['accuracy/test'])}")


if __name__ == '__main__':
    main()
