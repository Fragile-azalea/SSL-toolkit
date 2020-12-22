from allinone import SEMI_DATASET_REGISTRY, SEMI_TRAINER_REGISTRY
from allinone.data.data import semi_svhn, SemiDataset
import numpy as np
import gzip
import os
import hydra
from torchvision import transforms as tf
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torch.nn import ConvTranspose2d
from torch.nn import functional as F
from homura.vision import MODEL_REGISTRY
from homura.reporters import TQDMReporter, TensorboardReporter
from homura.optim import SGD
from homura.trainers import SupervisedTrainer
from managpu import GpuManager
GpuManager().set_by_memory(1)


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


@hydra.main(config_path="config", config_name='test_ladder.yml')
def main(args):
    #     import sys
    #     sys.path.append('/home/kp600168/semi/SSL-toolkit')

    print(args)
    mnist = SemiDataset(10, '/gdata/MNIST/', DealDataset, 10, label_transform=tf.Compose(
        [tf.ToPILImage(), tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
        unlabel_transform=tf.Compose(
        [tf.ToPILImage(), tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
        test_transform=tf.Compose(
        [tf.ToPILImage(), tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
    )
    # mnist = SEMI_DATASET_REGISTRY('mix')(MNIST, args.dataset, 10, [tf.Resize(
    #     (32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))], semi_size=args.semi_size)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers, return_num_classes=True)
    lenet = MODEL_REGISTRY(args.model)(num_classes=num_classes)
    kwargs = {
        'bn_list': [
            lenet.bn1, lenet.bn2, lenet.bn3, ], 'sigma_list': [
            0.3, 0.3, 0.3, ], 'v_list': [
                ConvTranspose2d(
                    16, 6, 10, 2), ConvTranspose2d(
                        120, 16, 10), ], 'lam_list': [
                            0.1, 0.01, 0.01, ], }
    trainer = SEMI_TRAINER_REGISTRY('Ladder')(lenet,
                                              SGD(lr=args.lr_256 * args.batch_size / 256,
                                                  momentum=0.9),
                                              F.cross_entropy,
                                              **kwargs,
                                              reporters=[TQDMReporter()])

    for _ in trainer.epoch_range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)

    # trainer = SupervisedTrainer(lenet, SGD(lr=args.lr_256 * args.batch_size /
    #                                        256, momentum=0.9), F.cross_entropy, reporters=[TQDMReporter()])
    # for _ in trainer.epoch_range(args.epochs):
    #     trainer.train(train_loader)
    #     trainer.test(test_loader)

    print(f"Max Accuracy={max(trainer.history['accuracy/test'])}")


if __name__ == '__main__':
    main()
