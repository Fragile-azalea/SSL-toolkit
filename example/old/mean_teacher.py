from MeanTeacher import MeanTeacherTrainerV2, TransformTwice
from homura.trainers import SupervisedTrainer
from homura.optim import SGD
from homura.reporters import TQDMReporter, TensorboardReporter
from utils import SemiVisionSet
from torch.nn import functional as F
from torch.nn import ConvTranspose2d
from torchvision.datasets import MNIST
from torchvision import transforms as tf
import argparse
from torch.cuda import device_count
from homura.metrics import accuracy
from lenet import LeNet5
from managpu import GpuManager
GpuManager().set_by_memory(1)


def main(args):
    mnist = SemiVisionSet(MNIST, args.dataset, 10, [], [TransformTwice(tf.Compose([tf.RandomResizedCrop((32, 32), (0.9, 1.0), (0.9, 1.1)), tf.ToTensor(
    ), tf.Normalize((0.1307,), (0.3081,))]))], [tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))], semi_size=59900)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=4, return_num_classes=True)
    # val_loader.transform = train_loader.transform
    lenet = LeNet5(num_classes=num_classes)
    trainer = MeanTeacherTrainerV2(lenet, SGD(lr=args.lr_256 * args.batch_size /
                                              256, momentum=0.9), F.cross_entropy, 0.01, 0.99, reporters=[TQDMReporter()])
    for _ in trainer.epoch_range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)

    print(f"Max Accuracy={max(trainer.history['accuracy/test'])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Working on MNIST')
    parser.add_argument('--dataset', type=str,
                        default='/home/kp600168/.torch/data/')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr-256', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    print(args)
    main(args)
