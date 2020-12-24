from utils import change_val_to_unlabel, SemiVisionSet
from Ladder import LadderTrainerV2
from homura.trainers import SupervisedTrainer
from homura.optim import SGD
from homura.reporters import TQDMReporter, TensorboardReporter
from homura.vision.data import VisionSet
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
    mnist = SemiVisionSet(MNIST, args.dataset, 10, [tf.Resize(
        (32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))], semi_size=59900)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=4, return_num_classes=True)
    lenet = LeNet5(num_classes=num_classes)
    bn_list = [lenet.bn1, lenet.bn2, lenet.bn3]
    sigma_list = [0.3, 0.3, 0.3]
    v_list = [ConvTranspose2d(16, 6, 10, 2), ConvTranspose2d(120, 16, 10)]
    lam_list = [0.1, 0.01, 0.01]
    trainer = LadderTrainerV2(lenet, SGD(lr=args.lr_256 * args.batch_size / 256, momentum=0.9), F.cross_entropy,
                              bn_list, sigma_list, v_list, lam_list, reporters=[TQDMReporter()])

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
    parser = argparse.ArgumentParser(description='Working on MNIST')
    parser.add_argument('--dataset', type=str,
                        default='/home/kp600168/.torch/data/')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr-256', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    print(args)
    main(args)
