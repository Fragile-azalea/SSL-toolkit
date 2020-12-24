from AdversariallyLearnedInference import Generator_x, Generator_z, Discriminator_x, Discriminator_z, Discriminator_x_z, AdversariallyLearnedInferenceTrainerV2
from homura.trainers import SupervisedTrainer
from homura.optim import Adam
from homura.reporters import TQDMReporter, TensorboardReporter
from homura.vision.data import VisionSet
from torch.nn import functional as F
from torch.nn import ConvTranspose2d
from torchvision.datasets import CIFAR10
from torchvision import transforms as tf
import argparse
from torch.cuda import device_count
from homura.metrics import accuracy
from utils import SemiVisionSet
from managpu import GpuManager
GpuManager().set_by_memory(1)


def main(args):
    cifar = SemiVisionSet(CIFAR10, args.dataset, 10, [tf.ToTensor(), tf.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))], semi_size=40000)
    train_loader, test_loader, num_classes = cifar(
        args.batch_size, num_workers=4, return_num_classes=True, use_prefetcher=True)
    model_dict = {
        'generator_x': Generator_x(),
        'generator_z': Generator_z(),
        'discriminator_x': Discriminator_x(),
        'discriminator_z': Discriminator_z(),
        'discriminator_x_z': Discriminator_x_z(num_classes),
    }
    trainer = AdversariallyLearnedInferenceTrainerV2(model_dict, Adam(lr=args.lr_100 * args.batch_size / 100, betas=(
        0.5, 1 - 1e-3)), F.cross_entropy, 1, reporters=[TQDMReporter(), TensorboardReporter('.')])

    for _ in trainer.epoch_range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)

    print(f"Max Accuracy={max(trainer.history['accuracy/test'])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Working on MNIST')
    parser.add_argument('--dataset', type=str,
                        default='/home/kp600168/.torch/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr-100', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=6475)
    args = parser.parse_args()
    print(args)
    main(args)
