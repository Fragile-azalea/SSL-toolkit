import argparse
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms as tf
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from DeSSL import MODEL_REGISTRY, TRANSFORM_REGISTRY, SCHEDULER_REGISTRY, loadding_config
from DeSSL.trainer import MixMatch
from DeSSL.data import SemiDataset


def main(args):
    unlabel_transform = TRANSFORM_REGISTRY('ManyTimes')(tf.Compose([tf.RandomResizedCrop(
        ((32, 32)), (0.9, 1.0), (0.9, 1.1)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]), 4)
    mnist = SemiDataset(
        root=args.root,
        num_labels_per_class=args.num_labels_per_class,
        dataset=MNIST,
        num_classes=10,
        label_transform=tf.Compose(
            [tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
        unlabel_transform=unlabel_transform,
        test_transform=tf.Compose(
            [tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]),
    )
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers, return_num_classes=True)
    optimizer = {'optimizer': SGD, 'lr': args.lr_256 *
                 args.batch_size / 256, 'momentum': 0.9}
    lr_scheduler = {'lr_scheduler': LambdaLR,
                    'lr_lambda': lambda epoch: epoch * 0.18 + 0.1 if epoch < 5 else 1.}
    model = MODEL_REGISTRY(args.model)(num_classes=num_classes)
    mean_teacher = MixMatch((train_loader, test_loader),
                            optimizer,
                            lr_scheduler,
                            model,
                            0.5,
                            0.2,
                            SCHEDULER_REGISTRY('identity')(1.))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(mean_teacher)


if __name__ == '__main__':
    parser = loadding_config('config/mix_match.yml')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(profiler="simple", max_epochs=150)
    args = parser.parse_args()
    print(args)
    main(args)
