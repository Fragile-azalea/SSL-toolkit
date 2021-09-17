import argparse
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms as tf
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from DeSSL import MODEL_REGISTRY, TRANSFORM_REGISTRY, SCHEDULER_REGISTRY
from DeSSL.trainer import MeanTeacher
from DeSSL.data import SemiDataset


def main(args):
    unlabel_transform = TRANSFORM_REGISTRY('twice')(tf.Compose([tf.RandomResizedCrop(
        ((32, 32)), (0.9, 1.0), (0.9, 1.1)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]))

    mnist = SemiDataset(
        root=args.root,
        num_labels_per_class=50,
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
    mean_teacher = MeanTeacher((train_loader, test_loader),
                               optimizer,
                               lr_scheduler,
                               model,
                               SCHEDULER_REGISTRY('identity')(0.01),
                               SCHEDULER_REGISTRY('lambda')(
        lambda epoch: min(1 - 1 / (1 + epoch), 0.99)))

    callbacks = [ModelCheckpoint(monitor="val_acc1",
                                 filename="{val_acc1:.2f}",
                                 save_weights_only=True,
                                 mode='max'), ]

    plugins = DDPPlugin(
        find_unused_parameters=False) if args.accelerator == 'ddp' else None

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        plugins=plugins)

    trainer.fit(mean_teacher)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Working on MeanTeacher")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--root", type=str, default='/hdd1/public_data/')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--model", type=str, default="LeNet5")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr_256", type=float, default=0.005)
    parser.set_defaults(profiler="simple", max_epochs=150)
    args = parser.parse_args()
    print(args)
    main(args)
