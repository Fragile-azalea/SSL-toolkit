# -*- coding: UTF-8 -*-
import argparse
from argparse import Namespace

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from DeSSL import MODEL_REGISTRY, SEMI_DATASET_REGISTRY
from DeSSL.trainer import Ladder


def main(args):
    mnist = SEMI_DATASET_REGISTRY(args.dataset)(args.root, 100, download=True)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers)
    optimizer = {'optimizer': Adam, 'lr': args.lr_256 * args.batch_size / 256}
    lr_scheduler = {'lr_scheduler': LambdaLR, 'lr_lambda': lambda epoch: epoch *
                    0.18 + 0.1 if epoch < 5 else (1. if epoch < 50 else 1.5 - epoch / 100)}
    num_neurons = [1000, 500, 250, 250, 250, 10, ]
    sigma_noise = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, ]
    kwargs = {'num_neurons': num_neurons,
              'sigma_noise': sigma_noise,
              'input_sigma_noise': 0.3}
    lenet = MODEL_REGISTRY(args.model)((28, 28), **kwargs)
    lam_list = [1000., 10., 0.1, 0.1, 0.1, 0.1, 0.1]

    ladder = Ladder((train_loader, test_loader), optimizer,
                    lr_scheduler, lenet, lam_list)
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

    trainer.fit(ladder)


if __name__ == '__main__':
    # python on_ucf11.py --model ClassifierRNN --rank 40,60,48,48,48 --init woho
    parser = argparse.ArgumentParser(description="Working on Ladder")

    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--root", type=str, default='/hdd1/public_data/')
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--model", type=str, default="Ladder_MLP")
    parser.add_argument("--dataset", type=str, default="semi_mnist")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr_256", type=float, default=0.0512)
    parser.set_defaults(profiler="simple", max_epochs=150)
    args = parser.parse_args()
    main(args)
