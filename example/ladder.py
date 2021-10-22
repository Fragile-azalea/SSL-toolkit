# -*- coding: UTF-8 -*-
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from DeSSL import MODEL_REGISTRY, SEMI_DATASET_REGISTRY, loadding_config
from DeSSL.trainer import Ladder


def main(args):
    mnist = SEMI_DATASET_REGISTRY(args.dataset)(args.root, 10)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers)
    optimizer = {'optimizer': Adam, 'lr': args.lr_256 * args.batch_size / 256}
    lr_scheduler = {'lr_scheduler': LambdaLR, 'lr_lambda': lambda epoch: epoch *
                    0.18 + 0.1 if epoch < 5 else (1. if epoch < 50 else 1.5 - epoch / 100)}

    lenet = MODEL_REGISTRY(args.model)(**vars(args))
    ladder = Ladder((train_loader, test_loader), optimizer,
                    lr_scheduler, lenet, args.lam_list)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(ladder)


if __name__ == '__main__':
    parser = loadding_config('config/ladder.yml')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(profiler="simple", max_epochs=150)
    args = parser.parse_args()
    print(args)
    main(args)
