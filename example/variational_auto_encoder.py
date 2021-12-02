from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms as tf
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from DeSSL import MODEL_REGISTRY, SEMI_DATASET_REGISTRY, SCHEDULER_REGISTRY, loadding_config
from DeSSL.trainer import VariationalAutoEncoder
from DeSSL.data import SemiDataset


def main(args):
    mnist = SEMI_DATASET_REGISTRY(args.dataset)(
        args.root, args.num_labels_per_class)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers, return_num_classes=True)
    vae = MODEL_REGISTRY(args.vae)()
    toy = MODEL_REGISTRY(args.model)(num_classes=num_classes)
    optimizer = {'optimizer': Adam, 'lr': args.lr_256 * args.batch_size / 256}
    lr_scheduler = {'lr_scheduler': LambdaLR,
                    'lr_lambda': lambda epoch: epoch * 0.18 + 0.1 if epoch < 5 else 1.}
    variational_auto_encoder = VariationalAutoEncoder((train_loader, test_loader),
                                                      {'vae': vae, 'encoder': toy},
                                                      optimizer,
                                                      lr_scheduler)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(variational_auto_encoder)


if __name__ == '__main__':
    parser = loadding_config('config/variational_auto_encoder.yml')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(profiler="simple", max_epochs=150)
    args = parser.parse_args()
    print(args)
    main(args)
