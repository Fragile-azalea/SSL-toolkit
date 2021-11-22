from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms as tf
import pytorch_lightning as pl
from DeSSL import MODEL_REGISTRY, TRANSFORM_REGISTRY, SCHEDULER_REGISTRY, loadding_config
from DeSSL.trainer import AdversariallyLearnedInference
from DeSSL.data import semi_cifar10


def main(args):
    unlabel_transform = TRANSFORM_REGISTRY('twice')(tf.Compose([tf.RandomResizedCrop(
        ((32, 32)), (0.9, 1.0), (0.9, 1.1)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]))

    cifar = semi_cifar10(root=args.root, unlabel_transform=tf.RandomResizedCrop((32, 32)),
                         num_labels_per_class=args.num_labels_per_class)
    train_loader, test_loader, num_classes = cifar(
        args.batch_size, num_workers=args.num_workers, return_num_classes=True)

    optimizer = {'optimizer': Adam, 'lr': args.lr_256 *
                 args.batch_size / 256, 'betas': (0.5, 1 - 1e-3)}
    lr_scheduler = {'lr_scheduler': LambdaLR,
                    'lr_lambda': lambda epoch: epoch * 0.18 + 0.1 if epoch < 5 else 1.}
    model_dict = {
        'generator_x': MODEL_REGISTRY('Generator_x')(),
        'generator_z': MODEL_REGISTRY('Generator_z')(),
        'discriminator_x': MODEL_REGISTRY('Discriminator_x')(),
        'discriminator_z': MODEL_REGISTRY('Discriminator_z')(),
        'discriminator_x_z': MODEL_REGISTRY('Discriminator_x_z')(num_classes),
    }

    mean_teacher = AdversariallyLearnedInference((train_loader, test_loader),
                                                 optimizer,
                                                 optimizer,
                                                 lr_scheduler,
                                                 model_dict,
                                                 SCHEDULER_REGISTRY('identity')(1.))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(mean_teacher)


if __name__ == '__main__':
    parser = loadding_config('config/adversarially_learned_inference.yml')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(profiler="simple", max_epochs=150)
    args = parser.parse_args()
    print(args)
    main(args)
