from homura.trainers import SupervisedTrainer
from homura.optim import SGD
from homura.vision import MODEL_REGISTRY
from homura.reporters import TQDMReporter, TensorboardReporter
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms as tf
from allinone import SEMI_DATASET_REGISTRY, SEMI_TRAINER_REGISTRY, TRANSFORM_REGISTRY, SCHEDULER_REGISTRY
from allinone.data import SemiDataset
from managpu import GpuManager
import hydra
import logging
GpuManager().set_by_memory(1)

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name='base.yml')
def main(args):
    logger.info(args)
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
    kwargs = {
        'model': MODEL_REGISTRY(args.model)(num_classes=num_classes),
        'optimizer': SGD(lr=args.lr_256 * args.batch_size / 256, momentum=0.9),
        'loss_f': F.cross_entropy,
        'consistency_weight': SCHEDULER_REGISTRY('identity')(0.00),
        'alpha': SCHEDULER_REGISTRY('lambda')(lambda epoch: min(1 - 1 / (1 + epoch), 0.99)),
        'dataset_type': 'split',
    }
    logger.info(kwargs)
    trainer = SEMI_TRAINER_REGISTRY('meanteacher')(
        **kwargs, reporters=[TQDMReporter()])
    for _ in trainer.epoch_range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)

    logger.info(f"Max Accuracy={max(trainer.history['accuracy/test'])}")


if __name__ == '__main__':
    main()
