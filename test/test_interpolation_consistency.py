from homura.trainers import SupervisedTrainer
from homura.optim import SGD
from homura.reporters import TQDMReporter, TensorboardReporter
from homura.vision import MODEL_REGISTRY
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms as tf
import hydra
from managpu import GpuManager
from itertools import cycle
GpuManager().set_by_memory(1)


@hydra.main(config_path="config", config_name='test_ladder.yml')
def main(args):
    import sys
    sys.path.append('/home/kp600168/semi')
    from allinone import SEMI_DATASET_REGISTRY, SEMI_TRAINER_REGISTRY, SCHEDULER_REGISTRY
    print(args)
    mnist = SEMI_DATASET_REGISTRY('mix')(MNIST, args.dataset, 10, [tf.Resize(
        (32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))], semi_size=args.semi_size)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers, return_num_classes=True)
    lenet = MODEL_REGISTRY(args.model)(num_classes=num_classes)
    kwargs = {
        'model': MODEL_REGISTRY(args.model)(num_classes=num_classes),
        'optimizer': SGD(lr=args.lr_256 * args.batch_size / 256, momentum=0.9),
        'loss_f': F.cross_entropy,
        'consistency_weight': SCHEDULER_REGISTRY('identity')(1e-5),
        'alpha': SCHEDULER_REGISTRY('lambda')(lambda epoch: min(1 - 1 / (1 + epoch), 0.99)),
        'beta': 0.02
    }
    trainer = SEMI_TRAINER_REGISTRY('InterpolationConsistency')(
        **kwargs, reporters=[TQDMReporter()])

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
    main()
