from homura.trainers import SupervisedTrainer
from homura.optim import SGD
from homura.vision import MODEL_REGISTRY
from homura.reporters import TQDMReporter, TensorboardReporter
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms as tf
from torch import randn
from managpu import GpuManager
import hydra
GpuManager().set_by_memory(1)


@hydra.main(config_path="config", config_name='test_ladder.yml')
def main(args):
    import sys
    sys.path.append('/home/kp600168/semi')
    from allinone import SEMI_DATASET_REGISTRY, SEMI_TRAINER_REGISTRY, TRANSFORM_REGISTRY, SCHEDULER_REGISTRY
    print(args)
    unlabel_transform = TRANSFORM_REGISTRY('twice')(tf.Compose([tf.RandomResizedCrop(
        ((32, 32)), (0.9, 1.0), (0.9, 1.1)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]))
    mnist = SEMI_DATASET_REGISTRY('mix')(MNIST, args.dataset, 10, [], [unlabel_transform], [tf.Resize(
        (32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))], semi_size=args.semi_size)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=4, return_num_classes=True)
    kwargs = {
        'model': MODEL_REGISTRY(args.model)(num_classes=num_classes),
        'optimizer': SGD(lr=args.lr_256 * args.batch_size / 256, momentum=0.9),
        'loss_f': F.cross_entropy,
        'consistency_weight': SCHEDULER_REGISTRY('identity')(0.01),
        'alpha': SCHEDULER_REGISTRY('lambda')(lambda epoch: min(1 - 1 / (1 + epoch), 0.99)),
        'dataset_type': 'mix',
    }
    trainer = SEMI_TRAINER_REGISTRY('meanteacher')(
        **kwargs, reporters=[TQDMReporter()])
    for _ in trainer.epoch_range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)

    # mnist = SEMI_DATASET_REGISTRY('split')(MNIST, args.dataset, 10, [], [unlabel_transform.transform], [tf.Resize(
    #     (32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))], semi_size=args.semi_size, unlabel_transform=unlabel_transform)
    # train_loader, test_loader, unlabel_loader, num_classes = mnist(
    #     args.batch_size, num_workers=0, return_num_classes=True)
    # kwargs = {
    #     'model': MODEL_REGISTRY(args.model)(num_classes=num_classes),
    #     'optimizer': SGD(lr=args.lr_256 * args.batch_size / 256, momentum=0.9),
    #     'loss_f': F.cross_entropy,
    #     'consistency_weight': SCHEDULER_REGISTRY('identity')(0.01),
    #     'alpha': SCHEDULER_REGISTRY('lambda')(lambda epoch: min(1 - 1 / (1 + epoch), 0.99)),
    #     'dataset_type': 'split',
    # }
    # trainer = SEMI_TRAINER_REGISTRY('meanteacher')(
    #     **kwargs, reporters=[TQDMReporter()])
    # for _ in trainer.epoch_range(args.epochs):
    #     trainer.train(zip(train_loader, unlabel_loader))
    #     trainer.test(test_loader)

    print(f"Max Accuracy={max(trainer.history['accuracy/test'])}")


if __name__ == '__main__':
    main()
