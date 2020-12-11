from homura.trainers import SupervisedTrainer
from homura.optim import SGD
from homura.reporters import TQDMReporter, TensorboardReporter
from homura.vision import MODEL_REGISTRY
from torch.nn import functional as F
from torch.nn import ConvTranspose2d
from torchvision.datasets import MNIST
from torchvision import transforms as tf
import hydra
from managpu import GpuManager
from itertools import cycle
GpuManager().set_by_memory(1)


@hydra.main(config_path="config", config_name='test_ladder.yml')
def main(args):
    import sys
    sys.path.append('/home/kp600168/semi/SSL-toolkit')
    from allinone import SEMI_DATASET_REGISTRY, SEMI_TRAINER_REGISTRY
    print(args)
    mnist = SEMI_DATASET_REGISTRY('split')(MNIST, args.dataset, 10, [tf.Resize(
        (32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))], semi_size=args.semi_size)
    train_loader, test_loader, _, num_classes = mnist(
        args.batch_size, num_workers=0, return_num_classes=True)
    lenet = MODEL_REGISTRY(args.model)(num_classes=num_classes)
    kwargs = {
        'bn_list': [lenet.bn1, lenet.bn2, lenet.bn3],
        'sigma_list': [0.3, 0.3, 0.3],
        'v_list': [ConvTranspose2d(16, 6, 10, 2), ConvTranspose2d(120, 16, 10)],
        'lam_list': [0.1, 0.01, 0.01],
    }
    trainer = SEMI_TRAINER_REGISTRY('Ladder')(lenet, SGD(
        lr=args.lr_256 * args.batch_size / 256, momentum=0.9), F.cross_entropy, **kwargs, reporters=[TQDMReporter()])

    # for _ in trainer.epoch_range(args.epochs):
    #     trainer.train(train_loader)
    #     trainer.test(test_loader)

    trainer = SupervisedTrainer(lenet, SGD(lr=args.lr_256 * args.batch_size /
                                           256, momentum=0.9), F.cross_entropy, reporters=[TQDMReporter()])
    for _ in trainer.epoch_range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)

    print(f"Max Accuracy={max(trainer.history['accuracy/test'])}")


if __name__ == '__main__':
    main()
