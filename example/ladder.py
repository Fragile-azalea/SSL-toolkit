from allinone import SEMI_TRAINER_REGISTRY
from allinone.data import SEMI_MNIST
import hydra
import logging
from torchvision import transforms as tf
from torch.nn import ConvTranspose2d, Sequential, Upsample, Tanh
from torch.nn import functional as F
from homura.vision import MODEL_REGISTRY
from homura.reporters import TQDMReporter, TensorboardReporter
from homura.optim import SGD
from managpu import GpuManager
GpuManager().set_by_memory(1)


logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name='base.yml')
def main(args):
    logger.info(args)
    mnist = SEMI_MNIST[args.dataset](args.root)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers, return_num_classes=True)
    lenet = MODEL_REGISTRY(args.model)(num_classes=num_classes)
#     from random import choice
#     lam_list = [choice([0.001, 0.01, 0.1, 1.,]), choice(
#         [0.001, 0.01, 0.1, 1.,]), choice([0.001, 0.01, 0.1, 1.,]), ]
    lam_list = [1.0, 0.1, 0.1, ]
    logger.info(lam_list)
    v1 = Sequential(
        ConvTranspose2d(16, 6, 5),
        Upsample(scale_factor=2),
    )
    v2 = Sequential(
        ConvTranspose2d(120, 16, 5),
        Upsample(scale_factor=2),
    )
    kwargs = {
        'bn_list': [
            lenet.bn1, lenet.bn2, lenet.bn3, ], 'sigma_list': [
            0.3, 0.3, 0.3, ], 'v_list': [v1, v2, ], 'lam_list': lam_list, }
    trainer = SEMI_TRAINER_REGISTRY('Ladder')(lenet,
                                              SGD(lr=args.lr_256 * args.batch_size / 256,
                                                  momentum=0.9),
                                              F.cross_entropy,
                                              **kwargs,
                                              reporters=[TQDMReporter()])

    for _ in trainer.epoch_range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)

    logger.info(f"Max Accuracy={max(trainer.history['accuracy/test'])}")


if __name__ == '__main__':
    main()
