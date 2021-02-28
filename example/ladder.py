from allinone import SEMI_TRAINER_REGISTRY
from allinone.data import SEMI_MNIST
import hydra
import logging
from torch.nn import functional as F
from homura.vision import MODEL_REGISTRY
from homura.reporters import TQDMReporter, TensorboardReporter
from homura.optim import Adam
from managpu import GpuManager
from homura.lr_scheduler import LambdaLR
GpuManager().set_by_memory(1)


logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name='ladderv2.yml')
def main(args):
    logger.info(args)
    mnist = SEMI_MNIST[args.dataset](args.root)
    num_neurons = [1000, 500, 250, 250, 250, 10, ]
    sigma_noise = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, ]
    kwargs = {'num_neurons': num_neurons,
              'sigma_noise': sigma_noise, 'input_sigma_noise': 0.3}
    # sigma_noise = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ]
    # kwargs = {'num_neurons': num_neurons,
    #           'sigma_noise': sigma_noise, 'input_sigma_noise': 0.0}
    logger.info(kwargs)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers, return_num_classes=True, pin_memory=False)
    lenet = MODEL_REGISTRY(args.model)((28, 28), **kwargs)
    # lam_list = [1000., 10., 0.1, 0.1, 0.1, 0.1, 0.1]
    lam_list = [0.] * 7
    lr_scheduler = LambdaLR(lambda epoch: epoch * 0.18 + 0.1 if epoch < 5 else (1. if epoch <
                                                                                50 else 1.5 - epoch / 100))
    kwargs = {'lam_list': lam_list, 'scheduler': lr_scheduler}
    logger.info(kwargs)
    trainer = SEMI_TRAINER_REGISTRY('Ladder')(
        lenet,
        Adam(
            lr=args.lr_256 *
            args.batch_size /
            256
        ),
        F.nll_loss,
        **kwargs,
        reporters=[
            TQDMReporter()])

    for _ in trainer.epoch_range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)
        trainer.scheduler.step()

    logger.info(f"Max Accuracy={max(trainer.history['accuracy/test'])}")


if __name__ == '__main__':
    main()
