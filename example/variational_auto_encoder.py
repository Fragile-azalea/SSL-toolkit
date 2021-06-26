from torch import zeros, randn
from torchvision.utils import make_grid
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


@hydra.main(config_path="config", config_name='variational_auto_encoder.yml')
def main(args):
    logger.info(args)
    mnist = SEMI_MNIST[args.dataset](args.root)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers, return_num_classes=True, pin_memory=False)
    vae = MODEL_REGISTRY(args.vae)()
    toy = MODEL_REGISTRY(args.model)(num_classes=num_classes)
    logger.info({'encoder_model': toy, 'vae_model': vae})
    trainer = SEMI_TRAINER_REGISTRY('VariationalAutoEncoder')(toy, vae, Adam(
        lr=args.lr_256 * args.batch_size / 256), F.binary_cross_entropy, reporters=[TQDMReporter(), TensorboardReporter('.')])

    B = 64
    H = W = 28
    for _ in trainer.epoch_range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)
        for i in range(num_classes):
            u_target = zeros((64, num_classes), device=trainer.device)
            u_target[:, i] = 1
            u_output, _, __ = trainer.model['vae'](
                randn(B, H * W, device=trainer.device), u_target)
            u_output = u_output.view(B, -1, H, W)
            trainer.reporter.add_image(
                str(i), make_grid(u_output, range=(-1, 1)), trainer.epoch)

    print(f"Max Accuracy={max(trainer.history['accuracy/test'])}")


if __name__ == '__main__':
    main()
