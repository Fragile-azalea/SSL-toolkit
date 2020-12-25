from allinone import SEMI_TRAINER_REGISTRY
from allinone.data import SEMI_MNIST
import hydra
from torchvision import transforms as tf
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torch.nn import ConvTranspose2d
from torch.nn import functional as F
from homura.vision import MODEL_REGISTRY
from homura.reporters import TQDMReporter, TensorboardReporter
from homura.optim import SGD
from homura.trainers import SupervisedTrainer
from managpu import GpuManager
GpuManager().set_by_memory(1)


@hydra.main(config_path="config", config_name='ladder.yml')
def main(args):
    print(args)
    mnist = SEMI_MNIST[args.dataset](args.root)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers, return_num_classes=True)
    lenet = MODEL_REGISTRY(args.model)(num_classes=num_classes)
    kwargs = {
        'bn_list': [
            lenet.bn1, lenet.bn2, lenet.bn3, ], 'sigma_list': [
            0.3, 0.3, 0.3, ], 'v_list': [
                ConvTranspose2d(
                    16, 6, 10, 2), ConvTranspose2d(
                        120, 16, 10), ], 'lam_list': [
                            0.1, 0.01, 0.01, ], }
    trainer = SEMI_TRAINER_REGISTRY('Ladder')(lenet,
                                              SGD(lr=args.lr_256 * args.batch_size / 256,
                                                  momentum=0.9),
                                              F.cross_entropy,
                                              **kwargs,
                                              reporters=[TQDMReporter()])

    for _ in trainer.epoch_range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)

    print(f"Max Accuracy={max(trainer.history['accuracy/test'])}")


if __name__ == '__main__':
    main()
