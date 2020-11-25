from homura.trainers import SupervisedTrainer
from homura.optim import Adam
from homura.reporters import TQDMReporter, TensorboardReporter
from homura.vision import MODEL_REGISTRY
from torch.nn import functional as F
from torch.nn import ConvTranspose2d
from torchvision.datasets import CIFAR10
from torchvision import transforms as tf
import hydra
from managpu import GpuManager
from itertools import cycle
GpuManager().set_by_memory(1)


@hydra.main(config_path="config", config_name='test_adversarially_learned_inference.yml')
def main(args):
    import sys
    sys.path.append('/home/kp600168/semi')
    from allinone import SEMI_DATASET_REGISTRY, SEMI_TRAINER_REGISTRY, SCHEDULER_REGISTRY
    print(args)
    cifar = SEMI_DATASET_REGISTRY('mix')(CIFAR10, args.dataset, 10, [tf.RandomResizedCrop(
        (32, 32)), tf.ToTensor(), tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))], semi_size=args.semi_size)
    train_loader, test_loader, num_classes = cifar(
        args.batch_size, num_workers=args.num_workers, return_num_classes=True)
    model_dict = {
        'generator_x': MODEL_REGISTRY('Generator_x')(),
        'generator_z': MODEL_REGISTRY('Generator_z')(),
        'discriminator_x': MODEL_REGISTRY('Discriminator_x')(),
        'discriminator_z': MODEL_REGISTRY('Discriminator_z')(),
        'discriminator_x_z': MODEL_REGISTRY('Discriminator_x_z')(num_classes),
    }
    kwargs = {
        'model_dict': model_dict,
        'optimizer': Adam(lr=args.lr_100 * args.batch_size / 100, betas=(0.5, 1 - 1e-3)),
        'loss_f': F.cross_entropy,
        'consistency_weight': SCHEDULER_REGISTRY('identity')(1.),
    }
    trainer = SEMI_TRAINER_REGISTRY('AdversariallyLearnedInference')(
        **kwargs, reporters=[TQDMReporter(), TensorboardReporter('.')])

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
