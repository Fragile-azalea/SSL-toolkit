from homura.trainers import SupervisedTrainer
from homura.optim import Adam
from homura.reporters import TQDMReporter, TensorboardReporter
from homura.vision import MODEL_REGISTRY
from torch.nn import functional as F
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms as tf
import hydra
from managpu import GpuManager
from itertools import cycle
GpuManager().set_by_memory(1)
        

@hydra.main(config_path="config", config_name='test_variational_auto_encoder.yml')
def main(args):
    import sys
    sys.path.append('/home/kp600168/semi/SSL-toolkit')
    from allinone import SEMI_DATASET_REGISTRY, SEMI_TRAINER_REGISTRY
    print(args)
    mnist = SEMI_DATASET_REGISTRY('mix')(MNIST, args.dataset, 10, [
        tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))], semi_size=args.semi_size)
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers, return_num_classes=True)
    vae = MODEL_REGISTRY('VAE')()
    toy = MODEL_REGISTRY('toynet')(num_classes=num_classes)
    trainer = SEMI_TRAINER_REGISTRY('VariationalAutoEncoder')(toy,vae, Adam(lr=2e-3), F.binary_cross_entropy, reporters=[TQDMReporter()])

    for _ in trainer.epoch_range(args.epochs):
        trainer.train(train_loader)
        trainer.test(test_loader)

    print(f"Max Accuracy={max(trainer.history['accuracy/test'])}")


if __name__ == '__main__':
    main()
