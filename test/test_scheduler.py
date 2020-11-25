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
    sys.path.append('/home/kp600168/semi')
    from allinone import SCHEDULER_REGISTRY, SEMI_DATASET_REGISTRY
    print(args)
    print(SCHEDULER_REGISTRY.catalogue())
    a = SCHEDULER_REGISTRY('linear')(5, 10)
    a.step()
    print(a())
    a = SCHEDULER_REGISTRY('identity')(5)
    a.step()
    print(a())
    a = SCHEDULER_REGISTRY('lambda')(lambda x: x + 1)
    a.step()
    print(a())


if __name__ == '__main__':
    main()
