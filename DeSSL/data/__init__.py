from DeSSL import Registry
SEMI_DATASET_REGISTRY = Registry('semi_dataset')
ACCELERATOR_REGISTRY = Registry('accelerator')
while True:
    from .dataset import *
    from .accelerator import *
    break
