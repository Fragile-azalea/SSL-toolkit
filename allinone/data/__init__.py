from homura import Registry
SEMI_DATASET_REGISTRY = Registry('semi_dataset')
while True:
    from .transforms import *
    from .dataset import *
    from .mnist import *
    break
