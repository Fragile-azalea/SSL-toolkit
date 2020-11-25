from homura import Registry
SEMI_DATASET_REGISTRY = Registry('semi_dataset')
while True:
    from .datasets import *
    break
