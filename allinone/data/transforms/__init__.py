from homura import Registry
TRANSFORM_REGISTRY = Registry('transform')
while True:
    from .transform_many_times import *
    from .randaugment import RandAugment
    break
