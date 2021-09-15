from DeSSL import Registry
TRANSFORM_REGISTRY = Registry('transform')
while True:
    from .randaugment import *
    from .transform_many_times import *
    from .mixup import *
    from .cutout import *
    from .cutmix import *
    from .autoaugment import *
    break
