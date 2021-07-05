from homura import Registry
TRANSFORM_REGISTRY = Registry('transform')
while True:
    from .randaugment import *
    from .transform_many_times import *
    from .mixup import *
    break
