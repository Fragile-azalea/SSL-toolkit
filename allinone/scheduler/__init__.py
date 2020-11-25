from homura import Registry

SCHEDULER_REGISTRY = Registry('scheduler')

while True:
    from .scheduler import Linear
    break
