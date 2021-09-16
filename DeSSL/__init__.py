from .register import Registry

while True:
    # Registry.import_modules('DeSSL.trainer')
    from .scheduler import SCHEDULER_REGISTRY
    from .trainer import SEMI_TRAINER_REGISTRY
    from .model import MODEL_REGISTRY
    from .data import SEMI_DATASET_REGISTRY
    from .transforms import TRANSFORM_REGISTRY
    break
