from torchvision import models
from . import MODEL_REGISTRY

__all__ = ['resnet18', 'resnet34', 'resnet50',
           'resnet101', 'resnet152', 'resnext50_32x4d']


@MODEL_REGISTRY.register
def resnet18(*args, **kwargs):
    return models.resnet18(*args, **kwargs)


@MODEL_REGISTRY.register
def resnet34(*args, **kwargs):
    return models.resnet34(*args, **kwargs)


@MODEL_REGISTRY.register
def resnet50(*args, **kwargs):
    return models.resnet50(*args, **kwargs)


@MODEL_REGISTRY.register
def resnet101(*args, **kwargs):
    return models.resnet101(*args, **kwargs)


@MODEL_REGISTRY.register
def resnet152(*args, **kwargs):
    return models.resnet152(*args, **kwargs)


@MODEL_REGISTRY.register
def resnext50_32x4d(*args, **kwargs):
    return models.resnext50_32x4d(*args, **kwargs)
