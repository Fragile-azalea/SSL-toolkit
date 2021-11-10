from torchvision import models
from torchvision.models.resnet import ResNet
from . import MODEL_REGISTRY
from typing import Any

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']


@MODEL_REGISTRY.register
def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    '''
    ResNet-18 model from "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>_.

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
    '''
    return models.resnet18(pretrained, progress, **kwargs)


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


@MODEL_REGISTRY.register
def resnext101_32x8d(*args, **kwargs):
    return models.resnext101_32x8d(*args, **kwargs)


@MODEL_REGISTRY.register
def wide_resnet50_2(*args, **kwargs):
    return models.wide_resnet50_2(*args, **kwargs)


@MODEL_REGISTRY.register
def wide_resnet101_2(*args, **kwargs):
    return models.wide_resnet101_2(*args, **kwargs)
