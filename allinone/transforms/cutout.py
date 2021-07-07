from typing import Tuple
import numpy as np
import torch
from PIL.Image import Image
from PIL.ImageDraw import Draw
from . import TRANSFORM_REGISTRY

__all__ = ['TensorCutout', 'ImageCutout']


@TRANSFORM_REGISTRY.register
class TensorCutout():
    r'''
    `Cutout <https://arxiv.org/abs/1708.04552>`_ Augmentation for Tensor.

    Args:
        length: side length of cutout part. 

    Example:
        >>> from torchvision import transforms as tf
        >>> transforms = tf.Compose([tf.ToTensor(), TensorCutout(10)])

    Returns:
        An augmented tensor.
    '''

    def __init__(self, length: int):
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


@TRANSFORM_REGISTRY.register
class ImageCutout():
    r'''
    `Cutout <https://arxiv.org/abs/1708.04552>`_ Augmentation for Image.

    Args:
        length: side length of cutout part. 

    Example:
        >>> from torchvision import transforms as tf
        >>> transforms = tf.Compose([ImageCutout(10), tf.ToTensor()])

    Returns:
        An augmented image.
    '''

    def __init__(self, length: int, color: Tuple[int, int, int] = (127, 127, 127)):
        self.length = length
        self.color = color

    def __call__(self, img: Image) -> Image:
        h, w = img.size
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        img = img.copy()
        Draw(img).rectangle((x1, y1, x2, y2), self.color)
        return img
