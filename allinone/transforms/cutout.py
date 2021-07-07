from typing import Tuple
import numpy as np
import torch
from PIL.Image import Image
from PIL.ImageDraw import Draw
from . import TRANSFORM_REGISTRY

__all__ = ['TensorCutOut', 'ImageCutOut']


@TRANSFORM_REGISTRY.register
class TensorCutOut():
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img: torch.Tensor):
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
class ImageCutOut():
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, length: int, color: Tuple[int, int, int] = (127, 127, 127)):
        self.length = length
        self.color = color

    def __call__(self, img: Image):
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
