from typing import Callable
from torchvision import transforms as tf
from . import TRANSFORM_REGISTRY


__all__ = ['ManyTimes', 'Twice']


@TRANSFORM_REGISTRY.register
class IdentityAndManyTimes:
    '''
    This class changes an image to a normalized tensor image and a series of augmented image.

    Args:
        transform: A list of image augmentation.
        norm: A list of image normalization.
        n: The times that the transform perform.
    '''

    def __init__(self,
                 transform: list,
                 norm: list,
                 n: int):
        self.transform = tf.Compose(transform + norm)
        self.norm = tf.Compose(norm)
        self.n = n

    def __call__(self, inp):
        return (self.norm(inp), *(self.transform(inp) for _ in range(self.n)))


@TRANSFORM_REGISTRY.register
class ManyTimes:
    '''
    This class transfers an image to a series of augmented images.

    Args:
        transform: The transform for augmentation and normalization of images.
        n: The times that the transform performs.

    Returns:
        The tuple of augmented images.
    '''

    def __init__(self,
                 transform: Callable,
                 n: int):
        self.transform = transform
        self.n = n

    def __call__(self, inp) -> tuple:
        '''
            Call of this class.

            Args:
                inp: something importance.
        '''
        return (*(self.transform(inp) for _ in range(self.n)),)

    def __str__(self):
        return 'transform:%s\ntimes:%d' % (str(self.transform), self.n)


@TRANSFORM_REGISTRY.register
def Twice(transform: Callable) -> ManyTimes:
    '''
    The easy call method of ManyTimes(transform, 2).

    Args:
        transform: The transform for augmentation and normalization of images.

    Returns:
        The class of ManyTimes(transform, 2).
    '''
    return ManyTimes(transform, 2)
