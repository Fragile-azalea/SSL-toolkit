from torchvision import transforms as tf
from . import TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register
class IdentityAndManyTimes:
    '''
    this class changes an image to a normalized tensor image and a series of augmented image.
    Attributes:
        transform: a list of image augmentation
        norm: a list of image normalization
        n: the times that the transform perform
    '''

    def __init__(self,
                 transform: list,
                 norm: list,
                 n: int):
        self.transform = tf.Compose(transform + norm)
        self.norm = tf.Compose(norm)
        self.n = n

    def __call__(self, inp):
        return self.norm(inp), *(self.transform(inp) for _ in range(self.n))


@TRANSFORM_REGISTRY.register
class ManyTimes:
    '''
    this class changes an image to a series of augmented image.
    Attributes:
        transform: a transform for image augmentation and normalization
        n: the times that the transform perform
    '''

    def __init__(self,
                 transform,
                 n: int):
        self.transform = transform
        self.n = n

    def __call__(self, inp):
        return *(self.transform(inp) for _ in range(self.n)),

    def __str__(self):
        return 'transform:%s\ntimes:%d' % (str(self.transform), self.n)


@TRANSFORM_REGISTRY.register
def Twice(transform):
    return ManyTimes(transform, 2)
