from numpy.core.numeric import tensordot
import pytest
from allinone.data import SemiDataset
from allinone.transforms import ImageCutout, TensorCutout
from torchvision.datasets import MNIST
from PIL.Image import Image
from torchvision import transforms as tf
from torch import Tensor


@pytest.mark.parametrize('root, num_labels_per_class, num_classes', [('/home/kp600168/.torch/data/', 50, 10), ])
def test_cutout(root, num_labels_per_class, num_classes):
    semi_mnist = SemiDataset(root, num_labels_per_class, MNIST,
                             num_classes, ImageCutout(10, (127)))
    for data, target in semi_mnist.label_dataset:
        assert isinstance(data, Image) == True
        break
    semi_mnist = SemiDataset(root, num_labels_per_class, MNIST,
                             num_classes, tf.Compose([tf.ToTensor(), TensorCutout(10)]))
    for data, target in semi_mnist.label_dataset:
        assert isinstance(data, Tensor) == True and list(
            data.shape) == [1, 28, 28]
        break
