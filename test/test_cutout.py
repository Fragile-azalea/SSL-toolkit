from numpy.core.numeric import tensordot
import pytest
from DeSSL.data import SemiDataset
from DeSSL.transforms import ImageCutout, TensorCutout
from torchvision.datasets import MNIST
from PIL.Image import Image
from torchvision import transforms as tf
from torch import Tensor


def test_cutout():
    from DeSSL import loadding_config

    parser = loadding_config('config/base.yml')
    args = parser.parse_args([])

    semi_mnist = SemiDataset(args.root, args.num_labels_per_class, MNIST,
                             args.num_classes, ImageCutout(10, (127)))
    for data, target in semi_mnist.label_dataset:
        assert isinstance(data, Image) == True
        break
    semi_mnist = SemiDataset(args.root, args.num_labels_per_class, MNIST,
                             args.num_classes, tf.Compose([tf.ToTensor(), TensorCutout(10)]))
    for data, target in semi_mnist.label_dataset:
        assert isinstance(data, Tensor) == True and list(
            data.shape) == [1, 28, 28]
        break
