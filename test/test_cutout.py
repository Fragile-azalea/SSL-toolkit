import pytest
from allinone.data import SemiDataset
from allinone.transforms import ImageCutOut
from torchvision.datasets import MNIST
from PIL.Image import Image


@pytest.mark.parametrize('root, num_labels_per_class, num_classes', [('/home/kp600168/.torch/data/', 50, 10), ])
def test_cutout(root, num_labels_per_class, num_classes):
    semi_mnist = SemiDataset(root, num_labels_per_class, MNIST,
                             num_classes, ImageCutOut(10, (127)))
    for data, target in semi_mnist.label_dataset:
        assert isinstance(data, Image) == True
        break
