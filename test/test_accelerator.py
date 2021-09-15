import pytest
import torch


@pytest.mark.parametrize('root, num_labels_per_class, num_classes, batch_size', [('/home/kp600168/.torch/data/', 50, 10, 256), ])
def test_accelerator(root, num_labels_per_class, num_classes, batch_size):
    from DeSSL.data import accelerator, SemiDataset
    from torchvision.datasets import MNIST, FashionMNIST

    FastMNIST = accelerator(torch.device('cuda:0'), MNIST, 0.1307, 0.3081)
    semi_mnist = SemiDataset(root, num_labels_per_class, FastMNIST, 10)
    train_loader, test_loader, classes = semi_mnist(
        batch_size, num_workers=0, pin_memory=False)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [batch_size, 1, 28, 28]
        break
    for data, target in test_loader:
        assert list(data.shape) == [batch_size * 2, 1, 28, 28]
        break
    assert classes == num_classes

    FastMNIST = accelerator(torch.device('cuda:0'), FashionMNIST, 0.286, 0.352)
    semi_mnist = SemiDataset(root, num_labels_per_class, FastMNIST, 10)
    train_loader, test_loader, classes = semi_mnist(
        batch_size, num_workers=0, pin_memory=False)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [batch_size, 1, 28, 28]
        break
    for data, target in test_loader:
        assert list(data.shape) == [batch_size * 2, 1, 28, 28]
        break
    assert classes == num_classes


@pytest.mark.parametrize('root, num_labels_per_class, num_classes, batch_size', [('/home/kp600168/.torch/data/', 50, 10, 256), ])
def test_register(root, num_labels_per_class, num_classes, batch_size):
    from DeSSL.data import ACCELERATOR_REGISTRY, SemiDataset

    FastMNIST = ACCELERATOR_REGISTRY('mnist')(torch.device('cuda:0'))
    semi_mnist = SemiDataset(root, num_labels_per_class, FastMNIST, 10)
    train_loader, test_loader, classes = semi_mnist(
        batch_size, num_workers=0, pin_memory=False)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [batch_size, 1, 28, 28]
        break
    for data, target in test_loader:
        assert list(data.shape) == [batch_size * 2, 1, 28, 28]
        break
    assert classes == num_classes

    FastMNIST = FastMNIST = ACCELERATOR_REGISTRY(
        'fashionmnist')(torch.device('cuda:0'))
    semi_mnist = SemiDataset(root, num_labels_per_class, FastMNIST, 10)
    train_loader, test_loader, classes = semi_mnist(
        batch_size, num_workers=0, pin_memory=False)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [batch_size, 1, 28, 28]
        break
    for data, target in test_loader:
        assert list(data.shape) == [batch_size * 2, 1, 28, 28]
        break
    assert classes == num_classes
