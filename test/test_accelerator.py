import pytest
import torch


def test_accelerator():
    from DeSSL.data import accelerator, SemiDataset
    from torchvision.datasets import MNIST, FashionMNIST
    from DeSSL import loadding_config

    parser = loadding_config('config/base.yml')
    args = parser.parse_args([])

    FastMNIST = accelerator(torch.device('cuda:0'), MNIST, 0.1307, 0.3081)
    semi_mnist = SemiDataset(
        args.root, args.num_labels_per_class, FastMNIST, 10)
    train_loader, test_loader, classes = semi_mnist(
        args.batch_size,   num_workers=0, pin_memory=False)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [args.batch_size, 1, 28, 28]
        break
    for data, target in test_loader:
        assert list(data.shape) == [args.batch_size * 2, 1, 28, 28]
        break
    assert classes == args.num_classes

    FastMNIST = accelerator(torch.device('cuda:0'), FashionMNIST, 0.286, 0.352)
    semi_mnist = SemiDataset(
        args.root, args.num_labels_per_class, FastMNIST, 10)
    train_loader, test_loader, classes = semi_mnist(
        args.batch_size, num_workers=0, pin_memory=False)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [args.batch_size, 1, 28, 28]
        break
    for data, target in test_loader:
        assert list(data.shape) == [args.batch_size * 2, 1, 28, 28]
        break
    assert classes == args.num_classes


def test_register():
    from DeSSL.data import ACCELERATOR_REGISTRY, SemiDataset
    from DeSSL import loadding_config

    parser = loadding_config('config/base.yml')
    args = parser.parse_args([])

    FastMNIST = ACCELERATOR_REGISTRY('mnist')(torch.device('cuda:0'))
    semi_mnist = SemiDataset(
        args.root, args.num_labels_per_class, FastMNIST, 10)
    train_loader, test_loader, classes = semi_mnist(
        args.batch_size, num_workers=0, pin_memory=False)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [args.batch_size, 1, 28, 28]
        break
    for data, target in test_loader:
        assert list(data.shape) == [args.batch_size * 2, 1, 28, 28]
        break
    assert classes == args.num_classes

    FastMNIST = FastMNIST = ACCELERATOR_REGISTRY(
        'fashionmnist')(torch.device('cuda:0'))
    semi_mnist = SemiDataset(
        args.root, args.num_labels_per_class, FastMNIST, 10)
    train_loader, test_loader, classes = semi_mnist(
        args.batch_size, num_workers=0, pin_memory=False)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [args.batch_size, 1, 28, 28]
        break
    for data, target in test_loader:
        assert list(data.shape) == [args.batch_size * 2, 1, 28, 28]
        break
    assert classes == args.num_classes
