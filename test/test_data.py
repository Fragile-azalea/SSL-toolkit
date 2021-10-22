from pytest import raises
import pytest


def test_register():
    from DeSSL.data import SEMI_DATASET_REGISTRY
    from DeSSL import loadding_config

    parser = loadding_config('config/base.yml')
    args = parser.parse_args([])

    semi_cifar = SEMI_DATASET_REGISTRY(
        'semi_cifar10')(args.root, args.num_labels_per_class)
    train_loader, test_loader, classes = semi_cifar(
        args.batch_size, num_workers=args.num_workers)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [args.batch_size, 3, 32, 32]
        break
    for data, target in test_loader:
        assert list(data.shape) == [args.batch_size * 2, 3, 32, 32]
        break
    assert classes == args.num_classes


def test_data():
    from DeSSL.data import SemiDataset, semi_cifar10
    from torchvision.datasets import CIFAR10
    from torchvision import transforms as tf
    from DeSSL import loadding_config

    parser = loadding_config('config/base.yml')
    args = parser.parse_args([])

    semi_cifar = SemiDataset(args.root, args.num_labels_per_class, CIFAR10,
                             args.num_classes, tf.ToTensor(), tf.ToTensor(), tf.ToTensor())
    train_loader, test_loader, classes = semi_cifar(
        args.batch_size, num_workers=args.num_workers)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [args.batch_size, 3, 32, 32]
        break
    for data, target in test_loader:
        assert list(data.shape) == [args.batch_size * 2, 3, 32, 32]
        break
    assert classes == args.num_classes

    semi_cifar = semi_cifar10(args.root, args.num_labels_per_class)
    train_loader, test_loader, classes = semi_cifar(
        args.batch_size, num_workers=args.num_workers)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [args.batch_size, 3, 32, 32]
        break
    for data, target in test_loader:
        assert list(data.shape) == [args.batch_size * 2, 3, 32, 32]
        break
    assert classes == args.num_classes


def test_include_labeled_data():
    from DeSSL.data import SemiDataset
    from torchvision.datasets import MNIST
    from torchvision import transforms as tf
    from DeSSL import loadding_config

    parser = loadding_config('config/base.yml')
    args = parser.parse_args([])

    mnist = MNIST(args.root)
    semi_mnist = SemiDataset(args.root, args.num_labels_per_class, MNIST,
                             args.num_classes, tf.ToTensor(), tf.ToTensor(), tf.ToTensor())
    assert len(semi_mnist.unlabel_dataset) == len(mnist)
    uninclude_semi_mnist = SemiDataset(args.root, args.num_labels_per_class, MNIST, args.num_classes, tf.ToTensor(
    ), tf.ToTensor(), tf.ToTensor(), include_labeled_data=False)
    assert len(uninclude_semi_mnist.unlabel_dataset) + \
        len(uninclude_semi_mnist.label_dataset) == len(mnist)


def test_import_data():
    with raises(ImportError):
        from DeSSL import SemiDataset, semi_mnist
