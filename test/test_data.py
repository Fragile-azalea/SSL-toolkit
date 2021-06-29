from pytest import raises
import pytest


@pytest.mark.parametrize('root, num_labels_per_class, num_classes, batch_size, num_workers', [('/home/kp600168/.torch/data/', 50, 10, 256, 4), ])
def test_register(root, num_labels_per_class, num_classes, batch_size, num_workers):
    from allinone.data import SEMI_DATASET_REGISTRY
    semi_cifar = SEMI_DATASET_REGISTRY(
        'semi_cifar10')(root, num_labels_per_class)
    train_loader, test_loader, classes = semi_cifar(
        batch_size, num_workers=num_workers)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [batch_size, 3, 32, 32]
        break
    for data, target in test_loader:
        assert list(data.shape) == [batch_size * 2, 3, 32, 32]
        break
    assert classes == num_classes


@pytest.mark.parametrize('root, num_labels_per_class, num_classes, batch_size, num_workers', [('/home/kp600168/.torch/data/', 50, 10, 256, 4), ])
def test_data(root, num_labels_per_class, num_classes, batch_size, num_workers):
    from allinone.data import SemiDataset, semi_cifar10
    from torchvision.datasets import CIFAR10
    from torchvision import transforms as tf
    semi_cifar = SemiDataset(root, num_labels_per_class, CIFAR10,
                             num_classes, tf.ToTensor(), tf.ToTensor(), tf.ToTensor())
    train_loader, test_loader, classes = semi_cifar(
        batch_size, num_workers=num_workers)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [batch_size, 3, 32, 32]
        break
    for data, target in test_loader:
        assert list(data.shape) == [batch_size * 2, 3, 32, 32]
        break
    assert classes == num_classes

    semi_cifar = semi_cifar10(root, num_labels_per_class)
    train_loader, test_loader, classes = semi_cifar(
        batch_size, num_workers=num_workers)
    for label, unlabel in train_loader:
        label_data, label_target = label
        unlabel_data, unlabel_target = unlabel
        assert list(label_data.shape) == [batch_size, 3, 32, 32]
        break
    for data, target in test_loader:
        assert list(data.shape) == [batch_size * 2, 3, 32, 32]
        break
    assert classes == num_classes


@pytest.mark.parametrize('root, num_labels_per_class, num_classes, batch_size, num_workers', [('/home/kp600168/.torch/data/', 50, 10, 256, 4), ])
def test_include_labeled_data(root, num_labels_per_class, num_classes, batch_size, num_workers):
    from allinone.data import SemiDataset
    from torchvision.datasets import MNIST
    from torchvision import transforms as tf
    mnist = MNIST(root)
    semi_mnist = SemiDataset(root, num_labels_per_class, MNIST,
                             num_classes, tf.ToTensor(), tf.ToTensor(), tf.ToTensor())
    assert len(semi_mnist.unlabel_dataset) == len(mnist)
    uninclude_semi_mnist = SemiDataset(root, num_labels_per_class, MNIST, num_classes, tf.ToTensor(
    ), tf.ToTensor(), tf.ToTensor(), include_labeled_data=False)
    assert len(uninclude_semi_mnist.unlabel_dataset) + \
        len(uninclude_semi_mnist.label_dataset) == len(mnist)


def test_import_data():
    with raises(ImportError):
        from allinone import SemiDataset, semi_mnist
