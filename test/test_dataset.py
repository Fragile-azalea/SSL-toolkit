from torchvision import transforms as tf
from torchvision.datasets import MNIST
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from allinone import *
    print(SEMI_DATASET_REGISTRY.catalogue())
    dataset = SEMI_DATASET_REGISTRY('Mix')(MNIST, '/home/kp600168/.torch/data/', 10, [
        tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))], semi_size=59900)
    train_loader, test_loader = dataset(256, num_workers=4)
    dataset = SEMI_DATASET_REGISTRY('Split')(MNIST, '/home/kp600168/.torch/data/', 10, [
        tf.Resize((32, 32)), tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))], semi_size=59000)
    train_loader, test_loader, unlabel_loader, return_num_classes = dataset(
        256, num_workers=4, unlabel_batch_size=100, return_num_classes=True)
    print(train_loader, test_loader, unlabel_loader, return_num_classes)

    for data in train_loader:
        input, target = data
        print(input.shape, target.shape)
