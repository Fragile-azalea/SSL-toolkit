from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms as tf
from tqdm import tqdm
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from allinone.data.data import SemiDataset, semi_mnist

    # x = SemiDataset(10, '/home/kp600168/.torch/data/', CIFAR10,
    #                 10, tf.ToTensor(), tf.ToTensor(), tf.ToTensor())
    # a, b, c = x(64, num_workers=4)
    # for i in a:
    #     print(i)

    # x = SemiDataset(10, '/home/kp600168/.torch/data/', MNIST,
    #                 10, tf.ToTensor(), tf.ToTensor(), tf.ToTensor())
    # a, b, c = x(64, num_workers=4)
    # for i in a:
    #     print(i)

    x = semi_mnist(50, '/home/kp600168/.torch/data/')
    a, b, c = x(256, num_workers=4)
    with tqdm(a) as t:
        for data in t:
            pass
