import argparse
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import yaml


def main(args):
    MNIST(args.data_path, download=True)
    FashionMNIST(args.data_path, download=True)
    CIFAR10(args.data_path, download=True)

    with open('config/base.yml', 'r') as fin:
        cfg = yaml.load(fin.read(), yaml.SafeLoader)
    cfg['root'] = args.data_path
    with open('config/base.yml', 'w') as fin:
        yaml.dump(cfg, fin, default_flow_style=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Download")
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
