import torch
import torch.nn as nn
from . import MODEL_REGISTRY


@MODEL_REGISTRY.register
class LeNet5(nn.Module):
    """
    The `LeNet-5 <https://ieeexplore.ieee.org/document/726791>`_ for measuring the performance of the algorithms on MNIST.

    Args:
        num_classes: The number of categories. 
    """

    def __init__(self, num_classes: int = 10):
        super(LeNet5, self).__init__()
        # 32 * 32 -> 28 * 28
        self.c1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        # 28 * 28 -> 14 * 14
        self.s2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        # 14 * 14 -> 10 * 10
        self.c3 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        # 10 * 10 -> 5 * 5
        self.s4 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        # 5 * 5 -> 1 * 1
        self.c5 = nn.Conv2d(16, 120, 5)
        self.bn3 = nn.BatchNorm2d(120)
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, num_classes)

    def forward(self, img):
        output = self.c1(img)
        output = self.bn1(output)
        output = torch.tanh(output)
        output = self.s2(output)
        output = self.c3(output)
        output = self.bn2(output)
        output = torch.tanh(output)
        output = self.s4(output)
        output = self.c5(output)
        output = self.bn3(output)
        output = torch.tanh(output)
        output = output.view(img.size(0), -1)
        output = self.fc6(output)

        output = torch.tanh(output)
        output = self.fc7(output)
        return output


@MODEL_REGISTRY.register
class LeNet5_SVHN(nn.Module):
    """
    The `LeNet-5 <https://ieeexplore.ieee.org/document/726791>`_ for measuring the performance of the algorithms on SVHN.

    Args:
        num_classes: The number of categories. 
    """

    def __init__(self, num_classes: int = 10):
        super(LeNet5_SVHN, self).__init__()
        # 32 * 32 -> 28 * 28
        self.c1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        # 28 * 28 -> 14 * 14
        self.s2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        # 14 * 14 -> 10 * 10
        self.c3 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        # 10 * 10 -> 5 * 5
        self.s4 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        # 5 * 5 -> 1 * 1
        self.c5 = nn.Conv2d(16, 120, 5)
        self.bn3 = nn.BatchNorm2d(120)
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, num_classes)

    def forward(self, img):
        output = self.c1(img)
        output = self.bn1(output)
        output = torch.tanh(output)
        output = self.s2(output)
        output = self.c3(output)
        output = self.bn2(output)
        output = torch.tanh(output)
        output = self.s4(output)
        output = self.c5(output)
        output = self.bn3(output)
        output = torch.tanh(output)
        output = output.view(img.size(0), -1)
        output = self.fc6(output)

        output = torch.tanh(output)
        output = self.fc7(output)
        return output
