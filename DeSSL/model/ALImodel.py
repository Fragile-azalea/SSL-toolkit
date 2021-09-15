from torch import nn, cat
from . import MODEL_REGISTRY

__all__ = ['Generator_z',
           'Generator_x',
           'Discriminator_x',
           'Discriminator_z',
           'Discriminator_x_z',
           ]


def reset_all_parameters(module):
    if hasattr(module, 'weight'):
        nn.init.normal_(module.weight, std=0.01)
    if hasattr(module, 'bias'):
        nn.init.zeros_(module.bias)


@MODEL_REGISTRY.register
class Generator_z(nn.Module):
    '''
    The Generator of z for CIFAR10 reported by `Adversarially Learned Inference <https://arxiv.org/pdf/1606.00704>`_.
    '''

    def __init__(self):
        super(Generator_z, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, 64, 4, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, 4),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, 4, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 512, 4),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(512, 64, 1),
        )
        self.apply(reset_all_parameters)

    def forward(self, input):
        return self.net(input)


@MODEL_REGISTRY.register
class Generator_x(nn.Module):
    '''
    The Generator of x for CIFAR10 reported by `Adversarially Learned Inference <https://arxiv.org/pdf/1606.00704>`_.
    '''

    def __init__(self):
        super(Generator_x, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 256, 4),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(64, 32, 4, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(32, 32, 5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(32, 32, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(32, 3, 1),
            nn.Tanh(),
        )
        self.apply(reset_all_parameters)

    def forward(self, input):
        return self.net(input)


class Maxout(nn.Module):
    def __init__(self, module, *args, pieces=2, **kwargs):
        super(Maxout, self).__init__()
        self.list = [module(*args, **kwargs) for _ in range(pieces)]
        self.list = nn.ModuleList(self.list)
        self.pieces = pieces

    def forward(self, input):
        output = [self.list[i](input) for i in range(self.pieces)]
        ret = output[0]
        for i in range(1, self.pieces):
            ret = ret.max(output[i])
        return ret


@MODEL_REGISTRY.register
class Discriminator_x(nn.Module):
    '''
    The Discriminator of x for CIFAR10 reported by `Adversarially Learned Inference <https://arxiv.org/pdf/1606.00704>`_.
    '''

    def __init__(self):
        super(Discriminator_x, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout2d(0.2),
            Maxout(nn.Conv2d, 3, 32, 5),
            nn.Dropout2d(0.5),
            Maxout(nn.Conv2d, 32, 64, 4, 2),
            nn.Dropout2d(0.5),
            Maxout(nn.Conv2d, 64, 128, 4),
            nn.Dropout2d(0.5),
            Maxout(nn.Conv2d, 128, 256, 4, 2),
            nn.Dropout2d(0.5),
            Maxout(nn.Conv2d, 256, 512, 4),
        )
        self.apply(reset_all_parameters)

    def forward(self, input):
        return self.net(input)


@MODEL_REGISTRY.register
class Discriminator_z(nn.Module):
    '''
    The Discriminator of z for CIFAR10 reported by `Adversarially Learned Inference <https://arxiv.org/pdf/1606.00704>`_.
    '''

    def __init__(self):
        super(Discriminator_z, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout2d(0.2),
            Maxout(nn.Conv2d, 64, 512, 1),
            nn.Dropout2d(0.5),
            Maxout(nn.Conv2d, 512, 512, 1),
        )
        self.apply(reset_all_parameters)

    def forward(self, input):
        return self.net(input)


@MODEL_REGISTRY.register
class Discriminator_x_z(nn.Module):
    '''
    The Discriminator of x and z for CIFAR10 reported by `Adversarially Learned Inference <https://arxiv.org/pdf/1606.00704>`_.
    '''

    def __init__(self, num_classes: int = 10):
        super(Discriminator_x_z, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout2d(0.5),
            Maxout(nn.Conv2d, 1024, 1024, 1),
            nn.Dropout2d(0.5),
            Maxout(nn.Conv2d, 1024, 1024, 1),
            nn.Dropout2d(0.5),
            nn.Flatten(),
        )
        self.fc = nn.Linear(1024, num_classes)
        self.discriminator = nn.Linear(1024, 1)
        self.apply(reset_all_parameters)

    def forward(self, x, z):
        x = self.net(cat((x, z), dim=1))
        return self.fc(x), self.discriminator(x)
