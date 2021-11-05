from DeSSL import Registry
MODEL_REGISTRY = Registry('model')
while True:
    from .lenet import LeNet5, LeNet5_SVHN
    from .ALImodel import Generator_x, Generator_z, Discriminator_x, Discriminator_z, Discriminator_x_z
    from .VAE import VAE
    from .toy import ToyNet
    from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d
    break
