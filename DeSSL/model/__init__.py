from DeSSL import Registry
MODEL_REGISTRY = Registry('model')
while True:
    from .lenet import LeNet5, LeNet5_SVHN
    from .ALImodel import Generator_x, Generator_z, Discriminator_x, Discriminator_z, Discriminator_x_z
    from .VAE import VAE
    from .toy import ToyNet
    break
