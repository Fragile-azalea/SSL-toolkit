import torch
from torch import nn
from torch.nn import functional as F
from homura.vision import MODEL_REGISTRY


@MODEL_REGISTRY.register
class VAE(nn.Module):
    '''
    Based on `VAE <https://github.com/pytorch/examples/blob/master/vae/main.py>`_.
    '''

    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(794, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        x = torch.cat((x.view(-1, 784), y), dim=1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == "__main__":
    vae = VAE()
    y = vae(torch.randn((3, 1, 28, 28)))
    print(y[0].shape)
