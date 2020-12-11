from torch import nn
from homura.vision import MODEL_REGISTRY

@MODEL_REGISTRY.register
class ToyNet(nn.Module):
    '''
    The toy NN for measuring the performance of the algorithms on MNIST.
    
    Args:
        num_classes: The number of categories. 
    '''
    def __init__(self, num_classes: int):
        super(ToyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(784, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, num_classes),
        )
    def forward(self, input):
        return self.net(input)