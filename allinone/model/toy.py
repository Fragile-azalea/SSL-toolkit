from torch import nn
import torch
from homura.vision import MODEL_REGISTRY
from math import prod
from typing import List
from functools import partial


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


def batch_normalization(batch, return_mean_and_std=False):
    batch_mean = torch.mean(batch, dim=0, keepdim=True)
    batch_std = torch.std(batch, dim=0, keepdim=True)
    batch = (batch - batch_mean) / (batch_std + 1e-7)
    if return_mean_and_std:
        return batch, batch_mean, batch_std
    else:
        return batch


@MODEL_REGISTRY.register
class Ladder_MLP(nn.Module):
    '''
    Args:
        input_shape: The shape of inputs.
        num_neurons: The list of neurons.
        sigma_noise: The list of the :math:`\sigma` of gaussian noise.
    '''

    def __init__(
        self,
        input_shape: tuple,
        num_neurons: List[int],
        sigma_noise: List[float],
    ):
        super(Ladder_MLP, self).__init__()
        input_dim = prod(input_shape)
        input_dim_list = [input_dim] + num_neurons[:-1]
        self.encoder = nn.ModuleList([nn.Linear(input_dim,
                                                output_dim, False) for input_dim,
                                      output_dim in zip(input_dim_list, num_neurons)])
        self.decoder = nn.ModuleList([nn.Linear(output_dim,
                                                input_dim, False) for input_dim,
                                      output_dim in zip(input_dim_list, num_neurons)])
        self.bn = nn.ModuleList(
            [nn.BatchNorm1d(dim, affine=False) for dim in num_neurons])
        self.inv_bn = nn.ModuleList(
            [nn.BatchNorm1d(dim, affine=False) for dim in num_neurons])
        self.factor = nn.ParameterList(
            [nn.Parameter(torch.ones(dim)) for dim in num_neurons])
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for dim in num_neurons])
        self.sigma_noise = sigma_noise

        self.a0 = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for dim in num_neurons])
        self.a1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for dim in num_neurons])
        self.a2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for dim in num_neurons])
        self.a3 = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for dim in num_neurons])
        self.a4 = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for dim in num_neurons])
        self.a5 = nn.ParameterList(
            [nn.Parameter(torch.ones(dim)) for dim in num_neurons])
        self.a6 = nn.ParameterList(
            [nn.Parameter(torch.ones(dim)) for dim in num_neurons])
        self.a7 = nn.ParameterList(
            [nn.Parameter(torch.ones(dim)) for dim in num_neurons])
        self.a8 = nn.ParameterList(
            [nn.Parameter(torch.ones(dim)) for dim in num_neurons])
        self.a9 = nn.ParameterList(
            [nn.Parameter(torch.ones(dim)) for dim in num_neurons])

    def clear_path(self, *input):
        self.z = []
        self.mean = []
        self.std = []
        input = [*map(partial(torch.flatten, start_dim=1), input)]
        for net, bn, factor, bias in zip(
                self.encoder, self.bn, self.factor, self.bias):
            input = [*map(net, input)]
            input[0] = bn(input[0])
            if len(input) == 2:
                input[1], batch_mean, batch_std = batch_normalization(
                    input[1], True)
                self.mean.append(batch_mean)
                self.std.append(batch_std)
                self.z.append(input[1])
            if net == self.encoder[-1]:  # last layer
                output = [*map(lambda x: x * factor, input)]
                output = [*map(partial(nn.functional.softmax, dim=1), output)]
            else:
                output = [*map(lambda x: x + bias, input)]
                output = [
                    *map(partial(nn.functional.relu, inplace=True), output)]
        return output[0]

    def noise_path(self, *input):
        self.noise_z = []
        input = map(partial(torch.flatten, start_dim=1), input)
        for net, bn, factor, bias, sigma in zip(
                self.encoder, self.bn, self.factor, self.bias, self.sigma_noise):
            input = [*map(net, input)]
            input = [*map(batch_normalization, input)]
            input = [*map(lambda x: x + torch.randn_like(x) * sigma, input)]
            self.noise_z.append(input[1])
            if net == self.encoder[-1]:  # last layer
                output = map(lambda x: x * factor, input)
                output = [*map(partial(nn.functional.softmax, dim=1), output)]
                self.noise_h = output[1]
            else:
                output = map(lambda x: x + bias, input)
                output = [
                    *map(partial(nn.functional.relu, inplace=True), output)]
        return output[0]

    def decoder_path(self):
        input = self.noise_h
        self.denoise_z = []
        for noise_z, net, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 in zip(self.noise_z[::-1],
                                                                        self.decoder[::-1],
                                                                        self.a0[::-1],
                                                                        self.a1[::-1],
                                                                        self.a2[::-1],
                                                                        self.a3[::-1],
                                                                        self.a4[::-1],
                                                                        self.a5[::-1],
                                                                        self.a6[::-1],
                                                                        self.a7[::-1],
                                                                        self.a8[::-1],
                                                                        self.a9[::-1]):
            '''Normalization'''
            input = batch_normalization(input)
            mu = a0 * torch.sigmoid(a1 * input + a2) + a3 * input + a4
            std = a5 * torch.sigmoid(a6 * input + a7) + a8 * input + a9
            input = (noise_z - mu) * std + mu
            self.denoise_z.append(input)
            input = net(input)
        self.denoise_z = self.denoise_z[::-1]

    def forward(self, path_name, *input):
        if path_name == 'clear':
            return self.clear_path(*input)
        elif path_name == 'noise':
            return self.noise_path(*input)
        else:
            self.decoder_path()

    def get_loss_d(self, lam_list):
        loss = 0.
        for lam, denoise_z, z, mean, std in zip(
                lam_list, self.denoise_z, self.z, self.mean, self.std):
            denoise_z = (denoise_z - mean) / (std + 1e-7)
            loss = loss + lam * \
                torch.mean(torch.norm(denoise_z - z, dim=1) / z.shape[1])
        return loss


if __name__ == '__main__':
    mlp = Ladder_MLP(
        (28, 28), [
            1000, 500, 250, 250, 250, 10, ], [
            0.3, 0.3, 0.3, 0.3, 0.3, 0.3, ])
    label = torch.randn([128, 1, 28, 28])
    unlabel = torch.randn([256, 1, 28, 28])
    output = mlp('clear', label)
    print(output.shape)
    output = mlp('clear', label, unlabel)
    print(output.shape)
    output = mlp('noise', label, unlabel)
    print(output.shape)
    mlp('decoder')
    print(mlp.get_loss_d([1000., 10., 0.1, 0.1, 0.1, 0.1, 0.1]))
