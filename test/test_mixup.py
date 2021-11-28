from DeSSL.transforms import mixup_for_one_hot
import torch


def test_mixup():
    input = torch.randn((32, 3, 64, 64))
    target = torch.randn((32, 10))
    gamma = torch.rand((32,))
    indices = torch.randperm(input.size(
        0), device=input.device, dtype=torch.long)
    mix_input, mix_target = mixup_for_one_hot(input, target, gamma, indices)
