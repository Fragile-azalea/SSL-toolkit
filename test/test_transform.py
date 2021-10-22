from DeSSL import transforms
from DeSSL.data.dataset import semi_cifar10
from DeSSL.transforms import mixup, IntegerMixLoss, OneHotMixLoss, CIFAR10Policy
from torchvision import transforms as tf
from DeSSL import TRANSFORM_REGISTRY
from PIL import Image
import torch
import pytest


def test_many_times_transfrom():
    input = Image.new('RGB', (224, 224), color='white')
    transform = TRANSFORM_REGISTRY('ManyTimes')
    transform = transform(tf.Compose([tf.Resize((32, 32)), tf.ToTensor()]), 3)
    output = transform(input)
    assert isinstance(output, tuple)
    assert len(output) == 3
    assert isinstance(output[0], torch.Tensor)
    assert output[0].shape == torch.Size([3, 32, 32])


def test_twice_transfrom():
    input = Image.new('RGB', (224, 224), color='white')
    transform = TRANSFORM_REGISTRY('Twice')
    transform = transform(tf.Compose([tf.Resize((64, 64)), tf.ToTensor()]))
    output = transform(input)
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert isinstance(output[0], torch.Tensor)
    assert output[0].shape == torch.Size([3, 64, 64])


def test_random_augment_transfrom():
    input = Image.new('RGB', (224, 224), color='white')
    transform = TRANSFORM_REGISTRY('RandAugment')
    transform = transform(5, 3)
    intermediate = transform(input)
    assert isinstance(intermediate, Image.Image)
    norm = tf.Compose([tf.Resize((64, 64)), tf.ToTensor()])
    output = norm(intermediate)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([3, 64, 64])


def test_autoaugment_transfrom():
    from DeSSL import loadding_config

    parser = loadding_config('config/base.yml')
    args = parser.parse_args([])

    from torchvision.datasets import CIFAR10
    cifar = CIFAR10(args.root, transform=tf.Compose(
        [tf.RandomCrop(32, 4, fill=128), CIFAR10Policy()]))
    for data, target in cifar:
        assert isinstance(data, Image.Image) == True
        break


def test_mixup():
    device = torch.device('cuda:0')
    input = torch.randn([256, 3, 32, 32], device=device)
    target = torch.randn([256, 10], device=device)
    target = torch.nn.functional.softmax(target, dim=-1)

    mix_input, target_a, target_b = TRANSFORM_REGISTRY(
        'mixup_for_integer')(input, target.argmax(dim=-1), 0.1)
    assert mix_input.shape == torch.Size([256, 3, 32, 32])
    assert torch.equal(target.argmax(dim=-1), target_a)
    assert target_b.shape == torch.Size([256])

    mix_input, mix_target = TRANSFORM_REGISTRY(
        'mixup_for_one_hot')(input, target, 0.1)
    assert mix_input.shape == torch.Size([256, 3, 32, 32])
    assert mix_target.shape == torch.Size([256, 10])

    indices = torch.randperm(input.size(
        0), device=input.device, dtype=torch.long)
    mix_input, mix_target = TRANSFORM_REGISTRY(
        'mixup_for_one_hot')(input, target, 0.1, indices)
    assert (input * 0.1 + input[indices] * 0.9 - mix_input).abs().max() < 1e-5
    assert (target * 0.1 + target[indices] *
            0.9 - mix_target).abs().max() < 1e-5


def test_mixloss():
    input = torch.randn([256, 10])
    label = torch.randint(0, 10, (256, ))
    gamma = 0.9
    mix_input, label, perm_label = mixup.mixup_for_integer(input, label, gamma)
    criterion = IntegerMixLoss()
    loss = criterion(mix_input, gamma, label, perm_label)
    assert isinstance(loss, torch.Tensor)
    label = torch.rand([256, 10])
    mix_input, mix_label = mixup.mixup_for_one_hot(input, label, gamma)
    one_hot = OneHotMixLoss()
    loss = one_hot(mix_input, mix_label)
    assert isinstance(loss, torch.Tensor)
