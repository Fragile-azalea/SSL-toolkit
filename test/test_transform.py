from torchvision import transforms as tf
from allinone import TRANSFORM_REGISTRY
from PIL import Image
import torch

def test_many_times_transfrom():
    input = Image.new('RGB', (224, 224), color = 'white')
    transform = TRANSFORM_REGISTRY('ManyTimes')
    transform = transform(tf.Compose([tf.Resize((32, 32)), tf.ToTensor()]), 3)
    output = transform(input)
    assert isinstance(output, tuple) 
    assert len(output) == 3
    assert isinstance(output[0], torch.Tensor)
    assert output[0].shape == torch.Size([3, 32, 32])
    
def test_twice_transfrom():
    input = Image.new('RGB', (224, 224), color = 'white')
    transform = TRANSFORM_REGISTRY('Twice')
    transform = transform(tf.Compose([tf.Resize((64, 64)), tf.ToTensor()]))
    output = transform(input)
    assert isinstance(output, tuple) 
    assert len(output) == 2
    assert isinstance(output[0], torch.Tensor)
    assert output[0].shape == torch.Size([3, 64, 64])

def test_random_augment_transfrom():
    input = Image.new('RGB', (224, 224), color = 'white')
    transform = TRANSFORM_REGISTRY('RandAugment')
    transform = transform(5, 3)
    intermediate = transform(input)
    assert isinstance(intermediate, Image.Image)
    norm = tf.Compose([tf.Resize((64, 64)), tf.ToTensor()])
    output = norm(intermediate)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([3, 64, 64])