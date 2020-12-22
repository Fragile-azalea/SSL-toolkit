from allinone import MODEL_REGISTRY
from torch.nn import Module
from allinone.model import LeNet5
import torch
from pytest import raises


def test_model_registry():
    lenet5 = MODEL_REGISTRY('lenet5')
    assert isinstance(lenet5(), Module)


def test_lenet5():
    lenet5 = LeNet5()
    lenet5.eval()
    input = torch.randn((16, 1, 32, 32))
    with torch.no_grad():
        output = lenet5(input)
    assert output.shape == torch.Size([16, 10])


def test_import_model():
    with raises(ImportError):
        from allinone import LeNet5
