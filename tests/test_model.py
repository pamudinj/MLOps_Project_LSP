import torch
from torch import nn

from pathmnist_mlops.model import Model


def test_model_instance():
    """
    Test model is a PyTorch module.
    """

    model = Model()

    assert isinstance(model, nn.Module,)


def test_model_forward_pass():
    """
    Test model forward pass.
    """

    model = Model()

    x = torch.rand(4, 3, 28, 28,)

    output = model(x)

    assert isinstance(output, torch.Tensor,)

    assert output.shape[0] == 4


def test_model_output_dimensions():
    """
    Test output dimensions.
    """

    model = Model()

    x = torch.rand(2, 3, 28, 28,)

    output = model(x)

    assert output.ndim == 2
