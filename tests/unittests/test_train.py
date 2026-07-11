import torch
from torch import nn
from torch.optim import Adam

from pathmnist_mlops.train import PathMNISTClassifier


def test_classifier_initialization():
    """Test classifier initialization."""

    model = PathMNISTClassifier()

    assert isinstance(model.model, nn.Module)
    assert isinstance(model.criterion, nn.CrossEntropyLoss)


def test_classifier_forward():
    """Test classifier forward pass."""

    model = PathMNISTClassifier()

    x = torch.randn(2, 3, 28, 28)

    output = model(x)

    assert output.shape == (2, 9)


def test_configure_optimizers():
    """Test optimizer creation."""

    model = PathMNISTClassifier()

    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, Adam)


def test_training_step():
    """Test training step returns a loss."""

    model = PathMNISTClassifier()

    images = torch.randn(4, 3, 28, 28)
    labels = torch.randint(0, 9, (4,))

    batch = (images, labels)

    loss = model.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_validation_step():
    """Test validation step executes."""

    model = PathMNISTClassifier()

    images = torch.randn(4, 3, 28, 28)
    labels = torch.randint(0, 9, (4,))

    batch = (images, labels)

    result = model.validation_step(batch, 0)

    assert result is None


def test_optimizer_learning_rate():
    """Test optimizer learning rate."""

    model = PathMNISTClassifier(lr=5e-4)

    optimizer = model.configure_optimizers()

    assert optimizer.param_groups[0]["lr"] == 5e-4


def test_hyperparameters_saved():
    """Test Lightning hyperparameters."""

    model = PathMNISTClassifier(lr=0.002)

    assert model.hparams.lr == 0.002
