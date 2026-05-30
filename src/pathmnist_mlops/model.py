import torch
from torch import nn


class Model(nn.Module):
    """
    Convolutional Neural Network for PathMNIST classification.

    The network consists of two convolutional blocks followed by
    a fully connected classifier. It takes RGB images of size
    28x28 as input and predicts one of the 9 PathMNIST classes.
    """

    def __init__(self, num_classes: int = 9):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 28, 28).

        Returns:
            Output logits of shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = Model()
    x = torch.rand(1, 3, 28, 28)
    print(f"Output shape of model: {model(x).shape}")
