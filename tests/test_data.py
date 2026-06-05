import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from pathmnist_mlops.data import (
    MyDataset,
    get_dataloaders,
)


def test_dataset_instance():
    """
    Test dataset is a PyTorch Dataset.
    """

    dataset = MyDataset(
        split="train",
    )

    assert isinstance(
        dataset,
        Dataset,
    )


def test_dataset_sample():
    """
    Test one dataset sample.
    """

    dataset = MyDataset(
        split="train",
        transform=transforms.ToTensor(),
    )

    image, label = dataset[0]

    assert isinstance(
        image,
        torch.Tensor,
    )

    assert isinstance(
        label,
        torch.Tensor,
    )


def test_dataloader_batch():
    """
    Test dataloader batch shapes.
    """

    train_loader, _, _ = get_dataloaders(batch_size=4)

    images, labels = next(iter(train_loader))

    assert images.shape[0] == 4

    assert labels.shape[0] == 4

    assert isinstance(
        train_loader,
        DataLoader,
    )
