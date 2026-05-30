from pathlib import Path

# import typer
import torch
from medmnist import PathMNIST  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore


class MyDataset(Dataset):
    """My custom dataset wrapping PathMNIST."""

    def __init__(self, split: str = "train", transform=None) -> None:
        self.dataset = PathMNIST(split=split, transform=transform, download=True)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

        image, label = self.dataset[index]
        label = torch.tensor(label).long().squeeze()
        return image, label


def get_dataloaders(batch_size: int = 32):
    transform = transforms.ToTensor()

    train = MyDataset(split="train", transform=transform)
    val = MyDataset(split="val", transform=transform)
    test = MyDataset(split="test", transform=transform)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=7), # identified bottleneck using profiling: set num_workers = 7
        DataLoader(val, batch_size=batch_size, num_workers=7),
        DataLoader(test, batch_size=batch_size, num_workers=7),
    )


def preprocess(data_path: Path, output_folder: Path) -> None:
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )
    print("PathMNIST requires no additional preprocessing.")


if __name__ == "__main__":
    train_loader, _, _ = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Label: {labels[0]}")
