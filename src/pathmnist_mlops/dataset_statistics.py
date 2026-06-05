from torchvision import transforms

from pathmnist_mlops.data import MyDataset


def dataset_statistics() -> None:
    """
    Print basic PathMNIST
    dataset statistics.
    """

    train_dataset = MyDataset(
        split="train",
        transform=transforms.ToTensor(),
    )

    print("PathMNIST Dataset")

    print(f"Number of training samples: {len(train_dataset)}")

    image, label = train_dataset[0]

    print(f"Image shape: {image.shape}")

    print(f"Example label: {label}")


if __name__ == "__main__":
    dataset_statistics()
