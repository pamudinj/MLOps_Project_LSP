from pathlib import Path

import torch
from data import get_dataloaders
from model import Model
from torch import nn
from torch.optim import Adam


def train():
    """
    Train a CNN on the PathMNIST dataset.

    The function loads the training and validation datasets,
    trains the model for a fixed number of epochs, evaluates
    validation accuracy after each epoch, and saves the
    trained model to disk.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders(batch_size=64)

    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),lr=1e-3)
    epochs = 20

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs} | "f"Loss: {train_loss:.4f} | " f"Val Acc: {val_acc:.4f}")

    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(),"models/pathmnist_cnn.pt")
    print("Model saved.")


if __name__ == "__main__":
    train()
