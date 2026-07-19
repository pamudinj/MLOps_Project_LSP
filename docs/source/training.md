# Training Pipeline

The training pipeline is implemented using PyTorch Lightning.

Main components:

- Hydra configuration
- PathMNIST dataloaders
- CNN classifier
- Early stopping
- Model checkpointing
- W&B experiment tracking
- Vertex AI training jobs

During training the following metrics are logged:

- Training loss
- Training accuracy
- Validation loss
- Validation accuracy

The best checkpoint is uploaded automatically to the Weights & Biases Model Registry.