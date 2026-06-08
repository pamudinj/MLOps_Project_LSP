from pathlib import Path

import torch

from pathmnist_mlops.train import (
    PathMNISTClassifier,
)


def export_onnx() -> None:
    """
    Export the trained PathMNIST model
    to ONNX format.
    """

    checkpoint_path = "models/pathmnist_cnn-epoch=14-val_acc=0.8864.ckpt"

    model = PathMNISTClassifier.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )

    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        28,
        28,
    )

    Path("models").mkdir(exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        "models/pathmnist_model.onnx",
        export_params=True,
        opset_version=18,
        dynamo=False,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print("ONNX model exported successfully.")


if __name__ == "__main__":
    export_onnx()
