from pathlib import Path

import torch

from pathmnist_mlops.evaluate import load_model_from_wandb


def export_onnx() -> None:
    """Export the best trained PathMNIST model from the W&B Model Registry
    to ONNX format for deployment with the FastAPI inference service."""

    model = load_model_from_wandb(device=torch.device("cpu"))

    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        28,
        28,
    )

    Path("models").mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),
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
