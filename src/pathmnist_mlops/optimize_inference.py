"""
Benchmark inference-speed optimizations for the PathMNIST model (M31).

Investigates the effect of quantization, torch.compile and pruning on
inference latency/throughput, following the exercises in
https://skaftenicki.github.io/dtu_mlops/latest/s9_scalable_applications/inference/

Usage:
    # with a trained checkpoint on disk
    python src/pathmnist_mlops/optimize_inference.py --checkpoint models/pathmnist_cnn-....ckpt

    # pull the best model from the W&B registry (needs .env with WANDB_* vars)
    python src/pathmnist_mlops/optimize_inference.py --use-wandb

    # no checkpoint available: benchmarks a randomly initialised model.
    # Latency numbers are still meaningful (they only depend on the
    # architecture/graph, not the weight values) but accuracy numbers are not.
    python src/pathmnist_mlops/optimize_inference.py
"""

import copy
import io
import logging
import time
from pathlib import Path

import torch
import typer
from torch import nn
from torch.nn.utils import prune

from pathmnist_mlops.evaluate import load_model_from_wandb
from pathmnist_mlops.train import PathMNISTClassifier

torch.manual_seed(55)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parents[2] / "reports"
IMAGE_SHAPE = (3, 28, 28)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_baseline_model(checkpoint: str | None, use_wandb: bool, device: torch.device) -> nn.Module:
    """Load the trained PathMNIST CNN, falling back to random init if unavailable."""
    if checkpoint:
        logger.info(f"Loading checkpoint from {checkpoint}")
        lit_model = PathMNISTClassifier.load_from_checkpoint(checkpoint, map_location=device)
    elif use_wandb:
        logger.info("Pulling best model artifact from W&B...")
        lit_model = load_model_from_wandb(device)
    else:
        logger.warning(
            "No --checkpoint given and --use-wandb not set: benchmarking a randomly "
            "initialised model. Latency/throughput/size numbers are still valid since "
            "they only depend on the architecture, not the trained weights - but any "
            "accuracy numbers below are meaningless and should be ignored."
        )
        lit_model = PathMNISTClassifier()

    lit_model.eval()
    return lit_model.model.to(device)


def try_get_test_loader(batch_size: int):
    """Best-effort attempt to get a real test dataloader for accuracy checks."""
    try:
        from pathmnist_mlops.data import get_dataloaders

        _, _, test_loader = get_dataloaders(batch_size=batch_size)
        return test_loader
    except Exception as e:  # noqa: BLE001 - purely best-effort, data may not be downloadable here
        logger.warning(f"Could not load PathMNIST test data ({e}). Skipping accuracy checks.")
        return None


# --------------------------------------------------------------------------- #
# Benchmark utilities
# --------------------------------------------------------------------------- #
def benchmark_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_reps: int = 50,
    warmup: int = 10,
) -> tuple[float, float]:
    """Return (avg latency in ms per batch, throughput in images/sec)."""
    model.eval()
    device = input_tensor.device
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

        tic = time.perf_counter()
        for _ in range(n_reps):
            _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        toc = time.perf_counter()

    avg_ms = (toc - tic) / n_reps * 1000
    throughput = input_tensor.shape[0] / (avg_ms / 1000)
    return avg_ms, throughput


def model_size_mb(model: nn.Module) -> float:
    """Size in MB of the model's state_dict when serialized (no disk write needed)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024**2)


def accuracy(model: nn.Module, test_loader, device: torch.device) -> float | None:
    if test_loader is None:
        return None
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long()
            preds = model(images).argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total if total else None


# --------------------------------------------------------------------------- #
# Optimization variants
# --------------------------------------------------------------------------- #
def select_quantization_engine() -> str | None:
    """
    Pick a usable quantized-CPU backend and activate it.

    torch.backends.quantized.engine defaults to "none" until set explicitly, even
    when a real engine (e.g. qnnpack on Apple Silicon, x86/fbgemm on Intel/AMD) is
    compiled into the wheel. Without this, every quantized op raises:
    "RuntimeError: Didn't find engine for operation ... NoQEngine".
    """
    engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if not engines:
        logger.warning(
            "No quantized CPU engine (qnnpack/fbgemm/x86) is available in this PyTorch "
            "build. Quantization variants will be skipped."
        )
        return None
    engine = (
        "x86"
        if "x86" in engines
        else ("fbgemm" if "fbgemm" in engines else ("qnnpack" if "qnnpack" in engines else engines[0]))
    )
    torch.backends.quantized.engine = engine
    return engine


def apply_dynamic_quantization(model: nn.Module) -> nn.Module | None:
    """Dynamic (weight-only) int8 quantization of Linear layers. CPU only."""
    if select_quantization_engine() is None:
        return None
    try:
        return torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(model).to("cpu"),
            {nn.Linear},
            dtype=torch.qint8,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Dynamic quantization failed ({e}). Skipping this variant.")
        return None


def apply_static_quantization(model: nn.Module, calibration_loader) -> nn.Module | None:
    """Full int8 (weights + activations) static quantization. CPU only, needs calibration data."""
    try:
        engine = select_quantization_engine()
        if engine is None:
            return None

        fp32_model = copy.deepcopy(model).to("cpu").eval()
        wrapped = torch.ao.quantization.QuantWrapper(fp32_model)
        wrapped.qconfig = torch.ao.quantization.get_default_qconfig(engine)
        torch.ao.quantization.prepare(wrapped, inplace=True)

        # Calibrate with a handful of representative batches (real data if we have it,
        # otherwise random tensors matching the input distribution as a fallback).
        with torch.no_grad():
            if calibration_loader is not None:
                for i, (images, _) in enumerate(calibration_loader):
                    wrapped(images)
                    if i >= 10:
                        break
            else:
                logger.warning(
                    "No real data available for calibration - using random tensors. "
                    "Static quantization accuracy will not be representative."
                )
                for _ in range(10):
                    wrapped(torch.rand(32, *IMAGE_SHAPE))

        torch.ao.quantization.convert(wrapped, inplace=True)
        return wrapped
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Static quantization failed ({e}). Skipping this variant.")
        return None


def apply_pruning(model: nn.Module, amount: float = 0.3, method: str = "local") -> nn.Module:
    """L1 unstructured pruning of all Conv2d/Linear weights, made permanent."""
    pruned = copy.deepcopy(model)
    prunable = [(m, "weight") for m in pruned.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    if method == "global":
        prune.global_unstructured(prunable, pruning_method=prune.L1Unstructured, amount=amount)
    else:
        for module, name in prunable:
            prune.l1_unstructured(module, name=name, amount=amount)

    for module, name in prunable:
        prune.remove(module, name)

    return pruned


def sparsity_report(model: nn.Module) -> str:
    lines = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            sparsity = 100 * float(torch.sum(module.weight == 0) / module.weight.numel())
            lines.append(f"  {name}: {sparsity:.1f}% zeroed weights")
    return "\n".join(lines)


def apply_compile(model: nn.Module, dummy_input: torch.Tensor) -> nn.Module | None:
    """
    torch.compile() itself never raises - compilation is lazy and only happens on the
    first real forward pass. So we have to trigger that pass here, inside the try/except,
    to actually catch backend/toolchain failures (e.g. missing C++ compiler headers).
    """
    try:
        compiled = torch.compile(model)
        with torch.no_grad():
            compiled(dummy_input)
        return compiled
    except Exception as e:  # noqa: BLE001
        logger.warning(f"torch.compile failed ({e}). Skipping this variant.")
        try:
            torch._dynamo.reset()
        except Exception:  # noqa: BLE001
            pass
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def optimize_inference(
    checkpoint: str = typer.Option(None, help="Path to a trained .ckpt file."),
    use_wandb: bool = typer.Option(False, help="Pull the best model from the W&B registry instead."),
    device: str = typer.Option("cpu", help="Device for the baseline/pruned/compiled variants."),
    batch_size: int = typer.Option(64, help="Batch size used for the benchmark input."),
    n_reps: int = typer.Option(50, help="Number of timed forward passes per variant."),
    warmup: int = typer.Option(10, help="Number of untimed warmup forward passes."),
    prune_amount: float = typer.Option(0.3, help="Fraction of weights to prune."),
) -> None:
    """Benchmark quantization, pruning and torch.compile against the baseline model."""
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device == "cuda" and dev.type == "cpu":
        logger.warning("CUDA requested but not available, falling back to CPU.")

    baseline = load_baseline_model(checkpoint, use_wandb, dev)
    dummy_input = torch.rand(batch_size, *IMAGE_SHAPE, device=dev)
    test_loader = try_get_test_loader(batch_size)

    results = []

    def record(name: str, model: nn.Module | None, input_tensor: torch.Tensor):
        if model is None:
            return
        try:
            latency_ms, throughput = benchmark_latency(model, input_tensor, n_reps, warmup)
            acc = accuracy(model, test_loader, input_tensor.device)
            size = model_size_mb(model)
        except Exception as e:  # noqa: BLE001 - one bad variant shouldn't kill the whole run
            logger.warning(f"{name}: benchmarking failed ({e}). Skipping this variant.")
            return
        results.append(
            {
                "variant": name,
                "latency_ms_per_batch": round(latency_ms, 3),
                "throughput_img_per_sec": round(throughput, 1),
                "size_mb": round(size, 3),
                "accuracy": None if acc is None else round(acc, 4),
            }
        )
        logger.info(f"{name}: {latency_ms:.3f} ms/batch, {throughput:.1f} img/s, {size:.3f} MB")

    # 1. Baseline fp32
    logger.info("=== Baseline (fp32) ===")
    record("baseline_fp32", baseline, dummy_input)

    # 2. Dynamic quantization (CPU only)
    logger.info("=== Dynamic quantization (int8, Linear layers) ===")
    dynamic_q = apply_dynamic_quantization(baseline)
    record("dynamic_quantization_int8", dynamic_q, torch.rand(batch_size, *IMAGE_SHAPE))

    # 3. Static quantization (CPU only, needs calibration)
    logger.info("=== Static quantization (int8, weights + activations) ===")
    static_q = apply_static_quantization(baseline, test_loader)
    record("static_quantization_int8", static_q, torch.rand(batch_size, *IMAGE_SHAPE))

    # 4. Pruning - local per-layer L1
    logger.info("=== Pruning (local, per-layer L1) ===")
    pruned_local = apply_pruning(baseline, amount=prune_amount, method="local")
    logger.info(f"Sparsity after local pruning:\n{sparsity_report(pruned_local)}")
    record("pruned_local_l1", pruned_local, dummy_input)

    # 5. Pruning - global L1
    logger.info("=== Pruning (global L1) ===")
    pruned_global = apply_pruning(baseline, amount=prune_amount, method="global")
    logger.info(f"Sparsity after global pruning:\n{sparsity_report(pruned_global)}")
    record("pruned_global_l1", pruned_global, dummy_input)

    # 6. torch.compile
    logger.info("=== torch.compile ===")
    compiled = apply_compile(baseline, dummy_input)
    record("torch_compiled", compiled, dummy_input)

    # --- report ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_DIR / "inference_optimization_results.md"

    header = "| variant | latency (ms/batch) | throughput (img/s) | size (MB) | accuracy |"
    sep = "|---|---|---|---|---|"
    rows = [
        f"| {r['variant']} | {r['latency_ms_per_batch']} | {r['throughput_img_per_sec']} "
        f"| {r['size_mb']} | {r['accuracy'] if r['accuracy'] is not None else 'n/a'} |"
        for r in results
    ]
    table = "\n".join([header, sep, *rows])

    baseline_ms = results[0]["latency_ms_per_batch"]
    speedups = "\n".join(
        f"- {r['variant']}: {baseline_ms / r['latency_ms_per_batch']:.2f}x baseline" for r in results[1:]
    )

    md_path.write_text(
        f"# M31 - Inference optimization results\n\n"
        f"Batch size: {batch_size}, device: {dev}, reps: {n_reps} (+{warmup} warmup)\n\n"
        f"{table}\n\n"
        f"## Speedup vs. baseline\n\n{speedups}\n\n"
        f"## Notes\n\n"
        f"- Dynamic/static quantization run on CPU regardless of `--device` "
        f"(PyTorch's eager-mode int8 kernels are CPU-only).\n"
        f"- Pruning here is unstructured: weights are zeroed but the tensor stays dense, "
        f"so PyTorch's dense matmul kernels do not skip the zeros - expect little to no "
        f"latency improvement from pruning alone unless converted to a sparse format.\n"
        f"- `accuracy: n/a` means no PathMNIST test data was available in this environment; "
        f"rerun with data present to get real accuracy numbers.\n",
        encoding="utf-8",
    )

    print("\n" + table)
    print(f"\nResults written to {md_path}")


if __name__ == "__main__":
    typer.run(optimize_inference)
