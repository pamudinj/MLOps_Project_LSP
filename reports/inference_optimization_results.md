# M31 - Inference optimization results

Batch size: 64, device: cpu, reps: 50 (+10 warmup)

| variant | latency (ms/batch) | throughput (img/s) | size (MB) | accuracy |
|---|---|---|---|---|
| baseline_fp32 | 8.01 | 7989.6 | 1.613 | 0.8394 |
| dynamic_quantization_int8 | 8.357 | 7658.7 | 0.463 | 0.8401 |
| static_quantization_int8 | 4.842 | 13216.6 | 0.41 | 0.7726 |
| pruned_local_l1 | 8.013 | 7986.6 | 1.613 | 0.7825 |
| pruned_global_l1 | 8.434 | 7588.3 | 1.613 | 0.8405 |

## Speedup vs. baseline

- dynamic_quantization_int8: 0.96x baseline
- static_quantization_int8: 1.65x baseline
- pruned_local_l1: 1.00x baseline
- pruned_global_l1: 0.95x baseline

## Notes

- Dynamic/static quantization run on CPU regardless of `--device` (PyTorch's eager-mode int8 kernels are CPU-only).
- Pruning here is unstructured: weights are zeroed but the tensor stays dense, so PyTorch's dense matmul kernels do not skip the zeros - expect little to no latency improvement from pruning alone unless converted to a sparse format.
- `accuracy: n/a` means no PathMNIST test data was available in this environment; rerun with data present to get real accuracy numbers.
