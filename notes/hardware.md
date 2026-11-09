# Hardware ceilings

Fill this in before any benchmark. All "% of peak" numbers elsewhere use the measured row as the denominator.

## Device

| Item | Value |
| --- | --- |
| GPU | TBD |
| SMs | TBD |
| Shared memory / SM | TBD |
| L2 size | TBD |
| HBM capacity | TBD |
| `ncu` counter access | TBD (ok / `ERR_NVGPUCTRPERM`) |

## Roofline

| | FP16/BF16 Tensor Core (TFLOPS) | HBM bandwidth (TB/s) | Ridge point (FLOPs/byte) |
| --- | --- | --- | --- |
| Theoretical (spec sheet) | TBD | TBD | TBD |
| Measured | TBD (large GEMM) | TBD (copy kernel) | TBD |

Ridge point = peak FLOPs ÷ peak bandwidth. Kernels with arithmetic intensity below it are bandwidth-bound on this device.

## How the measured row was obtained

- Compute ceiling: TBD (matrix size, dtype, library)
- Bandwidth ceiling: TBD (buffer size, read+write accounting)
