# flash-attention-triton

A from-scratch FlashAttention forward kernel in [Triton](https://github.com/triton-lang/triton), with benchmarks and roofline analysis.

The goal is not just a fast kernel but numbers I measured myself and can explain: where the time goes, which resource bounds each configuration, and why the gap to the reference implementation is what it is.

**Target:** ≥ 70% of PyTorch SDPA's flash backend throughput. If the kernel falls short, the deliverable is a profiling-backed explanation of the gap.

## Status

> 🚧 Work in progress. Nothing below is measured yet.

**Phase 0: environment and ceilings**
- [ ] Confirm GPU model and Nsight Compute counter access (`ncu` without `ERR_NVGPUCTRPERM`)
- [ ] Record theoretical peaks (Tensor Core FP16/BF16, HBM bandwidth) and the roofline ridge point
- [ ] Measure achievable ceilings: large GEMM (compute) and copy kernel (bandwidth)
- [ ] Triton tutorials with an `ncu` pass on each: vector add, fused softmax, matmul

**Phase 1: FlashAttention forward**
- [ ] Naive 3-step PyTorch attention: latency, peak memory, hand-counted HBM traffic vs. N
- [ ] Triton forward kernel with online softmax, `allclose` against the naive reference
- [ ] Benchmark: naive / mine / Triton tutorial fused attention / SDPA flash backend
- [ ] Tile-parameter sweep (`BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`) with an explanation of the winner
- [ ] Roofline plot and write-up

Stretch goals (only after the above): backward pass, FlashAttention-2 sequence-dimension parallelism, FP8.

## Environment

All numbers in this repo are tied to the versions below. Results from a different stack are not comparable.

| Component | Version |
| --- | --- |
| GPU | TBD |
| Driver | TBD |
| CUDA | TBD |
| PyTorch | TBD |
| Triton | TBD |
| Nsight Compute | TBD |

Hardware ceilings (theoretical and measured) live in [`notes/hardware.md`](notes/hardware.md). Every "% of peak" figure in this repo uses the **measured** ceiling as the denominator.

## Layout

```
kernels/   Triton kernels and the naive PyTorch reference
bench/     Benchmark scripts (timing via triton.testing.do_bench)
profiles/  Nsight Compute / Nsight Systems summaries
notes/     Hardware ceilings, tutorial notes, write-ups
```

## Experiment matrix

Batch and head count are fixed at a size that fills the GPU (e.g. B=4, H=16).

| Variable | Values | What it shows |
| --- | --- | --- |
| Sequence length N | 512, 1k, 2k, 4k, 8k, 16k | When naive OOMs; when FA turns compute-bound |
| Head dim d | 64, 128 | Tile size vs. SRAM pressure |
| Mask | causal, non-causal | Benefit of skipping whole tiles |
| Precision | fp16, bf16 | Numerical error vs. speed |
| Tile params | `BLOCK_M`/`BLOCK_N` ∈ {64, 128} | Tuning sweep |

For each configuration: latency, achieved TFLOPS, % of measured peak, DRAM bytes read/written (`dram__bytes_read.sum`, `dram__bytes_write.sum`), arithmetic intensity, occupancy, register spills.

Forward FLOPs (two matmuls, `QKᵀ` and `PV`); causal is roughly half:

$$
\text{FLOPs} = 4 \cdot B \cdot H \cdot N^2 \cdot d
$$

## Hypothesis to verify

Naive attention materializes the N×N score and probability matrices in HBM, so its traffic grows as Θ(N²). FlashAttention never writes the N×N matrices; the paper bounds its HBM accesses at Θ(N²d²/M) for SRAM size M, which is far below N² for typical d. In practice, K/V tiles re-read across query blocks may be served from L2, so measured DRAM bytes could look closer to Θ(N·d). This repo checks the measured byte counts against both models.

## References

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023
- Milakov & Gimelshein, *Online normalizer calculation for softmax*, 2018
- Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, CACM 2009
- Hwu, Kirk & El Hajj, *Programming Massively Parallel Processors*, 4th ed.
- Triton tutorials: vector add, fused softmax, matrix multiplication, fused attention
- Nsight Compute documentation: Roofline and Memory Workload Analysis sections

## License

[MIT](LICENSE)
