# CLAUDE.md

Guidance for Claude Code in this repo. Reply to Steve in Traditional Chinese; keep code, comments, and committed docs in English.

If `HANDOFF.md` exists (git-ignored, local only), read it before doing anything else.

## What this repo is

A from-scratch FlashAttention forward kernel in Triton, benchmarked against naive PyTorch, the Triton fused-attention tutorial, and PyTorch SDPA's flash backend, with roofline analysis from Nsight Compute. Target: ≥ 70% of SDPA flash throughput; if it falls short, explain the gap with profiling data. Roadmap and experiment matrix are in `README.md`.

This is a portfolio piece. Its value is that Steve wrote the kernel himself and can defend every number under interview questioning. Optimize for his understanding, not for how fast code gets written.

## Collaboration rules

**Steve writes, Claude coaches:**
- The Triton FlashAttention kernel(s) in `kernels/`. Do not write or rewrite kernel bodies unless he explicitly asks for code. Explain concepts, review what he wrote, point at where a bug is and why, or ask the question that leads him there.
- Until his kernel passes the numerical check, do not show or paraphrase the official Triton fused-attention tutorial, FlashAttention-2 source, or other reference kernels. Concepts from the papers are fine.
- The analysis: hand-counted HBM traffic, roofline ridge points, why a tile config wins. Check his math and push back on weak reasoning instead of supplying the answer.

**Claude may write directly:** benchmark harnesses, the naive PyTorch reference, environment reporting, CSV output and plotting, profiling wrapper scripts, README/notes formatting.

**Scope guard:** backward pass, FlashAttention-2 sequence-dimension parallelism, and FP8 are stretch goals. If they come up before the Phase 1 minimum is done (correct kernel, benchmark, tile sweep, roofline write-up), say so.

## Measurement rules

- Timing: `triton.testing.do_bench` only. No wall-clock timing around asynchronous CUDA calls.
- Forward FLOPs = 4·B·H·N²·d; causal counts half. State the formula next to any TFLOPS figure.
- "% of peak" uses the **measured** ceilings in `notes/hardware.md`, never spec-sheet peaks.
- DRAM traffic comes from Nsight Compute `dram__bytes_read.sum` and `dram__bytes_write.sum`.
- Correctness before performance: no benchmark number is reported for a kernel that hasn't passed `allclose` against the naive reference. Record max absolute error per dtype.
- Never invent or estimate numbers in README or notes. A value is either measured in this repo or cited from an official spec sheet with its source; otherwise it stays TBD.
- Every result file records GPU, driver, CUDA, PyTorch, and Triton versions.

## Environment

- The dev machine may be macOS with no NVIDIA GPU. Triton has no macOS wheels, so kernels and benchmarks cannot run there. Check `nvidia-smi` before trying to run anything; without it, limit work to editing and static review.
- GPU host and pinned versions: TBD (see the README Environment table). Once known, pin them in README and a requirements file.
- If `ncu` fails with `ERR_NVGPUCTRPERM`, performance counters are admin-only. Fall back to Nsight Systems and the PyTorch profiler, and compute arithmetic intensity by hand.

## Repo conventions

- `kernels/` Triton kernels + naive reference · `bench/` benchmark scripts · `profiles/` Nsight CSV exports (raw `.ncu-rep`/`.nsys-rep` are git-ignored) · `notes/` hardware ceilings and write-ups.
- Public repo: public models and synthetic inputs only, no proprietary code or data. Personal notes and plans stay out of git.
- GitHub account: `lzrong0203`. Commit identity must be set in every new clone:
  ```
  git config user.name "Steve Lai"
  git config user.email "10355534+lzrong0203@users.noreply.github.com"
  ```
  Never commit with a work email.
- Commit or push only when Steve asks.
