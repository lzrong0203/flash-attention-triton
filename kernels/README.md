# kernels

- Naive PyTorch reference (QKᵀ → softmax → ·V), used as the correctness oracle
- Triton FlashAttention forward (online softmax, tiled over K/V)
