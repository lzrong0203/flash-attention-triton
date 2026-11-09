# profiles

Nsight Compute and Nsight Systems results.

Raw `.ncu-rep` / `.nsys-rep` files are git-ignored (large and device-specific). Commit CSV exports instead, e.g.

```
ncu --csv --page raw -i report.ncu-rep > report.csv
```

Each file name should encode the configuration (kernel, N, d, mask, dtype, tile params).
