# Documentation

These pages describe how the fork works. The root [README](../README.md) covers what the repo is and how to run it; the pages below cover internals.

Start here if you change training, quantization, or kernels. Each page states its scope up front and points to the fastest reproduction command.

| Page | Covers | Read it when you |
|------|--------|------------------|
| [quantization.md](quantization.md) | LC-QAT codebooks, flags, KD anchoring, EfQAT freezing, denoiser distillation, sigma-conditioned codebooks, per-channel tables, activation LUTs | Touch `models/quant/` or any `--lcqat`, `--kd`, `--efqat` flag |
| [kernels.md](kernels.md) | Mul-less GEMV, index-linear and sparse-CSR kernels, quantized export, benchmarks | Touch `ops/native/cpu/`, `ops/dispatch.py`, or `scripts/export_lcqat.py` |
| [sparseprop.md](sparseprop.md) | SparseProp mechanism, flags, defect fixes, measured cost | Touch sparsity masks, `sparseprop.cpp`, or `--sparseprop-*` flags |
| [diffusionblocks.md](diffusionblocks.md) | Block-wise training engine, `--db-blocks=0` baseline, CPU behaviour | Touch `training/diffusion_blocks.py` or any `--db-*` flag |
| [dard.md](dard.md) | DARD offline rule mining, rule weighting, order-violation loss, DiffusionBlocks attachment | Touch `models/dard/` or any `--dard-*` flag |
| [architecture.md](architecture.md) | Package layout, functional-core / imperative-shell boundary | Add a module or move code across packages |

Experiment logs and generated tables live in [`dev/`](../dev), not here:

- [`dev/LEADERBOARD.md`](../dev/LEADERBOARD.md) — time-to-GPT-2 notes and the ablation table from `scripts/lcqat_ablation.py`
- [`dev/STACK_COMPARE.md`](../dev/STACK_COMPARE.md) — four-arm LC-QAT / SparseProp / DiffusionBlocks sweep
- [`dev/LOG.md`](../dev/LOG.md) — training experiment log
- [`dev/lcqat_spec.md`](../dev/lcqat_spec.md) — LC-QAT spec, system symbiosis, and measured deviations

Contributor guardrails live in [`AGENTS.md`](../AGENTS.md).
