# Project agent instructions

`AGENTS.md` for `Gabz4200/LCQAT_nanochat`. Applies to this repo only. Project
instructions override the global defaults at `~/.agents/AGENTS.md` when they
conflict, and take precedence over inferred conventions.

## Scope

This is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat)
retargeted as a research vehicle for **post-training quantization for cheap
inference on commodity hardware**: LC-QAT codebooks, SparseProp sparse
backprop, DiffusionBlocks block-wise training, mul-less / mul-light CPU GEMV
kernels, quantized KV-cache attention, and export to a stripped inference
artifact. The LC-QAT and SparseProp stages are **always-on by default** in
`base_train`, `chat_sft`, and `chat_rl` (disable with `--no-lcqat` /
`--no-sparseprop`). The DiffusionBlocks engine (`DiffusionBlockEngine`) is the
default training engine (`--db-blocks`, default 4).

The repo is **CPU-first**: `pyproject.toml` pins `torch` from the PyTorch CPU
index; the CUDA extra was dropped for the LC-QAT runtime. Re-add a GPU extra
per the inline comment there if you need CUDA/GPU training.

## Commands

Use the repo's `uv` setup. Activate it first:

```bash
uv sync --extra cpu --group dev   # CPU wheels + dev tools (pytest, ruff, pyrefly, taichi, aislop)
source .venv/bin/activate
```

Run scripts under `runs/` set up the venv and data themselves:
- `bash runs/runcpu.sh` — tiny toy run (d6, ~30 min on M3 Max) exercising all stages.
- `bash runs/speedrun.sh` — 8×H100 GPT-2 speedrun (`--no-lcqat --no-sparseprop`
  reproduces the historical float speedrun; `--fp8` is dropped, it conflicts
  with the default-on LC-QAT and needs CUDA).
- `bash runs/miniseries.sh` / `bash runs/scaling_laws.sh` — depth sweeps;
  `--db-blocks` scales with depth (4/6/8).

Run the test suite: `pytest tests/` (add `-m "not slow"` to skip slow runs).
LC-QAT ops parity tests JIT a C++ kernel, so a compiler (`g++`/`clang++`) must
be on `PATH`; on macOS MPS set `NANOCHAT_DTYPE=bfloat16` for the runtime tests.

Format: `uv run ruff format . && uv run ruff check .`
Type-check: `uv run pyrefly check .`
Lint gates: `uv run pyrefly check .` exits non-zero on unresolved types; ruff
errors fail the aislop gate.

## Engineering guardrails for this repo

- **The core does not import the shell.** `nanochat/models/` must not import from
  `modules/`, `training/`, `data/`, `callbacks/`, `utils/` or `tasks/`. Report a
  model-layer fact with `warnings.warn` or by returning a value, not by calling
  `print0`. A model decision that the layers consume (e.g. `COMPUTE_DTYPE` in
  `nanochat/models/dtype.py`) belongs in the core, with the shell free to
  re-export it. `nanochat/models/ -> nanochat/ops/` is the one sanctioned
  exception: `ops` is the kernel boundary. `nanochat/modules/experiments/` is
  harness, not model math — measurements there may import `training/`.
- **Behavior over bytes.** Quantization rewrites of `Linear` must preserve the
  forward value (straight-through estimator) and must pass three-way parity
  `naive == cpu == gpu` (see `nanochat/ops/` references).
- **JIT kernels are cached per user.** Native C++ ops under
  `nanochat/ops/native/cpu/` build into `~/.cache/torch_extensions`. Any
  kernel change is a cache-bust for downstream users; keep diffs minimal and
  comment the C++/Python contract.
- **No silent backend fallback.** `dispatch_gemv`, `dispatch_index_linear`, and
  `dispatch_quant_attn` raise if the requested backend can't run — never
  silently degrade to `naive`.
- **Export is lossy.** `scripts.export_lcqat` freezes codebooks into static
  FP32 LUTs and emits `uint8` index buffers — not resumable for training. Mark
  any non-resumable artifact clearly.
- **Always-on defaults.** If you add a training flag, decide whether it is on
  by default (LC-QAT-style) or opt-in (`--fp8`). Keep `--no-*` disables for
  always-on stages and document them in the README and `--help`.
- **Sparsity is static.** SparseProp masks are materialised once; the AVX2
  backward kernel walks `nnz` entries. Don't make the mask dynamic per step.
- **Hardware knobs exist for a reason.** `--depth` (model size), `--db-blocks`
  (block count), `--sparseprop-sparsity` (0–1), `--codebook-lr` — these are the
  tuning surfaces. Don't hide them behind config files.

## File layout reference

The package follows a functional-core / imperative-shell split:

| Package | Role |
|---------|------|
| `nanochat/models/` | **Functional core** — pure tensor math, no I/O or optimizer |
| `nanochat/models/quant/` | LC-QAT: codebooks, `LCQATLinear`, export, ablation |
| `nanochat/ops/` | Kernel layer — dispatcher, references, native C++, Taichi |
| `nanochat/modules/` | Imperative shell — checkpoints, inference engine, eval |
| `nanochat/training/` | DiffusionBlocks engine + optimizers |
| `nanochat/data/` | Ingestion boundary — dataloader, dataset, tokenizer |
| `nanochat/callbacks/` | Side-effect observers — W&B logging, GC, run summary |
| `nanochat/utils/` | `COMPUTE_DTYPE` and misc helpers |

Specific files:

- Transformer backbone: `nanochat/models/backbone.py` (`GPT`, `GPTConfig`, `Linear`)
- Typed contracts: `nanochat/models/io.py` (`LayerQuantSpec`, `CodebookSpec`)
- Training entry points: `scripts/{base_train,chat_sft,chat_rl,base_eval,chat_eval,chat_cli}.py`
- `base_train` internals: `scripts/_train/{build,loop,eval}.py`
- Quantization core: `nanochat/models/quant/{retrofit,linear,codebook,kd,efqat,lut,packing,export,sparseprop}.py`
- Ablation experiments: `nanochat/modules/experiments/` (one module each)
- Native kernels: `nanochat/ops/native/cpu/{gemv,index_linear,quant_attn,sparseprop}.cpp`
- Op dispatch: `nanochat/ops/{dispatch,gemv,index_linear,quant_attn,sparseprop}.py` (+ `references/` oracles)
- Kernel loaders: `nanochat/ops/kernels/{cpu_loader,gpu_loader}.py`
- DiffusionBlocks: `nanochat/training/diffusion_blocks.py` (`DiffusionBlockEngine`, `EquiProbabilityPartitioner`)
- Tests: `tests/test_lcqat_*` (parity + opcheck + runtime), `tests/test_sparseprop*`,
  `tests/test_dbcpu_*`, `tests/test_numerical_fingerprint.py` (behavior gate),
  `tests/test_architecture_boundary.py` (core->shell import rule),
  `tests/test_adamw_cpu.py`, `tests/test_calculator.py`
