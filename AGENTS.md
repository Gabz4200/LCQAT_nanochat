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

## Setup

```bash
uv sync --extra cpu --group dev   # CPU wheels + dev tools (pytest, ruff, pyrefly, taichi, aislop)
source .venv/bin/activate
```

A bare `uv sync` deliberately does NOT install torch — `import nanochat`
fails until you run the command above. That is the intended signal, not a bug.

Host constraints: small host (~7.6 GB RAM + swap). Use `--depth 6` for smoke
runs; a depth-20 training run OOMs if anything else is running. The full test
suite takes ~10 min; prefer targeted files during iteration.

## Commands

Run scripts under `runs/` (they set up the venv and data themselves):
- `bash runs/runcpu.sh` — tiny toy run (d6, ~30 min on M3 Max) exercising all stages.
- `bash runs/speedrun.sh` — 8×H100 GPT-2 speedrun (`--no-lcqat --no-sparseprop`
  reproduces the historical float speedrun; `--fp8` is dropped, it conflicts
  with the default-on LC-QAT and needs CUDA).
- `bash runs/miniseries.sh` / `bash runs/scaling_laws.sh` — depth sweeps;
  `--db-blocks` scales with depth (4/6/8).
- `bash runs/stackcompare.sh` — uses `--db-blocks=0` for the conventional LM
  baseline arm.

Tests: `pytest tests/` (add `-m "not slow"` to skip slow runs; single file:
`pytest tests/test_<name>.py -q`). `pythonpath = ["."]` in pytest config is
required for bare `pytest`. LC-QAT ops parity tests JIT a C++ kernel, so a
compiler (`g++`/`clang++`) must be on `PATH`; on macOS MPS set
`NANOCHAT_DTYPE=bfloat16` for the runtime tests. Taichi GPU tests skip
automatically when no Vulkan device is present (`requires_vulkan` marks).

Gates — all must exit 0, in this order:
```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
uv run pytest
uv run aislop scan .
```
Ruff policy lives in `pyproject.toml` (`select = ["E4","E7","E9","F","I"]`;
this ruff build defaults to near-ALL rules, so don't expand the select).
`pyrefly check` exits non-zero on unresolved types; ruff errors fail the
aislop gate. A `.pre-commit-config.yaml` exists (canonical hook repos,
`fail_fast: true`); the pre-commit binary is not installed, so run the gates
above directly. After `uv add`, torch flips to CUDA — always follow with
`uv sync --extra cpu --group dev`.

Format: `uv run ruff format . && uv run ruff check .`

## Testing conventions (TDD)

- Write the failing test first (RED), then the minimal implementation
  (GREEN), then clean up (REFACTOR). A test that passes on first run proves
  nothing. Bug fixes start with a reproduction test (Prove-It pattern).
- Test names read like specifications: `test_when_<condition>_then_<outcome>`.
- New kernel or objective: pin `naive == cpu == gpu` parity against an
  independent oracle (plain Python loops, not tensor ops that could share the
  bug), plus `torch.library` opcheck, `torch.compile` composition, the
  unknown-backend error, and input-validation errors.
- Coverage over the DB engine must assert behavior, not implementation:
  gradient reach (`lm_head` untouched by EDM, trained by AR), block/noise
  ownership, mask semantics row-by-row.
- Gotcha: nanochat zero-initializes `attn.c_proj`/`mlp.c_proj`, so untouched
  tiny models have NO-OP blocks and parity assertions pass vacuously — always
  build test models with `tests.conftest.build_active_tiny_gpt` (or
  `make_engine(..., active=True)`).
- Gotcha: `test_dbcpu_objective.py` was merged into `test_dbcpu_denoise.py`
  (20.6% duplication); keep new DB tests in `tests/test_db_denoise_kernel.py`
  (kernel stack) and `tests/test_db_edm_behavior.py` (mapping, masking,
  EDM-vs-AR) instead of new files per slice.

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
- **No silent backend fallback.** `dispatch_gemv`, `dispatch_index_linear`,
  `dispatch_quant_attn`, `dispatch_sparseprop_*`, and `dispatch_db_denoise`
  raise if the requested backend can't run — never silently degrade to `naive`.
- **Export is lossy.** `scripts.export_lcqat` freezes codebooks into static
  FP32 LUTs and emits `uint8` index buffers — not resumable for training. Mark
  any non-resumable artifact clearly.
- **Always-on defaults.** If you add a training flag, decide whether it is on
  by default (LC-QAT-style) or opt-in (`--fp8`). Keep `--no-*` disables for
  always-on stages and document them in the README and `--help`. New
  denoiser-only flags must be rejected (not ignored) under `--db-blocks=0`;
  see the flag-validation tuple in `scripts/_train/build.py`.
- **Sparsity is static.** SparseProp masks are materialised once; the AVX2
  backward kernel walks `nnz` entries. Don't make the mask dynamic per step.
- **Hardware knobs exist for a reason.** `--depth` (model size), `--db-blocks`
  (block count), `--sparseprop-sparsity` (0–1), `--codebook-lr` — these are the
  tuning surfaces. Don't hide them behind config files.
- **DiffusionBlocks map is versioned.** Block 0 (earliest layers) owns the
  HIGHEST noise range (paper Fig. 6 / App. C); `range_for_block` is the single
  owner — never re-derive the map from `boundaries()[b]`. `meta["db"]` stamps
  `noise_map_version`; the loader rejects anything but the current version.
  Hand-built `meta["db"]` dicts in tests must carry the stamp too.
- **EDM loss goes through the kernel.** `denoise_step` computes
  `weight * mean((pred - clean)^2)` via `dispatch_db_denoise`
  (`--db-denoise-backend`, default `cpu`); gradients flow through
  `DbDenoiseLossFunction`'s analytic backward, never by differentiating the
  compiled op (same split as the SparseProp training path).
- **Clean-past is structural.** The denoiser input is `[clean | noisy]` with
  suffix-only loss and an always-explicit 2T mask — plain 2T-causal leaks clean
  future into noisy tokens (verified row-by-row). Raw EDM and AR loss values
  live on different scales and must never be compared.

## Gotchas (learned the hard way)

- The post-edit hook reformats files and runs aislop — expect churn on every edit.
- `aislop-ignore-next-line <rule> -- reason` must sit on the line DIRECTLY
  above the finding.
- Multiple cpp_extensions sharing a torch namespace MUST use
  `TORCH_LIBRARY_FRAGMENT` — a second `TORCH_LIBRARY(ns)` SIGABRTs in dlopen
  static init.
- Taichi/Vulkan silently computes garbage on directly-passed host tensors —
  always stage via `from_numpy`/`to_numpy` (`_to_ti` in `gpu_loader.py`).
- torch 2.9–2.13 CPU inductor can emit use-before-assign buffers in the
  LC-QAT compiled backward; torch>=2.14 is pinned for this reason.
- `~/.cache/nanochat` holds the tokenizer + ClimbMix shards; tests needing
  them expect 2 shards FLAT in `~/.cache/base_data_climbmix` (last = val).
- torchrun 2-rank CPU d6 smoke works; keep `--total-batch-size` a multiple of
  world tokens.

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
- EDM loss kernel: `nanochat/ops/{db_denoise,dispatch}.py`,
  `nanochat/ops/references/db_denoise_reference.py`,
  `nanochat/ops/native/cpu/db_denoise.cpp`
- Ablation experiments: `nanochat/modules/experiments/` (one module each)
- Native kernels: `nanochat/ops/native/cpu/{gemv,index_linear,quant_attn,sparseprop,db_denoise}.cpp`
- Op dispatch: `nanochat/ops/{dispatch,gemv,index_linear,quant_attn,sparseprop,db_denoise}.py` (+ `references/` oracles)
- Kernel loaders: `nanochat/ops/kernels/{cpu_loader,gpu_loader}.py`
- DiffusionBlocks: `nanochat/training/diffusion_blocks.py` (`DiffusionBlockEngine`, `EquiProbabilityPartitioner`)
- Tests: `tests/test_lcqat_*` (parity + opcheck + runtime), `tests/test_sparseprop*`,
  `tests/test_dbcpu_*`, `tests/test_db_{denoise_kernel,edm_behavior}.py`,
  `tests/test_numerical_fingerprint.py` (behavior gate),
  `tests/test_architecture_boundary.py` (core->shell import rule),
  `tests/test_adamw_cpu.py`, `tests/test_calculator.py`

## Refactor rules (behavior preservation)

Refactors here are structural only — no changes to math, losses, or numerical
dynamics. Bracket every structural change with verification:

- Baseline first: `git status --porcelain` must be clean; run the affected
  tests (or a `--depth 6` smoke) once on the clean tree before touching code.
- Verify after each step; on failure revert, diagnose, fix, retry. The final
  `git diff` must show only moves and boundary fixes.
- The manual training loop is the research artifact (`scripts/_train/loop.py`)
  and stays manual — but shell boundaries still hold: no `.to()`/`.cuda()` in
  model forwards, side effects isolated in `callbacks/`, models constructible
  standalone from typed kwargs.

Structural rules (from the functional-core / imperative-shell split):

- **Flat math.** Compose blocks with `nn.ModuleList`/`nn.ModuleDict`; no deep
  inheritance chains. A forward pass must read like a formula top-to-bottom.
- **Typed contracts.** Cross-boundary forward I/O uses frozen `@dataclass`
  specs (`nanochat/models/io.py`: `LayerQuantSpec`, `CodebookSpec`) — never
  anonymous tuples or magic dicts. A defined-but-unused contract is a
  violation: use it or delete it.
- **Explicit state.** No hidden mutable `self.state` in models; recurrent or
  cached state (e.g. KV-cache) is passed in and returned explicitly.
- **Config isolation.** Models take typed kwargs, never argparse namespaces or
  config dicts. Flags are parsed in `scripts/` and threaded as values
  (`--db-blocks` → `num_blocks: int`, never the namespace).
- **Kernel contract.** Every custom op has three faces: a pure-PyTorch oracle
  in `nanochat/ops/references/`, a C++ CPU kernel in
  `nanochat/ops/native/cpu/`, a Taichi GPU kernel in
  `nanochat/ops/kernels/gpu_loader.py` — routed through `ops/dispatch.py`
  with an explicit `backend` argument. Models never import kernels directly.
  New differentiable behavior needs `torch.library` opcheck + FakeTensor
  coverage; training gradients flow via an autograd Function with an analytic
  backward, never by differentiating the compiled op.

## PR and commit guidelines

- Conventional commits: `feat(scope):`, `fix(scope):`, footers for breaking
  changes (`BREAKING CHANGE: ...`). One logical change per commit; present
  tense, imperative mood, <72-char subject.
- Never `--force` push, never `--no-verify`, never touch git config. If a
  hook-gated commit fails, fix and create a NEW commit.
- A mapping or objective change is a checkpoint-breaking change: bump the
  version stamp, reject old checkpoints loudly, and update every hand-built
  test meta plus `docs/diffusionblocks.md` in the same commit.
- Docs that describe behavior (`docs/diffusionblocks.md`, README `--help`
  text) are part of the change, not an afterthought.
