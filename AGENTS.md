# AGENTS.md

`AGENTS.md` for `Gabz4200/LCQAT_nanochat`. Applies to this repo only. Project instructions override the global defaults at `~/.agents/AGENTS.md` on conflict, and take precedence over inferred conventions.

## Project Overview

Fork of [karpathy/nanochat](https://github.com/karpathy/nanochat), retargeted as a research vehicle for post-training quantization and cheap inference on commodity hardware. Research stages: LC-QAT codebooks, SparseProp sparse backprop, DiffusionBlocks block-wise training, DARD algebraic rule distillation, mul-less / mul-light CPU GEMV kernels, quantized KV-cache attention, export to a stripped inference artifact.

LC-QAT, SparseProp, and DARD stay on by default in `base_train`, `chat_sft`, and `chat_rl`. Disable with `--no-lcqat` / `--no-sparseprop` / `--no-dard`. `DiffusionBlockEngine` is the default training engine (`--db-blocks`, default 4). `--fp8` conflicts with default-on LC-QAT, needs CUDA, and is dropped from the standard runs.

The repo is CPU-first: `pyproject.toml` pins `torch` from the PyTorch CPU index and drops the CUDA extra for the LC-QAT runtime. Re-add a GPU extra per the inline comment there for CUDA/GPU training. torch>=2.14 is pinned; torch 2.9-2.13 CPU inductor can emit use-before-assign buffers in the LC-QAT compiled backward.

Layout follows a functional-core / imperative-shell split:

| Package | Role |
|---------|------|
| `nanochat/models/` | Functional core: pure tensor math, no I/O or optimizer |
| `nanochat/models/quant/` | LC-QAT: codebooks, `LCQATLinear`, export, ablation |
| `nanochat/models/dard/` | DARD: offline AML mining, rule weighting and CSR compilation, differentiable rule loss, CLI flags |
| `nanochat/ops/` | Kernel layer: dispatcher, references, native C++, Taichi |
| `nanochat/modules/` | Imperative shell: checkpoints, inference engine, eval |
| `nanochat/training/` | DiffusionBlocks engine and optimizers |
| `nanochat/data/` | Ingestion boundary: dataloader, dataset, tokenizer |
| `nanochat/callbacks/` | Side-effect observers: W&B logging, GC, run summary |
| `nanochat/utils/` | `COMPUTE_DTYPE` and misc helpers |

Key files:

- Transformer backbone: `nanochat/models/backbone.py` (`GPT`, `GPTConfig`, `Linear`)
- Typed contracts: `nanochat/models/io.py` (`LayerQuantSpec`, `CodebookSpec`)
- Training entry points: `scripts/{base_train,chat_sft,chat_rl,base_eval,chat_eval,chat_cli}.py`
- `base_train` internals: `scripts/_train/{build,loop,eval}.py`
- Quantization core: `nanochat/models/quant/{retrofit,linear,codebook,kd,efqat,lut,packing,export,sparseprop}.py`
- DARD: `nanochat/models/dard/{shard,atomize,weighting,compile,loss,cli,engine_hook}.py` plus `nanochat/ops/references/dard_reference.py` (parity oracle)
- EDM loss kernel: `nanochat/ops/{db_denoise,dispatch}.py`, `nanochat/ops/references/db_denoise_reference.py`, `nanochat/ops/native/cpu/db_denoise.cpp`
- Ablation experiments: `nanochat/modules/experiments/` (one module each)
- Native kernels: `nanochat/ops/native/cpu/{gemv,index_linear,quant_attn,sparseprop,db_denoise}.cpp`
- Op dispatch: `nanochat/ops/{dispatch,gemv,index_linear,quant_attn,sparseprop,db_denoise}.py` plus `references/` oracles
- Kernel loaders: `nanochat/ops/kernels/{cpu_loader,gpu_loader}.py`
- DiffusionBlocks: `nanochat/training/diffusion_blocks.py` (`DiffusionBlockEngine`, `EquiProbabilityPartitioner`)

## Setup Commands

```bash
uv sync --extra cpu --group dev   # CPU wheels + dev tools (pytest, ruff, pyrefly, taichi, aislop)
source .venv/bin/activate
```

A bare `uv sync` does not install torch; `import nanochat` fails until you run the command above. Treat that as the intended signal. After `uv add`, torch flips to CUDA, so follow every `uv add` with `uv sync --extra cpu --group dev`.

Environment notes:

- Keep a C++ compiler (`g++`/`clang++`) on `PATH`; LC-QAT ops parity tests JIT a C++ kernel.
- On macOS MPS, set `NANOCHAT_DTYPE=bfloat16` for the runtime tests.
- `~/.cache/nanochat` holds the tokenizer and ClimbMix shards. Tests that need data expect 2 shards flat in `~/.cache/base_data_climbmix` (last shard is val).
- Small host (~7.6 GB RAM plus swap): use `--depth 6` for smoke runs. A depth-20 run OOMs when anything else runs. Full suite takes ~10 min; prefer targeted files during iteration.
- torchrun 2-rank CPU d6 smoke works; keep `--total-batch-size` a multiple of world tokens.

## Development Workflow

Scripts under `runs/` set up venv and data themselves:

- `bash runs/runcpu.sh`: tiny toy run (d6, ~30 min on M3 Max) exercising all stages.
- `bash runs/speedrun.sh`: 8xH100 GPT-2 speedrun. `--no-lcqat --no-sparseprop` reproduces the historical float speedrun.
- `bash runs/miniseries.sh` / `bash runs/scaling_laws.sh`: depth sweeps; `--db-blocks` scales with depth (4/6/8).
- `bash runs/stackcompare.sh`: uses `--db-blocks=0` for the conventional LM baseline arm.

The manual training loop is the research artifact (`scripts/_train/loop.py`) and stays manual. Parse flags in `scripts/` and thread them as values (`--db-blocks` becomes `num_blocks: int`); models take typed kwargs, never argparse namespaces or config dicts.

When you add a training flag, decide whether it stays on by default (LC-QAT-style) or opt-in (`--fp8`). Keep `--no-*` disables for always-on stages and document them in README and `--help`. New denoiser-only flags must be rejected, not ignored, under `--db-blocks=0`; see the flag-validation tuple in `scripts/_train/build.py`. DARD stays out of that rejection tuple: it is a vocabulary-space loss and both objectives define one. Tune through hardware knobs (`--depth`, `--db-blocks`, `--sparseprop-sparsity`, `--codebook-lr`); do not hide them behind config files.

Structural changes stay structural: no changes to math, losses, or numerical dynamics. Baseline first (`git status --porcelain` clean; run affected tests or a `--depth 6` smoke on the clean tree), verify after each step, and revert-diagnose-fix-retry on failure. The final `git diff` shows moves and boundary fixes only.

## Testing

Run tests:

```bash
pytest tests/
pytest tests/ -m "not slow"
pytest tests/test_<name>.py -q
```

`pythonpath = ["."]` in pytest config is required for bare `pytest`. Taichi GPU tests skip automatically without a Vulkan device (`requires_vulkan` marks).

Write the failing test first (RED), then the minimal implementation (GREEN), then clean up (REFACTOR). A test that passes on first run proves nothing. Bug fixes start with a reproduction test. Name tests as specifications: `test_when_<condition>_then_<outcome>`.

For a new kernel or objective, pin `naive == cpu == gpu` parity against an independent oracle (plain Python loops, not tensor ops that could share the bug), plus `torch.library` opcheck, `torch.compile` composition, the unknown-backend error, and input-validation errors. For the DB engine, assert behavior, not implementation: gradient reach (`lm_head` untouched by EDM, trained by AR), block and noise ownership, mask semantics row-by-row.

Test models need active blocks: nanochat zero-initializes `attn.c_proj`/`mlp.c_proj`, so untouched tiny models have no-op blocks and parity assertions pass vacuously. Build test models with `tests.conftest.build_active_tiny_gpt` (or `make_engine(..., active=True)`). Keep new DB tests in `tests/test_db_denoise_kernel.py` (kernel stack) and `tests/test_db_edm_behavior.py` (mapping, masking, EDM-vs-AR); do not open a new file per slice (`test_dbcpu_objective.py` already merged into `test_dbcpu_denoise.py`).

Quality gates, in order; all must exit 0:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
uv run pytest
uv run aislop scan .
```

Known test files: `tests/test_lcqat_*` (parity, opcheck, runtime), `tests/test_sparseprop*`, `tests/test_dard_{mining,loss,diffusion_blocks,smoke,overhead}.py`, `tests/test_dbcpu_*`, `tests/test_db_{denoise_kernel,edm_behavior}.py`, `tests/test_numerical_fingerprint.py` (behavior gate), `tests/test_architecture_boundary.py` (core-to-shell import rule), `tests/test_adamw_cpu.py`, `tests/test_calculator.py`.

## Code Style

Format before review:

```bash
uv run ruff format . && uv run ruff check .
```

Ruff policy lives in `pyproject.toml` (`select = ["E4","E7","E9","F","I"]`); this ruff build defaults to near-ALL rules, so do not expand the select. `pyrefly check` exits non-zero on unresolved types, and ruff errors fail the aislop gate. A `.pre-commit-config.yaml` exists (canonical hook repos, `fail_fast: true`); the pre-commit binary is not installed, so run the gates directly. The post-edit hook reformats files and runs aislop, so expect churn on every edit. Place `aislop-ignore-next-line <rule> -- reason` on the line directly above the finding.

Boundaries:

- `nanochat/models/` does not import from `modules/`, `training/`, `data/`, `callbacks/`, `utils/`, or `tasks/`. Report a model-layer fact with `warnings.warn` or a return value, never `print0`. A model decision the layers consume (for example `COMPUTE_DTYPE` in `nanochat/models/dtype.py`) lives in the core; the shell re-exports it. `nanochat/models/` to `nanochat/ops/` is the one sanctioned exception (`ops` is the kernel boundary). `nanochat/modules/experiments/` is harness, not model math; measurements there may import `training/`.
- Compose blocks with `nn.ModuleList`/`nn.ModuleDict`; avoid deep inheritance chains. A forward pass reads like a formula top-to-bottom.
- Cross-boundary forward I/O uses frozen `@dataclass` specs (`nanochat/models/io.py`: `LayerQuantSpec`, `CodebookSpec`), never anonymous tuples or magic dicts. A defined-but-unused contract is a violation; use it or delete it.
- Keep model state explicit: no hidden mutable `self.state`; pass recurrent or cached state (for example KV-cache) in and return it explicitly.
- No `.to()`/`.cuda()` in model forwards; isolate side effects in `callbacks/`; keep models constructible standalone from typed kwargs.

Kernel contract: every custom op has three faces (pure-PyTorch oracle in `nanochat/ops/references/`, C++ CPU kernel in `nanochat/ops/native/cpu/`, Taichi GPU kernel in `nanochat/ops/kernels/gpu_loader.py`), routed through `ops/dispatch.py` with an explicit `backend` argument. Models never import kernels directly. New differentiable behavior needs `torch.library` opcheck plus FakeTensor coverage; training gradients flow through an autograd Function with an analytic backward, never by differentiating the compiled op.

Quantization and sparsity rules:

- Quantization rewrites of `Linear` preserve the forward value (straight-through estimator) and pass three-way parity `naive == cpu == gpu` (see `nanochat/ops/` references).
- `dispatch_gemv`, `dispatch_index_linear`, `dispatch_quant_attn`, `dispatch_sparseprop_*`, and `dispatch_db_denoise` raise when the requested backend cannot run; they never degrade silently to `naive`.
- SparseProp masks are materialized once and stay static; the AVX2 backward kernel walks `nnz` entries. Do not make the mask dynamic per step.
- `denoise_step` computes `weight * mean((pred - clean)^2)` through `dispatch_db_denoise` (`--db-denoise-backend`, default `cpu`); gradients flow through `DbDenoiseLossFunction`'s analytic backward, never by differentiating the compiled op.
- Clean-past is structural: the denoiser input is `[clean | noisy]` with suffix-only loss and an always-explicit 2T mask. Plain 2T-causal leaks clean future into noisy tokens (verified row-by-row). Raw EDM and AR loss values live on different scales; never compare them.
- Block 0 (earliest layers) owns the highest noise range (paper Fig. 6 / App. C); `range_for_block` is the single owner, never re-derive the map from `boundaries()[b]`. `meta["db"]` stamps `noise_map_version`; the loader rejects anything but the current version, and hand-built `meta["db"]` dicts in tests carry the stamp too.
- `nanochat/models/dard/atomize.py` is integer set algebra for offline mining; never call it from a training step. The loop sees only the compiled `DardRuleSet` (two binary CSR matrices plus fp32 weights). If online code needs a rule's token sets, the compilation misses a field.
- A DARD conclusion never contains its premise; `_successors` enforces disjoint token sets. A conclusion containing its premise forces `v(conclusion) >= v(premise)`, pins every penalty at its floor, and silences the feature while the loss looks healthy.
- DARD weights stay fp32: `eps_floor = 1e-6` sits below fp8 E5M2's smallest subnormal (`2^-16`), so bf16/fp8 autocast rounds the floor to zero and kills tail rules. `DardRuleLoss` runs its block under `autocast(enabled=False)`, and `DardRuleSet.__post_init__` rejects non-fp32 weights.
- DARD gives `lm_head` a gradient under EDM (it otherwise has none because its target is embeddings). The penalty attaches at the same `lm_head` projection `generate()` samples from. The engine publishes `last_dard_logits` and does not add the penalty itself; the loop owns balancing, so apply the lambda in exactly one place.

## Build/Deploy

Native C++ ops under `nanochat/ops/native/cpu/` build into `~/.cache/torch_extensions`, cached per user. Treat any kernel change as a cache-bust for downstream users: keep diffs minimal and comment the C++/Python contract. Multiple cpp_extensions sharing a torch namespace must use `TORCH_LIBRARY_FRAGMENT`; a second `TORCH_LIBRARY(ns)` SIGABRTs in dlopen static init.

Export is lossy: `scripts.export_lcqat` freezes codebooks into static FP32 LUTs and emits `uint8` index buffers. Mark any non-resumable artifact clearly; exported weights do not resume training.

A mapping or objective change breaks checkpoints: bump the version stamp, reject old checkpoints loudly, and update every hand-built test meta plus `docs/diffusionblocks.md` in the same commit.

## PR Guidelines

- Use conventional commits (`feat(scope):`, `fix(scope):`, footers such as `BREAKING CHANGE: ...`). Keep one logical change per commit, present tense, imperative mood, subject under 72 chars.
- Never `--force` push, never `--no-verify`, never touch git config. When a hook-gated commit fails, fix and create a new commit.
- Include behavior docs in the change: `docs/diffusionblocks.md` and README `--help` text ship with the code, not after it.
- EDM does not train `lm_head`. Under `--db-objective edm` the loss does score-matching on denoised embeddings, so `lm_head` gets no gradient (`.grad is None`). `tests/test_db_edm_behavior.py` pins this. `lm_head` is ~23% of a small model's parameters and the only map from hidden state to token ids, so an EDM-trained checkpoint cannot generate text; it projects through random-weight logits. Any run with generation or coherence as an acceptance criterion must use `--db-objective ce`. DiffusionBlocks applies in both modes; only the block objective differs.

## Debugging/Troubleshooting

- Expect post-edit churn: the hook reformats and runs aislop on every edit.
- Taichi/Vulkan silently computes garbage on directly passed host tensors. Stage through `from_numpy`/`to_numpy` (`_to_ti` in `gpu_loader.py`).
- An EDM checkpoint that cannot generate text is usually not a bug: EDM leaves `lm_head` untrained (see PR Guidelines). Switch acceptance runs to `--db-objective ce`.
- A healthy-looking DARD loss with no effect often means premise/conclusion overlap; check `_successors` before tuning the lambda.
- A `uint8` export that will not resume training is expected; export artifacts are inference-only by design.
- Version-stamp rejections on load mean the checkpoint predates the current noise map or objective; do not bypass the loader check, rebuild or migrate the checkpoint.
- OOM on a depth-20 run with other processes alive is expected on this host; drop to `--depth 6` for smokes and keep `--total-batch-size` a multiple of world tokens on torchrun.
- Missing-tokenizer or missing-shard failures usually mean the cache layout is wrong; confirm 2 shards sit flat in `~/.cache/base_data_climbmix`.
