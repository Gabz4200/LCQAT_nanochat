# Kernels and the quantized runtime

The decode path. Native CPU kernels live in `nanochat/ops/native/cpu/`, are JIT-built
by torch on first use (needs `g++`/`clang++`, cached under `~/.cache/torch_extensions`),
and are dispatched through `nanochat/ops/dispatch.py`.

**No silent fallback.** `dispatch_gemv`, `dispatch_index_linear`,
`dispatch_quant_attn` and `sparse_index_linear` raise when the requested backend
cannot run. A quiet downgrade to a dense path would report success for work the
caller believed was sparse.

## Mul-less GEMV + index-linear kernels

For quantized decode (`K_W = 3` ternary), the runtime routes through a backend dispatcher:

- `naive` — pure PyTorch oracle (`nanochat/ops/references/`), for parity tests and debugging;
- `cpu` — C++ kernel under `nanochat/ops/native/cpu/`, JIT-built by torch on first use (needs `g++`/`clang++`; build cached under `~/.cache/torch_extensions`);
- `gpu` — Taichi kernel (`nanochat/ops/kernels/gpu_loader.py`), loaded lazily (needs `taichi` + a Vulkan ICD).

Each kernel decides separately how much ISA to spend, and the reasoning is in the file header rather than left to be inferred from the code:

| Kernel | Path | Why that path |
|---|---|---|
| `gemv.cpp` | AVX-512, AVX2, scalar, **selected at runtime via cpuid** | The ternary GEMV is FMA-bound, so all three are written and the extension still builds and runs on a machine with no AVX-512 |
| `sparseprop.cpp` | AVX2 FMA, parallel over rows (dW, forward) / columns (dX) | The gradient vectorizes over the batch, so a fixed-width vector unit is exactly the right shape |
| `index_linear.cpp` | `-O3` scalar, one thread per output element | **Storage-bound, not compute-bound.** Weights are read in place in their packed format — no materialized fp32 matrix — so the win is memory, and SIMD is deferred until `scripts/gemv_bench.py` says otherwise |
| `quant_attn.cpp` | `-O3` scalar, LUT gathers | The decode working set is L1/L2 resident and the bottleneck is LUT gathers, not FMA throughput |

`dispatch_index_linear` generalizes the packed format to **any** `K >= 3`; `index_format_for_k` picks the layout (`packing.py`): trits at K=3 (5 per byte), nibbles for K<=15 (2 per byte), raw `uint8` for K<=255, `int32` above. Tests in `tests/test_lcqat_index_linear.py`.

`dispatch_sparse_index_linear` / `sparse_index_linear` is the CSR counterpart (`nanochat/ops/sparse_index_linear.py`): the weight is a `(row_ptr, col_indices, alphabet)` listing of surviving slots rather than a dense index matrix, so a heavily pruned layer touches only the weights it kept, and the `alphabet` holds **resolved FP32 values** rather than codebook IDs — the export planner already decided which levels survive and knows each value, so the kernel never needs the codebook. A stored slot whose value is exactly `0.0` is skipped rather than accumulated, which is on the hot path in practice because LC-QAT's zero anchor makes `0.0` a real weight and magnitude pruning stores a pruned slot as a real entry.

That path is `cpu`-only, and it **raises** for any other backend. There is deliberately no dense fallback: silently degrading a sparse layer to a dense one while still reporting success is the exact bug the sparsity flags exist to make visible.

A requested backend that cannot run raises — it never silently falls back to naive. Three-way parity (`naive == cpu == gpu`), `torch.library.opcheck`, and `torch.compile` composition are covered by `tests/test_lcqat_ops.py`.

## Export

```bash
uv run python -m scripts.export_lcqat --source sft --out exports/lcqat_sft.pt
```

Freezes codebooks into static FP32 LUTs, replaces FP32 weight matrices with `uint8` index buffers, and saves a minimal artifact for the quantized inference runtime (not resumable for training). After export, `load_model` auto-detects the LC-QAT state from the artifact.

Export works on the default configuration: LC-QAT, SparseProp and DiffusionBlocks
all at their defaults, and the artifact reloads. Three things had to line up for
that, because each covers a different subtree of what `load_model` returns.

The engine is not an `nn.Module` — it owns three — so the export walk needs a
name-addressable tree. `DiffusionBlockEngine` now exposes `named_modules()` and
`get_submodule()` alongside the `state_dict` / `named_parameters` views it
already had, with the same `db_adapters` / `db_denoise_heads` prefixes, so the
adapters and denoise heads are exported with the base transformer rather than
skipped.

Loading back is a separate problem from exporting. `build_model` retrofits the
bare GPT and then constructs the engine from plain float `nn.Linear`s, and the
engine's load is `strict=False`, so engine-owned codebook and sparsity keys were
discarded without a word — the model came back structurally valid and
numerically wrong. Both retrofits are now gated on the `db_*` state keys that
say the subtrees were quantized, rather than on the base model's meta.

One asymmetry remains, and it is deliberate: the exported artifact carries the
sparse CSR buffers, but the reloaded layer is a plain `LCQATLinear` and no
forward path routes to `sparse_index_linear_cpu` yet — the dense
`packed_weight_indices` is what actually runs. Exporting the CSR form ahead of a
runtime that consumes it is harmless because both buffers are written and are
numerically identical.

## Benchmarks

Three benchmark scripts live in `scripts/`:

- `scripts/gemv_bench.py` — LC-QAT index-fetch matmul paths vs `torch.nn.functional.linear`, at the PRD's suggested shapes (m ∈ {768, 4096}, n ∈ {768, 2048}). Sanity-checks every backend against the naive oracle before timing, so a number can never come from a kernel that computes the wrong function. `uv run python -m scripts.gemv_bench`.
- `scripts/kv_budget_bench.py` — reproduces the LC-QAT PRD section 7 memory-budget table (heterogeneous packed weights, FP32 codebook LUTs, 32K packed 4-bit KV cache, C++ workspace constant). With `--alloc-kv`, it allocates and page-touches a real `QuantizedKVCache` at the PRD's implied dims to check resident growth against `storage_bytes`. `uv run python -m scripts.kv_budget_bench [--alloc-kv]`.
- `scripts/infer_bench.py` — end-to-end decode latency, throughput, memory and bandwidth utilization of a real checkpoint, sweeping the decode batch size. Intelligence metrics say nothing about what a model costs to *run*; this is the other axis. Prefill is compute-bound and batched, decode is bandwidth-bound and nearly free to batch until compute saturates, and the sweep traces the curve between the two. Prints a human-readable card and, as the very last line, one compact JSON document so scripts can consume it. `uv run python -m scripts.infer_bench -i base -g d12 | tail -1 | jq .sweep`.

Reach for `gemv_bench.py` when touching anything under `ops/native/cpu/`. It is the evidence for whether an ISA change helped, and the SparseProp and index-linear sections above are both conclusions drawn from it.
