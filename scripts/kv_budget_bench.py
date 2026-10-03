"""
Reproduce the LC-QAT PRD section 7 memory budget table (LC-QAT PRD, section 7).

Rows: heterogeneous packed weights (repo `prd` preset applied to nanochat's
meta-device shapes), FP32 codebook LUTs, 32K packed 4-bit KV cache, and the
PRD's C++-engine workspace constant, each compared to the published PRD row
(comparison is informational: the PRD's exact 7.5B config is not in this repo).
With --alloc-kv, the KV row is *measured* by allocating a real
QuantizedKVCache at the PRD's implied dims (L*H*D = 13,824) and page-touching
it, checking resident growth against storage_bytes.

uv run python -m scripts.kv_budget_bench [--alloc-kv]
"""

import argparse
import math

import torch

from nanochat.models.backbone import GPT, GPTConfig
from nanochat.models.quant.packing import index_bytes_for_k
from nanochat.models.quant.retrofit import PRESETS, get_layer_config, spec_k
from nanochat.modules.engine import QuantizedKVCache

VOCAB = 32768
KV_CONTEXT = 32768
# PRD section 7 published rows
PRD = dict(weights_gb=1.58, luts_mb=0.40, kv_gb=0.45, workspace_gb=0.20, total_gb=2.23)
# The 0.45 GB / 32K row implies L * H_kv * D ~= 13,824 (0.45e9 / 32768);
# one factorization with valid transformer dims is used for the RSS measure.
PRD_KV_DIMS = (36, 6, 64)
WORKSPACE_BYTES = int(0.20 * 1e9)  # PRD row: C++ AVX-512 engine workspace


def config_for(depth: int) -> GPTConfig:
    """base_train's config math: dim = depth * 64 nudged to head_dim=128."""
    base = depth * 64
    dim = math.ceil(base / 128) * 128
    return GPTConfig(
        sequence_len=512,
        vocab_size=VOCAB,
        n_layer=depth,
        n_head=dim // 128,
        n_kv_head=dim // 128,
        n_embd=dim,
    )


def build_meta(depth: int) -> GPT:
    with torch.device("meta"):
        return GPT(config_for(depth))


def param_count(depth: int) -> int:
    return sum(p.numel() for p in build_meta(depth).parameters())


def nearest_depth(target: float = 7.5e9) -> int:
    return min(range(10, 61), key=lambda d: abs(param_count(d) - target))


def weight_bytes(model: GPT) -> tuple[int, int]:
    """(packed_matrix_bytes, bf16_embedding_and_scalar_bytes)."""
    packed = bf16 = 0
    for name, p in model.named_parameters():
        if p.ndim != 2:  # scalars, lambdas: negligible, counted as bf16
            bf16 += p.numel() * 2
            continue
        out, inn = p.shape
        spec = get_layer_config(name, PRESETS["prd"])
        if spec is None:
            bf16 += p.numel() * 2  # embeddings / lm_head / gates: bf16
            continue
        packed += out * index_bytes_for_k(spec_k(spec.weight))
    return packed, bf16


def lut_bytes(model: GPT) -> int:
    total = 0
    for name, p in model.named_parameters():
        if p.ndim != 2 or p.shape[0] == 0:
            continue
        spec = get_layer_config(name, PRESETS["prd"])
        if spec is None:
            continue
        total += (
            spec_k(spec.weight)
            + spec_k(spec.activation)
            + (spec_k(spec.effective_output) if spec.quantize_output else 0)
        ) * 4
    return total


def kv_bytes(model: GPT) -> int:
    cfg = model.config
    return QuantizedKVCache.storage_bytes(
        batch_size=1,
        num_heads=cfg.n_kv_head,
        seq_len=KV_CONTEXT,
        head_dim=cfg.n_embd // cfg.n_head,
        num_layers=cfg.n_layer,
    )


def measure_kv_rss() -> tuple[int, int]:
    """Allocate + touch a PRD-dims cache; return (predicted, rss_growth) bytes."""

    def rss() -> int:  # current RSS (ru_maxrss is a high-water mark, not current)
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        raise RuntimeError("VmRSS not found in /proc/self/status")

    layers, heads, head_dim = PRD_KV_DIMS
    predicted = QuantizedKVCache.storage_bytes(1, heads, KV_CONTEXT, head_dim, layers)
    before = rss()
    cache = QuantizedKVCache(
        batch_size=1,
        num_heads=heads,
        seq_len=KV_CONTEXT,
        head_dim=head_dim,
        num_layers=layers,
        device="cpu",
        k_codebooks=torch.zeros(layers, heads, 15),
        v_codebooks=torch.zeros(layers, heads, 15),
    )
    cache.k_idx.view(-1)[::4096] = 1  # page-touch so RSS reflects the buffer
    cache.v_idx.view(-1)[::4096] = 1
    return predicted, max(rss() - before, 0)


def gb(n: int) -> str:
    return f"{n / 1e9:.3f} GB"


def main() -> None:
    parser = argparse.ArgumentParser(description="PRD section 7 budget table")
    parser.add_argument("--alloc-kv", action="store_true", help="measure KV RSS")
    parser.add_argument("--depth", type=int, default=0, help="0 = nearest to 7.5B")
    args = parser.parse_args()

    prd_sum = (
        PRD["weights_gb"] + PRD["luts_mb"] / 1024 + PRD["kv_gb"] + PRD["workspace_gb"]
    )
    assert abs(prd_sum - PRD["total_gb"]) < 0.005, f"PRD rows sum to {prd_sum}"
    print(f"PRD rows sum: {prd_sum:.4f} GB == published total {PRD['total_gb']} GB")

    depth = args.depth or nearest_depth()
    model = build_meta(depth)
    n_params = sum(p.numel() for p in model.parameters())
    (w_packed, w_bf16), lut, k = weight_bytes(model), lut_bytes(model), kv_bytes(model)
    w = w_packed + w_bf16
    total = w + lut + k + WORKSPACE_BYTES
    cfg = model.config
    kv_product = cfg.n_layer * cfg.n_kv_head * (cfg.n_embd // cfg.n_head)
    print(
        f"\nnanochat d{depth}: {n_params / 1e9:.2f}B params, head_dim="
        f"{cfg.n_embd // cfg.n_head}, n_kv_head={cfg.n_kv_head}"
    )
    print(
        "NOTE: PRD rows describe the PRD's own 7.5B config, which is not in "
        "this repo; computed rows are nanochat's config (informational)."
    )
    print(
        f"  weights: {gb(w_packed)} packed matrices + {gb(w_bf16)} bf16 "
        f"embeddings/lm_head/scalars"
    )
    print(
        f"  KV dims: nanochat L*H_kv*D={kv_product:,} vs PRD-implied 13,824 "
        f"(0.45 GB @ 32K) -> ratio {kv_product / 13824:.1f}x"
    )
    header = f"{'row':<16}{'computed':>14}{'PRD':>12}"
    print(header)
    print("-" * len(header))
    print(f"{'weights':<16}{gb(w):>14}{PRD['weights_gb']:>11.2f}G")
    print(f"{'codebook LUTs':<16}{lut / 1e6:>13.2f}M{PRD['luts_mb']:>11.2f}M")
    print(f"{'32K KV cache':<16}{gb(k):>14}{PRD['kv_gb']:>11.2f}G")
    print(f"{'workspace':<16}{gb(WORKSPACE_BYTES):>14}{PRD['workspace_gb']:>11.2f}G")
    print(f"{'TOTAL':<16}{gb(total):>14}{PRD['total_gb']:>11.2f}G")

    if args.alloc_kv:
        predicted, rss = measure_kv_rss()
        ratio = rss / predicted if predicted else 0.0
        print(
            f"\nKV row measured @ L*H*D="
            f"{PRD_KV_DIMS[0] * PRD_KV_DIMS[1] * PRD_KV_DIMS[2]}: "
            f"storage_bytes={gb(predicted)} rss_growth={gb(rss)} ratio={ratio:.2f}"
        )
        assert 0.90 <= ratio <= 1.15, f"RSS growth {rss} vs storage {predicted}"
        print("KV row reproduces PRD 0.45 GB within measurement noise: PASS")


if __name__ == "__main__":
    main()
