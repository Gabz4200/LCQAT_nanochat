"""
Microbench: LC-QAT index-fetch matmul paths vs torch F.linear.

Two tables at the PRD's suggested shapes (m in {768, 4096},
n in {768, 2048}):
  1. dispatch_gemv (K=3 trit storage) backends;
  2. dispatch_index_linear (K in {3, 15, 255, 257} storage formats) at
     decode shape T=1.
Both sanity-check the naive backend against F.linear on dequantized
weights first, so no timing row can compare against a wrong baseline.
These are guard numbers for the quantized runtime (memory is the
optimization), not a mul-less performance claim.

uv run python -m scripts.gemv_bench
"""

import argparse
import statistics
import time

import torch

from nanochat.models.quant.packing import pack_weight_indices
from nanochat.ops import dispatch_gemv, dispatch_index_linear
from nanochat.ops.kernels.gpu_loader import vulkan_available


def dequantize(act_indices, act_lut, weight_indices, scale_neg, scale_pos):
    """Same math as the quantized path, expressed as dense tensors for F.linear."""
    x = act_lut[act_indices.long()]
    levels = torch.tensor([-scale_neg, 0.0, scale_pos])
    w = levels[weight_indices.long()]
    return x, w


def time_fn(fn, iters: int, warmup: int = 5) -> float:
    """Median wall time per call, in milliseconds."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def bench_gemv(args) -> None:
    backends = ["naive", "cpu"] + (["gpu"] if vulkan_available() else [])
    print(
        f"\n== lcqat_gemv_k3 (K=3 storage)  backends: {backends}  "
        f"iters: {args.iters} (median) =="
    )
    header = f"{'m':>5} {'n':>5} {'backend':>8} {'ms':>10} {'vs F.linear':>12}"
    print(header)
    print("-" * len(header))
    for m in args.m:
        for n in args.n:
            torch.manual_seed(0)
            act = torch.randint(0, 15, (n,), dtype=torch.uint8)
            lut = torch.linspace(-2.0, 2.0, 15)
            w = torch.randint(0, 3, (m, n), dtype=torch.uint8)
            s_neg, s_pos = 0.37, 0.42
            x, wd = dequantize(act, lut, w, s_neg, s_pos)

            base = time_fn(lambda: torch.nn.functional.linear(x, wd), args.iters)
            out = dispatch_gemv(act, lut, w, s_neg, s_pos, backend="naive")
            expected = torch.nn.functional.linear(x, wd)
            assert torch.allclose(out, expected, atol=1e-5), (
                "naive backend disagrees with F.linear dequant baseline"
            )
            for backend in backends:
                ms = time_fn(
                    lambda b=backend: dispatch_gemv(
                        act, lut, w, s_neg, s_pos, backend=b
                    ),
                    args.iters,
                )
                print(f"{m:>5} {n:>5} {backend:>8} {ms:>10.3f} {base / ms:>11.2f}x")
            print(f"{'':>5} {'':>5} {'linear':>8} {base:>10.3f} {'1.00x':>12}")


def bench_index_linear(args) -> None:
    backends = ["naive", "cpu"] + (["gpu"] if vulkan_available() else [])
    print(
        f"\n== lcqat_index_linear (K-selected storage, T=1 decode)  "
        f"backends: {backends}  iters: {args.iters} (median) =="
    )
    header = f"{'m':>5} {'n':>5} {'K':>4} {'backend':>8} {'ms':>10} {'vs F.linear':>12}"
    print(header)
    print("-" * len(header))
    for m in args.m:
        for n in args.n:
            for k in (3, 15, 255, 257):
                torch.manual_seed(k)
                act = torch.randint(0, 15, (1, n), dtype=torch.uint8)
                act_lut = torch.linspace(-2.0, 2.0, 15)
                w = torch.randint(0, k, (m, n))
                w = w.to(torch.int32 if k > 255 else torch.uint8)
                w_lut = torch.sort(torch.randn(k)).values
                packed, fmt = pack_weight_indices(w, k)
                case = dict(
                    act_indices=act,
                    act_lut=act_lut,
                    weight_indices=packed,
                    weight_lut=w_lut,
                    n=n,
                    format=fmt,
                )
                x = act_lut[act.long()]
                wd = w_lut[w.long()]
                base = time_fn(lambda: torch.nn.functional.linear(x, wd), args.iters)
                out = dispatch_index_linear(**case, backend="naive")
                assert torch.allclose(
                    out, torch.nn.functional.linear(x, wd), atol=1e-5
                ), "naive index backend disagrees with F.linear dequant baseline"
                for backend in backends:
                    ms = time_fn(
                        lambda b=backend, c=case: dispatch_index_linear(**c, backend=b),
                        args.iters,
                    )
                    print(
                        f"{m:>5} {n:>5} {k:>4} {backend:>8} {ms:>10.3f} "
                        f"{base / ms:>11.2f}x"
                    )
                print(
                    f"{'':>5} {'':>5} {'':>4} {'linear':>8} {base:>10.3f} {'1.00x':>12}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="LC-QAT index-fetch microbench")
    parser.add_argument("--m", type=int, nargs="+", default=[768, 4096])
    parser.add_argument("--n", type=int, nargs="+", default=[768, 2048])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--op",
        choices=["gemv", "index", "both"],
        default="both",
        help="which op family to bench",
    )
    args = parser.parse_args()
    if args.op in ("gemv", "both"):
        bench_gemv(args)
    if args.op in ("index", "both"):
        bench_index_linear(args)


if __name__ == "__main__":
    main()
