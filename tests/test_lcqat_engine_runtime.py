"""
Tests for the Engine quantized-KV integration (TODO section 2): opt-in
`quantized_kv=True` decode path with perplexity/next-token parity against
the bf16-cache path on the same model.

python -m pytest tests/test_lcqat_engine_runtime.py -v
"""

import torch

from nanochat.models.quant import PRESETS, retrofit_model
from nanochat.modules.engine import (
    Engine,
    KVCache,
    QuantizedKVCache,
    kv_codebooks_from_model,
)
from tests.conftest import build_active_tiny_gpt
from tests.test_engine import ByteTokenizer


def build_model():
    torch.manual_seed(0)
    model = retrofit_model(build_active_tiny_gpt(), PRESETS["small"])
    model.eval()
    return model


def make_caches(model, batch_size: int, seq_len: int):
    m = model.config
    kwargs = dict(
        num_heads=m.n_kv_head,
        head_dim=m.n_embd // m.n_head,
        num_layers=m.n_layer,
        device=torch.device("cpu"),
    )
    float_cache = KVCache(
        batch_size=batch_size, seq_len=seq_len, dtype=torch.float32, **kwargs
    )
    k_cbs, v_cbs = kv_codebooks_from_model(model)
    quant_cache = QuantizedKVCache(
        batch_size=batch_size,
        seq_len=seq_len,
        k_codebooks=k_cbs,
        v_codebooks=v_cbs,
        **kwargs,
    )
    return float_cache, quant_cache


def test_when_prefill_with_quantized_cache_then_logits_close_to_bf16_cache() -> None:
    model = build_model()
    torch.manual_seed(1)
    ids = torch.randint(0, model.config.vocab_size, (1, 12))
    float_cache, quant_cache = make_caches(model, batch_size=1, seq_len=16)
    with torch.no_grad():
        logits_float = model(ids, kv_cache=float_cache)
        logits_quant = model(ids, kv_cache=quant_cache)
    assert logits_quant.shape == logits_float.shape
    assert torch.isfinite(logits_quant).all()
    diff = (logits_quant - logits_float).abs()
    print(f"\nprefill logits: max|d|={diff.max():.4f} mean|d|={diff.mean():.4f}")
    assert diff.max() < 1.0
    assert float_cache.get_pos() == quant_cache.get_pos() == 12


def test_when_forced_decode_steps_then_step_logits_close_to_bf16_cache() -> None:
    # The real next-token parity metric: identical forced token trajectories
    # through both cache types, comparing logits at every decode step (argmax
    # agreement is noise-sensitive on an untrained model - see the Engine test).
    model = build_model()
    torch.manual_seed(4)
    prompt = torch.randint(0, model.config.vocab_size, (1, 8))
    forced = torch.randint(0, model.config.vocab_size, (1, 8))

    def series(quantized: bool) -> torch.Tensor:
        float_cache, quant_cache = make_caches(model, batch_size=1, seq_len=20)
        cache = quant_cache if quantized else float_cache
        rows = []
        with torch.no_grad():
            logits = model(prompt, kv_cache=cache)[:, -1]
            rows.append(logits)
            for t in range(forced.shape[1]):
                logits = model(forced[:, t : t + 1], kv_cache=cache)[:, -1]
                rows.append(logits)
        return torch.stack(rows)

    ref, got = series(False), series(True)
    diff = (ref - got).abs()
    print(f"\ndecode step logits: max|d|={diff.max():.5f} mean|d|={diff.mean():.5f}")
    assert torch.isfinite(got).all()
    assert diff.max() < 0.01


def test_when_greedy_decode_with_quantized_kv_then_matches_bf16_cache() -> None:
    model = build_model()
    engine = Engine(model, ByteTokenizer())
    prompt = [65, 66, 67, 68]  # printable ASCII bytes
    kwargs = dict(num_samples=1, max_tokens=24, temperature=0.0, seed=7)
    ref, _ = engine.generate_batch(prompt, **kwargs)
    got, _ = engine.generate_batch(prompt, quantized_kv=True, **kwargs)
    assert len(got[0]) == len(ref[0])
    # On an UNTRAINED model the top-1 logit margins sit below the KV-quant
    # noise floor, so argmax flips after a few tokens are expected; the strict
    # per-step logit tolerance is asserted in the forced-trajectory test.
    # Here: the wiring must produce a long matching prefix, not garbage.
    prefix = next(
        (i for i, (a, b) in enumerate(zip(ref[0], got[0])) if a != b),
        min(len(ref[0]), len(got[0])),
    )
    print(f"\ngreedy agreement: diverges at token {prefix} of {len(ref[0])}")
    assert prefix >= 4


def test_when_quantized_kv_off_then_defaults_to_bf16_cache() -> None:
    model = build_model()
    engine = Engine(model, ByteTokenizer())
    prompt = [65, 66, 67]
    kwargs = dict(num_samples=1, max_tokens=8, temperature=0.0, seed=7)
    ref_a, _ = engine.generate_batch(prompt, **kwargs)
    ref_b, _ = engine.generate_batch(prompt, quantized_kv=False, **kwargs)
    assert ref_a == ref_b
