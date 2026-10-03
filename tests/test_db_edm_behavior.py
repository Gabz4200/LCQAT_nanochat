"""
Behavioral review of the DiffusionBlocks implementation against the paper
(SakanaAI DiffusionBlocks, arXiv 2506.14202), the reference implementation,
and plain autoregressive training.

Covers the four review findings plus the two preservation guarantees:

1. Block/noise mapping (paper Fig. 6, App. C): block 0 is the earliest layers
   and must own the HIGHEST noise range; block B-1 (latest layers) the lowest.
2. Clean-past AR conditioning (paper App. E.4): the denoiser input concatenates
   the clean sequence with the noisy sequence under a modified causal mask, so
   noisy future tokens condition on clean past tokens. Memory doubles in
   sequence length for the active block only; the forward stays a single pass.
3. EDM-versus-plain-AR: the two objectives are different kinds of training --
   different loss, different gradient reach (notably `lm_head`), different
   forward cost. Raw loss values must never be compared across them.
4. Plain-AR preservation: nothing in this pass may change `--db-blocks=0`
   training or the `train_step` escape hatch.
5. Checkpoint version guard: the corrected mapping is a breaking change, so
   `meta["db"]` carries `noise_map_version == 2` and the loader loudly rejects
   anything else instead of training on a silently reinterpreted schedule.

python -m pytest tests/test_db_edm_behavior.py -v
"""

import pytest
import torch

from nanochat.modules.experiments.tiny_models import make_engine
from nanochat.training.diffusion_blocks import (
    NOISE_MAP_VERSION,
    EquiProbabilityPartitioner,
)


def test_when_block_zero_then_it_owns_the_highest_noise_range() -> None:
    """Paper Fig. 6 / App. C: earliest layers denoise the highest noise."""
    part = EquiProbabilityPartitioner(num_blocks=4)
    lo0, hi0 = part.range_for_block(0)
    lo3, hi3 = part.range_for_block(3)
    assert float(lo0) > float(hi3), (
        f"block 0 [{float(lo0):.4g}, {float(hi0):.4g}] must sit above "
        f"block 3 [{float(lo3):.4g}, {float(hi3):.4g}]"
    )
    for _ in range(20):
        assert (
            float(part.sample_sigma(0, overlap=0.0))
            >= float(part.sample_sigma(3, overlap=0.0)) - 1e-6
        )


def test_when_partition_meta_then_it_versions_the_noise_map() -> None:
    meta = EquiProbabilityPartitioner(num_blocks=2).to_meta()
    assert meta["noise_map_version"] == NOISE_MAP_VERSION == 2


def test_when_old_noise_map_checkpoint_then_loader_rejects_it(tmp_path) -> None:
    """A v1 checkpoint would silently retarget every block's noise range."""
    import json

    from nanochat.modules.checkpoint_manager import build_model
    from tests.test_w0_ckpt_guard import _config_kwargs, _engine_matching_tokenizer

    engine, _ = _engine_matching_tokenizer()
    sd = engine.state_dict()
    step = 1
    torch.save(sd, tmp_path / f"model_{step:06d}.pt")
    meta = {
        "model_config": _config_kwargs(engine.model),
        # Deliberately the old mapping: version key absent means v1.
        "db": {"num_blocks": 2},
    }
    with open(tmp_path / f"meta_{step:06d}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with pytest.raises(RuntimeError, match="noise_map_version"):
        build_model(str(tmp_path), step, torch.device("cpu"), phase="eval")


def test_when_denoise_step_then_noisy_suffix_conditions_on_clean_prefix() -> None:
    """Paper App. E.4: the block input is [clean | noisy], not all-noisy."""
    seen: list[torch.Tensor] = []
    engine = make_engine(2)
    orig = engine._run_block_denoiser

    def spy(block_idx, noisy, sigma, seq_len, attn_mask=None, **kwargs):
        seen.append(noisy.detach().clone())
        return orig(block_idx, noisy, sigma, seq_len, attn_mask=attn_mask, **kwargs)

    engine._run_block_denoiser = spy  # type: ignore[method-assign]
    torch.manual_seed(0)
    idx = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        clean = torch.nn.functional.normalize(
            engine.model.transformer.wte(idx).float(), dim=-1
        )
    loss, _ = engine.denoise_step(idx, block_idx=0, clean=clean, backend="naive")
    assert torch.isfinite(loss)
    assert len(seen) == 1
    full, t = seen[0], idx.size(1)
    assert full.shape[1] == 2 * t, "input must be [clean | noisy] concatenation"
    # The clean half is exactly the target embeddings: conditioning on the
    # clean past, not on a second noisy draw.
    assert torch.equal(full[:, :t, :], clean)


def test_when_packed_mask_then_concat_mask_respects_documents() -> None:
    """The 2T mask must keep noisy tokens inside their own document's past."""
    from nanochat.training.diffusion_blocks import block_diagonal_mask

    engine = make_engine(2)
    t = 8
    mask = block_diagonal_mask([5, 3], t, device="cpu")
    full = engine._concat_causal_mask(mask)
    assert full.shape == (1, 1, 2 * t, 2 * t)
    full_bool = full if full.dtype == torch.bool else full > float("-inf")
    # Clean rows never see the noisy half: no information leakage into prefix.
    assert not full_bool[0, 0, :t, t:].any()
    # A noisy token in doc 2 (position t + 6) sees clean doc-2 past (5..6),
    # but not clean doc-1 positions (0..4) nor clean future (7).
    assert full_bool[0, 0, t + 6, 5:8].tolist() == [True, True, False]
    assert not full_bool[0, 0, t + 6, 0:5].any()
    # Same document rule on the noisy half itself.
    assert full_bool[0, 0, t + 6, t + 5 : t + 8].tolist() == [True, True, False]


def test_when_engine_backend_cpu_then_matches_naive() -> None:
    """The compiled kernel must not change training, only how it is computed."""
    torch.manual_seed(3)
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 12))
    torch.manual_seed(3)
    loss_naive, _ = engine.denoise_step(idx, block_idx=1, backend="naive")
    torch.manual_seed(3)
    loss_cpu, _ = engine.denoise_step(idx, block_idx=1, backend="cpu")
    assert torch.allclose(loss_naive, loss_cpu, atol=1e-5, rtol=1e-5)


def test_when_edm_versus_plain_ar_then_objectives_differ_in_kind() -> None:
    """Direct EDM-vs-plain-AR behavioral comparison on the same batch.

    Same batch, same weights, two different kinds of training: the EDM step
    denoises embeddings (weighted MSE, no logits, `lm_head` untouched, only the
    active block forwards) while the plain-AR step predicts next tokens
    (cross-entropy over the full depth, `lm_head` trained). Their raw loss
    values live on different scales and must never be compared directly.
    """
    torch.manual_seed(0)
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))

    loss_edm, _ = engine.denoise_step(idx, block_idx=0, backend="naive")
    assert torch.isfinite(loss_edm)
    loss_edm.backward()
    edm_with_grad = {n for n, p in engine.named_parameters() if p.grad is not None}
    assert not any(n.startswith("lm_head.") for n in edm_with_grad), (
        "the EDM objective predicts embeddings, never tokens: lm_head has no loss "
        "to attach a gradient to"
    )
    engine.zero_grad(set_to_none=True)

    loss_ar = engine.train_step(idx, idx, block_idx=0)
    assert torch.isfinite(loss_ar)
    loss_ar.backward()
    ar_with_grad = {n for n, p in engine.named_parameters() if p.grad is not None}
    assert any(n.startswith("lm_head.") for n in ar_with_grad), (
        "plain next-token cross-entropy trains lm_head"
    )


def test_when_train_step_then_denoise_backend_cannot_change_it() -> None:
    """Plain-AR preservation: the escape hatch ignores the kernel backend."""
    engine = make_engine(2)
    idx = torch.randint(0, 128, (2, 16))
    engine.denoise_backend = "naive"
    loss_naive = engine.train_step(idx, idx, block_idx=0)
    engine.denoise_backend = "cpu"
    loss_cpu = engine.train_step(idx, idx, block_idx=0)
    assert torch.equal(loss_naive, loss_cpu)
