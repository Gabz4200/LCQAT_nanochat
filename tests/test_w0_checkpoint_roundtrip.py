"""Checkpoint round-trip: the W0.2 regressions.

`base_train` and `chat_sft` used to save `orig_model.state_dict()` -- the bare
GPT -- while writing `meta["db"]`. The loader therefore took the DiffusionBlocks
branch, found no `db_adapters.*` / `db_denoise_head.*` keys, skipped both via its
`if adapters_sd:` guards, and loaded a **zero-initialized** denoise head and
adapter. The model looked like it had resumed and was silently zero.

Separately, `sparsity_mask` was `persistent=False`, so a resume rebuilt a fresh
*random* mask while reusing the weights that had been pruned against the previous
one.
"""

import torch

from nanochat.models.quant import PRESETS, inject_sparseprop_layers, retrofit_model
from nanochat.models.quant.retrofit import DEFAULT_PRESET
from nanochat.models.quant.sparseprop import SparsePropLinear
from nanochat.modules.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.training.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
)
from tests.conftest import build_active_tiny_gpt


def _build_engine(num_blocks=2):
    torch.manual_seed(0)
    model = retrofit_model(build_active_tiny_gpt(), PRESETS[DEFAULT_PRESET])
    engine = DiffusionBlockEngine(
        model, EquiProbabilityPartitioner(num_blocks=num_blocks)
    )
    engine.apply_lcqat(PRESETS[DEFAULT_PRESET])
    engine.apply_sparseprop(sparsity=0.75, with_lcqat=True)
    return engine


def test_when_engine_state_dict_then_it_carries_adapters_and_heads():
    engine = _build_engine()
    sd = engine.state_dict()
    assert any(k.startswith("db_adapters.") for k in sd)
    assert any(k.startswith("db_denoise_heads.") for k in sd)
    # Per-block heads: one head's worth of keys per block, not one shared head.
    head_keys = {k for k in sd if k.startswith("db_denoise_heads.")}
    for b in range(2):
        assert f"db_denoise_heads.{b}.weight" in head_keys


def test_when_save_and_load_then_engine_state_is_identical(tmp_path):
    engine = _build_engine()
    # Give every parameter a distinct value, so an identity check is meaningful.
    with torch.no_grad():
        for i, (_, p) in enumerate(engine.named_parameters()):
            p.fill_(0.01 * (i + 1))
    state = {k: v.clone() for k, v in engine.state_dict().items()}

    save_checkpoint(
        str(tmp_path),
        1,
        state,
        {},
        {"model_config": {}, "db": {"num_blocks": 2, "noise_map_version": 2}},
    )
    loaded, _, meta = load_checkpoint(
        str(tmp_path), 1, torch.device("cpu"), load_optimizer=False
    )
    assert meta["db"]["num_blocks"] == 2
    for key, value in loaded.items():
        assert torch.equal(value, state[key]), f"{key} changed across the round-trip"


def test_when_sparse_structure_buffers_are_persistent():
    """The W0.5 regression: a resume used to re-roll a random mask."""
    torch.manual_seed(0)
    model = retrofit_model(build_active_tiny_gpt(), PRESETS[DEFAULT_PRESET])
    inject_sparseprop_layers(model, sparsity=0.75, with_lcqat=True)
    sd = model.state_dict()
    sparse = [m for m in model.modules() if isinstance(m, SparsePropLinear)]
    assert sparse, "no sparse layers were injected"
    for module in sparse:
        prefix = None
        for name, mod in model.named_modules():
            if mod is module:
                prefix = name
                break
        assert f"{prefix}.sparsity_mask" in sd, "mask must be persistent"
        for buf in ("w_ptr", "w_col", "w_ptr_csc", "w_row"):
            assert f"{prefix}.{buf}" in sd, f"{buf} must be a persistent buffer"


def test_when_save_then_load_then_the_mask_is_the_same_pattern():
    """Not just present: the *same* pattern, so pruned weights stay consistent."""
    torch.manual_seed(0)
    model = retrofit_model(build_active_tiny_gpt(), PRESETS[DEFAULT_PRESET])
    inject_sparseprop_layers(model, sparsity=0.75, with_lcqat=True)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    mask_keys = [k for k in state if k.endswith("sparsity_mask")]
    assert mask_keys
    # And the pattern is genuinely sparse, not a dense placeholder.
    for key in mask_keys:
        mask = model.state_dict()[key]
        density = mask.float().mean().item()
        assert 0.0 < density < 1.0, f"{key} density {density} is not sparse"
