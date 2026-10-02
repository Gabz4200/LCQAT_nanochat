"""The asymmetric codebook split and the `asym` preset (W2.1-W2.4).

The motivation is concrete and lives in `nanochat/gpt.py`:

    x = F.relu(x).square()   # MLP.forward

so the MLP hidden tensor -- `4 * n_embd`, the largest activation in the network
-- is **non-negative**. It is quantized twice: by `mlp.c_fc`'s output quantizer
and by `mlp.c_proj`'s activation quantizer. Under the old symmetric 15-level
codebook, 7 of those 15 levels sat below a value the tensor can never take.

`m_neg = 0` gives a one-sided codebook: 8 levels, all usable, even K, which the
old odd-K constraint made impossible.
"""

import pytest
import torch

from nanochat.lcqat import PRESETS, LCQATLinear, retrofit_model
from nanochat.lcqat.codebook import MemoryEfficientLearnedCodebook
from nanochat.lcqat.retrofit import (
    DEFAULT_PRESET,
    get_layer_config,
    prepare_lcqat_before_load,
    spec_k,
    spec_split,
)
from tests.conftest import build_active_tiny_gpt

# K values at real bit boundaries. Even K is only legal because of the
# asymmetric split (m_neg = 0 + 1 anchor + m_pos).
EVEN_K = (4, 8, 16, 255)


def test_when_default_preset_then_it_is_asym():
    assert DEFAULT_PRESET == "asym"
    assert "asym" in PRESETS


def test_when_asym_preset_then_mlp_non_negative_tensors_get_one_sided_codebooks():
    cfg = PRESETS[DEFAULT_PRESET]
    c_fc = get_layer_config("transformer.h.0.mlp.c_fc", cfg)
    assert c_fc is not None
    _, _, quantize_out, out_spec = c_fc
    assert quantize_out is True
    assert out_spec == (0, 7)

    c_proj = get_layer_config("transformer.h.0.mlp.c_proj", cfg)
    assert c_proj is not None
    _, act_spec, _, _ = c_proj
    assert act_spec == (0, 7)


def test_when_asym_preset_then_attention_tensors_keep_both_sides():
    """Attention inputs are RMSNorm'd and signed; a one-sided codebook would throw
    away the entire negative half of every attention activation."""
    cfg = PRESETS[DEFAULT_PRESET]
    for name in (
        "transformer.h.0.attn.c_q",
        "transformer.h.0.attn.c_k",
        "transformer.h.0.attn.c_v",
        "transformer.h.0.attn.c_proj",
    ):
        spec = get_layer_config(name, cfg)
        assert spec is not None
        assert spec[0][0] > 0, f"{name} weight codebook should have negative levels"
        assert spec[1][0] > 0, f"{name} activation codebook should have negative levels"


def test_when_one_sided_codebook_then_every_level_is_non_negative():
    """The payoff: no level is spent below the tensor's minimum."""
    cb = MemoryEfficientLearnedCodebook(m_neg=0, m_pos=7, init_min=0.0, init_max=2.0)
    book = cb.get_codebook()
    assert book.numel() == 8
    assert (book >= 0).all()
    assert book[0].item() == 0.0, "anchor at index m_neg == 0"


def test_when_one_sided_codebook_then_negative_input_clamps_to_the_anchor():
    cb = MemoryEfficientLearnedCodebook(m_neg=0, m_pos=7, init_min=0.0, init_max=2.0)
    out = cb(torch.tensor([-100.0, -1e-9, 0.0]))
    assert (out.value == 0.0).all()
    assert (out.indices == 0).all()


@pytest.mark.parametrize("m_neg,m_pos", [(3, 0), (0, 3), (0, 7), (0, 14)])
def test_when_exact_k_then_split_reconstructs_k(m_neg, m_pos):
    cb = MemoryEfficientLearnedCodebook(m_neg=m_neg, m_pos=m_pos)
    assert cb.K == spec_k((m_neg, m_pos))


def test_when_even_k_then_no_odd_constraint_remains():
    """K=4 and K=8 are real bit boundaries the old odd-K rule forbade."""
    for k in EVEN_K:
        m_neg, m_pos = spec_split(k)
        assert m_neg + 1 + m_pos == k
        cb = MemoryEfficientLearnedCodebook(m_neg=m_neg, m_pos=m_pos)
        assert cb.K == k


def test_when_from_k_then_odd_splits_evenly_and_even_gains_the_extra_level():
    odd = MemoryEfficientLearnedCodebook.from_k(15)
    assert (odd.m_neg, odd.m_pos) == (7, 7)
    # An even K cannot be symmetric; the extra level goes positive.
    even = MemoryEfficientLearnedCodebook.from_k(8)
    assert (even.m_neg, even.m_pos) == (3, 4)
    assert even.K == 8


def test_when_anchor_then_bucketize_of_zero_is_exactly_the_anchor_index():
    """The structural guarantee W3's pruning depends on.

    A codebook *level* of 0.0 lies strictly between two midpoints, so
    bucketize(0.0) returns exactly m_neg in any regime. It is an index
    assignment rather than a computed value, so unlike the level ordering it
    does not degrade with floating-point precision.
    """
    for m_neg in (0, 1, 5, 7):
        cb = MemoryEfficientLearnedCodebook(m_neg=m_neg, m_pos=7)
        assert cb.bucketize(torch.zeros(1)).item() == m_neg
        assert (cb(torch.zeros(8)).value == 0.0).all()


def test_when_asym_applied_to_a_model_then_mlp_codebooks_are_one_sided():
    model = retrofit_model(build_active_tiny_gpt(), PRESETS[DEFAULT_PRESET])
    c_fc = model.get_submodule("transformer.h.0.mlp.c_fc")
    c_proj = model.get_submodule("transformer.h.0.mlp.c_proj")
    assert isinstance(c_fc, LCQATLinear) and isinstance(c_proj, LCQATLinear)
    for quantizer in (c_fc.out_quantizer, c_proj.act_quantizer):
        assert quantizer.m_neg == 0 and quantizer.m_pos == 7
        assert (quantizer.get_codebook() >= 0).all()


def test_when_asym_model_forward_then_activation_never_goes_negative():
    """End-to-end: the one-sided codebook emits no negative activation."""
    torch.manual_seed(0)
    model = retrofit_model(build_active_tiny_gpt(), PRESETS[DEFAULT_PRESET])
    c_proj = model.get_submodule("transformer.h.0.mlp.c_proj")
    x = torch.rand(2, 16, c_proj.in_features) * 4.0  # relu^2-like
    with torch.no_grad():
        quantized = c_proj.act_quantizer(x)
    assert (quantized.value >= 0).all()


def test_when_asym_config_round_trips_through_prepare_before_load():
    """Splits survive the meta JSON round-trip (as lists) and rebuild the same
    codebooks, which is what makes a resume faithful."""
    model = retrofit_model(build_active_tiny_gpt(), PRESETS[DEFAULT_PRESET])
    state = {k: v.clone() for k, v in model.state_dict().items()}
    fresh = build_active_tiny_gpt()
    active = prepare_lcqat_before_load(
        fresh, state, PRESETS[DEFAULT_PRESET].as_dict(), None
    )
    assert active == PRESETS[DEFAULT_PRESET]
    fresh.load_state_dict(state, strict=True)
    c_proj = fresh.get_submodule("transformer.h.0.mlp.c_proj")
    assert c_proj.act_quantizer.m_neg == 0
    assert c_proj.act_quantizer.m_pos == 7
