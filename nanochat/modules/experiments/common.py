"""Shared probe builders for the paired LC-QAT ablation experiments.

Nothing here measures anything on its own. Every name is used by at least two
of the nine experiments, or by the CLI driver in `scripts/lcqat_ablation.py`,
which owns the argparse surface, the claim checks and the leaderboard writer.
"""

from __future__ import annotations

import argparse

import torch

from nanochat.models.quant.ablation_metrics import select_quantizer
from nanochat.models.quant.retrofit import PRESETS, retrofit_model
from nanochat.modules.experiments.tiny_models import (
    build_active_tiny_gpt,
    make_engine,
)

#: Preset the `asym` experiment treats as the improvement, and the one it beats.
BASELINE_PRESET = "small"
VARIANT_PRESET = "asym"
#: The two objectives `--db-objective` selects between.
OBJECTIVE_CE = "ce"
OBJECTIVE_EDM = "edm"
#: The two block-sampling modes `--db-block-sampling` selects between.
SAMPLING_STEP = "step"
SAMPLING_MICRO = "micro"


def probe_layer(preset: str):
    """Retrofit a fresh tiny model and return the MLP down-projection.

    `c_proj` is chosen because it is the layer whose *activation* input is
    `relu^2` and therefore non-negative -- the tensor `asym` exists to serve.
    A model that shares no state with other arms is required, since retrofitting
    mutates in place.
    """
    model = build_active_tiny_gpt()
    retrofit_model(model, PRESETS[preset])
    return model.transformer.h[0].mlp.c_proj


def non_negative_probe(in_features: int, rows: int, seed: int) -> torch.Tensor:
    """A probe with the shape of the real `relu^2` input: non-negative.

    `asym`'s justification is that `mlp.forward` computes `F.relu(x).square()`,
    so the tensor `c_proj` actually quantizes never goes below zero. Measuring
    a symmetric signed probe would test the opposite of what the preset is for,
    and would report a loss that says nothing about the claim.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, in_features, generator=gen).relu().square()


def expected_1_over_sqrt_n(n_elements: int) -> float:
    """The `1/sqrt(N)` factor the PRD 2.4 scaling applies."""
    return 1.0 / (n_elements**0.5)


def build_probe_engine(args: argparse.Namespace):
    """A tiny DiffusionBlocks engine, depth-configurable and with live zero-inits.

    `make_engine(active=True)` is what makes the gradient-reachability
    measurements meaningful: the denoise heads and the adapter output layer are
    zero-initialized by design, and a zero tensor transmits no gradient, so
    every "did the gradient arrive" assertion would pass vacuously against a
    fresh engine.

    Depth is threaded through `make_engine`'s own `n_layer` argument rather than
    done here. `make_engine` builds its own model, so resizing a model built in
    this function and then handing `n_layer=None` to `make_engine` would build a
    *second* model at the default depth and silently discard the first -- the
    knob would appear to work while measuring the wrong depth.

    The engine is deterministic across arms, which is what pairing requires:
    two arms at a given seed see a byte-identical model, so a per-seed
    difference in the result is a difference in the probe. Probes are seeded
    separately by `block_probe_tensors`.
    """
    return make_engine(
        num_blocks=args.ablation_blocks, n_layer=args.ablation_n_layer, active=True
    )


def block_probe_tensors(
    args: argparse.Namespace, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(value probe, token ids, CE targets)` for the block-level probes.

    The value tensor is the `relu^2`-shaped one the driver already uses for the
    activation quantizers: `mlp.forward` computes `F.relu(x).square()`, so the
    tensors the real blocks see at their input are non-negative. It is the
    *clean* target the block's residual identity reconstructs.

    Token ids and CE targets both exist because the two objectives take
    different arguments. `denoise_step(idx, ...)` reads `idx` for its sequence
    length only, and `train_step(idx, targets, ...)` needs real targets to
    return a scalar cross-entropy: `GPT.forward` returns *logits* when
    `targets is None`, and backpropagating through logits makes
    `float(loss.detach())` raise rather than measure anything. All three come
    from one generator so both arms see identical tokens.
    """
    model = build_active_tiny_gpt()
    n_embd = int(model.config.n_embd)
    vocab = int(model.config.vocab_size)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(
        0, vocab, (args.ablation_batch, args.ablation_seq), generator=gen
    )
    targets = torch.randint(
        0, vocab, (args.ablation_batch, args.ablation_seq), generator=gen
    )
    probe = non_negative_probe(
        n_embd, args.ablation_batch * args.ablation_seq, seed
    ).view(args.ablation_batch, args.ablation_seq, n_embd)
    return probe, idx, targets


def engine_named_parameters(engine) -> list[tuple[str, torch.nn.Parameter]]:
    """The engine's parameters, as `(name, param)` in one deterministic order."""
    return list(engine.named_parameters())


def codebook_of(layer, which: str = "out") -> torch.Tensor:
    """A quantizer's codebook as an FP32 tensor.

    `which` follows `select_quantizer`'s roles: `"act"` is the layer's
    activation (input) quantizer, `"out"` its output quantizer, and `"weight"`
    its weight quantizer. Read off the layer rather than rebuilt, so the table
    measured is the one the model actually uses -- and so a layer that lacks the
    requested quantizer raises here instead of silently falling back to the
    weight quantizer, which would measure something else entirely.
    """
    quantizer = select_quantizer(layer, which)
    return quantizer.get_codebook().detach().to(torch.float32)


def count_codebook_cost(module) -> tuple[int, int]:
    """`(parameter count, artifact bytes)` for a quantizer or codebook module.

    Bytes are the trainable parameters at fp32 plus every persistent buffer at
    its own element size -- i.e. what the exported artifact actually carries,
    which is the cost `--db-sigma-codebook` is meant to be judged on. Buffers
    that are not persistent in `state_dict` (the input/output codebooks a
    `LearnableIndexLut` keeps for its own indexing) are excluded, because the
    owning quantizer already ships them in the checkpoint and counting them
    twice would overstate the artifact.
    """
    params = sum(p.numel() for p in module.parameters())
    # Persistence has to be read off `state_dict`, not off `named_buffers`:
    # the latter yields non-persistent buffers too, so filtering by it is a
    # tautology and every derived buffer gets priced as if it shipped.
    persistent = set(module.state_dict())
    buffers = sum(
        b.numel() * b.element_size()
        for name, b in module.named_buffers()
        if name in persistent
    )
    return params, params * 4 + buffers
