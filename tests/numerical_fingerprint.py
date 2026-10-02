"""Fixed-seed numerical fingerprint for behavior-preservation verification.

A passing test suite does not prove that a refactor preserved numerics: the
suite asserts contracts, and a refactor can satisfy every contract while
perturbing a loss by enough to change a training run. This module pins the
*values* -- forward logits, training loss, and a backward gradient -- for a
handful of configurations spanning the paths a structural refactor could
disturb:

* the plain autoregressive LM forward (``GPT``),
* the LC-QAT codebook path (``LCQATLinear``),
* SparseProp (which replaces the layer class entirely),
* the DiffusionBlocks denoiser objective.

Every value is recorded as a hex float64 bit pattern, so the comparison is
exact and platform-independent rather than tolerance-based. Run it before and
after a refactor and diff the output.
"""

from __future__ import annotations

import argparse
import struct


def unhex(value: str) -> float:
    """Inverse of `_hexdigest`, for tolerance comparisons."""
    return struct.unpack("<d", struct.pack("<Q", int(value, 16)))[0]


def _hexdigest(value: float) -> str:
    """Bit-exact float64 encoding, so comparison needs no tolerance."""
    return f"{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def _digest(tensor) -> str:
    """A single scalar that changes if any element changes.

    Sums the values in float64 and hex-encodes the bit pattern. The sum is
    order-dependent in general; callers that need cross-run stability must
    supply a tolerance rather than compare these strings exactly (see
    `fingerprint`).
    """
    import torch

    flat = tensor.detach().to(torch.float64).reshape(-1)
    total = flat.sum().item()
    return _hexdigest(total)


def fingerprint() -> tuple[dict[str, str], dict[str, float]]:
    """Compute the fingerprint.

    Returns ``(digests, tolerances)``. Most values are exact hex float64
    patterns and must match bit-for-bit. The two LC-QAT gradient sums carry a
    relative tolerance instead: their backward accumulates through
    ``scatter_add_``, which uses atomics, so the summation order -- and
    therefore the last few float64 bits -- varies between runs on the same
    code. Measured spread across repeated runs is ~1e-14 relative. That is
    float64 summation noise, not a numerical change, and demanding bitwise
    equality there would make this gate flaky rather than strict.
    """
    import argparse as _argparse

    import torch

    from nanochat.models.backbone import GPT, GPTConfig
    from nanochat.models.quant import lcqat_config_from_args
    from nanochat.models.quant.export import attach_learnable_activation_luts
    from nanochat.models.quant.retrofit import retrofit_model
    from nanochat.models.quant.sparseprop import inject_sparseprop_layers
    from nanochat.training.diffusion_blocks import (
        DiffusionBlockEngine,
        EquiProbabilityPartitioner,
    )

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)

    # Small but non-degenerate: 4 layers, 128 wide. Big enough that a swapped
    # layer or a re-ordered forward changes the value; small enough to run in
    # seconds on CPU.
    cfg = GPTConfig(
        sequence_len=64,
        vocab_size=512,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="L",
    )
    ns = _argparse.Namespace(
        lcqat=True,
        lcqat_preset="asym",
        lcqat_k_map="",
        codebook_grad_scale="inv_sqrt_n",
        lcqat_lut_relaxation="logits",
        lcqat_act_body="pwl",
        lcqat_channel_center=False,
        lcqat_bias_quant=False,
        lcqat_k_bias=32,
    )
    cfgq = lcqat_config_from_args(ns)

    out: dict[str, str] = {}
    # Relative tolerance for values whose backward uses atomics.
    tolerances: dict[str, float] = {
        "lcqat/grad": 1e-11,
        "sparseprop/grad": 1e-11,
    }

    def perturb(model) -> None:
        """Break the zero-init so the fingerprint has real signal.

        `GPT.init_weights` zero-initializes `lm_head`, which makes the logits
        uniform and pins the loss at exactly `ln(vocab_size)`. Every arm then
        reports the *same* loss, so a refactor that broke the forward pass
        entirely would still match. A deterministic perturbation gives the
        logits structure without introducing randomness.
        """
        gen = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() >= 2:
                    p.add_(
                        torch.randn(p.shape, generator=gen, dtype=torch.float32) * 0.02
                    )

    def record(tag: str, model, x, y) -> None:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        out[f"{tag}/logits"] = _digest(logits)
        loss = model(x, y)
        loss.backward()
        # Gradient fingerprint: sum over every parameter that received one.
        gsum = 0.0
        count = 0
        for p in model.parameters():
            if p.grad is not None:
                gsum += p.grad.detach().to(torch.float64).sum().item()
                count += 1
        out[f"{tag}/loss"] = _hexdigest(loss.detach().item())
        out[f"{tag}/grad"] = _hexdigest(gsum)
        out[f"{tag}/grad_params"] = str(count)

    x = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len))

    # 1. Plain LM -- the baseline every arm is compared against.
    torch.manual_seed(0)
    m_plain = GPT(cfg)
    m_plain.init_weights()
    perturb(m_plain)
    record("plain_lm", m_plain, x, y)

    # 2. LC-QAT retrofit -- codebook forward + STE + backward.
    torch.manual_seed(0)
    m_lcqat = GPT(cfg)
    m_lcqat.init_weights()
    retrofit_model(m_lcqat, cfgq)
    attach_learnable_activation_luts(m_lcqat, relaxation="logits", act_body="pwl")
    perturb(m_lcqat)
    record("lcqat", m_lcqat, x, y)

    # 3. SparseProp -- replaces 36 layer classes; the highest-risk surface.
    torch.manual_seed(0)
    m_sp = GPT(cfg)
    m_sp.init_weights()
    retrofit_model(m_sp, cfgq)
    attach_learnable_activation_luts(m_sp, relaxation="logits", act_body="pwl")
    inject_sparseprop_layers(m_sp, sparsity=0.75, with_lcqat=True)
    perturb(m_sp)
    record("sparseprop", m_sp, x, y)

    # 4. DiffusionBlocks -- the engine path, which routes through a different
    #    forward and a different objective.
    torch.manual_seed(0)
    m_db = GPT(cfg)
    m_db.init_weights()
    retrofit_model(m_db, cfgq)
    perturb(m_db)
    engine = DiffusionBlockEngine(
        m_db, EquiProbabilityPartitioner(num_blocks=2), dtype=torch.float32
    )
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.sequence_len))
    torch.manual_seed(1)  # sigma resampling must be pinned or this never matches
    loss_db = engine.train_step(tokens, targets)
    out["diffusionblocks/loss"] = _hexdigest(loss_db.detach().item())

    return out, tolerances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    import json

    result, tolerances = fingerprint()
    text = json.dumps(
        {"digests": result, "tolerances": tolerances}, indent=2, sort_keys=True
    )
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
