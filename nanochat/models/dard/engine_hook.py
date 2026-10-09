"""DARD's attachment to the DiffusionBlocks engine.

**The problem this solves.** DARD's penalty is a function of a *vocabulary
distribution*: `softmax(logits)`. `denoise_step` returns denoised *embeddings*,
and the EDM objective has no logits anywhere, so there is nothing for the
penalty to attach to. Rejecting DARD under `--db-objective edm` (this repo's
default) would make the feature unusable in the default configuration.

**The resolution.** The denoiser is already a classifier: `generate()` projects
the denoised state through `lm_head` and samples from the result (see
`DiffusionBlockEngine.generate`, the `softcap * tanh(logits / softcap)` step).
DARD attaches at exactly that projection -- `denoiser_logits` below reproduces
it verbatim -- so the distribution DARD penalizes is the same one the sampler's
own next-token decision uses. No new inference concept, and no second
vocabulary head.

**A consequence that is a real behavioural change, stated rather than
hidden.** `lm_head` currently receives **no** gradient from the EDM objective
-- a property `tests/test_db_edm_behavior.py` pins deliberately, since the
denoiser's targets are embeddings and gradient into `lm_head` would be
untargeted. DARD under EDM creates a new, legitimate gradient path into
`lm_head`: it is now being trained, through the rule penalty, to put mass on
rule-consistent tokens. `test_dard_diffusion_blocks.py` pins both the old
property (DARD off: no gradient) and the new one (DARD on: gradient present,
and confined to the aux term).
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Matches the logit softcap in `GPT.forward` and `DiffusionBlockEngine.generate`. Duplicated as a module constant rather than imported so this file does not depend on backbone internals; the test that checks it asserts the two agree.
LOGCAP_SOFTMAX = 15.0


def denoiser_logits(
    head: nn.Module, hidden: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """Project denoised `hidden` to a capped vocabulary logit tensor.

    Mirrors `GPT.forward`'s tail and `DiffusionBlockEngine.generate`'s logit
    step: slice the padded vocabulary, cast to fp32, softcap. The softcap
    matters numerically, not cosmetically -- without it, a denoiser whose
    hidden state has drifted produces logits large enough that
    `softmax` saturates and the rule penalty's gradients vanish, which is the
    underfailure PRD §5 is about.

    Args:
        head: the `lm_head` (possibly LC-QAT-retrofitted; the call goes through
            its `forward`, so the quantized path is used when one is installed).
        hidden: `(B, T, n_embd)` denoised states.
        vocab_size: slice the padded vocabulary down to the real one.

    Returns:
        `(B, T, vocab_size)` float32 logits, in the graph, so DARD's penalty
        backpropagates into `lm_head` and the active block.
    """
    logits = head(hidden)
    if logits.size(-1) > vocab_size:
        logits = logits[..., :vocab_size]
    logits = logits.float()
    return LOGCAP_SOFTMAX * torch.tanh(logits / LOGCAP_SOFTMAX)


__all__ = ["LOGCAP_SOFTMAX", "denoiser_logits"]
