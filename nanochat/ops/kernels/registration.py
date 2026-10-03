"""Registration of the inference-only custom ops.

Every compiled op in this package is inference-only: it has a real forward and
no gradient. Rather than let autograd produce a silent wrong answer -- or, worse,
a `NotImplementedError` from deep inside a dispatch -- each op registers an
autograd stub that raises with the same message pointing at the STE path that
training actually uses.

The five facades (`gemv`, `index_linear`, `quant_attn`, `sparseprop`,
`sparse_index_linear`) all need this, and their copies had already drifted in
wording and in guard style. They share this one implementation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def register_inference_only_op(
    op_name: str,
    fake_fn: Callable,
    message: str | None = None,
) -> None:
    """Register `op_name`'s FakeTensor kernel and its always-raising autograd.

    Args:
        op_name: the ``nanochat::`` qualified operator name.
        fake_fn: the FakeTensor/meta kernel, with the op's real signature. It
            must return a correctly shaped and typed tensor so a compiled
            region traces without materializing data.
        message: overrides the default backward error. Only needed when the
            stock sentence would misdescribe the op.

    Callers keep their own module-level `_fake_registered` guard; this does not
    track registration state itself, because each op's extension load and its
    fake registration must both happen exactly once and the two live together
    in the facade's `_ensure_*_op`.
    """
    torch.library.register_fake(op_name, fake_fn)

    text = message or (
        f"{op_name} is inference-only and defines no gradient; "
        "training runs the STE path in F.linear instead"
    )

    def _backward(ctx, *grad_outputs):
        raise RuntimeError(text)

    torch.library.register_autograd(op_name, _backward)
