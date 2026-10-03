"""
Public facade for the DiffusionBlocks EDM denoising-loss kernel.

The loss `weight * mean((pred - clean)^2)` is the exact objective every
`denoise_step` optimizes, on three backends: a pure-PyTorch naive reference
(testing oracle), a compiled C++ op (`cpu`), and a Taichi/Vulkan kernel
(`gpu`). Compiled backends load lazily on first use and raise rather than
falling back.

The compiled forward op is inference-only (autograd raises and points here):
training gradients reach `pred`/`clean` through `DbDenoiseLossFunction`'s
analytic backward (`dL/dp = 2w(p-c)/N`), never by differentiating through the
kernel -- the same split the SparseProp training path uses, where the layer's
backward routes through the kernels from inside its own Function.
"""

import torch

from nanochat.ops.kernels.registration import register_inference_only_op
from nanochat.ops.references.db_denoise_reference import validate_db_denoise_inputs

_fake_registered = False


def _ensure_cpu_op() -> None:
    """Build/import the C++ extension once and register the FakeTensor kernel."""
    global _fake_registered
    from nanochat.ops.kernels.cpu_loader import load_cpu_db_denoise_extension

    load_cpu_db_denoise_extension()
    if not _fake_registered:

        def _lcqat_db_denoise_fake(pred, clean, weight):
            return torch.empty((), dtype=torch.float32, device=pred.device)

        register_inference_only_op(
            "nanochat::lcqat_db_denoise",
            _lcqat_db_denoise_fake,
            "nanochat::lcqat_db_denoise is the compiled EDM-loss forward and "
            "defines no gradient of its own; training runs it inside "
            "DbDenoiseLossFunction, whose analytic backward carries the "
            "gradient to the denoiser prediction",
        )
        _fake_registered = True


def db_denoise_cpu(
    pred: torch.Tensor, clean: torch.Tensor, weight: float
) -> torch.Tensor:
    """Run the compiled C++ EDM loss on CPU."""
    validate_db_denoise_inputs(pred, clean, weight)
    _ensure_cpu_op()
    return torch.ops.nanochat.lcqat_db_denoise(
        pred.contiguous(), clean.contiguous(), float(weight)
    )


def db_denoise_gpu(
    pred: torch.Tensor, clean: torch.Tensor, weight: float
) -> torch.Tensor:
    """Run the Taichi/Vulkan EDM loss (loaded lazily)."""
    validate_db_denoise_inputs(pred, clean, weight)
    from nanochat.ops.kernels.gpu_loader import run_db_denoise

    return run_db_denoise(pred.contiguous(), clean.contiguous(), float(weight))


class DbDenoiseLossFunction(torch.autograd.Function):
    """EDM loss whose forward dispatches to a backend kernel.

    Forward computes the scalar on the requested backend; backward is the
    analytic gradient of `w * mean((p - c)^2)` in pure PyTorch, so the loss
    stays differentiable no matter which backend computed it. `weight` and
    `backend` are plain (non-tensor) arguments and take no gradient.
    """

    @staticmethod
    def forward(ctx, pred, clean, weight: float, backend: str):
        from nanochat.ops.dispatch import dispatch_db_denoise

        ctx.weight = float(weight)
        ctx.numel = pred.numel()
        ctx.save_for_backward(pred.detach(), clean.detach())
        with torch.no_grad():
            return dispatch_db_denoise(
                pred.detach(), clean.detach(), float(weight), backend
            )

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.weight
        n = ctx.numel
        pred, clean = ctx.saved_tensors
        diff = pred - clean
        scale = grad_output * (2.0 * weight / n)
        # `clean` is the detached target: it takes no gradient.
        return scale * diff, None, None, None


def db_denoise_loss(
    pred: torch.Tensor, clean: torch.Tensor, weight: float, backend: str = "cpu"
) -> torch.Tensor:
    """Differentiable EDM loss on the requested backend (default: compiled CPU)."""
    return DbDenoiseLossFunction.apply(pred, clean, float(weight), backend)
