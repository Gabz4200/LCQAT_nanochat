import torch
import torch.nn as nn
import torch.nn.functional as F

#: Which front stage feeds the learned lookup in `QuantizedLinear`.
#: `codebook` = none (feed the shadow parameter straight through, the original
#: behaviour). `classic` = uniform fake-quant first, so the lookup trains
#: against an inference-like input distribution.
LUT_FRONT_CODEBOOK = "codebook"
LUT_FRONT_CLASSIC = "classic"
LUT_FRONTS = (LUT_FRONT_CODEBOOK, LUT_FRONT_CLASSIC)


# Ideal for activations
class SmoothPWL(nn.Module):
    def __init__(
        self,
        knots: int = 10,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.index = nn.Parameter(torch.linspace(-1.0, 1.0, knots))
        self.weights = nn.Parameter(torch.ones(knots))
        self.bias = nn.Parameter(torch.ones(knots))

    def forward(self, x: torch.Tensor):
        # Broadcast x against knots across any shape:
        # x: (*shape) -> x.unsqueeze(-1): (*shape, 1)
        # self.index: (knots,) -> broadcasts to (*shape, knots)
        diff = x.unsqueeze(-1) - self.index
        inverse_squared_distances = 1.0 / (diff.pow(2) + self.eps)

        # Softmax over the knot dimension (last dimension)
        weights_softmax = F.softmax(inverse_squared_distances, dim=-1)

        # Compute convex combination weights and biases
        used_weight = (self.weights * weights_softmax).sum(dim=-1)
        used_bias = (self.bias * weights_softmax).sum(dim=-1)

        # Apply transformation element-wise, preserving original shape of x
        return used_weight * x + used_bias


# Ideal for quantization
class LearnedLookup(nn.Module):
    """Proximity lookup: select an output level by distance to a knot grid.

    The knots are stored FP32 and exposed in fp8 only through
    `index_fp8()`. Two reasons, both forced rather than stylistic:

    * `torch.linspace` has no fp8 CPU kernel, so constructing the grid directly
      in fp8 raises `NotImplementedError`. The grid is therefore built in FP32
      and cast, which is also the only order that works on any device.
    * An fp8 `nn.Parameter` has no usable gradient. Training needs the FP32
      shadow; fp8 is the *export* saving, and `nanochat/models/quant/learnable_lut.py`
      reaches the same split through `fp8_knots()`.
    """

    def __init__(
        self,
        entries: int = 15,
        eps: float = 1e-6,
        index_dtype: torch.dtype = torch.float8_e4m3fn,
        output_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.eps = eps
        # Storage dtype the knots are *exported* in. The parameter stays FP32 so
        # it can train; see the class docstring.
        self.index_dtype = index_dtype
        self.index = nn.Parameter(torch.linspace(-1.0, 1.0, entries))
        self.outputs = nn.Parameter(
            torch.linspace(-1.0, 1.0, entries, dtype=output_dtype)
        )

    def index_fp8(self) -> torch.Tensor:
        """The knots in their export dtype, for packing into the artifact."""
        return self.index.detach().to(self.index_dtype)

    def forward(self, x: torch.Tensor):
        # Broadcast x against knots across any shape:
        # x: (*shape) -> x.unsqueeze(-1): (*shape, 1)
        # self.index: (knots,) -> broadcasts to (*shape, knots)
        diff = x.unsqueeze(-1) - self.index
        inverse_squared_distances = 1.0 / (diff.pow(2) + self.eps)

        # Softmax over the knot dimension (last dimension)
        weights_softmax = F.softmax(inverse_squared_distances, dim=-1)

        # Compute convex combination outputs
        used_output = (self.outputs * weights_softmax).sum(dim=-1)

        # Apply transformation element-wise, preserving original shape of x
        return used_output


class ClassicUniformFakeQuant(nn.Module):
    """Classic uniform fake-quant: the standard QAT front stage.

    Symmetric, per-tensor, with a fixed scale derived from the observed range:

        scale = max(|x|) / ((K - 1) / 2)
        q     = clamp(round(x / scale), -(K - 1) // 2, (K - 1) // 2)
        out   = q * scale            (with identity STE through round)

    This is deliberately *not* LC-QAT. LC-QAT's codebook is learnable, non-uniform
    and asymmetric, with the exact-zero anchor that SparseProp depends on; the
    point of this stage is to be the thing LC-QAT is being compared against --
    a rigid uniform grid -- so the learned LUT behind it sees inputs that look
    like the ones it will see at inference.

    The scale is fixed rather than learned (no LSQ). A learnable scale would be a
    second grid to tune and would confound the comparison: the question this
    stage answers is "does a LUT correction help after uniform quantization",
    and a learned scale would answer a different question.

    **Exact zero is automatic.** `round(0) == 0` and `0 * scale == 0.0` for any
    `K`, so the SparseProp structural-zero contract holds here without a pin --
    this is the one place the zero survives for free. The RBF correction *after*
    it still needs its own knot-and-pin, because a convex combination of learned
    levels does not preserve zero (measured -4.47e-08 without the pin).

    **Only the front is STE.** The correction behind it is smooth, so the stack
    is shadow -> fake-quant (STE) -> smooth correction -> `F.linear`. Two STEs in
    series would be a step-function-of-a-step-function and train nothing useful;
    one STE followed by a smooth map gives the shadow weight a real Jacobian.

    The parameters stay FP32. Building them in the target dtype instead (int,
    fp8) would kill the gradient outright: an integer `nn.Parameter` has no
    usable `.grad`, and fp8 optimizer state diverges. The target dtype belongs
    at export, not in the training graph.
    """

    def __init__(self, entries: int = 255, eps: float = 1e-8):
        super().__init__()
        if entries < 3:
            raise ValueError(f"entries must be >= 3, got {entries}")
        self.entries = int(entries)
        # Odd entry count so the grid is symmetric around exact zero.
        self.levels = (self.entries - 1) // 2
        self.eps = float(eps)

    @torch.no_grad()
    def scale_for(self, x: torch.Tensor) -> torch.Tensor:
        """The fixed per-tensor scale for a tensor.

        `max(|x|)` with a zero fallback, so an all-zero tensor (every row of a
        zero-initialised `c_proj`) does not divide by zero. The floor is `eps`
        rather than 1.0 because a degenerate scale would otherwise quantize the
        whole tensor onto one level.
        """
        amax = x.detach().abs().max()
        return (amax / max(self.levels, 1)).clamp_min(self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize `x`, passing the gradient straight through."""
        scale = self.scale_for(x)
        q = torch.clamp(torch.round(x / scale), -float(self.levels), float(self.levels))
        x_q = q * scale
        # STE identity: `x_q` is a step function of `x`, so the gradient flows
        # to the shadow weight unattenuated.
        return x_q + (x - x.detach())


class QuantizedLinear(nn.Module):
    """Uniform-quantize first, then correct with the learned lookup.

    The front stage exists because of a train/inference mismatch that the plain
    version had. Feeding `self.weight` straight into `LearnedLookup` means the
    lookup sees continuous weights during training but a *quantized* weight
    distribution at inference, so the correction learns to compensate for an
    input shape it will never be given. With `lut_front="classic"` the weight
    crosses a uniform grid first and the lookup sees what it will see.

    `lut_front="codebook"` (the default) skips the front stage entirely and
    reproduces the original behaviour, so the comparison is available without
    changing any existing result. Nothing is replaced either way --
    `LearnedLookup` is still the quantizer/corrector, it just gets a more
    inference-like input.

    Both `LearnedLookup`s exist in both modes, so this is a genuine A/B over the
    input distribution and not a change of module.

    Args:
        lut_front: `"classic"` for the uniform front stage, `"codebook"` for none.
        quantize_bias: whether to run the bias through the same two stages. The
            shipped `LCQATLinear` consumes bias in FP32, so this is the one
            genuine capability gap here; it stays opt-in and unmeasured.
        entries: uniform grid size. 255 is the 8-bit index alphabet LC-QAT
            already packs for; 15 is the shipped default. Kept as a parameter so
            the comparison is a measurement rather than an assumption.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-6,
        input_dtype: torch.dtype = torch.float8_e4m3fn,
        output_dtype: torch.dtype = torch.float32,
        lut_front: str = LUT_FRONT_CODEBOOK,
        quantize_bias: bool = False,
        entries: int = 256,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.lut_front = lut_front
        self.quantize_bias_enabled = bool(quantize_bias)
        if lut_front not in (LUT_FRONT_CODEBOOK, LUT_FRONT_CLASSIC):
            raise ValueError(
                f"lut_front must be one of "
                f"{(LUT_FRONT_CODEBOOK, LUT_FRONT_CLASSIC)}, got {lut_front!r}"
            )

        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias: nn.Parameter | None = (
            nn.Parameter(torch.randn(out_features)) if bias else None
        )

        self.quantize_weight = LearnedLookup(
            entries=entries, eps=eps, index_dtype=input_dtype, output_dtype=output_dtype
        )
        self.quantize_bias = (
            LearnedLookup(
                entries=entries,
                eps=eps,
                index_dtype=input_dtype,
                output_dtype=output_dtype,
            )
            if bias
            else None
        )
        # Present only in `classic` mode, and parameter-free: the scale is read
        # off the tensor each call rather than stored, so there is no second
        # learnable grid to tune.
        self.fake_quant = (
            ClassicUniformFakeQuant(entries=entries)
            if lut_front == LUT_FRONT_CLASSIC
            else None
        )

    def _correct(self, param: torch.Tensor, lookup: "LearnedLookup") -> torch.Tensor:
        """Run one parameter through the front stage (if any) then the lookup."""
        if self.fake_quant is not None:
            param = self.fake_quant(param)
        return lookup(param)

    def forward(self, x: torch.Tensor):
        quantized_weight = self._correct(self.weight, self.quantize_weight)
        quantized_bias = None
        if self.bias is not None and self.quantize_bias_enabled:
            quantized_bias = self._correct(self.bias, self.quantize_bias)
        # Perform linear transformation with quantized weights and biases
        return F.linear(x, quantized_weight, quantized_bias)


class TestMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = QuantizedLinear(input_dim, hidden_dim)
        self.activation = SmoothPWL(knots=10)
        self.fc2 = QuantizedLinear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
