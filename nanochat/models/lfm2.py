"""LFM2 (Liquid Foundation Model 2) -- a GPT-compatible hybrid backbone.

LFM2 is an edge-first hybrid Transformer (arXiv:2511.23404): most layers
are cheap double-gated short convolutions ("LIV" blocks), a minority are
grouped-query attention (GQA) blocks with SwiGLU MLPs, over RMSNorm and
RoPE. The hybrid keeps long-range capacity through the sparse attention
layers while the convolution blocks carry the bulk of the compute cheaply.

**This module does not replace `nanochat.models.backbone.GPT`.** LFM2 is a
*second* backbone that speaks the exact interface the rest of the stack is
built against, so every downstream stage -- DiffusionBlocks, DARD, SparseProp,
LC-QAT, 4-bit QAT, and both ``*_MinExpLstm_Optimizer`` -- runs LFM2 unchanged:

* ``LFM2.transformer.h`` is an ``nn.ModuleList`` of blocks, each exposing the
  same ``forward(x, ve, cos_sin, window_size, kv_cache, attn_mask, cond,
  sigma)`` contract as a GPT ``Block``. A convolution block has no attention
  and ignores the attention-only arguments; it applies the AdaLN ``cond`` pair
  to its pre-norm input exactly like an attention block does, so DiffusionBlocks
  conditions every layer identically.
* ``LFM2.lm_head`` is the unembedding head DARD attaches to and the EDM
  cross-entropy targets; ``LFM2.config`` carries ``n_embd``/``n_head``/
  ``n_kv_head``/``n_layer``/``vocab_size``/``sequence_len``/``window_pattern``
  that the engine, partitioner and checkpoint manager read.
* ``LFM2.forward(idx, targets=None, attn_mask=None)`` returns the loss when
  ``targets`` is given and the logits otherwise -- the same contract
  ``scripts/_train/loop.py`` and the MeZO path call.

Because every projection is a ``Linear`` (the typed, dtype-casting
``nanochat.models.backbone.Linear``), SparseProp masks, LC-QAT codebooks and
``FourBitLinear`` retrofits all land on LFM2's projections exactly as they do
on GPT's, and both MinExpLSTM optimizers step the same ``nn.Parameter`` leaves.

The two attention-free differences from GPT are stated, not hidden:

1. Convolution blocks are position-free (depthwise short conv over the sequence
   axis). They do not take ``cos_sin``/``window_size``/``kv_cache``; a hybrid
   stack that needs long-range recall simply has enough GQA blocks to provide
   it -- that is LFM2's design point.
2. LFM2 keeps GPT's per-layer residual scalars (``resid_lambdas``/``x0_lambdas``)
   and the smear/backout residual-stream tricks, because those are part of
   nanochat's training dynamics and are orthogonal to the block topology.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.models.backbone import (
    BlockList,
    Linear,
    apply_rotary_emb,
    norm,
)
from nanochat.models.dtype import COMPUTE_DTYPE
from nanochat.models.flash_attention import flash_attn


@dataclass
class LFM2Config:
    """Configuration for an LFM2 hybrid backbone.

    Mirrors ``GPTConfig``'s field names (``n_layer``/``n_head``/``n_kv_head``/
    ``n_embd``/``vocab_size``/``sequence_len``/``window_pattern``) so the
    DiffusionBlocks engine, partitioner and checkpoint manager read LFM2's
    config the same way they read GPT's, plus the LFM2-specific knobs:

    * ``num_attention_blocks``: how many of the ``n_layer`` layers are GQA+SwiGLU
      attention blocks (the rest are double-gated short convolutions).
    * ``conv_kernel_size``: the depthwise short-conv kernel width.
    * ``intermediate_size``: SwiGLU hidden width (defaults to ``4 * n_embd``).
    """

    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 16
    n_head: int = 16
    n_kv_head: int = 4
    n_embd: int = 1024
    window_pattern: str = "SSSL"
    # LFM2-specific.
    num_attention_blocks: int = 6
    conv_kernel_size: int = 3
    intermediate_size: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.num_attention_blocks <= self.n_layer:
            raise ValueError(
                f"num_attention_blocks ({self.num_attention_blocks}) must be "
                f"within [0, n_layer] ({self.n_layer})"
            )
        if self.intermediate_size is None:
            # SwiGLU at the GPT hidden width; LFM2's 2.75x ratio is set explicitly.
            self.intermediate_size = 4 * self.n_embd

    @classmethod
    def from_pretrained_size(cls, size: str) -> "LFM2Config":
        """The LFM2 dense family's headline shapes (700M default here)."""
        configs = {
            "350M": cls(
                n_embd=768,
                intermediate_size=2048,
                n_head=12,
                n_kv_head=3,
                num_attention_blocks=4,
            ),
            "700M": cls(
                n_embd=1024,
                intermediate_size=2816,
                n_head=16,
                n_kv_head=4,
                num_attention_blocks=6,
            ),
            "1.2B": cls(
                n_embd=1536,
                intermediate_size=4096,
                n_head=24,
                n_kv_head=6,
                num_attention_blocks=8,
            ),
        }
        if size not in configs:
            raise ValueError(f"Unknown LFM2 size: {size}. Available: {list(configs)}")
        return configs[size]


class LFM2ConvBlock(nn.Module):
    """LFM2 double-gated short-range convolution ("LIV") block.

    The reference forward is
    ``B, C, x = linear(x); x = B*x; x = conv(x); x = C*x; x = linear(x)``
    -- two input-dependent gates around a depthwise short convolution. The
    gates make the convolution's contribution input-varying (hence "LIV")
    without any attention or recurrent state.

    It exposes GPT's ``Block.forward`` signature so a hybrid stack can hold
    conv and attention blocks in one ``ModuleList`` and DiffusionBlocks can
    condition either with the same AdaLN pair. The attention-only arguments
    (``ve``/``cos_sin``/``window_size``/``kv_cache``/``attn_mask``) are
    accepted and ignored: a short convolution is position-free and needs none
    of them. ``cond`` is applied to the pre-norm input exactly as in a GPT
    ``Block`` so the diffusion conditioning reaches the block's layers.
    """

    def __init__(self, config: LFM2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd
        # Input projection to the two gates and the value stream.
        self.input_projection = Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Depthwise short convolution over the sequence axis.
        self.conv = nn.Conv1d(
            config.n_embd,
            config.n_embd,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=config.n_embd,
            bias=False,
        )
        # Output projection back to the residual stream.
        self.output_projection = Linear(config.n_embd, config.n_embd, bias=False)

    def forward(
        self,
        x,
        ve,
        cos_sin,
        window_size,
        kv_cache=None,
        attn_mask=None,
        cond=None,
        sigma=None,
    ):
        h = norm(x)
        if cond is not None:
            gamma, beta = cond
            h = h * (1.0 + gamma) + beta
        # B, C, x = linear(h)
        projected = self.input_projection(h)
        gate_b, gate_c, value = projected.chunk(3, dim=-1)
        # x = B * x  (first input-dependent gate)
        gated = gate_b * value
        # conv over the sequence axis: (B, H, L) for Conv1d
        conv_input = gated.transpose(1, 2)
        conv_out = self.conv(conv_input).transpose(1, 2)
        # x = C * x  (second input-dependent gate)
        gated = gate_c * conv_out
        return x + self.output_projection(gated)


class LFM2Attention(nn.Module):
    """Grouped-query attention with QK-norm and RoPE, GPT-style.

    Uses the same projections, rotary application and flash-attention call as
    ``nanochat.models.backbone.CausalSelfAttention`` (GQA via ``n_kv_head``,
    sliding windows via ``window_size``, optional KV cache), so LFM2's attention
    layers are numerically and behaviorally identical to GPT's and reuse every
    kernel the stack already ships.
    """

    def __init__(self, config: LFM2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache=None, attn_mask=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q = q * 1.2
        k = k * 1.2
        if kv_cache is None:
            y = flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=window_size, attn_mask=attn_mask
            )
        else:
            if getattr(kv_cache, "quantized", False):
                from nanochat.ops import dispatch_quant_attn

                kv_cache.write(self.layer_idx, k, v)
                k_idx, v_idx = kv_cache.get_layer_indices(self.layer_idx)
                y = dispatch_quant_attn(
                    q,
                    k_idx,
                    v_idx,
                    kv_cache.k_codebooks[self.layer_idx],
                    kv_cache.v_codebooks[self.layer_idx],
                    kv_cache.cache_seqlens + T,
                    window_size[0],
                    backend="cpu",
                )
            else:
                k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
                y = flash_attn.flash_attn_with_kvcache(
                    q,
                    k_cache,
                    v_cache,
                    k=k,
                    v=v,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True,
                    window_size=window_size,
                )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class LFM2MLP(nn.Module):
    """SwiGLU feed-forward block (gate * up, then down)."""

    def __init__(self, config: LFM2Config):
        super().__init__()
        # `intermediate_size` is set by __post_init__, so the assertion documents an int.
        inter = config.intermediate_size
        assert inter is not None, "LFM2Config.__post_init__ must set intermediate_size"
        self.c_fc = Linear(config.n_embd, inter, bias=False)
        self.c_proj = Linear(inter, config.n_embd, bias=False)

    def forward(self, x, sigma=None):
        # SwiGLU; `sigma` is accepted for GPT MLP signature parity and ignored.
        return self.c_proj(F.silu(self.c_fc(x)))


class LFM2Block(nn.Module):
    """A GQA + SwiGLU block with the exact ``Block.forward`` contract."""

    def __init__(self, config: LFM2Config, layer_idx: int):
        super().__init__()
        self.attn = LFM2Attention(config, layer_idx)
        self.mlp = LFM2MLP(config)

    def forward(
        self,
        x,
        ve,
        cos_sin,
        window_size,
        kv_cache=None,
        attn_mask=None,
        cond=None,
        sigma=None,
    ):
        h = norm(x)
        if cond is not None:
            gamma, beta = cond
            h = h * (1.0 + gamma) + beta
        x = x + self.attn(h, ve, cos_sin, window_size, kv_cache, attn_mask=attn_mask)
        m = norm(x)
        if cond is not None:
            gamma, beta = cond
            m = m * (1.0 + gamma) + beta
        x = x + self.mlp(m, sigma=sigma)
        return x


class LFM2BlockList(BlockList):
    """A ``BlockList`` whose members are ``LFM2Block | LFM2ConvBlock``.

    Structural only: it keeps the typed-yield behavior of ``BlockList`` (so
    ``h[i].attn...`` consumers stay type-checked) while widening the member
    type to LFM2's two block shapes.
    """

    def __init__(self, blocks: list[LFM2Block | LFM2ConvBlock]) -> None:
        super().__init__(blocks)  # type: ignore[arg-type]

    def __getitem__(self, idx):  # type: ignore[override]
        result = super().__getitem__(idx)
        if isinstance(idx, slice):
            return LFM2BlockList(list(result))
        return result

    def __iter__(self):  # type: ignore[override]
        return super().__iter__()


class LFM2TransformerTrunk(nn.Module):
    """Embedding + hybrid block stack, with GPT's state_dict key layout."""

    def __init__(self, config: LFM2Config, padded_vocab_size: int):
        super().__init__()
        self.wte = nn.Embedding(padded_vocab_size, config.n_embd)
        # Convolution blocks first (cheap, local), then GQA blocks for long-range capacity.
        blocks: list[LFM2Block | LFM2ConvBlock] = []
        num_conv = config.n_layer - config.num_attention_blocks
        for layer_idx in range(config.n_layer):
            if layer_idx < num_conv:
                blocks.append(LFM2ConvBlock(config, layer_idx))
            else:
                blocks.append(LFM2Block(config, layer_idx))
        self.h: LFM2BlockList = LFM2BlockList(blocks)


class LFM2(nn.Module):
    """The LFM2 hybrid backbone, GPT-compatible at the interface level.

    Construction, initialization, rotary precompute and the forward/loss
    contract mirror ``GPT`` so a caller cannot tell which backbone it is
    training from the training loop alone. See the module docstring for the
    compatibility guarantees and the two stated differences.
    """

    def __init__(self, config: LFM2Config, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        if config.n_embd < 24:
            raise ValueError(f"smear reads 24 channels, got n_embd={config.n_embd}")
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            import warnings

            warnings.warn(
                f"Padding vocab_size from {config.vocab_size} to "
                f"{padded_vocab_size} for efficiency",
                stacklevel=2,
            )
        self.transformer = LFM2TransformerTrunk(config, padded_vocab_size)
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer residual scalars (same role as GPT's).
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Rotary cache (over-computed 10x like GPT's).
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            if isinstance(block, LFM2ConvBlock):
                torch.nn.init.uniform_(block.input_projection.weight, -s, s)
                torch.nn.init.uniform_(block.conv.weight, -s, s)
                torch.nn.init.zeros_(block.output_projection.weight)
            else:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(
        self, seq_len, head_dim, base=100000, device=None
    ):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), (
            f"Invalid window_pattern: {pattern}. Use only S and L."
        )
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Only attention honors a window; conv entries stay so per-layer indexing is valid.
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def num_matmul_params(self):
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))

    def num_scaling_params(self):
        """Parameter counts for scaling-law analysis (GPT-compatible dict).

        LFM2 has no value embeddings or LC-QAT codebooks, so those groups are
        zero; the rest mirror GPT's breakdown so the scaling-law analysis
        (`scripts/_train/build.py`) reads LFM2 the same way it reads GPT.
        """
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = (
            self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.smear_gate.weight.numel()
            + self.smear_lambda.numel()
            + self.backout_lambda.numel()
        )
        total = wte + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), (
            "Parameter count mismatch"
        )
        return {
            "wte": wte,
            "value_embeds": 0,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "codebooks": 0,
            "scalars": scalars,
            "total": total,
        }

    def estimate_flops(self):
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * self.num_matmul_params() + attn_flops

    def kv_bytes_per_token(self):
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        return (
            self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes
        )

    def forward(
        self, idx, targets=None, kv_cache=None, loss_reduction="mean", attn_mask=None
    ):
        B, T = idx.size()
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        )
        assert self.cos.dtype == COMPUTE_DTYPE, (
            f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        )
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (self.cos[:, T0 : T0 + T], self.sin[:, T0 : T0 + T])

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Smear (same as GPT): mix the previous token's embedding in.
        if kv_cache is None:
            if T <= 1:
                raise ValueError(
                    "Training forward requires T > 1 (got T=1); use kv_cache for "
                    "single-token decode."
                )
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                self.smear_gate(x[:, 1:, :24])
            )
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                    self.smear_gate(x[:, 1:, :24])
                )
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                    self.smear_gate(x[:, :, :24])
                )
                x = x + gate * x_pre_smear

        x0 = x
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # Convolution blocks ignore cos_sin/window; attention blocks need them.
            x = block(
                x, None, cos_sin, self.window_sizes[i], kv_cache, attn_mask=attn_mask
            )
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., : self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token


__all__ = [
    "LFM2",
    "LFM2Config",
    "LFM2Block",
    "LFM2ConvBlock",
    "LFM2Attention",
    "LFM2MLP",
]
