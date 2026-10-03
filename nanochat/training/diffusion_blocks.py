"""DiffusionBlocks-CPU: block-wise AR training + diffusion inference for constrained CPUs."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.models.backbone import maybe_sigma_call


def sinusoidal_noise_embedding(log_sigma: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / max(half - 1, 1))
    args = log_sigma.unsqueeze(-1) * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


class NoiseConditionedBlockAdapter(nn.Module):
    """Per-layer AdaLN conditioning for one block.

    Emits `n_layers` independent `(gamma, beta)` pairs -- one per layer of the
    block -- rather than a single pair for the whole group. The paper conditions
    *inside* the block (Step 3), so each layer responds to sigma differently; one
    modulation per block cannot express that, and applying a modulation to the
    block's input stream would also normalize the residual stream, discarding
    the residual identity that the paper's Euler-step interpretation depends on.

    `forward` returns a list of `(gamma, beta)` tuples, each shaped
    `(B, 1, n_embd)` and broadcast over the sequence axis, which is exactly what
    `Block.forward(cond=...)` consumes. The output layer is zero-initialized, so
    a fresh model has `gamma=0, beta=0` and the conditioning is an exact no-op.
    """

    def __init__(
        self,
        n_embd: int,
        cond_dim: int = 32,
        n_layers: int = 1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.n_layers = int(n_layers)
        self.n_embd = int(n_embd)
        # NOTE: do NOT keep a `self.out = self.mlp[-1]` alias here. That alias
        # shares the same nn.Linear Parameter object under two attribute paths
        # (`mlp.2` and `out`), so nn.Module registers it twice and the optimizer
        # emits "parameter group with duplicate parameters". The final linear is
        # reachable as self.mlp[-1]; nothing reads self.out.
        factory = {"device": device, "dtype": dtype}
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 4 * cond_dim, **factory),
            nn.SiLU(),
            nn.Linear(4 * cond_dim, 2 * self.n_layers * n_embd, **factory),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, x: torch.Tensor, sigma: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return one `(gamma, beta)` pair per layer, each `(B, 1, n_embd)`.

        The `nn.Sequential` is walked by hand rather than called, so a
        retrofitted (sigma-conditioned) Linear inside it can be given the noise
        level. `Sequential.forward` has no way to pass a per-layer keyword, and
        silently skipping sigma would make `--db-sigma-codebook` a no-op on
        these two layers while still reporting itself as enabled.
        """
        c = sinusoidal_noise_embedding(torch.log(sigma), self.cond_dim).to(x.dtype)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                c = maybe_sigma_call(layer, c, sigma)
            else:
                c = layer(c)
        c = c.reshape(c.size(0), self.n_layers, 2 * self.n_embd)
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_c in c.unbind(dim=1):
            gamma, beta = layer_c.chunk(2, dim=-1)
            out.append((gamma.unsqueeze(1), beta.unsqueeze(1)))
        return out


def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / 2.0**0.5))


def _standard_normal_icdf(q: torch.Tensor) -> torch.Tensor:
    return 2.0**0.5 * torch.erfinv(2.0 * q - 1.0)


class EquiProbabilityPartitioner:
    """Maps [sigma_min, sigma_max] into B equi-probability-mass intervals."""

    def __init__(
        self,
        num_blocks: int = 4,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        distribution: str = "log_normal",
    ) -> None:
        assert num_blocks >= 1
        assert 0.0 < sigma_min < sigma_max
        assert distribution == "log_normal"
        self.num_blocks = num_blocks
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std

    def _cdf(self, sigma: float) -> torch.Tensor:
        return _standard_normal_cdf(
            torch.tensor(
                (math.log(sigma) - self.p_mean) / self.p_std, dtype=torch.float64
            )
        )

    def sample_sigma(
        self,
        block_idx: int,
        generator: torch.Generator | None = None,
        overlap: float = 0.0,
    ) -> torch.Tensor:
        bounds = self.boundaries()
        lo, hi = bounds[block_idx].item(), bounds[block_idx + 1].item()
        alpha = (hi / lo) ** overlap
        q_lo = self._cdf(lo / alpha).clamp(1e-6, 1.0 - 1e-6)
        q_hi = self._cdf(hi * alpha).clamp(1e-6, 1.0 - 1e-6)
        u = q_lo + torch.rand((), generator=generator, dtype=torch.float64) * (
            q_hi - q_lo
        )
        return torch.exp(self.p_mean + self.p_std * _standard_normal_icdf(u)).float()

    def boundaries(self) -> torch.Tensor:
        lo = self._cdf(self.sigma_min)
        hi = self._cdf(self.sigma_max)
        frac = torch.linspace(0.0, 1.0, self.num_blocks + 1, dtype=torch.float64)
        q = lo + frac * (hi - lo)
        q = q.clamp(1e-6, 1.0 - 1e-6)
        sigmas = torch.exp(self.p_mean + self.p_std * _standard_normal_icdf(q))
        sigmas[0] = self.sigma_min
        sigmas[-1] = self.sigma_max
        return sigmas


def c_noise(sigma: torch.Tensor) -> torch.Tensor:
    return torch.log(sigma) / 4.0


def edm_preconditioning(
    sigma: torch.Tensor, sigma_data: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    var = sigma.square() + sigma_data**2
    c_in = 1.0 / var.sqrt()
    c_out = sigma * sigma_data / var.sqrt()
    w = var / (sigma * sigma_data).square()
    return c_in, c_out, w


def block_diagonal_mask(
    seq_lens: list[int] | torch.Tensor,
    seq_len: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Build a block-diagonal causal attention mask for packed sequences.

    Positions attend to earlier positions only if they are in the same sequence/document.
    Output shape: (1, 1, seq_len, seq_len) suitable for broadcast across (B, H, T, T).
    """
    if isinstance(seq_lens, torch.Tensor):
        lens_list = seq_lens.tolist()
    else:
        lens_list = list(seq_lens)
    assert sum(lens_list) <= seq_len, "Sum of sequence lengths exceeds seq_len"

    doc_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    curr = 0
    for doc_idx, length in enumerate(lens_list):
        doc_ids[curr : curr + length] = doc_idx
        curr += length
    if curr < seq_len:
        doc_ids[curr:] = len(lens_list)

    row = doc_ids.unsqueeze(1)
    col = doc_ids.unsqueeze(0)
    same_doc = row == col

    t_idx = torch.arange(seq_len, device=device)
    causal = t_idx.unsqueeze(1) >= t_idx.unsqueeze(0)

    mask = same_doc & causal
    mask = mask.unsqueeze(0).unsqueeze(0)
    if dtype == torch.bool:
        return mask
    out = torch.zeros_like(mask, dtype=dtype)
    out.masked_fill_(~mask, float("-inf"))
    return out


def _wrap_sparseprop(
    linear: nn.Linear, sparsity: float, with_lcqat: bool, in_place: bool = False
):
    """Wrap one engine-owned Linear with SparseProp sparse backward.

    The engine's layers have no nanochat role names, so they are converted
    directly rather than through `inject_sparseprop_layers`' name filter.

    `in_place=True` re-parents the quantizers instead of building a new module.
    That matters when the layer is an `LCQATLinear` that `apply_lcqat` already
    installed *inside* a live `nn.Sequential` slot: building a fresh wrapper
    would leave the original module registered under its old path, and every
    codebook parameter would then be reachable twice -- which the optimizer
    rejects, and which silently doubled the codebook count before the partition
    check existed. Re-parenting in place keeps one registration per parameter.
    """
    from nanochat.models.quant.linear import LCQATLinear
    from nanochat.models.quant.sparseprop import (
        SparsePropLinear,
        SparsePropLinearLCQAT,
    )

    if isinstance(linear, LCQATLinear):
        if not with_lcqat:
            # LC-QAT without SparseProp: leave the module alone. Forcing a
            # wrapper would drop the quantized inference path (see
            # SparsePropLinearLCQAT's docstring).
            return linear
        if in_place:
            _reparent_sparseprop_in_place(linear, sparsity)
            return linear
        return SparsePropLinearLCQAT.from_lcqat(linear, sparsity=sparsity)
    if in_place:
        _reparent_sparseprop_in_place(linear, sparsity)
        return linear
    return SparsePropLinear.from_linear(linear, sparsity=sparsity)


def _reparent_sparseprop_in_place(linear, sparsity: float) -> None:
    """Convert a Linear to sparse backward without replacing the module object.

    Mutates the module's class and installs the CSR/CSC buffers in place, so the
    object identity -- and therefore its position in the module tree, its
    parameter names, and its checkpoint keys -- is preserved.

    Substituting a `SparsePropLinearLCQAT` for an `LCQATLinear` that is still
    registered elsewhere in the tree would make every codebook parameter
    reachable under two paths, which AdamW rejects ("duplicate parameters") and
    which silently doubled the codebook parameter count before
    `verify_partition` caught it.
    """
    from nanochat.models.quant.linear import LCQATLinear
    from nanochat.models.quant.sparseprop import (
        SparsePropLinear,
        SparsePropLinearLCQAT,
    )

    # The class swap needs the SparsePropLinear.__init__ state, which we build
    # on a throwaway module and then adopt. Nothing is allocated twice: the
    # weight and quantizers are the *same objects*, shared with the scratch
    # module, which is discarded immediately.
    if isinstance(linear, LCQATLinear):
        scratch = SparsePropLinearLCQAT(linear, sparsity=sparsity)
        new_cls = SparsePropLinearLCQAT
    else:
        scratch = SparsePropLinear.from_linear(linear, sparsity=sparsity)
        new_cls = SparsePropLinear
    linear.__class__ = new_cls
    linear._buffers.update({k: v for k, v in scratch._buffers.items() if v is not None})
    linear._non_persistent_buffers_set = set(scratch._non_persistent_buffers_set)
    linear.sparsity = sparsity
    linear._build_sparse_structure()


def _retrofit_plain_linear(
    linear: nn.Linear,
    layer_config,
    roles: tuple[str, str],
    lcqat_linear_cls,
    spec_k,
    spec_split,
):
    """Retrofit one engine-owned `nn.Linear` to LC-QAT using explicit roles.

    `retrofit_model` resolves roles from nanochat module *names*
    (`attn.c_q`, `mlp.c_fc`, ...), which the engine's own layers do not have:
    the denoise heads and adapter MLPs are plain `nn.ModuleList` /
    `nn.Sequential` members with no such names. Passing the role field names
    directly reuses the config's level allocations without inventing a naming
    convention for them.
    """
    weight_spec = getattr(layer_config, roles[0])
    act_spec = getattr(layer_config, roles[1])
    return lcqat_linear_cls.from_float(
        linear,
        K_weight=spec_k(weight_spec),
        K_act=spec_k(act_spec),
        quantize_out=False,
        K_weight_split=spec_split(weight_spec),
        K_act_split=spec_split(act_spec),
        grad_scale=getattr(layer_config, "grad_scale", "inv_sqrt_n"),
    )


def _layer_groups(n_layer: int, num_blocks: int) -> list[list[int]]:
    """Split `n_layer` layer indices into `num_blocks` contiguous groups.

    The first `n_layer % num_blocks` groups get one extra layer, so the split is
    balanced and the groups stay contiguous (which the block-diagonal attention
    mask and the sequential sampler both rely on).
    """
    base, rem = divmod(n_layer, num_blocks)
    groups: list[list[int]] = []
    start = 0
    for b in range(num_blocks):
        size = base + (1 if b < rem else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


class DiffusionBlockEngine:
    """Block-isolated trainer and diffusion sampler for nanochat GPT.

    Converts transformer depth into B independent diffusion denoising blocks.
    Each block is trained on an equi-probability noise range via EDM denoising
    or local token prediction, isolating backward passes to L/B layers.
    During inference, sequential Euler steps from sigma_max to sigma_min
    progressively denoise token representations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        partitioner: EquiProbabilityPartitioner,
        dtype: torch.dtype = torch.float32,
        cond_dim: int = 32,
        device=None,
    ) -> None:
        self.model = model
        self.partitioner = partitioner
        self.dtype = dtype
        n_layer = len(model.transformer.h)
        assert partitioner.num_blocks <= n_layer
        n_embd = model.config.n_embd
        # `device` is load-bearing, not cosmetic: the engine's own parameters
        # live in the optimizer (parameters()), so landing them off-device means
        # their .grad stays None and AdamW skips them without a word.
        factory = {"device": device, "dtype": dtype}
        # Per-block layer counts must be known before the adapters are built,
        # since each adapter emits one (gamma, beta) per layer of its block.
        groups = _layer_groups(n_layer, partitioner.num_blocks)
        self.adapters = nn.ModuleList(
            [
                NoiseConditionedBlockAdapter(
                    n_embd, cond_dim, n_layers=len(groups[b]), **factory
                )
                for b in range(partitioner.num_blocks)
            ]
        )
        # One denoise head per block, not one shared head. Each block trains on a
        # disjoint equi-probability sigma-range whose activation distributions
        # differ, and under LC-QAT a single head carries a single quantized
        # K_act codebook -- one codebook cannot cover B disjoint distributions.
        # Per-block heads let each specialize to its own noise range.
        self.denoise_heads = nn.ModuleList(
            [
                nn.Linear(n_embd, n_embd, **factory)
                for _ in range(partitioner.num_blocks)
            ]
        )
        for head in self.denoise_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # Optional EfQAT freezer, consulted by `_requires_grad_for` rather than
        # mutating requires_grad behind its back.
        self.freezer = None
        # Optional float-twin KD anchor for the EDM objective (see
        # `set_distiller`). None means "no anchor", which is the default.
        self.distiller = None
        # The KD term from the most recent `denoise_step`, or 0.0 when no
        # distiller is installed. Exposed because the *returned* loss is a
        # convex mix of the data and anchor terms, so from it alone there is no
        # way to tell a working anchor from one that never fired.
        self.last_kd_loss = 0.0

    @property
    def _owned(self) -> tuple[nn.Module, ...]:
        """The subtrees this engine owns, in iteration order.

        The single source of truth for "what does this engine contain". Every
        whole-tree operation (`train`, `zero_grad`, `to`, `parameters`,
        `modules`, the SparseProp count, `cpu_adamw_for`) iterates exactly
        these, so adding a fourth owned subtree is a one-line change here
        rather than a silent omission in one of them -- which is precisely
        the failure the `modules()` docstring describes. `state_dict`,
        `load_state_dict` and `named_parameters` deliberately do NOT use it:
        their per-subtree key prefixes are a contract with the checkpoint and
        with `_requires_grad_for`'s block routing.
        """
        return (self.model, self.adapters, self.denoise_heads)

    def train(self, mode: bool = True) -> "DiffusionBlockEngine":
        for owned in self._owned:
            owned.train(mode)
        return self

    # aislop-ignore-next-line security/eval -- standard PyTorch nn.Module eval method
    def eval(self) -> "DiffusionBlockEngine":
        return self.train(False)

    @property
    def config(self):
        return getattr(self.model, "config", None)

    def get_device(self) -> torch.device:
        if hasattr(self.model, "get_device"):
            return self.model.get_device()
        return next(self.model.parameters()).device

    def zero_grad(self, set_to_none: bool = True) -> None:
        for owned in self._owned:
            owned.zero_grad(set_to_none=set_to_none)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs) -> "DiffusionBlockEngine":
        for owned in self._owned:
            owned.to(*args, **kwargs)
        return self

    def set_distiller(self, distiller) -> "DiffusionBlockEngine":
        """Attach a `DenoiserDistiller` (float-twin KD) to the EDM objective.

        Passing None disables denoiser distillation, which is the default: with
        no distiller `denoise_step` computes the EDM term and nothing else, byte
        for byte the objective it had before this flag existed.

        The distiller is duck-typed on `alpha` and `__call__(student_pred,
        teacher_pred_fn, weight=)`, which is the same contract `KDLoss` uses for
        the CE objective -- the engine does not need to know which anchor is
        installed, only that the returned scalar is a term it must add.
        """
        self.distiller = distiller
        return self

    def set_freezer(self, freezer) -> "DiffusionBlockEngine":
        """Attach a `SelectiveFreezer` (EfQAT) as the `requires_grad` arbiter.

        Passing None disables it. The engine consults `freezer.is_trainable(name)`
        before enabling a block, so a freeze survives subsequent steps.

        Any object exposing `is_trainable(name) -> bool` works, which includes
        `BlockLatchFreezer` (per-block permanent freeze). The duck-typed
        contract is the whole point: the engine must not need to know which
        freeze policy is installed, only that a name may or may not train.
        """
        self.freezer = freezer
        return self

    def freezer_metadata(self) -> dict | None:
        """Latch/freeze state for checkpoint metadata, or None if no freezer.

        Freezing decisions are not recoverable from the state_dict -- a frozen
        parameter looks exactly like a converged one -- so a resumed run would
        silently resume training a block the previous run had latched. The
        freezer owns its own serialization; the engine only forwards it.
        """
        if self.freezer is None:
            return None
        meta = getattr(self.freezer, "metadata", None)
        return meta() if callable(meta) else {}

    def apply_lcqat(self, layer_config) -> int:
        """Retrofit the engine's adapter MLPs + denoise heads to LC-QAT.

        The base transformer is retrofitted separately (via `retrofit_model` on
        `self.model`); this covers the engine-owned Linear layers so the whole
        training pipeline is LC-QAT. Returns the number of LCQATLinear modules
        created.

        `retrofit_model` matches on nanochat module-name suffixes
        (`attn.c_q`, `mlp.c_fc`, ...), which the adapter MLP's `nn.Sequential`
        does not use, so the engine-owned layers are retrofitted directly here
        rather than through the name-based role filter.
        """
        from nanochat.models.quant.linear import LCQATLinear
        from nanochat.models.quant.retrofit import spec_k, spec_split

        created = 0
        for i, head in enumerate(self.denoise_heads):
            self.denoise_heads[i] = _retrofit_plain_linear(
                head,
                layer_config,
                ("o_weight", "o_act"),
                LCQATLinear,
                spec_k,
                spec_split,
            )
            created += 1
        for i, adapter in enumerate(self.adapters):
            for j, linear in enumerate(adapter.mlp):
                if isinstance(linear, nn.Linear) and not isinstance(
                    linear, LCQATLinear
                ):
                    # The adapter MLP is (in, hidden, out): a c_fc-like expansion
                    # followed by a c_proj-like contraction onto 2*n_layers*n_embd.
                    roles = (
                        ("fc_weight", "fc_act")
                        if j == 0
                        else ("down_weight", "down_act")
                    )
                    adapter.mlp[j] = _retrofit_plain_linear(
                        linear, layer_config, roles, LCQATLinear, spec_k, spec_split
                    )
                    created += 1
        return created

    def apply_sparseprop(
        self,
        sparsity: float = 0.75,
        target_modules: list[str] | None = None,
        with_lcqat: bool = False,
    ) -> int:
        """Retrofit the engine's Linear modules with SparseProp sparse backward.

        Replaces nn.Linear modules in the base transformer and denoise heads
        with SparsePropLinear (drop-in, O(nnz) backward via AVX2 kernels).
        When ``with_lcqat=True``, existing LCQATLinear modules are wrapped as
        SparsePropLinearLCQAT so the codebook quantizers and sparse backprop
        coexist in the same module (LC-QAT + SparseProp). Module name grouping
        used by the DB-CPU partitioner is preserved because SparsePropLinear
        subclasses nanochat Linear. Returns the number of SparseProp modules
        created.
        """
        from nanochat.models.quant.sparseprop import (
            SparsePropLinear,
            SparsePropLinearLCQAT,
            inject_sparseprop_layers,
        )

        inject_sparseprop_layers(
            self.model,
            sparsity=sparsity,
            target_modules=target_modules,
            with_lcqat=with_lcqat,
        )
        for head in self.denoise_heads:
            _wrap_sparseprop(head, sparsity, with_lcqat, in_place=True)
        # Engine-owned layers are converted IN PLACE (class swap + buffer
        # adoption), never by substituting a new module. `apply_lcqat` put an
        # LCQATLinear into each `nn.Sequential` slot, so replacing that slot
        # would leave the original reachable at its old path and register every
        # codebook parameter twice -- which AdamW rejects and which silently
        # doubled the codebook count before `verify_partition` existed.
        for adapter in self.adapters:
            for linear in adapter.mlp:
                if isinstance(linear, SparsePropLinear):
                    continue
                if isinstance(linear, nn.Linear):
                    _wrap_sparseprop(linear, sparsity, with_lcqat, in_place=True)
        return sum(
            1
            for sub in self.modules()
            if isinstance(sub, (SparsePropLinear, SparsePropLinearLCQAT))
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Export full engine state (base model + adapters + denoise heads).

        Callers must use this, not `self.model.state_dict()`: the engine owns
        `db_adapters.*` and `db_denoise_heads.*` on top of the bare GPT, and a
        checkpoint that declares `meta["db"]` without them reloads a silently
        zero engine.
        """
        sd = self.model.state_dict()
        for k, v in self.adapters.state_dict().items():
            sd[f"db_adapters.{k}"] = v
        for k, v in self.denoise_heads.state_dict().items():
            sd[f"db_denoise_heads.{k}"] = v
        return sd

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = False
    ) -> None:
        """Load engine state from a unified checkpoint.

        Migrates a legacy single `db_denoise_head.*` block onto denoise head 0.
        A legacy checkpoint genuinely cannot be split across B blocks (there was
        only ever one head), so that is a loud error rather than a silent
        partial load.
        """
        model_sd = {}
        adapters_sd = {}
        heads_sd = {}
        legacy_head_sd = {}
        for k, v in state_dict.items():
            if k.startswith("db_adapters."):
                adapters_sd[k[len("db_adapters.") :]] = v
            elif k.startswith("db_denoise_heads."):
                heads_sd[k[len("db_denoise_heads.") :]] = v
            elif k.startswith("db_denoise_head."):
                legacy_head_sd[k[len("db_denoise_head.") :]] = v
            else:
                model_sd[k] = v
        if legacy_head_sd and self.partitioner.num_blocks > 1:
            raise RuntimeError(
                "checkpoint carries a legacy single `db_denoise_head.*` block, "
                f"but this engine has {self.partitioner.num_blocks} blocks and "
                "therefore {self.partitioner.num_blocks} denoise heads. A legacy "
                "checkpoint had only one head and cannot be split; retrain, or "
                "load it into a 1-block engine."
            )
        if legacy_head_sd:
            heads_sd.update({f"0.{k}": v for k, v in legacy_head_sd.items()})
        self.model.load_state_dict(model_sd, strict=strict)
        if adapters_sd:
            self.adapters.load_state_dict(adapters_sd, strict=strict)
        if heads_sd:
            self.denoise_heads.load_state_dict(heads_sd, strict=strict)

    def named_parameters(self, recurse: bool = True, remove_duplicate: bool = False):
        """Yield (name, param) for model + adapters + denoise heads.

        `build_qat_param_groups` and `SelectiveFreezer` need a single
        `named_parameters()` view over the whole engine tree, so we forward
        into the three owned subtrees with the `db_` prefix for adapter/head
        params (matching the `state_dict` key prefix). The prefixes are also what
        `_requires_grad_for` parses to route gradients to one block's adapter
        and head, so they are a contract, not cosmetics.
        """
        for name, p in self.model.named_parameters("", recurse, remove_duplicate):
            yield name, p
        # named_parameters(prefix) already inserts the prefix, so do not add it
        # again -- doing so produced `db_adapters.db_adapters.1....`, which broke
        # both the block routing in `_requires_grad_for` and any consumer that
        # parses these names.
        for name, p in self.adapters.named_parameters(
            "db_adapters", recurse, remove_duplicate
        ):
            yield name, p
        for name, p in self.denoise_heads.named_parameters(
            "db_denoise_heads", recurse, remove_duplicate
        ):
            yield name, p

    def parameters(self) -> list[torch.nn.Parameter]:
        """Model + adapter + denoise head params for the optimizer."""
        return [p for owned in self._owned for p in owned.parameters()]

    def modules(self):
        """Model + adapter + denoise head modules, yielding the three subtrees.

        The engine is not an `nn.Module` (it owns three of them), so it cannot
        inherit `modules()`. Every consumer that walks the engine tree -- the
        SparseProp pruning schedule, `apply_sparseprop`'s own count -- needs this
        view, and duplicating the three-way iteration at each call site is how
        the set of owned subtrees drifts.
        """
        for owned in self._owned:
            yield from owned.modules()

    def block_layers(self) -> list[list[int]]:
        return _layer_groups(len(self.model.transformer.h), self.partitioner.num_blocks)

    def _requires_grad_for(self, name: str, block_idx: int) -> bool:
        """Single arbiter for `requires_grad`, per parameter name.

        Owns three rules that used to be spread across `_activate_block` and
        `denoise_step` (which had already drifted apart):

        1. Only the active block's transformer layers get gradients.
        2. Everything outside `transformer.h` (embeddings, lm_head, scalars) is
           shared across blocks, so it always trains.
        3. A freezer (EfQAT `SelectiveFreezer`, or `BlockLatchFreezer` for the
           per-block permanent latch) can veto either. Consulted *before*
           enabling rather than after, which is what makes freezing stick: the
           old order had `_activate_block` re-enable everything on the next
           micro-step, silently undoing EfQAT. It is also what makes a
           permanent latch permanent -- the veto runs on every activation, for
           every sampled block, so no later `_activate_block` can revive it.
        """
        if self.freezer is not None and not self.freezer.is_trainable(name):
            return False
        if name.startswith("transformer.h."):
            layer_idx = name.split(".")[2]
            return layer_idx.isdigit() and int(layer_idx) in set(
                self.block_layers()[block_idx]
            )
        # One adapter and one denoise head per block: only the active block's own
        # may train, since each is specialized to a single noise range. The block
        # index is the ModuleList position right after the prefix, e.g.
        # `db_denoise_heads.1.weight` -> parts[1] == "1".
        for prefix in ("db_adapters.", "db_denoise_heads."):
            if name.startswith(prefix):
                parts = name.split(".")
                return len(parts) > 1 and parts[1] == str(block_idx)
        return True

    def _apply_requires_grad(self, block_idx: int) -> None:
        """Set `requires_grad` across the whole engine for one active block."""
        for name, p in self.named_parameters():
            wanted = self._requires_grad_for(name, block_idx)
            p.requires_grad_(wanted)
            if not wanted:
                p.grad = None

    def live_blocks(self) -> list[int]:
        """Block indices that can still train.

        A block latched by `BlockLatchFreezer` is retired: every one of its
        parameters has `requires_grad=False`, so a forward through it produces a
        loss with no `grad_fn` at all (the clean-embedding input is detached by
        the EDM objective). Sampling it would not merely waste compute, it would
        crash the backward. So the sampler draws only from live blocks.

        Any other freezer (the middle-band `SelectiveFreezer`) freezes a subset
        of names rather than whole blocks, so all blocks remain live.
        """
        if self.freezer is None:
            return list(range(self.partitioner.num_blocks))
        frozen = getattr(self.freezer, "is_latched", None)
        if not callable(frozen):
            return list(range(self.partitioner.num_blocks))
        return [b for b in range(self.partitioner.num_blocks) if not frozen(b)]

    def sample_block(self, generator: torch.Generator | None = None) -> int:
        """Draw a block index for one optimizer step.

        Sampled once per *step*, not once per micro-step: accumulating gradients
        from several blocks into one update turns block-wise training into
        ordinary gradient accumulation, which preserves the memory saving but
        destroys the noise-range specialization the method rests on.

        Drawn uniformly over `live_blocks()`, so a permanently latched block is
        never resampled. Raises when every block has been latched: that means
        training is over, and continuing would silently do nothing.
        """
        live = self.live_blocks()
        if not live:
            raise RuntimeError(
                "every diffusion block has been permanently latched by the "
                "EfQAT freezer; there is nothing left to train. Stop the run."
            )
        return live[int(torch.randint(len(live), (1,), generator=generator).item())]

    def _activate_block(self, block_idx: int | None = None) -> int:
        """Enable gradients for one block and return the block index used."""
        b = self.sample_block() if block_idx is None else int(block_idx)
        self._apply_requires_grad(b)
        self.train()
        return b

    def train_step(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        block_idx: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full-depth next-token cross-entropy through the whole model.

        This is the *escape hatch* objective, not the DiffusionBlocks one: the
        forward runs all L layers, so only the backward is block-isolated. The
        EDM objective in `denoise_step` is the real one.
        """
        self._activate_block(block_idx)
        return self.model(idx, targets, attn_mask=attn_mask)

    def logprobs(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        block_idx: int | None = None,
    ) -> torch.Tensor:
        """Compute per-token cross-entropy loss without reduction, for RL / GRPO."""
        self._activate_block(block_idx)
        loss = self.model(idx, targets, loss_reduction="none")
        if loss.dim() == 1 and loss.numel() == idx.numel():
            loss = loss.view_as(idx)
        return loss

    def _run_block_denoiser(
        self,
        block_idx: int,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        seq_len: int,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run block `block_idx` on `noisy` and return its denoiser prediction.

        Shared by the student and by the KD teacher twin so both see byte-identical
        arithmetic: same layers, same per-layer AdaLN conditioning, same rotary
        tables, same head. The teacher must differ from the student *only* in
        its weights, so any per-call setup that lived in `denoise_step` would
        have to be duplicated exactly -- and a drift between the two copies would
        show up as a KD term that is nonzero even at alpha=1 with identical
        weights, which is indistinguishable from a real quantization gap.

        `seq_len` is passed rather than read from `noisy` because the rotary
        tables are indexed by it and the teacher is given the same tensor.
        """
        groups = self.block_layers()
        head_dim = self.model.config.n_embd // self.model.config.n_head
        cos, sin = self.model._precompute_rotary_embeddings(seq_len, head_dim)
        # Per-layer AdaLN: each layer of the block gets its own (gamma, beta)
        # rather than one modulation for the whole group. The paper conditions
        # inside the block (Step 3), and a single block-level modulation leaves
        # the deeper layers of a block unable to specialize within its noise
        # range.
        conds = self.adapters[block_idx](noisy, sigma.view(1))
        # Sigma-conditioned LC-QAT codebooks resolve against the *batch*
        # dimension, so the engine's `(B,)` sigma is reshaped to `(B, 1, 1)`
        # here. Passed on every step rather than only when conditioning is on:
        # the block's own `sigma` parameter defaults to `None` and the float
        # layers ignore it, so the cost is one kwarg and the alternative is two
        # near-identical loops that must be kept in sync.
        sigma_b = sigma.view(-1, 1, 1)
        h = noisy
        for i, layer in enumerate(groups[block_idx]):
            h = self.model.transformer.h[layer](
                h,
                None,
                (cos, sin),
                (seq_len, 0),
                None,
                attn_mask=attn_mask,
                cond=conds[i],
                sigma=sigma_b,
            )
        # `sigma` is forwarded for the same reason as in the block loop: a
        # sigma-conditioned head needs it, and a static one ignores it. The
        # duck-typed helper keeps a plain `nn.Linear` head working unchanged.
        return maybe_sigma_call(self.denoise_heads[block_idx], h, sigma_b)

    def denoise_step(
        self,
        idx: torch.Tensor,
        block_idx: int | None = None,
        overlap: float = 0.0,
        generator: torch.Generator | None = None,
        attn_mask: torch.Tensor | None = None,
        clean: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EDM score-matching step for one block (the real DiffusionBlocks path).

        Only block `b`'s layers execute, which is where the B-fold activation
        memory reduction comes from: gradients exist for L/B layers, not L.

        Args:
            idx: token ids (B, T). Used to build the diffusion target when
                `clean` is not supplied.
            block_idx: block to train. Defaults to a fresh sample; pass an
                explicit index (sampled once per optimizer step) to keep every
                micro-step in a step on the same block.
            overlap: log-sigma overlap extension gamma (DiffusionBlocks App. C;
                0.05 for vision/diffusion, 0.1 for text).
            clean: precomputed L2-normalized target embeddings. Pass this when
                the same batch is reused across micro-steps -- computing it here
                would redo the `wte` lookup and normalization every micro-step.
            attn_mask: optional attention mask, threaded into the blocks (so
                packed sequences can be block-diagonal).
        """
        b = self._activate_block(block_idx)
        # The embedding table is used twice, and the two uses need different
        # gradient treatment:
        #
        # * as the **input** base: `noisy = c_in * (clean + sigma * eps)`. The
        #   paper conditions the denoiser on clean token embeddings (App. B), and
        #   those embeddings come from `wte`, so this path must stay in the graph
        #   or the embedding table never trains -- while `generate()` needs it to
        #   build the prefix.
        # * as the **target**: `loss = w(sigma) * ||pred - clean||^2`. This must
        #   be detached. It is the "clean data" of the denoising problem, and if
        #   it were differentiable the model could reduce the loss by shrinking
        #   the embeddings rather than by denoising well.
        #
        # When `clean` is supplied by the caller it is already a hoisted,
        # no-grad tensor, so it is used for both (this is the fast path, and the
        # embedding then trains via whatever else touches it).
        if clean is None:
            clean_input = F.normalize(self.model.transformer.wte(idx).float(), dim=-1)
            clean = clean_input.detach()
        else:
            clean_input = clean
        sigma = self.partitioner.sample_sigma(b, generator=generator, overlap=overlap)
        c_in, _, w = edm_preconditioning(sigma, self.partitioner.sigma_data)
        noisy = c_in * (clean_input + sigma * torch.randn_like(clean))
        t = idx.size(1)
        pred = self._run_block_denoiser(b, noisy, sigma, t, attn_mask=attn_mask)
        loss = (w * (pred - clean).square()).mean()
        # Reset first: a stale value from a previous call must not survive into
        # a step that ran without a distiller.
        self.last_kd_loss = 0.0
        if self.distiller is not None:
            # Denoiser distillation (KD, PRD 3.1, EDM form). The float twin
            # re-runs the *same* block on the *same* `noisy` and sigma -- no new
            # noise is sampled for it -- so the only difference between the two
            # predictions is quantization, which is what the anchor is supposed
            # to measure. `alpha` mixes the anchor against the data term rather
            # than adding a second, unnormalized objective.
            alpha = self.distiller.alpha
            loss_kd = self.distiller(
                pred,
                lambda: self.distiller.teacher._run_block_denoiser(
                    b, noisy, sigma, t, attn_mask=attn_mask
                ),
                weight=w,
            )
            loss = (1.0 - alpha) * loss + alpha * loss_kd
            self.last_kd_loss = float(loss_kd.detach())
        return loss, sigma

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor | list[int] | None = None,
        max_new_tokens: int = 20,
        num_steps: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
    ) -> list[int]:
        """Euler diffusion sampler for token generation.

        Sequential ODE steps from sigma_max to sigma_min (Eq. 4-5 in paper).
        For each step, applies the corresponding block b where sigma in [sigma_b, sigma_{b-1}],
        Euler updating the noisy embedding z, followed by classification projection.
        """
        torch.manual_seed(seed)
        device = self.model.transformer.wte.weight.device
        n_embd = self.model.config.n_embd
        head_dim = n_embd // self.model.config.n_head
        groups = self.block_layers()
        b_count = self.partitioner.num_blocks

        # Schedule of noise levels from sigma_max down to sigma_min
        if num_steps is not None and num_steps != b_count:
            sigmas = torch.exp(
                torch.linspace(
                    math.log(self.partitioner.sigma_min),
                    math.log(self.partitioner.sigma_max),
                    num_steps + 1,
                    device=device,
                )
            )
        else:
            sigmas = self.partitioner.boundaries().to(device)
        # Partition boundaries: sigmas[0]=sigma_min, sigmas[b_count]=sigma_max
        step_sigmas = torch.flip(sigmas, dims=[0])  # sigma_max down to sigma_min

        if idx is None or (isinstance(idx, (list, torch.Tensor)) and len(idx) == 0):
            cur_tokens = torch.empty((1, 0), dtype=torch.long, device=device)
        elif isinstance(idx, list):
            cur_tokens = torch.tensor([idx], dtype=torch.long, device=device)
        else:
            cur_tokens = idx.to(device)
            if cur_tokens.dim() == 1:
                cur_tokens = cur_tokens.unsqueeze(0)

        out_tokens = cur_tokens[0].tolist()

        for _ in range(max_new_tokens):
            seq_len = cur_tokens.size(1) + 1
            cos, sin = self.model._precompute_rotary_embeddings(seq_len, head_dim)

            # Target token initial state is pure noise: z ~ N(0, sigma_max^2 I)
            z_curr = torch.randn((1, 1, n_embd), device=device) * step_sigmas[0]

            for s in range(len(step_sigmas) - 1):
                sigma_curr = step_sigmas[s]
                sigma_next = step_sigmas[s + 1]
                delta_sigma = sigma_curr - sigma_next

                # Determine which block handles this noise level
                # Intervals are [sigma_b, sigma_{b-1}] where b in [0, B-1]
                b_idx = 0
                for b in range(b_count):
                    if sigmas[b] <= sigma_curr <= sigmas[b + 1]:
                        b_idx = b
                        break

                c_in, _, _ = edm_preconditioning(
                    sigma_curr, self.partitioner.sigma_data
                )

                # Prefix tokens if any
                if cur_tokens.size(1) > 0:
                    clean_prefix = F.normalize(
                        self.model.transformer.wte(cur_tokens).float(), dim=-1
                    )
                    full_z = torch.cat([clean_prefix, c_in * z_curr], dim=1)
                else:
                    full_z = c_in * z_curr

                # Per-layer AdaLN, mirroring denoise_step: the adapter returns
                # one (gamma, beta) per layer of the active block, consumed
                # inside each layer rather than applied to the block's input.
                conds = self.adapters[b_idx](full_z, sigma_curr.view(1))

                h = full_z
                for i, layer in enumerate(groups[b_idx]):
                    h = self.model.transformer.h[layer](
                        h, None, (cos, sin), (seq_len, 0), None, cond=conds[i]
                    )

                target_h = h[:, -1:, :]
                pred = self.denoise_heads[b_idx](target_h)

                # Euler update step: z_{i} = z_{i-1} + (delta_sigma / sigma_{i-1}) * (z_{i-1} - pred)
                z_curr = z_curr + (delta_sigma / sigma_curr) * (z_curr - pred)

            # Logit projection with softcap
            logits = self.model.lm_head(z_curr[:, -1, :])
            logits = logits[..., : self.model.config.vocab_size].float()
            softcap = 15.0
            logits = softcap * torch.tanh(logits / softcap)

            if temperature == 0.0:
                next_tok = torch.argmax(logits, dim=-1).item()
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()

            out_tokens.append(next_tok)
            cur_tokens = torch.cat(
                [cur_tokens, torch.tensor([[next_tok]], device=device)], dim=1
            )

        return out_tokens


def configure_cpu_training(num_threads: int = 4) -> None:
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)


def cpu_adamw_for(engine: DiffusionBlockEngine, lr: float = 3e-4) -> torch.optim.AdamW:
    """AdamW over every parameter the engine owns, fused for the CPU path.

    Takes `engine.parameters()` rather than re-listing the owned subtrees: a
    hand-written list here silently drops the parameters of any subtree added
    later, and AdamW says nothing when it is handed a short list.
    """
    return torch.optim.AdamW(engine.parameters(), lr=lr, weight_decay=0.01, fused=True)


def packed_lm_batch(
    tokens: list[int] | torch.Tensor, seq_len: int, batch_size: int, step: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cyclic packed LM batch: concatenated stream cut to exact seq_len, targets shifted by one."""
    n = len(tokens)
    base = (step * batch_size * seq_len) % n
    idx = torch.tensor(
        [
            [tokens[(base + i * seq_len + j) % n] for j in range(seq_len)]
            for i in range(batch_size)
        ]
    )
    targets = torch.tensor(
        [
            [tokens[(base + i * seq_len + j + 1) % n] for j in range(seq_len)]
            for i in range(batch_size)
        ]
    )
    return idx, targets


def pack_sequences(docs: list[list[int]], seq_len: int) -> list[list[int]]:
    stream: list[int] = [tok for doc in docs for tok in doc]
    n = len(stream) - len(stream) % seq_len
    return [stream[i : i + seq_len] for i in range(0, n, seq_len)]
