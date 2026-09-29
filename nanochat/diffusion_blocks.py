"""DiffusionBlocks-CPU: block-wise AR training + diffusion inference for constrained CPUs."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_noise_embedding(log_sigma: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / max(half - 1, 1))
    args = log_sigma.unsqueeze(-1) * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


class NoiseConditionedBlockAdapter(nn.Module):
    """AdaLN conditioning: sinusoidal sigma embed -> 2-layer SiLU MLP -> gamma/beta."""

    def __init__(self, n_embd: int, cond_dim: int = 32) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        # NOTE: do NOT keep a `self.out = self.mlp[-1]` alias here. That alias
        # shares the same nn.Linear Parameter object under two attribute paths
        # (`mlp.2` and `out`), so nn.Module registers it twice and the optimizer
        # emits "parameter group with duplicate parameters". The final linear is
        # reachable as self.mlp[-1]; nothing reads self.out.
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, 4 * cond_dim),
            nn.SiLU(),
            nn.Linear(4 * cond_dim, 2 * n_embd),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        c = self.mlp(
            sinusoidal_noise_embedding(torch.log(sigma), self.cond_dim).to(x.dtype)
        )
        gamma, beta = c.chunk(2, dim=-1)
        return F.rms_norm(x, (x.size(-1),)) * (
            1.0 + gamma.unsqueeze(1)
        ) + beta.unsqueeze(1)


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
        self.distribution = distribution

    @property
    def noise_boundaries(self) -> torch.Tensor:
        return self.boundaries()

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
        lo = _standard_normal_cdf(
            torch.tensor(
                (math.log(self.sigma_min) - self.p_mean) / self.p_std,
                dtype=torch.float64,
            )
        )
        hi = _standard_normal_cdf(
            torch.tensor(
                (math.log(self.sigma_max) - self.p_mean) / self.p_std,
                dtype=torch.float64,
            )
        )
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
    ) -> None:
        self.model = model
        self.partitioner = partitioner
        self.dtype = dtype
        n_layer = len(model.transformer.h)
        assert partitioner.num_blocks <= n_layer
        n_embd = model.config.n_embd
        self.adapters = nn.ModuleList(
            [
                NoiseConditionedBlockAdapter(n_embd, cond_dim)
                for _ in range(partitioner.num_blocks)
            ]
        )
        self.denoise_head = nn.Linear(n_embd, n_embd)
        nn.init.zeros_(self.denoise_head.weight)
        nn.init.zeros_(self.denoise_head.bias)

    def train(self, mode: bool = True) -> "DiffusionBlockEngine":
        self.model.train(mode)
        self.adapters.train(mode)
        self.denoise_head.train(mode)
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
        self.model.zero_grad(set_to_none=set_to_none)
        self.adapters.zero_grad(set_to_none=set_to_none)
        self.denoise_head.zero_grad(set_to_none=set_to_none)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs) -> "DiffusionBlockEngine":
        self.model.to(*args, **kwargs)
        self.adapters.to(*args, **kwargs)
        self.denoise_head.to(*args, **kwargs)
        return self

    def apply_lcqat(self, layer_config) -> int:
        """Retrofit the diffusion engine's adapter MLPs + denoise_head to LC-QAT.

        The base transformer is retrofitted separately (via `retrofit_model` on
        `self.model`); this method covers the engine-owned Linear layers that
        `base_train`'s `--lcqat` flag must also quantize so the whole training
        pipeline is LC-QAT (PRD: "the only training method that exists must
        use it"). Returns the number of LCQATLinear modules created.
        """
        from nanochat.lcqat import retrofit_model

        # The denoise_head is a plain nn.Linear; retrofit it in place.
        self.denoise_head = retrofit_model(self.denoise_head, layer_config)
        # Each NoiseConditionedBlockAdapter owns 2 Linear layers (c_fc, c_proj);
        # retrofit them too. retrofit_model returns a new module, so replace
        # the ModuleList slot in place.
        for i, adapter in enumerate(self.adapters):
            self.adapters[i] = retrofit_model(adapter, layer_config)
        return (
            sum(
                1
                for m in self.adapters
                for sub in m.modules()
                if sub.__class__.__name__ == "LCQATLinear"
            )
            + 1
        )

    def apply_sparseprop(
        self,
        sparsity: float = 0.75,
        target_modules: list[str] | None = None,
        with_lcqat: bool = False,
    ) -> int:
        """Retrofit the engine's Linear modules with SparseProp sparse backward.

        Replaces nn.Linear modules in the base transformer and denoise_head
        with SparsePropLinear (drop-in, O(nnz) backward via AVX2 kernels).
        When ``with_lcqat=True``, existing LCQATLinear modules are wrapped as
        SparsePropLinearLCQAT so the codebook quantizers and sparse backprop
        coexist in the same module (LC-QAT + SparseProp). Module name grouping
        used by the DB-CPU partitioner is preserved because SparsePropLinear
        subclasses nanochat Linear. Returns the number of SparseProp modules
        created.
        """
        from nanochat.lcqat.sparseprop import (
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
        inject_sparseprop_layers(
            self.denoise_head,
            sparsity=sparsity,
            target_modules=target_modules,
            with_lcqat=with_lcqat,
        )
        for i, adapter in enumerate(self.adapters):
            self.adapters[i] = inject_sparseprop_layers(
                adapter,
                sparsity=sparsity,
                target_modules=target_modules,
                with_lcqat=with_lcqat,
            )
        count = (
            sum(
                1
                for m in self.adapters
                for sub in m.modules()
                if isinstance(sub, (SparsePropLinear, SparsePropLinearLCQAT))
            )
            + sum(
                1
                for m in self.model.modules()
                if isinstance(m, (SparsePropLinear, SparsePropLinearLCQAT))
            )
            + sum(
                1
                for m in self.denoise_head.modules()
                if isinstance(m, (SparsePropLinear, SparsePropLinearLCQAT))
            )
        )
        return count

    def kv_codebooks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-(layer, head) K/V codebooks for the QuantizedKVCache (PRD 7.1).

        Reads each block's attn.c_k/c_v out-quantizer. The PRD 7.1 storage
        layout is per (layer, head), so the module codebook is repeated across
        that layer's KV heads. Returns fp32 [n_layers, n_kv_head, K].
        """
        from nanochat.engine import kv_codebooks_from_model

        return kv_codebooks_from_model(self.model)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Export full engine state (base model + adapters + denoise_head)."""
        sd = self.model.state_dict()
        for k, v in self.adapters.state_dict().items():
            sd[f"db_adapters.{k}"] = v
        for k, v in self.denoise_head.state_dict().items():
            sd[f"db_denoise_head.{k}"] = v
        return sd

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = False
    ) -> None:
        """Load engine state from unified checkpoint."""
        model_sd = {}
        adapters_sd = {}
        head_sd = {}
        for k, v in state_dict.items():
            if k.startswith("db_adapters."):
                adapters_sd[k[len("db_adapters.") :]] = v
            elif k.startswith("db_denoise_head."):
                head_sd[k[len("db_denoise_head.") :]] = v
            else:
                model_sd[k] = v
        self.model.load_state_dict(model_sd, strict=strict)
        if adapters_sd:
            self.adapters.load_state_dict(adapters_sd, strict=strict)
        if head_sd:
            self.denoise_head.load_state_dict(head_sd, strict=strict)

    def named_parameters(self, recurse: bool = True, remove_duplicate: bool = False):
        """Yield (name, param) for model + adapters + denoise_head.

        `build_qat_param_groups` and `SelectiveFreezer` need a single
        `named_parameters()` view over the whole engine tree, so we forward
        into the three owned subtrees with the `db_` prefix for adapter/head
        params (matching the `state_dict` key prefix).
        """
        for name, p in self.model.named_parameters("", recurse, remove_duplicate):
            yield name, p
        for name, p in self.adapters.named_parameters(
            "db_adapters", recurse, remove_duplicate
        ):
            yield f"db_adapters.{name}", p
        for name, p in self.denoise_head.named_parameters(
            "db_denoise_head", recurse, remove_duplicate
        ):
            yield f"db_denoise_head.{name}", p

    def parameters(self) -> list[torch.nn.Parameter]:
        """Model + adapter + denoise_head params for the optimizer."""
        params = list(self.model.parameters()) + list(self.adapters.parameters())
        params += list(self.denoise_head.parameters())
        return params

    def block_layers(self) -> list[list[int]]:
        n_layer = len(self.model.transformer.h)
        base, rem = divmod(n_layer, self.partitioner.num_blocks)
        groups: list[list[int]] = []
        start = 0
        for b in range(self.partitioner.num_blocks):
            size = base + (1 if b < rem else 0)
            groups.append(list(range(start, start + size)))
            start += size
        return groups

    def _block_boundary_indices(self) -> list[int]:
        """Cumulative layer counts per block (used to build block-diagonal masks)."""
        groups = self.block_layers()
        out = [0]
        for g in groups:
            out.append(out[-1] + len(g))
        return out

    def _activate_block(self, block_idx: int | None = None) -> None:
        groups = self.block_layers()
        b = torch.randint(len(groups), (1,)).item() if block_idx is None else block_idx
        active = set(groups[b])
        for i, block in enumerate(self.model.transformer.h):
            block.requires_grad_(i in active)
            if i not in active:
                for p in block.parameters():
                    p.grad = None
        for name, p in self.model.named_parameters():
            if not name.startswith("transformer.h."):
                p.requires_grad_(True)
        self.model.train()

    def train_step(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor,
        block_idx: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

    def denoise_step(
        self,
        idx: torch.Tensor,
        block_idx: int | None = None,
        overlap: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = self.block_layers()
        b = torch.randint(len(groups), (1,)).item() if block_idx is None else block_idx
        for i, block in enumerate(self.model.transformer.h):
            block.requires_grad_(i in set(groups[b]))
            if i not in groups[b]:
                for p in block.parameters():
                    p.grad = None
        for name, p in self.model.named_parameters():
            if not name.startswith("transformer.h."):
                p.requires_grad_(False)
                p.grad = None
        self.adapters.requires_grad_(False)
        self.adapters[b].requires_grad_(True)
        with torch.no_grad():
            clean = F.normalize(self.model.transformer.wte(idx).float(), dim=-1)
        sigma = self.partitioner.sample_sigma(b, generator=generator, overlap=overlap)
        c_in, _, w = edm_preconditioning(sigma, self.partitioner.sigma_data)
        noisy = c_in * (clean + sigma * torch.randn_like(clean))
        h = self.adapters[b](noisy, sigma.view(1))
        t = idx.size(1)
        head_dim = self.model.config.n_embd // self.model.config.n_head
        cos, sin = self.model._precompute_rotary_embeddings(t, head_dim)
        for layer in groups[b]:
            h = self.model.transformer.h[layer](h, None, (cos, sin), (t, 0), None)
        pred = self.denoise_head(h)
        loss = (w * (pred - clean).square()).mean()
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

                h = self.adapters[b_idx](full_z, sigma_curr.view(1))

                for layer in groups[b_idx]:
                    h = self.model.transformer.h[layer](
                        h, None, (cos, sin), (seq_len, 0), None
                    )

                target_h = h[:, -1:, :]
                pred = self.denoise_head(target_h)

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
    params = list(engine.model.parameters()) + list(engine.adapters.parameters())
    params += list(engine.denoise_head.parameters())
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.01, fused=True)


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
