"""
Model construction and derived run configuration for `scripts.base_train`.

Holds everything that assembles what gets trained: the bare `GPT`, the LC-QAT
retrofit, SparseProp injection, the learned activation tables, FP8 conversion
and the `disable_fp8` context manager, the DiffusionBlocks engine, the
scaling-law derived batch size / LR scale / weight decay, the AdamW
parameter groups, and the KD / EfQAT anchors.

Moved out of `scripts/base_train.py` verbatim: the statements and their order
are unchanged, only the module they live in. Each step takes the values the
previous one produced as explicit parameters instead of reading module
globals, so the entry point drives the same sequence.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch

from nanochat.models.backbone import GPT, GPTConfig, Linear
from nanochat.models.quant import (
    finish_lcqat_after_load,
    lcqat_config_from_args,
    prepare_lcqat_before_load,
    retrofit_model,
    retrofit_summary,
)
from nanochat.models.quant.efqat import SelectiveFreezer
from nanochat.models.quant.kd import DenoiserDistiller, KDLoss
from nanochat.models.quant.optimizer import build_qat_param_groups, verify_partition
from nanochat.models.quant.w6 import (
    build_denoiser_teacher,
    install_sigma_codebooks,
    make_latch_freezer,
)
from nanochat.modules.checkpoint_manager import load_checkpoint
from nanochat.training.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
)
from nanochat.utils.common import COMPUTE_DTYPE, get_base_dir, print0


@dataclass
class BaseModel:
    """The bare GPT plus everything the retrofit/load path decided about it."""

    model: GPT
    orig_model: GPT
    model_config: GPTConfig
    model_config_kwargs: dict
    lcqat_requested: object | None
    lcqat_active: object | None
    checkpoint_dir: str
    resuming: bool
    resumed_db_blocks: int | None
    resumed_sparse: bool
    meta_data: dict | None
    optimizer_data: dict | None


@dataclass
class EngineSetup:
    """`engine` is None in LM mode; every downstream step reads it that way."""

    use_diffusion_blocks: bool
    num_db_blocks: int
    engine: DiffusionBlockEngine | None
    float_twin: GPT | None
    n_twin_stripped: int


@dataclass
class RunConfig:
    """The scaling-law derived run configuration."""

    total_batch_size: int
    dmodel_lr_scale: float
    batch_lr_scale: float
    weight_decay_scaled: float
    num_params: int
    num_flops_per_token: float
    num_scaling_params: int
    target_tokens: int


def build_model_meta(depth, args, vocab_size):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta


def build_base_model(args, vocab_size, device, ddp_rank, sparse_schedule):
    """Build the model, retrofit it, and load the checkpoint when resuming."""
    # Build the model, move to device, init the weights
    model = build_model_meta(
        args.depth, args, vocab_size
    )  # 1) Build on meta device (only shapes/dtypes, no data)
    model_config = model.config
    model_config_kwargs = asdict(model_config)
    print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
    model.to_empty(
        device=device
    )  # 2) All tensors get storage on target device but with uninitialized (garbage) data
    model.init_weights()  # 3) All tensors get initialized

    # LC-QAT retrofit (must happen before torch.compile, the optimizer, and FP8 conversion)
    lcqat_requested = lcqat_config_from_args(args) if args.lcqat else None
    lcqat_active = None

    # If we are resuming, overwrite the model parameters with those of the checkpoint
    base_dir = get_base_dir()
    output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"  # e.g. d12
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
    resuming = args.resume_from_step != -1
    # Unbound on the plain-float fresh-run path (no resume, no LC-QAT retrofit);
    # the two branches below assign them.
    resumed_sparse = False
    meta_data = None
    optimizer_data = None
    if resuming:
        print0(f"Resuming optimization from step {args.resume_from_step}")
        model_data, optimizer_data, meta_data = load_checkpoint(
            checkpoint_dir,
            args.resume_from_step,
            device,
            load_optimizer=True,
            rank=ddp_rank,
        )
        lcqat_active = prepare_lcqat_before_load(
            model, model_data, meta_data.get("lcqat"), lcqat_requested
        )
        # The diffusion engine owns db_adapters.* / db_denoise_head.* on top of the
        # bare GPT; those keys do not belong in the base model's state_dict.
        # checkpoint_manager.build_model does the same strip.
        # SparseProp must be injected BEFORE the load, not after: the checkpoint
        # carries `sparsity_mask` / `w_ptr` / `w_col` / `w_ptr_csc` / `w_row` as
        # persistent buffers, so loading into a model that has not been injected yet
        # fails with "unexpected key(s)". Injecting here also means the saved mask is
        # what gets used -- previously the mask was non-persistent, so a resume
        # silently re-rolled a random pattern while reusing the checkpoint's weights.
        if args.sparseprop:
            from nanochat.models.quant.sparseprop import inject_sparseprop_layers

            inject_sparseprop_layers(
                model,
                sparsity=args.sparseprop_sparsity,
                with_lcqat=lcqat_active is not None,
            )
            # No pruning here: the mask built above is a placeholder that
            # load_state_dict immediately overwrites with the checkpoint's own. A
            # gradual ramp resumes from its position in the global step counter, so
            # it needs no per-layer state either.
            resumed_sparse = True
        else:
            resumed_sparse = False
        base_model_data = {
            k: v
            for k, v in model_data.items()
            if not k.startswith("db_adapters.")
            and not k.startswith("db_denoise_heads.")
            and not k.startswith("db_denoise_head.")
        }
        model.load_state_dict(base_model_data, strict=True, assign=True)
        if lcqat_active is None:
            lcqat_active = finish_lcqat_after_load(model, lcqat_requested)
        # The block partition is checkpoint provenance, not a fresh choice: resuming
        # with a different --db-blocks would silently re-partition the model and
        # change which parameters receive gradients every micro-step.
        resumed_db_blocks = (meta_data.get("db") or {}).get("num_blocks")
        del model_data, base_model_data  # free up this memory after the copy
    elif lcqat_requested is not None:
        retrofit_model(model, lcqat_requested)
        # D9: attach the learned activation tables. Must run after `retrofit_model`
        # (it walks the LCQATLinear pairs) and *unconditionally* -- gating on
        # "are the flags non-default" would skip the table when both flags are at
        # their defaults, which is exactly the case that needs a trainable table to
        # exist at all. Attaching with the defaults is a value no-op: the table
        # starts at the same bake export would have produced.
        from nanochat.models.quant.export import attach_learnable_activation_luts

        n_luts = attach_learnable_activation_luts(
            model,
            relaxation=lcqat_requested.lut_relaxation,
            act_body=lcqat_requested.act_body,
        )
        print0(f"Learned activation LUTs attached: {n_luts}")
        lcqat_active = lcqat_requested
        # Fresh run: inject SparseProp into the base transformer here so the two
        # retrofit passes are symmetric with the resume branch above. Must run after
        # the LC-QAT retrofit, since `with_lcqat=True` wraps the LCQATLinears.
        if args.sparseprop:
            from nanochat.models.quant.sparseprop import inject_sparseprop_layers

            inject_sparseprop_layers(
                model,
                sparsity=args.sparseprop_sparsity,
                with_lcqat=True,
            )
        resumed_sparse = False
    resumed_db_blocks = resumed_db_blocks if resuming else None
    if resuming and args.sparseprop:
        # A checkpoint trained without sparsity has no mask buffers to load, so the
        # freshly-injected mask is what will be used. State that rather than
        # letting it look like a restored pattern.
        ckpt_sparse = (meta_data or {}).get("sparseprop", {}).get("enabled")
        if ckpt_sparse is not True:
            print0(
                "SparseProp resumed with a freshly-built magnitude mask: the "
                "checkpoint did not record sparsity (this is expected on a first "
                "resume from a pre-SparseProp run)"
            )
    elif args.sparseprop:
        # Fresh run: apply the schedule's initial target (one-shot at the full
        # sparsity unless --sparseprop-start-frac ramps it in).
        achieved = sparse_schedule.apply(model, 0)
        if achieved is not None:
            print0(
                f"SparseProp pruned to {achieved:.4f} sparsity "
                f"(scope={sparse_schedule.scope}, magnitude criterion)"
            )
        else:
            # `inject_sparseprop_layers` above already masked every layer, so this
            # is the common case, not an error: the schedule declines to prune only
            # when it has no event at step 0. Reporting the injected state anyway,
            # because a run that silently masked 75% of its weights is not something
            # anyone should have to infer from the absence of a log line.
            from nanochat.models.quant.pruning import collect as _collect_sparse_layers

            masked = _collect_sparse_layers(model)
            if masked:
                total = sum(layer.weight.numel() for layer in masked)
                kept = sum(int(layer.sparsity_mask.sum()) for layer in masked)
                print0(
                    f"SparseProp injected {len(masked)} sparse Linear layers; "
                    f"{1.0 - kept / total:.4f} of their weights are exact zeros "
                    "(gradual pruning has no event at step 0)"
                )
    if lcqat_active is not None:
        print0(f"LC-QAT enabled: {retrofit_summary(model)}")

    return BaseModel(
        model=model,
        orig_model=model,
        model_config=model_config,
        model_config_kwargs=model_config_kwargs,
        lcqat_requested=lcqat_requested,
        lcqat_active=lcqat_active,
        checkpoint_dir=checkpoint_dir,
        resuming=resuming,
        resumed_db_blocks=resumed_db_blocks,
        resumed_sparse=resumed_sparse,
        meta_data=meta_data,
        optimizer_data=optimizer_data,
    )


def convert_to_fp8(model, args, device_type, lcqat_active):
    """Convert Linear layers to Float8Linear if --fp8 is set."""
    if args.fp8 and lcqat_active is not None:
        raise SystemExit(
            "--fp8 and --lcqat are mutually exclusive (both convert Linear layers)"
        )

    if args.fp8:
        if device_type != "cuda":
            print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
        else:
            # our custom fp8 is simpler than torchao, written for exact API compatibility
            # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
            import torch.nn as nn

            from nanochat.models.fp8 import (
                Float8LinearConfig,
                convert_to_float8_training,
            )

            # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
            def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
                if not isinstance(mod, nn.Linear):
                    return False
                if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                    return False
                if min(mod.in_features, mod.out_features) < 128:
                    return False
                return True

            fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
            num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
            convert_to_float8_training(
                model, config=fp8_config, module_filter_fn=fp8_module_filter
            )
            num_fp8 = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
            num_skipped = num_linear - num_fp8
            print0(
                f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)"
            )


# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if "Float8" in type(module).__name__:
            if "." in name:
                parent_name, attr_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)


def build_engine(
    args, model, device, lcqat_active, resuming, resumed_db_blocks, sparse_schedule
):
    """Initialize DiffusionBlocks Engine for block-wise training.

    `--db-blocks <= 0` trains a plain autoregressive LM instead: no partitioner,
    no adapters, no denoise heads, no block isolation. Everything downstream then
    runs against the bare `model`, which is why `engine` is None rather than a
    degenerate one-block engine. A one-block engine still runs the denoiser
    objective and still gradients one block, so it is not a baseline for this.
    """
    use_diffusion_blocks = args.db_blocks > 0
    num_db_blocks = min(args.db_blocks, args.depth) if use_diffusion_blocks else 0
    if resumed_db_blocks is not None and resumed_db_blocks != num_db_blocks:
        raise SystemExit(
            f"--db-blocks {num_db_blocks} does not match the checkpoint's "
            f"{resumed_db_blocks} blocks; the block partition is checkpoint "
            "provenance and cannot be changed on resume"
        )
    if not use_diffusion_blocks:
        print0(
            "DiffusionBlocks disabled (--db-blocks<=0): plain next-token LM training, "
            "every layer trains every step"
        )
        # These are all denoiser-only concepts. Rejecting rather than ignoring them
        # keeps a flag the user typed from looking like it did something.
        for flag, value in (
            ("--db-objective", args.db_objective),
            ("--db-block-sampling", args.db_block_sampling),
            ("--kd-denoiser-alpha", args.kd_denoiser_alpha),
            ("--efqat-latch-blocks", args.efqat_latch_blocks),
        ):
            if (
                value not in (0, 0.0, "", None)
                and not (flag == "--db-objective" and value == "edm")
                and not (flag == "--db-block-sampling" and value == "step")
            ):
                raise SystemExit(
                    f"{flag}={value!r} is meaningless with --db-blocks<=0: there are "
                    "no diffusion blocks to sample, condition or latch. Drop the flag "
                    "or set --db-blocks to a positive count."
                )
        if args.db_sigma_codebook:
            raise SystemExit(
                f"--db-sigma-codebook={args.db_sigma_codebook!r} requires diffusion "
                "blocks to condition on. Drop it or set --db-blocks to a positive count."
            )

    # device/dtype: the engine owns adapters + denoise heads, which are created here.
    # Without them they land on CPU while their siblings are on `device`, and since
    # they are in the optimizer but never see a forward on that device, their .grad
    # stays None and AdamW silently skips them forever.
    engine = None
    float_twin, n_twin_stripped = (None, 0)
    if use_diffusion_blocks:
        partitioner = EquiProbabilityPartitioner(
            num_blocks=num_db_blocks,
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=0.5,
        )
        engine = DiffusionBlockEngine(
            model, partitioner, device=device, dtype=COMPUTE_DTYPE
        )
        print0(
            f"Initialized DiffusionBlocks Engine with {num_db_blocks} independent blocks"
        )

        # PRD: "the only training method that exists must use it" — when LC-QAT is
        # enabled, retrofit the engine-owned Linear layers (adapters + denoise heads)
        # too, so the whole training pipeline is LC-QAT.
        if lcqat_active is not None:
            n_lcqat = engine.apply_lcqat(lcqat_active)
            print0(f"LC-QAT retrofitted {n_lcqat} diffusion-engine Linear layers")

        # Sigma-conditioned activation codebooks (PRD 3.2). Opt-in: a DiffusionBlocks
        # engine's blocks train on disjoint noise ranges, so one activation codebook has
        # to span all of them, and most of its levels are spent on values that never
        # occur. Two mechanisms with very different costs, so both are opt-in and the
        # default recipe is untouched.
        if args.db_sigma_codebook:
            num_anchors = args.db_sigma_anchors or num_db_blocks
            n_conditioned = install_sigma_codebooks(
                engine, args.db_sigma_codebook, args.db_sigma_anchors, num_db_blocks
            )
            print0(
                f"sigma-conditioned activation codebooks: {args.db_sigma_codebook} on "
                f"{n_conditioned} layers"
                + (f", {num_anchors} anchors" if num_anchors > 1 else "")
            )

        # SparseProp on the engine-owned layers (adapters + per-block denoise heads).
        # The base transformer is handled separately: on a fresh run by the injection
        # below, and on a resume by the pre-load injection in the resume branch, which
        # has to happen before load_state_dict can see the checkpoint's mask buffers.
        if args.sparseprop:
            n_sparse = engine.apply_sparseprop(
                sparsity=args.sparseprop_sparsity,
                with_lcqat=lcqat_active is not None,
            )
            print0(f"SparseProp injected {n_sparse} sparse Linear layers")
            # `inject_sparseprop_layers` places a per-layer magnitude mask as a
            # placeholder. The base transformer's masks were already set above; this
            # only touches the engine-owned layers (adapters + per-block denoise heads),
            # and on a resume it is skipped so the checkpoint's masks survive.
            if not resuming:
                engine_achieved = sparse_schedule.apply(engine, 0)
                if engine_achieved is not None:
                    print0(
                        f"SparseProp pruned engine layers to {engine_achieved:.4f} "
                        f"sparsity (scope={sparse_schedule.scope})"
                    )

        # Denoiser distillation teacher (PRD 3.1, EDM form). Taken *here*, before the
        # retrofits below, so on a fresh run the copy is already float: LC-QAT codebooks
        # and SparseProp sparsity are exactly what --kd-denoiser-alpha measures.
        if args.kd_denoiser_alpha > 0.0:
            float_twin, n_twin_stripped = build_denoiser_teacher(
                engine, args.kd_denoiser_alpha, args.db_objective
            )
            print0(
                f"KD denoiser twin: float copy of the engine "
                f"({n_twin_stripped} LC-QAT layers stripped)"
            )

    return EngineSetup(
        use_diffusion_blocks=use_diffusion_blocks,
        num_db_blocks=num_db_blocks,
        engine=engine,
        float_twin=float_twin,
        n_twin_stripped=n_twin_stripped,
    )


def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts["transformer_matrices"] + params_counts["lm_head"]
    return scaling_params


def assemble_run_config(args, model, model_config, vocab_size):
    """Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay."""
    # Get the parameter counts of our model
    param_counts = model.num_scaling_params()
    print0("Parameter counts:")
    for key, value in param_counts.items():
        print0(f"{key:24s}: {value:,}")
    num_params = param_counts["total"]
    num_flops_per_token = model.estimate_flops()
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    # 1) Use scaling laws to determine the optimal training horizon in tokens
    # The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
    # We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
    num_scaling_params = get_scaling_params(model)
    target_tokens = int(
        args.target_param_data_ratio * num_scaling_params
    )  # optimal tokens for the model we are about to train

    # Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
    d12_ref = build_model_meta(12, args, vocab_size)  # creates the model on meta device
    D_REF = args.target_param_data_ratio * get_scaling_params(
        d12_ref
    )  # compute-optimal d12 training horizon in tokens (measured empirically)
    B_REF = 2**19  # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

    # 2) Now that we have the token horizon, we can calculate the optimal batch size
    # We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
    # The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
    total_batch_size = args.total_batch_size  # user-provided override is possible
    if total_batch_size == -1:
        batch_size_ratio = target_tokens / D_REF
        predicted_batch_size = B_REF * batch_size_ratio**0.383
        total_batch_size = 2 ** round(
            math.log2(predicted_batch_size)
        )  # clamp to nearest power of 2 for efficiency
        print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

    # 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
    # AdamW LRs scale with 1/sqrt(n_embd), tuned at 768 (same recipe the LC-QAT
    # optimizer port carries over).
    dmodel_lr_scale = (model_config.n_embd / 768) ** -0.5
    print0(
        f"Scaling the AdamW LRs by 1/sqrt({model_config.n_embd}/768) = {dmodel_lr_scale:.6f}"
    )

    batch_lr_scale = 1.0
    batch_ratio = total_batch_size / B_REF  # B/B_ref
    if batch_ratio != 1.0:
        # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
        batch_lr_scale = batch_ratio**0.5  # η ∝ √(B/B_ref)
        print0(
            f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})"
        )

    # 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
    # We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
    # Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
    # Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
    # λ = λ_ref · √(B/B_ref) · (D_ref/D)
    # Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
    weight_decay_scaled = (
        args.weight_decay
        * math.sqrt(total_batch_size / B_REF)
        * (D_REF / target_tokens)
    )
    if weight_decay_scaled != args.weight_decay:
        print0(
            f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}"
        )

    return RunConfig(
        total_batch_size=total_batch_size,
        dmodel_lr_scale=dmodel_lr_scale,
        batch_lr_scale=batch_lr_scale,
        weight_decay_scaled=weight_decay_scaled,
        num_params=num_params,
        num_flops_per_token=num_flops_per_token,
        num_scaling_params=num_scaling_params,
        target_tokens=target_tokens,
    )


def build_optimizer(
    args,
    model,
    engine,
    use_diffusion_blocks,
    run_config,
    device_type,
    resuming,
    optimizer_data,
):
    """Initialize the Optimizer (AdamW-only for DiffusionBlocks engine).

    PRD section 5: codebook delta params (`raw_pos_deltas`, `raw_neg_deltas`)
    get their own AdamW group with a dedicated LR and zero weight decay,
    distinct from the matrix-weight group.
    Everything the optimizer owns: the engine when DiffusionBlocks is on, the
    bare GPT otherwise. `build_qat_param_groups` accepts either, since it only
    needs something exposing `named_parameters()`.
    """
    trainable_root = engine if use_diffusion_blocks else model
    param_groups = build_qat_param_groups(
        trainable_root,
        matrix_lr=args.matrix_lr * run_config.batch_lr_scale,
        weight_decay=run_config.weight_decay_scaled,
        codebook_lr=args.codebook_lr * run_config.batch_lr_scale,
        matrix_betas=(0.8, 0.95),
        matrix_eps=1e-10,
        embedding_lr=args.embedding_lr,
        unembedding_lr=args.unembedding_lr,
        scalar_lr=args.scalar_lr,
        dmodel_lr_scale=run_config.dmodel_lr_scale,
    )
    # Fail here rather than shipping a run where a whole role silently never trains
    # (which is what happened to --embedding-lr/--unembedding-lr/--scalar-lr while
    # the group builder emitted only matrix + codebook).
    verify_partition(trainable_root, param_groups)
    optimizer = make_adamw(param_groups, device_type)

    if resuming:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        restore_optimizer_momentum(optimizer, optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr

    return optimizer, trainable_root


def make_adamw(param_groups, device_type: str):
    """AdamW over `param_groups`, fused on CPU, with each group's LR recorded.

    `initial_lr` is what a resume restores: `load_state_dict` overwrites
    param_group LRs with the saved values, and a stage that wants its own
    schedule (SFT, or a warm start at a different horizon) has to put them back
    afterwards. Recording them at construction is the only place the fresh value
    is still known.
    """
    print0(
        "Optimizer groups: "
        + ", ".join(f"{g['role']}={len(g['params'])}" for g in param_groups)
    )
    optimizer = torch.optim.AdamW(param_groups, fused=(device_type == "cpu"))
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def restore_optimizer_momentum(
    optimizer, optimizer_data, source: str = "checkpoint"
) -> bool:
    """Warm-start `optimizer` from `optimizer_data`, or start fresh.

    `load_state_dict` overwrites param_group metadata (LRs, betas, weight
    decay) with the saved values. Callers that built fresh LRs must save them
    via `initial_lr` and restore them after this returns; see
    `build_optimizer` for the full sequence.

    `source` names the checkpoint in the one message that quotes it, so the
    base-train and chat-SFT callers keep their existing wording.

    Returns whether momentum was actually loaded. Two failure shapes are
    handled, and they are not the same:

    * A saved state whose total parameter count differs from the current model
      (a different stage or architecture -- e.g. a Muon-based pretrain
      checkpoint). The saved state's integer parameter indices map onto the
      *old* flat parameter list, not the current `Parameter` objects, so any
      positional copy would attach momentum to the wrong parameters. Start
      fresh.
    * The same parameter count under a different grouping. Copy the momentum
      buffers by `Parameter` identity and leave the current groups (and their
      LRs) intact.
    """
    try:
        optimizer.load_state_dict(optimizer_data)
        return True
    except ValueError:
        # Expected, not exceptional: a state saved under a different param
        # grouping cannot be loaded positionally, and that is one of the two
        # cases this function exists to handle. Both recovery paths below check
        # the parameter counts before any momentum is copied.
        pass

    saved_param_count = sum(
        len(g["params"]) for g in optimizer_data.get("param_groups", [])
    )
    current_param_count = sum(len(g["params"]) for g in optimizer.param_groups)
    if saved_param_count != current_param_count:
        print0(
            f"Pretrained optimizer skipped: saved {saved_param_count} params "
            f"vs. current {current_param_count} (architecture mismatch); "
            f"starting with a fresh optimizer"
        )
        return False
    id_to_param = {
        id(p): group for group in optimizer.param_groups for p in group["params"]
    }
    copied = 0
    for pid, state in optimizer_data.get("state", {}).items():
        target = id_to_param.get(pid)
        if target is None:
            continue
        optimizer.state[id(target)] = state
        copied += 1
    print0(
        f"Loaded optimizer momentum for {copied}/{len(id_to_param)} params "
        f"from {source} (group layout mismatch; LRs reset)"
    )
    return True


def build_kd_loss_fn(args, device, lcqat_active):
    """Knowledge Distillation teacher (PRD 3.1): optionally load a frozen FP32
    teacher and anchor QAT with a KL-divergence loss. The teacher is built from
    a separate checkpoint so it stays float (no codebook deltas)."""
    kd_loss_fn: KDLoss | None = None
    if args.kd_alpha > 0.0:
        # The objective conflict is checked FIRST, before the teacher prerequisite.
        # Both are real and both must block the run, but they are not equally
        # informative: "kd-alpha is incompatible with edm" is a statement about the
        # two flags the user actually typed, whereas "requires --kd-teacher-source"
        # asks for a third flag they may not have been thinking about. Checking the
        # prerequisite first masks the incompatibility behind an unrelated error --
        # which is exactly what happened: `--kd-alpha 0.5 --db-objective edm`
        # reported only the missing teacher, and a user who then supplied one was
        # told about the objective conflict only on the next run.
        #
        # It also avoids a pointless checkpoint load for a configuration that is
        # going to exit regardless.
        if args.db_objective == "edm":
            # Fail loudly at startup. The logit-KD loss is a cross-entropy against
            # the teacher's softmax over tokens; the EDM objective predicts a
            # denoised embedding and has no logits, so the two cannot be combined.
            # Silently skipping would look like KD "not helping" rather than "never
            # applied".
            raise SystemExit(
                "--kd-alpha > 0 (logit KL against an FP teacher) is incompatible with "
                "--db-objective edm, which predicts denoised embeddings rather than "
                "next-token logits. Use --db-objective ce with KD, or drop --kd-alpha "
                "to use the EDM objective. For an anchor under the EDM objective, use "
                "--kd-denoiser-alpha, which distills the float denoiser instead."
            )
        if args.kd_teacher_source is None or args.kd_teacher_tag is None:
            raise SystemExit(
                "--kd-alpha > 0 requires --kd-teacher-source and --kd-teacher-tag"
            )
        if lcqat_active is None:
            raise SystemExit(
                "KD anchoring (PRD 3.1) requires LC-QAT; LC-QAT is on by default -- "
                "drop --no-lcqat to enable it"
            )
        teacher_checkpoint_dir = os.path.join(
            get_base_dir(), "checkpoints", args.kd_teacher_source
        )
        print0(
            f"Loading frozen KD teacher from {teacher_checkpoint_dir} step {args.kd_teacher_tag}"
        )
        teacher_state, _, teacher_meta = load_checkpoint(
            teacher_checkpoint_dir, int(args.kd_teacher_tag), device
        )
        from nanochat.models.backbone import GPTConfig

        teacher_config = GPTConfig(**teacher_meta["model_config"])
        teacher = GPT(teacher_config)
        # Teacher is float; load raw weights (strip compiled _orig_mod prefix).
        with torch.no_grad():
            clean_state = {
                k.removeprefix("_orig_mod."): v for k, v in teacher_state.items()
            }
            teacher.load_state_dict(clean_state, strict=False)
        teacher.to(device=device, dtype=COMPUTE_DTYPE)
        teacher.eval()
        kd_loss_fn = KDLoss(teacher, alpha=args.kd_alpha, tau=args.kd_temperature)
        del teacher_state
        print0(f"KD teacher loaded: alpha={args.kd_alpha}, tau={args.kd_temperature}")
    return kd_loss_fn


def build_efqat_freezer(args, model, engine, use_diffusion_blocks, lcqat_active):
    """EfQAT selective layer freezing (PRD 3.2): freeze middle-layer codebook +
    weight gradients after a warmup window, keeping only critical outlier
    layers (embeddings, attn q/k, output) trainable."""
    efqat_freezer: SelectiveFreezer | None = None
    if lcqat_active is not None and args.efqat_freeze_after >= 0:
        efqat_freezer = SelectiveFreezer(
            model,
            warmup_steps=args.efqat_freeze_after,
            freeze_middle_frac=args.efqat_freeze_frac,
        )
        print0(
            f"EfQAT selective freezing enabled: freezing after {args.efqat_freeze_after} steps "
            f"(middle {args.efqat_freeze_frac:.0%} of {len(efqat_freezer.model.transformer.h)} layers)"
        )
        # The engine consults the freezer before enabling a block, so the freeze
        # survives; previously `_activate_block` re-enabled everything each step.
        if use_diffusion_blocks:
            engine.set_freezer(efqat_freezer)
        else:
            # No `_activate_block` runs in LM mode, so nothing would otherwise ever
            # re-enable the frozen band and the veto has to be applied here.
            efqat_freezer.freeze()
            n_frozen_now = sum(1 for p in model.parameters() if not p.requires_grad)
            print0(
                f"EfQAT froze {n_frozen_now} parameters (LM mode: no engine arbiter)"
            )
    return efqat_freezer


def build_kd_denoiser(args, engine, float_twin):
    """Denoiser distillation (PRD 3.1, EDM form): hand the frozen float twin to the
    engine, which adds the KD term to the EDM objective inside `denoise_step`.
    Defined unconditionally because the per-step logging reads it on every step,
    not only on the ones where the anchor is installed."""
    kd_denoiser = None
    if float_twin is not None:
        kd_denoiser = DenoiserDistiller(float_twin, alpha=args.kd_denoiser_alpha)
        engine.set_distiller(kd_denoiser)  # engine is not None: kd-denoiser-alpha is
        # rejected outright when DiffusionBlocks is off (see the startup guard).
        print0(
            f"KD denoiser distillation enabled: alpha={args.kd_denoiser_alpha}, "
            f"anchor = w(sigma)*||D_quant - D_float||^2 on the same noisy input"
        )
    return kd_denoiser


def build_latch_freezer(
    args, engine, num_db_blocks, use_diffusion_blocks, lcqat_active, resuming, meta_data
):
    """EfQAT per-block permanent latching (PRD 3.2). The middle-band freezer is
    global and one-shot; this one is per diffusion block and, crucially,
    *permanent*: once a block's noise range is specialized, its quantization
    parameters must never move again, however many later steps resample it. That
    only holds if the veto runs inside `_requires_grad_for`, i.e. before
    `_activate_block` re-enables the block -- which `set_freezer` arranges."""
    block_latch_freezer = None
    efqat_latch_targets: list[int] = []
    if lcqat_active is not None and use_diffusion_blocks:
        block_latch_freezer, efqat_latch_targets = make_latch_freezer(
            engine, args.efqat_latch_blocks, num_db_blocks
        )
    if block_latch_freezer is not None:
        print0(
            f"EfQAT per-block latch enabled: blocks {efqat_latch_targets} freeze "
            f"permanently at step {max(args.efqat_latch_after, 0)}"
        )
        # Latches are not recoverable from the state_dict -- a frozen parameter
        # looks exactly like a converged one -- so a resumed run would silently
        # start training a block the previous run had retired.
        if resuming and meta_data:
            block_latch_freezer.load_metadata(meta_data.get("efqat_latch"))
    if block_latch_freezer is not None:
        engine.set_freezer(block_latch_freezer)
    return block_latch_freezer, efqat_latch_targets


def build_grad_scaler():
    """GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)"""
    scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
    if scaler is not None:
        print0("GradScaler enabled for fp16 training")
    return scaler
