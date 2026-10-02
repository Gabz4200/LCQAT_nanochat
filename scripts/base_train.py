"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import gc
import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_peak_flops,
    is_ddp_initialized,
    print0,
    print_banner,
)
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.diffusion_blocks import DiffusionBlockEngine, EquiProbabilityPartitioner
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.lcqat import (
    finish_lcqat_after_load,
    lcqat_config_from_args,
    prepare_lcqat_before_load,
    retrofit_model,
    retrofit_summary,
)
from nanochat.lcqat.efqat import SelectiveFreezer
from nanochat.lcqat.kd import DenoiserDistiller, KDLoss
from nanochat.lcqat.optimizer import build_qat_param_groups, verify_partition
from nanochat.lcqat.w6 import (
    build_denoiser_teacher,
    describe_sigma_codebooks,
    install_sigma_codebooks,
    make_latch_freezer,
)
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from scripts.base_eval import evaluate_core

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument(
    "--run",
    type=str,
    default="dummy",
    help="wandb run name ('dummy' disables wandb logging)",
)
# Runtime
parser.add_argument(
    "--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)"
)
# FP8 training
parser.add_argument(
    "--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)"
)
parser.add_argument(
    "--fp8-recipe",
    type=str,
    default="tensorwise",
    choices=["rowwise", "tensorwise"],
    help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)",
)
parser.add_argument(
    "--no-lcqat",
    action="store_false",
    dest="lcqat",
    default=True,
    help="disable LC-QAT (default: LC-QAT is always on; pass this to run plain float training)",
)
parser.add_argument(
    "--lcqat-preset",
    type=str,
    default="asym",
    choices=["asym", "small", "prd"],
    help=(
        "per-layer K allocation. 'asym' (default) uses the same total level "
        "counts as 'small' but splits them by sign: m_neg=0 for the two "
        "non-negative MLP tensors (`relu(x).square()` feeds mlp.c_proj), which "
        "turns a symmetric 15 with 7 dead levels into 8 usable ones. 'small' is "
        "the symmetric maximum-compression table, 'prd' is the PRD table "
        "(8-bit down_proj). Each K also accepts an explicit split via "
        "--lcqat-k-map, e.g. '0-7/0-7'."
    ),
)
parser.add_argument(
    "--lcqat-k-map",
    type=str,
    default="",
    help="override K per module substring, e.g. 'mlp.c_proj:255/255,attn.c_v:15/15'",
)
parser.add_argument(
    "--codebook-lr",
    type=float,
    default=1e-3,
    help="learning rate for codebook step parameters (PRD: 10-50x network weights)",
)
parser.add_argument(
    "--codebook-grad-scale",
    type=str,
    default="inv_sqrt_n",
    choices=["none", "inv_sqrt_n"],
    help=(
        "PRD 2.4 codebook gradient scaling. 'inv_sqrt_n' (default) scales the "
        "codebook gradient by 1/sqrt(numel) -- in a 4096x4096 layer 16.7M "
        "elements pool into one K-entry codebook, and unscaled the step "
        "parameters oscillate relative to the weights. 'none' uses the plain "
        "STE. Note the two interact multiplicatively with --codebook-lr: with "
        "N = B*T*D in the millions the activation codebook gradient is ~1000x "
        "smaller under 'inv_sqrt_n', so this is a real trade, not a free win."
    ),
)
# Knowledge Distillation (PRD 3.1)
parser.add_argument(
    "--kd-teacher-source",
    type=str,
    default=None,
    help="checkpoint source tag to load the FP32 teacher for KD anchoring (PRD 3.1)",
)
parser.add_argument(
    "--kd-teacher-tag",
    type=str,
    default=None,
    help="checkpoint tag of the teacher model (PRD 3.1)",
)
parser.add_argument(
    "--kd-alpha",
    type=float,
    default=0.0,
    help=(
        "weight of the KL-distillation term (PRD 3.1): L_total = (1-a)*CE + a*KD. "
        "0.0 disables KD. A value > 0 requires --kd-teacher-source and "
        "--kd-teacher-tag, and is incompatible with --db-objective edm."
    ),
)
parser.add_argument(
    "--kd-temperature",
    type=float,
    default=2.0,
    help="softmax temperature for KL distillation (PRD 3.1)",
)
# Denoiser distillation: the EDM-native form of the same anchor. --kd-alpha
# needs logits, which the denoising objective never produces, so this is a
# separate opt-in path rather than a reuse of it. Registered in add_w6_args.
# EfQAT selective layer freezing (PRD 3.2)
parser.add_argument(
    "--efqat-freeze-after",
    type=int,
    default=-1,
    help="freeze middle-layer codebook+weight grads after N steps (PRD 3.2), -1 = disabled",
)
parser.add_argument(
    "--no-sparseprop",
    action="store_false",
    dest="sparseprop",
    default=True,
    help="disable SparseProp sparse backprop (default: SparseProp is always on)",
)
parser.add_argument(
    "--sparseprop-sparsity",
    type=float,
    default=0.75,
    help="sparsity level for SparseProp (fraction of weights pruned, 0.0-1.0)",
)
# Scope / gradual schedule / dense-below-threshold. Registered through the
# shared helper so base_train, chat_sft and chat_rl cannot drift apart.
from nanochat.lcqat.pruning import (  # noqa: E402
    add_sparseprop_pruning_args,
    schedule_from_args,
)

add_sparseprop_pruning_args(parser)
parser.add_argument(
    "--efqat-freeze-frac",
    type=float,
    default=0.5,
    help="fraction of middle transformer layers to freeze (PRD 3.2)",
)
# W6 DiffusionBlocks features (sigma codebooks / EfQAT per-block latching /
# denoiser KD) plus --lcqat-channel-center. Registered through the shared
# helper so base_train, chat_sft and chat_rl cannot drift apart on the same
# hardware knob, the same argument add_sparseprop_pruning_args makes above.
from nanochat.lcqat.w6 import add_w6_args  # noqa: E402

add_w6_args(parser)
# Model architecture
parser.add_argument(
    "--depth", type=int, default=20, help="depth of the Transformer model"
)
parser.add_argument(
    "--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio"
)
parser.add_argument(
    "--head-dim", type=int, default=128, help="target head dimension for attention"
)
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument(
    "--window-pattern",
    type=str,
    default="SSSL",
    help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')",
)
# Training horizon (only one used, in order of precedence)
parser.add_argument(
    "--num-iterations",
    type=int,
    default=-1,
    help="explicit number of optimization steps (-1 = disable)",
)
parser.add_argument(
    "--target-flops",
    type=float,
    default=-1.0,
    help="calculate num_iterations to reach target_flops (-1 = disable)",
)
parser.add_argument(
    "--target-param-data-ratio",
    type=float,
    default=12,
    help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)",
)
# Optimization
parser.add_argument(
    "--device-batch-size",
    type=int,
    default=32,
    help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.",
)
parser.add_argument(
    "--total-batch-size",
    type=int,
    default=-1,
    help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)",
)
parser.add_argument(
    "--embedding-lr",
    type=float,
    default=0.3,
    help="learning rate for embedding parameters (Adam)",
)
parser.add_argument(
    "--unembedding-lr",
    type=float,
    default=0.008,
    help="learning rate for unembedding parameters (Adam)",
)
parser.add_argument(
    "--db-blocks",
    type=int,
    default=4,
    help="number of diffusion blocks for block-wise training (default: 4)",
)
parser.add_argument(
    "--db-objective",
    type=str,
    default="edm",
    choices=["edm", "ce"],
    help=(
        "block-wise training objective. 'edm' (default) is the DiffusionBlocks "
        "method: only the active block's layers run, so activations are "
        "O(L/B) instead of O(L), and the block is trained by score matching "
        "against its own equi-probability noise range. 'ce' is the escape "
        "hatch: a full-depth next-token cross-entropy with block-isolated "
        "gradients, which saves backward memory but no forward FLOPs and gives "
        "no noise-range specialization."
    ),
)
parser.add_argument(
    "--db-overlap",
    type=float,
    default=0.1,
    help=(
        "log-sigma overlap extension gamma between adjacent blocks "
        "(DiffusionBlocks App. C). 0.0 = disjoint intervals, larger = smoother "
        "transitions. Paper uses 0.05 for vision/diffusion, 0.1 for text."
    ),
)
parser.add_argument(
    "--db-block-sampling",
    type=str,
    default="step",
    choices=["step", "micro"],
    help=(
        "when to draw the active block. 'step' (default) draws once per "
        "optimizer step, so every micro-step in a step trains the same block. "
        "'micro' redraws per micro-step, which degrades block-wise training "
        "into ordinary gradient accumulation (memory saving kept, noise-range "
        "specialization lost)."
    ),
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.28,
    help="weight decay for the AdamW optimizer (for weights)",
)
parser.add_argument(
    "--matrix-lr",
    type=float,
    default=0.02,
    help="learning rate for matrix parameters (AdamW)",
)
parser.add_argument(
    "--scalar-lr",
    type=float,
    default=0.5,
    help="learning rate for scalars (resid_lambdas, x0_lambdas)",
)
parser.add_argument(
    "--init-lr-frac",
    type=float,
    default=0.01,
    help="initial LR as a fraction of the base LR, ramped up during warmup",
)
parser.add_argument(
    "--warmup-steps", type=int, default=40, help="number of steps for LR warmup"
)
parser.add_argument(
    "--warmdown-ratio",
    type=float,
    default=0.65,
    help="ratio of iterations for LR warmdown",
)
parser.add_argument(
    "--final-lr-frac",
    type=float,
    default=0.05,
    help="final LR as fraction of initial LR",
)
parser.add_argument(
    "--resume-from-step",
    type=int,
    default=-1,
    help="resume training from this step (-1 = disable)",
)
# Evaluation
parser.add_argument(
    "--eval-every",
    type=int,
    default=250,
    help="evaluate val bpb every N steps (-1 = disable)",
)
parser.add_argument(
    "--eval-tokens",
    type=int,
    default=80 * 524288,
    help="number of tokens to evaluate val loss on",
)
parser.add_argument(
    "--core-metric-every",
    type=int,
    default=2000,
    help="evaluate CORE metric every N steps (-1 = disable)",
)
parser.add_argument(
    "--core-metric-max-per-task",
    type=int,
    default=500,
    help="examples per task for CORE metric",
)
parser.add_argument(
    "--sample-every",
    type=int,
    default=2000,
    help="sample from model every N steps (-1 = disable)",
)
parser.add_argument(
    "--save-every",
    type=int,
    default=-1,
    help="save checkpoints every N steps (-1 = only at end)",
)
# Output
parser.add_argument(
    "--model-tag",
    type=str,
    default=None,
    help="override model tag for checkpoint directory name",
)
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# The SparseProp pruning schedule: scope, gradual ramp, dense-below-threshold.
# Built here (right after parsing) so every setup path sees the same object and
# the training loop can call `sparse_schedule.apply(engine, step)`.
sparse_schedule = schedule_from_args(args)
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat", name=args.run, config=user_config)
)

# Flash Attention status
from nanochat.flash_attention import USE_FA3

using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(
            f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback"
        )
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(
            f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible."
        )
        print0(
            "WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns."
        )
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model


def build_model_meta(depth):
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


# Build the model, move to device, init the weights
model = build_model_meta(
    args.depth
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
        from nanochat.lcqat.sparseprop import inject_sparseprop_layers

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
    from nanochat.lcqat.export import attach_learnable_activation_luts

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
        from nanochat.lcqat.sparseprop import inject_sparseprop_layers

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
if lcqat_active is not None:
    print0(f"LC-QAT enabled: {retrofit_summary(model)}")

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

if args.fp8 and lcqat_active is not None:
    raise SystemExit(
        "--fp8 and --lcqat are mutually exclusive (both convert Linear layers)"
    )

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training

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


# -----------------------------------------------------------------------------
# Compile the model

orig_model = model  # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)

# Initialize DiffusionBlocks Engine for block-wise training
num_db_blocks = min(args.db_blocks, args.depth)
if resumed_db_blocks is not None and resumed_db_blocks != num_db_blocks:
    raise SystemExit(
        f"--db-blocks {num_db_blocks} does not match the checkpoint's "
        f"{resumed_db_blocks} blocks; the block partition is checkpoint "
        "provenance and cannot be changed on resume"
    )
partitioner = EquiProbabilityPartitioner(
    num_blocks=num_db_blocks,
    sigma_min=0.002,
    sigma_max=80.0,
    sigma_data=0.5,
)
# device/dtype: the engine owns adapters + denoise heads, which are created here.
# Without them they land on CPU while their siblings are on `device`, and since
# they are in the optimizer but never see a forward on that device, their .grad
# stays None and AdamW silently skips them forever.
engine = DiffusionBlockEngine(model, partitioner, device=device, dtype=COMPUTE_DTYPE)
print0(f"Initialized DiffusionBlocks Engine with {num_db_blocks} independent blocks")

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
float_twin, n_twin_stripped = (None, 0)
if args.kd_denoiser_alpha > 0.0:
    float_twin, n_twin_stripped = build_denoiser_teacher(
        engine, args.kd_denoiser_alpha, args.db_objective
    )
    print0(
        f"KD denoiser twin: float copy of the engine "
        f"({n_twin_stripped} LC-QAT layers stripped)"
    )


# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

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
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts["transformer_matrices"] + params_counts["lm_head"]
    return scaling_params


num_scaling_params = get_scaling_params(model)
target_tokens = int(
    args.target_param_data_ratio * num_scaling_params
)  # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12)  # creates the model on meta device
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
    args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
)
if weight_decay_scaled != args.weight_decay:
    print0(
        f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}"
    )

# -----------------------------------------------------------------------------
# Initialize the Optimizer (AdamW-only for DiffusionBlocks engine)
# PRD section 5: codebook delta params (`raw_pos_deltas`, `raw_neg_deltas`)
# get their own AdamW group with a dedicated LR and zero weight decay,
# distinct from the matrix-weight group.
param_groups = build_qat_param_groups(
    engine,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    codebook_lr=args.codebook_lr * batch_lr_scale,
    matrix_betas=(0.8, 0.95),
    matrix_eps=1e-10,
    embedding_lr=args.embedding_lr,
    unembedding_lr=args.unembedding_lr,
    scalar_lr=args.scalar_lr,
    dmodel_lr_scale=dmodel_lr_scale,
)
# Fail here rather than shipping a run where a whole role silently never trains
# (which is what happened to --embedding-lr/--unembedding-lr/--scalar-lr while
# the group builder emitted only matrix + codebook).
verify_partition(engine, param_groups)
print0(
    "Optimizer groups: "
    + ", ".join(f"{g['role']}={len(g['params'])}" for g in param_groups)
)
optimizer = torch.optim.AdamW(
    param_groups,
    fused=(device_type == "cpu"),
)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

if resuming:
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    try:
        optimizer.load_state_dict(optimizer_data)
    except ValueError:
        # The saved optimizer was produced by an older code path whose
        # parameter grouping differs from the current one (e.g. a Muon
        # stage that has since been removed, or duplicate codebook params
        # from a non-flattened SparsePropLinearLCQAT). load_state_dict
        # hard-fails on group-count mismatch.
        saved_param_count = sum(
            len(g["params"]) for g in optimizer_data.get("param_groups", [])
        )
        current_param_count = sum(len(g["params"]) for g in optimizer.param_groups)
        if saved_param_count != current_param_count:
            # Different param count (different architecture/stage, e.g. a
            # Muon-based pretrain checkpoint vs. the current AdamW-only
            # model). The saved state's integer param indices map onto the
            # old flat param list, not the current Parameter objects, so any
            # positional copy would attach momentum to the wrong params.
            # Warm-start is unsafe here; start fresh.
            print0(
                f"Pretrained optimizer skipped: saved {saved_param_count} params "
                f"vs. current {current_param_count} (architecture mismatch); "
                f"starting with a fresh optimizer"
            )
        else:
            # Same param count, different grouping: copy momentum buffers
            # per Parameter identity and keep our fresh param groups
            # (LRs, betas, weight decay) intact.
            id_to_param = {
                id(p): p for group in optimizer.param_groups for p in group["params"]
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
                f"from checkpoint (group layout mismatch; LRs reset)"
            )
    del optimizer_data
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr

# -----------------------------------------------------------------------------
# Knowledge Distillation teacher (PRD 3.1): optionally load a frozen FP32
# teacher and anchor QAT with a KL-divergence loss. The teacher is built from
# a separate checkpoint so it stays float (no codebook deltas).
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
    from nanochat.gpt import GPTConfig

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

# The `--kd-alpha` / `--db-objective edm` incompatibility is checked at the top of
# the `kd_alpha > 0` block above, before the teacher is loaded.

# -----------------------------------------------------------------------------
# EfQAT selective layer freezing (PRD 3.2): freeze middle-layer codebook +
# weight gradients after a warmup window, keeping only critical outlier
# layers (embeddings, attn q/k, output) trainable.
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
    engine.set_freezer(efqat_freezer)

# -----------------------------------------------------------------------------
# Denoiser distillation (PRD 3.1, EDM form): hand the frozen float twin to the
# engine, which adds the KD term to the EDM objective inside `denoise_step`.
# Defined unconditionally because the per-step logging reads it on every step,
# not only on the ones where the anchor is installed.
kd_denoiser = None
if float_twin is not None:
    kd_denoiser = DenoiserDistiller(float_twin, alpha=args.kd_denoiser_alpha)
    engine.set_distiller(kd_denoiser)
    print0(
        f"KD denoiser distillation enabled: alpha={args.kd_denoiser_alpha}, "
        f"anchor = w(sigma)*||D_quant - D_float||^2 on the same noisy input"
    )

# -----------------------------------------------------------------------------
# EfQAT per-block permanent latching (PRD 3.2). The middle-band freezer above
# is global and one-shot; this one is per diffusion block and, crucially,
# *permanent*: once a block's noise range is specialized, its quantization
# parameters must never move again, however many later steps resample it. That
# only holds if the veto runs inside `_requires_grad_for`, i.e. before
# `_activate_block` re-enables the block -- which `set_freezer` arranges.
block_latch_freezer = None
efqat_latch_targets: list[int] = []
if lcqat_active is not None:
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
    engine.set_freezer(block_latch_freezer)

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = (
    None if not resuming else meta_data["dataloader_state_dict"]
)
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    args.device_batch_size,
    args.max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device
)
x, y, dataloader_state_dict = next(
    train_loader
)  # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert (
    args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
)
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(
        f"Calculated number of iterations from target data:param ratio: {num_iterations:,}"
    )
else:
    raise ValueError("No training horizon specified")
total_tokens = (
    total_batch_size * num_iterations
)  # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(
    f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}"
)  # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")


def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        frac = it / warmup_iters
        return frac * 1.0 + (1 - frac) * args.init_lr_frac
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac


# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None  # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0  # EMA of training loss
    total_training_time = 0  # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = (
    args.device_batch_size * args.max_seq_len
)  # tokens per iteration for a single rank
world_tokens_per_fwdbwd = (
    tokens_per_fwdbwd * ddp_world_size
)  # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0, (
    f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
)
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(
    f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}"
)
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(
    f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}"
)
if args.db_block_sampling == "micro" and grad_accum_steps > 1:
    print0(
        "WARNING: --db-block-sampling micro with gradient accumulation is LOSSY, "
        "not just a different sampling strategy. `_apply_requires_grad` clears "
        "p.grad for parameters the active block does not own, so each micro-step "
        "erases the previous one's gradients and only the last block sampled "
        "contributes to the optimizer step. Use the default 'step' unless you "
        "are reproducing the ablation."
    )

# Packed batches arrive as document boundaries; `block_diagonal_mask` turns them
# into a causal + same-document mask so a document cannot attend across a
# neighbour's tokens. `denoise_step` calls the transformer blocks directly, so
# the mask has to be handed to it explicitly (the CE path threads it through
# GPT.forward). None means "no packing": the model's attention is already
# causal, so a full causal mask would be a no-op.
train_attn_mask: torch.Tensor | None = None

# Go!
while True:
    last_step = (
        step == num_iterations
    )  # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (
            args.device_batch_size * args.max_seq_len * ddp_world_size
        )
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            }
        )
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (
        last_step or (step > 0 and step % args.core_metric_every == 0)
    ):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(
                orig_model,
                tokenizer,
                device,
                max_per_task=args.core_metric_max_per_task,
            )
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            }
        )
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if (
        args.sample_every > 0
        and master_process
        and (last_step or (step > 0 and step % args.sample_every == 0))
    ):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        # `ar_engine` is the autoregressive sampler; it must not shadow the
        # DiffusionBlockEngine bound to `engine`, which is what train_step runs on.
        ar_engine = Engine(orig_model, tokenizer)  # orig_model avoids recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = ar_engine.generate_batch(
                    tokens, num_samples=1, max_tokens=16, temperature=0
                )
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (
        step > 0
        and step != args.resume_from_step
        and args.save_every > 0
        and step % args.save_every == 0
    ):
        save_checkpoint(
            checkpoint_dir,
            step,
            # engine.state_dict(), not orig_model.state_dict(): the engine owns
            # db_adapters.* / db_denoise_heads.* on top of the bare GPT, and meta
            # below declares meta["db"]. Saving the bare model writes a
            # checkpoint that claims a diffusion engine but carries none of its
            # parameters, so every resume silently reloads zero adapters/heads.
            engine.state_dict(),
            optimizer.state_dict(),  # optimizer state
            {  # metadata saved as json
                "step": step,
                "val_bpb": val_bpb,  # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config,  # inputs to the training script
                "lcqat": lcqat_active.as_dict() if lcqat_active is not None else None,
                # KD anchoring state. Two independent, incompatible-in-practice
                # anchors: `alpha` is the logit KL (CE objective), `denoiser_alpha`
                # is the float-twin denoiser anchor (EDM objective). Both default
                # to 0.0 = off, and both are recorded so a resume can see that a
                # run's objective differs from the default rather than silently
                # changing it.
                "kd": {
                    "alpha": args.kd_alpha,
                    "tau": args.kd_temperature,
                    "denoiser_alpha": args.kd_denoiser_alpha,
                },
                "db": {
                    "num_blocks": num_db_blocks,
                    "sigma_min": 0.002,
                    "sigma_max": 80.0,
                    "sigma_data": 0.5,
                },
                # EfQAT per-block latch: which blocks have been permanently
                # retired. Metadata only -- the tensors themselves are already
                # in engine.state_dict() -- but the freeze decision is not
                # recoverable from tensor values, so it has to be written out
                # or a resume silently restarts training a converged block.
                "efqat_latch": engine.freezer_metadata(),
                "sigma_codebook": describe_sigma_codebooks(args),
                "sparseprop": {
                    "enabled": args.sparseprop,
                    "sparsity": args.sparseprop_sparsity,
                    "scope": sparse_schedule.scope,
                    "start_frac": sparse_schedule.start_frac,
                    "every": sparse_schedule.every,
                    "ramp_steps": sparse_schedule.ramp_steps,
                    "dense_threshold": sparse_schedule.dense_threshold,
                },
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {  # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    # EfQAT: drive the selective freezer from the global step (PRD 3.2), then
    # hand it to the engine as the single `requires_grad` arbiter. Without the
    # handoff the engine re-enables every transformer parameter on the next
    # block activation, silently undoing the freeze.
    if efqat_freezer is not None:
        if efqat_freezer.update(step):
            n_frozen = sum(1 for p in engine.parameters() if not p.requires_grad)
            print0(f"EfQAT froze {n_frozen} parameters at step {step}")
    # EfQAT per-block latch (PRD 3.2). Fires before the forward, so the retired
    # block is already frozen in the very step the latch lands; latching after
    # the forward would let one more update through. `>=` rather than `==` so a
    # resumed run (which starts above the threshold) still latches, and
    # `latch_blocks` is idempotent so the re-fire is free.
    if block_latch_freezer is not None and step >= max(args.efqat_latch_after, 0):
        n_new = block_latch_freezer.latch_blocks(efqat_latch_targets)
        if n_new:
            print0(
                f"EfQAT latched blocks {efqat_latch_targets} permanently at step "
                f"{step} ({n_new} parameters frozen; total latched: "
                f"{block_latch_freezer.latched_blocks()})"
            )
    # Gradual magnitude pruning. Runs BEFORE the forward so the mask and the
    # forward agree in the same step, and so the weights it prunes are the ones
    # the optimizer is about to write into. Mask changes are monotone (a pruned
    # weight is exactly 0.0, so |W| == 0 and it can never be re-selected), which
    # is what makes a resumed ramp safe.
    if args.sparseprop:
        pruned = sparse_schedule.apply(engine, step)
        if pruned is not None:
            n_sparse_above = len(sparse_schedule.layers_above_threshold(engine))
            print0(
                f"Step {step:05d} | SparseProp pruned to {pruned:.4f} sparsity; "
                f"{n_sparse_above} layers past the "
                f"{sparse_schedule.dense_threshold:.0%} sparse-kernel threshold"
            )
    # The block is drawn ONCE per optimizer step, not once per micro-step.
    # Accumulating gradients from several blocks into one update would preserve
    # the memory saving but destroy the noise-range specialization that
    # DiffusionBlocks depends on.
    block_idx = (
        engine.sample_block()
        if args.db_objective == "edm" and args.db_block_sampling == "step"
        else None
    )
    # Hoisted out of the micro-step loop: denoise_step reuses the same (x, y)
    # across grad_accum_steps, so the embedding lookup + L2 normalization would
    # otherwise be redone once per micro-step.
    clean = None
    if args.db_objective == "edm":
        with torch.no_grad():
            clean = F.normalize(engine.model.transformer.wte(x).float(), dim=-1)

    # Reported alongside the total so an enabled anchor is observable in the
    # log: a KD term that stays at 0.0 for a whole run means the twin is not
    # actually being compared against, not that the gap vanished.
    step_kd_logged = 0.0

    for micro_step in range(grad_accum_steps):
        if args.db_objective == "edm":
            # `block_idx=None` under --db-block-sampling micro, which redraws
            # per micro-step (the ablation arm).
            loss, sigma = engine.denoise_step(
                x,
                block_idx=block_idx,
                overlap=args.db_overlap,
                attn_mask=train_attn_mask,
                clean=clean,
            )
            if engine.distiller is not None:
                step_kd_logged = step_kd_logged + float(engine.last_kd_loss)
        elif kd_loss_fn is not None:
            # KD anchoring needs logits, so the CE objective asks for them
            # explicitly (targets=None) and computes the CE term here.
            student_logits = engine.model(x, targets=None, attn_mask=train_attn_mask)
            # Mirror GPT.forward's loss exactly: the dataloader already shifted
            # y, so no extra shift here.
            loss_ce = F.cross_entropy(
                student_logits.reshape(-1, student_logits.size(-1)),
                y.reshape(-1),
                ignore_index=-1,
            )
            loss_kd = kd_loss_fn(student_logits, x)
            loss = (1.0 - kd_loss_fn.alpha) * loss_ce + kd_loss_fn.alpha * loss_kd
        else:
            loss = engine.train_step(
                x, y, block_idx=block_idx, attn_mask=train_attn_mask
            )
        train_loss = loss.detach()  # for logging
        loss = (
            loss / grad_accum_steps
        )  # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(
            train_loader
        )  # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()  # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9  # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = (
        ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    )  # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (
        1 - ema_beta ** (step + 1)
    )  # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds / 60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    kd_mean = step_kd_logged / max(grad_accum_steps, 1)
    kd_str = f" | kd: {kd_mean:.6f}" if kd_denoiser else ""
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f}{kd_str} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time / 60:.2f}m{eta_str}"
    )
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        # The float-vs-quantized denoiser gap this step's anchor actually saw.
        # Logged separately from `train/loss` because the returned loss mixes it
        # with the data term, so a silently-dead anchor is otherwise invisible.
        if kd_denoiser is not None:
            log_data["train/kd_denoiser"] = kd_mean
        # EfQAT latch progress: how many diffusion blocks have been permanently
        # retired. Plateaus at `num_db_blocks` once all of them have finished
        # specializing to their noise range.
        if block_latch_freezer is not None:
            log_data["train/efqat_latched_blocks"] = float(
                len(block_latch_freezer.latched_blocks())
            )
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect()  # manually collect a lot of garbage from setup
        gc.freeze()  # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable()  # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0:  # every 5000 steps...
        gc.collect()  # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# cleanup
wandb_run.finish()  # wandb run finish
compute_cleanup()
