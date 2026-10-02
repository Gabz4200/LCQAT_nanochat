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

import torch
import wandb

from nanochat.data.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.data.tokenizer import get_token_bytes, get_tokenizer
from nanochat.models.flash_attention import HAS_FA3
from nanochat.utils.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_peak_flops,
    print0,
    print_banner,
)
from scripts._train.build import (
    assemble_run_config,
    build_base_model,
    build_efqat_freezer,
    build_engine,
    build_grad_scaler,
    build_kd_denoiser,
    build_kd_loss_fn,
    build_latch_freezer,
    build_optimizer,
    convert_to_fp8,
)
from scripts._train.eval import EvalContext
from scripts._train.loop import LoopContext, train_loop

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
from nanochat.models.quant.pruning import (  # noqa: E402
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
from nanochat.models.quant.w6 import add_w6_args  # noqa: E402

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
    help=(
        "number of diffusion blocks for block-wise training (default: 4). "
        "0 (or any negative value) disables DiffusionBlocks entirely and trains "
        "a plain autoregressive LM by next-token cross-entropy: no partitioner, "
        "no denoise heads, no block isolation, every layer trains every step. "
        "Use it for a conventional baseline -- --db-objective ce is NOT that "
        "baseline, because it still routes through the engine and still "
        "gradients only one block at a time."
    ),
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
from nanochat.models.flash_attention import USE_FA3  # noqa: E402

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

base = build_base_model(args, vocab_size, device, ddp_rank, sparse_schedule)
model = base.model
orig_model = base.model  # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model_config = base.model_config
model_config_kwargs = base.model_config_kwargs
lcqat_requested = base.lcqat_requested
lcqat_active = base.lcqat_active
checkpoint_dir = base.checkpoint_dir
resuming = base.resuming
resumed_db_blocks = base.resumed_db_blocks
resumed_sparse = base.resumed_sparse
meta_data = base.meta_data
optimizer_data = base.optimizer_data

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

convert_to_fp8(model, args, device_type, lcqat_active)

# -----------------------------------------------------------------------------
# Compile the model

# Initialize DiffusionBlocks Engine for block-wise training.
db = build_engine(
    args,
    model,
    device,
    lcqat_active,
    resuming,
    resumed_db_blocks,
    sparse_schedule,
)
use_diffusion_blocks = db.use_diffusion_blocks
num_db_blocks = db.num_db_blocks
engine = db.engine
float_twin = db.float_twin
n_twin_stripped = db.n_twin_stripped

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

run_config = assemble_run_config(args, model, model_config, vocab_size)
total_batch_size = run_config.total_batch_size
dmodel_lr_scale = run_config.dmodel_lr_scale
batch_lr_scale = run_config.batch_lr_scale
weight_decay_scaled = run_config.weight_decay_scaled
num_params = run_config.num_params
num_flops_per_token = run_config.num_flops_per_token
num_scaling_params = run_config.num_scaling_params
target_tokens = run_config.target_tokens

# -----------------------------------------------------------------------------
# Initialize the Optimizer (AdamW-only for DiffusionBlocks engine)

optimizer, trainable_root = build_optimizer(
    args,
    model,
    engine,
    use_diffusion_blocks,
    run_config,
    device_type,
    resuming,
    optimizer_data,
)

# The `--kd-alpha` / `--db-objective edm` incompatibility is checked inside
# build_kd_loss_fn, before the teacher is loaded.
kd_loss_fn = build_kd_loss_fn(args, device, lcqat_active)

# -----------------------------------------------------------------------------
# EfQAT selective layer freezing (PRD 3.2): freeze middle-layer codebook +
# weight gradients after a warmup window, keeping only critical outlier
# layers (embeddings, attn q/k, output) trainable.
efqat_freezer = build_efqat_freezer(
    args, model, engine, use_diffusion_blocks, lcqat_active
)

# -----------------------------------------------------------------------------
# Denoiser distillation (PRD 3.1, EDM form): hand the frozen float twin to the
# engine, which adds the KD term to the EDM objective inside `denoise_step`.
kd_denoiser = build_kd_denoiser(args, engine, float_twin)

# -----------------------------------------------------------------------------
# EfQAT per-block permanent latching (PRD 3.2). The middle-band freezer above
# is global and one-shot; this one is per diffusion block and, crucially,
# *permanent*: once a block's noise range is specialized, its quantization
# parameters must never move again, however many later steps resample it. That
# only holds if the veto runs inside `_requires_grad_for`, i.e. before
# `_activate_block` re-enables the block -- which `set_freezer` arranges.
block_latch_freezer, efqat_latch_targets = build_latch_freezer(
    args,
    engine,
    num_db_blocks,
    use_diffusion_blocks,
    lcqat_active,
    resuming,
    meta_data,
)

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = build_grad_scaler()

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

# -----------------------------------------------------------------------------
# Training loop

eval_ctx = EvalContext(
    model=model,
    orig_model=orig_model,
    tokenizer=tokenizer,
    device=device,
    args=args,
    wandb_run=wandb_run,
    ddp_world_size=ddp_world_size,
    token_bytes=token_bytes,
    build_val_loader=build_val_loader,
)
train_loop(
    LoopContext(
        args=args,
        model=model,
        orig_model=orig_model,
        engine=engine,
        optimizer=optimizer,
        scaler=scaler,
        trainable_root=trainable_root,
        tokenizer=tokenizer,
        train_loader=train_loader,
        x=x,
        y=y,
        dataloader_state_dict=dataloader_state_dict,
        wandb_run=wandb_run,
        device=device,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        synchronize=synchronize,
        get_max_memory=get_max_memory,
        gpu_peak_flops=gpu_peak_flops,
        token_bytes=token_bytes,
        build_val_loader=build_val_loader,
        sparse_schedule=sparse_schedule,
        kd_loss_fn=kd_loss_fn,
        kd_denoiser=kd_denoiser,
        efqat_freezer=efqat_freezer,
        block_latch_freezer=block_latch_freezer,
        efqat_latch_targets=efqat_latch_targets,
        use_diffusion_blocks=use_diffusion_blocks,
        num_db_blocks=num_db_blocks,
        checkpoint_dir=checkpoint_dir,
        model_config_kwargs=model_config_kwargs,
        lcqat_active=lcqat_active,
        user_config=user_config,
        resuming=resuming,
        meta_data=meta_data,
        total_batch_size=total_batch_size,
        num_iterations=num_iterations,
        num_flops_per_token=num_flops_per_token,
        num_scaling_params=num_scaling_params,
        master_process=master_process,
        eval_ctx=eval_ctx,
    )
)

# cleanup
wandb_run.finish()  # wandb run finish
compute_cleanup()
