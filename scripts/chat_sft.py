"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
"""

import argparse
import gc
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from nanochat.data.tokenizer import get_token_bytes
from nanochat.models.flash_attention import HAS_FA3
from nanochat.models.quant import lcqat_config_from_args, retrofit_summary
from nanochat.models.quant.retrofit import LayerKConfig
from nanochat.modules.checkpoint_manager import (
    load_model,
    load_optimizer_state,
    save_checkpoint,
)
from nanochat.modules.engine import Engine
from nanochat.modules.loss_eval import evaluate_bpb
from nanochat.tasks.common import TaskMixture
from nanochat.tasks.gsm8k import GSM8K
from nanochat.tasks.mmlu import MMLU
from nanochat.tasks.smoltalk import SmolTalk
from nanochat.training.diffusion_blocks import (
    DiffusionBlockEngine,
    EquiProbabilityPartitioner,
)
from nanochat.utils.common import (
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
)
from scripts.chat_eval import run_chat_eval

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
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
# Model loading
parser.add_argument(
    "--model-tag", type=str, default=None, help="model tag to load from"
)
parser.add_argument(
    "--model-step", type=int, default=None, help="model step to load from"
)
parser.add_argument(
    "--load-optimizer",
    type=int,
    default=1,
    help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes)",
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
        "per-layer K allocation. 'asym' (default) splits each codebook by sign, "
        "giving m_neg=0 to the two non-negative MLP tensors so no level is spent "
        "on a sign they never take. 'small' is the symmetric max-compression "
        "table, 'prd' the PRD table."
    ),
)
parser.add_argument(
    "--lcqat-k-map",
    type=str,
    default="",
    help="override K per module substring, e.g. 'mlp.c_proj:255/255'",
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
# Training horizon
parser.add_argument(
    "--num-iterations",
    type=int,
    default=-1,
    help="number of optimization steps (-1 = full epoch)",
)
# Batch sizes (default: inherit from pretrained checkpoint)
parser.add_argument(
    "--max-seq-len",
    type=int,
    default=None,
    help="max context length (default: inherit from pretrain)",
)
parser.add_argument(
    "--device-batch-size",
    type=int,
    default=None,
    help="per-device batch size (default: inherit from pretrain)",
)
parser.add_argument(
    "--total-batch-size",
    type=int,
    default=None,
    help="total batch size in tokens (default: inherit from pretrain)",
)
# Optimization (default: inherit from pretrained checkpoint)
parser.add_argument(
    "--embedding-lr",
    type=float,
    default=None,
    help="learning rate for embedding parameters (Adam) (default: inherit from pretrain)",
)
parser.add_argument(
    "--unembedding-lr",
    type=float,
    default=None,
    help="learning rate for unembedding parameters (Adam) (default: inherit from pretrain)",
)
parser.add_argument(
    "--matrix-lr",
    type=float,
    default=None,
    help="learning rate for matrix parameters (AdamW) (default: inherit from pretrain)",
)
parser.add_argument(
    "--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR"
)
parser.add_argument(
    "--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup"
)
parser.add_argument(
    "--warmdown-ratio",
    type=float,
    default=0.5,
    help="ratio of iterations for LR warmdown",
)
parser.add_argument(
    "--db-objective",
    type=str,
    default="edm",
    choices=["edm", "ce"],
    help=(
        "block-wise training objective. 'edm' (default) is the DiffusionBlocks "
        "method: only the active block's layers run, so activations are "
        "O(L/B). 'ce' is the escape hatch (full-depth next-token cross-entropy "
        "with block-isolated gradients). Should match the pretraining "
        "objective -- switching mid-pipeline changes what is being optimized."
    ),
)
parser.add_argument(
    "--db-overlap",
    type=float,
    default=0.1,
    help="log-sigma overlap between adjacent blocks (DiffusionBlocks App. C)",
)
parser.add_argument(
    "--db-block-sampling",
    type=str,
    default="step",
    choices=["step", "micro"],
    help="draw the active block once per optimizer step (default) or per micro-step",
)
parser.add_argument(
    "--db-blocks",
    type=int,
    default=4,
    help="number of diffusion blocks for block-wise training (default: 4)",
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
from nanochat.models.quant.pruning import (  # noqa: E402
    add_sparseprop_pruning_args,
    schedule_from_args,
)

add_sparseprop_pruning_args(parser)
# W6 DiffusionBlocks features (sigma codebooks / EfQAT per-block latching /
# denoiser KD) plus --lcqat-channel-center, registered through the same shared
# helper base_train uses so the three entry points cannot drift apart.
from nanochat.models.quant.w6 import (  # noqa: E402
    add_w6_args,
    build_denoiser_teacher,
    describe_sigma_codebooks,
    install_sigma_codebooks,
    make_latch_freezer,
)

add_w6_args(parser)
parser.add_argument(
    "--final-lr-frac",
    type=float,
    default=0.0,
    help="final LR as fraction of initial LR",
)
# Evaluation
parser.add_argument(
    "--eval-every",
    type=int,
    default=200,
    help="evaluate val bpb every N steps (-1 = disable)",
)
parser.add_argument(
    "--eval-tokens",
    type=int,
    default=40 * 524288,
    help="number of tokens to evaluate val loss on",
)
parser.add_argument(
    "--chatcore-every",
    type=int,
    default=200,
    help="evaluate ChatCORE metric every N steps (-1 = disable)",
)
parser.add_argument(
    "--chatcore-max-cat",
    type=int,
    default=-1,
    help="max problems per categorical task for ChatCORE",
)
parser.add_argument(
    "--chatcore-max-sample",
    type=int,
    default=24,
    help="max problems per generative task for ChatCORE",
)
# Data mixture
parser.add_argument(
    "--mmlu-epochs",
    type=int,
    default=3,
    help="number of epochs of MMLU in training mixture (teaches Multiple Choice)",
)
parser.add_argument(
    "--gsm8k-epochs",
    type=int,
    default=4,
    help="number of epochs of GSM8K in training mixture (teaches Math and Tool Use)",
)
args = parser.parse_args()
user_config = vars(args).copy()
# Shared SparseProp pruning schedule (scope / gradual ramp / dense threshold).
sparse_schedule = schedule_from_args(args)
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat-sft", name=args.run, config=user_config)
)

# Flash Attention status
if not HAS_FA3:
    print0(
        "WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient."
    )

# Load the model and tokenizer (LC-QAT configs are detected from the checkpoint meta)
lcqat_requested = lcqat_config_from_args(args) if args.lcqat else None
model, tokenizer, meta = load_model(
    "base",
    device,
    phase="train",
    model_tag=args.model_tag,
    step=args.model_step,
    lcqat=lcqat_requested,
)
lcqat_started_from_float = lcqat_requested is not None and meta.get("lcqat") is None
lcqat_meta = meta.get("lcqat")
if lcqat_meta is None and lcqat_requested is not None:
    lcqat_meta = lcqat_requested.as_dict()
if lcqat_requested is not None:
    print0(f"LC-QAT SFT: {retrofit_summary(model)}")

# Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len", 2048, meta),
    ("device_batch_size", 32, meta),
    ("total_batch_size", 524288, meta),
    ("embedding_lr", 0.3, pretrain_user_config),
    ("unembedding_lr", 0.004, pretrain_user_config),
    ("matrix_lr", 0.02, pretrain_user_config),
    ("db_blocks", 4, pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(
            f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}"
        )
    else:
        print0(f"Using {name}={arg_val}")

orig_model = model
if isinstance(model, DiffusionBlockEngine):
    engine = model
    base_model = engine.model
    num_db_blocks = engine.partitioner.num_blocks
else:
    base_model = model
    num_db_blocks = min(args.db_blocks, base_model.config.n_layer)
    partitioner = EquiProbabilityPartitioner(
        num_blocks=num_db_blocks,
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
    )
    engine = DiffusionBlockEngine(base_model, partitioner)

# LC-QAT: retrofit the engine-owned Linear layers (adapters + denoise heads)
# so the whole training pipeline is LC-QAT. The base model was already
# retrofitted by load_model/retrofit_model during checkpoint load.
lcqat_active = lcqat_requested if lcqat_started_from_float else lcqat_meta
if lcqat_active is not None and not isinstance(lcqat_active, LayerKConfig):
    lcqat_active = LayerKConfig.from_dict(lcqat_active)
if lcqat_active is not None:
    n_lcqat = engine.apply_lcqat(lcqat_active)
    print0(f"LC-QAT retrofitted {n_lcqat} diffusion-engine Linear layers")

# Sigma-conditioned activation codebooks (PRD 3.2). Must follow apply_lcqat --
# that is what creates the act quantizers to replace -- and precede SparseProp,
# which rewraps the same modules.
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

# SparseProp: unstructured sparsity with AVX2 sparse backprop.
if args.sparseprop:
    n_sparse = engine.apply_sparseprop(
        sparsity=args.sparseprop_sparsity,
        with_lcqat=lcqat_active is not None,
    )
    print0(f"SparseProp injected {n_sparse} sparse Linear layers")
    engine_achieved = sparse_schedule.apply(engine, 0)
    if engine_achieved is not None:
        print0(
            f"SparseProp pruned engine layers to {engine_achieved:.4f} sparsity "
            f"(scope={sparse_schedule.scope})"
        )

# Denoiser-distillation teacher (PRD 3.1, EDM form). Taken *here*, after the
# engine load but before the training loop, so the twin inherits the checkpoint's
# trained weights and `build_float_twin` strips the LC-QAT layers the retrofits
# above installed. On a fresh run the base model was never retrofitted either.
float_twin, n_twin_stripped = (None, 0)
if args.kd_denoiser_alpha > 0.0:
    float_twin, n_twin_stripped = build_denoiser_teacher(
        engine, args.kd_denoiser_alpha, args.db_objective
    )
    print0(
        f"KD denoiser twin: float copy of the engine "
        f"({n_twin_stripped} LC-QAT layers stripped)"
    )

# Denoiser distillation (PRD 3.1, EDM form): hand the frozen float twin to the
# engine, which adds the KD term to the EDM objective inside `denoise_step`.
# Defined unconditionally because the per-step logging reads it on every step,
# not only on the ones where the anchor is installed.
kd_denoiser = None
if float_twin is not None:
    from nanochat.models.quant.kd import DenoiserDistiller

    kd_denoiser = DenoiserDistiller(float_twin, alpha=args.kd_denoiser_alpha)
    engine.set_distiller(kd_denoiser)
    print0(
        f"KD denoiser distillation enabled: alpha={args.kd_denoiser_alpha}, "
        f"anchor = w(sigma)*||D_quant - D_float||^2 on the same noisy input"
    )

# EfQAT per-block permanent latching (PRD 3.2). `>=` at the latch site so a
# resumed run (which starts above the threshold) still latches; `latch_blocks`
# is idempotent so the re-fire is free.
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
    engine.set_freezer(block_latch_freezer)

depth = base_model.config.n_layer
num_flops_per_token = base_model.estimate_flops()
tokens_per_fwdbwd = (
    args.device_batch_size * args.max_seq_len
)  # tokens per iteration for a single rank
world_tokens_per_fwdbwd = (
    tokens_per_fwdbwd * ddp_world_size
)  # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0, (
    f"total_batch_size ({args.total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
)
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(
    f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}"
)
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(
    f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}"
)
if args.db_block_sampling == "micro" and grad_accum_steps > 1:
    print0(
        "WARNING: --db-block-sampling micro with gradient accumulation is LOSSY. "
        "`_apply_requires_grad` clears p.grad for parameters the active block "
        "does not own, so each micro-step erases the previous one's gradients. "
        "Use the default 'step' unless reproducing the ablation."
    )
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer (AdamW-only for DiffusionBlocks engine)
# PRD section 5: codebook delta params get their own AdamW group with a
# dedicated LR and zero weight decay, distinct from the matrix-weight group.
from nanochat.models.quant.optimizer import build_qat_param_groups, verify_partition

param_groups = build_qat_param_groups(
    engine,
    matrix_lr=args.matrix_lr if args.matrix_lr is not None else 3e-4,
    weight_decay=0.0,
    codebook_lr=args.codebook_lr if args.codebook_lr is not None else 1e-3,
    matrix_betas=(0.8, 0.95),
    matrix_eps=1e-10,
)
# Fail here rather than shipping a run where a whole role silently never trains.
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

# Optionally warm-start optimizer from pretrained checkpoint (momentum buffers etc.)
# Note: load_state_dict overwrites param_group metadata (LRs, betas, etc.) with the
# pretrained values. Since pretraining warmdown brings LRs to ~0, we must save and
# restore our fresh SFT LRs after loading.
base_dir = get_base_dir()
if args.load_optimizer and lcqat_started_from_float:
    # Retrofit added a codebook param group that the float pretrain optimizer
    # state does not have; warm-starting would fail the group-size check.
    print0(
        "LC-QAT starting from float checkpoint: using fresh optimizer (no warm-start)"
    )
elif args.load_optimizer:
    optimizer_data = load_optimizer_state(
        "base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step
    )
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        loaded_momentum = False
        try:
            optimizer.load_state_dict(optimizer_data)
            loaded_momentum = True
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
                # The saved and current models have a different number of
                # parameters (different architecture/stage, e.g. a Muon-based
                # pretrain checkpoint vs. the current AdamW-only model). The
                # saved state's integer param indices map onto the old flat
                # param list, not the current Parameter objects, so any
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
                    id(p): p
                    for group in optimizer.param_groups
                    for p in group["params"]
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
                    f"from pretrained checkpoint (group layout mismatch; LRs reset)"
                )
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        if loaded_momentum:
            print0(
                "Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)"
            )
    else:
        print0(
            "WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)"
        )

# GradScaler for fp16 training (bf16/fp32 don't need it)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# Override the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# SFT data mixture and DataLoader
train_tasks = [
    SmolTalk(split="train"),  # 460K rows of general conversations
    *[
        MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)
    ],  # 100K rows per epoch
    *[
        GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)
    ],  # 8K rows per epoch
]
train_dataset = TaskMixture(train_tasks)
print0(
    f"Training mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})"
)
val_dataset = TaskMixture(
    [
        SmolTalk(split="test"),  # 24K rows in test set
        MMLU(
            subset="all", split="test", stop=5200
        ),  # 14K rows in test set, use only 5.2K to match the train ratios
        GSM8K(
            subset="main", split="test", stop=420
        ),  # 1.32K rows in test set, use only 420 to match the train ratios
    ]
)  # total: 24K + 5.2K + 0.42K ~= 29.6K rows
# DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
# A big problem is that we don't know the final num_iterations in advance. So we create
# these two global variables and update them from within the data generator.
last_step = (
    False  # we will toggle this to True when we reach the end of the training dataset
)
approx_progress = 0.0  # will go from 0 to 1 over the course of the epoch
current_epoch = 1  # track epoch for logging


def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    BOS-aligned dataloader for SFT with bestfit-pad packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Padding positions have targets masked with -1 (ignore_index for cross-entropy).
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()

    # Conversation buffer: list of (token_ids, loss_mask) tuples
    conv_buffer = []
    cursor = ddp_rank  # Each rank processes different conversations (for fetching)
    consumed = ddp_rank  # Track actual consumption separately from buffering
    epoch = 1
    it = 0  # iteration counter

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
                # Note: last_step is now triggered based on consumption, not fetching

    while True:
        rows = []
        mask_rows = []
        row_lengths = []  # Track actual content length (excluding padding) for each row
        for _ in range(args.device_batch_size):
            row = []
            mask_row = []
            padded = False
            while len(row) < row_capacity:
                # Ensure buffer has conversations
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                # Find largest conversation that fits entirely
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    # Found a conversation that fits - use it entirely
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size  # Track actual consumption
                else:
                    # No conversation fits - pad the remainder instead of cropping
                    # This ensures we never discard any tokens
                    content_len = len(row)
                    row.extend([bos_token] * remaining)  # Pad with BOS tokens
                    mask_row.extend([0] * remaining)
                    padded = True
                    break  # Row is now full (with padding)

            # Track content length: full row if no padding, otherwise the length before padding
            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        # Stopping condition to respect num_iterations, if given
        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True

        # Update progress tracking (based on consumed, not cursor, to account for buffering)
        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                approx_progress = it / args.num_iterations
            else:
                approx_progress = consumed / dataset_size
            # Trigger last_step when we've consumed enough (instead of when cursor wraps)
            if consumed >= dataset_size:
                last_step = True

        # Build tensors
        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = (
            batch_tensor[:, :-1]
            .to(device=device, dtype=torch.int32, non_blocking=use_cuda)
            .contiguous()
        )
        targets = (
            batch_tensor[:, 1:]
            .to(device=device, dtype=torch.int64, non_blocking=use_cuda)
            .contiguous()
        )

        # Apply the loss mask from render_conversation (mask=1 for assistant completions,
        # mask=0 for user prompts, BOS, special tokens, tool outputs). mask[1:] aligns
        # with targets (shifted by 1). Unmasked positions get -1 (ignore_index).
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1

        # Mask out padding positions in targets (set to -1 = ignore_index)
        # For each row, positions >= (content_length - 1) in targets should be masked
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1 :] = -1

        yield inputs, targets


train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0  # will go from 0 to 1 over the course of the epoch


# Learning rate schedule (linear warmup, constant, linear warmdown)
# Same shape as base_train but uses progress (0→1) instead of absolute step counts,
# because SFT doesn't always know num_iterations in advance (dataset-driven stopping).
def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac


# -----------------------------------------------------------------------------
# Training loop
x, y = next(train_loader)  # prefetch the very first batch of data
min_val_bpb = float("inf")
smooth_train_loss = 0  # EMA of training loss
ema_beta = 0.9  # EMA decay factor
total_training_time = 0  # total wall-clock time of training
step = 0
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (
            args.device_batch_size * args.max_seq_len * ddp_world_size
        )
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
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

    # once in a while: estimate the ChatCORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    chatcore_results = {}
    if args.chatcore_every > 0 and (
        last_step or (step > 0 and step % args.chatcore_every == 0)
    ):
        model.eval()
        # ar_engine is the autoregressive sampler; it must not shadow the
        # DiffusionBlockEngine bound to `engine`, which train_step runs on.
        ar_engine = Engine(orig_model, tokenizer)
        all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval"]
        categorical_tasks = {"ARC-Easy", "ARC-Challenge", "MMLU"}
        baseline_accuracies = {
            "ARC-Easy": 0.25,
            "ARC-Challenge": 0.25,
            "MMLU": 0.25,
            "GSM8K": 0.0,
            "HumanEval": 0.0,
        }
        task_results = {}
        for task_name in all_tasks:
            limit = (
                args.chatcore_max_cat
                if task_name in categorical_tasks
                else args.chatcore_max_sample
            )
            max_problems = None if limit < 0 else limit  # -1 means no limit
            acc = run_chat_eval(
                task_name,
                orig_model,
                tokenizer,
                ar_engine,
                batch_size=args.device_batch_size,
                max_problems=max_problems,
            )
            task_results[task_name] = acc
            print0(f"  {task_name}: {100 * acc:.2f}%")

        # Compute ChatCORE metrics (mean centered accuracy, ranges from 0=random to 1=perfect)
        def centered_mean(tasks):
            return sum(
                (task_results[t] - baseline_accuracies[t])
                / (1.0 - baseline_accuracies[t])
                for t in tasks
            ) / len(tasks)

        chatcore = centered_mean(all_tasks)
        chatcore_cat = centered_mean(categorical_tasks)
        print0(
            f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}"
        )
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "chatcore_metric": chatcore,
                "chatcore_cat": chatcore_cat,
                **{
                    f"chatcore/{task_name}": acc
                    for task_name, acc in task_results.items()
                },
            }
        )
        model.train()

    # save checkpoint at the end of the run (all ranks participate so each saves its optimizer shard)
    if last_step:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"  # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            # engine.state_dict(), not orig_model.state_dict(): the engine owns
            # db_adapters.* / db_denoise_heads.*, and this checkpoint declares
            # meta["db"] below. Saving the bare model would write a checkpoint
            # that claims a diffusion engine but carries none of its parameters.
            engine.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb,  # loss at last step
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config,  # inputs to the training script
                "lcqat": lcqat_meta,
                "db": {
                    "num_blocks": num_db_blocks,
                    "sigma_min": 0.002,
                    "sigma_max": 80.0,
                    "sigma_data": 0.5,
                    "sigma_codebook": describe_sigma_codebooks(args),
                },
                "sparseprop": {
                    "enabled": args.sparseprop,
                    "sparsity": args.sparseprop_sparsity,
                    "scope": sparse_schedule.scope,
                    "start_frac": sparse_schedule.start_frac,
                    "every": sparse_schedule.every,
                    "ramp_steps": sparse_schedule.ramp_steps,
                    "dense_threshold": sparse_schedule.dense_threshold,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    # EfQAT per-block latch (PRD 3.2). Fires before the forward, so the retired
    # block is already frozen in the very step the latch lands; latching after
    # the forward would let one more update through.
    if block_latch_freezer is not None and step >= max(args.efqat_latch_after, 0):
        n_new = block_latch_freezer.latch_blocks(efqat_latch_targets)
        if n_new:
            print0(
                f"EfQAT latched blocks {efqat_latch_targets} permanently at step "
                f"{step} ({n_new} parameters frozen; total latched: "
                f"{block_latch_freezer.latched_blocks()})"
            )
    # Gradual magnitude pruning, before the forward so the mask and the values
    # agree in the same step. Monotone: a pruned weight is exactly 0.0 and can
    # never win the |W| ranking again.
    if args.sparseprop:
        pruned = sparse_schedule.apply(engine, step)
        if pruned is not None:
            print0(
                f"Step {step:05d} | SparseProp pruned to {pruned:.4f} sparsity; "
                f"{len(sparse_schedule.layers_above_threshold(engine))} layers "
                f"past the {sparse_schedule.dense_threshold:.0%} threshold"
            )
    # Block drawn once per optimizer step (see base_train: per-micro-step
    # sampling would turn block-wise training into ordinary gradient
    # accumulation). The diffusion target is hoisted for the same reason.
    block_idx = (
        engine.sample_block()
        if args.db_objective == "edm" and args.db_block_sampling == "step"
        else None
    )
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
            loss, _ = engine.denoise_step(
                x, block_idx=block_idx, overlap=args.db_overlap, clean=clean
            )
            if engine.distiller is not None:
                step_kd_logged = step_kd_logged + float(engine.last_kd_loss)
        else:
            loss = engine.train_step(x, y)
        train_loss = loss.detach()  # for logging
        loss = (
            loss / grad_accum_steps
        )  # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y = next(
            train_loader
        )  # prefetch the next batch while the GPU is busy with forward/backward
        progress = max(
            progress, approx_progress
        )  # only increase progress monotonically
    # step the optimizer
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1

    # logging
    smooth_train_loss = (
        ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    )  # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (
        1 - ema_beta ** (step + 1)
    )  # debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    print0(
        f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time / 60:.2f}m"
    )
    if step % 10 == 0:
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
                "train/epoch": current_epoch,
            }
        )

    # The garbage collector spends ~500ms scanning for cycles quite frequently.
    # We manually manage it to avoid these pauses during training.
    if step == 1:
        gc.collect()  # manually collect a lot of garbage from setup
        gc.freeze()  # freeze all currently surviving objects and exclude them from GC
        gc.disable()  # disable GC entirely except:
    elif step % 5000 == 0:  # every 5000 steps...
        gc.collect()  # manually collect, just to be safe for very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# cleanup
wandb_run.finish()  # wandb run finish
compute_cleanup()
