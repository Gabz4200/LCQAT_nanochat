"""
Utilities for saving and loading model/optim/state checkpoints.
"""

import json
import logging
import os
import re

import torch

from nanochat.common import COMPUTE_DTYPE, get_base_dir, setup_default_logging
from nanochat.diffusion_blocks import DiffusionBlockEngine, EquiProbabilityPartitioner
from nanochat.gpt import GPT, GPTConfig
from nanochat.lcqat.retrofit import (
    LayerKConfig,
    finish_lcqat_after_load,
    is_exported_lcqat_state,
    prepare_lcqat_before_load,
)
from nanochat.tokenizer import get_tokenizer

setup_default_logging()
logger = logging.getLogger(__name__)


def log0(message):
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(message)


def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        log0("Patching missing window_pattern in model config to 'L'")


def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0("Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0("Patching missing x0_lambdas in model data to 0.0")


def save_checkpoint(
    checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0
):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(
            checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt"
        )
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(
            checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt"
        )
        optimizer_data = torch.load(optimizer_path, map_location=device)
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase, lcqat=None):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training

    `lcqat` optionally requests starting QAT from a float checkpoint: a
    LayerKConfig instance (or its dict form). LC-QAT checkpoints are detected
    from their state keys and retrofitted automatically with the config saved
    in their meta (falls back to `lcqat`, then the small preset).
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    if lcqat is not None and not isinstance(lcqat, LayerKConfig):
        lcqat = LayerKConfig.from_dict(lcqat)
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, step, device, load_optimizer=False
    )
    if is_exported_lcqat_state(model_data) and phase == "train":
        raise RuntimeError(
            "exported LC-QAT artifact is inference-only (weights stripped); "
            "cannot train or resume from it - load a training checkpoint instead"
        )
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()  # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    # Check if DiffusionBlocks state was saved in metadata or state keys
    # Do this BEFORE the strict load so malformed db checkpoints fail with
    # the intended message instead of an incidental LUT key error.
    db_meta = meta_data.get("db")
    if db_meta is not None:
        # The engine's own parameters (adapters, denoise heads) are zero-initialized
        # at construction (diffusion_blocks.py NoiseConditionedBlockAdapter /
        # denoise_head). A checkpoint that declares meta["db"] but carries no
        # db_* keys therefore loads a *silently zero* engine. That is always a
        # bug in the writer, so fail loudly instead of evaluating garbage.
        # Two required families, each satisfied by either the current or the
        # legacy key prefix: `db_denoise_heads.{b}.*` (per-block, W1.3) or
        # `db_denoise_head.*` (the single shared head that preceded it).
        families = {
            "db_adapters.": ("db_adapters.",),
            "db_denoise_head": ("db_denoise_heads.", "db_denoise_head."),
        }
        missing = [
            family
            for family, prefixes in families.items()
            if not any(k.startswith(p) for p in prefixes for k in model_data)
        ]
        if missing:
            raise RuntimeError(
                f"checkpoint declares meta['db'] but its state_dict is missing "
                f"{missing} keys: the DiffusionBlocks engine would load with "
                "zero-initialized adapters/denoise heads. This checkpoint was "
                "written with `orig_model.state_dict()` instead of "
                "`engine.state_dict()`; retrain or re-save it."
            )

    # Early legacy-head check: if meta has num_blocks > 1 but state has the
    # legacy single-head prefix (db_denoise_head.) and NOT the per-block
    # prefix (db_denoise_heads.), fail with the intended message before
    # the strict base load (which would complain about LUT keys instead).
    if db_meta is not None:
        num_blocks = db_meta.get("num_blocks", 4)
        has_legacy_head = any(k.startswith("db_denoise_head.") for k in model_data)
        has_per_block_heads = any(k.startswith("db_denoise_heads.") for k in model_data)
        if num_blocks > 1 and has_legacy_head and not has_per_block_heads:
            raise RuntimeError(
                f"checkpoint declares meta['db'] with num_blocks={num_blocks} "
                "but state_dict has legacy single-head keys (db_denoise_head.) "
                "and no per-block heads (db_denoise_heads.). A legacy checkpoint "
                "with one head cannot be split across multiple blocks; retrain "
                "or re-save it with the current format."
            )

    # LC-QAT: retrofit before load when the checkpoint already has codebooks,
    # otherwise retrofit after load only when QAT start was requested
    lcqat_active = prepare_lcqat_before_load(
        model, model_data, meta_data.get("lcqat"), lcqat
    )
    # SparseProp must also be injected BEFORE the load, for the same reason
    # LC-QAT is: the checkpoint carries `sparsity_mask` / `w_ptr` / `w_col` /
    # `w_ptr_csc` / `w_row` as persistent buffers, so loading into a model that
    # was not injected fails with "unexpected key(s)". The training entry points
    # do this themselves, but they inject *before* calling build_model; an eval
    # or inference path that goes straight to load_model did not, so a model
    # trained with the always-on SparseProp default could not be loaded for
    # evaluation at all. Rebuild it here from meta["sparseprop"], which records
    # the settings the checkpoint was written with.
    sparseprop_meta = meta_data.get("sparseprop")
    if sparseprop_meta and sparseprop_meta.get("enabled"):
        from nanochat.lcqat.sparseprop import inject_sparseprop_layers

        # No pruning: the mask built here is a placeholder that the strict load
        # immediately overwrites with the checkpoint's own.
        inject_sparseprop_layers(
            model,
            sparsity=float(sparseprop_meta.get("sparsity", 0.75)),
            with_lcqat=lcqat_active is not None,
        )
    # Strip db_ keys for base model load
    base_model_data = {
        k: v
        for k, v in model_data.items()
        if not k.startswith("db_adapters.")
        and not k.startswith("db_denoise_heads.")
        and not k.startswith("db_denoise_head.")
    }
    model.load_state_dict(base_model_data, strict=True, assign=True)
    if lcqat_active is None:
        finish_lcqat_after_load(model, lcqat)

    # DiffusionBlocks engine construction (top level, only when db_meta exists)
    if db_meta is not None:
        num_blocks = db_meta.get("num_blocks", 4)
        sigma_min = db_meta.get("sigma_min", 0.002)
        sigma_max = db_meta.get("sigma_max", 80.0)
        sigma_data = db_meta.get("sigma_data", 0.5)
        partitioner = EquiProbabilityPartitioner(
            num_blocks=num_blocks,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
        )
        engine = DiffusionBlockEngine(
            model, partitioner, device=device, dtype=COMPUTE_DTYPE
        )
        engine.load_state_dict(model_data, strict=False)
        target_model = engine
    else:
        target_model = model

    # Put the model in the right training phase / mode
    if phase == "eval":
        target_model.eval()
    else:
        target_model.train()
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], (
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    )
    return target_model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [
        f
        for f in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, f))
    ]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True
    )
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if re.search(r"model_(\d+)\.pt$", f)
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = max(int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files)
    return last_step


def load_model_from_dir(
    checkpoints_dir, device, phase, model_tag=None, step=None, lcqat=None
):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(
        checkpoint_dir, step, device, phase, lcqat=lcqat
    )
    return model, tokenizer, meta_data


def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)


def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    model_dir = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    if not os.path.exists(optimizer_path):
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        return None
    log0(f"Loading optimizer state from {optimizer_path}")
    optimizer_data = torch.load(optimizer_path, map_location=device)
    return optimizer_data
