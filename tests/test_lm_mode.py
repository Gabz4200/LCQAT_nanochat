"""`--db-blocks=0`: plain autoregressive LM training, with no DiffusionBlocks.

The mode exists because `--db-objective ce` is *not* a DiffusionBlocks-off
baseline. `ce` still routes the forward through `DiffusionBlockEngine`, so
`train_step` calls `_activate_block`, which samples a block and leaves gradients
enabled for that block's layers only. Every step therefore still trains 1/B of
the network. A comparison that used `ce` as its "no DiffusionBlocks" arm would
have been comparing one-block-against-one-block and calling it a baseline.

These tests pin the two properties that make the mode usable as a baseline:

1. `--db-blocks=0` trains every layer on every step, with no engine built.
2. The resulting checkpoint is loadable by `base_eval` as a bare `GPT`, because
   `meta["db"]` is written as `None` and the loader gates engine reconstruction
   on that key's presence.
"""

import os

import pytest
import torch

from tests.conftest import run_base_train


def _run(extra, tag, timeout=900):
    """A per-test checkpoint directory, so runs cannot resume each other's."""
    return run_base_train(extra, model_tag=tag, timeout=timeout)


def _assert_ok(proc):
    assert proc.returncode == 0, (
        f"base_train exited {proc.returncode}\n"
        f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-3000:]}"
    )


@pytest.mark.slow
def test_when_db_blocks_zero_then_no_engine_is_built():
    """The engine must not exist at all, not exist degenerately.

    A one-block engine would satisfy "no diffusion blocks are active" while
    still routing through `denoise_step`, still sampling a block, and still
    holding denoise heads in the optimizer. The mode has to skip construction.
    """
    proc = _run(["--db-blocks=0", "--no-sparseprop"], "lmmode-test-noengine")
    _assert_ok(proc)
    assert "Initialized DiffusionBlocks Engine" not in proc.stdout, (
        "an engine was constructed despite --db-blocks=0"
    )
    assert "DiffusionBlocks disabled" in proc.stdout
    assert "step 00001" in proc.stdout, "training loop did not run"


@pytest.mark.slow
def test_when_db_blocks_zero_then_every_layer_trains():
    """No block isolation: every transformer layer must receive a gradient.

    This is the property that distinguishes the mode from `--db-objective ce`.
    It is checked by comparing against the block-isolated run rather than by
    inspecting the training loop, so it fails if the isolation ever leaks back
    in through the shared code path.
    """
    from nanochat.models.backbone import GPT, GPTConfig
    from nanochat.models.quant.retrofit import PRESETS, retrofit_model

    config = GPTConfig(
        sequence_len=64,
        vocab_size=512,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="L",
    )
    model = GPT(config)
    retrofit_model(model, PRESETS["asym"])
    idx = torch.randint(0, config.vocab_size, (2, 8))
    loss = model(idx, targets=idx)
    loss.backward()

    missing = [
        name
        for name, p in model.named_parameters()
        if name.startswith("transformer.h.") and p.grad is None
    ]
    assert not missing, f"layers with no gradient in LM mode: {missing}"


@pytest.mark.slow
def test_when_db_blocks_zero_then_checkpoint_meta_db_is_none():
    """`meta["db"]` must be None so the loader skips engine reconstruction.

    `build_model` treats the *presence* of `meta["db"]` as "this checkpoint has
    a diffusion engine" and builds one. A zeroed-but-present dict would make it
    build an engine whose adapters and denoise heads are absent from the
    state_dict, which is the silent-zero failure the loader guards against.
    """
    tag = "lmmode-test-meta"
    proc = _run(["--db-blocks=0", "--no-sparseprop"], tag)
    _assert_ok(proc)

    from nanochat.modules.checkpoint_manager import load_checkpoint
    from nanochat.utils.common import get_base_dir

    ckpt_dir = os.path.join(get_base_dir(), "base_checkpoints", tag)
    assert os.path.isdir(ckpt_dir), f"no checkpoint directory at {ckpt_dir}"
    # Through load_checkpoint rather than torch.load on a glob: the directory
    # also holds optim_*.pt, which sorts after model_*.pt, so a `sorted(...)[-1]`
    # reads the optimizer state and reports its keys instead of the
    # checkpoint's.
    model_data, _optim, meta = load_checkpoint(ckpt_dir, 2, "cpu", load_optimizer=False)
    assert "db" in meta, "meta has no 'db' key at all"
    assert meta["db"] is None, f"meta['db'] is {meta['db']!r}, expected None"
    # And no engine parameters leaked into the state_dict.
    db_keys = [k for k in model_data if k.startswith("db_")]
    assert not db_keys, f"engine keys in an LM-mode checkpoint: {db_keys[:5]}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "flag,value",
    [
        ("--db-sigma-codebook", "conditioned"),
        ("--kd-denoiser-alpha", "0.5"),
        ("--efqat-latch-blocks", "1"),
    ],
)
def test_when_denoiser_flag_with_db_blocks_zero_then_exits_nonzero(flag, value):
    """Denoiser-only flags must be rejected, not silently ignored.

    Silently ignoring one would produce a run that looks like the flag was
    tested and produces the same numbers as not passing it at all, which is the
    failure mode the ablation harness is supposed to prevent.
    """
    proc = _run(["--db-blocks=0", flag, value], "lmmode-test-reject")
    assert proc.returncode != 0, (
        f"{flag}={value} was silently accepted with --db-blocks=0"
    )
    assert "meaningless with --db-blocks<=0" in proc.stdout + proc.stderr or (
        "requires diffusion blocks" in proc.stdout + proc.stderr
    ), f"rejection did not explain the conflict; got:\n{proc.stdout[-1500:]}"


@pytest.mark.slow
def test_when_db_blocks_zero_with_sparseprop_then_sparsity_is_applied():
    """SparseProp must actually prune in LM mode, and say so.

    The masking happens inside `inject_sparseprop_layers`, not in the gradual
    schedule, so with the default `--sparseprop-every=0` the schedule declines
    to prune and there is no "pruned to ..." line. Without an explicit report the
    run is indistinguishable from one where SparseProp silently did nothing.
    """
    proc = _run(["--db-blocks=0"], "lmmode-test-sparse")
    _assert_ok(proc)
    assert "SparseProp injected" in proc.stdout, (
        "SparseProp did not report any injected layers in LM mode"
    )
    assert "exact zeros" in proc.stdout, (
        "SparseProp did not report the masked fraction in LM mode"
    )
