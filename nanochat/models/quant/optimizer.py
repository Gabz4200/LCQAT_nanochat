"""
LC-QAT-aware optimizer builder (PRD section 5: codebook params get their own
AdamW group with a dedicated LR and no weight decay).

`base_train`/`chat_sft`/`chat_rl` operate on the `DiffusionBlockEngine` (model +
adapters + per-block denoise heads), not the bare GPT. This module partitions the
*whole* engine tree by parameter-name role. It replaces `GPT.setup_optimizer`,
which was deleted: that method only knew about `transformer.h`, so it silently
dropped the engine's adapters and denoise heads, and it ignored the
`--embedding-lr` / `--unembedding-lr` / `--scalar-lr` family that `base_train`
parses.

Role assignment is a partition: every parameter lands in exactly one group, and
`verify_partition` asserts it. That assertion is the point of the module --
`base_train` accepted `--embedding-lr` / `--unembedding-lr` / `--scalar-lr` while
the previous two-group builder silently discarded all three, so embeddings and
scalars silently trained at the matrix LR.
"""

from __future__ import annotations

import torch.nn as nn

#: Parameter-name suffixes identifying a codebook step parameter. Stable
#: contract shared with `nanochat.models.backbone._is_codebook_param` and relied on by
#: SparseProp's re-parenting, which must survive module surgery.
CODEBOOK_SUFFIXES = ("raw_pos_deltas", "raw_neg_deltas")

#: Suffixes identifying a *learned activation-LUT* parameter
#: (`LearnableIndexLut.logits`, or its proximity form's `knots`/`levels`).
#:
#: These are table step parameters, not matrix weights, so they belong to the
#: codebook group -- their own LR, and no weight decay. Left out of
#: `CODEBOOK_SUFFIXES` deliberately: that frozenset is the PRD 5 *name
#: contract* that SparseProp's re-parenting and `retrofit.is_lcqat_state` both
#: key on to decide whether a checkpoint is LC-QAT at all. Widening it would
#: make every model carrying a learnable LUT report itself as quantized even
#: with no codebook parameters at all.
LEARNED_LUT_SUFFIXES = (
    "learnable_activation_lut.logits",
    "learnable_activation_lut.knots",
    "learnable_activation_lut.levels",
)

#: Scalar / gate parameter names, by role. Matched exactly: a `startswith` here
#: would swallow unrelated submodules.
_SCALAR_ROLES: dict[str, str] = {
    "resid_lambdas": "resid",
    "x0_lambdas": "x0",
    "smear_gate.weight": "smear",
    "smear_lambda": "smear",
    "backout_lambda": "smear",
}

# Per-role optimizer hyperparameters, ported from the (now deleted)
# `GPT.setup_optimizer` so the
# tuned recipe carries over unchanged. `matrix` takes the caller's values.
_ROLE_BETAS: dict[str, tuple[float, float]] = {
    "lm_head": (0.8, 0.96),
    "embed": (0.8, 0.995),
    "value_embed": (0.8, 0.995),
    "resid": (0.8, 0.95),
    "x0": (0.96, 0.95),
    "smear": (0.8, 0.95),
    "codebook": (0.8, 0.95),
}
_ROLE_WD: dict[str, float] = {
    "lm_head": 0.01,
    "embed": 0.001,
    "value_embed": 0.01,
    "resid": 0.05,
    "x0": 0.0,
    "smear": 0.0,
    "codebook": 0.0,  # PRD 5: codebook steps get no weight decay
}

#: Group order in the emitted list, so the layout is deterministic and
#: checkpoint optimizer warm-starts stay comparable across runs.
ROLE_ORDER = (
    "lm_head",
    "embed",
    "value_embed",
    "resid",
    "x0",
    "smear",
    "codebook",
    "matrix",
)


def is_codebook_param(name: str) -> bool:
    """True if `name` is a codebook step parameter (PRD 5 name contract)."""
    return name.endswith(CODEBOOK_SUFFIXES)


def is_learned_lut_param(name: str) -> bool:
    """True if `name` is a learned activation-LUT table parameter."""
    return name.endswith(LEARNED_LUT_SUFFIXES)


def role_for_name(name: str) -> str:
    """Return the optimizer role for a fully-qualified parameter name.

    Order matters: codebook params live *inside* `transformer.h`, so the codebook
    check must precede the matrix catch-all.
    """
    if is_codebook_param(name) or is_learned_lut_param(name):
        return "codebook"
    if name in _SCALAR_ROLES:
        return _SCALAR_ROLES[name]
    if name.startswith("lm_head."):
        return "lm_head"
    if name.startswith("transformer.wte."):
        return "embed"
    if name.startswith("value_embeds."):
        return "value_embed"
    # Everything else: transformer block weights, engine-owned adapters and
    # denoise heads, and any module added later under the model tree.
    return "matrix"


def build_qat_param_groups(
    model: nn.Module,
    matrix_lr: float,
    weight_decay: float,
    codebook_lr: float = 1e-3,
    matrix_betas: tuple[float, float] = (0.8, 0.95),
    matrix_eps: float = 1e-10,
    embedding_lr: float = 0.3,
    unembedding_lr: float = 0.008,
    scalar_lr: float = 0.5,
    dmodel_lr_scale: float = 1.0,
) -> list[dict]:
    """Build AdamW param groups over the whole engine tree, partitioned by role.

    Works on a bare `GPT` or on a `DiffusionBlockEngine` (anything exposing
    `named_parameters()`), which is the point: the method this replaces only ever
    saw `transformer.h`, so the engine's adapters and denoise heads fell outside
    every group, and `--embedding-lr` / `--unembedding-lr` / `--scalar-lr` were
    parsed and then discarded.

    Per-role betas and weight decay are ported from the tuned recipe so they
    carry over unchanged. LRs are the caller's, scaled by `dmodel_lr_scale`
    (which `base_train` computes as `(n_embd / 768) ** -0.5`).

    Returns groups in `ROLE_ORDER`, each tagged with `role=` so the layout is
    inspectable and a missing role is visible rather than silent.
    """
    by_role: dict[str, list[nn.Parameter]] = {role: [] for role in ROLE_ORDER}
    for name, p in model.named_parameters():
        by_role[role_for_name(name)].append(p)

    lrs = {
        "lm_head": unembedding_lr * dmodel_lr_scale,
        "embed": embedding_lr * dmodel_lr_scale,
        "value_embed": embedding_lr * dmodel_lr_scale * 0.5,
        "resid": scalar_lr * 0.01,
        "x0": scalar_lr,
        "smear": 0.2,
        "codebook": codebook_lr * dmodel_lr_scale,
        "matrix": matrix_lr,
    }
    groups = []
    for role in ROLE_ORDER:
        params = by_role[role]
        if not params:
            continue
        groups.append(
            dict(
                kind="adamw",
                role=role,
                params=params,
                lr=lrs[role],
                betas=(
                    matrix_betas
                    if role == "matrix"
                    else _ROLE_BETAS.get(role, matrix_betas)
                ),
                eps=matrix_eps,
                weight_decay=(
                    weight_decay if role == "matrix" else _ROLE_WD.get(role, 0.0)
                ),
            )
        )
    return groups


def verify_partition(model: nn.Module, groups: list[dict]) -> None:
    """Assert every model parameter appears in exactly one group.

    Guards the failure mode this module exists to prevent: a group builder that
    silently omits a role, leaving those parameters untrained with no error.
    Duplicates are equally fatal -- AdamW raises on those, but only once the
    optimizer is constructed, which is later than it should be caught.
    """
    seen: dict[int, str] = {}
    for group in groups:
        role = group.get("role", "<unlabelled>")
        for p in group["params"]:
            pid = id(p)
            if pid in seen:
                raise AssertionError(
                    f"parameter is in two optimizer groups ({seen[pid]} and "
                    f"{role}); AdamW would reject this"
                )
            seen[pid] = role
    missing = [name for name, p in model.named_parameters() if id(p) not in seen]
    if missing:
        raise AssertionError(
            f"{len(missing)} parameter(s) are in no optimizer group, e.g. "
            f"{missing[:5]}; they would never receive an update"
        )
