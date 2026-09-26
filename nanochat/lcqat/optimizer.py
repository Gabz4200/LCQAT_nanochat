"""
LC-QAT-aware optimizer builder (PRD section 5: codebook params get their own
AdamW group with a dedicated LR and no weight decay).

`base_train`/`chat_sft`/`chat_rl` operate on the DiffusionBlockEngine (model +
adapters + denoise_head), not the bare GPT. The canonical `GPT.setup_optimizer`
only knows about `transformer.h` params, so this helper scans the *whole*
module tree and splits any `raw_pos_deltas` / `raw_neg_deltas` parameters out
into a dedicated codebook group, leaving every other parameter in the matrix
group.
"""

from __future__ import annotations

import torch.nn as nn

CodebookParam = object  # sentinel for group tagging


def split_codebook_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition `model.parameters()` into (matrix_params, codebook_params).

    A parameter is a codebook step parameter iff its name ends with
    `raw_pos_deltas` or `raw_neg_deltas` (the stable name contract from
    `nanochat.gpt._is_codebook_param`).
    """
    matrix_params: list[nn.Parameter] = []
    codebook_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if name.endswith(("raw_pos_deltas", "raw_neg_deltas")):
            codebook_params.append(p)
        else:
            matrix_params.append(p)
    return matrix_params, codebook_params


def build_qat_param_groups(
    model: nn.Module,
    matrix_lr: float,
    weight_decay: float,
    codebook_lr: float = 1e-3,
    matrix_betas: tuple[float, float] = (0.8, 0.95),
    matrix_eps: float = 1e-10,
) -> list[dict]:
    """Build AdamW param groups that separate codebook deltas from matrices.

    The codebook group uses a dedicated LR with zero weight decay (PRD 5);
    the matrix group keeps the caller's LR / weight decay. Other per-role
    groups (embeddings, scalars, denoise_head) are left to the caller; this
    helper only enforces the codebook/matrix split that LC-QAT requires.
    """
    matrix_params, codebook_params = split_codebook_params(model)
    if not codebook_params:
        # No LC-QAT retrofit; fall back to a single matrix group.
        return [
            dict(
                kind="adamw",
                params=matrix_params,
                lr=matrix_lr,
                betas=matrix_betas,
                eps=matrix_eps,
                weight_decay=weight_decay,
            )
        ]
    return [
        dict(
            kind="adamw",
            params=matrix_params,
            lr=matrix_lr,
            betas=matrix_betas,
            eps=matrix_eps,
            weight_decay=weight_decay,
        ),
        dict(
            kind="codebook",
            params=codebook_params,
            lr=codebook_lr,
            betas=matrix_betas,
            eps=matrix_eps,
            weight_decay=0.0,
        ),
    ]
