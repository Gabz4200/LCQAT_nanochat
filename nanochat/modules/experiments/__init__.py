"""The nine paired LC-QAT ablation experiments, one module per experiment.

`scripts/lcqat_ablation.py` owns the CLI surface, the claim checks and the
leaderboard writer. Every measurement lives here and is re-exported below, so
`from nanochat.modules.experiments import run_act_lut` and the driver's own
`from scripts.lcqat_ablation import run_act_lut` both resolve to the same
function object.
"""

from __future__ import annotations

from nanochat.modules.experiments.act_lut import (
    K_MATCHED_BUDGET,
    SMOOTHPWL_DEFAULT_INIT_RANGE,
    build_activation_lut,
    matched_budget_codebooks,
    piecewise_linear,
    run_act_lut,
)
from nanochat.modules.experiments.asym_vs_small import run_asym_vs_small
from nanochat.modules.experiments.bias_quant import (
    BIAS_CODEBOOK_DEAD,
    BIAS_CODEBOOK_LIVE,
    _act_kwargs,
    _weight_kwargs,
    bias_probe,
    run_bias_quant,
)
from nanochat.modules.experiments.block_sampling import (
    SAMPLING_FAMILIES,
    STEP_ARM_BLOCK,
    _accumulate_one_step,
    _length_only_idx,
    _per_block_reference,
    _spread,
    grad_retained_fraction,
    run_block_sampling,
)
from nanochat.modules.experiments.common import (
    BASELINE_PRESET,
    OBJECTIVE_CE,
    OBJECTIVE_EDM,
    SAMPLING_MICRO,
    SAMPLING_STEP,
    VARIANT_PRESET,
    block_probe_tensors,
    build_probe_engine,
    codebook_of,
    count_codebook_cost,
    engine_named_parameters,
    expected_1_over_sqrt_n,
    non_negative_probe,
    probe_layer,
)
from nanochat.modules.experiments.grad_scale import run_grad_scale
from nanochat.modules.experiments.objective import (
    _run_objective_arm,
    run_objective,
)
from nanochat.modules.experiments.overlap import (
    OVERLAP_FLOORS,
    measure_overlap_out_of_band,
    run_overlap,
    sorted_overlap_sweep,
)
from nanochat.modules.experiments.sigma_cond import run_sigma_cond
from nanochat.modules.experiments.sparsity import (
    SCOPES,
    SPARSITY_FLOORS,
    measure_sparsity_nmse,
    pruned_lcqat_layers,
    run_sparsity,
    scope_mask_disagreement,
    sparsity_sweep,
)

__all__ = [
    "BASELINE_PRESET",
    "BIAS_CODEBOOK_DEAD",
    "BIAS_CODEBOOK_LIVE",
    "K_MATCHED_BUDGET",
    "OBJECTIVE_CE",
    "OBJECTIVE_EDM",
    "OVERLAP_FLOORS",
    "SAMPLING_FAMILIES",
    "SAMPLING_MICRO",
    "SAMPLING_STEP",
    "SCOPES",
    "SMOOTHPWL_DEFAULT_INIT_RANGE",
    "SPARSITY_FLOORS",
    "STEP_ARM_BLOCK",
    "VARIANT_PRESET",
    # Underscore-prefixed, and re-exported only so `scripts.lcqat_ablation`
    # keeps the attribute it had before the split. Nothing here imports them.
    "_accumulate_one_step",
    "_act_kwargs",
    "_length_only_idx",
    "_per_block_reference",
    "_run_objective_arm",
    "_spread",
    "_weight_kwargs",
    "bias_probe",
    "block_probe_tensors",
    "build_activation_lut",
    "build_probe_engine",
    "codebook_of",
    "count_codebook_cost",
    "engine_named_parameters",
    "expected_1_over_sqrt_n",
    "grad_retained_fraction",
    "matched_budget_codebooks",
    "measure_overlap_out_of_band",
    "measure_sparsity_nmse",
    "non_negative_probe",
    "piecewise_linear",
    "probe_layer",
    "pruned_lcqat_layers",
    "run_act_lut",
    "run_asym_vs_small",
    "run_bias_quant",
    "run_block_sampling",
    "run_grad_scale",
    "run_objective",
    "run_overlap",
    "run_sigma_cond",
    "run_sparsity",
    "scope_mask_disagreement",
    "sorted_overlap_sweep",
    "sparsity_sweep",
]
