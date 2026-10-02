# Refactor baseline

Recorded on commit 693135f (pre-refactor), torch 2.14.0+cpu, Python 3.12.13, OMP_NUM_THREADS=4.

```
=== BASELINE 2026-10-02T10:33:28Z ===
HEAD=693135fb9bad18753fa095368cfe58d36d6b718a
OMP_NUM_THREADS=4
--- pytest ---
  /home/gabz/Projects/LCQAT_nanochat/.venv/lib/python3.12/site-packages/taichi/_lib/utils.py:70: DeprecationWarning: 'locale.getdefaultlocale' is deprecated and slated for removal in Python 3.15. Use setlocale(), getencoding() and getlocale() instead.
    return path.encode(locale.getdefaultlocale()[1])

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1019 passed, 13 skipped, 1 warning in 431.83s (0:07:11)
```

The numerical fingerprint recorded at the same commit lives in `tests/numerical_reference.json`.
All 13 values were identical before and after the restructure.

## Dependency-direction pass

A post-restructure AST audit (real `import` statements, not text search) found
`nanochat/models/` still importing the imperative shell. Three genuine leaks
were closed:

| Leak | Fix |
|---|---|
| `backbone.py`, `fp8.py`, `flash_attention.py` imported `COMPUTE_DTYPE` from `nanochat/utils/common.py` | The dtype policy moved to `nanochat/models/dtype.py`; `utils/common.py` re-exports it for the shell. It is a model decision — every `Linear` casts to it in forward |
| `retrofit.py` and `backbone.py` called the shell's `print0` | Replaced with `warnings.warn`. Both were reporting a model-layer fact, and importing the logging shell inverted the direction |

`nanochat/models/quant/experiments/` was also moved to `nanochat/modules/experiments/`:
ablation measurements import `training/diffusion_blocks.py`, so they are harness
code, not model math. The audit now reports **0** core-to-shell imports.

`models/ -> nanochat.ops` remains, and is intended: `ops` is the sanctioned kernel
boundary between the core and the hardware.
