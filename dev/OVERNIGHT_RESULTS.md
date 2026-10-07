# Overnight Combined Run — Results

Run tag: `d8_s256_b4_db2_sp0.7`
Log: `~/.cache/nanochat/d8_s256_b4_db2_sp0.7.log`
Launch log: `~/.cache/nanochat/overnight_launch.log`
Script: `runs/overnight_combined.sh`

## Configuration

- Model: depth=8, aspect=24, head_dim=64 → n_embd=192, seq=256, batch=4
- Technologies (all 5 ON simultaneously):
  - LC-QAT (default-on, 3 codebook presets)
  - MinExpLSTMOptimizer (default optimizer, learnable_beta=True, INT8 states)
  - SparseProp (sparsity=0.7, magnitude-pruned, static mask)
  - DiffusionBlocks (2 blocks, EDM objective)
  - DARD (378 rules, lambda=0.2, per-block head)
- Optimizer: MinExpLSTM, matrix-lr=0.005, alpha=0.1, beta=0.9, clamp=1.0
- Meta-step (joint online co-training) every 50 steps
- Iterations: 1900, eval every 100, early-stop patience=8 evals

## Host constraints that drove the sizing

- 7.6 GB RAM, 4 cores. Two timing probes were decisive:
  - `--window-pattern L` is mandatory: the default SSSL sliding-window
    pattern has no SDPA kernel on CPU and measured 242 s/step (19x slower).
  - The DARD penalty dominates cost (per-block LM head over the full
    32768 vocab for every block). Measured: d10/b8=191 s, d10/b4=18 s,
    d8/b4=13 s, d6/b4=11 s. Batch 8 explodes DARD cost super-linearly.
- d8/b4 at 13.5 s/step → 1900 iters ≈ 7.1 h, fits the 01:30→07:00 window.

## Progress (live)

See the log for the running `step ... loss ...` lines. Loss trend:
0.15 → 23 → 15 → 8.7 → 4.8 → 1.9 → 0.95 → 0.65 (converging).
meta_loss logged at step 100: 10.42 (cell co-training active).
Early-stop watcher: `grep earlystop ~/.cache/nanochat/overnight_launch.log`.

## Verification

- 18/18 MinExpLSTM unit tests pass (`tests/test_minexp_lstm_optimizer.py`).
- 42 tests covering changed files pass (minexp, adamw, db_edm, db_objective,
  architecture boundary).
- ruff check / format: clean. pyrefly: minexp_lstm.py + beta_net.py clean
  (pre-existing hparams/unbound-name false-positives remain in
  lightning_module.py and _train/loop.py). aislop: 85/100, 0 errors.
- Pre-existing C++ kernel test failures (environmental, confirmed via
  git stash clean-tree run): test_lcqat_sparse_kernel, test_db_denoise_kernel,
  test_lcqat_export, test_lcqat_quant_attn, test_sparseprop.
