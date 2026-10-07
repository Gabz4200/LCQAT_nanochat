#!/bin/bash
# Overnight combined-technology training run.
#
# All five technologies ON simultaneously:
#   - LC-QAT (default-on, codebook params in their own group)
#   - MinExpLSTMOptimizer (default optimizer, learnable beta,
#     INT8 states, joint online co-training every K steps)
#   - SparseProp (0.7 sparsity, magnitude-pruned, static mask)
#   - DiffusionBlocks (block-wise EDM objective, 4 blocks)
#   - DARD (vocabulary-space algebraic rule loss)
#
# Scaled for the host (7.6 GB RAM, 4 cores) to run ~7.5 h
# without OOM: depth 12, seq 512, device-batch 8. Meta-step
# every 20 optimizer steps (the co-training forward+backward
# roughly doubles per-step cost, so a lower frequency keeps
# enough iterations in the window). Early stop on the eval
# loss plateau so the run does not burn hours on a converged
# model.
#
# Run as:  bash runs/overnight_combined.sh
set -euo pipefail

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
# No wandb API key on this host: run offline so the
# overnight run does not block on an interactive login.
export WANDB_MODE=disabled
export WANDB_SILENT=true
# Line-buffered stdout: piped to tee, Python would otherwise
# block-buffer (4 KB) and the log would sit silent for minutes.
export PYTHONUNBUFFERED=1
mkdir -p "$NANOCHAT_BASE_DIR"
[ -d ".venv" ] || uv venv
uv sync --extra cpu --group dev
source .venv/bin/activate

WANDB_RUN=${WANDB_RUN:-overnight_combined}
# Sizing for the 7.6 GB RAM / 4-core host, ~7.5 h window
# (01:30 -> 07:00). CRITICAL findings from timing probes:
#   - --window-pattern L is mandatory: the default SSSL
#     sliding-window pattern has no SDPA kernel on CPU and
#     measured 242 s/step (19x slower than L).
#   - The DARD penalty is the dominant cost: it runs a
#     per-block LM head over the full 32768 vocab for
#     EVERY block, so the step cost scales with depth x
#     batch x vocab. Measured: d10/b8 = 191 s/step,
#     d10/b4 = 18 s, d8/b4 = 13 s, d6/b4 = 11 s.
#   - Batch 8 explodes the DARD cost super-linearly; batch
#     4 keeps it linear.
# d8 x aspect 24 -> n_embd 192, ~9.8M params, 13 s/step.
# 1900 iterations at 13 s = 6.9 h, leaving margin for the
# meta-step every 50 steps and the eval every 100 steps.
# matrix-lr 0.005 (default 0.02 destabilized the loss:
# 0.19 -> 25 in 3 steps); MinExpLSTM's EMA needs a lower
# LR than AdamW at this scale.
DEPTH=${DEPTH:-8}
SEQ=${SEQ:-256}
BATCH=${BATCH:-4}
TOTAL_BATCH=${TOTAL_BATCH:-1024}
ASPECT=${ASPECT:-24}
HEAD_DIM=${HEAD_DIM:-64}
DB_BLOCKS=${DB_BLOCKS:-2}
SPARSITY=${SPARSITY:-0.7}
META_EVERY=${META_EVERY:-50}
ITERS=${ITERS:-1900}
EVAL_EVERY=${EVAL_EVERY:-100}
MATRIX_LR=${MATRIX_LR:-0.005}
# Early stop: halt after this many consecutive evals without
# improvement (eval loss is the signal; the loop prints it
# every EVAL_EVERY steps and this wrapper watches the log).
PATIENCE=${PATIENCE:-8}

RUN_TAG="d${DEPTH}_s${SEQ}_b${BATCH}_db${DB_BLOCKS}_sp${SPARSITY}"

echo "=== overnight combined run: $RUN_TAG ==="
echo "RAM: $(free -g | awk '/Mem:/{print $2"G total, "$7"G avail"}')"

# Launch training in the background, tee to a log so the
# early-stop watcher can read eval losses as they appear.
LOG="$HOME/.cache/nanochat/${RUN_TAG}.log"
python -m scripts.base_train \
    --depth="$DEPTH" \
    --aspect-ratio="$ASPECT" \
    --head-dim="$HEAD_DIM" \
    --window-pattern=L \
    --max-seq-len="$SEQ" \
    --device-batch-size="$BATCH" \
    --total-batch-size="$TOTAL_BATCH" \
    --eval-every="$EVAL_EVERY" \
    --eval-tokens=262144 \
    --core-metric-every=-1 \
    --sample-every=500 \
    --save-every=500 \
    --num-iterations="$ITERS" \
    --db-blocks="$DB_BLOCKS" \
    --db-objective=edm \
    --sparseprop-sparsity="$SPARSITY" \
    --sparseprop-ramp-steps=200 \
    --optimizer=minexplstm \
    --matrix-lr="$MATRIX_LR" \
    --minexp-alpha=0.1 \
    --minexp-beta=0.9 \
    --minexp-clamp=1.0 \
    --minexp-learnable-beta \
    --minexp-meta-every="$META_EVERY" \
    --minexp-meta-lr=1e-3 \
    --run="$WANDB_RUN" \
    2>&1 | tee "$LOG" &

TRAIN_PID=$!
echo "training pid=$TRAIN_PID, log=$LOG"

# Early-stop watcher: poll the log for eval losses and kill
# the run when the eval loss stops improving for PATIENCE
# consecutive evals. The loop prints a line containing
# "val_bpb" (or "eval") on every eval; the watcher extracts
# the latest float and tracks the running minimum.
python - "$LOG" "$PATIENCE" "$TRAIN_PID" <<'PYEOF' &
import re, sys, time, os, signal

log_path, patience_s, pid_s = sys.argv[1], sys.argv[2], sys.argv[3]
patience = int(patience_s)
pid = int(pid_s)
# Matches the eval-loss line the loop prints:
# "Step 00100 | Validation bpb: 5.123456"
pat = re.compile(r"Validation bpb:\s*([0-9]*\.?[0-9]+)", re.I)

best = float("inf")
bad = 0
last_size = 0
while True:
    if not os.path.exists(log_path):
        time.sleep(5)
        continue
    try:
        with open(log_path, "r") as f:
            f.seek(last_size)
            chunk = f.read()
            last_size += len(chunk)
    except OSError:
        time.sleep(5)
        continue
    for m in pat.finditer(chunk):
        val = float(m.group(1))
        if val < best - 1e-6:
            best = val
            bad = 0
        else:
            bad += 1
        print(f"[earlystop] eval={val:.4f} best={best:.4f} bad={bad}/{patience}", flush=True)
        if bad >= patience:
            print(f"[earlystop] plateau for {patience} evals; stopping pid {pid}", flush=True)
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            sys.exit(0)
    time.sleep(30)
PYEOF

WATCH_PID=$!
echo "early-stop watcher pid=$WATCH_PID (patience=$PATIENCE)"

# Wait for training to finish (or be killed by the watcher).
wait $TRAIN_PID
RC=$?
echo "=== training exited rc=$RC ==="
# Stop the watcher if training finished on its own.
kill $WATCH_PID 2>/dev/null || true
exit $RC
