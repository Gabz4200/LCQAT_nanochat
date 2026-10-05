#!/bin/bash
# night5: same as night4 but --db-objective=ce.
# All four technologies on: LC-QAT + SparseProp + DiffusionBlocks + DARD.
#
# Sized to this host by measurement, not by wishful thinking (see docs/dard.md
# "Measured"). The box is a 4-core i5-8250U: measured bf16 matmul ceiling
# 0.7 TFLOP/s, and a full-depth AR micro-step at depth 6 costs 5.2 s against
# DiffusionBlocks' 1.96 s -- a real 2.7x wall-clock win, but DiffusionBlocks is
# a memory/FLOPs optimisation, not a token-throughput one, so it does not change
# how many tokens fit in the budget.
#
# Budget: ~1043 tok/s measured at depth 6 / B=3. Depth 4 is smaller still.
# 700 steps at a 32k batch is ~23M tokens, which is the honest ceiling here.
#
# --db-objective=ce, NOT edm: under EDM the objective is score-matching on
# denoised EMBEDDINGS, so lm_head receives no gradient at all (verified: grad is
# None). lm_head is 8.39M params -- 22.9% of this model -- and it is the only
# component mapping hidden state to token ids. night4 ran with the default edm
# and therefore produced word-salad BY CONSTRUCTION, not from undertraining.
# DiffusionBlocks still owns the block structure and the sparse/quantized layers;
# only the block objective changes.
#
# Depth 4 rather than 6 deliberately: at a fixed token budget a smaller model is
# far less undertrained. The compute-optimal size for 24M tokens is ~9.5M
# params, so every parameter I add is paid for twice -- once in the forward,
# once in having that much less data per weight.

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu --group dev
source .venv/bin/activate

python -m scripts.base_train \
    --depth=4 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=4 \
    --total-batch-size=32768 \
    --num-iterations=100 \
    --eval-every=25 \
    --eval-tokens=131072 \
    --core-metric-every=-1 \
    --sample-every=50 \
    --save-every=25 \
    --db-blocks=2 \
    --db-objective=ce \
    --run=dummy \
    --model-tag=night4
