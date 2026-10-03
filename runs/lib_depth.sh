#!/bin/bash
# Depth-dependent training flags shared by runs/miniseries.sh and
# runs/scaling_laws.sh.
#
# Both scripts sweep the same depth list and therefore need the same two
# ladders: how large a per-device batch fits before OOM, and how many
# DiffusionBlocks to scale the engine to. They were duplicated verbatim in both
# files, which meant a threshold change had to be made twice and a missed edit
# would silently make two sweeps non-comparable.

# device_batch_arg <depth> -> "--device-batch-size=N", reduced at larger depths
# to stay inside memory.
device_batch_arg() {
    local d=$1
    if [ $d -ge 28 ]; then
        echo "--device-batch-size=8"
    elif [ $d -ge 20 ]; then
        echo "--device-batch-size=16"
    else
        echo "--device-batch-size=32"
    fi
}

# db_blocks_arg <depth> -> the bare block count, for `--db-blocks=N`.
# The DiffusionBlocks engine is on by default; the block count scales with
# depth so the B-fold activation-memory argument holds at every depth.
db_blocks_arg() {
    local d=$1
    if [ $d -ge 26 ]; then
        echo "8"
    elif [ $d -ge 16 ]; then
        echo "6"
    else
        echo "4"
    fi
}