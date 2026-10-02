"""
Export a trained LC-QAT checkpoint into a stripped inference artifact:
codebooks frozen to static FP32 LUTs, FP32 shadow weights replaced by uint8
index matrices (LC-QAT PRD section 8). Run from the project root:

python -m scripts.export_lcqat --source base --out exports/model.pt

The artifact is consumed by the quantized inference runtime (see
the quantized runtime); it is not a resumable training checkpoint.
"""

import argparse
import os

from nanochat.models.quant import export_lcqat_checkpoint, is_lcqat_state
from nanochat.modules.checkpoint_manager import load_model
from nanochat.utils.common import autodetect_device_type, compute_init, print0

parser = argparse.ArgumentParser(description="Export a stripped LC-QAT checkpoint")
parser.add_argument(
    "-i",
    "--source",
    type=str,
    default="base",
    help="Source of the model: base|sft|rl",
)
parser.add_argument(
    "-g", "--model-tag", type=str, default=None, help="Model tag to load"
)
parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
parser.add_argument(
    "-o",
    "--out",
    type=str,
    default=None,
    help="Output path (default: <source>_checkpoints/<tag>/lcqat_export_<step>.pt)",
)
parser.add_argument(
    "--device-type",
    type=str,
    default="",
    choices=["cuda", "cpu", "mps"],
    help="Device type: cuda|cpu|mps. empty => autodetect",
)
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(
    args.source, device, phase="eval", model_tag=args.model_tag, step=args.step
)

if not is_lcqat_state(model.state_dict()):
    raise SystemExit(
        "This checkpoint was not trained with LC-QAT (no codebook parameters found). "
        "Train with --lcqat first, then export."
    )

step = meta.get("step", args.step)
out_path = args.out
if out_path is None:
    out_path = os.path.join("exports", f"lcqat_{args.source}_{step}.pt")
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

state = export_lcqat_checkpoint(model, out_path)
num_index_buffers = sum(1 for k in state if k.endswith("packed_weight_indices"))
size_mb = os.path.getsize(out_path) / (1024 * 1024)
print0(
    f"Exported LC-QAT artifact to {out_path} "
    f"({num_index_buffers} quantized modules, {size_mb:.1f} MiB)"
)
