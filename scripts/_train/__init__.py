"""
Implementation of the `scripts.base_train` pretraining entry point.

Split out of `scripts/base_train.py` for structure only -- `build` assembles the
model and the derived run config, `eval` holds the in-loop validation / CORE /
sampling / run summary, and `loop` runs the training loop. The entry point
itself still parses args and drives everything at module import time, exactly as
before.
"""
