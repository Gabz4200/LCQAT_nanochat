# Architecture

This page maps the package layout and the one boundary that matters: pure tensor math in `nanochat/models/` versus hardware, persistence, and lifecycle in the shell around it. Read it before adding a module or moving code across packages.

Fastest check that the boundary holds:

```bash
uv run pytest tests/test_architecture_boundary.py -q
```

The core never imports the shell. `nanochat/models/` must not import from `modules/`, `training/`, `data/`, `callbacks/`, `utils/`, or `tasks/`. Report a model-layer fact with `warnings.warn` or by returning a value, never with `print0`. A decision the layers consume (for example `COMPUTE_DTYPE` in `nanochat/models/dtype.py`) belongs in the core; the shell may re-export it. The one sanctioned exception runs downward: `nanochat/models/` may import `nanochat/ops/`, the kernel boundary. `nanochat/modules/experiments/` is harness, not model math, so measurements there may import `training/`.

Repo-specific guardrails (always-on flags, the export contract, the CPU index policy) live in [`AGENTS.md`](../AGENTS.md).

## Package layout

```
.
├── nanochat
│   ├── models/                          # functional core: pure tensor math
│   │   ├── backbone.py                  # the GPT nn.Module Transformer
│   │   ├── lfm2.py                      # the LFM2 hybrid backbone (conv + GQA, arXiv 2511.23404)
│   │   ├── dtype.py                     # COMPUTE_DTYPE policy (model-layer decision)
│   │   ├── io.py                        # typed contracts (LayerQuantSpec)
│   │   ├── flash_attention.py           # Flash Attention 3 / SDPA dispatch
│   │   ├── fp8.py                       # Float8Linear conversion
│   │   └── quant/                       # LC-QAT: codebooks, layers, export
│   │       ├── codebook.py              # asymmetric learned codebook (STE)
│   │       ├── linear.py                # LCQATLinear module
│   │       ├── lut.py                   # Activation LUTs (relu2/silu/gelu/tanh/sigmoid)
│   │       ├── learnable_lut.py         # LearnableIndexLut (softmax-relaxed LUT)
│   │       ├── per_channel.py           # PerChannelValueCenteredQuantizer
│   │       ├── product_lut.py           # fused 2-D product LUT
│   │       ├── activation.py            # SmoothPWL activation bodies
│   │       ├── bias_quant.py            # opt-in bias codebook
│   │       ├── sigma_codebook.py        # sigma-conditioned / modulated codebooks
│   │       ├── retrofit.py              # LayerKConfig, PRESETS, retrofit_model
│   │       ├── w6.py                    # shared CLI surface for the DB-coupled flags
│   │       ├── sparseprop.py            # SparsePropLinear, magnitude pruning
│   │       ├── pruning.py               # gradual pruning schedule
│   │       ├── sparse_artifact.py       # CSR sparse export planning + packing
│   │   ├── packing.py               # trit/nibble/uint8 index packing
│   │   ├── fourbit.py               # FourBitLinear: packed-nibble 4-bit QAT
│   │   ├── kd.py                    # KDLoss / DenoiserDistiller
│   │       ├── efqat.py                 # SelectiveFreezer, BlockLatchFreezer
│   │       ├── optimizer.py             # build_qat_param_groups
│   │       ├── export.py                # export_lcqat_checkpoint, wire_activation_luts
│   │       ├── ablation.py              # ablation experiment implementations
│   │       ├── ablation_metrics.py      # ablation measurement protocol
│   │       └── reference/               # pure-PyTorch reference implementations
│   ├── ops/                             # kernel layer: dispatcher + backends
│   │   ├── dispatch.py                  # dispatch_gemv / dispatch_quant_attn / dispatch_index_linear /
│   │   │                                #   dispatch_db_denoise / dispatch_sparseprop_{forward,backward}
│   │   ├── gemv.py                      # mul-less GEMV backends
│   │   ├── index_linear.py              # index-fetch matmul, any K >= 3
│   │   ├── sparse_index_linear.py       # CSR variant; naive oracle + cpu, no GPU/dense fallback
│   │   ├── quant_attn.py                # quantized KV-cache attention
│   │   ├── sparseprop.py                # AVX2 sparse forward/backward
│   │   ├── references/                  # naive PyTorch oracles (CI ground truth)
│   │   ├── native/cpu/                  # C++: gemv, index_linear, quant_attn, sparseprop
│   │   └── kernels/                     # cpu_loader.py, gpu_loader.py (Taichi), registration.py
│   ├── modules/                         # imperative shell: runtime + persistence
│   │   ├── checkpoint_manager.py        # save/load model checkpoints
│   │   ├── experiments/                 # one module per ablation experiment
│   │   ├── engine.py                    # inference engine, KV cache, calculator tool
│   │   ├── execution.py                 # sandboxed Python execution
│   │   ├── core_eval.py                 # DCLM CORE score
│   │   └── loss_eval.py                 # bits-per-byte evaluation
│   ├── training/                        # training-time machinery
│   │   ├── diffusion_blocks.py          # DiffusionBlockEngine, EquiProbabilityPartitioner,
│   │   │                                #   configure_cpu_training, cpu_adamw_for
│   │   ├── minexp_lstm.py               # SGD_MinExpLstm_Optimizer + cell re-export
│   │   │                                #   (joint online learned optimizer, AdamW replacement)
│   │   ├── mezo_minexplstm.py           # MeZo_MinExpLstm_Optimizer: zeroth-order SPSA
│   │   │                                #   variant of the same cell (dual forward, no grad)
│   │   ├── cell.py                      # SGDMinExpLSTMCellFiLM4D: 4D-latent FiLM cell,
│   │   │                                #   lambda-coupled beta predictor
│   │   ├── batched_step.py              # shape-grouped dual-EMA batched cell forward
│   │   ├── lightning_module.py          # MinExpLSTMLightningModule + LR tuning (Tuner),
│   │   │                                #   JointTrainableModule (PRD 5.3 manual-opt path)
│   │   └── optim.py                     # fused AdamW (Muon removed) + MinExpLSTM re-exports
│   ├── data/                            # ingestion boundary
│   │   ├── dataloader.py                # tokenizing distributed data loader
│   │   ├── dataset.py                   # download/read utils for pretraining data
│   │   └── tokenizer.py                 # BPE tokenizer wrapper in GPT-4 style
│   ├── callbacks/                       # side-effect observers for the loop
│   │   └── training.py                  # W&B logging, GC management, run summary
│   ├── utils/
│   │   └── common.py                    # misc shell utilities (re-exports COMPUTE_DTYPE)
│   └── tasks/                           # evaluation task mixtures
│       ├── common.py                    # TaskMixture | TaskSequence
│       └── arc.py / gsm8k.py / humaneval.py / mmlu.py / smoltalk.py
├── scripts                              # entry points (the imperative shell)
│   ├── _cli.py                          # add_common_cli_args (db/lcqat args live with their owners)
│   ├── _train/                          # base_train split by concern
│   │   ├── build.py                     # model construction, LC-QAT + SparseProp wiring
│   │   ├── loop.py                      # the training loop
│   │   └── eval.py                      # validation, CORE metric, sampling, checkpoints
│   ├── base_train.py  base_eval.py  chat_sft.py  chat_rl.py  chat_eval.py
│   ├── chat_cli.py  export_lcqat.py  tok_train.py  tok_eval.py
│   └── lcqat_ablation.py  gemv_bench.py  infer_bench.py  kv_budget_bench.py
├── runs                                 # shell harnesses
│   ├── runcpu.sh                        # end-to-end CPU demo (d6)
│   ├── stackcompare.sh                  # four-arm LC-QAT/SparseProp/DiffusionBlocks sweep
│   ├── speedrun.sh  miniseries.sh  scaling_laws.sh
│   └── lib_depth.sh                     # depth-dependent flags shared by the two sweeps
├── tests                                # mirrors the source layout
│   ├── test_lcqat_*.py                  # parity, opcheck, runtime, wiring
│   ├── test_sparseprop*.py              # kernel parity, pruning, integration
│   ├── test_dbcpu_*.py                  # DiffusionBlocks CPU
│   ├── test_lm_mode.py                  # plain-LM path (--db-blocks=0)
│   ├── test_numerical_fingerprint.py    # behavior-preservation gate
│   ├── test_architecture_boundary.py    # core must not import the shell
│   └── test_w0_*.py  test_w2_*.py  ...
├── dev
│   ├── LEADERBOARD.md                   # Time-to-GPT-2 docs + generated ablation table
│   ├── STACK_COMPARE.md                 # four-arm SparseProp/DiffusionBlocks comparison
│   ├── LOG.md                           # training experiment log
│   ├── lcqat_spec.md                    # LC-QAT spec, symbiosis, measured deviations
│   ├── REFACTOR_BASELINE.md
│   └── *_analysis.ipynb, *.png, *.pdf
├── docs                                 # reference documentation (docs/README.md indexes it)
│   ├── quantization.md                 # LC-QAT, KD, EfQAT, sigma-conditioned, per-channel
│   ├── kernels.md                      # mul-less GEMV, index-linear, sparse CSR, export, benchmarks
│   ├── sparseprop.md                   # sparsity mechanism, defect fixes, measured cost
│   ├── diffusionblocks.md              # the block-wise engine and its CPU behaviour
│   └── architecture.md                 # this file
├── AGENTS.md                            # engineering guardrails for this repo
├── pyproject.toml                       # CPU-only torch wheel (the `cpu` extra)
└── uv.lock
```

## MeZO-MinExpLSTM stack

The zeroth-order sibling of the SGD optimizer: the same `SGDMinExpLSTMCellFiLM4D` cell and FiLM head, but the update direction comes from a SPSA estimate `(L(θ+εz) - L(θ-εz)) / (2ε)` instead of a materialized gradient, so training needs only forward passes.

- `MeZoJointTrainableModule` (manual optimization): dual forward passes, seed-replayed perturbation, FiLM-modulated update, online REINFORCE meta-step.
- `MeZo_MinExpLstm_Optimizer`: `naive` (reference), `cpu` (C++), `gpu` (Taichi) backends; three-way parity pinned in `tests/test_mezo_lightning.py`.
- **Multi-probe (PRD v2.1):** `--mezo-probes M` (default 1) samples M independent directional probes per step, each a full dual-forward cycle around the same base state, and aggregates them with the order-invariant arithmetic mean. The FIR and MeZO vectors are linear in `mu`, so the batch mean equals the single-probe step with the mean `mu`, and the aggregation is permutation-equivariant (swapping probe seeds yields identical updates, pinned in `tests/test_mezo_multiprobe.py`). M=4 is the PRD target: variance reduction 1/M at the cost of 2M forward passes. Memory footprint is unchanged (1.5 bytes/parameter): the M probes are averaged before the single recurrent-state writeback, so the state stays O(1) in M.

## Two backbones

`--backbone {gpt,lfm2}` selects the model the whole stack trains. Both implement the same interface (`forward(idx, targets=None, ...)`, `transformer.h`, `lm_head`, `_precompute_rotary_embeddings`, `generate`), so every optimizer, quantization stage, and DiffusionBlocks objective works on either without a branch at the call site.

- **GPT** (`nanochat/models/backbone.py`): the all-attention Transformer. Default.
- **LFM2** (`nanochat/models/lfm2.py`, arXiv 2511.23404): a hybrid of gated short convolutions and a small fraction of GQA attention blocks (`num_attention_blocks ≈ depth × 6/16`, capped), with SwiGLU MLPs, pre-norm RMSNorm, and RoPE with QK-norm. The attention ratio is the model's own; the rest of the depth is convolution. The backbone is chosen at construction in `scripts/_train/build.py` and recorded in the saved config, so `checkpoint_manager.build_model` rebuilds the stack a checkpoint was trained with (it detects LFM2 by `num_attention_blocks` in the config) rather than always loading a GPT.

The DiffusionBlocks engine type is `GPT | LFM2` for the same reason: it reads `model.transformer.h` and `model._precompute_rotary_embeddings`, which both backbones provide. See [docs/diffusionblocks.md](diffusionblocks.md#sequential-vs-parallel-block-execution) for the sequential/parallel block-execution choice.

## 4-bit QAT

`--qat-4bit` (mutually exclusive with `--lcqat`) retrofits every eligible `Linear` to `FourBitLinear` (`nanochat/models/quant/fourbit.py`): an fp32 master weight plus a packed-nibble mirror (uint8, two 4-bit weights per byte) that the straight-through-estimator forward reads, refreshed by `quantize_weight_()` after each step. All weights are real 4 bits at inference; the fp32 master exists only so the optimizer still updates in fp32. Retrofit happens in `scripts/_train/build.py` after backbone construction and before LC-QAT/SparseProp injection (which `FourBitLinear` is skipped by, since it is not a plain `nn.Linear`).



## Learnable, data-dependent LR

Both MinExpLSTM optimizers (`SGD_` and `MeZo_`) use a **learnable, data-dependent learning rate**. The LR is a function of the current step's data, not a fixed hyperparameter:

- A tiny `nn.Linear(1, 1)` **LR predictor** maps the layer-wise RMS-normalized gradient magnitude (the same scale-invariant signal the cell and the beta predictor consume) to a logit.
- A **sigmoid** parametrizes the LR to (0, 1): strictly positive (the optimizer always moves) and bounded by 1 (the step is bounded). The predictor is zero-initialized with bias = `inv_sigmoid(lr)`, so a fresh optimizer's LR equals the requested scalar `lr` and the trajectory matches native SGD at step 0.
- The **AdamW meta-optimizer co-trains the LR predictor alongside the cell** (`meta_step`'s differentiable closure rebuild computes the LR from the rebuilt gradient, so the predictor sits on the gradient path). The LR adapts to the loss landscape; some steps move faster than others.
- The **AdamW meta-optimizer's own LR (`meta_lr`) stays a plain hyperparameter**; it is *not* learned. Lightning's LR finder (`tune_lr`) optimizes it before training starts (warmup), then training proceeds with the meta-optimizer co-training the cell + LR predictor.
- The fused CPU/GPU kernels consume the LR's *value* (the forward path); the LR's gradient flows through the differentiable meta-rebuild closure, never through the fused kernel (the same forward/kernel vs. backward/closure split the cell uses).

Pinned in `tests/test_learnable_lr.py` and `tests/test_sgd_minexplstm_film.py`.
