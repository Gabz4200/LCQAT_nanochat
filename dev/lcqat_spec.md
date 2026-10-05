# LC-QAT Specification: Learnable Codebook Quantization-Aware Training, System Symbiosis, and Measured Deviations

This file merges `dev/lc_qat_pseudoPaper.md` (§§1-6 and Appendix A) and `dev/combination.md` (§§1-5) into one reading order. It adds no facts. Equations, code blocks, tables, and numbers below match the sources. The two originals were deleted after the merge; their content survives only here.

## Abstract

Standard uniform Quantization-Aware Training (QAT) forces continuous model parameters onto rigid linear grids, inducing severe accuracy degradation at sub-byte and low-bit regimes ($K \le 15$, $\le 4\text{ bits/param}$). Conversely, multi-dimensional Vector Quantization (VQ) introduces the curse of dimensionality, suffering from codebook collapse and requiring complex auxiliary losses. We present **Learnable Codebook Quantization-Aware Training and Inference with LUTs (LC-QAT)**, an end-to-end framework that reformulates Large Language Models (LLMs) into adaptive, non-uniform 1D discrete state machines. LC-QAT operates strictly in 1D scalar space ($\mathbb{R}^1$) with module-level codebook ownership and element-wise independent indexing.

By parameterizing step increments via softplus-transformed latent variables, LC-QAT guarantees 1D monotonicity ($c_k < c_{k+1}$) and anchors an exact structural zero level ($c_{M_{\text{neg}}} = 0.0$) without manual projection steps. We derive a dual-gradient autograd formulation ($\text{out} = f(x) + x - \text{detach}(x)$) that simultaneously updates continuous shadow weights via Straight-Through Estimation (STE) and codebook boundaries via task loss gradients scaled by $1/\sqrt{N}$. At inference, LC-QAT eschews complex bit-sliced boolean kernels, employing L1 cache-resident 1D and 2D Look-Up Tables (LUTs) with standard Fused Multiply-Add (FMA) hardware instructions.

## Table of contents

- Part I, LC-QAT spec (from pseudoPaper §§1-6)
  - 1. Introduction
  - 2. Mathematical formulation and optimization mechanics
  - 3. System architecture and structural granularity
  - 4. Hardware execution engine and LUT accelerators
  - 5. PyTorch engineering and reference implementation
  - 6. Comparison matrix
- Part II, symbiosis and measurements (from combination.md §§1-4)
  - 7. The unified architecture stack
  - 8. Harmonizing the mechanics (why they do not conflict)
  - 9. The unified execution pipeline
  - 10. The four-combination measurement (CPU, measured)
- Part III, measured deviations appendix (one copy of the combination.md §5 / pseudoPaper Appendix A overlap)
  - 11. Measured deviations from the spec
    - 11.1 Where the implementation deviates from §2.2
    - 11.2 The activation relaxation (new, §2.3 did not specify one)
    - 11.3 The zero anchor does NOT transfer automatically to the activation
    - 11.4 Knot precision is free
    - 11.5 The activation body (`SmoothPWL`): the third measurement
    - 11.6 Bias quantization: capability shipped, measured, default off

---

# Part I: LC-QAT spec (from pseudoPaper §§1-6)

Part I states the LC-QAT spec. It defines asymmetric codebooks, softplus monotonicity, dual-gradient flow, 1/sqrt(N) scaling, system architecture, the LUT engine, the reference implementation, and the comparison matrix.

## 1. Introduction

Quantization is the primary mechanism for mitigating the memory bandwidth bottleneck of Large Language Models (LLMs) during autoregressive generation. However, pushing parameters and activations into sub-byte regimes ($K=3$ ternary, $K=15$ 4-bit) reveals fundamental breakdown points in existing compression approaches.

### 1.1 Limitations of Uniform Quantization

Standard uniform quantization maps continuous real numbers to integer buckets using a linear affine transformation:

$$W_q = \text{clamp}\left( \text{round}\left( \frac{W}{S} \right) + Z, \, Q_{\min}, \, Q_{\max} \right)$$

This forces all quantization centroids to be strictly equidistant ($\Delta = \text{const}$). Because LLM weights and activations display non-uniform, kurtotic distributions with heavy tails and asymmetric post-activation skewness (e.g., post-SiLU or post-GELU distributions), uniform grids waste quantization buckets on low-density regions while coarse-graining high-density central peaks.

### 1.2 Limitations of High-Dimensional Vector Quantization

Vector Quantization (VQ) techniques project multi-dimensional feature blocks ($\mathbb{R}^D$) into discrete codebook vectors. Operating in high-dimensional space subjects the optimization process to the curse of dimensionality. Codebook entries frequently drift into unassigned latent voids (a phenomenon known as _codebook collapse_), requiring heuristic mechanisms such as commitment losses ($\Vert{}z_e(x) - \text{sg}[e]\Vert{}_2^2$), Exponential Moving Average (EMA) cluster resets, or entropy penalties.

```
Standard Uniform Quantization (Rigid Equal Spacing):
---[ -1.0 ]-------[ -0.5 ]-------[  0.0  ]-------[ +0.5 ]-------[ +1.0 ]---

LC-QAT Non-Uniform Codebook (Learned Spacing Matched to Density):
---[-1.0]---[-0.3][-0.1][ 0.0 ][+0.1][+0.3]---------------[+1.0]---
                        ^
                  Exact Zero Anchor

```

### 1.3 The LC-QAT Solution

LC-QAT resolves this dichotomy by operating strictly in **1D scalar space ($\mathbb{R}^1$)**. Parameters and activations are discretized independently element-wise, but map to non-uniform, parametric codebook levels whose interval step sizes are optimized end-to-end via backpropagation. By maintaining strict 1D topological ordering and anchoring an explicit zero level, LC-QAT eliminates codebook collapse by design while adapting its quantization resolution to the exact empirical shape of model tensors.

---

## 2. Mathematical Formulation & Optimization Mechanics

### 2.1 Asymmetric Split Codebooks ($K = M_{\text{neg}} + 1 + M_{\text{pos}}$)

Activation tensors in modern LLM architectures display strong directional asymmetry. Post-ReLU activations are strictly non-negative ($x \ge 0$), whereas post-SiLU and post-GELU activations exhibit shallow negative troughs ($x \approx -0.28$) bounded by long positive tails.

To prevent resolution wasting, LC-QAT generalizes codebook cardinality $K$ into an asymmetric split comprising $M_{\text{neg}}$ negative levels, $1$ zero anchor, and $M_{\text{pos}}$ positive levels:

$$K = M_{\text{neg}} + 1 + M_{\text{pos}}$$

Index $M_{\text{neg}}$ is anchored strictly to **FP32 $0.0$**. Exact structural zeros, such as sparse activation outputs, incur zero quantization noise and consume no dynamic range from non-zero levels.

### 2.2 Softplus Parameterization for Guaranteed Monotonicity

To ensure that codebook levels remain strictly ordered ($c_0 < c_1 < \dots < c_{K-1}$) during unconstrained gradient updates without applying non-differentiable sorting or clamping steps, step increments $\delta_i^+, \delta_i^-$ are parameterized using latent variables $\rho_i^+, \rho_i^- \in \mathbb{R}$ transformed via the softplus function:

$$\delta_i^+ = \text{softplus}(\rho_i^+) = \ln\left(1 + e^{\rho_i^+}\right), \quad i \in \{1, \dots, M_{\text{pos}}\}$$

$$\delta_i^- = \text{softplus}(\rho_i^-) = \ln\left(1 + e^{\rho_i^-}\right), \quad i \in \{1, \dots, M_{\text{neg}}\}$$

The complete $K$-entry codebook vector $\mathbf{C} \in \mathbb{R}^K$ is assembled via prefix summation:

$$\mathbf{C}_{\text{neg}} = \left[ -\sum_{j=1}^{1} \delta_j^-, \, -\sum_{j=1}^{2} \delta_j^-, \, \dots, \, -\sum_{j=1}^{M_{\text{neg}}} \delta_j^- \right]$$

$$\mathbf{C}_{\text{pos}} = \left[ \sum_{j=1}^{1} \delta_j^+, \, \sum_{j=1}^{2} \delta_j^+, \, \dots, \, \sum_{j=1}^{M_{\text{pos}}} \delta_j^+ \right]$$

$$\mathbf{C} = \begin{bmatrix} \text{flip}(\mathbf{C}_{\text{neg}}) & 0.0 & \mathbf{C}_{\text{pos}} \end{bmatrix}^T$$

Because $\text{softplus}(z) > 0$ strictly for all $z \in \mathbb{R}$, strict monotonicity $c_k < c_{k+1}$ is guaranteed by construction.

### 2.3 Dual-Gradient Flow Mechanics

LC-QAT requires backpropagation to simultaneously update two distinct target sets during every backward pass:

1. **Continuous Shadow Weights ($x$):** Learning how to shift continuous parameters so their discretized assignments fall into low-loss codebook buckets.
2. **Codebook Parameters ($\mathbf{\rho}^+, \mathbf{\rho}^-$):** Learning how to adjust interval step boundaries so the non-uniform grid fits the global parameter distribution.

To enable dual-gradient flow within PyTorch's computational graph, the forward operator is formulated as:

$$\text{out} = f(x) + (x - \text{detach}(x))$$

Where $f(x) = \mathbf{C}[\text{bucketize}(x, \mathbf{B})]$ performs the discrete codebook lookup against boundary midpoints $B_k = \frac{C_k + C_{k+1}}{2}$.

```
                  ┌───► f(x) ──────────────► Codebook Params (ρ+, ρ-)
                  │     (Discrete Lookup)    (Learns Bin Spacing)
dL / d(out) ──────┤
                  │
                  └───► + x - detach(x) ───► Continuous Input x
                        (Identity STE)       (Learns Shadow Weights)

```

#### Derivation of the Gradient Split:

- **Gradient w.r.t. Continuous Input $x$:**

$$\frac{\partial \text{out}}{\partial x} = \frac{\partial f(x)}{\partial x} + \frac{\partial x}{\partial x} - \frac{\partial \text{detach}(x)}{\partial x}$$

Since $f(x)$ is a step function whose derivative is zero almost everywhere ($\frac{\partial f(x)}{\partial x} = 0$) and $\frac{\partial \text{detach}(x)}{\partial x} = 0$:

$$\frac{\partial \text{out}}{\partial x} = 0 + 1 - 0 = 1$$

This passes an unattenuated identity gradient back to continuous shadow weights and preceding network layers (Straight-Through Estimation).

- **Gradient w.r.t. Codebook Parameters $\rho_j^+$:**

$$\frac{\partial \text{out}}{\partial \rho_j^+} = \frac{\partial f(x)}{\partial \rho_j^+} + 0 - 0 = \frac{\partial f(x)}{\partial \rho_j^+}$$

Applying the chain rule through the cumulative prefix sum yields:

$$\frac{\partial \mathcal{L}}{\partial \rho_j^+} = \sum_{k=j}^{M_{\text{pos}}} \left( \sum_{\{i \mid Q_i = M_{\text{neg}} + k\}} \frac{\partial \mathcal{L}}{\partial \text{out}_i} \right) \cdot \sigma(\rho_j^+)$$

where $\sigma(z) = (1 + e^{-z})^{-1}$ is the logistic sigmoid function (the derivative of softplus).

### 2.4 Gradient Scaling ($1/\sqrt{N}$)

In a linear layer of size $4096 \times 4096$, $16.7\text{ million}$ elements map into a single $K$-entry codebook. Unscaled sum-reduction across millions of element gradients causes codebook step parameters ($\rho^+, \rho^-$) to oscillate or diverge relative to weight updates.

To stabilize co-adaptation, codebook parameter gradients are scaled down inversely proportional to the square root of the tensor element count ($N = \text{numel}(X)$) using backward hooks:

$$\nabla_{\mathbf{\rho}} \mathcal{L} \leftarrow \frac{1}{\sqrt{N}} \nabla_{\mathbf{\rho}} \mathcal{L}$$

---

## 3. System Architecture & Structural Granularity

### 3.1 Module Ownership vs. Element-Wise Application

LC-QAT establishes a structural separation between codebook allocation and index application:

```
                              LCQATLinear Module
                                      │
     ┌────────────────────────────────┴────────────────────────────────┐
     ▼                                                                 ▼
Weight Codebook C_W                                           Activation Codebook C_A
(Owned at Module Level)                                       (Owned at Module Level)
Size: K_W floats (~120 B)                                     Size: K_A floats (~120 B)
     │                                                                 │
     │ Applied Element-Wise Across                                     │ Applied Element-Wise Across
     ▼                                                                 ▼
Weight Tensor W [4096 x 4096]                                 Activation Tensor X [B, S, 4096]
Index Tensor Q_W (uint8)                                      Index Tensor Q_A (uint8)

```

1. **Module-Level Ownership:** A single `nn.Linear` layer owns exactly **one** weight codebook instance and **one** activation codebook instance. For a 7.5B parameter Transformer with ~200 linear layers, the total system-wide codebook memory footprint is negligible ($\sim 200 \times 2 \times 255 \times 4\text{ bytes} \approx 400\text{ KB}$).
2. **Element-Wise Independent Indexing:** Every weight $W_{ij}$ and activation $X_t$ receives an independent discrete index $Q_{ij} \in \{0, \dots, K-1\}$.
3. **Massive Gradient Pooling:** Millions of individual tensor elements aggregate their loss gradients into the single module-level codebook, producing a smooth, low-variance statistical signal for interval step optimization.

### 3.2 Theoretical Grounding & Integration

#### Alignment with Finite Scalar Quantization (FSQ)

FSQ demonstrates that independent 1D scalar quantization avoids the geometric collapse of vector quantization spaces. By maintaining strict 1D monotonicity ($c_k < c_{k+1}$) and an explicit zero anchor, LC-QAT eliminates codebook collapse by mathematical design, avoiding commitment losses or cluster resets.

#### Knowledge Distillation (KD) Anchoring (Polino et al.)

Sub-byte quantization ($K=3, 15$) compresses the discrete loss landscape into sharp local minima. LC-QAT anchors student QAT optimization using a KL-divergence loss evaluated against an unquantized FP32/BF16 teacher model:

$$\mathcal{L}_{\text{KD}} = \tau_{\text{KD}}^2 \, D_{\text{KL}}\left( \text{softmax}\left(\frac{Z_{\text{teacher}}}{\tau_{\text{KD}}}\right) \parallel \text{softmax}\left(\frac{Z_{\text{student}}}{\tau_{\text{KD}}}\right) \right)$$

$$\mathcal{L}_{\text{total}} = (1 - \alpha) \mathcal{L}_{\text{CE}}(Y, \hat{Y}_{\text{quant}}) + \alpha \mathcal{L}_{\text{KD}}$$

#### Selective Layer Freezing (EfQAT)

Updating shadow weights and codebook parameters across multi-billion parameter models spikes VRAM usage due to Adam optimizer momentum/variance states ($2 \times \text{FP32}$ per parameter). Following EfQAT efficiency strategies, LC-QAT implements **selective layer freezing**: middle transformer layers exhibiting stable activation distributions have their codebook updates and weight gradients frozen after initial warmup epochs, restricting active backward passes to critical outlier layers (e.g., embedding projections, attention keys/queries, and output heads).

---

## 4. Hardware Execution Engine & LUT Accelerators

LC-QAT avoids complex bit-sliced or popcount-based boolean execution kernels. Instead, deployment execution relies on **L1 data cache-resident Look-Up Tables (LUTs)** combined with hardware Fused Multiply-Add (FMA) instructions.

```
Packed Memory (RAM)               L1 Data Cache                      CPU/GPU Registers
┌──────────────────┐           ┌──────────────────┐               ┌──────────────────┐
│ uint8 Indices    │ ──Fetch──►│ FP32 Codebook    │ ──Translate──►│ FP32 ALU Vector  │
│ (Weights & Acts) │           │ (1 KB per Layer) │               │ Execution (FMA)  │
└──────────────────┘           └──────────────────┘               └──────────────────┘

```

### 4.1 Storage Hierarchy & Dequantize-on-Fetch

1. **RAM Storage:** Weights and activations are stored in system RAM as packed unsigned integers (`uint8` for $K \le 255$, packed 4-bit nibbles for $K=15$).
2. **L1 Cache Residency:** Layer codebooks ($\approx 1\text{ KB}$ per layer) remain locked inside the CPU/GPU L1 data cache.
3. **Execution Pipeline:** Execution units fetch packed byte indices from RAM, translate them into FP32 registers via L1 cache lookups in 1 CPU cycle, and execute standard hardware FMA vector dot products.

### 4.2 2D Fused Multiplication LUTs ($K_W=15, K_A=15$)

When both weights and activations use 4-bit codebooks ($K=15$), the execution engine bypasses online floating-point multiplications entirely. A combined 4-bit weight index $q_w$ and 4-bit activation index $q_a$ yield only $15 \times 15 = 225$ possible multiplication outcomes.

During model loading, the C++ runtime pre-computes a 256-entry floating-point table for the layer:

$$\mathbf{LUT}_{\text{mul}}[ (q_w \ll 4) \mid q_a ] = c_{\text{weight}}[q_w] \times c_{\text{activation}}[q_a]$$

At runtime, the execution kernel uses the combined 8-bit index byte to fetch the pre-calculated product directly from the L1 cache, executing matrix multiplication via direct vector additions.

---

## 5. PyTorch Engineering & Reference Implementation

### 5.1 Asymmetric Learned Codebook Primitive (`AsymmetricLearnedCodebook`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLearnedCodebook(nn.Module):
    """
    Parametric 1D Scalar Codebook supporting asymmetric splits (M_neg + 1 + M_pos = K).
    Anchors index M_neg strictly to FP32 0.0 and enforces monotonicity via softplus steps.
    """

    def __init__(
        self,
        m_neg: int = 127,
        m_pos: int = 127,
        init_min: float = -1.0,
        init_max: float = 1.0,
    ):
        super().__init__()
        self.m_neg = m_neg
        self.m_pos = m_pos
        self.K = m_neg + 1 + m_pos

        # Initialize negative steps (if M_neg > 0)
        if self.m_neg > 0:
            init_neg = torch.linspace(0, abs(init_min), self.m_neg + 1)[1:]
            neg_deltas = init_neg - torch.cat([torch.tensor([0.0]), init_neg[:-1]])
            self.raw_neg_deltas = nn.Parameter(torch.log(torch.exp(neg_deltas) - 1.0))
        else:
            self.register_parameter("raw_neg_deltas", None)

        # Initialize positive steps (if M_pos > 0)
        if self.m_pos > 0:
            init_pos = torch.linspace(0, init_max, self.m_pos + 1)[1:]
            pos_deltas = init_pos - torch.cat([torch.tensor([0.0]), init_pos[:-1]])
            self.raw_pos_deltas = nn.Parameter(torch.log(torch.exp(pos_deltas) - 1.0))
        else:
            self.register_parameter("raw_pos_deltas", None)

        # Buffer for compiled inference LUT
        self.register_buffer(
            "compiled_codebook",
            torch.empty(self.K, dtype=torch.float32),
            persistent=True,
        )
        self.is_compiled = False

    def get_codebook(self) -> torch.Tensor:
        if not self.training and self.is_compiled:
            return self.compiled_codebook

        zero = torch.tensor(
            [0.0], device=self.compiled_codebook.device, dtype=torch.float32
        )

        if self.m_neg > 0:
            neg_steps = F.softplus(self.raw_neg_deltas)
            neg_side = -torch.cumsum(neg_steps, dim=0)
            neg_part = torch.flip(neg_side, dims=[0])
        else:
            neg_part = torch.empty(0, device=zero.device, dtype=torch.float32)

        if self.m_pos > 0:
            pos_steps = F.softplus(self.raw_pos_deltas)
            pos_part = torch.cumsum(pos_steps, dim=0)
        else:
            pos_part = torch.empty(0, device=zero.device, dtype=torch.float32)

        cb = torch.cat([neg_part, zero, pos_part])

        if not self.training:
            self.compiled_codebook.copy_(cb)
            self.is_compiled = True

        return cb

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codebook = self.get_codebook().to(torch.float32)
        x_fp32 = x.to(torch.float32)

        # Calculate decision midpoints
        midpoints = (codebook[:-1] + codebook[1:]) / 2.0
        indices = torch.bucketize(x_fp32.detach(), midpoints)
        x_dequant = codebook[indices]

        # Dual-Gradient Autograd formulation: f(x) + (x - detach(x))
        x_q = x_dequant + (x_fp32 - x_fp32.detach())

        target_dtype = torch.uint8 if self.K <= 255 else torch.int32
        return x_q.to(x.dtype), indices.to(target_dtype)

    def register_gradient_scaling_hook(self, tensor_numel: int):
        """Attaches 1/sqrt(N) gradient scaling hooks to step parameters."""
        scale = 1.0 / (tensor_numel**0.5)
        if self.raw_neg_deltas is not None:
            self.raw_neg_deltas.register_hook(lambda grad: grad * scale)
        if self.raw_pos_deltas is not None:
            self.raw_pos_deltas.register_hook(lambda grad: grad * scale)

    def compile_for_inference(self):
        """Free trainable parameters and lock codebook into a static buffer."""
        cb = self.get_codebook().detach().clone()
        self.compiled_codebook.copy_(cb)
        self.is_compiled = True

        if hasattr(self, "raw_neg_deltas") and self.raw_neg_deltas is not None:
            del self.raw_neg_deltas
            self.raw_neg_deltas = None
        if hasattr(self, "raw_pos_deltas") and self.raw_pos_deltas is not None:
            del self.raw_pos_deltas
            self.raw_pos_deltas = None
```

### 5.2 Dual-Quantized Linear Operator (`LCQATLinear`)

```python
class LCQATLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear executing dual scalar quantization
    on weights and activations using module-owned codebooks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        m_neg_w: int = 1,
        m_pos_w: int = 1,
        m_neg_a: int = 7,
        m_pos_a: int = 7,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.weight_quantizer = AsymmetricLearnedCodebook(
            m_neg=m_neg_w, m_pos=m_pos_w, init_min=-0.1, init_max=0.1
        )
        self.act_quantizer = AsymmetricLearnedCodebook(
            m_neg=m_neg_a, m_pos=m_pos_a, init_min=-2.0, init_max=2.0
        )

        # Attach gradient scaling hooks based on tensor sizes
        self.weight_quantizer.register_gradient_scaling_hook(out_features * in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q, _ = self.act_quantizer(x)
        w_q, _ = self.weight_quantizer(self.weight)
        return F.linear(x_q, w_q, self.bias)

    @classmethod
    def from_float(
        cls,
        mod: nn.Linear,
        m_neg_w: int = 1,
        m_pos_w: int = 1,
        m_neg_a: int = 7,
        m_pos_a: int = 7,
    ) -> "LCQATLinear":
        new_mod = cls(
            mod.in_features,
            mod.out_features,
            bias=(mod.bias is not None),
            m_neg_w=m_neg_w,
            m_pos_w=m_pos_w,
            m_neg_a=m_neg_a,
            m_pos_a=m_pos_a,
        )
        new_mod.weight.data.copy_(mod.weight.data)
        if mod.bias is not None:
            new_mod.bias.data.copy_(mod.bias.data)

        w_max = mod.weight.data.abs().max().item()
        new_mod.weight_quantizer = AsymmetricLearnedCodebook(
            m_neg=m_neg_w, m_pos=m_pos_w, init_min=-w_max, init_max=w_max
        )
        new_mod.weight_quantizer.register_gradient_scaling_hook(
            mod.in_features * mod.out_features
        )
        return new_mod
```

### 5.3 Export Pipeline

```python
@torch.no_grad()
def export_lcqat_checkpoint(model: nn.Module, export_path: str):
    """
    Compiles codebooks into static buffers, strips continuous shadow parameters,
    and exports a compact index state_dict.
    """
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, LCQATLinear):
            module.weight_quantizer.compile_for_inference()
            module.act_quantizer.compile_for_inference()

            # Unpack discrete indices for weight parameters
            _, w_indices = module.weight_quantizer(module.weight)
            module.register_buffer(
                "packed_weight_indices", w_indices.to(torch.uint8), persistent=True
            )

            # Delete shadow weight parameter from RAM
            del module.weight

    torch.save(model.state_dict(), export_path)
```

---

## 6. Comparison matrix

| Feature / Metric        | Standard Uniform QAT                      | Vector Quantization (VQ-VAE)                  | LC-QAT Framework                                  |
| ----------------------- | ----------------------------------------- | --------------------------------------------- | ------------------------------------------------- |
| **Quantization Domain** | 1D Uniform Grid ($\Delta = \text{const}$) | $\mathbb{R}^D$ Multi-Dimensional Vector Space | **1D Non-Uniform Scalar ($\mathbb{R}^1$)**        |
| **Codebook Spacing**    | Fixed Linear                              | Free Vector Clusters                          | **Learned Softplus Steps ($\rho^+, \rho^-$)**     |
| **Zero Preservation**   | Approximate / Scale Offset                | None                                          | **Strict $0.0$ Anchor at Index $M_{\text{neg}}$** |
| **Codebook Collapse?**  | No                                        | High (Requires Commitment Loss/EMA)           | **No (Prevented by 1D Topology)**                 |
| **Ownership Scope**     | Per-Layer Scale $S$ / Zero $Z$            | Global Embedding Dictionary                   | **Module-Level Codebook, Element-Wise Indexing**  |
| **Inference Hardware**  | INT4/INT8 ALU Math                        | Distance Metric Searching                     | **FP32 Dequant-on-Fetch via L1 LUT + FMA**        |

---

# Part II: symbiosis and measurements (from combination.md §§1-4)

Part II states how DiffusionBlocks, LC-QAT, and SparseProp combine and what the four-arm run measured. It keeps the design argument separate from the measured results.

> NOTE (2026-10-03, carried from combination.md): §§7-9 predate the removed HANDOFF_symbiosis.md (deleted 2026-10-05; see git history); paths and names there may be stale.

## 7. The unified architecture stack

Block-wise diffusion training, learnable 1D quantization, and sparse backpropagation combine into one engine for training and deploying LLMs. The three methods work at orthogonal levels of the stack: macro-architecture (depth/memory), meso-architecture (precision/bit-width), and micro-execution (compute/sparsity). Their benefits stack multiplicatively instead of cancelling.

Each method owns one bottleneck:

- **Macro (DiffusionBlocks):** the transformer splits into $B$ independent blocks, each trained as a denoiser over its own noise range. Memory falls by ~$B$, since gradients cover one block at a time.
- **Meso (LC-QAT):** each block's linear layers become dual-quantized `LCQATLinear` modules: weights and activations on non-uniform 1D scalar codebooks with softplus step increments ($K=3$ ternary or $K=15$ 4-bit). The activation between two quantized layers is a learned radial-basis map over a learnable knot grid (see §9 and Part III).
- **Micro (SparseProp):** unstructured sparse weights pack into CSR/CSC, and the backward pass uses AVX2 SIMD to skip multiplications and gradient work for weights pruned to exactly zero.

## 8. Harmonizing the mechanics (why they do not conflict)

Quantization, sparsity, and block-wise training often interfere: quantization noise destroys sparse zeros, and local training destabilizes global grids. Three bridges resolve this.

### 8.1 The LC-QAT zero anchor enables SparseProp

Standard uniform quantization applies scale plus zero-point offset, turning exact structural zeros into approximate floats. SparseProp then has nothing to skip.

LC-QAT avoids this with an asymmetric split codebook ($K = M_{\text{neg}} + 1 + M_{\text{pos}}$) whose index $M_{\text{neg}}$ sits at FP32 $0.0$, fixed and exempt from gradient updates. Pruned weights stay exact zeros, pack into CSR/CSC, and the backward pass scales with layer density.

### 8.2 Diffusion denoising drives LC-QAT dual-gradients

Standard QAT adapts codebooks to the global task loss. DiffusionBlocks trains on local denoising score matching without BPTT across blocks. LC-QAT's dual-gradient form ($\text{out} = f(x) + x - \text{detach}(x)$) does not care where the loss comes from: the active block receives noisy input $(x, y+\sigma\epsilon)$, the local loss backpropagates, the STE updates the continuous shadow weights, and the codebook steps ($\rho^+, \rho^-$) take sum-reduced gradients scaled by $1/\sqrt{N}$. Each block's codebook fits its own noise band's activation distribution.

### 8.3 Equi-probability partitioning balances sparse parameters

Each block handles an equal share of the training distribution's cumulative probability mass. Without this, a block stuck with a hard noise band could not take heavy sparsity without accuracy loss. Equal mass keeps the sparsity budget even across blocks.

## 9. The unified execution pipeline

### 9.1 Training (single-block forward/backward)

1. **Block + noise:** pick block $b$, sample $\sigma$ from $[\sigma_b, \sigma_{b-1}]$.
2. **Sparse forward:** feed $z_{\sigma}$ through the block; `LCQATLinear` layers discretize the active non-zero shadow weights onto the codebook.
3. **Loss:** $\mathcal{L} \leftarrow w(\sigma) \cdot Loss(\hat{y}, y)$.
4. **Sparse AVX2 backward:** compute $\partial L / \partial X$, $\partial L / \partial W$ over CSR; `vfmadd` touches non-zero entries only, skipping the $0.0$-anchored weights.
5. **Dual update:** sparse gradients update the non-zero shadow weights (STE) and the codebook boundaries ($\rho^+, \rho^-$).

### 9.2 Inference (sequential LUT execution)

1. **Sequential state machine:** generation starts from pure noise $z_0$ and passes through the blocks in order; one block is resident at a time.
2. **Dequantize-on-fetch:** the active block's parameters arrive from RAM as packed 4-bit indices ($K_W=15$); L1-resident 2D fused-multiplication LUTs translate them.
3. **Zero-skipping FMA:** the kernel fetches the combined 8-bit weight/activation index pair, reads the precomputed FP32 product, and skips fetch and lookup entirely for structural-zero indices.

## 10. The four-combination measurement (CPU, measured)

Sections 7-9 are the design argument. This section is what the build does, measured after the fact on a 4-core CPU host.

**Harness:** `runs/stackcompare.sh`, derived from `runs/runcpu.sh`. Four training runs at depth 6, `n_embd` 384, vocab 32768, device-batch 4, seq 512, 2 gradient-accumulation micro-steps, 30 iterations (~123k tokens). LC-QAT is on in every arm; exactly one switch separates each arm from its neighbour. Full output in [`dev/STACK_COMPARE.md`](STACK_COMPARE.md). (The source says "Four real training runs"; the word "real" here marks measured runs as opposed to projections, and the harness sentence above gives the checkable detail.)

| arm | DiffusionBlocks | LC-QAT | SparseProp | min val bpb | final loss | median step | tok/sec |
|---|---|---|---|---|---|---|---|
| `lcqat` | no | yes | no | 2.365218 | 8.375339 | 17599.8 ms | 232 |
| `lcqat_db` | yes | yes | no | 3.200110 | 0.130895 | 5718.8 ms | 716 |
| `lcqat_db_sp` | yes | yes | yes | 3.200110 | 0.620494 | 4876.2 ms | 839 |
| `lcqat_sp` | no | yes | yes | 2.364559 | 8.370389 | 15874.0 ms | 258 |

### 10.1 What each comparison can and cannot say

The arms form a 2x2, but **only the two diagonal pairs are clean**:

| comparison | changes | verdict |
|---|---|---|
| `lcqat_sp` vs `lcqat` | SparseProp only, plain LM | **-9.8%** step time |
| `lcqat_db_sp` vs `lcqat_db` | SparseProp only, denoiser | **-14.7%** step time |
| `lcqat_db` vs `lcqat` | DB *and* objective | timing only, **-67.5%** |
| `lcqat_db_sp` vs `lcqat_sp` | DB *and* objective | timing only, **-69.3%** |

The cross pairs change the objective too: `edm` trains a denoiser against a resampled $\sigma$, plain LM trains next-token cross-entropy. **Their losses are not comparable and are never pooled here.** Step times compare soundly because both arms move the same 4096 tokens per step.

### 10.2 The multiplicative claim nearly holds, slightly exceeded

Section 7 claims the benefits stack multiplicatively. From the two clean one-switch factors:

* DiffusionBlocks alone: 17599.8 / 5718.8 = **3.078x**
* SparseProp alone: 17599.8 / 15874.0 = **1.109x**
* Product: **3.412x**, predicting 5158 ms for the full stack
* Measured `lcqat_db_sp`: 4876 ms = **3.609x**, **5.5% faster than predicted**

The claim is not just unfalsified but slightly *exceeded*. The likely reason: the two mechanisms relieve different bottlenecks (activation memory and depth vs. backward FLOPs), so neither eats the other's saving. At 5.5% on a 30-iteration run, read this as "consistent with multiplicative", not proof.

### 10.3 SparseProp is not slower, and how that was shown

SparseProp was a removal candidate on this host. It survived; the path there is the reusable part.

**Measured, in order** (all at the sweep's own shapes):

| scope | dense | sparse | delta |
|---|---|---|---|
| `base_train` step, 6 samples each | 17352-17949 ms | 15904-16348 ms | **-10%** |
| in-process fwd+bwd, one micro-step | 7534 ms | 7592 ms | +0.8% |
| in-process `AdamW(fused=True)` alone | 251 ms | 245 ms | -2.4% |
| in-process backward onto existing grads | 7424 ms | 7733 ms | +4.2% |
| full step, 2 micro-steps, no dataloader | 17591 ms | 15839 ms | **-10%** |

**Five defects found and fixed on the way**; four were *speedups from doing less work*:

1. **AVX2 `nnz` kernels → masked dense GEMMs.** The CSR walk does 4x fewer MACs yet loses 10.9x forward / 4.7x backward: per-`nnz` gathers cannot vectorise like a packed micro-kernel. "Fewer MACs" was the wrong model. Exact, not approximate: pruned weights hold the LC-QAT zero anchor (§8.1), so dense form is algebraically identical.
2. **A differentiable gather on the weight view**, scattering gradients across the whole dense buffer (301 ms of a 532 ms step).
3. **A bypassed `_quantize`**, skipping the STE (486 ms) *and* the PRD 2.4 $1/\sqrt{N}$ codebook scale: codebooks trained $\sqrt{N}$ too hot.
4. **An `out_quantizer` re-parented but never called**: 24 layers trained with unquantized output.
5. **A dead learned activation LUT.** A manual gather bypassed `LearnableIndexLut.forward`, dropping the straight-through relaxation, so `--lcqat-lut-relaxation` and `--lcqat-act-body` **silently did nothing on every SparseProp run** and the table got no gradient.

Defects 4 and 5 ran *faster* than the code they replaced. The general trap: a path beating its replacement may just skip work the other owes. Reverting fix 5 changes 100% of activation elements (max abs 12.0), not rounding.

**The residual is unexplained and left that way.** The last table row reproduces the trainer's gap outside the trainer while in-process micro-benchmarks show parity. Ruled out: the optimizer's parameter list is byte-identical between arms (same roles, shapes, tensor counts), and profiler op counts match exactly (`mm` 111/111, `bucketize` 198/198, `index` 114/114). SparseProp skips no work and pays nothing extra in the optimizer. What remains (allocator state, thread scheduling) is a hypothesis, not a finding. The decision stands regardless: the claim is negative, and SparseProp is **not** slower on either pair.

### 10.4 Quality is a tie; this is not a quality result

Validation bpb is flat within each pair (2.365218 vs 2.364559 plain LM; 3.200110 vs 3.200110 denoiser). §§7-10 argue the methods compose, not that sparsity improves accuracy, and 30 iterations at depth 6 cannot show it. **This section is about cost, not quality.** Quality needs longer runs at a depth where sparsity bites.

---

# Part III: measured deviations appendix

Part III keeps one copy of the overlap. combination.md §5 and pseudoPaper Appendix A describe the same relaxation, anchor pin, knot precision, and body measurements. Appendix A is fuller, so it is the kept copy. The mapping below shows where each §5 claim lives in §11.

> Everything in §§1-6 above is the original specification. This appendix records a **measured deviation** from §2.2/§2.3 as implemented in this repo. It does not alter the spec's claims; it records where the implementation moved and why. Full detail was in `dev/HANDOFF_symbiosis.md` §10 (deleted 2026-10-05; see git history).

Overlap mapping (combination.md §5 → kept §11 location):

- §5 terminology box (proximity vs radial-basis) → §11.2 box, same text.
- §5 zero-anchor bridge (pin a knot at zero) → §11.3.
- §5 proximity-over-logits figures (30 vs 225 params, 6.24e-01 vs 9.97e-02) → §11.2 table and caveat.
- §5 knot precision (60 B to 15 B, bit-identical table) → §11.4.
- §5 activation body figures (0.48 vs 1.02, 0.67 with bypass) → §11.5 with audit status.
- §5 audit caveat block (three unverified constants) → the matching caveats in §11.2, §11.4, §11.5.
- §5 HANDOFF pointers → kept in §11.1, §11.5 as written.

Wording differences stay as written with provenance; this merge does not reconcile them:

- Strict monotonicity: §2.2 states the softplus construction guarantees strict order; §11.1 records the measured correction (non-decreasing; strictness needs distinct deltas). Both stay; §11.1 carries the status.
- Anchor wording: combination.md §5 says fp8 knots "preserve the exact `0.0` anchor"; §11.4 says no shipped knot sits at exactly `0.0` and `pin_mask` holds the anchor by pinning the level. Both wordings stay below with their source named.
- Body figures: combination.md §5 cites 0.48 vs 1.02 as a roughly 2x fit edge; §11.5 audits the same figures as an unestablished claim with an unidentified baseline. Both stay with their status labels.

## 11. Measured deviations from the spec

## 11.1 Where the implementation deviates from §2.2

§2.2 assembles the codebook by softplus prefix-summation and claims this "guarantees **strict** monotonicity $c_k < c_{k+1}$ by construction." That claim is false in two ways, both found by test:

1. Two equal latent deltas produce equal increments, hence **equal** levels. The true guarantee is monotone *non-decreasing*; strictness requires distinct deltas.
2. `softplus` saturates to the identity above $|\rho| \approx 30$ in **any** precision, so far-apart latents collapse to identical increments.

Neither affects the **exact zero anchor**, which *is* structural: it is an index assignment (`m_neg`), not a computed value, and holds unconditionally. That is the property §2.1 and SparseProp depend on, and it is intact.

## 11.2 The activation relaxation (new, §2.3 did not specify one)

The spec states the codebook but leaves the activation between two quantized layers unspecified. The implementation introduces a **proximity relaxation** over a learnable knot grid, selected by distance rather than by free logits:

$$\text{value}(x) = \sum_j \text{softmax}_j\!\Big(\frac{s^2}{(x - \text{knot}_j)^2 + \epsilon}\Big) \cdot \text{level}_j$$

where $s$ is the median local knot spacing (`PROXIMITY_SCALE_GRID_UNITS = 1.0` grid units). The scale factor is load-bearing rather than cosmetic: without it the nearest knot scores ~770 against ~48 for its neighbour, a softmax argument gap of ~700 that underflows to exactly 0 in FP32 and turns the selector back into a hard argmax with a zero knot gradient, the same saturation cliff this relaxation replaced the free-logit one to avoid.

> **Terminology (unified 2026-10-01).** "Radial basis" and "proximity" name **two different selectors that share one formula shape**. Throughout the `dev/` docs, **proximity relaxation** is the $K_{\text{in}} \to K_{\text{out}}$ *index-table* selector (`learnable_lut.py`, `--lcqat-lut-relaxation`), and **radial-basis body** is the *activation function* selector (`activation.py`, `--lcqat-act-body`). Shared definition: both assign by inverse-square distance to a learnable knot grid.

Measured against a free $(K_{\text{in}} \times K_{\text{out}})$ logit matrix at $K=15$:

| | free logits | proximity |
|---|---|---|
| parameters | 225 | **30** |
| max abs gradient at init | 9.97e-02 | **6.24e-01** ⚠️ |

⚠️ **Audit caveat (2026-10-01): the gradient magnitudes did not re-measure.** The parameter counts (225 vs 30) and the free-logit figure `9.97e-02` **reproduce exactly**. The proximity figure `6.24e-01` did not: it measures `1.0` on a symmetric 15-level codebook with an exact zero anchor, and `2.0` against a logits side of `3.5\times10^{-2}$ under the repo's own probe. The *direction* of the claim holds in both: the distance selector's gradient is larger, but the magnitude is configuration-sensitive and unconfirmed. The shipped test asserts only `prox_grad > logits_grad`, never a constant, so it cannot confirm the cited figure either way.

The *qualitative* argument is the central part and is unaffected by the re-measurement: a free-logit softmax saturates; once it does, the gradient is identically zero in FP32 and the table freezes permanently. A distance-based selector has no such cliff. §2.4's $1/\sqrt{N}$ scaling assumes the codebook gradient is *informative*; a frozen table would silently void it.

## 11.3 The zero anchor does NOT transfer automatically to the activation

This is the single most important line in this appendix. §2.1's anchor holds for the codebook. It does **not** hold for a learned smooth activation, which is a convex combination of learned levels and has no reason to pass through $0.0$.

Measured on the unmodified distance-selector form: input $0.0$ returns $-4.47\times10^{-8}$.

In this repo that is not a rounding curiosity. §2.1's anchor is the entire mechanism by which structural zeros survive into CSR/CSC packing. An activation emitting $-4.47\times10^{-8}$ where the codebook emits $0.0$ breaks that contract at the exact junction of the two methods.

**Resolution:** pin a knot at exactly $0.0$ and pin its level to exactly $0.0$, forcing the convex combination through the origin. Verified to produce exact $0.0$, and, unlike a `where`-style bypass, it keeps the operation a pure convex combination, which is what allows it to survive export to the integer LUT.

## 11.4 Knot precision is free

Knots stored as `float8_e4m3fn` retain strict increase over $[-1,1]$ at $K=15$. The resolved index table is **bit-identical** to the FP32-knot version. Storage falls from 60 B to 15 B per layer at zero behavioural cost.

Note that fp8 is not what preserves the LC-QAT zero anchor: on the shipped knots there is no knot at exactly $0.0$ at all (they straddle it). The anchor is held by `pin_mask`, which pins the *level* at the input-codebook zero index to exactly $0.0$. fp8 storage must preserve the *ordering*, which it does.

⚠️ **Audit caveat (2026-10-01): the cited "max knot error 0.107" did not re-measure.** It is $0.0268$ on a plain `linspace(-1,1,15)` grid and $0.0357$ on the nudged knots the implementation actually produces (each a quarter-gap off its input position). The two central claims (strict ordering survives fp8, and the resolved table is bit-identical) both hold and are pinned by tests.

## 11.5 The activation body (`SmoothPWL`): the third measurement

The spec states no activation *function* body, only the codebooks. §11.2 covers the index-table selector; this records the separately-selectable body (`nanochat/lcqat/activation.py`, `--lcqat-act-body`). (combination.md §5 names these paths under `nanochat/models/quant/`; the paths above are the pseudoPaper Appendix A wording and stay as written.)

**Continuous fit, matched parameter budget (15 floats):**

| | max abs error on `relu²` |
|---|---|
| PWL (15 free values, linear interp) | **1.02** |
| `SmoothPWL` (5 knots + 5 weights + 5 bias) | **0.48** |
| `SmoothPWL` + 0-bypass | **0.67** |

⚠️ **This table did not reproduce** (2026-10-01 audit), and the direction of the comparison is not reproducible either: **the baseline the `1.02` refers to is unidentified**. Every PWL baseline tried fits `relu²` *better* than the radial basis (free-interpolation PWL $0.004$ vs shipped body $0.236$), the opposite of $1.02$-vs-$0.48$. Note also that the reference `SmoothPWL` in `nanochat/lcqat/reference/simple.py` has **no `fit` method**, so `0.48` could only have come from the `activation.py` port's `fit_from_callable`; the reference class was never the measured subject. Treat the 2× advantage as an **unestablished claim** pending the per-preset re-measurement the removed HANDOFF §10.4-D required (file deleted 2026-10-05; see git history).

Keeping `index`, `weights` and `bias` all free `nn.Parameter` is **not** in question: the basis is a strictly richer family than a fixed grid, and the shipped tests assert all three stay free. Only the *magnitude* of the fit advantage is unresolved.

## 11.6 Bias quantization: capability shipped, measured, default off

The spec's `LCQATLinear` (§5.2) quantizes weights and activations; bias is consumed in FP32. Quantizing bias is orthogonal to everything above and costs one more codebook per layer.

> **Status as of 2026-10-01: implemented and measured, default off.** The `quantize_bias` path (`nanochat/lcqat/bias_quant.py`, 46 tests in `tests/test_lcqat_bias_quant.py`) is opt-in, so the shipped path is unchanged and existing checkpoints stay bit-identical. It requires `bias=True` and is guarded against the packed-buffer export path. §5.2's operator as written still consumes bias in FP32 unless the flag is set.

**Measured** (`scripts/lcqat_ablation.py --experiment bias_quant --seeds 8`; rows `bias_quant_nmse`, `bias_quant_cost`, `bias_quant_liveness` in `dev/LEADERBOARD.md`):

| quantity | value |
|---|---|
| bias NMSE, FP32 baseline | 0 (it round-trips through nothing, so it scores 0 exactly) |
| bias NMSE, quantized at `K=15` | **0.0130425** over 8 seeds |
| codebook levels actually hit | 14 / 15 |
| extra parameters per layer | **14** (one shared 1-D table, `K=15`, independent of `out_features`) |
| extra storage per layer | **116 B** (FP32) |
| codebook liveness | gradient live; L2 movement 1.73e-04 over 5 steps, against a 5.77e-07 ULP floor |

The codebook is initialized from the actual bias before measuring (`init_from_tensor()`), because the quantizer's default act_init span (−2..2, fitted to relu² outputs) is ~40× wider than a std-0.02 bias. Without that, every entry buckets onto the zero anchor, 1 level is hit, the error is exactly 1.0 and the gradient is identically zero, a false "does not train". `from_float()` performs the same initialization, so retrofit takes this path automatically.

Liveness is gated on **both** a non-zero gradient and movement at or above the ULP floor. The gradient alone would call the §9.8 dead-codebook case live; movement alone would call a codebook that is merely too small to update at this learning rate dead.

**What this does and does not establish.** It establishes that a 15-level bias codebook reconstructs a bias vector to 1.3e-2 NMSE for 14 parameters / 116 B per layer, and that the codebook trains. It does **not** establish that encoding bias is worth doing on a trained model: the micro-probe trains only the codebook, and the end-to-end effect on loss is not measured. The harness deliberately gates liveness only, not the error value: encoding either a pass or a fail threshold would be a check that can only pass. The cost row is a price tag, not a comparison anyone can win.
