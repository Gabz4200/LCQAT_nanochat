# LC-QAT: Learnable Codebook Quantization-Aware Training and LUT-Accelerated Inference for Large Language Models

## Abstract

Standard uniform Quantization-Aware Training (QAT) forces continuous model parameters onto rigid linear grids, inducing severe accuracy degradation at sub-byte and low-bit regimes ($K \le 15$, $\le 4\text{ bits/param}$). Conversely, multi-dimensional Vector Quantization (VQ) introduces the curse of dimensionality, suffering from codebook collapse and requiring complex auxiliary losses. We present **Learnable Codebook Quantization-Aware Training and Inference with LUTs (LC-QAT)**, an end-to-end framework that reformulates Large Language Models (LLMs) into adaptive, non-uniform 1D discrete state machines. LC-QAT operates strictly in 1D scalar space ($\mathbb{R}^1$) with module-level codebook ownership and element-wise independent indexing.

By parameterizing step increments via softplus-transformed latent variables, LC-QAT guarantees 1D monotonicity ($c_k < c_{k+1}$) and anchors an exact structural zero level ($c_{M_{\text{neg}}} = 0.0$) without manual projection steps. We derive a dual-gradient autograd formulation ($\text{out} = f(x) + x - \text{detach}(x)$) that simultaneously updates continuous shadow weights via Straight-Through Estimation (STE) and codebook boundaries via task loss gradients scaled by $1/\sqrt{N}$. At inference, LC-QAT eschews complex bit-sliced boolean kernels, employing L1 cache-resident 1D and 2D Look-Up Tables (LUTs) with standard Fused Multiply-Add (FMA) hardware instructions.

---

## 1. Introduction

Quantization is the primary mechanism for mitigating the memory bandwidth bottleneck of Large Language Models (LLMs) during autoregressive generation. However, pushing parameters and activations into sub-byte regimes ($K=3$ ternary, $K=15$ 4-bit) reveals fundamental breakdown points in existing compression paradigms.

### 1.1 Limitations of Uniform Quantization

Standard uniform quantization maps continuous real numbers to integer buckets using a linear affine transformation:

$$W_q = \text{clamp}\left( \text{round}\left( \frac{W}{S} \right) + Z, \, Q_{\min}, \, Q_{\max} \right)$$

This forces all quantization centroids to be strictly equidistant ($\Delta = \text{const}$). Because LLM weights and activations display non-uniform, kurtotic distributions with heavy tails and asymmetric post-activation skewness (e.g., post-SiLU or post-GELU distributions), uniform grids waste quantization buckets on low-density regions while coarse-graining high-density central peaks.

### 1.2 Limitations of High-Dimensional Vector Quantization

Vector Quantization (VQ) techniques project multi-dimensional feature blocks ($\mathbb{R}^D$) into discrete codebook vectors. Operating in high-dimensional space subjects the optimization process to the curse of dimensionality. Codebook entries frequently drift into unassigned latent voids—a phenomenon known as _codebook collapse_—requiring heuristic mechanisms such as commitment losses ($\Vert{}z_e(x) - \text{sg}[e]\Vert{}_2^2$), Exponential Moving Average (EMA) cluster resets, or entropy penalties.

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

Index $M_{\text{neg}}$ is anchored strictly to **FP32 $0.0$**. Exact structural zeros—such as sparse activation outputs—incur zero quantization noise and consume no dynamic range from non-zero levels.

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
    def __init__(self, m_neg: int = 127, m_pos: int = 127, init_min: float = -1.0, init_max: float = 1.0):
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
            self.register_parameter('raw_neg_deltas', None)

        # Initialize positive steps (if M_pos > 0)
        if self.m_pos > 0:
            init_pos = torch.linspace(0, init_max, self.m_pos + 1)[1:]
            pos_deltas = init_pos - torch.cat([torch.tensor([0.0]), init_pos[:-1]])
            self.raw_pos_deltas = nn.Parameter(torch.log(torch.exp(pos_deltas) - 1.0))
        else:
            self.register_parameter('raw_pos_deltas', None)

        # Buffer for compiled inference LUT
        self.register_buffer("compiled_codebook", torch.empty(self.K, dtype=torch.float32), persistent=True)
        self.is_compiled = False

    def get_codebook(self) -> torch.Tensor:
        if not self.training and self.is_compiled:
            return self.compiled_codebook

        zero = torch.tensor([0.0], device=self.compiled_codebook.device, dtype=torch.float32)

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
        scale = 1.0 / (tensor_numel ** 0.5)
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
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 m_neg_w: int = 1, m_pos_w: int = 1,
                 m_neg_a: int = 7, m_pos_a: int = 7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.weight_quantizer = AsymmetricLearnedCodebook(m_neg=m_neg_w, m_pos=m_pos_w, init_min=-0.1, init_max=0.1)
        self.act_quantizer = AsymmetricLearnedCodebook(m_neg=m_neg_a, m_pos=m_pos_a, init_min=-2.0, init_max=2.0)

        # Attach gradient scaling hooks based on tensor sizes
        self.weight_quantizer.register_gradient_scaling_hook(out_features * in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q, _ = self.act_quantizer(x)
        w_q, _ = self.weight_quantizer(self.weight)
        return F.linear(x_q, w_q, self.bias)

    @classmethod
    def from_float(cls, mod: nn.Linear, m_neg_w: int = 1, m_pos_w: int = 1,
                   m_neg_a: int = 7, m_pos_a: int = 7) -> "LCQATLinear":
        new_mod = cls(mod.in_features, mod.out_features, bias=(mod.bias is not None),
                      m_neg_w=m_neg_w, m_pos_w=m_pos_w, m_neg_a=m_neg_a, m_pos_a=m_pos_a)
        new_mod.weight.data.copy_(mod.weight.data)
        if mod.bias is not None:
            new_mod.bias.data.copy_(mod.bias.data)

        w_max = mod.weight.data.abs().max().item()
        new_mod.weight_quantizer = AsymmetricLearnedCodebook(
            m_neg=m_neg_w, m_pos=m_pos_w, init_min=-w_max, init_max=w_max
        )
        new_mod.weight_quantizer.register_gradient_scaling_hook(mod.in_features * mod.out_features)
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
            module.register_buffer("packed_weight_indices", w_indices.to(torch.uint8), persistent=True)

            # Delete shadow weight parameter from RAM
            del module.weight

    torch.save(model.state_dict(), export_path)

```

---

## 6. Comparative Paradigm Matrix

| Feature / Metric        | Standard Uniform QAT                      | Vector Quantization (VQ-VAE)                  | LC-QAT Framework                                  |
| ----------------------- | ----------------------------------------- | --------------------------------------------- | ------------------------------------------------- |
| **Quantization Domain** | 1D Uniform Grid ($\Delta = \text{const}$) | $\mathbb{R}^D$ Multi-Dimensional Vector Space | **1D Non-Uniform Scalar ($\mathbb{R}^1$)**        |
| **Codebook Spacing**    | Fixed Linear                              | Free Vector Clusters                          | **Learned Softplus Steps ($\rho^+, \rho^-$)**     |
| **Zero Preservation**   | Approximate / Scale Offset                | None                                          | **Strict $0.0$ Anchor at Index $M_{\text{neg}}$** |
| **Codebook Collapse?**  | No                                        | High (Requires Commitment Loss/EMA)           | **No (Prevented by 1D Topology)**                 |
| **Ownership Scope**     | Per-Layer Scale $S$ / Zero $Z$            | Global Embedding Dictionary                   | **Module-Level Codebook, Element-Wise Indexing**  |
| **Inference Hardware**  | INT4/INT8 ALU Math                        | Distance Metric Searching                     | **FP32 Dequant-on-Fetch via L1 LUT + FMA**        |
