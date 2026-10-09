"""Coordinate-wise MinExpLSTM cell with 4D latent FiLM modulation.

``SGDMinExpLSTMCellFiLM4D`` is the learned half of the
``SGD-MinExpLSTM-Optimizer`` (PRD v3.2). It operates scalar-by-scalar
over parameter coordinates -- no cross-channel mixing, so the whole
step vectorizes across SIMD lanes and GPU warps (PRD principle 1).

Math anchors:

* MinLSTM (Feng et al. 2024, arXiv:2410.01201): the gates depend only
  on the current input ``x``, never on ``h_{t-1}``, so the recurrence
  ``h_t = hat_phi * h_{t-1} + (1 - hat_phi) * tilde_h`` is a linear
  scan, not a sequential LSTM.
* Log-domain exponential gating (Horuz et al. 2025, arXiv:2508.03614):
  ``exp(a) / (exp(a) + exp(b))`` reduces to ``sigmoid(a - b)`` -- no
  overflow-prone ``exp()`` and no division.
* FiLM modulation (Perez et al. 2017, arXiv:1709.07871): the head emits
  a gain ``gamma`` and shift ``delta`` applied as
  ``(1 + gamma) * base + delta``.
* Input-dependent decay (Gu & Dao 2023, arXiv:2312.00752): the EMA
  horizons are data-dependent, here via the lambda-coupled predictor.
* Scale-invariant inputs (Metz et al. 2022, arXiv:2203.00845): every
  input is layer-wise RMS-normalized before it reaches the cell.

The FiLM head (``film_proj``) is zero-initialized, so a fresh cell
emits ``gamma == 0`` and ``delta == 0`` everywhere and the
optimizer's first step is exactly plain SGD -- the residual anchor
that keeps an untrained learned optimizer from destroying a model
(He et al. 2015, arXiv:1512.03385).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SGDMinExpLSTMCellFiLM4D(nn.Module):
    """Minimal exponential-gated LSTM cell with a FiLM head, elementwise.

    Input ``x = [g_norm, ema_fast_norm, ema_slow_norm]`` (all
    RMS-normalized, PRD section 3.1/3.3); hidden state ``h`` is a
    scalar per coordinate. The scalar hidden state projects to a 4D
    latent ``z = ReLU(W_out h + b_out)``, then to the 2D FiLM
    parameters ``[gamma, delta] = W_film z + b_film`` (PRD 3.4/3.5).
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 1,
        latent_dim: int = 4,
    ) -> None:
        super().__init__()
        if input_size != 3:
            raise ValueError(f"input_size must be 3, got {input_size}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        # Log-domain gate projections (PRD 3.4).
        self.phi_proj = nn.Linear(input_size, hidden_size)
        self.iota_proj = nn.Linear(input_size, hidden_size)
        self.h_proj = nn.Linear(input_size, hidden_size)
        # 1D -> 4D latent feature projection (PRD 3.5).
        self.out_proj = nn.Linear(hidden_size, latent_dim)
        # 4D -> 2D FiLM head (PRD 3.5), zero-initialized: the step-0 fallback safeguard (PRD principle 2).
        self.film_proj = nn.Linear(latent_dim, 2)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        # Lambda-coupled beta predictor (PRD 3.2): a single shared scalar affine map, one gate evaluation for both horizons.
        self.w_beta = nn.Parameter(torch.tensor([0.1]))
        self.b_beta = nn.Parameter(torch.tensor([0.0]))

    def predict_betas(self, g_norm: Tensor, lam: float = 0.10) -> tuple[Tensor, Tensor]:
        """Per-coordinate (fast, slow) EMA decays, lambda-coupled.

        ``u = w_beta * g_norm + b_beta`` through a hard sigmoid
        (AVX2-friendly, PRD 3.2), then
        ``beta_slow = 2*lam + (0.999 - 2*lam) * sigma`` and
        ``beta_fast = beta_slow - lam``. The fixed ``lam`` spacing
        keeps the two horizons distinct (no mode collapse) while a
        single predictor halves gate cost (PRD principle 3).
        """
        if not 0.0 < lam <= 0.4:
            raise ValueError(f"lam must be in (0, 0.4], got {lam}")
        u_t = g_norm * self.w_beta + self.b_beta
        sigma_u = torch.clamp(0.5 + 0.125 * u_t, 0.0, 1.0)
        min_beta_slow = 2.0 * lam
        beta_slow = min_beta_slow + (0.999 - min_beta_slow) * sigma_u
        beta_fast = beta_slow - lam
        return beta_fast, beta_slow

    def forward(self, x: Tensor, h_prev: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run one elementwise recurrence step.

        ``x``: normalized input ``[..., 3]``.
        ``h_prev``: previous hidden state ``[..., 1]``.
        Returns ``(gamma, delta, h_next)`` with ``gamma``/``delta``
        ``[...]`` and ``h_next`` ``[..., 1]``.
        """
        phi_val = self.phi_proj(x)
        iota_val = self.iota_proj(x)
        hat_phi = torch.sigmoid(phi_val - iota_val)
        tilde_h = self.h_proj(x)
        h_next = hat_phi * h_prev + (1.0 - hat_phi) * tilde_h
        z = torch.relu(self.out_proj(h_next))
        film_out = self.film_proj(z)
        gamma = film_out[..., 0]
        delta = film_out[..., 1]
        return gamma, delta, h_next


# Historical name from the pre-PRD-v3.2 design (single EMA + BetaNet MLP). It now names this cell; prefer ``SGDMinExpLSTMCellFiLM4D``.
MinExpLSTMCell = SGDMinExpLSTMCellFiLM4D

__all__ = ["MinExpLSTMCell", "SGDMinExpLSTMCellFiLM4D"]
