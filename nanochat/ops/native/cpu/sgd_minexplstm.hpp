/**
 * SGD-MinExpLSTM CPU Kernel Header.
 * Declares the dispatcher function for the fused optimizer step.
 */

#pragma once

#include <ATen/ATen.h>

namespace sgd_minexplstm_cpu {

// Dispatcher function - called from torch binding
void sgd_minexplstm_step(
    at::Tensor grad,
    at::Tensor weight,
    at::Tensor ema_fast_int8,
    at::Tensor ema_slow_int8,
    at::Tensor h_int8,
    at::Tensor scale_fast,
    at::Tensor scale_slow,
    at::Tensor scale_h,
    // Cell parameters
    at::Tensor phi_w, at::Tensor phi_b,
    at::Tensor iota_w, at::Tensor iota_b,
    at::Tensor h_w, at::Tensor h_b,
    at::Tensor out_w, at::Tensor out_b,
    at::Tensor film_w, at::Tensor film_b,
    at::Tensor w_beta, at::Tensor b_beta,
    // Hyperparameters
    double lr, double wd, double lam, double clamp_val
);

} // namespace sgd_minexplstm_cpu