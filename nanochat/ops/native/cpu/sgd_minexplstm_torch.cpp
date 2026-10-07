/**
 * Torch binding for SGD-MinExpLSTM CPU kernel.
 */

#include <torch/extension.h>
#include "sgd_minexplstm.hpp"

TORCH_LIBRARY_FRAGMENT(nanochat, m) {
    m.def("sgd_minexplstm_step(Tensor grad, Tensor weight, "
            "Tensor ema_fast_int8, Tensor ema_slow_int8, Tensor h_int8, "
            "Tensor scale_fast, Tensor scale_slow, Tensor scale_h, "
            "Tensor phi_w, Tensor phi_b, Tensor iota_w, Tensor iota_b, "
            "Tensor h_w, Tensor h_b, Tensor out_w, Tensor out_b, "
            "Tensor film_w, Tensor film_b, "
            "Tensor w_beta, Tensor b_beta, "
            "float lr, float wd, float lam, float clamp_val) -> ()");
    m.impl("sgd_minexplstm_step", torch::kCPU, &sgd_minexplstm_cpu::sgd_minexplstm_step);
}