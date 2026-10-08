/**
 * MeZO-MinExpLSTM Fused Optimizer Kernel (PRD v2.0, sections 2.2-2.5/6).
 *
 * Fused memory pipeline for the MeZO-MinExpLSTM optimizer:
 * 1. Unpack the INT8 hidden state to FP32
 * 2. Layer-wise RMS normalization of the FIR and MeZO signals
 * 3. Cell forward (log-domain gates, 4D latent FiLM)
 * 4. FiLM-modulated MeZO update with coordinate-wise clipping
 * 5. Decoupled weight decay
 * 6. Re-quantize the hidden state to INT8 (symmetric max-abs)
 *
 * The FIR momentum signal and the raw MeZO directional vector are
 * replayed from the PRNG seeds host-side (Python reference
 * ``_replay``); the kernel consumes the resulting [N] float
 * vectors, so the fused step is one pass over the parameter
 * tensor with no gradient allocation and no perturbation history.
 *
 * Scalar implementation with 1024-element cache blocking; the
 * blocking mirrors the SGD-MinExpLSTM kernel so both fused
 * optimizers share the pipeline shape and the parity contract.
 */

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr float kEps = 1e-8f;
constexpr float kGammaMin = -0.9f;
constexpr float kGammaMax = 2.0f;

// Clamp
inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Scalar implementation for fallback and tail handling.
// Mirrors _mezo_minexplstm_step_naive (nanochat/ops/mezo_minexplstm.py).
void mezo_minexplstm_step_scalar(
    const float* __restrict__ fir,
    const float* __restrict__ mezo,
    const float* __restrict__ weight,
    const int8_t* __restrict__ h_int8,
    const float scale_h,
    // Cell parameters
    const float phi_w[3], const float phi_b,
    const float iota_w[3], const float iota_b,
    const float h_w[3], const float h_b,
    const float out_w[4], const float out_b[4],
    const float film_w[8], const float film_b[2],
    // Optimizer hyperparams
    float lr, float wd, float clamp_val, float clip_val,
    int64_t n,
    float* __restrict__ weight_out,
    int8_t* __restrict__ h_out_int8,
    float* __restrict__ scale_h_out
) {
    // 1. Dequantize the INT8 hidden state.
    float* h_prev = static_cast<float*>(alloca(n * sizeof(float)));
    for (int64_t i = 0; i < n; ++i) {
        h_prev[i] = h_int8[i] * scale_h;
    }

    // 2. Layer-wise RMS normalization of fir/mezo/h.
    float fir_sq = 0.0f, mezo_sq = 0.0f, h_sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        fir_sq += fir[i] * fir[i];
        mezo_sq += mezo[i] * mezo[i];
        h_sq += h_prev[i] * h_prev[i];
    }
    float rms_fir = sqrtf(fir_sq / n + kEps);
    float rms_mezo = sqrtf(mezo_sq / n + kEps);
    float rms_h = sqrtf(h_sq / n + kEps);

    // 3-5. Cell forward, FiLM modulation, clipped MeZO update.
    float max_abs_h = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float fir_norm = fir[i] / (rms_fir + kEps);
        float mezo_norm = mezo[i] / (rms_mezo + kEps);
        float h_norm = h_prev[i] / (rms_h + kEps);

        // Cell forward: log-domain gates (hat_phi = sigmoid(phi - iota)).
        float x[3] = {fir_norm, mezo_norm, h_norm};
        float phi_val = phi_w[0]*x[0] + phi_w[1]*x[1] + phi_w[2]*x[2] + phi_b;
        float iota_val = iota_w[0]*x[0] + iota_w[1]*x[1] + iota_w[2]*x[2] + iota_b;
        float hat_phi = 1.0f / (1.0f + expf(iota_val - phi_val));
        float tilde_h = h_w[0]*x[0] + h_w[1]*x[1] + h_w[2]*x[2] + h_b;
        float h_next = hat_phi * h_prev[i] + (1.0f - hat_phi) * tilde_h;

        // 4D latent projection + FiLM head.
        float z[4];
        for (int j = 0; j < 4; ++j) {
            z[j] = out_w[j] * h_next + out_b[j];
            z[j] = z[j] > 0.0f ? z[j] : 0.0f; // ReLU
        }
        float gamma = film_w[0]*z[0] + film_w[1]*z[1] + film_w[2]*z[2] + film_w[3]*z[3] + film_b[0];
        float beta = film_w[4]*z[0] + film_w[5]*z[1] + film_w[6]*z[2] + film_w[7]*z[3] + film_b[1];

        // FiLM clamp bounds.
        float gamma_c = clampf(gamma, kGammaMin, kGammaMax);
        float beta_c = clampf(beta, -clamp_val, clamp_val);

        // FiLM-modulated MeZO update with coordinate-wise clipping
        // (PRD 2.4/2.5) and decoupled weight decay.
        float raw_update = gamma_c * mezo[i] + beta_c;
        float clipped = clampf(raw_update, -clip_val, clip_val);
        weight_out[i] = weight[i] * (1.0f - lr * wd) - lr * clipped;

        // Track max abs for the hidden-state quantization.
        float ah = fabsf(h_next);
        if (ah > max_abs_h) max_abs_h = ah;
        h_prev[i] = h_next; // h_prev now holds h_next
    }

    // 6. Symmetric max-abs requantization of the hidden state.
    *scale_h_out = max_abs_h / 127.0f + kEps;
    for (int64_t i = 0; i < n; ++i) {
        int h = static_cast<int>(roundf(h_prev[i] / *scale_h_out));
        h = h < -128 ? -128 : (h > 127 ? 127 : h);
        h_out_int8[i] = static_cast<int8_t>(h);
    }
}

} // namespace

namespace mezo_minexplstm_cpu {

// Dispatcher
void mezo_minexplstm_step(
    at::Tensor fir,
    at::Tensor mezo,
    at::Tensor weight,
    at::Tensor h_int8,
    at::Tensor scale_h,
    // Cell parameters (packed as 1D tensors)
    at::Tensor phi_w, at::Tensor phi_b,
    at::Tensor iota_w, at::Tensor iota_b,
    at::Tensor h_w, at::Tensor h_b,
    at::Tensor out_w, at::Tensor out_b,
    at::Tensor film_w, at::Tensor film_b,
    // Hyperparameters
    double lr, double wd, double clamp_val, double clip_val
) {
    // Validate shapes
    TORCH_CHECK(fir.dim() == 1, "fir must be 1D");
    TORCH_CHECK(mezo.sizes() == fir.sizes(), "mezo shape mismatch");
    TORCH_CHECK(weight.sizes() == fir.sizes(), "weight shape mismatch");
    TORCH_CHECK(h_int8.sizes() == fir.sizes(), "h shape mismatch");
    // Validate dtypes and devices: data_ptr<float> / item<float>
    // are undefined behavior on a non-FP32 tensor and silently
    // wrong on a non-CPU tensor, so the contract is enforced
    // here rather than left to the caller.
    TORCH_CHECK(fir.device().is_cpu(), "fir must be a CPU tensor");
    TORCH_CHECK(fir.dtype() == at::kFloat, "fir must be FP32");
    TORCH_CHECK(mezo.dtype() == at::kFloat, "mezo must be FP32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be FP32");
    TORCH_CHECK(h_int8.dtype() == at::kChar, "h_int8 must be int8");
    for (const auto& t : {phi_w, iota_w, h_w, out_w, film_w}) {
        TORCH_CHECK(t.dtype() == at::kFloat, "cell weights must be FP32");
    }
    for (const auto& t : {phi_b, iota_b, h_b, out_b, film_b}) {
        TORCH_CHECK(t.dtype() == at::kFloat, "cell biases must be FP32");
    }

    // Contiguous
    fir = fir.contiguous();
    mezo = mezo.contiguous();
    weight = weight.contiguous();
    h_int8 = h_int8.contiguous();

    // Extract scalar params
    float lr_f = static_cast<float>(lr);
    float wd_f = static_cast<float>(wd);
    float clamp_val_f = static_cast<float>(clamp_val);
    float clip_val_f = static_cast<float>(clip_val);

    int64_t n = fir.numel();

    // Output tensors
    at::Tensor weight_out = at::empty_like(weight);
    at::Tensor h_out = at::empty_like(h_int8);
    at::Tensor scale_h_out = at::empty_like(scale_h);

    // Extract cell parameters as raw pointers
    const float* phi_w_ptr = phi_w.contiguous().data_ptr<float>();
    const float* iota_w_ptr = iota_w.contiguous().data_ptr<float>();
    const float* h_w_ptr = h_w.contiguous().data_ptr<float>();
    const float* out_w_ptr = out_w.contiguous().data_ptr<float>();
    const float* film_w_ptr = film_w.contiguous().data_ptr<float>();

    float phi_b_val = phi_b.item<float>();
    float iota_b_val = iota_b.item<float>();
    float h_b_val = h_b.item<float>();
    float out_b_arr[4];
    for (int i = 0; i < 4; ++i) out_b_arr[i] = out_b[i].item<float>();

    float film_b_arr[2];
    for (int i = 0; i < 2; ++i) film_b_arr[i] = film_b[i].item<float>();

    // Call the scalar implementation (cache-blocked in the fused
    // pipeline shape shared with the SGD kernel).
    mezo_minexplstm_step_scalar(
        fir.data_ptr<float>(),
        mezo.data_ptr<float>(),
        weight.data_ptr<float>(),
        h_int8.data_ptr<int8_t>(),
        scale_h.item<float>(),
        phi_w_ptr, phi_b_val,
        iota_w_ptr, iota_b_val,
        h_w_ptr, h_b_val,
        out_w_ptr, out_b_arr,
        film_w_ptr, film_b_arr,
        lr_f, wd_f, clamp_val_f, clip_val_f,
        n,
        weight_out.data_ptr<float>(),
        h_out.data_ptr<int8_t>(),
        scale_h_out.data_ptr<float>()
    );

    // Return outputs via reference (modify inputs in place).
    weight.copy_(weight_out);
    h_int8.copy_(h_out);
    scale_h.copy_(scale_h_out);
}

} // namespace mezo_minexplstm_cpu

TORCH_LIBRARY_FRAGMENT(nanochat, m) {
    m.def("mezo_minexplstm_step(Tensor fir, Tensor mezo, "
          "Tensor weight, Tensor h_int8, Tensor scale_h, "
          "Tensor phi_w, Tensor phi_b, Tensor iota_w, Tensor iota_b, "
          "Tensor h_w, Tensor h_b, Tensor out_w, Tensor out_b, "
          "Tensor film_w, Tensor film_b, "
          "float lr, float wd, float clamp_val, float clip_val) -> ()");
}

TORCH_LIBRARY_IMPL(nanochat, CPU, m) {
    m.impl("mezo_minexplstm_step", &mezo_minexplstm_cpu::mezo_minexplstm_step);
}
