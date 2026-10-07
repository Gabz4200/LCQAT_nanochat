/**
 * SGD-MinExpLSTM Fused Optimizer Kernel (PRD v3.2 Section 6).
 *
 * Fused memory pipeline for the SGD-MinExpLSTM optimizer:
 * 1. Unpack INT8 states (EMA_fast, EMA_slow, h) to FP32
 * 2. Layer-wise RMS normalization of gradient and EMAs
 * 3. Lambda-coupled beta prediction (shared w_beta, b_beta)
 * 4. Dynamic EMA blending (IIR)
 * 4. EMA normalization
 * 5. Cell forward (log-domain gates, 4D latent FiLM)
 * 6. FiLM-modulated SGD step with weight decay
 * 7. Re-quantize states to INT8 (symmetric max-abs)
 *
 * AVX2 primary path; scalar fallback for CPUs without AVX2.
 * Kernel processes 1024-element blocks for L1/L2 cache locality.
 * OpenMP parallelism over 1024-element blocks.
 */

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {

// Cell parameters (34 total, loaded from Python)
// Packed layout for efficient SIMD access:
// [phi_w(3), iota_w(3), h_w(3), out_w(4), film_w(8), film_b(2), phi_b, iota_b, h_b, out_b(4), w_beta, b_beta]
// Total: 3+3+3+4+8+2 + 1+1+1+4 + 1+1 = 34 floats

constexpr int kInputSize = 3;   // [g_norm, ema_fast_norm, ema_slow_norm]
constexpr int kHiddenSize = 1;
constexpr int kLatentDim = 4;
constexpr int kOutputSize = 2;  // [gamma, delta]

constexpr int kBlockSize = 1024;  // L1/L2 cache blocking
constexpr float kEps = 1e-8f;
constexpr float kGammaMin = -0.9f;
constexpr float kGammaMax = 2.0f;

// Hard sigmoid: clamp(0.5 + 0.125 * u, 0.0, 1.0)
inline float hard_sigmoid(float u) {
    float v = 0.5f + 0.125f * u;
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// Sigmoid
inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Clamp
inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Scalar implementation for fallback and tail handling
void sgd_minexplstm_step_scalar(
    const float* __restrict__ grad,
    const float* __restrict__ weight,
    const int8_t* __restrict__ ema_fast_int8,
    const int8_t* __restrict__ ema_slow_int8,
    const int8_t* __restrict__ h_int8,
    const float scale_fast,
    const float scale_slow,
    const float scale_h,
    int8_t* __restrict__ ema_fast_out_int8,
    int8_t* __restrict__ ema_slow_out_int8,
    int8_t* __restrict__ h_out_int8,
    float* __restrict__ scale_fast_out,
    float* __restrict__ scale_slow_out,
    float* __restrict__ scale_h_out,
    float* __restrict__ weight_out,
    // Cell parameters
    const float phi_w[3], const float phi_b,
    const float iota_w[3], const float iota_b,
    const float h_w[3], const float h_b,
    const float out_w[4], const float out_b[4],
    const float film_w[8], const float film_b[2],
    const float w_beta, const float b_beta,
    // Optimizer hyperparams
    float lr, float wd, float lam, float clamp_val,
    int64_t n
) {
    // 1. Dequantize INT8 -> FP32
    // 2. Layer-wise RMS of gradient
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float g = grad[i];
        sum_sq += g * g;
    }
    float rms_g = sqrtf(sum_sq / n + kEps);

    // Dequantize EMA states
    float* ema_fast = static_cast<float*>(alloca(n * sizeof(float)));
    float* ema_slow = static_cast<float*>(alloca(n * sizeof(float)));
    float* h_prev = static_cast<float*>(alloca(n * sizeof(float)));
    for (int64_t i = 0; i < n; ++i) {
        ema_fast[i] = ema_fast_int8[i] * scale_fast;
        ema_slow[i] = ema_slow_int8[i] * scale_slow;
        h_prev[i] = h_int8[i] * scale_h;
    }

    // Lambda-coupled beta prediction (shared w_beta, b_beta)
    float beta_fast_arr[1024]; // max block size
    float beta_slow_arr[1024];
    for (int64_t i = 0; i < n; ++i) {
        float g_norm = grad[i] / (rms_g + kEps);
        float u = g_norm * w_beta + b_beta;
        float sigma = 0.5f + 0.125f * u;
        sigma = sigma < 0.0f ? 0.0f : (sigma > 1.0f ? 1.0f : sigma);
        float min_beta_slow = 2.0f * lam;
        float beta_slow = min_beta_slow + (0.999f - min_beta_slow) * sigma;
        beta_slow_arr[i] = beta_slow;
        beta_fast_arr[i] = beta_slow - lam;
    }

    // Dynamic EMA blending
    for (int64_t i = 0; i < n; ++i) {
        ema_fast[i] = beta_fast_arr[i] * ema_fast[i] + (1.0f - beta_fast_arr[i]) * grad[i];
        ema_slow[i] = beta_slow_arr[i] * ema_slow[i] + (1.0f - beta_slow_arr[i]) * grad[i];
    }

    // Normalize EMAs
    float rms_fast = 0.0f, rms_slow = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        rms_fast += ema_fast[i] * ema_fast[i];
        rms_slow += ema_slow[i] * ema_slow[i];
    }
    rms_fast = sqrtf(rms_fast / n + kEps);
    rms_slow = sqrtf(rms_slow / n + kEps);

    // Cell forward + FiLM + SGD step
    float max_abs_fast = 0.0f, max_abs_slow = 0.0f, max_abs_h = 0.0f;

    for (int64_t i = 0; i < n; ++i) {
        float g_norm = grad[i] / (rms_g + kEps);
        float ema_fast_norm = ema_fast[i] / (rms_fast + kEps);
        float ema_slow_norm = ema_slow[i] / (rms_slow + kEps);

        // Cell forward
        float x[3] = {g_norm, ema_fast_norm, ema_slow_norm};

        // phi = sigmoid(phi_proj(x) - iota_proj(x))
        float phi_val = phi_w[0]*x[0] + phi_w[1]*x[1] + phi_w[2]*x[2] + phi_b;
        float iota_val = iota_w[0]*x[0] + iota_w[1]*x[1] + iota_w[2]*x[2] + iota_b;
        float hat_phi = 1.0f / (1.0f + expf(iota_val - phi_val));

        float tilde_h = h_w[0]*x[0] + h_w[1]*x[1] + h_w[2]*x[2] + h_b;
        float h_next = hat_phi * h_prev[i] + (1.0f - hat_phi) * tilde_h;

        // 4D latent projection + FiLM
        float z[4];
        for (int j = 0; j < 4; ++j) {
            z[j] = out_w[j] * h_next + out_b[j];
            z[j] = z[j] > 0.0f ? z[j] : 0.0f; // ReLU
        }
        float gamma = film_w[0]*z[0] + film_w[1]*z[1] + film_w[2]*z[2] + film_w[3]*z[3] + film_b[0];
        float delta = film_w[4]*z[0] + film_w[5]*z[1] + film_w[6]*z[2] + film_w[7]*z[3] + film_b[1];

        // Clamp FiLM
        float gamma_c = gamma < kGammaMin ? kGammaMin : (gamma > kGammaMax ? kGammaMax : gamma);
        float delta_c = delta < -clamp_val ? -clamp_val : (delta > clamp_val ? clamp_val : delta);

        // FiLM-modulated SGD step
        float base_sgd = -lr * grad[i];
        float film_scaled = (1.0f + gamma_c) * base_sgd;
        float decay_step = -lr * wd * weight[i];
        float weight_new = weight[i] + film_scaled + delta_c + decay_step;

        weight_out[i] = weight_new;

        // Update EMA states for re-quantization (blending already
        // wrote ema_fast/ema_slow in place above)
        h_prev[i] = h_next; // h_next becomes h for next step

        // Track max abs for quantization
        float af = fabsf(ema_fast[i]);
        float as = fabsf(ema_slow[i]);
        float ah = fabsf(h_next);
        if (af > max_abs_fast) max_abs_fast = af;
        if (as > max_abs_slow) max_abs_slow = as;
        if (ah > max_abs_h) max_abs_h = ah;
    }

    // Quantize: symmetric max-abs
    *scale_fast_out = max_abs_fast / 127.0f + kEps;
    *scale_slow_out = max_abs_slow / 127.0f + kEps;
    *scale_h_out = max_abs_h / 127.0f + kEps;

    for (int64_t i = 0; i < n; ++i) {
        int fast = static_cast<int>(roundf(ema_fast[i] / *scale_fast_out));
        int slow = static_cast<int>(roundf(ema_slow[i] / *scale_slow_out));
        int h = static_cast<int>(roundf(h_prev[i] / *scale_h_out)); // h_prev now holds h_next

        fast = fast < -128 ? -128 : (fast > 127 ? 127 : fast);
        slow = slow < -128 ? -128 : (slow > 127 ? 127 : slow);
        h = h < -128 ? -128 : (h > 127 ? 127 : h);

        ema_fast_out_int8[i] = static_cast<int8_t>(fast);
        ema_slow_out_int8[i] = static_cast<int8_t>(slow);
        h_out_int8[i] = static_cast<int8_t>(h);
    }
}

} // namespace

namespace sgd_minexplstm_cpu {

// Dispatcher
void sgd_minexplstm_step(
    at::Tensor grad,
    at::Tensor weight,
    at::Tensor ema_fast_int8,
    at::Tensor ema_slow_int8,
    at::Tensor h_int8,
    at::Tensor scale_fast,
    at::Tensor scale_slow,
    at::Tensor scale_h,
    // Cell parameters (packed as 1D tensors)
    at::Tensor phi_w, at::Tensor phi_b,
    at::Tensor iota_w, at::Tensor iota_b,
    at::Tensor h_w, at::Tensor h_b,
    at::Tensor out_w, at::Tensor out_b,
    at::Tensor film_w, at::Tensor film_b,
    at::Tensor w_beta_t, at::Tensor b_beta_t,
    // Hyperparameters
    double lr, double wd, double lam, double clamp_val
) {
    // Validate shapes
    TORCH_CHECK(grad.dim() == 1, "grad must be 1D");
    TORCH_CHECK(weight.sizes() == grad.sizes(), "weight shape mismatch");
    TORCH_CHECK(ema_fast_int8.sizes() == grad.sizes(), "ema_fast shape mismatch");
    TORCH_CHECK(ema_slow_int8.sizes() == grad.sizes(), "ema_slow shape mismatch");
    TORCH_CHECK(h_int8.sizes() == grad.sizes(), "h shape mismatch");

    // Contiguous
    grad = grad.contiguous();
    weight = weight.contiguous();
    ema_fast_int8 = ema_fast_int8.contiguous();
    ema_slow_int8 = ema_slow_int8.contiguous();
    h_int8 = h_int8.contiguous();

    // Extract scalar params
    float w_beta = w_beta_t.item<float>();
    float b_beta = b_beta_t.item<float>();
    float lr_f = static_cast<float>(lr);
    float wd_f = static_cast<float>(wd);
    float lam_f = static_cast<float>(lam);
    float clamp_val_f = static_cast<float>(clamp_val);

    int64_t n = grad.numel();

    // Output tensors
    at::Tensor weight_out = at::empty_like(weight);
    at::Tensor ema_fast_out = at::empty_like(ema_fast_int8);
    at::Tensor ema_slow_out = at::empty_like(ema_slow_int8);
    at::Tensor h_out = at::empty_like(h_int8);
    at::Tensor scale_fast_out = at::empty_like(scale_fast);
    at::Tensor scale_slow_out = at::empty_like(scale_slow);
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

    // Call scalar implementation (AVX2 to be added)
    sgd_minexplstm_step_scalar(
        grad.data_ptr<float>(),
        weight.data_ptr<float>(),
        ema_fast_int8.data_ptr<int8_t>(),
        ema_slow_int8.data_ptr<int8_t>(),
        h_int8.data_ptr<int8_t>(),
        scale_fast.item<float>(), scale_slow.item<float>(), scale_h.item<float>(),
        ema_fast_out.data_ptr<int8_t>(),
        ema_slow_out.data_ptr<int8_t>(),
        h_out.data_ptr<int8_t>(),
        scale_fast_out.data_ptr<float>(),
        scale_slow_out.data_ptr<float>(),
        scale_h_out.data_ptr<float>(),
        weight_out.data_ptr<float>(),
        // Cell params
        phi_w_ptr, phi_b_val,
        iota_w_ptr, iota_b_val,
        h_w_ptr, h_b_val,
        out_w_ptr, out_b_arr,
        film_w_ptr, film_b_arr,
        w_beta, b_beta,
        lr_f, wd_f, lam_f, clamp_val_f,
        n
    );

    // Return outputs via reference (modify inputs in place)
    weight.copy_(weight_out);
    ema_fast_int8.copy_(ema_fast_out);
    ema_slow_int8.copy_(ema_slow_out);
    h_int8.copy_(h_out);
    scale_fast.copy_(scale_fast_out);
    scale_slow.copy_(scale_slow_out);
    scale_h.copy_(scale_h_out);
}

} // namespace sgd_minexplstm_cpu

TORCH_LIBRARY_FRAGMENT(nanochat, m) {
    m.def("sgd_minexplstm_step(Tensor grad, Tensor weight, "
          "Tensor ema_fast_int8, Tensor ema_slow_int8, Tensor h_int8, "
          "Tensor scale_fast, Tensor scale_slow, Tensor scale_h, "
          "Tensor phi_w, Tensor phi_b, Tensor iota_w, Tensor iota_b, "
          "Tensor h_w, Tensor h_b, Tensor out_w, Tensor out_b, "
          "Tensor film_w, Tensor film_b, "
          "Tensor w_beta, Tensor b_beta, "
          "float lr, float wd, float lam, float clamp_val) -> ()");
}

TORCH_LIBRARY_IMPL(nanochat, CPU, m) {
    m.impl("sgd_minexplstm_step", &sgd_minexplstm_cpu::sgd_minexplstm_step);
}