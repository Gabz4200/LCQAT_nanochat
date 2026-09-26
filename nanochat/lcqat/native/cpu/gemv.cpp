// LC-QAT mul-less ternary GEMV (PRD section 5.2).
//
// K_W=3 weights are decoded from row-wise packed trits (5 per byte); activation
// indices arrive as packed nibbles (2 per byte) and are resolved through the
// FP32 LUT (dequantize-on-fetch). The inner loop performs no floating-point
// multiplications of values: ternary states become conditional additions into
// separate positive/negative accumulators, and the two asymmetric codebook
// scales (delta minus / delta plus) are applied once per output row.
//
// AVX-512 is the primary path; AVX2 and a scalar loop are runtime-selected via
// cpuid so the extension builds and runs on machines without AVX-512.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <immintrin.h>

#include <cstdint>

namespace {

constexpr int64_t kTritsPerByte = 5;

inline int decode_trit(const uint8_t* row, int64_t j) {
    static constexpr int kPow3[5] = {1, 3, 9, 27, 81};
    return (row[j / kTritsPerByte] / kPow3[j % kTritsPerByte]) % 3;
}

inline float lookup_act(const uint8_t* act_nibbles, const float* act_lut, int64_t j) {
    const uint8_t byte = act_nibbles[j >> 1];
    const uint8_t idx = (j & 1) ? (byte >> 4) : (byte & 0x0F);
    return act_lut[idx];
}

void gemv_row_scalar(
    const uint8_t* act_nibbles, const float* act_lut, const uint8_t* weight_row,
    int64_t n, float scale_neg, float scale_pos, float* out_row) {
    float pos = 0.0f;
    float neg = 0.0f;
    for (int64_t j = 0; j < n; ++j) {
        const float x = lookup_act(act_nibbles, act_lut, j);
        const int t = decode_trit(weight_row, j);
        if (t == 2) {
            pos += x;
        } else if (t == 0) {
            neg += x;
        }
    }
    out_row[0] = scale_pos * pos - scale_neg * neg;
}

__attribute__((target("avx512f"))) void gemv_row_avx512(
    const uint8_t* act_nibbles, const float* act_lut, const uint8_t* weight_row,
    int64_t n, float scale_neg, float scale_pos, float* out_row) {
    __m512 pos_acc = _mm512_setzero_ps();
    __m512 neg_acc = _mm512_setzero_ps();
    int64_t j = 0;
    for (; j + 16 <= n; j += 16) {
        // 1) Dequantize 16 activation indices through the L1-resident FP32 LUT
        alignas(64) float x_vals[16];
        for (int k = 0; k < 16; ++k) {
            x_vals[k] = lookup_act(act_nibbles, act_lut, j + k);
        }
        const __m512 x_vec = _mm512_load_ps(x_vals);

        // 2) Decode 16 trits into conditional add masks (no value multiplications)
        __mmask16 pos_mask = 0;
        __mmask16 neg_mask = 0;
        for (int k = 0; k < 16; ++k) {
            const int t = decode_trit(weight_row, j + k);
            if (t == 2) {
                pos_mask |= static_cast<__mmask16>(1) << k;
            } else if (t == 0) {
                neg_mask |= static_cast<__mmask16>(1) << k;
            }
        }
        pos_acc = _mm512_mask_add_ps(pos_acc, pos_mask, pos_acc, x_vec);
        neg_acc = _mm512_mask_add_ps(neg_acc, neg_mask, neg_acc, x_vec);
    }
    float row = scale_pos * _mm512_reduce_add_ps(pos_acc) -
        scale_neg * _mm512_reduce_add_ps(neg_acc);
    for (; j < n; ++j) {
        const float x = lookup_act(act_nibbles, act_lut, j);
        const int t = decode_trit(weight_row, j);
        if (t == 2) {
            row += scale_pos * x;
        } else if (t == 0) {
            row -= scale_neg * x;
        }
    }
    out_row[0] = row;
}

__attribute__((target("avx2"))) void gemv_row_avx2(
    const uint8_t* act_nibbles, const float* act_lut, const uint8_t* weight_row,
    int64_t n, float scale_neg, float scale_pos, float* out_row) {
    __m256 pos_acc = _mm256_setzero_ps();
    __m256 neg_acc = _mm256_setzero_ps();
    int64_t j = 0;
    for (; j + 8 <= n; j += 8) {
        alignas(32) float x_vals[8];
        alignas(32) float pos_vals[8];
        alignas(32) float neg_vals[8];
        for (int k = 0; k < 8; ++k) {
            const float x = lookup_act(act_nibbles, act_lut, j + k);
            const int t = decode_trit(weight_row, j + k);
            x_vals[k] = x;
            pos_vals[k] = (t == 2) ? x : 0.0f;
            neg_vals[k] = (t == 0) ? x : 0.0f;
        }
        pos_acc = _mm256_add_ps(pos_acc, _mm256_load_ps(pos_vals));
        neg_acc = _mm256_add_ps(neg_acc, _mm256_load_ps(neg_vals));
    }
    alignas(32) float pos_lane[8];
    alignas(32) float neg_lane[8];
    _mm256_store_ps(pos_lane, pos_acc);
    _mm256_store_ps(neg_lane, neg_acc);
    float pos = 0.0f;
    float neg = 0.0f;
    for (int k = 0; k < 8; ++k) {
        pos += pos_lane[k];
        neg += neg_lane[k];
    }
    float row = scale_pos * pos - scale_neg * neg;
    for (; j < n; ++j) {
        const float x = lookup_act(act_nibbles, act_lut, j);
        const int t = decode_trit(weight_row, j);
        if (t == 2) {
            row += scale_pos * x;
        } else if (t == 0) {
            row -= scale_neg * x;
        }
    }
    out_row[0] = row;
}

enum class RowPath { kAVX512, kAVX2, kScalar };

RowPath select_row_path() {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) {
        return RowPath::kAVX512;
    }
    if (__builtin_cpu_supports("avx2")) {
        return RowPath::kAVX2;
    }
    return RowPath::kScalar;
}

at::Tensor lcqat_gemv_k3(
    const at::Tensor& act_nibbles, const at::Tensor& act_lut,
    const at::Tensor& weight_trits, int64_t n, double scale_neg, double scale_pos) {
    TORCH_CHECK(
        act_nibbles.is_contiguous() && act_lut.is_contiguous() && weight_trits.is_contiguous(),
        "lcqat_gemv_k3 inputs must be contiguous");
    TORCH_CHECK(
        act_nibbles.device().is_cpu() && act_lut.device().is_cpu() && weight_trits.device().is_cpu(),
        "lcqat_gemv_k3 only supports CPU tensors");
    TORCH_CHECK(act_nibbles.scalar_type() == at::kByte, "act_nibbles must be uint8");
    TORCH_CHECK(weight_trits.scalar_type() == at::kByte, "weight_trits must be uint8");
    TORCH_CHECK(act_lut.scalar_type() == at::kFloat, "act_lut must be float32");
    TORCH_CHECK(
        act_nibbles.dim() == 1 && act_lut.dim() == 1 && weight_trits.dim() == 2,
        "expected shapes: act_nibbles [ceil(n/2)], act_lut [Ka], weight_trits [m, ceil(n/5)]");
    TORCH_CHECK(act_lut.numel() <= 16, "act_lut must fit 4-bit indices (<= 16 entries)");
    TORCH_CHECK(n >= 0 && n <= 2 * act_nibbles.numel(), "n out of range for act_nibbles");
    const int64_t m = weight_trits.size(0);
    const int64_t row_bytes = weight_trits.size(1);
    TORCH_CHECK(row_bytes * kTritsPerByte >= n, "weight_trits rows too narrow for n");

    // Every activation nibble (including the pad nibble of an odd tail) must
    // resolve inside the LUT; the kernel indexes it without further checks.
    const uint8_t* nib = act_nibbles.data_ptr<uint8_t>();
    const int64_t nibbles = act_nibbles.numel();
    for (int64_t b = 0; b < nibbles; ++b) {
        TORCH_CHECK(
            (nib[b] & 0x0F) < act_lut.numel() && (nib[b] >> 4) < act_lut.numel(),
            "activation index out of range for act_lut");
    }

    auto out = at::empty({m}, act_lut.options());
    const float* lut = act_lut.data_ptr<float>();
    const uint8_t* weights = weight_trits.data_ptr<uint8_t>();
    float* out_ptr = out.data_ptr<float>();
    const float neg = static_cast<float>(scale_neg);
    const float pos = static_cast<float>(scale_pos);
    const RowPath path = select_row_path();

    at::parallel_for(0, m, 1, [&](const int64_t begin, const int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const uint8_t* row = weights + i * row_bytes;
            switch (path) {
                case RowPath::kAVX512:
                    gemv_row_avx512(nib, lut, row, n, neg, pos, out_ptr + i);
                    break;
                case RowPath::kAVX2:
                    gemv_row_avx2(nib, lut, row, n, neg, pos, out_ptr + i);
                    break;
                default:
                    gemv_row_scalar(nib, lut, row, n, neg, pos, out_ptr + i);
            }
        }
    });
    return out;
}

}  // namespace

// Fragment: quant_attn.cpp's extension may already own the `nanochat`
// library in this process; TORCH_LIBRARY enforces single registration per
// namespace and aborts (uncaught exception in static init) on a second one.
TORCH_LIBRARY_FRAGMENT(nanochat, m) {
    m.def(
        "lcqat_gemv_k3(Tensor act_nibbles, Tensor act_lut, Tensor weight_trits, "
        "int n, float scale_neg, float scale_pos) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanochat, CPU, m) {
    m.impl("lcqat_gemv_k3", &lcqat_gemv_k3);
}
