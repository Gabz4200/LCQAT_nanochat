// K-agnostic index-weight linear (quantized runtime, TODO section 1).
//
// Weights arrive in their K-selected storage format (trits / nibbles /
// uint8 / int32, tag from nanochat.lcqat.packing FORMAT_*) and are
// resolved through the FP32 weight LUT inside the kernel
// (dequantize-on-fetch); activations are uint8 IDs resolved through the
// activation LUT. The packed storage is read in place - no materialized
// fp32 weight matrix, which is the memory point of the whole runtime.
//
// One thread per output element [T, m]; -O3 scalar. SIMD only if the
// bench in scripts/gemv_bench.py says it is needed.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <cstdint>

namespace {

constexpr int64_t kFormatTrits = 0;
constexpr int64_t kFormatNibbles = 1;
constexpr int64_t kFormatUint8 = 2;
constexpr int64_t kFormatInt32 = 3;

constexpr int64_t kTritsPerByte = 5;
constexpr int64_t kNibblesPerByte = 2;

int64_t expected_format(int64_t k) {
    if (k == 3) {
        return kFormatTrits;
    }
    if (k <= 15) {
        return kFormatNibbles;
    }
    if (k <= 255) {
        return kFormatUint8;
    }
    return kFormatInt32;
}

inline int decode_trit(const uint8_t* row, int64_t j) {
    static constexpr int kPow3[5] = {1, 3, 9, 27, 81};
    return (row[j / kTritsPerByte] / kPow3[j % kTritsPerByte]) % 3;
}

inline int decode_nibble(const uint8_t* row, int64_t j) {
    const uint8_t byte = row[j >> 1];
    return (j & 1) ? (byte >> 4) : (byte & 0x0F);
}

at::Tensor lcqat_index_linear(
    const at::Tensor& act_indices, const at::Tensor& act_lut,
    const at::Tensor& weight_indices, const at::Tensor& weight_lut,
    int64_t n, int64_t format) {
    TORCH_CHECK(
        act_indices.is_contiguous() && act_lut.is_contiguous() &&
            weight_indices.is_contiguous() && weight_lut.is_contiguous(),
        "lcqat_index_linear inputs must be contiguous");
    TORCH_CHECK(
        act_indices.device().is_cpu() && act_lut.device().is_cpu() &&
            weight_indices.device().is_cpu() && weight_lut.device().is_cpu(),
        "lcqat_index_linear only supports CPU tensors");
    TORCH_CHECK(
        act_indices.scalar_type() == at::kByte,
        "act_indices must be uint8 (unpacked activation IDs)");
    TORCH_CHECK(
        act_lut.scalar_type() == at::kFloat && weight_lut.scalar_type() == at::kFloat,
        "act_lut/weight_lut must be float32");
    TORCH_CHECK(
        act_indices.dim() == 2 && weight_indices.dim() == 2 &&
            act_lut.dim() == 1 && weight_lut.dim() == 1,
        "expected shapes: act_indices [t, n], act_lut [Ka], "
        "weight_indices [m, ...], weight_lut [Kw]");
    TORCH_CHECK(n > 0, "n must be positive");
    TORCH_CHECK(
        act_indices.size(1) == n, "act_indices width must equal n");

    const int64_t k_w = weight_lut.numel();
    TORCH_CHECK(
        k_w >= 3 && k_w % 2 == 1, "weight_lut K must be an odd integer >= 3");
    TORCH_CHECK(
        format == expected_format(k_w),
        "format does not match weight_lut K");
    const int64_t t = act_indices.size(0);
    const int64_t m = weight_indices.size(0);
    const int64_t width = weight_indices.size(1);
    switch (format) {
        case kFormatTrits:
            TORCH_CHECK(
                weight_indices.scalar_type() == at::kByte,
                "trit-packed weights must be uint8");
            TORCH_CHECK(
                width * kTritsPerByte >= n,
                "trit-packed rows too narrow for n");
            break;
        case kFormatNibbles:
            TORCH_CHECK(
                weight_indices.scalar_type() == at::kByte,
                "nibble-packed weights must be uint8");
            TORCH_CHECK(
                width * kNibblesPerByte >= n,
                "nibble-packed rows too narrow for n");
            break;
        case kFormatUint8:
            TORCH_CHECK(
                weight_indices.scalar_type() == at::kByte,
                "raw uint8 weights must be uint8");
            TORCH_CHECK(width == n, "raw weight rows must have width n");
            break;
        case kFormatInt32:
            TORCH_CHECK(
                weight_indices.scalar_type() == at::kInt,
                "raw K>255 weights must be int32");
            TORCH_CHECK(width == n, "raw weight rows must have width n");
            break;
        default:
            TORCH_CHECK(false, "unknown weight index format: ", format);
    }

    // Bound every index against its LUT before the kernel trusts it
    // (same contract as lcqat_gemv_k3; the kernel indexes unchecked).
    const int64_t k_a = act_lut.numel();
    TORCH_CHECK(
        k_a >= 3 && k_a % 2 == 1, "act_lut K must be an odd integer >= 3");
    const uint8_t* act = act_indices.data_ptr<uint8_t>();
    for (int64_t i = 0; i < act_indices.numel(); ++i) {
        TORCH_CHECK(act[i] < k_a, "act index out of range for act_lut");
    }
    if (format == kFormatInt32) {
        const int32_t* w = weight_indices.data_ptr<int32_t>();
        for (int64_t i = 0; i < weight_indices.numel(); ++i) {
            TORCH_CHECK(w[i] >= 0 && w[i] < k_w, "weight index out of range");
        }
    } else if (format == kFormatUint8) {
        const uint8_t* w = weight_indices.data_ptr<uint8_t>();
        for (int64_t i = 0; i < weight_indices.numel(); ++i) {
            TORCH_CHECK(w[i] < k_w, "weight index out of range");
        }
    } else if (format == kFormatNibbles) {
        const uint8_t* w = weight_indices.data_ptr<uint8_t>();
        for (int64_t i = 0; i < weight_indices.numel(); ++i) {
            TORCH_CHECK(
                (w[i] & 0x0F) < k_w && (w[i] >> 4) < k_w,
                "weight index out of range");
        }
    } else {
        // trits: decode every position (cheap pre-pass, cache-resident)
        for (int64_t i = 0; i < m; ++i) {
            const uint8_t* row =
                weight_indices.data_ptr<uint8_t>() + i * width;
            for (int64_t j = 0; j < n; ++j) {
                TORCH_CHECK(decode_trit(row, j) < k_w, "weight index out of range");
            }
        }
    }

    auto out = at::empty({t, m}, act_lut.options());
    const float* act_lut_ptr = act_lut.data_ptr<float>();
    const float* w_lut_ptr = weight_lut.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const uint8_t* w_u8 = format == kFormatInt32 ? nullptr : weight_indices.data_ptr<uint8_t>();
    const int32_t* w_i32 = format == kFormatInt32 ? weight_indices.data_ptr<int32_t>() : nullptr;
    const int64_t w_row = width;

    at::parallel_for(0, t * m, 1, [&](const int64_t begin, const int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t mi = linear % m;
            const int64_t ti = linear / m;
            const uint8_t* act_row = act + ti * n;
            float acc = 0.0f;
            switch (format) {
                case kFormatTrits: {
                    const uint8_t* w_row_ptr = w_u8 + mi * w_row;
                    for (int64_t j = 0; j < n; ++j) {
                        acc += act_lut_ptr[act_row[j]] *
                            w_lut_ptr[decode_trit(w_row_ptr, j)];
                    }
                    break;
                }
                case kFormatNibbles: {
                    const uint8_t* w_row_ptr = w_u8 + mi * w_row;
                    for (int64_t j = 0; j < n; ++j) {
                        acc += act_lut_ptr[act_row[j]] *
                            w_lut_ptr[decode_nibble(w_row_ptr, j)];
                    }
                    break;
                }
                case kFormatUint8: {
                    const uint8_t* w_row_ptr = w_u8 + mi * w_row;
                    for (int64_t j = 0; j < n; ++j) {
                        acc += act_lut_ptr[act_row[j]] * w_lut_ptr[w_row_ptr[j]];
                    }
                    break;
                }
                default: {
                    const int32_t* w_row_ptr = w_i32 + mi * w_row;
                    for (int64_t j = 0; j < n; ++j) {
                        acc += act_lut_ptr[act_row[j]] * w_lut_ptr[w_row_ptr[j]];
                    }
                }
            }
            out_ptr[linear] = acc;
        }
    });
    return out;
}

}  // namespace

// Fragment: other nanochat extensions may already own the `nanochat`
// library in this process (TORCH_LIBRARY enforces one registration).
TORCH_LIBRARY_FRAGMENT(nanochat, m) {
    m.def(
        "lcqat_index_linear(Tensor act_indices, Tensor act_lut, "
        "Tensor weight_indices, Tensor weight_lut, int n, int format) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanochat, CPU, m) {
    m.impl("lcqat_index_linear", &lcqat_index_linear);
}
