// Index-native quantized-KV decode attention (LC-QAT PRD section 7.1).
//
// K/V live as nibble-packed uint8 indices and are resolved to FP32 through
// per-head LUTs inside the kernel (dequantize-on-fetch), so the cache never
// materializes as bf16. Covers causal masking with an optional left sliding
// window, GQA (query head h reads KV head h / group), and any number of
// queries Tq ending at cache_seqlens (write-before-attend contract).
//
// Scalar + -O3: the decode working set (window rows) is L1/L2 resident and the
// bottleneck is LUT gathers, not FMA throughput. SIMD is a follow-up only if
// benchmarks demand it.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

inline uint8_t nibble_idx(const uint8_t* row, int64_t d) {
    const uint8_t byte = row[d >> 1];
    return (d & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0x0F);
}

at::Tensor lcqat_quant_attn(
    const at::Tensor& q, const at::Tensor& k_idx, const at::Tensor& v_idx,
    const at::Tensor& k_lut, const at::Tensor& v_lut,
    const at::Tensor& cache_seqlens, int64_t window_left) {
    TORCH_CHECK(
        q.is_contiguous() && k_idx.is_contiguous() && v_idx.is_contiguous() &&
            k_lut.is_contiguous() && v_lut.is_contiguous() &&
            cache_seqlens.is_contiguous(),
        "lcqat_quant_attn inputs must be contiguous");
    TORCH_CHECK(
        q.device().is_cpu() && k_idx.device().is_cpu() && v_idx.device().is_cpu() &&
            k_lut.device().is_cpu() && v_lut.device().is_cpu() &&
            cache_seqlens.device().is_cpu(),
        "lcqat_quant_attn only supports CPU tensors");
    TORCH_CHECK(
        q.scalar_type() == at::kFloat && k_lut.scalar_type() == at::kFloat &&
            v_lut.scalar_type() == at::kFloat,
        "q/k_lut/v_lut must be float32");
    TORCH_CHECK(
        k_idx.scalar_type() == at::kByte && v_idx.scalar_type() == at::kByte,
        "k_idx/v_idx must be uint8");
    TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt, "cache_seqlens must be int32");
    TORCH_CHECK(q.dim() == 4, "q must be [B, Tq, H, D]");
    TORCH_CHECK(k_idx.dim() == 4 && v_idx.sizes() == k_idx.sizes(), "k_idx/v_idx must match as [B, T, H_kv, nb]");

    const int64_t b = q.size(0);
    const int64_t tq = q.size(1);
    const int64_t n_head = q.size(2);
    const int64_t head_dim = q.size(3);
    const int64_t t = k_idx.size(1);
    const int64_t h_kv = k_idx.size(2);
    const int64_t nb = k_idx.size(3);
    TORCH_CHECK(k_idx.size(0) == b, "batch mismatch between q and cache");
    TORCH_CHECK(nb == (head_dim + 1) / 2, "cache last dim must hold head_dim nibbles");
    TORCH_CHECK(
        k_lut.dim() == 2 && v_lut.sizes() == k_lut.sizes() &&
            k_lut.size(0) == h_kv,
        "k_lut/v_lut must be [H_kv, K] matching the cache");
    TORCH_CHECK(n_head % h_kv == 0, "n_head must be divisible by H_kv");
    const int64_t k_size = k_lut.size(1);
    TORCH_CHECK(k_size >= 3 && k_size <= 16, "LUT K must fit 4-bit indices (3..16)");
    TORCH_CHECK(cache_seqlens.dim() == 1 && cache_seqlens.size(0) == b, "cache_seqlens must be [B]");
    TORCH_CHECK(window_left >= -1, "window_left must be >= -1");

    // Every nibble (including the pad nibble of an odd tail) must resolve
    // inside the LUT; the kernel indexes it without further checks.
    const uint8_t* k_ptr = k_idx.data_ptr<uint8_t>();
    const uint8_t* v_ptr = v_idx.data_ptr<uint8_t>();
    const int64_t cache_elems = k_idx.numel();
    for (int64_t i = 0; i < cache_elems; ++i) {
        TORCH_CHECK(
            (k_ptr[i] & 0x0F) < k_size && (k_ptr[i] >> 4) < k_size &&
                (v_ptr[i] & 0x0F) < k_size && (v_ptr[i] >> 4) < k_size,
            "cache index out of range for LUT");
    }
    const int32_t* seqlens = cache_seqlens.data_ptr<int32_t>();
    for (int64_t i = 0; i < b; ++i) {
        TORCH_CHECK(
            seqlens[i] >= tq && seqlens[i] <= t,
            "cache_seqlens must be in [Tq, T]");
    }

    auto out = at::zeros({b, tq, n_head, head_dim}, q.options());
    const float* q_ptr = q.data_ptr<float>();
    const float* k_lut_ptr = k_lut.data_ptr<float>();
    const float* v_lut_ptr = v_lut.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t group = n_head / h_kv;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    at::parallel_for(0, b * tq * n_head, 1, [&](const int64_t begin, const int64_t end) {
        std::vector<float> scores;
        std::vector<float> acc(head_dim, 0.0f);
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t h = linear % n_head;
            const int64_t i = (linear / n_head) % tq;
            const int64_t bi = linear / (n_head * tq);
            const int64_t kv = h / group;
            const int64_t s = seqlens[bi];
            const int64_t g = s - tq + i;  // global position of this query
            const int64_t lo = window_left < 0 ? 0 : std::max<int64_t>(0, g - window_left);
            const int64_t n = g - lo + 1;
            scores.assign(n, 0.0f);
            const float* q_row = q_ptr + ((bi * tq + i) * n_head + h) * head_dim;

            float mx = -INFINITY;
            for (int64_t j = lo; j <= g; ++j) {
                const uint8_t* k_row =
                    k_ptr + ((bi * t + j) * h_kv + kv) * nb;
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += q_row[d] * k_lut_ptr[kv * k_size + nibble_idx(k_row, d)];
                }
                const float sc = dot * scale;
                scores[j - lo] = sc;
                mx = std::max(mx, sc);
            }
            float sum = 0.0f;
            for (int64_t jj = 0; jj < n; ++jj) {
                const float e = std::exp(scores[jj] - mx);
                scores[jj] = e;
                sum += e;
            }
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (int64_t jj = 0; jj < n; ++jj) {
                const int64_t j = lo + jj;
                const uint8_t* v_row =
                    v_ptr + ((bi * t + j) * h_kv + kv) * nb;
                const float p = scores[jj] / sum;
                for (int64_t d = 0; d < head_dim; ++d) {
                    acc[d] += p * v_lut_ptr[kv * k_size + nibble_idx(v_row, d)];
                }
            }
            float* out_row = out_ptr + ((bi * tq + i) * n_head + h) * head_dim;
            for (int64_t d = 0; d < head_dim; ++d) {
                out_row[d] = acc[d];
            }
        }
    });
    return out;
}

}  // namespace

// Fragment: gemv.cpp's extension may already own the `nanochat` library in
// this process; TORCH_LIBRARY enforces single registration per namespace.
TORCH_LIBRARY_FRAGMENT(nanochat, m) {
    m.def(
        "lcqat_quant_attn(Tensor q, Tensor k_idx, Tensor v_idx, Tensor k_lut, "
        "Tensor v_lut, Tensor cache_seqlens, int window_left) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanochat, CPU, m) {
    m.impl("lcqat_quant_attn", &lcqat_quant_attn);
}
