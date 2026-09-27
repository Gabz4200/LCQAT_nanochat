/**
 * SparseProp AVX2 kernels for unstructured sparse backpropagation.
 *
 * Layout convention (matching nn.Linear y = x @ W^T):
 *   - weight W: [M, K]  (M = out_features rows, K = in_features cols)
 *   - input x:  [B, K]  -> kernel uses transposed [K, B]  (row-contiguous per in-feature)
 *   - output:   [B, M]  -> kernel uses transposed [M, B]
 *
 * Forward (SpMM):  y[j,:] += w_val * x[k,:] for each nnz (j,k)
 * Backward dW:     w_val[p] = dot(dy[j,:], x[k,:])   (per nnz)
 * Backward dX:     dx[k,:] += w_val * dy[j,:]        (per nnz)
 *
 * All three vectorize over the batch B with AVX2 FMA, parallel over rows (dW,
 * forward) or columns (dX) with no races. CSR for row-wise passes,
 * CSC for the column-wise dx pass.
 */
#include <torch/extension.h>
#include <immintrin.h>
#include <cstdint>

namespace {

constexpr int AVX2_W = 8;  // 256-bit / 32-bit float

// Per-row AVX2 helpers (target-attributed so intrinsics are legal).
__attribute__((target("avx2,fma")))
float hsum8_avx2(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s4 = _mm_add_ps(lo, hi);
  s4 = _mm_hadd_ps(s4, s4);
  s4 = _mm_hadd_ps(s4, s4);
  return _mm_cvtss_f32(s4);
}

// --- Forward per-row SpMM: y[row,:] += sum_p w_val[p]*x[col[p],:] ---
__attribute__((target("avx2,fma")))
void spmm_row_avx2(const float* __restrict gy_unused, // kept for signature uniformity, unused
                   const float* __restrict x_base,
                   const int32_t* __restrict cols,     // [nnz_in_row] column indices
                   const float* __restrict vals,       // [nnz_in_row] weight values
                   int nnz_in_row, int B,
                   float* __restrict y_row) {
  (void)gy_unused;
  for (int p = 0; p < nnz_in_row; ++p) {
    int kc = cols[p];
    float w = vals[p];
    const float* x_row = x_base + static_cast<int64_t>(kc) * B;
    __m256 vw = _mm256_set1_ps(w);
    int b = 0;
    for (; b + AVX2_W <= B; b += AVX2_W) {
      __m256 vy = _mm256_loadu_ps(y_row + b);
      __m256 vx = _mm256_loadu_ps(x_row + b);
      vy = _mm256_fmadd_ps(vw, vx, vy);
      _mm256_storeu_ps(y_row + b, vy);
    }
    for (; b < B; ++b) y_row[b] += w * x_row[b];
  }
}

// --- Backward dW per-row: for each nnz, dot product of dy[row,:] and x[col,:] ---
__attribute__((target("avx2,fma")))
void bw_dW_row_avx2(const float* __restrict dy_row,
                    const float* __restrict x_base,
                    const int32_t* __restrict cols,
                    int nnz_in_row, int B,
                    float* __restrict out_vals) {
  for (int p = 0; p < nnz_in_row; ++p) {
    int kc = cols[p];
    const float* x_row = x_base + static_cast<int64_t>(kc) * B;
    __m256 vacc = _mm256_setzero_ps();
    int b = 0;
    for (; b + AVX2_W <= B; b += AVX2_W) {
      __m256 vdy = _mm256_loadu_ps(dy_row + b);
      __m256 vx = _mm256_loadu_ps(x_row + b);
      vacc = _mm256_fmadd_ps(vdy, vx, vacc);
    }
    float acc = hsum8_avx2(vacc);
    for (; b < B; ++b) acc += dy_row[b] * x_row[b];
    out_vals[p] = acc;
  }
}

// --- Backward dX per-column: dx[col,:] += sum_p vals[p]*dy[row[p],:] ---
__attribute__((target("avx2,fma")))
void bw_dX_col_avx2(float* __restrict gx_col,
                    const float* __restrict dy_base,
                    const int32_t* __restrict rows,
                    const float* __restrict vals,
                    int nnz_in_col, int B) {
  for (int p = 0; p < nnz_in_col; ++p) {
    int mr = rows[p];
    float w = vals[p];
    const float* dy_row = dy_base + static_cast<int64_t>(mr) * B;
    __m256 vw = _mm256_set1_ps(w);
    int b = 0;
    for (; b + AVX2_W <= B; b += AVX2_W) {
      __m256 vgy = _mm256_loadu_ps(dy_row + b);
      __m256 vgx = _mm256_loadu_ps(gx_col + b);
      vgx = _mm256_fmadd_ps(vgy, vw, vgx);
      _mm256_storeu_ps(gx_col + b, vgx);
    }
    for (; b < B; ++b) gx_col[b] += w * dy_row[b];
  }
}

// --- Scalar fallbacks ---

void spmm_row_scalar(const float* __restrict x_base,
                     const int32_t* __restrict cols,
                     const float* __restrict vals,
                     int nnz_in_row, int B,
                     float* __restrict y_row) {
  for (int p = 0; p < nnz_in_row; ++p) {
    int kc = cols[p];
    float w = vals[p];
    const float* x_row = x_base + static_cast<int64_t>(kc) * B;
    for (int b = 0; b < B; ++b) y_row[b] += w * x_row[b];
  }
}

void bw_dW_row_scalar(const float* __restrict dy_row,
                      const float* __restrict x_base,
                      const int32_t* __restrict cols,
                      int nnz_in_row, int B,
                      float* __restrict out_vals) {
  for (int p = 0; p < nnz_in_row; ++p) {
    int kc = cols[p];
    const float* x_row = x_base + static_cast<int64_t>(kc) * B;
    float acc = 0.f;
    for (int b = 0; b < B; ++b) acc += dy_row[b] * x_row[b];
    out_vals[p] = acc;
  }
}

void bw_dX_col_scalar(float* __restrict gx_col,
                      const float* __restrict dy_base,
                      const int32_t* __restrict rows,
                      const float* __restrict vals,
                      int nnz_in_col, int B) {
  for (int p = 0; p < nnz_in_col; ++p) {
    int mr = rows[p];
    float w = vals[p];
    const float* dy_row = dy_base + static_cast<int64_t>(mr) * B;
    for (int b = 0; b < B; ++b) gx_col[b] += w * dy_row[b];
  }
}

// --- Runtime dispatch ---

enum class Path { kAVX2, kScalar };

Path select_path() {
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx2")) return Path::kAVX2;
  return Path::kScalar;
}

// ---------- Torch C++ API ----------

at::Tensor lcqat_sparseprop_forward(
    const at::Tensor& w_val,     // [nnz] float32 (CSR order)
    const at::Tensor& w_col,     // [nnz] int32 (CSR column indices)
    const at::Tensor& w_ptr,     // [M+1] int32 (CSR row pointers)
    const at::Tensor& x,         // [K, B] float32
    const at::Tensor& bias,      // [M] float32 or empty
    int64_t M) {
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(w_val.is_contiguous(), "w_val must be contiguous");
  TORCH_CHECK(w_col.is_contiguous(), "w_col must be contiguous");
  TORCH_CHECK(w_ptr.is_contiguous(), "w_ptr must be contiguous");
  TORCH_CHECK(x.device().is_cpu(), "x must be on CPU");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(w_val.scalar_type() == at::kFloat, "w_val must be float32");
  TORCH_CHECK(w_col.scalar_type() == at::kInt, "w_col must be int32");
  TORCH_CHECK(w_ptr.scalar_type() == at::kInt, "w_ptr must be int32");
  TORCH_CHECK(w_ptr.size(0) == M + 1, "w_ptr size must be M+1");

  int64_t K = x.size(0);
  int64_t B = x.size(1);

  auto y = at::zeros({M, B}, x.options());
  if (bias.defined() && bias.numel() > 0) {
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == M, "bias size must be M");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
      float* y_ptr = y.data_ptr<float>();
      const float* b_ptr = bias.data_ptr<float>();
      for (int64_t m = begin; m < end; ++m) {
        float* y_row = y_ptr + m * B;
        float bval = b_ptr[m];
        for (int64_t bb = 0; bb < B; ++bb) y_row[bb] = bval;
      }
    });
  }

  const float* x_ptr = x.data_ptr<float>();
  const float* wv_ptr = w_val.data_ptr<float>();
  const int32_t* wc_ptr = w_col.data_ptr<int32_t>();
  const int32_t* wp_ptr = w_ptr.data_ptr<int32_t>();
  float* y_ptr = y.data_ptr<float>();
  const Path path = select_path();

  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      int32_t p_start = wp_ptr[m];
      int32_t p_end = wp_ptr[m + 1];
      int nnz_in_row = p_end - p_start;
      float* y_row = y_ptr + m * B;
      if (path == Path::kAVX2) {
        spmm_row_avx2(nullptr, x_ptr, wc_ptr + p_start, wv_ptr + p_start,
                       nnz_in_row, B, y_row);
      } else {
        spmm_row_scalar(x_ptr, wc_ptr + p_start, wv_ptr + p_start,
                        nnz_in_row, B, y_row);
      }
    }
  });
  return y;
}

std::tuple<at::Tensor, at::Tensor> lcqat_sparseprop_backward(
    const at::Tensor& gY,          // [M, B] float32
    const at::Tensor& x,           // [K, B] float32
    const at::Tensor& w_val,       // [nnz] float32 (CSR order)
    const at::Tensor& w_col,       // [nnz] int32
    const at::Tensor& w_ptr,       // [M+1] int32
    const at::Tensor& w_val_csc,   // [nnz] float32 (CSC order)
    const at::Tensor& w_row,       // [nnz] int32
    const at::Tensor& w_cptr,      // [K+1] int32
    int64_t M, int64_t K) {
  TORCH_CHECK(gY.is_contiguous() && x.is_contiguous(), "gY and x must be contiguous");
  TORCH_CHECK(gY.scalar_type() == at::kFloat, "gY must be float32");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  TORCH_CHECK(w_val.scalar_type() == at::kFloat, "w_val must be float32");
  TORCH_CHECK(w_val_csc.scalar_type() == at::kFloat, "w_val_csc must be float32");
  TORCH_CHECK(gY.size(0) == M, "gY rows must be M");
  TORCH_CHECK(x.size(0) == K, "x rows must be K");
  TORCH_CHECK(gY.size(1) == x.size(1), "B mismatch");
  TORCH_CHECK(w_ptr.size(0) == M + 1, "w_ptr size must be M+1");
  TORCH_CHECK(w_cptr.size(0) == K + 1, "w_cptr size must be K+1");

  int64_t B = gY.size(1);
  int64_t nnz = w_val.size(0);

  auto gW_val = at::empty({nnz}, w_val.options());
  auto gX = at::zeros({K, B}, x.options());

  const float* gy_ptr = gY.data_ptr<float>();
  const float* x_ptr = x.data_ptr<float>();
  const float* wv_ptr = w_val.data_ptr<float>();
  const int32_t* wc_ptr = w_col.data_ptr<int32_t>();
  const int32_t* wp_ptr = w_ptr.data_ptr<int32_t>();
  const float* wvc_ptr = w_val_csc.data_ptr<float>();
  const int32_t* wr_ptr = w_row.data_ptr<int32_t>();
  const int32_t* wcptr_ptr = w_cptr.data_ptr<int32_t>();
  float* gw_ptr = gW_val.data_ptr<float>();
  float* gx_ptr = gX.data_ptr<float>();
  const Path path = select_path();

  // dW: parallel over output rows M (CSR), disjoint nnz per row -> no race
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      int32_t p_start = wp_ptr[m];
      int32_t p_end = wp_ptr[m + 1];
      int nnz_in_row = p_end - p_start;
      const float* dy_row = gy_ptr + static_cast<int64_t>(m) * B;
      if (path == Path::kAVX2) {
        bw_dW_row_avx2(dy_row, x_ptr, wc_ptr + p_start,
                       nnz_in_row, B, gw_ptr + p_start);
      } else {
        bw_dW_row_scalar(dy_row, x_ptr, wc_ptr + p_start,
                         nnz_in_row, B, gw_ptr + p_start);
      }
    }
  });

  // dX: parallel over input columns K (CSC), disjoint columns -> no race
  at::parallel_for(0, K, 1, [&](int64_t begin, int64_t end) {
    for (int64_t k = begin; k < end; ++k) {
      int32_t p_start = wcptr_ptr[k];
      int32_t p_end = wcptr_ptr[k + 1];
      int nnz_in_col = p_end - p_start;
      float* gx_col = gx_ptr + static_cast<int64_t>(k) * B;
      if (path == Path::kAVX2) {
        bw_dX_col_avx2(gx_col, gy_ptr,
                       wr_ptr + p_start, wvc_ptr + p_start,
                       nnz_in_col, B);
      } else {
        bw_dX_col_scalar(gx_col, gy_ptr,
                         wr_ptr + p_start, wvc_ptr + p_start,
                         nnz_in_col, B);
      }
    }
  });

  return std::make_tuple(gX, gW_val);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(nanochat, m) {
  m.def("lcqat_sparseprop_forward(Tensor w_val, Tensor w_col, Tensor w_ptr, "
        "Tensor x, Tensor bias, int M) -> Tensor");
  m.def("lcqat_sparseprop_backward(Tensor gY, Tensor x, "
        "Tensor w_val, Tensor w_col, Tensor w_ptr, "
        "Tensor w_val_csc, Tensor w_row, Tensor w_cptr, "
        "int M, int K) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(nanochat, CPU, m) {
  m.impl("lcqat_sparseprop_forward", &lcqat_sparseprop_forward);
  m.impl("lcqat_sparseprop_backward", &lcqat_sparseprop_backward);
}