// DiffusionBlocks EDM denoising loss (training kernel).
//
//   loss = weight * mean((pred - clean)^2)
//
// `pred` is the active block's denoiser prediction, `clean` the detached
// L2-normalized target embeddings, `weight` the scalar EDM weight w(sigma).
// DiffusionBlocks trains one scalar sigma per optimizer step, so the weight
// is a single float for the whole batch (not per-sample).
//
// One thread per chunk with per-thread partial sums: the reduction is a
// single scalar, so parallelizing over output elements would race. N is
// typically B*T*D (tens of thousands of elements); the serial tail that sums
// the partials is O(threads).
//
// This op is forward-only (inference-only registration): training gradients
// reach pred/clean through DbDenoiseLossFunction's analytic backward in
// nanochat/ops/db_denoise.py, never by differentiating through this kernel.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace {

at::Tensor lcqat_db_denoise(
    const at::Tensor& pred, const at::Tensor& clean, double weight) {
  TORCH_CHECK(pred.sizes() == clean.sizes(), "pred and clean must have the same shape");
  TORCH_CHECK(pred.numel() > 0, "pred and clean must be non-empty");
  TORCH_CHECK(
      pred.scalar_type() == at::kFloat && clean.scalar_type() == at::kFloat,
      "pred and clean must be float32");
  TORCH_CHECK(
      pred.device().is_cpu() && clean.device().is_cpu(),
      "pred and clean must be on CPU");
  TORCH_CHECK(weight >= 0.0, "weight must be non-negative");

  auto p = pred.contiguous();
  auto c = clean.contiguous();
  const float* p_ptr = p.data_ptr<float>();
  const float* c_ptr = c.data_ptr<float>();
  const int64_t n = p.numel();

  // Chunked partial sums: each thread owns one chunk, so no atomics.
  const int64_t n_threads = std::max<int64_t>(1, at::get_num_threads());
  const int64_t n_chunks = std::min(n_threads, n);
  std::vector<double> partials(n_chunks, 0.0);
  at::parallel_for(0, n_chunks, 1, [&](int64_t begin, int64_t end) {
    for (int64_t t = begin; t < end; ++t) {
      const int64_t start = t * n / n_chunks;
      const int64_t stop = (t + 1) * n / n_chunks;
      double acc = 0.0;
      for (int64_t i = start; i < stop; ++i) {
        const double d =
            static_cast<double>(p_ptr[i]) - static_cast<double>(c_ptr[i]);
        acc += d * d;
      }
      partials[t] = acc;
    }
  });
  double total = 0.0;
  for (double v : partials) total += v;
  return at::scalar_tensor(
      static_cast<float>(weight * total / static_cast<double>(n)), pred.options());
}

}  // namespace

// Fragment: other extensions may already own the `nanochat` library in this
// process; TORCH_LIBRARY enforces single registration per namespace and
// aborts (uncaught exception in static init) on a second one.
TORCH_LIBRARY_FRAGMENT(nanochat, m) {
  m.def("lcqat_db_denoise(Tensor pred, Tensor clean, float weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanochat, CPU, m) {
  m.impl("lcqat_db_denoise", &lcqat_db_denoise);
}
