#include <torch/extension.h>
#include <vector>

std::vector<int64_t> native_geometry_cuda(const at::Tensor& dense, at::Tensor out);
void mixed_sum_cuda(const at::Tensor& ids, const at::Tensor& weights,
                    const at::Tensor& lengths, const at::Tensor& use_sparse,
                    const at::Tensor& dense, at::Tensor partials, at::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("native_geometry", &native_geometry_cuda);
  m.def("mixed_sum_out", &mixed_sum_cuda,
        "Per-row GPU branch: exact sparse normal SUM / original dense vec4 SUM");
}
