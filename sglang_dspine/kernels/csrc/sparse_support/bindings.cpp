#include <torch/extension.h>
void sparse_air_cut_out(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,
                        at::Tensor,at::Tensor,at::Tensor,at::Tensor,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_air_cut_out", &sparse_air_cut_out);
}
