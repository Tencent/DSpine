#include <torch/extension.h>
void conditional_topk_out(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("conditional_topk_out",&conditional_topk_out);}
