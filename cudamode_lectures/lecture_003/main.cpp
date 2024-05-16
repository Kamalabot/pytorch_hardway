#include <torch/extension.h>
torch::Tensor rgb_to_gs(torch::Tensor input);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("rgb_to_gs", torch::wrap_pybind_function(rgb_to_gs), "rgb_to_gs");
}