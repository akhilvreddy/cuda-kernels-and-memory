#include <torch/extension.h>

// declaring my forward function
void leaky_relu_forward(torch::Tensor x, torch::Tensor out, float alpha);

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "Leaky ReLU forward (CUDA)");
}