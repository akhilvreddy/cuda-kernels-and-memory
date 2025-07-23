#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void leaky_relu_forward_kernel(const float* x, float* out, int N, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float val = x[i];
        out[i] = val > 0 ? val : alpha * val;
    }
}

void leaky_relu_forward(torch::Tensor x, torch::Tensor out, float alpha) {
    int N = x.numel();
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    leaky_relu_forward_kernel<<<gridSize, blockSize>>>(x.data_ptr<float>(), out.data_ptr<float>(), N, alpha);
}