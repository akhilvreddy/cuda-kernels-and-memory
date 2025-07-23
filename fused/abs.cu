#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void apply_abs(float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fabsf(x[i]);
    }
}

int main() {
    int N = 1024 * 1024;
    size_t size = N * sizeof(float);
    float* h_x = new float[N];
    float* h_out = new float[N];

    for (int i = 0; i < N; ++i)
        h_x[i] = (i - N / 2) * 0.01f;

    float *d_x, *d_out;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    apply_abs<<<gridSize, blockSize>>>(d_x, d_out, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    for (int i = N / 2 - 5; i < N / 2 + 5; ++i)
        std::cout << "abs(" << h_x[i] << ") = " << h_out[i] << std::endl;

    std::cout << "\n Abs GPU time: " << ms << " ms\n";

    delete[] h_x; delete[] h_out;
    cudaFree(d_x); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}