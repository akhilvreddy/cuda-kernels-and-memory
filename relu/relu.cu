#include <iostream>
#include <cuda_runtime.h>

__global__ void relu(float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] > 0 ? x[i] : 0.0f;
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);

    float *h_x = new float[N];
    float *h_out = new float[N];

    // input needs to have both negative and positive values (to show effects of ReLU)
    for (int i = 0; i < N; ++i) {
        h_x[i] = i - 512;  // range: -512 to +511
    }

    float *d_x, *d_out;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    // timer (just like vector_add)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    relu<<<gridSize, blockSize>>>(d_x, d_out, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // first 10 values
    for (int i = 510; i < 520; ++i) {
        std::cout << "ReLU(" << h_x[i] << ") = " << h_out[i] << std::endl;
    }

    std::cout << "GPU kernel execution time: " << ms << " ms" << std::endl;

    delete[] h_x;
    delete[] h_out;
    cudaFree(d_x);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}