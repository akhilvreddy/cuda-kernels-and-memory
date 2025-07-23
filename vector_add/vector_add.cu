#include <iostream>
#include <cuda_runtime.h>

// main CUDA kernel
__global__ void vector_add(float* a, float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// this is going to be run in colab
int main() {
    int N = 1024;
    size_t size = N * sizeof(float);

    // allocate memory on host (CPU)
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_out = new float[N];

    // input arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = float(i);
        h_b[i] = float(N - i);
    }

    // allocate memory on device (GPU)
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // start kernel with recorded time
    cudaEventRecord(start);
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // first 10 results
    for (int i = 0; i < 10; ++i) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_out[i] << std::endl;
    }

    // free memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_out;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return 0;
}