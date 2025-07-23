#include <iostream>
#include <cuda_runtime.h>

__global__ void warp_diverge(float* a, float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (i % 2 == 0)
            out[i] = a[i] * 2;
        else
            out[i] = b[i] * 3;
    }
}

int main() {
    int N = 1024 * 1024;
    size_t size = N * sizeof(float);
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_out = new float[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 0.001f;
        h_b[i] = i * 0.002f;
    }

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    warp_diverge<<<(N + 255)/256, 256>>>(d_a, d_b, d_out, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    for (int i = N / 2 - 5; i < N / 2 + 5; ++i)
        std::cout << "out[" << i << "] = " << h_out[i] << std::endl;

    std::cout << "\n Warp-divergent time: " << ms << " ms\n";

    delete[] h_a; delete[] h_b; delete[] h_out;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}