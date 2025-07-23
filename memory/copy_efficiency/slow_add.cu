// only using global memory
__global__ void slow_add_kernel(float* a, float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_out = new float[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 256, gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    slow_add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_out, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Slow add took " << ms << " ms\n";

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    delete[] h_a;
    delete[] h_b;
    delete[] h_out;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}