// non contiguous, bad access
__global__ void strided_copy(float* input, float* output, int stride, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * stride;
    if (idx < n) {
        output[idx] = input[idx];  // scattered memory access
    }
}