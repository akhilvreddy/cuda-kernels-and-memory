// contiguous memory
__global__ void coalesced_copy(float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i];  // memory coalescing
    }
}