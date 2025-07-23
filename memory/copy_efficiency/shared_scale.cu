// shared memory (fast)
__global__ void shared_scale(float* input, float* output, float scale, int n) {
    __shared__ float cache[256];  // blockDim.x <= 256

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cache[threadIdx.x] = input[i];
        __syncthreads();
        output[i] = cache[threadIdx.x] * scale;
    }
}