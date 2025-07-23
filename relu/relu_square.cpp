#include <iostream>
#include <chrono>
#include <cmath>

int main() {
    int N = 1024 * 1024;
    float* x = new float[N];
    float* out = new float[N];

    // init range -512K to +511K
    for (int i = 0; i < N; ++i)
        x[i] = i - N / 2;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        float val = x[i] > 0 ? x[i] : 0.0f;
        out[i] = val * val;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = end - start;

    // values near ReLU threshold (just like CUDA)
    for (int i = N / 2 - 5; i < N / 2 + 5; ++i) {
        std::cout << "ReLU^2(" << x[i] << ") = " << out[i] << std::endl;
    }

    std::cout << "\n⏱️ CPU ReLU+Square time: " << dur.count() << " ms\n";

    delete[] x;
    delete[] out;
    return 0;
}