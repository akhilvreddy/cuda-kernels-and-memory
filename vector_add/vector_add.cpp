#include <iostream>
#include <chrono>

int main() {
    int N = 1024;
    float *a = new float[N];
    float *b = new float[N];
    float *out = new float[N];

    for (int i = 0; i < N; ++i) {
        a[i] = float(i);
        b[i] = float(N - i);
    }

    // timer
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        out[i] = a[i] + b[i];
    }

    // end timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    for (int i = 0; i < 10; ++i) {
        std::cout << a[i] << " + " << b[i] << " = " << out[i] << std::endl;
    }

    std::cout << "CPU execution time: " << duration_ms.count() << " ms" << std::endl;

    delete[] a;
    delete[] b;
    delete[] out;
    return 0;
}