#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

__device__ unsigned long long fib(int n) {
    if (n <= 1) return n;
    unsigned long long a = 0, b = 1, c;
    for (int i = 2; i <= n; ++i) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

__global__ void fibonacci_kernel(unsigned long long* d_fib, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        // Fibonacci computation in parallel
        if (idx == 0) {
            d_fib[0] = 0;
        } else if (idx == 1) {
            d_fib[1] = 1;
        } else {
            d_fib[idx] = d_fib[idx-1] + d_fib[idx-2];  // Parallelize Fibonacci logic
        }
    }
}

void fibonacci_sequential(unsigned long long* fib, int N) {
    if (N > 0) fib[0] = 0;
    if (N > 1) fib[1] = 1;
    for (int i = 2; i < N; ++i) {
        fib[i] = fib[i - 1] + fib[i - 2];
    }
}

int main() {
    int Ns[] = {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20};
    int num_sizes = sizeof(Ns) / sizeof(Ns[0]);

    std::ofstream fout("times.csv");
    fout << "N,CPU_ms,GPU_kernel_ms,GPU_memcpy_ms,GPU_total_ms\n";

    for (int s = 0; s < num_sizes; ++s) {
        int N = Ns[s];
        unsigned long long *h_fib_cpu = new unsigned long long[N];
        unsigned long long *h_fib_gpu = new unsigned long long[N];
        unsigned long long *d_fib;

        cudaMalloc(&d_fib, N * sizeof(unsigned long long));

        // CPU timing
        auto start_cpu = std::chrono::high_resolution_clock::now();
        fibonacci_sequential(h_fib_cpu, N);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

        // GPU kernel timing
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        cudaEvent_t start_kernel, stop_kernel;
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);

        cudaEventRecord(start_kernel);
        fibonacci_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_fib, N);
        cudaEventRecord(stop_kernel);
        cudaEventSynchronize(stop_kernel);

        float gpu_kernel_ms = 0;
        cudaEventElapsedTime(&gpu_kernel_ms, start_kernel, stop_kernel);

        // Memory copy timing
        auto start_memcpy = std::chrono::high_resolution_clock::now();
        cudaMemcpy(h_fib_gpu, d_fib, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        auto end_memcpy = std::chrono::high_resolution_clock::now();
        double gpu_memcpy_ms = std::chrono::duration<double, std::milli>(end_memcpy - start_memcpy).count();

        // Total GPU time (kernel + memory copy)
        double gpu_total_ms = gpu_kernel_ms + gpu_memcpy_ms;

        // Save times to CSV
        fout << N << "," << cpu_ms << "," << gpu_kernel_ms << "," << gpu_memcpy_ms << "," << gpu_total_ms << "\n";

        // Cleanup
        delete[] h_fib_cpu;
        delete[] h_fib_gpu;
        cudaFree(d_fib);
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);
    }

    fout.close();
    return 0;
}

