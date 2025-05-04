#include <iostream>
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
        d_fib[idx] = fib(idx);  // Compute Fibonacci for the thread index
    }
}

int main() {
    int N = 1 << 20; // 2^20
    unsigned long long *d_fib, *h_fib;

    // Allocate host memory
    h_fib = (unsigned long long*)malloc(N * sizeof(unsigned long long));

    // Allocate device memory
    cudaMalloc((void**)&d_fib, N * sizeof(unsigned long long));

    // Set up execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    fibonacci_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_fib, N);

    // Copy results back to host
    cudaMemcpy(h_fib, d_fib, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Optional: Print the first few Fibonacci numbers to verify
    for (int i = 0; i < 10; ++i) {
        std::cout << "Fib(" << i << ") = " << h_fib[i] << std::endl;
    }

    // Free memory
    cudaFree(d_fib);
    free(h_fib);

    return 0;
}
