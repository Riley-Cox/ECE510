#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main(void)
{
  // Warm-up to initialize CUDA context (not timed)
  float *d_x_warmup, *d_y_warmup;
  cudaMalloc(&d_x_warmup, 256 * sizeof(float));
  cudaMalloc(&d_y_warmup, 256 * sizeof(float));
  saxpy<<<1, 256>>>(256, 2.0f, d_x_warmup, d_y_warmup);
  cudaDeviceSynchronize();
  cudaFree(d_x_warmup);
  cudaFree(d_y_warmup);

  // Start CSV file
  FILE *fp = fopen("timing.csv", "w");
  fprintf(fp, "N,TotalTime_ms,KernelTime_ms\n");

  for (int exp = 15; exp <= 25; exp++) {
    int N = 1 << exp;

    // Total time measurement start
    cudaEvent_t totalStart, totalStop;
    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalStop);
    cudaEventRecord(totalStart);

    // Host allocations
    float *x = (float*)malloc(N * sizeof(float));
    float *y = (float*)malloc(N * sizeof(float));

    // Device allocations
    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

    // Copy to device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel-only time measurement
    cudaEvent_t kernelStart, kernelStop;
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    cudaEventRecord(kernelStart);
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);
    cudaEventRecord(kernelStop);

    cudaEventSynchronize(kernelStop);

    float kernelTimeMs = 0;
    cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop);

    // Copy back to host
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check correctness
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
      maxError = fmaxf(maxError, fabsf(y[i] - 4.0f));

    // Total time measurement stop
    cudaEventRecord(totalStop);
    cudaEventSynchronize(totalStop);

    float totalTimeMs = 0;
    cudaEventElapsedTime(&totalTimeMs, totalStart, totalStop);

    // Print and log both times
    printf("N = 2^%d, Max error: %f, Total Time: %f ms, Kernel Time: %f ms\n",
           exp, maxError, totalTimeMs, kernelTimeMs);
    fprintf(fp, "%d,%f,%f\n", N, totalTimeMs, kernelTimeMs);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);
    cudaEventDestroy(totalStart);
    cudaEventDestroy(totalStop);
  }

  fclose(fp);
  return 0;
}
