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
  fprintf(fp, "N,Time_ms\n");

  for (int exp = 15; exp <= 25; exp++) {
    int N = 1 << exp;

    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);
    cudaEventRecord(stop);

    // Wait and measure time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for correctness
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
      maxError = fmaxf(maxError, fabsf(y[i] - 4.0f));

    printf("N = 2^%d, Max error: %f, Time: %f ms\n", exp, maxError, milliseconds);
    fprintf(fp, "%d,%f\n", N, milliseconds);

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  fclose(fp);
  return 0;
}
