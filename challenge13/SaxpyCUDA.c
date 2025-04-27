#include <stdio.h>
#include <cuda_runtime.h>
#include <fstream>

__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main(void)
{
    int start_exp = 15;
    int end_exp = 25;

    std:ofstream outFile("execution_times.csv");
    outFile << "MatrixSize,ExecutionTime_ms\n");

    for (int exp = start_exp; exp <= end_exp; exp++) {
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

        // Start recording
        cudaEventRecord(start);

        // Launch kernel
        saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

        // Stop recording
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Copy result back to host (optional for correctness checking)
        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Validate result
        float maxError = 0.0f;
        for (int i = 0; i < N; i++)
            maxError = fmax(maxError, fabs(y[i] - 4.0f));

        printf("Matrix size 2^%d (%d elements): Max error = %f, Execution time = %f ms\n",
               exp, N, maxError, milliseconds);
        
  	outFile << exp << "," << milliseconds << "\n";

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_x);
        cudaFree(d_y);
        free(x);
        free(y);
    }
	outFile.close();

    return 0;
}

