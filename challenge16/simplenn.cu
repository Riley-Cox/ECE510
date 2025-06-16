
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define INPUT_SIZE 4
#define HIDDEN_SIZE 5
#define OUTPUT_SIZE 1
#define NUM_RUNS 1000


__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void forward_hidden_layer(
    float *inputs, float *weights, float *biases, float *outputs)
{
    int i = threadIdx.x;
    if (i < HIDDEN_SIZE) {
        float sum = 0.0f;
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += inputs[j] * weights[i * INPUT_SIZE + j];
        }
        sum += biases[i];
        outputs[i] = sigmoid(sum);
    }
}

__global__ void forward_output_layer(
    float *hidden, float *weights, float *biases, float *output)
{
    float sum = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        sum += hidden[i] * weights[i];
    }
    sum += biases[0];
    *output = sigmoid(sum);
}

int main() {
    float h_input[INPUT_SIZE] = {1.0f, 0.5f, -0.5f, 0.2f};

    float h_hidden_weights[HIDDEN_SIZE * INPUT_SIZE] = {
         0.2f, -0.1f,  0.4f,  0.3f,
        -0.3f,  0.1f,  0.6f, -0.4f,
         0.5f,  0.2f, -0.5f,  0.1f,
        -0.2f,  0.4f,  0.2f,  0.3f,
         0.1f, -0.3f,  0.5f,  0.6f
    };

    float h_hidden_biases[HIDDEN_SIZE] = {0.1f, -0.2f, 0.0f, 0.3f, -0.1f};
    float h_output_weights[HIDDEN_SIZE] = {0.3f, -0.5f, 0.2f, 0.4f, 0.1f};
    float h_output_bias[1] = {0.05f};

    float h_hidden_out[HIDDEN_SIZE];
    float h_output;

    float *d_input, *d_hidden_weights, *d_hidden_biases, *d_hidden_out;
    float *d_output_weights, *d_output_bias, *d_output;

    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_hidden_weights, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_hidden_biases, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_hidden_out, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output_weights, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output_bias, sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_weights, h_hidden_weights, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_biases, h_hidden_biases, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_weights, h_output_weights, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_bias, h_output_bias, sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up run
    forward_hidden_layer<<<1, HIDDEN_SIZE>>>(d_input, d_hidden_weights, d_hidden_biases, d_hidden_out);
    forward_output_layer<<<1, 1>>>(d_hidden_out, d_output_weights, d_output_bias, d_output);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < NUM_RUNS; ++i) {
        forward_hidden_layer<<<1, HIDDEN_SIZE>>>(d_input, d_hidden_weights, d_hidden_biases, d_hidden_out);
        forward_output_layer<<<1, 1>>>(d_hidden_out, d_output_weights, d_output_bias, d_output);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time = elapsed_ms / NUM_RUNS;

    cudaMemcpy(h_hidden_out, d_hidden_out, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Final output: " << h_output << std::endl;
    std::cout << "[Benchmark] Total time for " << NUM_RUNS << " runs: " << elapsed_ms << " ms" << std::endl;
    std::cout << "[Benchmark] Average per inference: " << avg_time << " ms" << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_hidden_weights);
    cudaFree(d_hidden_biases);
    cudaFree(d_hidden_out);
    cudaFree(d_output_weights);
    cudaFree(d_output_bias);
    cudaFree(d_output);

    return 0;
}
