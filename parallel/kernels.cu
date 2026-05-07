/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */

#include <cuda_runtime.h>
#include "kernels.h"

__global__ void forward_relu_kernel(
    const float *x,
    const float *W,
    const float *b,
    float *out,
    int in_dim,
    int out_dim
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < out_dim) {
        float sum = b[j];

        for (int i = 0; i < in_dim; i++) {
            sum += x[i] * W[i * out_dim + j];
        }

        out[j] = sum > 0.0f ? sum : 0.0f;
    }
}

void launch_forward_relu(
    const float *x,
    const float *W,
    const float *b,
    float *out,
    int in_dim,
    int out_dim
) {
    int threads = 256;
    int blocks = (out_dim + threads - 1) / threads;

    forward_relu_kernel<<<blocks, threads>>>(
        x, W, b, out, in_dim, out_dim
    );
}

__global__ void forward_linear_kernel(
    const float *x,
    const float *W,
    const float *b,
    float *out,
    int in_dim,
    int out_dim
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < out_dim) {
        float sum = b[j];

        for (int i = 0; i < in_dim; i++) {
            sum += x[i] * W[i * out_dim + j];
        }

        out[j] = sum;
    }
}

void launch_forward_linear(
    const float *x,
    const float *W,
    const float *b,
    float *out,
    int in_dim,
    int out_dim
) {
    int threads = 256;
    int blocks = (out_dim + threads - 1) / threads;

    forward_linear_kernel<<<blocks, threads>>>(
        x, W, b, out, in_dim, out_dim
    );
}

__global__ void softmax_kernel(float *x, int n) {
    __shared__ float max_val;
    __shared__ float sum;

    if (threadIdx.x == 0) {
        max_val = x[0];
        for (int i = 1; i < n; i++) {
            if (x[i] > max_val) max_val = x[i];
        }

        sum = 0.0f;
        for (int i = 0; i < n; i++) {
            x[i] = expf(x[i] - max_val);
            sum += x[i];
        }

        for (int i = 0; i < n; i++) {
            x[i] /= sum;
        }
    }
}

void launch_softmax(float *x, int n) {
    softmax_kernel<<<1, 1>>>(x, n);
}

__global__ void output_delta_kernel(
    const float *out,
    const float *label,
    float *delta,
    int n
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < n) {
        delta[k] = label[k] - out[k];
    }
}

void launch_output_delta(
    const float *out,
    const float *label,
    float *delta,
    int n
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    output_delta_kernel<<<blocks, threads>>>(out, label, delta, n);
}

__global__ void hidden_delta_kernel(
    const float *next_delta,
    const float *W_next,
    const float *activated,
    float *delta,
    int curr_dim,
    int next_dim
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < curr_dim) {
        float sum = 0.0f;

        for (int k = 0; k < next_dim; k++) {
            sum += next_delta[k] * W_next[j * next_dim + k];
        }

        float deriv = activated[j] > 0.0f ? 1.0f : 0.0f;
        delta[j] = deriv * sum;
    }
}

void launch_hidden_delta(
    const float *next_delta,
    const float *W_next,
    const float *activated,
    float *delta,
    int curr_dim,
    int next_dim
) {
    int threads = 256;
    int blocks = (curr_dim + threads - 1) / threads;

    hidden_delta_kernel<<<blocks, threads>>>(
        next_delta,
        W_next,
        activated,
        delta,
        curr_dim,
        next_dim
    );
}

__global__ void update_weights_kernel(
    float *W,
    const float *input,
    const float *delta,
    float lr,
    int in_dim,
    int out_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = in_dim * out_dim;

    if (idx < total) {
        int i = idx / out_dim;
        int j = idx % out_dim;

        W[i * out_dim + j] += lr * input[i] * delta[j];
    }
}

void launch_update_weights(
    float *W,
    const float *input,
    const float *delta,
    float lr,
    int in_dim,
    int out_dim
) {
    int threads = 256;
    int total = in_dim * out_dim;
    int blocks = (total + threads - 1) / threads;

    update_weights_kernel<<<blocks, threads>>>(
        W, input, delta, lr, in_dim, out_dim
    );
}

__global__ void update_bias_kernel(
    float *b,
    const float *delta,
    float lr,
    int dim
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < dim) {
        b[j] += lr * delta[j];
    }
}

void launch_update_bias(
    float *b,
    const float *delta,
    float lr,
    int dim
) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;

    update_bias_kernel<<<blocks, threads>>>(b, delta, lr, dim);
}

__global__ void loss_kernel(
    const float *out,
    const float *label,
    float *loss,
    int n
) {
    int k = threadIdx.x;

    if (k < n) {
        float val = -label[k] * logf(out[k] + 1e-8f);
        atomicAdd(loss, val);
    }
}

void launch_loss(
    const float *out,
    const float *label,
    float *loss,
    int n
) {
    loss_kernel<<<1, 32>>>(out, label, loss, n);
}
