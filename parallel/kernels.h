/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel();

#ifndef KERNEL_H
#define KERNEL_H

void launch_forward_relu(
    const float *x,
    const float *W,
    const float *b,
    float *out,
    int in_dim,
    int out_dim
);

void launch_forward_linear(
    const float *x,
    const float *W,
    const float *b,
    float *out,
    int in_dim,
    int out_dim
);

void launch_softmax(float *x, int n);

void launch_output_delta(
    const float *out,
    const float *label,
    float *delta,
    int n
);

void launch_hidden_delta(
    const float *next_delta,
    const float *W_next,
    const float *activated,
    float *delta,
    int curr_dim,
    int next_dim
);

void launch_update_weights(
    float *W,
    const float *input,
    const float *delta,
    float lr,
    int in_dim,
    int out_dim
);

void launch_update_bias(
    float *b,
    const float *delta,
    float lr,
    int dim
);

void launch_loss(
    const float *out,
    const float *label,
    float *loss,
    int n
);
#endif
