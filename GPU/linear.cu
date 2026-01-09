#include "linear.h"
#include "../utils/utils.h"
#include <cuda_runtime.h>
#include <iostream>

__global__
void linear_forward_gpu(const float* __restrict__ inp,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int bs, int n_in, int n_out)
{
    int row_start = blockIdx.x * blockDim.x + threadIdx.x;  // batch index start
    int col_start = blockIdx.y * blockDim.y + threadIdx.y;  // output index start
    int row_stride = blockDim.x * gridDim.x;
    int col_stride = blockDim.y * gridDim.y;

    for (int row = row_start; row < bs; row += row_stride) {
        for (int col = col_start; col < n_out; col += col_stride) {
            float val = bias[col];
            for (int j = 0; j < n_in; j++) {
                int idx_i = row * n_in + j;
                int idx_w = j * n_out + col;
                val += inp[idx_i] * weights[idx_w];
            }
            out[row * n_out + col] = val;
        }
    }
}

__global__
void linear_backward_gpu(float* __restrict__ inp,
    const float* __restrict__ weights,
    const float* __restrict__ out,
    int bs, int n_in, int n_out)
{
    int row_start = blockIdx.x * blockDim.x + threadIdx.x;
    int j_start = blockIdx.y * blockDim.y + threadIdx.y;
    int row_stride = blockDim.x * gridDim.x;
    int j_stride = blockDim.y * gridDim.y;

    for (int row = row_start; row < bs; row += row_stride) {
        for (int j = j_start; j < n_in; j += j_stride) {
            float grad = 0.0f;
            for (int col = 0; col < n_out; col++) {
                int idx_o = row * n_out + col;
                int idx_w = j * n_out + col;
                grad += weights[idx_w] * out[idx_o];
            }
            inp[row * n_in + j] = grad;
        }
    }
}

__global__
void linear_update_gpu(const float* __restrict__ inp,
    float* __restrict__ weights,
    float* __restrict__ bias,
    const float* __restrict__ out,
    int bs, int n_in, int n_out, float lr)
{
    int j_start = blockIdx.x * blockDim.x + threadIdx.x;
    int col_start = blockIdx.y * blockDim.y + threadIdx.y;
    int j_stride = blockDim.x * gridDim.x;
    int col_stride = blockDim.y * gridDim.y;

    for (int j = j_start; j < n_in; j += j_stride) {
        for (int col = col_start; col < n_out; col += col_stride) {
            float grad_w = 0.0f;
            float grad_b = 0.0f;

            for (int row = 0; row < bs; row++) {
                int idx_i = row * n_in + j;
                int idx_o = row * n_out + col;
                grad_w += inp[idx_i] * out[idx_o];
                if (j == 0) grad_b += out[idx_o];
            }

            weights[j * n_out + col] -= lr * grad_w;
            if (j == 0) bias[col] -= lr * grad_b;
        }
    }
}

Linear_GPU::Linear_GPU(int _bs, int _n_in, int _n_out, float _lr,
    int _block_x, int _block_y,
    int _n_block_rows, int _n_block_cols)
{
    bs = _bs;
    n_in = _n_in;
    n_out = _n_out;
    lr = _lr;

    sz_weights = n_in * n_out;
    sz_out = bs * n_out;

    // Thread block config
    block_x = (_block_x > 0) ? _block_x : 32;
    block_y = (_block_y > 0) ? _block_y : 32;

    n_block_rows = (_n_block_rows > 0) ? _n_block_rows : (bs + block_x - 1) / block_x;
    n_block_cols = (_n_block_cols > 0) ? _n_block_cols : (n_out + block_y - 1) / block_y;

    cudaMallocManaged(&weights, sz_weights * sizeof(float));
    cudaMallocManaged(&bias, n_out * sizeof(float));

    kaiming_init(weights, n_in, n_out);
    cudaMemset(bias, 0, sizeof(float) * n_out);

    std::cout << "Linear_GPU created with blocks (" << block_x << "x" << block_y
        << ") and grid (" << n_block_rows << "x" << n_block_cols << ")" << std::endl;
}

void Linear_GPU::forward(float* _inp, float* _out) {
    inp = _inp;
    out = _out;

    dim3 n_blocks(n_block_rows, n_block_cols);
    dim3 n_threads(block_x, block_y);

    linear_forward_gpu << <n_blocks, n_threads >> > (inp, weights, bias, out, bs, n_in, n_out);
    cudaDeviceSynchronize();
}

void Linear_GPU::backward() {
    dim3 n_blocks((bs + block_x - 1) / block_x,
        (n_in + block_y - 1) / block_y);
    dim3 n_threads(block_x, block_y);

    linear_backward_gpu << <n_blocks, n_threads >> > (inp, weights, out, bs, n_in, n_out);
    cudaDeviceSynchronize();
}

void Linear_GPU::update() {
    dim3 n_blocks((n_in + block_x - 1) / block_x,
        (n_out + block_y - 1) / block_y);
    dim3 n_threads(block_x, block_y);

    linear_update_gpu << <n_blocks, n_threads >> > (inp, weights, bias, out, bs, n_in, n_out, lr);
    cudaDeviceSynchronize();
}
