#include "mse.h"
#include <cuda_runtime.h>

// Compute Mean Square Error loss
__global__
void mse_forward_gpu(float* inp, float* out, int sz_out, float* loss) {
    int idx_start = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = idx_start; idx < sz_out; idx += stride) {
        atomicAdd(loss, fdividef(powf(inp[idx] - out[idx], 2), sz_out));
    }
}

// Compute gradient of MSE loss
__global__
void mse_backward_gpu(float* inp, float* out, int sz_out) {
    int idx_start = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = idx_start; idx < sz_out; idx += stride) {
        inp[idx] = fdividef(2.0f * (inp[idx] - out[idx]), sz_out);
    }
}

// Initialize values of Module
MSE_GPU::MSE_GPU(int _sz_out, int _block_x, int _n_block_rows) {
    sz_out = _sz_out;
    block_x = (_block_x > 0) ? _block_x : 256;
    n_blocks = (_n_block_rows > 0) ? _n_block_rows : ((sz_out + block_x - 1) / block_x);
}

// Dummy method for compatibility with other modules and performing backpropagation. Store pointers
void MSE_GPU::forward(float* _inp, float* _out) {
    inp = _inp;
    out = _out;
}

// Calculate loss, cannot be used for backpropagation
float MSE_GPU::_forward(float* _inp, float* _out) {
    float* loss;
    cudaMallocManaged(&loss, sizeof(float));
    *loss = 0.0f;

    mse_forward_gpu << <n_blocks, block_x >> > (_inp, _out, sz_out, loss);
    cudaDeviceSynchronize();

    float result = *loss;
    cudaFree(loss);
    return result;
}

// Compute backward pass using stored pointers
void MSE_GPU::backward() {
    mse_backward_gpu << <n_blocks, block_x >> > (inp, out, sz_out);
    cudaDeviceSynchronize();
}
