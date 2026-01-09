#include "relu.h"
#include <cuda_runtime.h>

// Forward pass with grid-stride loop
__global__
void relu_forward_gpu(float* inp, float* out, int sz_out) {
    int idx_start = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = idx_start; idx < sz_out; idx += stride) {
        out[idx] = fmaxf(0.0f, inp[idx]);
    }
}

// Backward pass with grid-stride loop
__global__
void relu_backward_gpu(float* inp, float* out, int sz_out) {
    int idx_start = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = idx_start; idx < sz_out; idx += stride) {
        inp[idx] = (0 < inp[idx]) * out[idx];
    }
}

ReLU_GPU::ReLU_GPU(int _sz_out, int _block_x, int _n_block_rows) {
    sz_out = _sz_out;
    block_x = (_block_x > 0) ? _block_x : 256;
    n_blocks = (_n_block_rows > 0) ? _n_block_rows : ((sz_out + block_x - 1) / block_x);
}

void ReLU_GPU::forward(float* _inp, float* _out) {
    inp = _inp;
    out = _out;
    relu_forward_gpu << <n_blocks, block_x >> > (inp, out, sz_out);
    cudaDeviceSynchronize();
}

void ReLU_GPU::backward() {
    relu_backward_gpu << <n_blocks, block_x >> > (inp, out, sz_out);
    cudaDeviceSynchronize();
    //cudaFree(out);
}
