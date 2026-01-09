#ifndef RELU_GPU_H
#define RELU_GPU_H

#include "../utils/module.h"

class ReLU_GPU :public Module {
public:

	int n_blocks, block_x;
	ReLU_GPU(int _sz_out, int _block_x=0, int _n_block_rows=0);
	void forward(float* _inp, float* _out);
	void backward();
};

#endif