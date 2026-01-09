#ifndef MSE_GPU_H
#define MSE_GPU_H

#include "../utils/module.h"

class MSE_GPU : public Module {
public:
	float* inp, * out;
	int n_blocks, block_x;
	MSE_GPU(int _sz_out, int _block_x = 0, int _n_block_rows = 0);
	void forward(float* _inp, float* _out);
	float _forward(float* _inp, float* _out);
	void backward();
};

#endif // !MSE_GPU_H
