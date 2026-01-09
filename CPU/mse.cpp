#include "mse.h"

// Compute Mean Square Error loss
void mse_forward_cpu(float* inp, float* out, int sz_out, float* loss) {
	*loss = 0.0f;
	for (int i = 0; i < sz_out; i++) {
		*loss += (inp[i] - out[i]) * (inp[i] - out[i]);
	}
	*loss /= sz_out;
}

// Compute gradient of MSE loss
void mse_backward_cpu(float* inp, float* out, int sz_out) {
	for (int i = 0; i < sz_out; i++) {
		inp[i] = 2 * (inp[i] - out[i]) / sz_out;
	}
}

// Initialize values of Module
MSE_CPU::MSE_CPU(int _sz_out) {
	sz_out = _sz_out;
}

// Dummy method for compatibility with other modules and performing backpropagation. Store pointers
void MSE_CPU::forward(float* _inp, float* _out) {
	inp = _inp;
	out = _out;
}

// Calculate loss, cannot be used for backpropagation
float MSE_CPU::_forward(float* _inp, float* _out) {
	float loss;
	mse_forward_cpu(_inp, _out, sz_out, &loss);
	return loss;
}

// Compute backward pass using stored pointers
void MSE_CPU::backward() {
	mse_backward_cpu(inp, out, sz_out);
}