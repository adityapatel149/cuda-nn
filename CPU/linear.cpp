#include "linear.h"
#include "../utils/utils.h"

void linear_forward_cpu(float* inp, float* weights, float* bias, float* out, int bs, int n_in, int n_out) {
	int idx_i, idx_w, idx_o;

	for (int i = 0; i < bs; i++) {
		for (int k = 0; k < n_out; k++) {
			idx_o = i * n_out + k;
			out[idx_o] = bias[k];

			for (int j = 0; j < n_in; j++) {
				idx_i = i * n_in + j;
				idx_w = j * n_out + k;

				out[idx_o] += inp[idx_i] * weights[idx_w];
			}
		}
	}
}

void linear_backward_cpu(float* inp, float* weights, float* out, int bs, int n_in, int n_out) {
	int idx_i, idx_w, idx_o;
	for (int i = 0; i < bs; i++) {
		for (int k = 0; k < n_out; k++) {
			idx_o = i * n_out + k;

			for (int j = 0; j < n_in; j++) {
				idx_i = i * n_in + j;
				idx_w = j * n_out + k;

				inp[idx_i] += weights[idx_w] * out[idx_o];
			}
		}
	}
}

void linear_update_cpu(float* inp, float* weights, float* bias, float* out, int bs, int n_in, int n_out, float lr) {
	int idx_i, idx_w, idx_o;
	for (int i = 0; i < bs; i++) {
		for (int k = 0; k < n_out; k++) {
			idx_o = i * n_out + k;
			bias[k] -= lr * out[idx_o];

			for (int j = 0; j < n_in; j++) {
				idx_i = i * n_in + j;
				idx_w = j * n_out + k;

				weights[idx_w] -= lr * inp[idx_i] * out[idx_o];
			}
		}
	}
}

Linear_CPU::Linear_CPU(int _bs, int _n_in, int _n_out, float _lr) {
	bs = _bs;
	n_in = _n_in;
	n_out = _n_out;
	lr = _lr;

	sz_weights = n_in * n_out;
	sz_out = bs * n_out;
	weights = new float[sz_weights];
	bias = new float[n_out];

	kaiming_init(weights, n_in, n_out);
	init_zero(bias, n_out);
}


void Linear_CPU::forward(float* _inp, float* _out) {
	inp = _inp;
	out = _out;
	linear_forward_cpu(inp, weights, bias, out, bs, n_in, n_out);
}

void Linear_CPU::backward() {
	init_zero(inp, bs * n_in);
	linear_backward_cpu(inp, cp_weights, out, bs, n_in, n_out);

	delete[] cp_weights;
	delete[] out;
}

void Linear_CPU::update() {
	cp_weights = new float[n_in * n_out];
	cpy_array(cp_weights, weights, sz_weights);
	linear_update_cpu(inp, weights, bias, out, bs, n_in, n_out, lr);
}