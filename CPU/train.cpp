#include <iostream>

#include "mse.h"
#include "train.h"
#include "../utils/utils.h"


void train_cpu(Sequential_CPU seq, float* inp, float* targ, int bs, int n_in, int n_epochs) {
	MSE_CPU mse(bs);

	int sz_inp = bs * n_in;
	float* cp_inp = new float[sz_inp];
	float* out = new float[bs];
	for (int i = 0; i < n_epochs; i++) {
		cpy_array(cp_inp, inp, sz_inp);
		seq.forward(cp_inp, out); //Forward pass all linear layers

		mse.forward(seq.layers.back()->out, targ); 
		mse.backward(); 
		seq.update(); //Update and backward propagation of all layers
	}
	delete[] cp_inp;

	seq.forward(inp, out);
	float loss = mse._forward(seq.layers.back()->out, targ);
	std::cout << "The final loss is: " << loss << std::endl;
}
