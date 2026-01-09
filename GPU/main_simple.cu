#include<chrono>
#include<stdio.h>

#include "linear.h"
#include "relu.h"
#include "train.h"
#include "../data/read_csv.h"

int main() {
	std::chrono::steady_clock::time_point begin, end;

	int bs = 10000, n_in = 500, n_epochs = 5;
	int n_hidden = n_in / 2;
	float lr = 0.01f;

	// Optional GPU kernel configuration
	int block_x = 32;        // threads per block dimension
	int block_y = 32;
	int n_block_rows = 0;	// 0 -> auto compute
	int n_block_cols = 0;	// 0 -> auto compute

	// Allocate unified memory for input and target
	float* inp;
	float* targ;
	cudaMallocManaged(&inp, bs * n_in * sizeof(float));
	cudaMallocManaged(&targ, bs * sizeof(float));

	begin = std::chrono::steady_clock::now();
	read_csv(inp, "../data/x.csv");
	read_csv(targ, "../data/y.csv");
	end = std::chrono::steady_clock::now();
	std::cout << "Data reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0f << std::endl;


	Linear_GPU* lin1 = new Linear_GPU(bs, n_in, n_hidden, lr, block_x, block_y, n_block_rows, n_block_cols);
	//ReLU_GPU* relu1 = new ReLU_GPU(bs * n_hidden, block_x, n_block_rows);
	ReLU_GPU* relu1 = new ReLU_GPU(bs * n_hidden);
	Linear_GPU* lin2 = new Linear_GPU(bs, n_hidden, 1, lr, block_x, block_y, n_block_rows, n_block_cols);

	std::vector<Module*> layers = { lin1, relu1, lin2 };
	Sequential_GPU seq(layers);
	
	
	//cudaMemPrefetchAsync(inp, bs * n_in * sizeof(float), 0, 0);
	//cudaMemPrefetchAsync(targ, bs * sizeof(float), 0, 0);
	//cudaMemPrefetchAsync(lin1->weights, lin1->sz_weights * sizeof(float), 0,0);
	//cudaMemPrefetchAsync(lin2->weights, lin2->sz_weights * sizeof(float), 0,0);


	begin = std::chrono::steady_clock::now();
	//train_gpu(seq, inp, targ, bs, n_in, n_epochs, block_x, n_block_rows);
	train_gpu(seq, inp, targ, bs, n_in, n_epochs);
	end = std::chrono::steady_clock::now();
	std::cout << "Training time: "
		<< (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6f)
		<< " seconds" << std::endl;


	cudaFree(inp);
	cudaFree(targ);
	return 0;
}