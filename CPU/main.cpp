#include<chrono>
#include<stdio.h>

#include "linear.h"
#include "relu.h"
#include "train.h"
#include "../data/read_csv.h"

int main() {
	std::chrono::steady_clock::time_point begin, end;

	int bs = 1000, n_in = 50000, n_epochs = 1;
	int n_hidden = n_in / 2;
	float lr = 0.01f;

	float* inp = new float[bs * n_in], * targ = new float[bs];
	
	begin = std::chrono::steady_clock::now();
	read_csv(inp, "../data/x.csv");
	read_csv(targ, "../data/y.csv");
	end = std::chrono::steady_clock::now();
	std::cout << "Data reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0f << std::endl;


	Linear_CPU* lin1 = new Linear_CPU(bs, n_in, n_hidden, lr);
	ReLU_CPU* relu1 = new ReLU_CPU(bs * n_hidden);
	Linear_CPU* lin2 = new Linear_CPU(bs, n_hidden, 1, lr);

	std::vector<Module*> layers = { lin1, relu1, lin2 };
	Sequential_CPU seq(layers);

	begin = std::chrono::steady_clock::now();
	train_cpu(seq, inp, targ, bs, n_in, n_epochs);
	end = std::chrono::steady_clock::now();
	std::cout << "Training time: "
		<< (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6f)
		<< " seconds" << std::endl;

	delete[] inp;
	delete[] targ;
	return 0;
}