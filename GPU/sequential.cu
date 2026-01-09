#include "sequential.h"

// Perform forward pass through all layers

void sequential_forward_gpu(float* inp, std::vector<Module*> layers, float* out) {
	int sz_out;
	float* curr_out;

	for (int i = 0; i < layers.size(); i++) {
		Module* layer = layers[i];

		sz_out = layer->sz_out;

		cudaMallocManaged(&curr_out,sz_out*sizeof(float));
		layer->forward(inp, curr_out);
		
		// Set output of current layer as input for next layer
		inp = curr_out;
	}

	// Clean up memory
	cudaMallocManaged(&curr_out, sizeof(float));
	cudaFree(curr_out);
}

// Perform backward pass (update step) for all layers (from last to first)
void sequential_update_gpu(std::vector<Module*> layers) {
	for (int i = layers.size() - 1; i >= 0; i--){
		Module* layer = layers[i];

		//layer->update();
		layer->backward();
		layer->update();

	}
}

// Initialize sequence of layers
Sequential_GPU::Sequential_GPU(std::vector<Module*> _layers) {
	layers = _layers;
}

void Sequential_GPU::forward(float* inp, float* out) {
	sequential_forward_gpu(inp, layers, out);
}

void Sequential_GPU::update() {
	sequential_update_gpu(layers);
}

