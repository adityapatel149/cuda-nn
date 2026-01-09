#include "sequential.h"

// Perform forward pass through all layers

void sequential_forward_cpu(float* inp, std::vector<Module*> layers, float* out) {
	int sz_out;
	float* curr_out;

	for (int i = 0; i < layers.size(); i++) {
		Module* layer = layers[i];

		sz_out = layer->sz_out;

		curr_out = new float[sz_out];
		layer->forward(inp, curr_out);
		
		// Set output of current layer as input for next layer
		inp = curr_out;
	}

	// Clean up memory
	curr_out = new float[1];
	delete[] curr_out;
}

// Perform backward pass (update step) for all layers (from last to first)
void sequential_update_cpu(std::vector<Module*> layers) {
	for (int i = layers.size() - 1; i >= 0; i--){
		Module* layer = layers[i];

		layer->update();
		layer->backward();
	}
}

// Initialize sequence of layers
Sequential_CPU::Sequential_CPU(std::vector<Module*> _layers) {
	layers = _layers;
}

void Sequential_CPU::forward(float* inp, float* out) {
	sequential_forward_cpu(inp, layers, out);
}

void Sequential_CPU::update() {
	sequential_update_cpu(layers);
}

