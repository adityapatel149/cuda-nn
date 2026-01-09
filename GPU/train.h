#ifndef TRAIN_GPU_H
#define TRAIN_GPU_H

#include "sequential.h"

void train_gpu(Sequential_GPU seq, float* input, float* target, int bs, int n_in, int n_epochs, int _block_x = 0, int _n_block_rows = 0);

#endif
