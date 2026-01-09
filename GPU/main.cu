#include <chrono>
#include <stdio.h>
#include "linear.h"
#include "relu.h"
#include "train.h"
#include "../data/read_csv.h"

int main() {
    std::vector<std::pair<int, int>> block_configs = {
        {8, 8}, {16, 16}, {32, 8}, {32, 16}, {32, 32}, {64, 8}, {128, 4}
    };

    std::vector<std::pair<int, int>> grid_configs = {
        {0, 0}, {8, 8}, {16, 16}, {32, 32}, {64, 16}, {16, 64}
    };

    int bs = 10000, n_in = 500, n_epochs = 5;
    int n_hidden = n_in / 2;
    float lr = 0.01f;

    FILE* fp = fopen("results.csv", "w");
    if (!fp) {
        perror("fopen failed");
        return 1;
    }
    fprintf(fp, "block_x,block_y,n_block_rows,n_block_cols,time_sec\n");

    for (auto [block_x, block_y] : block_configs) {
        if (block_x * block_y > 1024) {
            printf("Skipping (%d, %d): exceeds 1024 threads\n", block_x, block_y);
            continue;
        }

        for (auto [n_block_rows, n_block_cols] : grid_configs) {
            printf("\n===== Config: block=(%d,%d), grid=(%d,%d) =====\n",
                block_x, block_y, n_block_rows, n_block_cols);

            // Allocate unified memory 
            float* inp;
            float* targ;
            cudaMallocManaged(&inp, bs * n_in * sizeof(float));
            cudaMallocManaged(&targ, bs * sizeof(float));
            read_csv(inp, "../data/x.csv");
            read_csv(targ, "../data/y.csv");

            Linear_GPU* lin1 = new Linear_GPU(bs, n_in, n_hidden, lr,
                block_x, block_y,
                n_block_rows, n_block_cols);
            ReLU_GPU* relu1 = new ReLU_GPU(bs * n_hidden, block_x, n_block_rows);
            Linear_GPU* lin2 = new Linear_GPU(bs, n_hidden, 1, lr,
                block_x, block_y,
                n_block_rows, n_block_cols);

            std::vector<Module*> layers = { lin1, relu1, lin2 };
            Sequential_GPU seq(layers);

            auto begin = std::chrono::steady_clock::now();
            train_gpu(seq, inp, targ, bs, n_in, n_epochs);
            auto end = std::chrono::steady_clock::now();

            double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;
            printf("Training time: %.3f s\n", elapsed);
            fprintf(fp, "%d,%d,%d,%d,%.3f\n", block_x, block_y, n_block_rows, n_block_cols, elapsed);

            // Safe cleanup
            cudaDeviceSynchronize();
            delete lin1;
            delete relu1;
            delete lin2;
            cudaDeviceSynchronize();
            cudaFree(inp);
            cudaFree(targ);

            // Reset device 
            cudaDeviceReset();
            cudaSetDevice(0); 
        }
    }

    fclose(fp);
    printf("\nResults saved to results.csv\n");
    return 0;
}
