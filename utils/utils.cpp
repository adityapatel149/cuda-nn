#include<math.h>
#include<iostream>
#include<random>

#include "utils.h"

// Max Difference between elements of two arrays
float max_diff(float* res1, float* res2, int n) {
	float diff, max = 0;
	for (int i = 0; i < n; i++) {
		diff = abs(res1[i] - res2[i]);
		max = (max < diff) ? diff: max;
	}
	return max;
}

// Return number of zeros in an array
int n_zeros(float* a, int n) {
	int r = 0;
	for (int i = 0; i < n; i++) {
		r += (!a[i]);
	}
	return r;
}

// Randomly fill an array with values between -1 and 1
void fill_array(float* a, int n) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> dist(0.0f, 1.0f); //(mean, standard deviation)

	for (int i = 0; i < n; i++) {
		a[i] = dist(gen);
	}
}


void test_res(float* res1, float* res2, int n) {
	int n_res1_zeros = n_zeros(res1, n), n_res2_zeros = n_zeros(res2, n);
	float max = max_diff(res1, res2, n);

	std::cout << "Number of zeros of res1: " << n_res1_zeros << std::endl;
	std::cout << "Number of zeros of res2: " << n_res2_zeros << std::endl;
	std::cout << "Maximum difference: " << max << std::endl;
	std::cout << "---------------------------------------" << std::endl;
}


void print_array(float* a, int n) {
	for (int i = 0; i < n; i++) {
		std::cout<< a[i] << std::endl;
	}
	std::cout << "---------------------------------------" << std::endl;
}


// Initialize an array with zeros
void init_zero(float* a, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = 0.0f;
	}
}

void cpy_array(float* a, float* b, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = b[i];
	}
}

void kaiming_init(float* w, int n_in, int n_out) {
	float std = sqrt(2 / (float)n_in);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> dist(0.0f, std);

	for (int i = 0; i < n_in*n_out; i++) {
		w[i] = dist(gen);
	}
}

int random_int(int min, int max) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(min, max);
	return dist(gen);
}