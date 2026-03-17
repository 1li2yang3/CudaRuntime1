#ifndef EUCLIDEAN_CUH
#define EUCLIDEAN_CUH

#include "common.cuh"

void launch_euclidean_batch_gpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time);

float launch_euclidean_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n);


#endif