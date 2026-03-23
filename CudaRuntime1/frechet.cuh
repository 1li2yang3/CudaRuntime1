#ifndef FRECHET_CUH
#define FRECHET_CUH

#include "common.cuh"

void launch_frechet_batch_gpu_wavefront(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m, float& gpu_time);

float launch_frechet_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n,int m);

#endif