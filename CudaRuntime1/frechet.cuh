#ifndef FRECHET_CUH
#define FRECHET_CUH

#include "common.cuh"

// GPU 批量 Fréchet 接口
void launch_frechet_batch_gpu_wavefront(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time);
// CPU 批量 Fréchet 接口
float launch_frechet_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n);

#endif