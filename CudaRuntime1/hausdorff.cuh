#ifndef HAUSDORFF_CUH
#define HAUSDORFF_CUH

#include "common.cuh"

// GPU 接口
void launch_hausdorff_batch_gpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time);

// CPU 接口
float launch_hausdorff_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n);
float launch_hausdorff_batch_cpu_grid(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n);

#endif