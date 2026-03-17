#ifndef LCSS_CUH
#define LCSS_CUH

#include "common.cuh"

// GPU 批量 LCSS 接口
void launch_lcss_batch_gpu_wavefront(const Point* h_t1, const Point* h_t2, float* h_results,int num_t, int n, float epsilon, float& gpu_time);

// CPU 批量 LCSS 接口
float launch_lcss_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float epsilon);


#endif