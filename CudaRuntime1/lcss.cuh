#ifndef LCSS_CUH
#define LCSS_CUH

#include "common.cuh"

void launch_lcss_batch_gpu_wavefront(const Point* h_t1, const Point* h_t2, float* h_results,int num_t, int n, float epsilon, float& gpu_time);

float launch_lcss_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float epsilon);


#endif