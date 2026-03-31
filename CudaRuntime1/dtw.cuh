#ifndef DTW_CUH
#define DTW_CUH

#include "common.cuh"

void launch_dtw_batch_gpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n,int m, float& gpu_time);
void launch_dtw_batch_gpu_knn(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m, float& gpu_time);
float launch_dtw_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n,int m);
#endif