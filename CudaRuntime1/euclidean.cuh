#ifndef EUCLIDEAN_CUH
#define EUCLIDEAN_CUH

#include "common.cuh"

void launch_euclidean_batch_gpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time);
void launch_euclidean_batch_gpu_2(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time);

float launch_euclidean_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n);
float launch_euclidean_batch_cpu_2(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n);
float launch_euclidean_batch_cpu_rtree(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n);


void launch_euclidean_batch_gpu_rtree_exact(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time);
void execute_rtree_csr_kernel_on_gpu(const Point* h_t1, const Point* h_t2,
    const int* h_candidates, const int* h_offsets,
    float* h_results, int num_t, int n, int total_candidates, float& gpu_time);


#endif