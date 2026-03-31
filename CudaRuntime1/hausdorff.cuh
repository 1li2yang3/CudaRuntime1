#ifndef HAUSDORFF_CUH
#define HAUSDORFF_CUH

#include "common.cuh"


float launch_hausdorff_batch_cpu_rtree(const Point* h_t1, int num_t1, int n,
    const Point* h_t2, int num_t2, int m,
    float* h_results, int top_k = 10);

float run_hausdorff_rtree_gpu_pipeline(const Point* h_t1, int num_t1, int n,
    const Point* h_t2, int num_t2, int m,
    float* h_results, int top_k = 10);


void launch_hausdorff_rtree_gpu(const Point* h_t1, int num_t1, int n,
    const Point* h_t2, int num_t2, int m,
    const int* h_candidate_indices, int top_k,
    float* h_results, float& gpu_time);
#endif