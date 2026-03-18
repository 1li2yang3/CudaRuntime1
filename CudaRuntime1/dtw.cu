#include "dtw.cuh"
#include <cmath>
#include <algorithm>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <vector>
#include <omp.h>
#include <cuda_awbarrier.h>
#include <iostream>

__global__ void dtw_kernel_global(const Point* t1_raw, const Point* t2_raw, float* results, float* global_dp, int num_t, int n) {
    int bid = blockIdx.x;
    if (bid >= num_t) return;
    size_t block_offset = (size_t)bid * 3 * (n + 1);
    float* d0 = global_dp + block_offset;
    float* d1 = global_dp + block_offset + (n + 1);
    float* d2 = global_dp + block_offset + 2 * (n + 1);

    int tid = threadIdx.x;

    for (int i = tid; i <= n; i += blockDim.x) {
        d0[i] = 1e20f;
        d1[i] = 1e20f;
        d2[i] = 1e20f;
    }
    __syncthreads(); 

    if (tid == 0) {
        d0[0] = 0.0f;
    }
    __syncthreads();

    for (int k = 2; k <= 2 * n; k++) {
        int i_start = max(1, k - n);
        int i_end = min(n, k - 1);

        for (int i = i_start + tid; i <= i_end; i += blockDim.x) {
            int j = k - i;

            Point p1 = t1_raw[bid * n + (i - 1)];
            Point p2 = t2_raw[bid * n + (j - 1)];

            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;
            float cost = sqrtf(dx * dx + dy * dy);

            float diag = d0[i - 1];
            float up = d1[i - 1];
            float left = d1[i];

            d2[i] = cost + fminf(diag, fminf(up, left));
        }

        __syncthreads();

        if (k == 2 && tid == 0) d0[0] = 1e20f;

        float* temp = d0;
        d0 = d1;
        d1 = d2;
        d2 = temp;

        for (int i = i_start + tid; i <= i_end; i += blockDim.x) {
            d2[i] = 1e20f;
        }
        __syncthreads();
    }

    if (tid == 0) {
        results[bid] = d1[n];
    }
}


void launch_dtw_batch_gpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    size_t traj_size = (size_t)num_t * n * sizeof(Point);
    size_t global_dp_size = (size_t)num_t * 3 * (n + 1) * sizeof(float);
    Point* d_t1, * d_t2;
    float* d_results;
    float* d_global_dp; 

    CHECK(cudaMalloc(&d_t1, traj_size));
    CHECK(cudaMalloc(&d_t2, traj_size));
    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));
    CHECK(cudaMalloc(&d_global_dp, global_dp_size)); 

    CHECK(cudaMemcpy(d_t1, h_t1, traj_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2, h_t2, traj_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threadsPerBlock = (n < 256) ? ((n / 32 + 1) * 32) : 256;
    int blocksPerGrid = num_t;

    dtw_kernel_global << <blocksPerGrid, threadsPerBlock >> > (d_t1, d_t2, d_results, d_global_dp, num_t, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    CHECK(cudaMemcpy(h_results, d_results, num_t * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_results);
    cudaFree(d_global_dp); 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK(cudaEventRecord(stop_all));
    CHECK(cudaEventSynchronize(stop_all));
    CHECK(cudaEventElapsedTime(&time_all, start_all, stop_all));
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);

    std::cout << "\n计算时间占比: " << gpu_time / time_all;
    gpu_time = time_all;

}