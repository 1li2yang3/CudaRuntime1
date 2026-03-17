#include "dtw.cuh"
#include <cmath>
#include <algorithm>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <vector>
#include <omp.h>
#include <cuda_awbarrier.h>
#include <iostream>


__global__ void dtw_kernel_wavefront(const Point* t1_raw, const Point* t2_raw, float* results, int num_t, int n) {
    int bid = blockIdx.x;
    if (bid >= num_t) return;

    extern __shared__ char s_mem[];

    Point* s_t1 = (Point*)s_mem;
    Point* s_t2 = (Point*)(s_mem + n * sizeof(Point));

    float* d0 = (float*)(s_mem + 2 * n * sizeof(Point));
    float* d1 = (float*)(s_mem + 2 * n * sizeof(Point) + (n + 1) * sizeof(float));
    float* d2 = (float*)(s_mem + 2 * n * sizeof(Point) + 2 * (n + 1) * sizeof(float));

    int tid = threadIdx.x;

    for (int i = tid; i < n; i += blockDim.x) {
        s_t1[i] = t1_raw[bid * n + i];
        s_t2[i] = t2_raw[bid * n + i];
    }

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

            float dx = s_t1[i - 1].x - s_t2[j - 1].x;
            float dy = s_t1[i - 1].y - s_t2[j - 1].y;
            float cost = __fsqrt_rn(dx * dx + dy * dy); 

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

        for (int i = tid; i <= n; i += blockDim.x) {
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

    size_t shared_mem_bytes = (n * 2 * sizeof(Point)) + ((n + 1) * 3 * sizeof(float));

    if (shared_mem_bytes > 100 * 1024) {
        std::cerr << "Error: n is too large (" << n << "). Shared memory limit exceeded!" << std::endl;
        return;
    }

    size_t traj_size = num_t * n * sizeof(Point);
    Point* d_t1_raw, * d_t2_raw;
    float* d_results;

    CHECK(cudaMalloc(&d_t1_raw, traj_size));
    CHECK(cudaMalloc(&d_t2_raw, traj_size));
    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));

    CHECK(cudaMemcpy(d_t1_raw, h_t1, traj_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2_raw, h_t2, traj_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaFuncSetAttribute(dtw_kernel_wavefront,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_mem_bytes);

    int threadsPerBlock = (n < 256) ? n : 256;
    int blocksPerGrid = num_t;

    dtw_kernel_wavefront << <blocksPerGrid, threadsPerBlock, shared_mem_bytes >> > (
        d_t1_raw, d_t2_raw, d_results, num_t, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    CHECK(cudaMemcpy(h_results, d_results, num_t * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaEventRecord(stop_all));
    CHECK(cudaEventSynchronize(stop_all));
    CHECK(cudaEventElapsedTime(&time_all, start_all, stop_all));

    cudaFree(d_t1_raw); cudaFree(d_t2_raw); cudaFree(d_results);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    std::cout << "\n计算时间占比: " << gpu_time / time_all;
    gpu_time = time_all;

}