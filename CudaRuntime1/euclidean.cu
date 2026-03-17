#include "euclidean.cuh"
#include <cmath>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <iostream>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void euclidean_kernel(const float2* t1_batch, const float2* t2_batch, float* results, int num_t, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= num_t) return;

    float local_sum = 0.0f;
    int offset = bid * n;

    for (int i = tid; i < n; i += blockDim.x) {
        float2 p1 = t1_batch[offset + i];
        float2 p2 = t2_batch[offset + i];

        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        local_sum += dx * dx + dy * dy;
    }

    float warp_sum = warpReduceSum(local_sum);

    static __shared__ float shared_warp_sums[32];
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    if (lane_id == 0) {
        shared_warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0) {
        float final_sum = (lane_id < num_warps) ? shared_warp_sums[lane_id] : 0.0f;
        final_sum = warpReduceSum(final_sum);

        if (tid == 0) {
            results[bid] = sqrtf(final_sum);
        }
    }
}

void launch_euclidean_batch_gpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    size_t total_points = (size_t)num_t * n;
    Point* d_t1_f32 = nullptr;
    Point* d_t2_f32 = nullptr;
    float* d_results = nullptr;

    CHECK(cudaMalloc(&d_t1_f32, total_points * sizeof(Point)));
    CHECK(cudaMalloc(&d_t2_f32, total_points * sizeof(Point)));
    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));

    CHECK(cudaMemcpy(d_t1_f32, h_t1, total_points * sizeof(Point), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2_f32, h_t2, total_points * sizeof(Point), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));

    euclidean_kernel << <num_t, 256 >> > ((const float2*)d_t1_f32, (const float2*)d_t2_f32, d_results, num_t, n);
    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    CHECK(cudaMemcpy(h_results, d_results, num_t * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaEventRecord(stop_all));
    CHECK(cudaEventSynchronize(stop_all));
    CHECK(cudaEventElapsedTime(&time_all, start_all, stop_all));

    cudaFree(d_t1_f32);
    cudaFree(d_t2_f32);
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "计算时间占比: " << gpu_time / time_all;

    gpu_time = time_all;
}