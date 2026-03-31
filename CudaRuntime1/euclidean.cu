#include "euclidean.cuh"
#include <cmath>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> 
#include <iostream>
#include <cfloat> // 引入 FLT_MAX

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {//当前线程向自己线程号加上偏移量的线程获取数据
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


__global__ void euclidean_rtree_csr_kernel_shared(const float2* t1_batch, const float2* t2_batch,
    const int* candidates, const int* offsets,
    float* results, int num_t, int n)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= num_t) return;

    extern __shared__ float2 s_t1[];

    int offset1 = bid * n;

    for (int i = tid; i < n; i += blockDim.x) {
        s_t1[i] = t1_batch[offset1 + i];
    }

    __syncthreads();

    float min_dist_sq = FLT_MAX;
    static __shared__ float shared_warp_sums[32];

    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    int start_idx = offsets[bid];
    int end_idx = offsets[bid + 1];


    for (int idx = start_idx; idx < end_idx; ++idx) {
        float local_sum = 0.0f;
        int t2_idx = candidates[idx];
        int offset2 = t2_idx * n;

        for (int i = tid; i < n; i += blockDim.x) {

            float2 p1 = s_t1[i];

            float2 p2 = t2_batch[offset2 + i];

            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;
            local_sum += dx * dx + dy * dy;
        }

        float warp_sum = warpReduceSum(local_sum);
        if (lane_id == 0) shared_warp_sums[warp_id] = warp_sum;
        __syncthreads();

        if (warp_id == 0) {
            float final_sum = (lane_id < num_warps) ? shared_warp_sums[lane_id] : 0.0f;
            final_sum = warpReduceSum(final_sum);

            if (tid == 0) {
                if (final_sum < min_dist_sq) {
                    min_dist_sq = final_sum;
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        results[bid] = sqrtf(min_dist_sq);
    }
}

void execute_rtree_csr_kernel_on_gpu(const Point* h_t1, const Point* h_t2,
    const int* h_candidates, const int* h_offsets,
    float* h_results, int num_t, int n, int total_candidates, float& gpu_time)
{
    size_t total_points = (size_t)num_t * n;
    Point* d_t1, * d_t2;
    float* d_results;
    int* d_candidates, * d_offsets;

    cudaMalloc(&d_t1, total_points * sizeof(Point));
    cudaMalloc(&d_t2, total_points * sizeof(Point));
    cudaMalloc(&d_results, num_t * sizeof(float));
    cudaMalloc(&d_candidates, total_candidates * sizeof(int));
    cudaMalloc(&d_offsets, (num_t + 1) * sizeof(int));

    cudaMemcpy(d_t1, h_t1, total_points * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t2, h_t2, total_points * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidates, h_candidates, total_candidates * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, (num_t + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    size_t shared_mem_bytes = n * sizeof(float2);

    euclidean_rtree_csr_kernel_shared << <num_t, 256, shared_mem_bytes >> > ((const float2*)d_t1, (const float2*)d_t2,
        d_candidates, d_offsets, d_results, num_t, n);
    cudaGetLastError();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(h_results, d_results, num_t * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_t1); cudaFree(d_t2);
    cudaFree(d_results); cudaFree(d_candidates); cudaFree(d_offsets);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}