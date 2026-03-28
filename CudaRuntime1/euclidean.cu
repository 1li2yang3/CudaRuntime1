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

__global__ void euclidean_kernel(const float2* t1_batch, const float2* t2_batch, float* results, int num_t, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= num_t) return;

    float local_sum = 0.0f;
    int offset = bid * n;

    for (int i = tid; i < n; i += blockDim.x) {//每次256个线程同时计算
        float2 p1 = t1_batch[offset + i];
        float2 p2 = t2_batch[offset + i];

        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        local_sum += dx * dx + dy * dy;
    }

    float warp_sum = warpReduceSum(local_sum);

    static __shared__ float shared_warp_sums[32];//共享内存
	int warp_id = tid / warpSize;//当前线程所在warp的id
	int lane_id = tid % warpSize;//当前线程在warp中的id，warpReduceSum后0号线程保存了warp的和

	if (lane_id == 0) {//每个warp的0号线程将warp的和保存到共享内存中
        shared_warp_sums[warp_id] = warp_sum;
    }
	__syncthreads();//同步所有线程

	int num_warps = (blockDim.x + warpSize - 1) / warpSize;//当前block中warp数
	if (warp_id == 0) {//0号warp负责将所有warp的和加起来
		float final_sum = (lane_id < num_warps) ? shared_warp_sums[lane_id] : 0.0f;//每个线程取一个warp的和，之后的线程取0
        final_sum = warpReduceSum(final_sum);

		if (tid == 0) {//0号线程存结果
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

    cudaFree(d_t1_f32);
    cudaFree(d_t2_f32);
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK(cudaEventRecord(stop_all));
    CHECK(cudaEventSynchronize(stop_all));
    CHECK(cudaEventElapsedTime(&time_all, start_all, stop_all));
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);
    std::cout << "计算时间占比: " << gpu_time / time_all;
    gpu_time = time_all;
}


// 新的Kernel: 计算T1中每条轨迹对T2中所有轨迹的最小距离
__global__ void euclidean_min_kernel(const float2* t1_batch, const float2* t2_batch, float* results, int num_t, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x; // 当前Block对应T1中的第 bid 条轨迹

    if (bid >= num_t) return;

    int offset1 = bid * n; // T1轨迹的起始偏移量
    float min_dist = FLT_MAX; // 初始化最小距离为浮点数最大值

    static __shared__ float shared_warp_sums[32]; // 共享内存

    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    // 遍历 T2 中的所有轨迹
    for (int j = 0; j < num_t; ++j) {
        float local_sum = 0.0f;
        int offset2 = j * n; // T2当前遍历轨迹的起始偏移量

        // 每个线程计算一部分点对的距离平方和
        for (int i = tid; i < n; i += blockDim.x) {
            float2 p1 = t1_batch[offset1 + i];
            float2 p2 = t2_batch[offset2 + i];

            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;
            local_sum += dx * dx + dy * dy;
        }

        // Warp内规约
        float warp_sum = warpReduceSum(local_sum);

        if (lane_id == 0) {
            shared_warp_sums[warp_id] = warp_sum;
        }
        __syncthreads(); // 同步，确保所有Warp的和都已写入共享内存

        // Block内规约（由0号Warp完成）
        if (warp_id == 0) {
            float final_sum = (lane_id < num_warps) ? shared_warp_sums[lane_id] : 0.0f;
            final_sum = warpReduceSum(final_sum);

            if (tid == 0) {
                float current_dist = sqrtf(final_sum);
                // 更新最小距离
                if (current_dist < min_dist) {
                    min_dist = current_dist;
                }
            }
        }

        // 核心：在进入下一轮 T2 轨迹循环前，必须全局同步！
        // 否则下一个循环中 lane_id==0 的线程可能会覆盖 shared_warp_sums，
        // 导致 0号Warp 还在读取上一轮数据时发生数据竞争。
        __syncthreads();
    }

    // 将T1中第bid条轨迹对应的最小距离写入结果数组
    if (tid == 0) {
        results[bid] = min_dist;
    }
}

// 接口保持不变
void launch_euclidean_batch_gpu_2(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time) {
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

    // 调用新的 Kernel 函数
    euclidean_min_kernel << <num_t, 256 >> > ((const float2*)d_t1_f32, (const float2*)d_t2_f32, d_results, num_t, n);
    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    CHECK(cudaMemcpy(h_results, d_results, num_t * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_t1_f32);
    cudaFree(d_t2_f32);
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK(cudaEventRecord(stop_all));
    CHECK(cudaEventSynchronize(stop_all));
    CHECK(cudaEventElapsedTime(&time_all, start_all, stop_all));
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);
    std::cout << "计算时间占比: " << gpu_time / time_all;
    gpu_time = time_all;
}