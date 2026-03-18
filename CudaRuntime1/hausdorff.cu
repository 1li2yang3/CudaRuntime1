#include "hausdorff.cuh"
#include <cmath>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_fp16.h>
#include <iostream>

__device__ float compute_directed_hausdorff(const float2* A, const float2* B, int n) {
	__shared__ float2 s_B[256];//点集B的共享内存
    __shared__ float s_global_max; //全局最大最短距离
	__shared__ float s_warp_max[8]; //每个warp的最大最短距离

	if (threadIdx.x == 0) s_global_max = 0.0f;//0号线程初始化最大距离
	__syncthreads();//同步，防止其他线程访问未初始化的s_global_max

	for (int tA = 0; tA < n; tA += blockDim.x) {//遍历点集A，分块加载，每次处理256个点
		int a_idx = tA + threadIdx.x;//偏移量+线程id=当前线程处理的点索引
		float2 a;//当前线程处理的点
        bool valid_a = (a_idx < n);
        if (valid_a) {
            a = A[a_idx];
        }

        float local_min_d2 = 1e38f;
        bool active = valid_a; 

        for (int tB = 0; tB < n; tB += blockDim.x) {
			int b_idx = tB + threadIdx.x;//与a_idx类似，计算当前线程处理的点索引
            if (b_idx < n) {
				s_B[threadIdx.x] = B[b_idx];//将点集B的当前块加载到共享内存
            }
			__syncthreads();//每次加载共享数据后都要同步

			int valid_j = (tB + blockDim.x < n) ? blockDim.x : (n - tB);//共享内存中有效点的数量，最后一个块可能不足256个点

            if (active) {
				for (int j = 0; j < valid_j; j++) {//计算当前点a与共享内存中每个点的距离，更新最短距离
                    float dx = a.x - s_B[j].x;
                    float dy = a.y - s_B[j].y;
                    float d2 = dx * dx + dy * dy;
                    if (d2 < local_min_d2) {
                        local_min_d2 = d2;
                    }
                }

                if (local_min_d2 <= s_global_max) {
                    active = false;
                }
            }
            __syncthreads();
        }

        float my_val = active ? local_min_d2 : 0.0f;
        unsigned int mask = 0xffffffff;

		for (int offset = 16; offset > 0; offset /= 2) {//warp内归约，计算每个warp内线程的最大最短距离
            my_val = fmaxf(my_val, __shfl_down_sync(mask, my_val, offset));
        }

        int lane = threadIdx.x % 32;//当前线程在warp中的id
        int wid = threadIdx.x / 32;//当前线程所在warp的id
		if (lane == 0) {//每个warp的第一个线程将该warp的最大最短距离写入共享内存
            s_warp_max[wid] = my_val;
        }
        __syncthreads();

		if (threadIdx.x < 32) {//前32个线程负责归约所有warp的结果，计算全局最大最短距离
			float final_val = (threadIdx.x < 8) ? s_warp_max[threadIdx.x] : 0.0f;//256个线程有8个warp，前8个线程分别处理一个warp的结果
			for (int offset = 16; offset > 0; offset /= 2) {//归约所有warp的结果，计算全局最大最短距离
                final_val = fmaxf(final_val, __shfl_down_sync(mask, final_val, offset));
            }
			if (threadIdx.x == 0) {//0号线程更新全局最大最短距离
                s_global_max = fmaxf(s_global_max, final_val);
            }
        }
        __syncthreads();
    }

    return s_global_max;
}

__global__ void hausdorff_tiled_kernel(const Point* t1_batch, const Point* t2_batch, float* results, int num_t, int n) {
    int bid = blockIdx.x;
    if (bid >= num_t) return;

    const float2* T1 = reinterpret_cast<const float2*>(t1_batch + bid * n);
    const float2* T2 = reinterpret_cast<const float2*>(t2_batch + bid * n);

    float max_d1 = compute_directed_hausdorff(T1, T2, n);
    float max_d2 = compute_directed_hausdorff(T2, T1, n);

    if (threadIdx.x == 0) {
        results[bid] = sqrtf(fmaxf(max_d1, max_d2));
    }
}

void launch_hausdorff_batch_gpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    Point* d_t1 = nullptr;
    Point* d_t2 = nullptr;
    float* d_results = nullptr;
    size_t total_points = (size_t)num_t * n;

    CHECK(cudaMalloc(&d_t1, total_points * sizeof(Point)));
    CHECK(cudaMalloc(&d_t2, total_points * sizeof(Point)));
    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));

    CHECK(cudaMemcpy(d_t1, h_t1, total_points * sizeof(Point), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2, h_t2, total_points * sizeof(Point), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));

    hausdorff_tiled_kernel << <num_t, 256 >> > (d_t1, d_t2, d_results, num_t, n);
    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    CHECK(cudaMemcpy(h_results, d_results, num_t * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_results);
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