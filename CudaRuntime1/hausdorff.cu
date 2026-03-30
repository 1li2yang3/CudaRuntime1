#include "hausdorff.cuh"
#include <cmath>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_fp16.h>
#include <iostream>


//__device__ float compute_directed_hausdorff(const float2* A, const float2* B, int lenA, int lenB) {
//    __shared__ float2 s_B[256];      // 点集B的共享内存，分块加载
//    __shared__ float s_global_max;   // 全局最大最短距离
//    __shared__ float s_warp_max[8];  // 每个warp的最大最短距离
//
//    if (threadIdx.x == 0) s_global_max = 0.0f;
//    __syncthreads();
//
//    // 遍历点集A，上限改为 lenA
//    for (int tA = 0; tA < lenA; tA += blockDim.x) {
//        int a_idx = tA + threadIdx.x;
//        bool valid_a = (a_idx < lenA);
//        float2 a;
//        if (valid_a) {
//            a = A[a_idx];
//        }
//
//        float local_min_d2 = 1e38f;
//        bool active = valid_a;
//
//        // 遍历点集B，上限改为 lenB
//        for (int tB = 0; tB < lenB; tB += blockDim.x) {
//            int b_idx = tB + threadIdx.x;
//            if (b_idx < lenB) {
//                s_B[threadIdx.x] = B[b_idx];
//            }
//            __syncthreads();
//
//            // 共享内存中有效点的数量受限于 lenB
//            int valid_j = (tB + blockDim.x < lenB) ? blockDim.x : (lenB - tB);
//
//            if (active) {
//                for (int j = 0; j < valid_j; j++) {
//                    float dx = a.x - s_B[j].x;
//                    float dy = a.y - s_B[j].y;
//                    float d2 = dx * dx + dy * dy;
//                    if (d2 < local_min_d2) {
//                        local_min_d2 = d2;
//                    }
//                }
//
//                // Early Exit 优化逻辑
//                if (local_min_d2 <= s_global_max) {
//                    active = false;
//                }
//            }
//            __syncthreads();
//        }
//
//        float my_val = active ? local_min_d2 : 0.0f;
//        unsigned int mask = 0xffffffff;
//
//        // warp内归约
//        for (int offset = 16; offset > 0; offset /= 2) {
//            my_val = fmaxf(my_val, __shfl_down_sync(mask, my_val, offset));
//        }
//
//        int lane = threadIdx.x % 32;
//        int wid = threadIdx.x / 32;
//        if (lane == 0) {
//            s_warp_max[wid] = my_val;
//        }
//        __syncthreads();
//
//        // 块内归约
//        if (threadIdx.x < 32) {
//            float final_val = (threadIdx.x < 8) ? s_warp_max[threadIdx.x] : 0.0f;
//            for (int offset = 16; offset > 0; offset /= 2) {
//                final_val = fmaxf(final_val, __shfl_down_sync(mask, final_val, offset));
//            }
//            if (threadIdx.x == 0) {
//                s_global_max = fmaxf(s_global_max, final_val);
//            }
//        }
//        __syncthreads();
//    }
//
//    return s_global_max;
//}
//
//__global__ void hausdorff_tiled_kernel(const Point* t1_batch, const Point* t2_batch, float* results, int num_t, int n, int m) {
//    int bid = blockIdx.x;
//    if (bid >= num_t) return;
//
//    // 依据各自的长度计算偏移量
//    const float2* T1 = reinterpret_cast<const float2*>(t1_batch + bid * n);
//    const float2* T2 = reinterpret_cast<const float2*>(t2_batch + bid * m);
//
//    // 双向计算：A->B传入(n, m)，B->A传入(m, n)
//    float max_d1 = compute_directed_hausdorff(T1, T2, n, m);
//    float max_d2 = compute_directed_hausdorff(T2, T1, m, n);
//
//    if (threadIdx.x == 0) {
//        results[bid] = sqrtf(fmaxf(max_d1, max_d2));
//    }
//}
//
//void launch_hausdorff_batch_gpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m, float& gpu_time) {
//    cudaEvent_t start_all, stop_all;
//    float time_all = 0.0f;
//    CHECK(cudaEventCreate(&start_all));
//    CHECK(cudaEventCreate(&stop_all));
//    CHECK(cudaEventRecord(start_all));
//
//    Point* d_t1 = nullptr;
//    Point* d_t2 = nullptr;
//    float* d_results = nullptr;
//
//    // 分别计算两条轨迹所需的总显存大小
//    size_t total_points_1 = (size_t)num_t * n;
//    size_t total_points_2 = (size_t)num_t * m;
//
//    CHECK(cudaMalloc(&d_t1, total_points_1 * sizeof(Point)));
//    CHECK(cudaMalloc(&d_t2, total_points_2 * sizeof(Point)));
//    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));
//
//    CHECK(cudaMemcpy(d_t1, h_t1, total_points_1 * sizeof(Point), cudaMemcpyHostToDevice));
//    CHECK(cudaMemcpy(d_t2, h_t2, total_points_2 * sizeof(Point), cudaMemcpyHostToDevice));
//
//    cudaEvent_t start, stop;
//    CHECK(cudaEventCreate(&start));
//    CHECK(cudaEventCreate(&stop));
//    CHECK(cudaEventRecord(start));
//
//    // 启动 Kernel 并传入 n 和 m，线程数依然固定 256
//    hausdorff_tiled_kernel << <num_t, 256 >> > (d_t1, d_t2, d_results, num_t, n, m);
//    CHECK(cudaGetLastError());
//
//    CHECK(cudaEventRecord(stop));
//    CHECK(cudaEventSynchronize(stop));
//    CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
//
//    CHECK(cudaMemcpy(h_results, d_results, num_t * sizeof(float), cudaMemcpyDeviceToHost));
//
//    cudaFree(d_t1);
//    cudaFree(d_t2);
//    cudaFree(d_results);
//    cudaEventDestroy(start);
//    cudaEventDestroy(stop);
//
//    CHECK(cudaEventRecord(stop_all));
//    CHECK(cudaEventSynchronize(stop_all));
//    CHECK(cudaEventElapsedTime(&time_all, start_all, stop_all));
//    cudaEventDestroy(start_all);
//    cudaEventDestroy(stop_all);
//
//    std::cout << "\n计算时间占比: " << gpu_time / time_all;
//    gpu_time = time_all;
//}




// 你的原有核心算法（保持不变，它在Block级别完全同步，可以在循环中被安全调用）
__device__ float compute_directed_hausdorff(const float2* A, const float2* B, int lenA, int lenB) {
    __shared__ float2 s_B[256];
    __shared__ float s_global_max;
    __shared__ float s_warp_max[8];

    if (threadIdx.x == 0) s_global_max = 0.0f;
    __syncthreads();

    for (int tA = 0; tA < lenA; tA += blockDim.x) {
        int a_idx = tA + threadIdx.x;
        bool valid_a = (a_idx < lenA);
        float2 a;
        if (valid_a) {
            a = A[a_idx];
        }

        float local_min_d2 = 1e38f;
        bool active = valid_a;

        for (int tB = 0; tB < lenB; tB += blockDim.x) {
            int b_idx = tB + threadIdx.x;
            if (b_idx < lenB) {
                s_B[threadIdx.x] = B[b_idx];
            }
            __syncthreads();

            int valid_j = (tB + blockDim.x < lenB) ? blockDim.x : (lenB - tB);

            if (active) {
                for (int j = 0; j < valid_j; j++) {
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

        for (int offset = 16; offset > 0; offset /= 2) {
            my_val = fmaxf(my_val, __shfl_down_sync(mask, my_val, offset));
        }

        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        if (lane == 0) {
            s_warp_max[wid] = my_val;
        }
        __syncthreads();

        if (threadIdx.x < 32) {
            float final_val = (threadIdx.x < 8) ? s_warp_max[threadIdx.x] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                final_val = fmaxf(final_val, __shfl_down_sync(mask, final_val, offset));
            }
            if (threadIdx.x == 0) {
                s_global_max = fmaxf(s_global_max, final_val);
            }
        }
        __syncthreads();
    }

    return s_global_max;
}

// 新的 Kernel：每个 Block 处理一条 T1 轨迹，并遍历其对应的 top_k 个 T2 候选者
__global__ void hausdorff_tiled_rtree_kernel(
    const Point* t1_batch, int num_t1, int n,
    const Point* t2_batch, int num_t2, int m,
    const int* candidate_indices, int top_k,
    float* results)
{
    int bid = blockIdx.x;
    if (bid >= num_t1) return;

    // T1 固定
    const float2* T1 = reinterpret_cast<const float2*>(t1_batch + bid * n);

    float best_hausdorff_d2 = 1e38f; // 记录 top_k 个候选者中的最小距离平方

    // 遍历该 T1 轨迹对应的 10 个候选 T2 轨迹的索引
    for (int k = 0; k < top_k; k++) {
        // 从 candidate_indices 中获取目标轨迹在 t2_batch 中的真实索引
        int t2_idx = candidate_indices[bid * top_k + k];
        const float2* T2 = reinterpret_cast<const float2*>(t2_batch + t2_idx * m);

        float max_d1 = compute_directed_hausdorff(T1, T2, n, m);
        float max_d2 = compute_directed_hausdorff(T2, T1, m, n);

        float current_max_d2 = fmaxf(max_d1, max_d2);

        // 只有 thread 0 需要更新最小值（虽然 __syncthreads() 保证了同步，但最好由一个线程写寄存器以防分支差异）
        if (threadIdx.x == 0) {
            if (current_max_d2 < best_hausdorff_d2) {
                best_hausdorff_d2 = current_max_d2;
            }
        }
        // 同步，确保所有线程等待 thread 0 更新完毕后再进行下一轮候选者的计算
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        results[bid] = sqrtf(best_hausdorff_d2);
    }
}

// CUDA 主机端启动函数
void launch_hausdorff_rtree_gpu(const Point* h_t1, int num_t1, int n,
    const Point* h_t2, int num_t2, int m,
    const int* h_candidate_indices, int top_k,
    float* h_results, float& gpu_time)
{
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    Point* d_t1 = nullptr;
    Point* d_t2 = nullptr;
    int* d_candidate_indices = nullptr;
    float* d_results = nullptr;

    size_t total_points_1 = (size_t)num_t1 * n;
    size_t total_points_2 = (size_t)num_t2 * m;
    size_t total_candidates = (size_t)num_t1 * top_k;

    CHECK(cudaMalloc(&d_t1, total_points_1 * sizeof(Point)));
    CHECK(cudaMalloc(&d_t2, total_points_2 * sizeof(Point)));
    CHECK(cudaMalloc(&d_candidate_indices, total_candidates * sizeof(int)));
    CHECK(cudaMalloc(&d_results, num_t1 * sizeof(float)));

    // 注意：这里需要把整个 T2 集合都传进显存，因为候选索引可能指向任意一条 T2
    CHECK(cudaMemcpy(d_t1, h_t1, total_points_1 * sizeof(Point), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2, h_t2, total_points_2 * sizeof(Point), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_candidate_indices, h_candidate_indices, total_candidates * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));

    // 启动 Kernel：Grid 大小等于 T1 轨迹的数量
    hausdorff_tiled_rtree_kernel << <num_t1, 256 >> > (d_t1, num_t1, n, d_t2, num_t2, m, d_candidate_indices, top_k, d_results);
    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    CHECK(cudaMemcpy(h_results, d_results, num_t1 * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_candidate_indices);
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