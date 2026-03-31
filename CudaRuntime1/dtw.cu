#include "dtw.cuh"
#include <cmath>
#include <algorithm>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <vector>
#include <omp.h>
#include <cuda_awbarrier.h>
#include <iostream>


struct Envelope {
    float min_x, max_x, min_y, max_y;
};

// 极速 LB_Keogh 过滤 Kernel
__global__ void lb_keogh_filter_kernel(const Envelope* envs_t1, const Point* t2_raw,
    float* lb_matrix, int num_t, int n, int m) {
    int t1_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t2_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (t1_idx >= num_t || t2_idx >= num_t) return;

    int env_offset = t1_idx * n;
    int t2_offset = t2_idx * m;
    float lb_dist = 0.0f;

    for (int i = 0; i < m; i++) {
        int env_idx = i * n / m;
        Point p2 = t2_raw[t2_offset + i];
        Envelope env = envs_t1[env_offset + env_idx];

        float dx = fmaxf(0.0f, fmaxf(env.min_x - p2.x, p2.x - env.max_x));
        float dy = fmaxf(0.0f, fmaxf(env.min_y - p2.y, p2.y - env.max_y));
        lb_dist += sqrtf(dx * dx + dy * dy);
    }
    lb_matrix[t1_idx * num_t + t2_idx] = lb_dist;
}

// 精确 DTW Kernel (利用动态共享内存处理 Pair)
__global__ void dtw_refine_kernel(const Point* t1_raw, const Point* t2_raw,
    const int2* pair_list, float* results,
    int num_pairs, int n, int m) {
    int bid = blockIdx.x;
    if (bid >= num_pairs) return;

    int tid = threadIdx.x;
    int t1_idx = pair_list[bid].x;
    int t2_idx = pair_list[bid].y;
    int offset1 = t1_idx * n;
    int offset2 = t2_idx * m;

    extern __shared__ float s_dp[];
    float* d0 = s_dp;
    float* d1 = s_dp + (n + 1);
    float* d2 = s_dp + 2 * (n + 1);

    for (int i = tid; i <= n; i += blockDim.x) {
        d0[i] = 1e20f; d1[i] = 1e20f; d2[i] = 1e20f;
    }
    __syncthreads();

    if (tid == 0) d0[0] = 0.0f;
    __syncthreads();

    for (int k = 2; k <= n + m; k++) {
        int i_start = max(1, k - m);
        int i_end = min(n, k - 1);

        for (int i = i_start + tid; i <= i_end; i += blockDim.x) {
            int j = k - i;
            Point p1 = t1_raw[offset1 + (i - 1)];
            Point p2 = t2_raw[offset2 + (j - 1)];

            float dx = p1.x - p2.x; float dy = p1.y - p2.y;
            float cost = sqrtf(dx * dx + dy * dy);

            float diag = d0[i - 1];
            float up = d1[i - 1];
            float left = d1[i];

            d2[i] = cost + fminf(diag, fminf(up, left));
        }
        __syncthreads();

        if (k == 2 && tid == 0) d0[0] = 1e20f;
        float* temp = d0; d0 = d1; d1 = d2; d2 = temp;

        for (int i = i_start + tid; i <= i_end; i += blockDim.x) {
            d2[i] = 1e20f;
        }
        __syncthreads();
    }

    if (tid == 0) results[bid] = d1[n];
}


void launch_dtw_batch_gpu_knn(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));


    int r = 5;
    std::vector<Envelope> h_envs(num_t * n);
#pragma omp parallel for
    for (int t = 0; t < num_t; t++) {
        for (int i = 0; i < n; i++) {
            float min_x = 1e20f, max_x = -1e20f, min_y = 1e20f, max_y = -1e20f;
            int start = std::max(0, i - r);
            int end = std::min(n - 1, i + r);
            for (int k = start; k <= end; k++) {
                Point p = h_t1[t * n + k];
                min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
                min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
            }
            h_envs[t * n + i] = { min_x, max_x, min_y, max_y };
        }
    }

    // 分配设备内存
    size_t traj_size1 = (size_t)num_t * n * sizeof(Point);
    size_t traj_size2 = (size_t)num_t * m * sizeof(Point);
    Point* d_t1, * d_t2;
    Envelope* d_envs;
    float* d_lb_matrix;

    CHECK(cudaMalloc(&d_t1, traj_size1));
    CHECK(cudaMalloc(&d_t2, traj_size2));
    CHECK(cudaMalloc(&d_envs, num_t * n * sizeof(Envelope)));
    CHECK(cudaMalloc(&d_lb_matrix, num_t * num_t * sizeof(float)));

    CHECK(cudaMemcpy(d_t1, h_t1, traj_size1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2, h_t2, traj_size2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_envs, h_envs.data(), num_t * n * sizeof(Envelope), cudaMemcpyHostToDevice));

    // 记录 GPU 纯计算时间 (LB + DTW)
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu); cudaEventCreate(&stop_gpu);
    float time_lb = 0.0f, time_dtw = 0.0f;

    // --- 阶段 B: 运行 LB_Keogh 粗筛 Kernel ---
    cudaEventRecord(start_gpu);
    dim3 block_lb(16, 16);
    dim3 grid_lb((num_t + block_lb.x - 1) / block_lb.x, (num_t + block_lb.y - 1) / block_lb.y);
    lb_keogh_filter_kernel << <grid_lb, block_lb >> > (d_envs, d_t2, d_lb_matrix, num_t, n, m);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time_lb, start_gpu, stop_gpu);

    // 拷回距离矩阵
    std::vector<float> h_lb_matrix(num_t * num_t);
    CHECK(cudaMemcpy(h_lb_matrix.data(), d_lb_matrix, num_t * num_t * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_envs);
    cudaFree(d_lb_matrix);

    // --- 阶段 C: CPU 端提取前 10 名候选对 ---
    int K = std::min(10, num_t);
    int num_pairs = num_t * K;
    std::vector<int2> h_pair_list(num_pairs);

#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        // pair <LB距离, t2索引>
        std::vector<std::pair<float, int>> row(num_t);
        for (int j = 0; j < num_t; j++) {
            row[j] = { h_lb_matrix[i * num_t + j], j };
        }
        std::partial_sort(row.begin(), row.begin() + K, row.end());
        for (int k = 0; k < K; k++) {
            h_pair_list[i * K + k] = make_int2(i, row[k].second);
        }
    }

    // --- 阶段 D: 运行精确 DTW Kernel 进行 Refine ---
    int2* d_pair_list;
    float* d_dtw_results;
    CHECK(cudaMalloc(&d_pair_list, num_pairs * sizeof(int2)));
    CHECK(cudaMalloc(&d_dtw_results, num_pairs * sizeof(float)));
    CHECK(cudaMemcpy(d_pair_list, h_pair_list.data(), num_pairs * sizeof(int2), cudaMemcpyHostToDevice));

    int max_diag_len = std::min(n, m);
    int threadsPerBlock = (max_diag_len < 256) ? ((max_diag_len / 32 + 1) * 32) : 256;
    size_t shared_mem_size = 3 * (n + 1) * sizeof(float);
    cudaFuncSetAttribute(dtw_refine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

    cudaEventRecord(start_gpu);
    // Grid大小刚好是提取出的对数 (num_t * 10)
    dtw_refine_kernel << <num_pairs, threadsPerBlock, shared_mem_size >> > (d_t1, d_t2, d_pair_list, d_dtw_results, num_pairs, n, m);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time_dtw, start_gpu, stop_gpu);

    // 累加 GPU 纯内核计算时间赋给传出参数
    gpu_time = time_lb + time_dtw;

    // --- 阶段 E: 写回最终的唯一结果 ---
    std::vector<float> h_dtw_results(num_pairs);
    CHECK(cudaMemcpy(h_dtw_results.data(), d_dtw_results, num_pairs * sizeof(float), cudaMemcpyDeviceToHost));

#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        float min_dtw = 1e20f;
        // int best_index = -1;  // <--- 如果以后需要索引用这个记录

        // 遍历当前 t1 的 10 个精确计算结果，找出最小值
        for (int k = 0; k < K; k++) {
            float dist = h_dtw_results[i * K + k];
            if (dist < min_dtw) {
                min_dtw = dist;
                // best_index = h_pair_list[i * K + k].y;
            }
        }
        // 按你的原接口，写入单个最相似的距离
        h_results[i] = min_dtw;
    }

    // --- 清理与统计 ---
    cudaFree(d_t1); cudaFree(d_t2);
    cudaFree(d_pair_list); cudaFree(d_dtw_results);
    cudaEventDestroy(start_gpu); cudaEventDestroy(stop_gpu);

    CHECK(cudaEventRecord(stop_all));
    CHECK(cudaEventSynchronize(stop_all));
    CHECK(cudaEventElapsedTime(&time_all, start_all, stop_all));
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);

    std::cout << "\n计算时间占比" << gpu_time / time_all ;
	gpu_time = time_all;

}