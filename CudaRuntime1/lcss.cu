#include "lcss.cuh"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>


struct Envelope {
    float min_x, max_x, min_y, max_y;
};

// 严格上界过滤 Kernel (计算落在 epsilon 膨胀包络盒内的最大可能匹配点数)
__global__ void lcss_strict_bound_filter_kernel(const Envelope* envs_t1, const Point* t2_raw,
    float* ub_matrix, int num_t, int n, int m, float epsilon) {
    int t1_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t2_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (t1_idx >= num_t || t2_idx >= num_t) return;

    int env_offset = t1_idx * n;
    int t2_offset = t2_idx * m;

    int possible_matches = 0;

    for (int i = 0; i < m; i++) {
        int env_idx = i * n / m;
        Point p2 = t2_raw[t2_offset + i];
        Envelope env = envs_t1[env_offset + env_idx];


        // 注意原代码中匹配条件是 abs(dx) < epsilon，所以这里是开区间
        if (p2.x > env.min_x - epsilon && p2.x < env.max_x + epsilon &&
            p2.y > env.min_y - epsilon && p2.y < env.max_y + epsilon) {
            possible_matches++; // 如果在区域内，代表它"有可能"成为最终LCSS的一个匹配点
        }
    }

    // 转换为与精算结果同量纲的相似度比例上界
    int min_len = min(n, m);
    ub_matrix[t1_idx * num_t + t2_idx] = (float)possible_matches / min_len;
}


__global__ void lcss_refine_kernel_m_1(const Point* __restrict__ t1_raw, const Point* __restrict__ t2_raw,
    const int2* pair_list, float* __restrict__ results,
    int num_pairs, int n, int m, float epsilon) {
    int bid = blockIdx.x;
    if (bid >= num_pairs) return;

    // 获取需要精确计算的真实轨迹索引
    int t1_idx = pair_list[bid].x;
    int t2_idx = pair_list[bid].y;

    // 引入动态共享内存，大小为 3 * n * sizeof(int)
    extern __shared__ int s_diagonals_flat[];

    int tid = threadIdx.x;

    // 独立计算两条轨迹的偏移量
    int offset1 = t1_idx * n;
    int offset2 = t2_idx * m;

    for (int k = 2; k <= n + m; k++) {
        int curr_buf = k % 3;
        int prev1_buf = (k - 1) % 3;
        int prev2_buf = (k - 2) % 3;

        int start_i = min(n, k - 1);
        int end_i = max(1, k - m);
        int num_elements = start_i - end_i + 1;

        for (int idx = tid; idx < num_elements; idx += blockDim.x) {
            int i = start_i - idx;
            int j = k - i;

            float dx = fabsf(t1_raw[offset1 + i - 1].x - t2_raw[offset2 + j - 1].x);
            float dy = fabsf(t1_raw[offset1 + i - 1].y - t2_raw[offset2 + j - 1].y);
            bool match = (dx < epsilon) && (dy < epsilon);

            int val = 0;
            if (match) {
                int prev_diag_val = 0;
                if (i > 1 && j > 1) {
                    int start_i_k2 = min(n, k - 3);
                    prev_diag_val = s_diagonals_flat[prev2_buf * n + (start_i_k2 - (i - 1))];
                }
                val = prev_diag_val + 1;
            }
            else {
                int left_val = 0;
                int up_val = 0;
                int start_i_k1 = min(n, k - 2);

                if (j > 1) left_val = s_diagonals_flat[prev1_buf * n + (start_i_k1 - i)];
                if (i > 1) up_val = s_diagonals_flat[prev1_buf * n + (start_i_k1 - (i - 1))];

                val = max(left_val, up_val);
            }

            s_diagonals_flat[curr_buf * n + idx] = val;
        }
        __syncthreads();
    }

    if (tid == 0) {
        int last_buf = (n + m) % 3;
        int min_len = min(n, m);
        results[bid] = (float)s_diagonals_flat[last_buf * n + 0] / min_len;
    }
}


void launch_lcss_batch_gpu_wavefront_knn(const Point* h_t1, const Point* h_t2, float* h_results,
    int num_t, int n, int m, float epsilon, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    // --- 阶段 A: CPU 端快速生成包络线 (Window r=5) ---
    int r = 5;
    std::vector<Envelope> h_envs(num_t * n);
#pragma omp parallel for
    for (int t = 0; t < num_t; t++) {
        for (int i = 0; i < n; i++) {
            float min_x = 1e20f, max_x = -1e20f, min_y = 1e20f, max_y = -1e20f;
            int start_idx = std::max(0, i - r);
            int end_idx = std::min(n - 1, i + r);
            for (int k = start_idx; k <= end_idx; k++) {
                Point p = h_t1[t * n + k];
                min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
                min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
            }
            h_envs[t * n + i] = { min_x, max_x, min_y, max_y };
        }
    }

    size_t pts_size1 = (size_t)num_t * n * sizeof(Point);
    size_t pts_size2 = (size_t)num_t * m * sizeof(Point);

    Point* d_t1_raw, * d_t2_raw;
    Envelope* d_envs;
    float* d_ub_matrix; // 改名：upper bound matrix

    CHECK(cudaMalloc(&d_t1_raw, pts_size1));
    CHECK(cudaMalloc(&d_t2_raw, pts_size2));
    CHECK(cudaMalloc(&d_envs, num_t * n * sizeof(Envelope)));
    CHECK(cudaMalloc(&d_ub_matrix, num_t * num_t * sizeof(float)));

    CHECK(cudaMemcpy(d_t1_raw, h_t1, pts_size1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2_raw, h_t2, pts_size2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_envs, h_envs.data(), num_t * n * sizeof(Envelope), cudaMemcpyHostToDevice));

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu); cudaEventCreate(&stop_gpu);
    float time_lb = 0.0f, time_lcss = 0.0f;

    // --- 阶段 B: 运行过滤 Kernel (使用严格上界方法，需传入 epsilon) ---
    cudaEventRecord(start_gpu);
    dim3 block_lb(16, 16);
    dim3 grid_lb((num_t + block_lb.x - 1) / block_lb.x, (num_t + block_lb.y - 1) / block_lb.y);
    lcss_strict_bound_filter_kernel << <grid_lb, block_lb >> > (d_envs, d_t2_raw, d_ub_matrix, num_t, n, m, epsilon);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time_lb, start_gpu, stop_gpu);

    std::vector<float> h_ub_matrix(num_t * num_t);
    CHECK(cudaMemcpy(h_ub_matrix.data(), d_ub_matrix, num_t * num_t * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_envs);
    cudaFree(d_ub_matrix);

    // --- 阶段 C: 提取上界相似度最大的 Top-10 候选对 ---
    int K = std::min(10, num_t);
    int num_pairs = num_t * K;
    std::vector<int2> h_pair_list(num_pairs);

#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        std::vector<std::pair<float, int>> row(num_t);
        for (int j = 0; j < num_t; j++) {
            row[j] = { h_ub_matrix[i * num_t + j], j };
        }

        // 【核心修改】：LCSS上界代表可能的最大相似度，越大越有潜力。所以用降序排序
        std::partial_sort(row.begin(), row.begin() + K, row.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

        for (int k = 0; k < K; k++) {
            h_pair_list[i * K + k] = make_int2(i, row[k].second);
        }
    }

    // --- 阶段 D: 运行精确 LCSS Kernel 进行 Refine ---
    int2* d_pair_list;
    float* d_lcss_results;
    CHECK(cudaMalloc(&d_pair_list, num_pairs * sizeof(int2)));
    CHECK(cudaMalloc(&d_lcss_results, num_pairs * sizeof(float)));
    CHECK(cudaMemcpy(d_pair_list, h_pair_list.data(), num_pairs * sizeof(int2), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    size_t shared_mem_size = 3 * n * sizeof(int);
    cudaFuncSetAttribute(lcss_refine_kernel_m_1, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

    cudaEventRecord(start_gpu);
    lcss_refine_kernel_m_1 << <num_pairs, threadsPerBlock, shared_mem_size >> > (d_t1_raw, d_t2_raw, d_pair_list, d_lcss_results, num_pairs, n, m, epsilon);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time_lcss, start_gpu, stop_gpu);

    gpu_time = time_lb + time_lcss;

    // --- 阶段 E: 写回最终结果 (LCSS是越大越好) ---
    std::vector<float> h_lcss_results(num_pairs);
    CHECK(cudaMemcpy(h_lcss_results.data(), d_lcss_results, num_pairs * sizeof(float), cudaMemcpyDeviceToHost));

#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        float max_lcss = -1.0f;

        for (int k = 0; k < K; k++) {
            float similarity = h_lcss_results[i * K + k];
            if (similarity > max_lcss) {
                max_lcss = similarity;
            }
        }
        h_results[i] = max_lcss;
    }

    // 清理资源
    cudaFree(d_t1_raw);
    cudaFree(d_t2_raw);
    cudaFree(d_pair_list);
    cudaFree(d_lcss_results);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    CHECK(cudaEventRecord(stop_all));
    CHECK(cudaEventSynchronize(stop_all));
    CHECK(cudaEventElapsedTime(&time_all, start_all, stop_all));
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);

    std::cout << "\n计算时间占比 " << gpu_time / time_all;
    gpu_time = time_all;
}