#include "frechet.cuh"
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <omp.h>


struct FrechetEnvelope {
    float min_x, max_x, min_y, max_y;
};

// 严格下界过滤 Kernel (计算包络线物理距离的最大值)
__global__ void frechet_lb_filter_kernel(const FrechetEnvelope* envs_t1, const Point* t2_raw,
    float* lb_matrix, int num_t, int n, int m) {
    int t1_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t2_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (t1_idx >= num_t || t2_idx >= num_t) return;

    int env_offset = t1_idx * n;
    int t2_offset = t2_idx * m;

    // Fréchet是瓶颈度量，初始下界为0
    float lb_dist = 0.0f;

    for (int i = 0; i < m; i++) {
        int env_idx = i * n / m;
        Point p2 = t2_raw[t2_offset + i];
        FrechetEnvelope env = envs_t1[env_offset + env_idx];

        // 计算点到包络线（矩形框）的最短距离
        float dx = fmaxf(0.0f, fmaxf(env.min_x - p2.x, p2.x - env.max_x));
        float dy = fmaxf(0.0f, fmaxf(env.min_y - p2.y, p2.y - env.max_y));
        float current_dist = sqrtf(dx * dx + dy * dy);

        // 取整个过程中的最大值，作为严格的 Fréchet 下界
        lb_dist = fmaxf(lb_dist, current_dist);
    }
    lb_matrix[t1_idx * num_t + t2_idx] = lb_dist;
}


__global__ void frechet_refine_kernel_m(const Point* t1, const Point* t2, const int2* pair_list,
    float* results, int num_pairs, int n, int m)
{
    int bid = blockIdx.x;
    if (bid >= num_pairs) return;

    // 获取需要精确计算的真实轨迹索引
    int t1_idx = pair_list[bid].x;
    int t2_idx = pair_list[bid].y;

    int tid = threadIdx.x;

    // 1. 独立计算两条轨迹的偏移量
    int offset1 = t1_idx * n;
    int offset2 = t2_idx * m;

    // 申请动态共享内存
    extern __shared__ float s_dp[];

    // 将一维的动态共享内存划分为三段，每段长度依然为 n (按行标 i 索引)
    float* dp_prev2 = s_dp;
    float* dp_prev1 = s_dp + n;
    float* dp_curr = s_dp + 2 * n;

    // 2. 循环次数变为对角线的总数：(n - 1) + (m - 1) + 1 = n + m - 1
    for (int k = 0; k < n + m - 1; k++) {
        // 3. 动态计算当前对角线上元素的起始位置和长度
        int start_i = max(0, k - m + 1);
        int end_i = min(k, n - 1);
        int len = end_i - start_i + 1;

        for (int step = tid; step < len; step += blockDim.x) {
            int i = start_i + step;
            int j = k - i;

            // 取点时使用各自的 offset
            float dx = t1[offset1 + i].x - t2[offset2 + j].x;
            float dy = t1[offset1 + i].y - t2[offset2 + j].y;
            float dist = sqrtf(dx * dx + dy * dy);

            float val = dist;
            if (k > 0) {
                float prev_min;
                if (i == 0) {
                    prev_min = dp_prev1[0];
                }
                else if (j == 0) {
                    prev_min = dp_prev1[i - 1];
                }
                else {
                    float left = dp_prev1[i];       // 对应dp[i][j-1]
                    float up = dp_prev1[i - 1];     // 对应dp[i-1][j]
                    float diag = dp_prev2[i - 1];   // 对应dp[i-1][j-1]
                    prev_min = fminf(fminf(left, up), diag);
                }
                val = fmaxf(dist, prev_min);
            }
            dp_curr[i] = val; // 写入共享内存
        }

        __syncthreads();

        // 滚动数组更新指针
        float* temp = dp_prev2;
        dp_prev2 = dp_prev1;
        dp_prev1 = dp_curr;
        dp_curr = temp;
    }

    if (tid == 0) {
        // 最终的计算结果一定落在 dp(n-1, m-1) 上，对应的行标 i 就是 n - 1
        results[bid] = dp_prev1[n - 1];
    }
}


void launch_frechet_batch_gpu_wavefront(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    // --- 阶段 A: CPU 端快速生成包络线 ( r=5) ---
    int r = 5;
    std::vector<FrechetEnvelope> h_envs(num_t * n);
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

    Point* d_t1, * d_t2;
    FrechetEnvelope* d_envs;
    float* d_lb_matrix;

    CHECK(cudaMalloc(&d_t1, pts_size1));
    CHECK(cudaMalloc(&d_t2, pts_size2));
    CHECK(cudaMalloc(&d_envs, num_t * n * sizeof(FrechetEnvelope)));
    CHECK(cudaMalloc(&d_lb_matrix, num_t * num_t * sizeof(float)));

    CHECK(cudaMemcpy(d_t1, h_t1, pts_size1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2, h_t2, pts_size2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_envs, h_envs.data(), num_t * n * sizeof(FrechetEnvelope), cudaMemcpyHostToDevice));

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu); cudaEventCreate(&stop_gpu);
    float time_lb = 0.0f, time_frechet = 0.0f;

    // --- 阶段 B: 运行过滤 Kernel，得出严格下界矩阵 ---
    cudaEventRecord(start_gpu);
    dim3 block_lb(16, 16);
    dim3 grid_lb((num_t + block_lb.x - 1) / block_lb.x, (num_t + block_lb.y - 1) / block_lb.y);
    frechet_lb_filter_kernel << <grid_lb, block_lb >> > (d_envs, d_t2, d_lb_matrix, num_t, n, m);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time_lb, start_gpu, stop_gpu);

    std::vector<float> h_lb_matrix(num_t * num_t);
    CHECK(cudaMemcpy(h_lb_matrix.data(), d_lb_matrix, num_t * num_t * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_envs);
    cudaFree(d_lb_matrix);

    // --- 阶段 C: 提取下界距离最小的 Top-10 候选对 ---
    int K = std::min(10, num_t);
    int num_pairs = num_t * K;
    std::vector<int2> h_pair_list(num_pairs);

#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        std::vector<std::pair<float, int>> row(num_t);
        for (int j = 0; j < num_t; j++) {
            row[j] = { h_lb_matrix[i * num_t + j], j };
        }
        // 按下界距离从小到大排序
        std::partial_sort(row.begin(), row.begin() + K, row.end());
        for (int k = 0; k < K; k++) {
            h_pair_list[i * K + k] = make_int2(i, row[k].second);
        }
    }

    // --- 阶段 D: 运行精确 Frechet Kernel 进行 Refine ---
    int2* d_pair_list;
    float* d_frechet_results;
    CHECK(cudaMalloc(&d_pair_list, num_pairs * sizeof(int2)));
    CHECK(cudaMalloc(&d_frechet_results, num_pairs * sizeof(float)));
    CHECK(cudaMemcpy(d_pair_list, h_pair_list.data(), num_pairs * sizeof(int2), cudaMemcpyHostToDevice));

    int max_diag_len = std::min(n, m);
    int threads = (max_diag_len < 256) ? max_diag_len : 256;
    size_t shared_mem_size = 3 * n * sizeof(float);
    cudaFuncSetAttribute(frechet_refine_kernel_m, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

    cudaEventRecord(start_gpu);
    // Grid大小配置为 num_pairs
    frechet_refine_kernel_m << <num_pairs, threads, shared_mem_size >> > (d_t1, d_t2, d_pair_list, d_frechet_results, num_pairs, n, m);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time_frechet, start_gpu, stop_gpu);

    gpu_time = time_lb + time_frechet;

    // --- 阶段 E: 写回最终结果 (Frechet是距离，取最小值) ---
    std::vector<float> h_frechet_results(num_pairs);
    CHECK(cudaMemcpy(h_frechet_results.data(), d_frechet_results, num_pairs * sizeof(float), cudaMemcpyDeviceToHost));

#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        float min_frechet = 1e20f; // 初始为无穷大

        // 遍历这 10 个精确计算的 Fréchet 结果，找出最小值
        for (int k = 0; k < K; k++) {
            float dist = h_frechet_results[i * K + k];
            if (dist < min_frechet) {
                min_frechet = dist;
            }
        }
        h_results[i] = min_frechet;
    }

    // 清理资源
    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_pair_list);
    cudaFree(d_frechet_results);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    CHECK(cudaEventRecord(stop_all));
    CHECK(cudaEventSynchronize(stop_all));
    CHECK(cudaEventElapsedTime(&time_all, start_all, stop_all));
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);

    std::cout << "\n计算时间占比  " << gpu_time / time_all ;
    gpu_time = time_all;
}