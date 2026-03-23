#include "frechet.cuh"
#include <stdio.h>
#include <algorithm>
#include <iostream>

__global__ void frechet_kernel_wavefront_m(const Point* t1, const Point* t2, float* results, int n, int m)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // 1. 独立计算两条轨迹的偏移量
    int offset1 = bid * n;
    int offset2 = bid * m;

    // 申请动态共享内存
    extern __shared__ float s_dp[];

    // 将一维的动态共享内存划分为三段，每段长度依然为 n (按行标 i 索引)
    float* dp_prev2 = s_dp;
    float* dp_prev1 = s_dp + n;
    float* dp_curr = s_dp + 2 * n;

    // 2. 循环次数变为对角线的总数：(n - 1) + (m - 1) + 1 = n + m - 1
    for (int k = 0; k < n + m - 1; k++) {
        // 3. 动态计算当前对角线上元素的起始位置和长度
        // i + j = k, 且 0 <= j <= m - 1 => k - m + 1 <= i
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
                    float up = dp_prev1[i - 1];   // 对应dp[i-1][j]
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

    // 1. 分别计算两条轨迹各自所需的显存大小
    size_t pts_size1 = (size_t)num_t * n * sizeof(Point);
    size_t pts_size2 = (size_t)num_t * m * sizeof(Point);

    Point* d_t1, * d_t2;
    float* d_results;

    CHECK(cudaMalloc(&d_t1, pts_size1));
    CHECK(cudaMalloc(&d_t2, pts_size2));
    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));

    CHECK(cudaMemcpy(d_t1, h_t1, pts_size1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2, h_t2, pts_size2, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blocks = num_t;
    // 2. 对角线上的最大元素数量为 min(n, m)，可以以此来优化线程块大小
    int max_diag_len = std::min(n, m);
    int threads = (max_diag_len < 256) ? max_diag_len : 256;

    // 3. 共享内存大小只需按照行跨度 n 申请（3 * n）
    // 注意：如果 m 远大于 n，按 n 申请是最省显存的；只要能存下最大的 i 即可。
    size_t shared_mem_size = 3 * n * sizeof(float);
    cudaFuncSetAttribute(frechet_kernel_wavefront_m, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

    // 4. 启动Kernel，加入参数 m
    frechet_kernel_wavefront_m << <blocks, threads, shared_mem_size >> > (d_t1, d_t2, d_results, n, m);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

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