#include "lcss.cuh"
#include <stdio.h>
#include <math.h>
#include <iostream>

__global__ void lcss_wavefront_kernel_m(const Point* __restrict__ t1_raw, const Point* __restrict__ t2_raw,
    float* __restrict__ results, int num_t, int n, int m, float epsilon) {

    int bid = blockIdx.x;
    if (bid >= num_t) return;

    // 引入动态共享内存，大小在 Host 启动时指定为 3 * n * sizeof(int)
    extern __shared__ int s_diagonals_flat[];

    int tid = threadIdx.x;

    // 独立计算两条轨迹的偏移量
    int offset1 = bid * n;
    int offset2 = bid * m;

    // 循环次数变为对角线的最大值：n + m
    for (int k = 2; k <= n + m; k++) {
        int curr_buf = k % 3;
        int prev1_buf = (k - 1) % 3;
        int prev2_buf = (k - 2) % 3;

        // i 的上界依然受限于 n
        int start_i = min(n, k - 1);
        // i 的下界受限于 m (因为 j = k - i <= m => i >= k - m)
        int end_i = max(1, k - m);
        int num_elements = start_i - end_i + 1;

        for (int idx = tid; idx < num_elements; idx += blockDim.x) {
            int i = start_i - idx;
            int j = k - i;

            // 使用各自的 offset 取点
            float dx = fabsf(t1_raw[offset1 + i - 1].x - t2_raw[offset2 + j - 1].x);
            float dy = fabsf(t1_raw[offset1 + i - 1].y - t2_raw[offset2 + j - 1].y);
            bool match = (dx < epsilon) && (dy < epsilon);

            int val = 0;
            if (match) {
                int prev_diag_val = 0;
                if (i > 1 && j > 1) {
                    // start_i_k2 的推导依然成立，因为它只和 n 有关
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

            // 写入共享内存的当前缓冲区
            s_diagonals_flat[curr_buf * n + idx] = val;
        }

        __syncthreads();
    }

    if (tid == 0) {
        // 最终结果位于 k = n + m 的对角线上
        int last_buf = (n + m) % 3;
        // 标准LCSS通常除以 min(n, m) 或者 max(n, m)，这里根据两条轨迹的情况除以较短的一条
        int min_len = min(n, m);
        results[bid] = (float)s_diagonals_flat[last_buf * n + 0] / min_len;
    }
}

void launch_lcss_batch_gpu_wavefront(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m, float epsilon, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    // 1. 分别计算大小
    size_t pts_size1 = (size_t)num_t * n * sizeof(Point);
    size_t pts_size2 = (size_t)num_t * m * sizeof(Point);

    Point* d_t1_raw, * d_t2_raw;
    float* d_results;

    CHECK(cudaMalloc(&d_t1_raw, pts_size1));
    CHECK(cudaMalloc(&d_t2_raw, pts_size2));
    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));

    CHECK(cudaMemcpy(d_t1_raw, h_t1, pts_size1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2_raw, h_t2, pts_size2, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threadsPerBlock = 256;

    // 2. 共享内存大小依然维持为 3 * n * sizeof(int)
    // 理由同上：对角线的映射是以行索引 i 为准的，最大就是 n。
    size_t shared_mem_size = 3 * n * sizeof(int);
    cudaFuncSetAttribute(lcss_wavefront_kernel_m, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

    // 3. 传入参数 m
    lcss_wavefront_kernel_m << <num_t, threadsPerBlock, shared_mem_size >> > (d_t1_raw, d_t2_raw, d_results, num_t, n, m, epsilon);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    CHECK(cudaMemcpy(h_results, d_results, num_t * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_t1_raw);
    cudaFree(d_t2_raw);
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