#include "dtw.cuh"
#include <cmath>
#include <algorithm>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <vector>
#include <omp.h>
#include <cuda_awbarrier.h>
#include <iostream>

__global__ void dtw_kernel_m(const Point* t1_raw, const Point* t2_raw, float* results, int num_t, int n, int m) {
    int bid = blockIdx.x;
    if (bid >= num_t) return;

    int tid = threadIdx.x;

    // 1. 独立计算偏移量
    int offset1 = bid * n;
    int offset2 = bid * m;

    // 引入动态共享内存，大小在 Host 端指定为 3 * (n + 1) * sizeof(float)
    extern __shared__ float s_dp[];

    // 划分共享内存为三行，每行长度为 n + 1 (因为 i 最大是 n)
    float* d0 = s_dp;                        // 上两行 k-2
    float* d1 = s_dp + (n + 1);              // 上一行 k-1
    float* d2 = s_dp + 2 * (n + 1);          // 当前行 k

    // 初始化DP表, 无穷大表示不可达
    for (int i = tid; i <= n; i += blockDim.x) {
        d0[i] = 1e20f;
        d1[i] = 1e20f;
        d2[i] = 1e20f;
    }
    __syncthreads();

    // 0号线程初始化虚空点 dp[0][0] = 0
    if (tid == 0) {
        d0[0] = 0.0f;
    }
    __syncthreads();

    // 2. 循环总次数改为 n + m
    for (int k = 2; k <= n + m; k++) {
        // 3. 根据 j 的最大值 m 推导 i 的下界
        int i_start = max(1, k - m); // j=k-i 且 1<=j<=m 推导出 i>=k-m
        int i_end = min(n, k - 1);

        for (int i = i_start + tid; i <= i_end; i += blockDim.x) {
            int j = k - i;

            // 4. 使用各自独立的 offset
            Point p1 = t1_raw[offset1 + (i - 1)];
            Point p2 = t2_raw[offset2 + (j - 1)];

            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;
            float cost = sqrtf(dx * dx + dy * dy);

            float diag = d0[i - 1]; // k-2行的i-1位置，即dp[i-1][j-1]
            float up = d1[i - 1];   // k-1行的i-1位置，即dp[i-1][j]
            float left = d1[i];     // k-1行的i位置，即dp[i][j-1]

            d2[i] = cost + fminf(diag, fminf(up, left)); // 计算当前元素dp[i][j]并写入共享内存
        }

        __syncthreads();

        // 计算完第一条反对角线后，虚空点dp[0][0]不再可达，重置为无穷大
        if (k == 2 && tid == 0) d0[0] = 1e20f;

        // 循环利用共享内存的三行，更新行指针
        float* temp = d0;
        d0 = d1;
        d1 = d2;
        d2 = temp;

        // 将新的 d2 滚动行重置为无穷大，为下一轮计算做准备
        for (int i = i_start + tid; i <= i_end; i += blockDim.x) {
            d2[i] = 1e20f;
        }
        __syncthreads();
    }

    // 0号线程将结果写回全局内存
    if (tid == 0) {
        // 最终结果位于 d1[n] 中（因为当 k=n+m 循环结束后，d2变成了d0，d1依然是最后一步求出的当前行结果）
        results[bid] = d1[n];
    }
}
void launch_dtw_batch_gpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    // 1. 分别计算大小
    size_t traj_size1 = (size_t)num_t * n * sizeof(Point);
    size_t traj_size2 = (size_t)num_t * m * sizeof(Point);

    Point* d_t1, * d_t2;
    float* d_results;

    // 分配点集和结果的内存
    CHECK(cudaMalloc(&d_t1, traj_size1));
    CHECK(cudaMalloc(&d_t2, traj_size2));
    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));

    CHECK(cudaMemcpy(d_t1, h_t1, traj_size1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2, h_t2, traj_size2, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 2. 线程块大小配置
    int max_diag_len = std::min(n, m);
    int threadsPerBlock = (max_diag_len < 256) ? ((max_diag_len / 32 + 1) * 32) : 256;
    int blocksPerGrid = num_t;

    // 3. 共享内存大小：3行，每行依然是 n + 1 个 float
    size_t shared_mem_size = 3 * (n + 1) * sizeof(float);
    cudaFuncSetAttribute(dtw_kernel_m, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);


    dtw_kernel_m << <blocksPerGrid, threadsPerBlock, shared_mem_size >> > (d_t1, d_t2, d_results, num_t, n, m);

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