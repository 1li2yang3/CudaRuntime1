#include "dtw.cuh"
#include <cmath>
#include <algorithm>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <vector>
#include <omp.h>
#include <cuda_awbarrier.h>
#include <iostream>

//__global__ void dtw_kernel(const Point* t1_raw, const Point* t2_raw, float* results, float* global_dp, int num_t, int n) {
//    int bid = blockIdx.x;
//    if (bid >= num_t) return;
//    size_t block_offset = (size_t)bid * 3 * (n + 1);//全局DP偏移量
//	float* d0 = global_dp + block_offset;//上两行 k-2
//	float* d1 = global_dp + block_offset + (n + 1);//上一行 k-1
//	float* d2 = global_dp + block_offset + 2 * (n + 1);//当前行 k
//
//    int tid = threadIdx.x;
//
//	for (int i = tid; i <= n; i += blockDim.x) {//初始化DP表,无穷大表示不可达
//        d0[i] = 1e20f;
//        d1[i] = 1e20f;
//        d2[i] = 1e20f;
//    }
//    __syncthreads(); 
//
//	if (tid == 0) {//0号线程初始化虚空点dp[0][0]=0
//        d0[0] = 0.0f;
//    }
//    __syncthreads();
//
//	for (int k = 2; k <= 2 * n; k++) {//定义k=i+j，沿反对角线计算DP，每条反对角线上的元素可以并行计算
//		int i_start = max(1, k - n);//j=k-i且1<=j<=n推导出i的范围，即1<=i<=n且1<=k-i<=n
//        int i_end = min(n, k - 1);
//
//		for (int i = i_start + tid; i <= i_end; i += blockDim.x) {//每个线程计算反对角线上的一个元素
//            int j = k - i;
//
//			Point p1 = t1_raw[bid * n + (i - 1)];//索引偏移，j-1是因为DP表多了一行和一列
//			Point p2 = t2_raw[bid * n + (j - 1)];//k=2时i=1,j=1，访问t1[0]和t2[0]
//
//            float dx = p1.x - p2.x;
//            float dy = p1.y - p2.y;
//            float cost = sqrtf(dx * dx + dy * dy);
//            // 正常状态转移方程为dp[i][j] = cost + std::min({ dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] });
//			float diag = d0[i - 1];//k-2行的i-1位置，即dp[i-1][j-1]
//			float up = d1[i - 1];//k-1行的i-1位置，即dp[i-1][j]
//			float left = d1[i];//k-1行的i位置，即dp[i][j-1]
//
//			d2[i] = cost + fminf(diag, fminf(up, left));//计算当前元素dp[i][j]
//        }
//
//        __syncthreads();
//
//		if (k == 2 && tid == 0) d0[0] = 1e20f;//计算完第一条反对角线后，虚空点dp[0][0]不再可达，重置为无穷大
//
//		float* temp = d0;//循环利用全局DP表的三行，更新行指针
//        d0 = d1;
//        d1 = d2;
//        d2 = temp;
//
//        for (int i = i_start + tid; i <= i_end; i += blockDim.x) {
//            d2[i] = 1e20f;
//        }
//        __syncthreads();
//    }
//
//	if (tid == 0) {//0号线程将结果写回全局内存
//        results[bid] = d1[n];
//    }
//}
//
//
//void launch_dtw_batch_gpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time) {
//    cudaEvent_t start_all, stop_all;
//    float time_all = 0.0f;
//    CHECK(cudaEventCreate(&start_all));
//    CHECK(cudaEventCreate(&stop_all));
//    CHECK(cudaEventRecord(start_all));
//
//    size_t traj_size = (size_t)num_t * n * sizeof(Point);
//	size_t global_dp_size = (size_t)num_t * 3 * (n + 1) * sizeof(float);//每个线程块需要3行(n+1)的DP表
//    Point* d_t1, * d_t2;
//    float* d_results;
//    float* d_global_dp; 
//
//    CHECK(cudaMalloc(&d_t1, traj_size));
//    CHECK(cudaMalloc(&d_t2, traj_size));
//    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));
//    CHECK(cudaMalloc(&d_global_dp, global_dp_size)); 
//
//    CHECK(cudaMemcpy(d_t1, h_t1, traj_size, cudaMemcpyHostToDevice));
//    CHECK(cudaMemcpy(d_t2, h_t2, traj_size, cudaMemcpyHostToDevice));
//
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start); cudaEventCreate(&stop);
//    cudaEventRecord(start);
//
//    int threadsPerBlock = (n < 256) ? ((n / 32 + 1) * 32) : 256;
//    int blocksPerGrid = num_t;
//
//    dtw_kernel << <blocksPerGrid, threadsPerBlock >> > (d_t1, d_t2, d_results, d_global_dp, num_t, n);
//
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&gpu_time, start, stop);
//
//    CHECK(cudaMemcpy(h_results, d_results, num_t * sizeof(float), cudaMemcpyDeviceToHost));
//
//    cudaFree(d_t1);
//    cudaFree(d_t2);
//    cudaFree(d_results);
//    cudaFree(d_global_dp); 
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
//
//}



//----共享内存版本----
__global__ void dtw_kernel(const Point* t1_raw, const Point* t2_raw, float* results, int num_t, int n) {
    int bid = blockIdx.x;
    if (bid >= num_t) return;

    int tid = threadIdx.x;

    // 引入动态共享内存，大小在 Host 端指定为 3 * (n + 1) * sizeof(float)
    extern __shared__ float s_dp[];

    // 划分共享内存为三行，每行长度为 n + 1
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

    // 沿反对角线计算DP
    for (int k = 2; k <= 2 * n; k++) {
        int i_start = max(1, k - n); // j=k-i且1<=j<=n推导出i的范围
        int i_end = min(n, k - 1);

        for (int i = i_start + tid; i <= i_end; i += blockDim.x) { // 每个线程计算反对角线上的一个元素
            int j = k - i;

            Point p1 = t1_raw[bid * n + (i - 1)]; // 索引偏移，j-1是因为DP表多了一行和一列
            Point p2 = t2_raw[bid * n + (j - 1)]; // k=2时i=1,j=1，访问t1[0]和t2[0]

            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;
            float cost = sqrtf(dx * dx + dy * dy);

            // 状态转移方程为 dp[i][j] = cost + std::min({ dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] });
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
        results[bid] = d1[n];
    }
}

void launch_dtw_batch_gpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    size_t traj_size = (size_t)num_t * n * sizeof(Point);

    Point* d_t1, * d_t2;
    float* d_results;

    // 分配点集和结果的内存
    CHECK(cudaMalloc(&d_t1, traj_size));
    CHECK(cudaMalloc(&d_t2, traj_size));
    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));

    // 注意：完全移除了 d_global_dp 的 cudaMalloc 分配

    CHECK(cudaMemcpy(d_t1, h_t1, traj_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2, h_t2, traj_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threadsPerBlock = (n < 256) ? ((n / 32 + 1) * 32) : 256;
    int blocksPerGrid = num_t;

    // 计算动态共享内存大小：3行，每行 n + 1 个 float
    size_t shared_mem_size = 3 * (n + 1) * sizeof(float);
    cudaFuncSetAttribute(dtw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    // 启动 kernel 时传入动态共享内存大小
    dtw_kernel << <blocksPerGrid, threadsPerBlock, shared_mem_size >> > (d_t1, d_t2, d_results, num_t, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    CHECK(cudaMemcpy(h_results, d_results, num_t * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_results);
    // 注意：移除了 cudaFree(d_global_dp);

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