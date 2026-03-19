#include "lcss.cuh"
#include <stdio.h>
#include <math.h>
#include <iostream>

__global__ void lcss_wavefront_kernel(const Point* __restrict__ t1_raw, const Point* __restrict__ t2_raw,
    float* __restrict__ results, int* __restrict__ global_diagonals_flat,int num_t, int n, float epsilon) {

    int bid = blockIdx.x;
    if (bid >= num_t) return;
	int* diagonals_flat = &global_diagonals_flat[bid * 3 * n];//滚动数组，分为3个区域，每个区域的意义也是滚动的
    int tid = threadIdx.x;

    for (int k = 2; k <= 2 * n; k++) {
        int curr_buf = k % 3;
        int prev1_buf = (k - 1) % 3;
        int prev2_buf = (k - 2) % 3;

		int start_i = min(n, k - 1);//i的上界 i+j=k，1<=i<=n,1<=j<=n，所以i的范围是max(1,k-n)到min(n,k-1)
		int end_i = max(1, k - n);//同上，i的范围下界
		int num_elements = start_i - end_i + 1;//当前对角线上的元素数量

		for (int idx = tid; idx < num_elements; idx += blockDim.x) {//多个线程处理当前对角线上的多个元素
			//idx是当前对角线上的元素索引，从0到num_elements-1，idx=0对应i=start_i，idx=num_elements-1对应i=end_i
			//i和idx的关系是i+idx=start_i，i和j的关系是i+j=k，dp数组的计算和存储是根据idx来进行的
			int i = start_i - idx;//根据idx计算i的值，i从start_i递减到end_i
            int j = k - i;

			float dx = fabsf(t1_raw[bid * n + i - 1].x - t2_raw[bid * n + j - 1].x);//-1是因为i和j是从1开始的，而数组索引是从0开始的
            float dy = fabsf(t1_raw[bid * n + i - 1].y - t2_raw[bid * n + j - 1].y);
            bool match = (dx < epsilon) && (dy < epsilon);

            int val = 0;
            if (match) {//对应的方程为dp[i][j] = dp[i - 1][j - 1] + 1;
                int prev_diag_val = 0;
                if (i > 1 && j > 1) {
					int start_i_k2 = min(n, k - 3);//对于dp[i-1][j-1]，i-1+j-1=k-2，所以i-1的上界是min(n, k - 3)
                    //从prev2_buf中获取dp[i-1][j-1]的值，i-1对应的索引是start_i_k2-(i-1)
					prev_diag_val = diagonals_flat[prev2_buf * n + (start_i_k2 - (i - 1))];
                    
                }
                val = prev_diag_val + 1;
            }
            else {//对应dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
                int left_val = 0;
                int up_val = 0;
				int start_i_k1 = min(n, k - 2);//对于dp[i][j-1]和dp[i-1][j]，i+j-1=k-1，所以i的上界是min(n, k - 2)

				if (j > 1) left_val = diagonals_flat[prev1_buf * n + (start_i_k1 - i)];//对于dp[i][j-1]
				if (i > 1) up_val = diagonals_flat[prev1_buf * n + (start_i_k1 - (i - 1))];//对于dp[i-1][j]

                val = max(left_val, up_val);
            }

			diagonals_flat[curr_buf * n + idx] = val;//将计算结果写入当前缓冲区，idx对应当前对角线上的元素索引
        }
        __syncthreads();
    }

    if (tid == 0) {
		int last_buf = (2 * n) % 3;//dp[n][n]对应的对角线是k=2*n,对应滚动数组的(2 * n) % 3号区域
        results[bid] = (float)diagonals_flat[last_buf * n + 0] / n;
    }
}

void launch_lcss_batch_gpu_wavefront(const Point* h_t1, const Point* h_t2, float* h_results,int num_t, int n, float epsilon, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    size_t pts_size = (size_t)num_t * n * sizeof(Point);
    size_t diag_size = (size_t)num_t * 3 * n * sizeof(int); 
    Point* d_t1_raw, * d_t2_raw;
    float* d_results;
    int* d_diagonals_flat; 

    CHECK(cudaMalloc(&d_t1_raw, pts_size));
    CHECK(cudaMalloc(&d_t2_raw, pts_size));
    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));
    CHECK(cudaMalloc(&d_diagonals_flat, diag_size));
    CHECK(cudaMemset(d_diagonals_flat, 0, diag_size));

    CHECK(cudaMemcpy(d_t1_raw, h_t1, pts_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2_raw, h_t2, pts_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threadsPerBlock = 256;
    lcss_wavefront_kernel << <num_t, threadsPerBlock >> > (d_t1_raw, d_t2_raw, d_results, d_diagonals_flat, num_t, n, epsilon);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    CHECK(cudaMemcpy(h_results, d_results, num_t * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_t1_raw);
    cudaFree(d_t2_raw);
    cudaFree(d_results);
    cudaFree(d_diagonals_flat);
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