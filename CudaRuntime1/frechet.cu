#include "frechet.cuh"
#include <stdio.h>
#include <algorithm>
#include <iostream>

__global__ void frechet_kernel_wavefront(const Point* t1, const Point* t2, float* results, int n,
    float* g_dp_prev2, float* g_dp_prev1, float* g_dp_curr)  
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int offset = bid * n;

    float* dp_prev2 = g_dp_prev2 + offset;
    float* dp_prev1 = g_dp_prev1 + offset;
    float* dp_curr = g_dp_curr + offset;

	for (int k = 0; k < 2 * n - 1; k++) {//这里没有虚拟边界，所以k从0到2n-1
		int start_i = max(0, k - n + 1);//i+j=k,0<=i<=n-1,0<=j<=n-1,所以i的范围是max(0,k-n+1)到min(k,n-1)
		int len = min(k, n - 1) - start_i + 1;//该条对角线上的元素个数

		for (int step = tid; step < len; step += blockDim.x) {//多个线程并行处理一条对角线上的元素
            int i = start_i + step;
            int j = k - i;

            float dx = t1[offset + i].x - t2[offset + j].x;
            float dy = t1[offset + i].y - t2[offset + j].y;
            float dist = sqrtf(dx * dx + dy * dy);

            float val = dist;
			if (k > 0) {//dp[0][0] = dist,所以k=0时不需要考虑前驱状态
                float prev_min;
				if (i == 0) {//第一行只能从左边来 对应dp[i][j-1],此时i=0
                    prev_min = dp_prev1[0];
                }
				else if (j == 0) {//第一列只能从上边来 对应dp[i-1][j],此时j=0
                    prev_min = dp_prev1[i - 1];
                }
                else {
					float left = dp_prev1[i];//对应dp[i][j-1]
					float up = dp_prev1[i - 1];//对应dp[i-1][j]
					float diag = dp_prev2[i - 1];//对应dp[i-1][j-1]
                    prev_min = fminf(fminf(left, up), diag);
                }
				val = fmaxf(dist, prev_min);//Frechet距离的递推关系是当前距离和前驱状态的最大值
            }
            dp_curr[i] = val;
        }

        __syncthreads();

		float* temp = dp_prev2;//滚动数组更新指针
        dp_prev2 = dp_prev1;
        dp_prev1 = dp_curr;
        dp_curr = temp;
    }

	if (tid == 0) {//0号线程写回结果
        results[bid] = dp_prev1[n - 1];
    }
}


void launch_frechet_batch_gpu_wavefront(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    size_t pts_size = (size_t)num_t * n * sizeof(Point);
    size_t dp_size = (size_t)num_t * n * sizeof(float); 

    Point* d_t1, * d_t2;
    float* d_results;
    float* d_dp_prev2, * d_dp_prev1, * d_dp_curr; 

    CHECK(cudaMalloc(&d_t1, pts_size));
    CHECK(cudaMalloc(&d_t2, pts_size));
    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));

    CHECK(cudaMalloc(&d_dp_prev2, dp_size));
    CHECK(cudaMalloc(&d_dp_prev1, dp_size));
    CHECK(cudaMalloc(&d_dp_curr, dp_size));

    CHECK(cudaMemcpy(d_t1, h_t1, pts_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2, h_t2, pts_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blocks = num_t;
    int threads = (n < 256) ? n : 256;

    frechet_kernel_wavefront << <blocks, threads >> > (d_t1, d_t2, d_results, n,d_dp_prev2, d_dp_prev1, d_dp_curr);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    CHECK(cudaMemcpy(h_results, d_results, num_t * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_results);
    cudaFree(d_dp_prev2);
    cudaFree(d_dp_prev1);
    cudaFree(d_dp_curr);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CHECK(cudaEventRecord(stop_all));
    CHECK(cudaEventSynchronize(stop_all));
    CHECK(cudaEventElapsedTime(&time_all, start_all, stop_all));
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);
    std::cout << "\n计算时间占比" << gpu_time / time_all;
    gpu_time = time_all;
}