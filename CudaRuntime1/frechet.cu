#include "frechet.cuh"
#include <stdio.h>
#include <algorithm>
#include <iostream>

__global__ void frechet_kernel_wavefront(const Point* t1, const Point* t2, float* results, int n) {
    int bx = blockIdx.x; 
    int tx = threadIdx.x;

    extern __shared__ char smem[];

    Point* s_t1 = (Point*)smem;
    Point* s_t2 = (Point*)(smem + n * sizeof(Point));
    float* dp_prev2 = (float*)(smem + 2 * n * sizeof(Point));                     
    float* dp_prev1 = (float*)(smem + 2 * n * sizeof(Point) + n * sizeof(float)); 
    float* dp_curr = (float*)(smem + 2 * n * sizeof(Point) + 2 * n * sizeof(float));

    int offset = bx * n;
    for (int i = tx; i < n; i += blockDim.x) {
        s_t1[i] = t1[offset + i];
        s_t2[i] = t2[offset + i];
    }
    __syncthreads(); 

    for (int k = 0; k < 2 * n - 1; k++) {
        int start_i = max(0, k - n + 1);
        int len = min(k, n - 1) - start_i + 1;

        for (int step = tx; step < len; step += blockDim.x) {
            int i = start_i + step;
            int j = k - i;

            float dx = s_t1[i].x - s_t2[j].x;
            float dy = s_t1[i].y - s_t2[j].y;
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
                    float left = dp_prev1[i];    
                    float up = dp_prev1[i - 1]; 
                    float diag = dp_prev2[i - 1]; 
                    prev_min = fminf(fminf(left, up), diag);
                }
                val = fmaxf(dist, prev_min);
            }
            dp_curr[i] = val;
        }

        __syncthreads();

        float* temp = dp_prev2;
        dp_prev2 = dp_prev1;
        dp_prev1 = dp_curr;
        dp_curr = temp;
    }


    if (tx == 0) {

        results[bx] = dp_prev1[n - 1];
    }
}


void launch_frechet_batch_gpu_wavefront(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    size_t pts_size = (size_t)num_t * n * sizeof(Point);

    Point* d_t1, * d_t2;
    float* d_results;
    CHECK(cudaMalloc(&d_t1, pts_size));
    CHECK(cudaMalloc(&d_t2, pts_size));
    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));

    CHECK(cudaMemcpy(d_t1, h_t1, pts_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2, h_t2, pts_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blocks = num_t;

    int threads = (n < 256) ? n : 256;

    size_t shared_mem_size = 2 * n * sizeof(Point) + 3 * n * sizeof(float);

    frechet_kernel_wavefront << <blocks, threads, shared_mem_size >> > (d_t1, d_t2, d_results, n);

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
    std::cout << "\n计算时间占比" << gpu_time / time_all;
    gpu_time = time_all; 
}