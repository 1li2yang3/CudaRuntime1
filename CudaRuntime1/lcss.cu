#include "lcss.cuh"
#include <stdio.h>
#include <math.h>
#include <iostream>


__global__ void lcss_wavefront_kernel(const Point* __restrict__ t1_raw,const Point* __restrict__ t2_raw,float* __restrict__ results,
    int num_t, int n, float epsilon) {
    int pair_idx = blockIdx.x;
    if (pair_idx >= num_t) return;

    extern __shared__ char s_mem[];

    Point* s_t1 = (Point*)s_mem;

    Point* s_t2 = (Point*)&s_t1[n];

    int* diagonals_flat = (int*)&s_t2[n];

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    for (int i = tid; i < n; i += bdim) {
        s_t1[i] = t1_raw[pair_idx * n + i];
        s_t2[i] = t2_raw[pair_idx * n + i];
    }
    __syncthreads();

    for (int k = 2; k <= 2 * n; k++) {
        int curr_buf = k % 3;
        int prev1_buf = (k - 1) % 3;
        int prev2_buf = (k - 2) % 3;

        int start_i = min(n, k - 1);
        int end_i = max(1, k - n);
        int num_elements = start_i - end_i + 1;

        for (int idx = tid; idx < num_elements; idx += bdim) {
            int i = start_i - idx;
            int j = k - i;

            float dx = fabsf(s_t1[i - 1].x - s_t2[j - 1].x);
            float dy = fabsf(s_t1[i - 1].y - s_t2[j - 1].y);
            bool match = (dx < epsilon) && (dy < epsilon);

            int val = 0;
            if (match) {
                int prev_diag_val = 0;
                if (i > 1 && j > 1) {
                    int start_i_k2 = min(n, k - 3);
                    prev_diag_val = diagonals_flat[prev2_buf * n + (start_i_k2 - (i - 1))];
                }
                val = prev_diag_val + 1;
            }
            else {
                int left_val = 0;
                int up_val = 0;
                int start_i_k1 = min(n, k - 2);

                if (j > 1) left_val = diagonals_flat[prev1_buf * n + (start_i_k1 - i)];
                if (i > 1) up_val = diagonals_flat[prev1_buf * n + (start_i_k1 - (i - 1))];

                val = max(left_val, up_val);
            }

            diagonals_flat[curr_buf * n + idx] = val;
        }
        __syncthreads();
    }

    if (tid == 0) {
        int last_buf = (2 * n) % 3;
        results[pair_idx] = (float)diagonals_flat[last_buf * n + 0] / n;
    }
}

void launch_lcss_batch_gpu_wavefront(const Point* h_t1, const Point* h_t2, float* h_results,
    int num_t, int n, float epsilon, float& gpu_time) {
    cudaEvent_t start_all, stop_all;
    float time_all = 0.0f;
    CHECK(cudaEventCreate(&start_all));
    CHECK(cudaEventCreate(&stop_all));
    CHECK(cudaEventRecord(start_all));

    size_t shared_mem_bytes = (2 * n * sizeof(Point)) + (3 * n * sizeof(int));

    if (shared_mem_bytes > 49152) {
        printf("Error: n is too large (%d). Shared memory exceeded 48KB limit!\n", n);
        return;
    }

    size_t pts_size = (size_t)num_t * n * sizeof(Point);
    Point* d_t1_raw, * d_t2_raw;
    float* d_results;

    CHECK(cudaMalloc(&d_t1_raw, pts_size));
    CHECK(cudaMalloc(&d_t2_raw, pts_size));
    CHECK(cudaMalloc(&d_results, num_t * sizeof(float)));

    CHECK(cudaMemcpy(d_t1_raw, h_t1, pts_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_t2_raw, h_t2, pts_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threadsPerBlock = 256;

    lcss_wavefront_kernel << <num_t, threadsPerBlock, shared_mem_bytes >> > (
        d_t1_raw, d_t2_raw, d_results, num_t, n, epsilon
        );

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
    std::cout << "\n计算时间占比" << gpu_time / time_all;
    gpu_time = time_all; 

}

