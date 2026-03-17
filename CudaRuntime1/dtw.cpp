#include "dtw.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h> 

float launch_dtw_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n) {
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int k = 0; k < num_t; k++) {
        int offset = k * n;
        std::vector<std::vector<float>> dp(n + 1, std::vector<float>(n + 1, 1e20f));
        dp[0][0] = 0;

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                float dx = h_t1[offset + i - 1].x - h_t2[offset + j - 1].x;
                float dy = h_t1[offset + i - 1].y - h_t2[offset + j - 1].y;
                float cost = sqrt(dx * dx + dy * dy);
                dp[i][j] = cost + std::min({ dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] });
            }
        }
        h_results[k] = dp[n][n];
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}



