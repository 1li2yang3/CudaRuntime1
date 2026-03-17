#include "lcss.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h> 

float launch_lcss_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float epsilon) {
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(static)
    for (int k = 0; k < num_t; k++) {
        int offset = k * n;
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(n + 1, 0));

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
        
                float dx = std::abs(h_t1[offset + i - 1].x - h_t2[offset + j - 1].x);
                float dy = std::abs(h_t1[offset + i - 1].y - h_t2[offset + j - 1].y);

                if (dx < epsilon && dy < epsilon) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }
                else {
                    dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        h_results[k] = (float)dp[n][n] / n;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}