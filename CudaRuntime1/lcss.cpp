#include "lcss.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h> 

float launch_lcss_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m, float epsilon) {
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(static)
    for (int k = 0; k < num_t; k++) {
        // 1. 独立计算两条轨迹的偏移量
        int offset1 = k * n;
        int offset2 = k * m;

        // 2. DP 表格大小修改为 (n + 1) * (m + 1)
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

        // i 代表 h_t1 的点，上限为 n
        for (int i = 1; i <= n; i++) {
            // 3. j 代表 h_t2 的点，上限改为 m
            for (int j = 1; j <= m; j++) {

                // 4. 使用各自独立的 offset 来获取坐标
                float dx = std::abs(h_t1[offset1 + i - 1].x - h_t2[offset2 + j - 1].x);
                float dy = std::abs(h_t1[offset1 + i - 1].y - h_t2[offset2 + j - 1].y);

                if (dx < epsilon && dy < epsilon) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }
                else {
                    dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        // 5. 获取 dp[n][m] 的结果，并为了和 GPU 版本保持一致，除以 min(n, m)
        int min_len = std::min(n, m);
        h_results[k] = (float)dp[n][m] / min_len;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}