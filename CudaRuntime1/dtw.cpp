#include "dtw.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h> 

float launch_dtw_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m) {
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for //启动多线程并行计算
    for (int k = 0; k < num_t; k++) {
        // 1. 独立计算两条轨迹的偏移量
        int offset1 = k * n;
        int offset2 = k * m;

        // 2. DP 表格大小修改为 (n + 1) * (m + 1)
        std::vector<std::vector<float>> dp(n + 1, std::vector<float>(m + 1, 1e20f));
        dp[0][0] = 0;

        // i 代表 h_t1 的点，上限维持为 n
        for (int i = 1; i <= n; i++) {
            // 3. j 代表 h_t2 的点，上限修改为 m
            for (int j = 1; j <= m; j++) {

                // 4. 使用各自独立的 offset 来获取坐标
                float dx = h_t1[offset1 + i - 1].x - h_t2[offset2 + j - 1].x;
                float dy = h_t1[offset1 + i - 1].y - h_t2[offset2 + j - 1].y;
                float cost = sqrt(dx * dx + dy * dy);

                dp[i][j] = cost + std::min({ dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] });
            }
        }

        // 5. 最终结果位于 dp[n][m]
        h_results[k] = dp[n][m];
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}
