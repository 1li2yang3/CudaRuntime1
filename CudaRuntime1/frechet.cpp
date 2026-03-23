#include "frechet.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h> 

float launch_frechet_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m) {
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
        // 1. DP表格大小修改为 (n + 1) * (m + 1)
        std::vector<float> dp((n + 1) * (m + 1), 0.0f);

        // 2. 索引计算时，列的跨度改为 (m + 1)
        auto get_dp = [&dp, m](int i, int j) -> float& {
            return dp[i * (m + 1) + j];
            };

        auto calc_dist = [](const Point& p1, const Point& p2) {
            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;
            return std::sqrt(dx * dx + dy * dy);
            };

#pragma omp for
        for (int k = 0; k < num_t; k++) {
            // 3. 两个数组在 batch 中的偏移量现在不同了
            int offset1 = k * n;
            int offset2 = k * m;

            // 初始化起点
            get_dp(1, 1) = calc_dist(h_t1[offset1], h_t2[offset2]);

            // 初始化第一列：h_t1 往前走，h_t2 停留在起点
            for (int i = 2; i <= n; i++) {
                get_dp(i, 1) = std::max(get_dp(i - 1, 1), calc_dist(h_t1[offset1 + i - 1], h_t2[offset2]));
            }

            // 初始化第一行：h_t1 停留在起点，h_t2 往前走 (注意这里上限变为 m)
            for (int j = 2; j <= m; j++) {
                get_dp(1, j) = std::max(get_dp(1, j - 1), calc_dist(h_t1[offset1], h_t2[offset2 + j - 1]));
            }

            // 填充 dp 表格
            for (int i = 2; i <= n; i++) {
                for (int j = 2; j <= m; j++) { // 注意内层循环的上限变为 m
                    // 当前两个点的距离
                    float d = calc_dist(h_t1[offset1 + i - 1], h_t2[offset2 + j - 1]);
                    // 之前三种走法的最小值
                    float prev_min = std::min({ get_dp(i - 1, j), get_dp(i, j - 1), get_dp(i - 1, j - 1) });
                    // 取 最大值 保障路径正确性
                    get_dp(i, j) = std::max(d, prev_min);
                }
            }

            // 最终结果位于 dp(n, m)
            h_results[k] = get_dp(n, m);
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}