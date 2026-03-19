#include "frechet.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h> 

float launch_frechet_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n) {
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
		std::vector<float> dp((n + 1) * (n + 1), 0.0f);//储存i到j的距离
        auto get_dp = [&dp, n](int i, int j) -> float& {
            return dp[i * (n + 1) + j];
            };
        auto calc_dist = [](const Point& p1, const Point& p2) {
            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;
            return std::sqrt(dx * dx + dy * dy);
            };

#pragma omp for
        for (int k = 0; k < num_t; k++) {//两个点往前走，找寻所有可能路径中，整条路径最宽的那段距离
            int offset = k * n;

            get_dp(1, 1) = calc_dist(h_t1[offset], h_t2[offset]);

			for (int i = 2; i <= n; i++) {//初始化第一行和第一列，表示其中一个点不动，另一个点往前走的情况
                get_dp(i, 1) = std::max(get_dp(i - 1, 1), calc_dist(h_t1[offset + i - 1], h_t2[offset]));
            }
            for (int j = 2; j <= n; j++) {
                get_dp(1, j) = std::max(get_dp(1, j - 1), calc_dist(h_t1[offset], h_t2[offset + j - 1]));
            }

			for (int i = 2; i <= n; i++) {//填充dp表格，表示两个点分别往前走的情况
                for (int j = 2; j <= n; j++) {
					float d = calc_dist(h_t1[offset + i - 1], h_t2[offset + j - 1]);//当前两个点的距离
					float prev_min = std::min({ get_dp(i - 1, j), get_dp(i, j - 1), get_dp(i - 1, j - 1) });//之前三种走法的最小值
					get_dp(i, j) = std::max(d, prev_min);//必须同时满足当前距离和之前最小值的要求，才能保证路径的连续性和正确性
                }
            }
			h_results[k] = get_dp(n, n);
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}