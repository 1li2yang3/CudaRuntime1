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
        std::vector<float> dp((n + 1) * (n + 1), 0.0f);
        auto get_dp = [&dp, n](int i, int j) -> float& {
            return dp[i * (n + 1) + j];
            };
        auto calc_dist = [](const Point& p1, const Point& p2) {
            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;
            return std::sqrt(dx * dx + dy * dy);
            };

#pragma omp for
        for (int k = 0; k < num_t; k++) {
            int offset = k * n;

            get_dp(1, 1) = calc_dist(h_t1[offset], h_t2[offset]);

            for (int i = 2; i <= n; i++) {
                get_dp(i, 1) = std::max(get_dp(i - 1, 1), calc_dist(h_t1[offset + i - 1], h_t2[offset]));
            }
            for (int j = 2; j <= n; j++) {
                get_dp(1, j) = std::max(get_dp(1, j - 1), calc_dist(h_t1[offset], h_t2[offset + j - 1]));
            }

            for (int i = 2; i <= n; i++) {
                for (int j = 2; j <= n; j++) {
                    float d = calc_dist(h_t1[offset + i - 1], h_t2[offset + j - 1]);
                    float prev_min = std::min({ get_dp(i - 1, j), get_dp(i, j - 1), get_dp(i - 1, j - 1) });
                    get_dp(i, j) = std::max(d, prev_min);
                }
            }
            h_results[k] = get_dp(n, n);
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}