#include "lcss.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h> 
#include <iostream>

struct LCSSEnvelope {
    float min_x, max_x, min_y, max_y;
};

float launch_lcss_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m, float epsilon) {
    auto start = std::chrono::high_resolution_clock::now();

    int r = 5;
    std::vector<LCSSEnvelope> envs(num_t * n);


#pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < num_t; t++) {
        for (int i = 0; i < n; i++) {
            float min_x = 1e20f, max_x = -1e20f, min_y = 1e20f, max_y = -1e20f;
            int start_idx = std::max(0, i - r);
            int end_idx = std::min(n - 1, i + r);

            for (int k = start_idx; k <= end_idx; k++) {
                Point p = h_t1[t * n + k];
                min_x = std::fmin(min_x, p.x); max_x = std::fmax(max_x, p.x);
                min_y = std::fmin(min_y, p.y); max_y = std::fmax(max_y, p.y);
            }

            envs.at(t * n + i) = { min_x, max_x, min_y, max_y };
        }
    }

    int K = std::min(10, num_t);
    int min_len = std::min(n, m);


#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < num_t; i++) {
        int offset1 = i * n;
        std::vector<std::pair<float, int>> ub_scores(num_t);

        for (int j = 0; j < num_t; j++) {
            int offset2 = j * m;
            int possible_matches = 0;

            for (int v = 0; v < m; v++) {

                int env_idx = static_cast<int>(std::floor((double)v * (double)n / (double)m));
                Point p2 = h_t2[offset2 + v];

                LCSSEnvelope env = envs.at(offset1 + env_idx);

                if (p2.x > env.min_x - epsilon && p2.x < env.max_x + epsilon &&
                    p2.y > env.min_y - epsilon && p2.y < env.max_y + epsilon) {
                    possible_matches++;
                }
            }
            float ub_similarity = (float)possible_matches / min_len;
            ub_scores.at(j) = { ub_similarity, j };
        }


        std::stable_sort(ub_scores.begin(), ub_scores.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

        float max_lcss = -1.0f;

        for (int k = 0; k < K; k++) {
            int t2_idx = ub_scores.at(k).second;
            int offset2 = t2_idx * m;

            std::vector<std::vector<int>> dp;
            for (int row_idx = 0; row_idx <= n; ++row_idx) {
                std::vector<int> row;
                for (int col_idx = 0; col_idx <= m; ++col_idx) {
                    row.push_back(0);
                }
                dp.push_back(row);
            }

            for (int u = 1; u <= n; u++) {
                Point p1 = h_t1[offset1 + u - 1];

                for (int v = 1; v <= m; v++) {
                    Point p2 = h_t2[offset2 + v - 1];

                    float dx = std::abs(p1.x - p2.x);
                    float dy = std::abs(p1.y - p2.y);

                    if (dx < epsilon && dy < epsilon) {
                        dp.at(u).at(v) = dp.at(u - 1).at(v - 1) + 1;
                    }
                    else {
                        dp.at(u).at(v) = std::max(dp.at(u - 1).at(v), dp.at(u).at(v - 1));
                    }
                }
            }

            float current_similarity = (float)dp.at(n).at(m) / min_len;
            if (current_similarity > max_lcss) {
                max_lcss = current_similarity;
            }
        }
        h_results[i] = max_lcss;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}