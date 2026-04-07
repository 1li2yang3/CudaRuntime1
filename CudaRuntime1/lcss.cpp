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

#pragma omp parallel for schedule(static)
    for (int t = 0; t < num_t; t++) {
        for (int i = 0; i < n; i++) {
            float min_x = 1e20f, max_x = -1e20f, min_y = 1e20f, max_y = -1e20f;
            int start_idx = std::max(0, i - r);
            int end_idx = std::min(n - 1, i + r);

            for (int k = start_idx; k <= end_idx; k++) {
                Point p = h_t1[t * n + k];
                min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
                min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
            }
            envs[t * n + i] = { min_x, max_x, min_y, max_y };
        }
    }

    // 提取前 10 名
    int K = std::min(10, num_t);
    int min_len = std::min(n, m); // 提取到外层，用于计算相似度比例

    // --- 阶段 B & C & D: 并行处理每条 h_t1 轨迹 ---
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_t; i++) {
        int offset1 = i * n;

        // --- 步骤 1：严格上界过滤计算 ---
        std::vector<std::pair<float, int>> ub_scores(num_t); // <上界最大相似度, t2索引>

        for (int j = 0; j < num_t; j++) {
            int offset2 = j * m;
            int possible_matches = 0; // 记录可能的最大匹配点数

            for (int v = 0; v < m; v++) {
                int env_idx = v * n / m; 
                Point p2 = h_t2[offset2 + v];
                LCSSEnvelope env = envs[offset1 + env_idx];


                if (p2.x > env.min_x - epsilon && p2.x < env.max_x + epsilon &&
                    p2.y > env.min_y - epsilon && p2.y < env.max_y + epsilon) {
                    possible_matches++; // 若在区域内，代表有可能成为匹配点
                }
            }
            // 转化为相似度比例
            float ub_similarity = (float)possible_matches / min_len;
            ub_scores[j] = { ub_similarity, j };
        }

        // --- 步骤 2：提取 Top-10 严格上界最大的候选者 ---

        std::partial_sort(ub_scores.begin(), ub_scores.begin() + K, ub_scores.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

        // --- 步骤 3：对这 10 个候选者进行精确 LCSS 计算 ---
        float max_lcss = -1.0f;

        for (int k = 0; k < K; k++) {
            int t2_idx = ub_scores[k].second;
            int offset2 = t2_idx * m;
            std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

            // 精确 LCSS 的 DP 过程
            for (int u = 1; u <= n; u++) {
                Point p1 = h_t1[offset1 + u - 1];

                for (int v = 1; v <= m; v++) {
                    Point p2 = h_t2[offset2 + v - 1];

                    float dx = std::abs(p1.x - p2.x);
                    float dy = std::abs(p1.y - p2.y);

                    if (dx < epsilon && dy < epsilon) {
                        dp[u][v] = dp[u - 1][v - 1] + 1;
                    }
                    else {
                        dp[u][v] = std::max(dp[u - 1][v], dp[u][v - 1]);
                    }
                }
            }

            // 最终 LCSS 匹配点数在 dp[n][m] 中
            float current_similarity = (float)dp[n][m] / min_len;

            if (current_similarity > max_lcss) {
                max_lcss = current_similarity;
            }
        }

        // --- 步骤 4：写入最大相似度结果 ---
        h_results[i] = max_lcss;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}