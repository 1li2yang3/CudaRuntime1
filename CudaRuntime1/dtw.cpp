#include "dtw.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h> 
#include <iostream>


struct Envelope {
    float min_x, max_x, min_y, max_y;
};

float launch_dtw_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m) {
    auto start = std::chrono::high_resolution_clock::now();

    int r = 5;
    std::vector<Envelope> envs(num_t * n);

#pragma omp parallel for
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

    int K = std::min(10, num_t); // 提取前 10 名

#pragma omp parallel for
	for (int i = 0; i < num_t; i++) {//对于每一个 t1，计算它与所有 t2 的距离，并找出最小的那个
        int offset1 = i * n;

        // LB_Keogh 过滤计算
        std::vector<std::pair<float, int>> lb_dists(num_t); // 存储 <LB距离, t2索引>

		for (int j = 0; j < num_t; j++) {//对于当前 t1，计算它与所有 t2 的 LB_Keogh 距离
            int offset2 = j * m;
            float lb_dist = 0.0f;

            for (int v = 0; v < m; v++) {
                int env_idx = v * n / m;
                Point p2 = h_t2[offset2 + v];
                Envelope env = envs[offset1 + env_idx];

                float dx = std::max(0.0f, std::max(env.min_x - p2.x, p2.x - env.max_x));
                float dy = std::max(0.0f, std::max(env.min_y - p2.y, p2.y - env.max_y));
                lb_dist += std::sqrt(dx * dx + dy * dy);
            }
            lb_dists[j] = { lb_dist, j };
        }

        // 提取候选者
        std::partial_sort(lb_dists.begin(), lb_dists.begin() + K, lb_dists.end());

        // --- 第三步：针对选出的 10 名候选者，进行精确 DTW 计算 ---
        float min_dtw = 1e20f;


        std::vector<float> prev_row(m + 1, 1e20f);
        std::vector<float> curr_row(m + 1, 1e20f);

        for (int k = 0; k < K; k++) {
            int t2_idx = lb_dists[k].second;
            int offset2 = t2_idx * m;

            // 初始化滚动数组的第一行 (对应原代码 dp[0][0] = 0)
            std::fill(prev_row.begin(), prev_row.end(), 1e20f);
            prev_row[0] = 0.0f;

            //  DTW 的 DP 过程
            for (int u = 1; u <= n; u++) {
                curr_row[0] = 1e20f; // dp[u][0] 总是无穷大

                Point p1 = h_t1[offset1 + u - 1];

                for (int v = 1; v <= m; v++) {
                    Point p2 = h_t2[offset2 + v - 1];

                    float dx = p1.x - p2.x;
                    float dy = p1.y - p2.y;
                    float cost = std::sqrt(dx * dx + dy * dy);

                    // 对应 dp[u][v] = cost + min(dp[u-1][v], dp[u][v-1], dp[u-1][v-1])
                    curr_row[v] = cost + std::min({ prev_row[v], curr_row[v - 1], prev_row[v - 1] });
                }
                // 滚动更新
                std::swap(prev_row, curr_row);
            }

            // 计算结果在 prev_row[m] 中（因为最后一次循环结束时执行了 swap）
            float current_dtw = prev_row[m];

            if (current_dtw < min_dtw) {
                min_dtw = current_dtw;
            }
        }

        // --- 第四步：写入最终最小值 ---
        h_results[i] = min_dtw;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}