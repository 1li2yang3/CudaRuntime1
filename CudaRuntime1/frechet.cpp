#include "frechet.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h> 
#include <iostream>

//float launch_frechet_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m) {
//    auto start = std::chrono::high_resolution_clock::now();
//
//#pragma omp parallel
//    {
//        // 1. DP表格大小修改为 (n + 1) * (m + 1)
//        std::vector<float> dp((n + 1) * (m + 1), 0.0f);
//
//        // 2. 索引计算时，列的跨度改为 (m + 1)
//        auto get_dp = [&dp, m](int i, int j) -> float& {
//            return dp[i * (m + 1) + j];
//            };
//
//        auto calc_dist = [](const Point& p1, const Point& p2) {
//            float dx = p1.x - p2.x;
//            float dy = p1.y - p2.y;
//            return std::sqrt(dx * dx + dy * dy);
//            };
//
//#pragma omp for
//        for (int k = 0; k < num_t; k++) {
//            // 3. 两个数组在 batch 中的偏移量现在不同了
//            int offset1 = k * n;
//            int offset2 = k * m;
//
//            // 初始化起点
//            get_dp(1, 1) = calc_dist(h_t1[offset1], h_t2[offset2]);
//
//            // 初始化第一列：h_t1 往前走，h_t2 停留在起点
//            for (int i = 2; i <= n; i++) {
//                get_dp(i, 1) = std::max(get_dp(i - 1, 1), calc_dist(h_t1[offset1 + i - 1], h_t2[offset2]));
//            }
//
//            // 初始化第一行：h_t1 停留在起点，h_t2 往前走 (注意这里上限变为 m)
//            for (int j = 2; j <= m; j++) {
//                get_dp(1, j) = std::max(get_dp(1, j - 1), calc_dist(h_t1[offset1], h_t2[offset2 + j - 1]));
//            }
//
//            // 填充 dp 表格
//            for (int i = 2; i <= n; i++) {
//                for (int j = 2; j <= m; j++) { // 注意内层循环的上限变为 m
//                    // 当前两个点的距离
//                    float d = calc_dist(h_t1[offset1 + i - 1], h_t2[offset2 + j - 1]);
//                    // 之前三种走法的最小值
//                    float prev_min = std::min({ get_dp(i - 1, j), get_dp(i, j - 1), get_dp(i - 1, j - 1) });
//                    // 取 最大值 保障路径正确性
//                    get_dp(i, j) = std::max(d, prev_min);
//                }
//            }
//
//            // 最终结果位于 dp(n, m)
//            h_results[k] = get_dp(n, m);
//        }
//    }
//
//    auto stop = std::chrono::high_resolution_clock::now();
//    return std::chrono::duration<float, std::milli>(stop - start).count();
//}






// ==========================================
// 1. 结构体定义 (加了 Frechet 前缀防冲突)
// ==========================================
struct FrechetEnvelope {
    float min_x, max_x, min_y, max_y;
};

// ==========================================
// 2. CPU 严格下界过滤+精算 主函数
// ==========================================
float launch_frechet_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m) {
    auto start = std::chrono::high_resolution_clock::now();

    // --- 阶段 A: 预计算 h_t1 的 2D 包络线 (Window r=5) ---
    int r = 5;
    std::vector<FrechetEnvelope> envs(num_t * n);

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

    // --- 阶段 B & C & D: 并行处理每条 h_t1 轨迹 ---
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_t; i++) {
        int offset1 = i * n;

        // --- 步骤 1：包络线严格下界过滤计算 ---
        std::vector<std::pair<float, int>> lb_dists(num_t); // <下界最大距离, t2索引>

        for (int j = 0; j < num_t; j++) {
            int offset2 = j * m;
            float lb_dist = 0.0f; // Fréchet 是瓶颈度量，初始下界为0

            for (int v = 0; v < m; v++) {
                int env_idx = v * n / m; // 长度映射
                Point p2 = h_t2[offset2 + v];
                FrechetEnvelope env = envs[offset1 + env_idx];

                float dx = std::max(0.0f, std::max(env.min_x - p2.x, p2.x - env.max_x));
                float dy = std::max(0.0f, std::max(env.min_y - p2.y, p2.y - env.max_y));
                float current_dist = std::sqrt(dx * dx + dy * dy);


                lb_dist = std::max(lb_dist, current_dist);
            }
            lb_dists[j] = { lb_dist, j };
        }

        // --- 步骤 2：提取 Top-10 严格下界最小的候选者 ---
        // 下界越小，真实的 Fréchet 距离才有可能小
        std::partial_sort(lb_dists.begin(), lb_dists.begin() + K, lb_dists.end());

        // --- 步骤 3：对这 10 个候选者进行精确 Fréchet 计算 ---
        float min_frechet = 1e20f; // Fréchet 是距离度量，越小越好，初始赋无穷大

        auto calc_dist = [](const Point& p1, const Point& p2) {
            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;
            return std::sqrt(dx * dx + dy * dy);
            };

        for (int k = 0; k < K; k++) {
            int t2_idx = lb_dists[k].second;
            int offset2 = t2_idx * m;


            std::vector<float> dp((n + 1) * (m + 1), 0.0f);

            auto get_dp = [&dp, m](int u, int v) -> float& {
                return dp[u * (m + 1) + v];
                };

            // 初始化起点
            get_dp(1, 1) = calc_dist(h_t1[offset1], h_t2[offset2]);

            // 初始化第一列：h_t1 往前走，h_t2 停留在起点
            for (int u = 2; u <= n; u++) {
                get_dp(u, 1) = std::max(get_dp(u - 1, 1), calc_dist(h_t1[offset1 + u - 1], h_t2[offset2]));
            }

            // 初始化第一行：h_t1 停留在起点，h_t2 往前走
            for (int v = 2; v <= m; v++) {
                get_dp(1, v) = std::max(get_dp(1, v - 1), calc_dist(h_t1[offset1], h_t2[offset2 + v - 1]));
            }

            // 填充 dp 表格
            for (int u = 2; u <= n; u++) {
                for (int v = 2; v <= m; v++) {
                    float d = calc_dist(h_t1[offset1 + u - 1], h_t2[offset2 + v - 1]);
                    float prev_min = std::min({ get_dp(u - 1, v), get_dp(u, v - 1), get_dp(u - 1, v - 1) });
                    get_dp(u, v) = std::max(d, prev_min);
                }
            }

            // 记录最小的 Fréchet 距离
            float current_frechet = get_dp(n, m);
            if (current_frechet < min_frechet) {
                min_frechet = current_frechet;
            }
        }

        // --- 步骤 4：写入最小 Fréchet 距离结果 ---
        h_results[i] = min_frechet;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}