#include "hausdorff.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>  


inline float compute_directed_hausdorff(const Point* source, int len_source, const Point* target, int len_target, float current_max_min_d2) {
    float min_x = target[0].x, max_x = target[0].x;
    float min_y = target[0].y, max_y = target[0].y;
    // 目标点集的边界框，受限于 len_target
    for (int i = 1; i < len_target; i++) {
        min_x = std::min(min_x, target[i].x); max_x = std::max(max_x, target[i].x);
        min_y = std::min(min_y, target[i].y); max_y = std::max(max_y, target[i].y);
    }

    // 网格数受限于目标点集的数量
    int G = std::max(1, (int)std::sqrt(len_target));
    float cell_w = (max_x - min_x) / G;
    float cell_h = (max_y - min_y) / G;
    if (cell_w < 1e-6f) cell_w = 1e-6f;
    if (cell_h < 1e-6f) cell_h = 1e-6f;

    std::vector<std::vector<int>> grid(G * G); // 网格索引存储target点的索引

    // 将目标点放入网格，受限于 len_target
    for (int i = 0; i < len_target; i++) {
        int cx = std::max(0, std::min(G - 1, (int)((target[i].x - min_x) / cell_w)));
        int cy = std::max(0, std::min(G - 1, (int)((target[i].y - min_y) / cell_h)));
        int idx = cy * G + cx;
        grid[idx].push_back(i);
    }

    float max_min_d2 = current_max_min_d2;

    // 遍历源点集，受限于 len_source
    for (int i = 0; i < len_source; i++) {
        float px = source[i].x;
        float py = source[i].y;
        float min_d2 = 1e20f;

        int cx0 = std::max(0, std::min(G - 1, (int)((px - min_x) / cell_w)));
        int cy0 = std::max(0, std::min(G - 1, (int)((py - min_y) / cell_h)));

        int r = 0;
        while (r <= G) { // 以(cx0, cy0)为中心，半径为r
            int cx_min = cx0 - r;
            int cx_max = cx0 + r;
            int cy_min = cy0 - r;
            int cy_max = cy0 + r;

            for (int cy = cy_min; cy <= cy_max; cy++) {
                for (int cx = cx_min; cx <= cx_max; cx++) {

					if (r > 0 && !(cy == cy_min || cy == cy_max || cx == cx_min || cx == cx_max)) {// 只检查边界上的格子，内层格子在之前的半径已经检查过了
                        continue;
                    }

                    if (cx >= 0 && cx < G && cy >= 0 && cy < G) {

                        float c_min_x = min_x + cx * cell_w;
                        float c_max_x = c_min_x + cell_w;
                        float c_min_y = min_y + cy * cell_h;
                        float c_max_y = c_min_y + cell_h;

                        float dx_cell = std::max({ 0.0f, c_min_x - px, px - c_max_x });
                        float dy_cell = std::max({ 0.0f, c_min_y - py, py - c_max_y });

                        if (dx_cell * dx_cell + dy_cell * dy_cell >= min_d2) {
                            continue;
                        }

                        int idx = cy * G + cx;
                        for (int curr : grid[idx]) {
                            float dx = px - target[curr].x;
                            float dy = py - target[curr].y;
                            float d2 = dx * dx + dy * dy;
                            if (d2 < min_d2) {
                                min_d2 = d2;
                            }
                            if (min_d2 <= max_min_d2) goto EARLY_PRUNE;
                        }
                    }
                }
            }

            float box_min_x = min_x + cx_min * cell_w;
            float box_max_x = min_x + (cx_max + 1) * cell_w;
            float box_min_y = min_y + cy_min * cell_h;
            float box_max_y = min_y + (cy_max + 1) * cell_h;

            float dx_box = std::max({ 0.0f, box_min_x - px, px - box_max_x });
            float dy_box = std::max({ 0.0f, box_min_y - py, py - box_max_y });

            if (dx_box * dx_box + dy_box * dy_box >= min_d2) {
                break;
            }

            r++;
        }
    EARLY_PRUNE:
        if (min_d2 > max_min_d2) {
            max_min_d2 = min_d2;
        }
    }
    return max_min_d2;
}

float launch_hausdorff_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, int m) {
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int k = 0; k < num_t; k++) {
        // 分离偏移量
        int offset1 = k * n;
        int offset2 = k * m;

        const Point* T1 = &h_t1[offset1];
        const Point* T2 = &h_t2[offset2];

        // T1 到 T2：源长度为 n，目标长度为 m
        float d1 = compute_directed_hausdorff(T1, n, T2, m, 0.0f);
        // T2 到 T1：源长度为 m，目标长度为 n (传入之前计算出的 d1 进行剪枝优化)
        float d2 = compute_directed_hausdorff(T2, m, T1, n, d1);

        h_results[k] = sqrtf(d2); // 最终结果开方
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}