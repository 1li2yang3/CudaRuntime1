#include "hausdorff.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>  
#include <iostream>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

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





namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

// 定义 Boost 中适配 R-tree 的数据结构
typedef bg::model::point<float, 2, bg::cs::cartesian> bg_point;
typedef bg::model::box<bg_point> bg_box;
// R-tree 中存储的元素对：<轨迹边界框, 轨迹在 h_t2 中的索引>
typedef std::pair<bg_box, int> rtree_value;


// 提取单条轨迹的边界框 (MBR)
inline bg_box get_trajectory_mbr(const Point* traj, int len) {
    float min_x = traj[0].x, max_x = traj[0].x;
    float min_y = traj[0].y, max_y = traj[0].y;
    for (int i = 1; i < len; ++i) {
        min_x = std::min(min_x, traj[i].x);
        max_x = std::max(max_x, traj[i].x);
        min_y = std::min(min_y, traj[i].y);
        max_y = std::max(max_y, traj[i].y);
    }
    return bg_box(bg_point(min_x, min_y), bg_point(max_x, max_y));
}


float launch_hausdorff_batch_cpu_rtree(const Point* h_t1, int num_t1, int n,
    const Point* h_t2, int num_t2, int m,
    float* h_results, int top_k ) {
    auto start = std::chrono::high_resolution_clock::now();

    // 1. 预处理：构建 h_t2 的 R-Tree
    std::vector<rtree_value> rtree_entries;
    rtree_entries.reserve(num_t2);
    for (int i = 0; i < num_t2; i++) {
        const Point* T2 = &h_t2[i * m];
        bg_box mbr = get_trajectory_mbr(T2, m);
        rtree_entries.push_back(std::make_pair(mbr, i));
    }

    // 使用 quadratic 算法构建 R-tree（批量插入性能更好）
    bgi::rtree<rtree_value, bgi::quadratic<16>> rtree(rtree_entries);

    // 2. 并行查询与精确计算
#pragma omp parallel for
    for (int i = 0; i < num_t1; i++) {
        const Point* T1 = &h_t1[i * n];
        bg_box mbr1 = get_trajectory_mbr(T1, n);

        // 使用 R-Tree 查出距离 T1 的 MBR 最近的 top_k 个 T2 轨迹
        std::vector<rtree_value> candidates;
        rtree.query(bgi::nearest(mbr1, top_k), std::back_inserter(candidates));

        float best_hausdorff_d2 = 1e20f; // 初始化为极大值，记录真实的最小距离平方

        // 在这 10 个候选者中进行精确的 Hausdorff 计算
        for (const auto& candidate : candidates) {
            int target_idx = candidate.second;
            const Point* T2_candidate = &h_t2[target_idx * m];

            // 复用你原来的核心算法
            float d1 = compute_directed_hausdorff(T1, n, T2_candidate, m, 0.0f);
            float d2 = compute_directed_hausdorff(T2_candidate, m, T1, n, d1);

            // 更新找到的最相似轨迹（距离越小越相似）
            if (d2 < best_hausdorff_d2) {
                best_hausdorff_d2 = d2;
            }
        }

        // 保存最接近的一条轨迹的真实距离开方
        h_results[i] = sqrtf(best_hausdorff_d2);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}