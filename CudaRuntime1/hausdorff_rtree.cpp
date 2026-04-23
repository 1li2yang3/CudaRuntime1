#include "hausdorff.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<float, 2, bg::cs::cartesian> bg_point;
typedef bg::model::box<bg_point> bg_box;
typedef std::pair<bg_box, int> rtree_value;

// 提取单条轨迹的 MBR
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

// 供主程序调用的 Pipeline 入口
float run_hausdorff_rtree_gpu_pipeline(const Point* h_t1, int num_t1, int n,
    const Point* h_t2, int num_t2, int m,
    float* h_results, int top_k)
{
    auto start_cpu = std::chrono::high_resolution_clock::now();

    // 1. 构建 h_t2 的 R-Tree
    std::vector<rtree_value> rtree_entries;
    rtree_entries.reserve(num_t2);
    for (int i = 0; i < num_t2; i++) {
        const Point* T2 = &h_t2[i * m];
        bg_box mbr = get_trajectory_mbr(T2, m);
        rtree_entries.push_back(std::make_pair(mbr, i));
    }
    bgi::rtree<rtree_value, bgi::quadratic<16>> rtree(rtree_entries);

    // 2. 查询 h_t1 每个轨迹的 top_k，存入平铺的一维数组
    // 数组大小为 num_t1 * top_k
    std::vector<int> candidate_indices(num_t1 * top_k, 0);

#pragma omp parallel for
    for (int i = 0; i < num_t1; i++) {
        const Point* T1 = &h_t1[i * n];
        bg_box mbr1 = get_trajectory_mbr(T1, n);

        std::vector<rtree_value> candidates;
        rtree.query(bgi::nearest(mbr1, top_k), std::back_inserter(candidates));

        // 将结果写入平铺数组
        for (int k = 0; k < candidates.size() && k < top_k; k++) {
            candidate_indices[i * top_k + k] = candidates[k].second;
        }

        // 如果候选数不足 top_k（极端情况），用最后一个凑数防止越界
        for (int k = candidates.size(); k < top_k; k++) {
            candidate_indices[i * top_k + k] = candidates.empty() ? 0 : candidates.back().second;
        }
    }

    
    
    float gpu_time = 0.0f;

    launch_hausdorff_rtree_gpu(h_t1, num_t1, n,h_t2, num_t2, m,candidate_indices.data(), top_k,h_results, gpu_time);

    auto stop_cpu = std::chrono::high_resolution_clock::now();

    float rtree_time = std::chrono::duration<float, std::milli>(stop_cpu - start_cpu).count();
    std::cout << "\n计算时间占比: " << gpu_time / rtree_time;

    return rtree_time;
}