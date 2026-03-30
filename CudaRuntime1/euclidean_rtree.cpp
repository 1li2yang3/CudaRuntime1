#include "euclidean.cuh"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath> 
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
typedef bg::model::point<float, 2, bg::cs::cartesian> BgPoint;
typedef bg::model::box<BgPoint> BgBox;
typedef std::pair<BgBox, int> RTreeValue;

inline double calc_exact_dist_sq(const Point* t1_pts, const Point* t2_pts, int n) {
    double sum_sq = 0.0;
    for (int j = 0; j < n; j++) {
        double dx = (double)t1_pts[j].x - (double)t2_pts[j].x;
        double dy = (double)t1_pts[j].y - (double)t2_pts[j].y;
        sum_sq += (dx * dx) + (dy * dy);
    }
    return sum_sq;
}

void launch_euclidean_batch_gpu_rtree_exact(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time) {
    auto start_all = std::chrono::high_resolution_clock::now();

    std::vector<BgBox> mbrs_t1(num_t);
    std::vector<RTreeValue> rtree_data_t2(num_t);

#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        float min_x = h_t1[i * n].x, max_x = h_t1[i * n].x;
        float min_y = h_t1[i * n].y, max_y = h_t1[i * n].y;
        for (int j = 1; j < n; j++) {
            min_x = std::min(min_x, h_t1[i * n + j].x); max_x = std::max(max_x, h_t1[i * n + j].x);
            min_y = std::min(min_y, h_t1[i * n + j].y); max_y = std::max(max_y, h_t1[i * n + j].y);
        }
        mbrs_t1[i] = BgBox(BgPoint(min_x, min_y), BgPoint(max_x, max_y));
    }

#pragma omp parallel for
    for (int k = 0; k < num_t; k++) {
        float min_x = h_t2[k * n].x, max_x = h_t2[k * n].x;
        float min_y = h_t2[k * n].y, max_y = h_t2[k * n].y;
        for (int j = 1; j < n; j++) {
            min_x = std::min(min_x, h_t2[k * n + j].x); max_x = std::max(max_x, h_t2[k * n + j].x);
            min_y = std::min(min_y, h_t2[k * n + j].y); max_y = std::max(max_y, h_t2[k * n + j].y);
        }
        rtree_data_t2[k] = std::make_pair(BgBox(BgPoint(min_x, min_y), BgPoint(max_x, max_y)), k);
    }

    bgi::rtree<RTreeValue, bgi::quadratic<16>> rtree(rtree_data_t2.begin(), rtree_data_t2.end());

    std::vector<std::vector<int>> all_candidates(num_t);


#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        double tight_upper_bound = std::numeric_limits<double>::max();
        int offset1 = i * n;

        // 步骤 1：找最近的 1 个，定出精确卡尺 (这部分极快，保留)
        auto query_first = rtree.qbegin(bgi::nearest(mbrs_t1[i], 1));
        if (query_first != rtree.qend()) {
            int first_k = query_first->second;
            tight_upper_bound = calc_exact_dist_sq(&h_t1[offset1], &h_t2[first_k * n], n);
        }

        // 步骤 2：生成膨胀搜索框
        all_candidates[i].reserve(32); // 预分配减少内存开销
        float radius = (float)std::sqrt(tight_upper_bound);

        float min_x = mbrs_t1[i].min_corner().get<0>();
        float min_y = mbrs_t1[i].min_corner().get<1>();
        float max_x = mbrs_t1[i].max_corner().get<0>();
        float max_y = mbrs_t1[i].max_corner().get<1>();

        BgBox search_box(BgPoint(min_x - radius, min_y - radius), BgPoint(max_x + radius, max_y + radius));

        // 步骤 3：直接范围相交查询，暴打 Priority Queue
        std::vector<RTreeValue> returned_values;
        rtree.query(bgi::intersects(search_box), std::back_inserter(returned_values));

        // 步骤 4：对捞出来的框做最后一遍卡尺精筛
        for (const auto& val : returned_values) {
            double mbr_dist_sq = bg::comparable_distance(mbrs_t1[i], val.first);
            if (mbr_dist_sq <= tight_upper_bound) {
                all_candidates[i].push_back(val.second);
            }
        }
    }

    std::vector<int> h_offsets(num_t + 1, 0);
    for (int i = 0; i < num_t; i++) {
        // 修复 size_t 到 int 的警告
        h_offsets[i + 1] = h_offsets[i] + static_cast<int>(all_candidates[i].size());
    }
    int total_candidates = h_offsets.back();
    std::vector<int> h_flat_candidates(total_candidates);

#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        std::copy(all_candidates[i].begin(), all_candidates[i].end(), h_flat_candidates.begin() + h_offsets[i]);
    }

    execute_rtree_csr_kernel_on_gpu(h_t1, h_t2, h_flat_candidates.data(), h_offsets.data(), h_results, num_t, n, total_candidates, gpu_time);

    auto stop_all = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop_all - start_all;

    std::cout << "计算时间占比: " << gpu_time / duration.count() << std::endl;
    gpu_time = duration.count();
}