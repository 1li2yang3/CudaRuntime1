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


void launch_euclidean_batch_gpu_rtree_exact(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n, float& gpu_time) {
    auto start_all = std::chrono::high_resolution_clock::now();

    std::vector<BgBox> mbrs_t1(num_t);
    std::vector<RTreeValue> rtree_data_t2(num_t);

    // 计算 t1 的 MBR
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

	// 计算 t2 的 MBR 和索引
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

    // 直接使用 KNN 查询获取最近的 k 个候选者
#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        // 防止数据总量不到 k 个导致越界
        unsigned int k_candidates = std::min(50, num_t);

        // 预分配空间，提高 vector 性能
        all_candidates[i].reserve(k_candidates);

        // R 树直接查询 MBR 距离最近的 k 个候选者
        auto query_it = rtree.qbegin(bgi::nearest(mbrs_t1[i], k_candidates));

        for (; query_it != rtree.qend(); ++query_it) {
            // 直接把捞出来的轨迹索引塞进候选名单
            all_candidates[i].push_back(query_it->second);
        }
    }

    // 将嵌套的 candidates 数组展平为 CSR 格式，准备传给 GPU
    std::vector<int> h_offsets(num_t + 1, 0);
    for (int i = 0; i < num_t; i++) {
        h_offsets[i + 1] = h_offsets[i] + static_cast<int>(all_candidates[i].size());
    }

    int total_candidates = h_offsets.back();
    std::vector<int> h_flat_candidates(total_candidates);

#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        std::copy(all_candidates[i].begin(), all_candidates[i].end(), h_flat_candidates.begin() + h_offsets[i]);
    }

    // 调用 GPU Kernel，计算这 k 个候选者的真实欧式距离并取最小值
    execute_rtree_csr_kernel_on_gpu(h_t1, h_t2, h_flat_candidates.data(), h_offsets.data(), h_results, num_t, n, total_candidates, gpu_time);

    auto stop_all = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop_all - start_all;

    std::cout << "计算时间占比: " << gpu_time / duration.count();
	gpu_time = duration.count();// 返回总耗时
}