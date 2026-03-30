#include "euclidean.cuh"
#include <cmath>
#include <chrono>
#include <omp.h> 
#include <vector>
#include <limits>
#include <queue>
#include <algorithm>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>

float launch_euclidean_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n) {
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        double sum = 0.0; 
        int offset = i * n;
        for (int j = 0; j < n; j++) {
            double dx = (double)h_t1[offset + j].x - (double)h_t2[offset + j].x;
            double dy = (double)h_t1[offset + j].y - (double)h_t2[offset + j].y;
            sum += std::pow(dx, 2) + std::pow(dy, 2);
        }
        h_results[i] = (float)sqrt(sum); 
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;

    return duration.count(); 
}




float launch_euclidean_batch_cpu_2(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    // OpenMP 并行化最外层循环，因为每个 T1 轨迹的计算是完全独立的
#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        // 初始化最小平方距离为一个极大值
        double min_dist_sq = std::numeric_limits<double>::max();
        int offset1 = i * n; // T1 当前轨迹的偏移量

        // 遍历 T2 中的所有轨迹
        for (int k = 0; k < num_t; k++) {
            double sum_sq = 0.0;
            int offset2 = k * n; // T2 当前遍历轨迹的偏移量

            // 计算 T1[i] 和 T2[k] 之间的距离平方和
            for (int j = 0; j < n; j++) {
                double dx = (double)h_t1[offset1 + j].x - (double)h_t2[offset2 + j].x;
                double dy = (double)h_t1[offset1 + j].y - (double)h_t2[offset2 + j].y;
                // 优化：直接使用乘法替代 std::pow(..., 2)，大幅提升性能
                //sum_sq += (dx * dx) + (dy * dy);
                sum_sq += std::pow(dx, 2) + std::pow(dy, 2);
            }

            // 更新最小距离的平方
            if (sum_sq < min_dist_sq) {
                min_dist_sq = sum_sq;
            }
        }

        // 优化：只需在最后找到最小值时，做一次开方操作
        h_results[i] = (float)std::sqrt(min_dist_sq);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;

    return duration.count();
}


namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
typedef bg::model::point<float, 2, bg::cs::cartesian> BgPoint;
typedef bg::model::box<BgPoint> BgBox;
typedef std::pair<BgBox, int> RTreeValue;
inline double calc_exact_dist_sq(const Point* t1_pts, const Point* t2_pts, int n, double current_min_sq) {
    double sum_sq = 0.0;
    for (int j = 0; j < n; j++) {
        double dx = (double)t1_pts[j].x - (double)t2_pts[j].x;
        double dy = (double)t1_pts[j].y - (double)t2_pts[j].y;
        sum_sq += std::pow(dx, 2) + std::pow(dy, 2);
        if (sum_sq >= current_min_sq) {
            return sum_sq;
        }
    }
    return sum_sq;
}

float launch_euclidean_batch_cpu_rtree(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<BgBox> mbrs_t1(num_t);
    std::vector<RTreeValue> rtree_data_t2(num_t);

#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        float min_x = h_t1[i * n].x, max_x = h_t1[i * n].x;
        float min_y = h_t1[i * n].y, max_y = h_t1[i * n].y;
        for (int j = 1; j < n; j++) {
            min_x = std::min(min_x, h_t1[i * n + j].x);
            max_x = std::max(max_x, h_t1[i * n + j].x);
            min_y = std::min(min_y, h_t1[i * n + j].y);
            max_y = std::max(max_y, h_t1[i * n + j].y);
        }
        mbrs_t1[i] = BgBox(BgPoint(min_x, min_y), BgPoint(max_x, max_y));
    }

#pragma omp parallel for
    for (int k = 0; k < num_t; k++) {
        float min_x = h_t2[k * n].x, max_x = h_t2[k * n].x;
        float min_y = h_t2[k * n].y, max_y = h_t2[k * n].y;
        for (int j = 1; j < n; j++) {
            min_x = std::min(min_x, h_t2[k * n + j].x);
            max_x = std::max(max_x, h_t2[k * n + j].x);
            min_y = std::min(min_y, h_t2[k * n + j].y);
            max_y = std::max(max_y, h_t2[k * n + j].y);
        }
        rtree_data_t2[k] = std::make_pair(BgBox(BgPoint(min_x, min_y), BgPoint(max_x, max_y)), k);
    }


    bgi::rtree<RTreeValue, bgi::quadratic<16>> rtree(rtree_data_t2.begin(), rtree_data_t2.end());

#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        double min_exact_dist_sq = std::numeric_limits<double>::max();
        int offset1 = i * n;


        auto query_it = rtree.qbegin(bgi::nearest(mbrs_t1[i], num_t));

        for (; query_it != rtree.qend(); ++query_it) {
            double mbr_dist_sq = bg::comparable_distance(mbrs_t1[i], query_it->first);
            if (mbr_dist_sq >= min_exact_dist_sq) {
                break;
            }
            int k = query_it->second;
            int offset2 = k * n;
            double exact_sum_sq = calc_exact_dist_sq(&h_t1[offset1], &h_t2[offset2], n, min_exact_dist_sq);
            if (exact_sum_sq < min_exact_dist_sq) {
                min_exact_dist_sq = exact_sum_sq;
            }
        }

        h_results[i] = (float)std::sqrt(min_exact_dist_sq);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;

    return duration.count();
}



