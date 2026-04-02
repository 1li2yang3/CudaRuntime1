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

// Boost.Geometry 命名空间别名与类型定义
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
// 定义二维笛卡尔坐标系下的点类型
typedef bg::model::point<float, 2, bg::cs::cartesian> BgPoint;
// 定义由上述点构成的矩形框（Box）类型
typedef bg::model::box<BgPoint> BgBox;
// R树存储的键值对类型：<轨迹的最小外接矩形, 轨迹的索引ID>
typedef std::pair<BgBox, int> RTreeValue;

/**
 * 辅助函数：计算两条轨迹（点序列）之间的精确平方欧氏距离
 * @param t1_pts, t2_pts 分别指向两条轨迹的点数组指针
 * @param n 轨迹包含的点数
 * @param current_min_sq 当前已知的最小距离的平方（用于提前终止优化）
 */
inline double calc_exact_dist_sq(const Point* t1_pts, const Point* t2_pts, int n, double current_min_sq) {
    double sum_sq = 0.0;
    for (int j = 0; j < n; j++) {
        double dx = (double)t1_pts[j].x - (double)t2_pts[j].x;
        double dy = (double)t1_pts[j].y - (double)t2_pts[j].y;
        sum_sq += std::pow(dx, 2) + std::pow(dy, 2);
        
        // 剪枝优化1 (Early Exit)：
        // 如果在累加过程中，部分距离的平方已经大于或等于当前的最小距离平方，
        // 说明这条轨迹不可能成为新的最近轨迹，直接提前返回，不再计算后续的点。
        if (sum_sq >= current_min_sq) {
            return sum_sq;
        }
    }
    return sum_sq;
}

/**
 * 主函数：批量计算集合 h_t1 中的轨迹到集合 h_t2 中轨迹的最短距离
 * @param h_t1, h_t2 轨迹集合1和2（展平的一维数组，每条轨迹连续存放）
 * @param h_results 存放结果的数组（存储 t1 中每条轨迹到 t2 的最短距离）
 * @param num_t 每个集合中包含的轨迹总数
 * @param n 每条轨迹包含的点数
 */
float launch_euclidean_batch_cpu_rtree(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n) {
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    
    // 存储 h_t1 的最小外接矩形 (MBR - Minimum Bounding Rectangle)
    std::vector<BgBox> mbrs_t1(num_t);
    // 存储 h_t2 的 MBR 及其对应的索引，用于构建 R 树
    std::vector<RTreeValue> rtree_data_t2(num_t);

    // 第一步：多线程并行计算 h_t1 中所有轨迹的 MBR (最小外接矩形)
#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        // 初始化外接矩形的边界为轨迹的第一个点
        float min_x = h_t1[i * n].x, max_x = h_t1[i * n].x;
        float min_y = h_t1[i * n].y, max_y = h_t1[i * n].y;
        // 遍历轨迹的其余点，扩展边界
        for (int j = 1; j < n; j++) {
            min_x = std::min(min_x, h_t1[i * n + j].x);
            max_x = std::max(max_x, h_t1[i * n + j].x);
            min_y = std::min(min_y, h_t1[i * n + j].y);
            max_y = std::max(max_y, h_t1[i * n + j].y);
        }
        mbrs_t1[i] = BgBox(BgPoint(min_x, min_y), BgPoint(max_x, max_y));
    }

    // 第二步：多线程并行计算 h_t2 中所有轨迹的 MBR，并与其索引打包
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

    // 第三步：使用 h_t2 的 MBR 数据构建 R 树 (采用 quadratic 分裂算法，节点容量设为16)
    // R树可以极大地加速空间最近邻查询
    bgi::rtree<RTreeValue, bgi::quadratic<16>> rtree(rtree_data_t2.begin(), rtree_data_t2.end());

    // 第四步：多线程并行查询 R 树（修改版：仅寻找最近的10个候选者）
#pragma omp parallel for
    for (int i = 0; i < num_t; i++) {
        double min_exact_dist_sq = std::numeric_limits<double>::max();
        int offset1 = i * n;

        // 使用 std::min 是为了防止总轨迹数(num_t)本身还不到10条时发生越界错误
        unsigned int k_candidates = std::min(50, num_t);

        // R树会直接替你筛选出 MBR 距离最近的 10 个结果
        auto query_it = rtree.qbegin(bgi::nearest(mbrs_t1[i], k_candidates));

        for (; query_it != rtree.qend(); ++query_it) {
            double mbr_dist_sq = bg::comparable_distance(mbrs_t1[i], query_it->first);

            // 【注意】：即使限制了 10 个，这个剪枝依然强烈建议保留！
            // 因为在算第 3 个候选者时，如果它的 MBR 距离已经大于前 2 个算出来的真实最短距离，
            // 那么剩下的第 4 到 10 个候选者就绝对不可能更近了，直接 break 可以省下这部分的计算。
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

    // 记录结束时间并计算耗时 (毫秒)
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;

    return duration.count();
}