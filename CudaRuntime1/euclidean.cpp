#include "euclidean.cuh"
#include <cmath>
#include <chrono>
#include <omp.h> 
#include <vector>
#include <limits> // 引入 limits 以使用 numeric_limits

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