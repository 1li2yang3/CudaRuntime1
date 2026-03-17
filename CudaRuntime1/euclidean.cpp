#include "euclidean.cuh"
#include <cmath>
#include <chrono>
#include <omp.h> 
#include <vector>


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

