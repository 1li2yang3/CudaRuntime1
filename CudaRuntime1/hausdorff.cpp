#include "hausdorff.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>  

inline float compute_directed_hausdorff(const Point* source, const Point* target, int n, float current_max_min_d2) {
    float min_x = target[0].x, max_x = target[0].x;
    float min_y = target[0].y, max_y = target[0].y;
    for (int i = 1; i < n; i++) {
        min_x = std::min(min_x, target[i].x); max_x = std::max(max_x, target[i].x);
        min_y = std::min(min_y, target[i].y); max_y = std::max(max_y, target[i].y);
    }
    int G = std::max(1, (int)std::sqrt(n));//网格数与边长
    float cell_w = (max_x - min_x) / G;
    float cell_h = (max_y - min_y) / G;
    if (cell_w < 1e-6f) cell_w = 1e-6f;
    if (cell_h < 1e-6f) cell_h = 1e-6f;

	std::vector<std::vector<int>> grid(G * G);//网格索引存储target点的索引，共有G*G个网格

    for (int i = 0; i < n; i++) {
        int cx = std::max(0, std::min(G - 1, (int)((target[i].x - min_x) / cell_w)));
        int cy = std::max(0, std::min(G - 1, (int)((target[i].y - min_y) / cell_h)));
        int idx = cy * G + cx;
        grid[idx].push_back(i); 
    }

    float max_min_d2 = current_max_min_d2;

    for (int i = 0; i < n; i++) {
        float px = source[i].x;
        float py = source[i].y;
        float min_d2 = 1e20f;

        int cx0 = std::max(0, std::min(G - 1, (int)((px - min_x) / cell_w)));
        int cy0 = std::max(0, std::min(G - 1, (int)((py - min_y) / cell_h)));

        int r = 0;
		while (r <= G) {//以(cx0, cy0)为中心，半径为r
            int cx_min = cx0 - r;
            int cx_max = cx0 + r;
            int cy_min = cy0 - r;
            int cy_max = cy0 + r;

            for (int cy = cy_min; cy <= cy_max; cy++) {
                for (int cx = cx_min; cx <= cx_max; cx++) {

					if (r > 0 && !(cy == cy_min || cy == cy_max || cx == cx_min || cx == cx_max)) {//如果不是边界网格，跳过
                        continue;
                    }

                    if (cx >= 0 && cx < G && cy >= 0 && cy < G) {

						float c_min_x = min_x + cx * cell_w;//当前网格边界
                        float c_max_x = c_min_x + cell_w;
                        float c_min_y = min_y + cy * cell_h;
                        float c_max_y = c_min_y + cell_h;

						float dx_cell = std::max({ 0.0f, c_min_x - px, px - c_max_x });//如果点在网格外，计算点到网格四边的最短距离
						float dy_cell = std::max({ 0.0f, c_min_y - py, py - c_max_y });//如果点在网格内，距离为0
						if (dx_cell * dx_cell + dy_cell * dy_cell >= min_d2) {//如果点到网格的最短距离已经大于当前最小距离，跳过
                            continue;
                        }

                        int idx = cy * G + cx;
						for (int curr : grid[idx]) {//遍历网格中的点
                            float dx = px - target[curr].x;
                            float dy = py - target[curr].y;
                            float d2 = dx * dx + dy * dy;
							if (d2 < min_d2) {//更新最小距离
                                min_d2 = d2;
                            }
							if (min_d2 <= max_min_d2) goto EARLY_PRUNE;//goto语句，如果最小距离小于等于全局最大最小距离，跳到下方
                        }
                    }
                }
            }

			float box_min_x = min_x + cx_min * cell_w;//当前半径外的边界框
            float box_max_x = min_x + (cx_max + 1) * cell_w;
            float box_min_y = min_y + cy_min * cell_h;
            float box_max_y = min_y + (cy_max + 1) * cell_h;

			float dx_box = std::max({ 0.0f, box_min_x - px, px - box_max_x });//计算点到边界框的最短距离
            float dy_box = std::max({ 0.0f, box_min_y - py, py - box_max_y });
			if (dx_box * dx_box + dy_box * dy_box >= min_d2) {//如果点到边界框的最短距离已经大于当前最小距离，则更外圈的网格都可以跳过
                break;
            }

            r++;
        }
	EARLY_PRUNE://跳过当前点，继续下一个
        if (min_d2 > max_min_d2) {
            max_min_d2 = min_d2;
        }
    }
    return max_min_d2;
}

float launch_hausdorff_batch_cpu(const Point* h_t1, const Point* h_t2, float* h_results, int num_t, int n) {
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for

    for (int k = 0; k < num_t; k++) {
        int offset = k * n;
        const Point* T1 = &h_t1[offset];
        const Point* T2 = &h_t2[offset];

        float d1 = compute_directed_hausdorff(T1, T2, n, 0.0f);
        float d2 = compute_directed_hausdorff(T2, T1, n, d1);
        h_results[k] = sqrtf(d2);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}