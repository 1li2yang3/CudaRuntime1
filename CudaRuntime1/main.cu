#include <iostream>
#include <vector>
#include <iomanip>
#include "euclidean.cuh"
#include "hausdorff.cuh"
#include "dtw.cuh"
#include "lcss.cuh"
#include "frechet.cuh"
#include <thread>

//void generate_random_points(std::vector<Point>& host_vec) {
//
//    for (auto& p : host_vec) {
//        p.x = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0f));
//        p.y = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0f));
//    }
//}
void generate_random_points(std::vector<Point>& host_vec) {
    if (host_vec.empty()) return;

    // 1. 生成第一个点 (在 0 到 100 内随机)
    host_vec[0].x = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0f));
    host_vec[0].y = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0f));

    // 2. 从第二个点开始，基于前一个点进行偏移
    for (size_t i = 1; i < host_vec.size(); ++i) {
        // 生成 [-0.5, 0.5] 范围内的随机偏移量
        // (rand() / RAND_MAX) 生成 0~1 的数，乘以 1.0 然后减去 0.5 得到 -0.5~0.5
        float dx = (static_cast<float>(rand()) / RAND_MAX) * 1.0f - 0.5f;
        float dy = (static_cast<float>(rand()) / RAND_MAX) * 1.0f - 0.5f;

        // 计算新坐标
        float new_x = host_vec[i - 1].x + dx;
        float new_y = host_vec[i - 1].y + dy;

        // 3. 边界处理：确保点始终在 [0, 100] 的范围内 (Clamp操作)
        host_vec[i].x = std::max(0.0f, std::min(100.0f, new_x));
        host_vec[i].y = std::max(0.0f, std::min(100.0f, new_y));
    }
}

void generate_base_trajectories(std::vector<Point>& t1_batch, int num_t, int n) {
    for (int k = 0; k < num_t; ++k) {
        int offset = k * n;

        // 让所有轨迹的起点都在地图中心附近，避免一开始就飞出边界
        t1_batch[offset].x = 50.0f;
        t1_batch[offset].y = 50.0f;

        for (int i = 1; i < n; ++i) {
            // 每次最多移动 [-1.0, 1.0] 的距离
            float dx = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
            float dy = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;

            float new_x = t1_batch[offset + i - 1].x + dx;
            float new_y = t1_batch[offset + i - 1].y + dy;

            // 限制在 0-100 范围内
            t1_batch[offset + i].x = std::max(0.0f, std::min(100.0f, new_x));
            t1_batch[offset + i].y = std::max(0.0f, std::min(100.0f, new_y));
        }
    }
}
void generate_similar_trajectories_nm(const std::vector<Point>& t1_batch, std::vector<Point>& t2_batch,
    int num_t, int n, int m, float noise_max) {
    for (int k = 0; k < num_t; ++k) {
        int offset1 = k * n; // t1 的偏移量按 n 计算
        int offset2 = k * m; // t2 的偏移量按 m 计算

        for (int j = 0; j < m; ++j) {
            // 核心逻辑：将 t2 的索引 j (0 到 m-1) 映射到 t1 的索引 src_idx (0 到 n-1)
            int src_idx = 0;
            if (m > 1) {
                // 按比例均匀映射，并四舍五入找最近的基准点
                src_idx = static_cast<int>(std::round(static_cast<float>(j) * (n - 1) / (m - 1)));
            }

            // 安全防御：确保索引不越界
            src_idx = std::max(0, std::min(n - 1, src_idx));

            // 取出对应的基准点坐标
            float base_x = t1_batch[offset1 + src_idx].x;
            float base_y = t1_batch[offset1 + src_idx].y;

            // 生成 [-noise_max, noise_max] 的随机噪声
            float noise_x = (static_cast<float>(rand()) / RAND_MAX) * noise_max * 2.0f - noise_max;
            float noise_y = (static_cast<float>(rand()) / RAND_MAX) * noise_max * 2.0f - noise_max;

            // 加上噪声，并限制在 0-100 的地图边界内
            t2_batch[offset2 + j].x = std::max(0.0f, std::min(100.0f, base_x + noise_x));
            t2_batch[offset2 + j].y = std::max(0.0f, std::min(100.0f, base_y + noise_y));
        }
    }
}


void test_euclidean() {
    const int num_t = 5000;
    const int n = 5000;
    std::vector<Point> h_t1(num_t * n);
    std::vector<Point> h_t2(num_t * n);
    std::vector<float> gpu_results(num_t);
    std::vector<float> cpu_results(num_t);
    /*generate_random_points(h_t1);
    generate_random_points(h_t2);*/
    generate_base_trajectories(h_t1, num_t, n);
    float noise_max = 2.0f;
    generate_similar_trajectories_nm(h_t1, h_t2, num_t, n, n, noise_max);

    float cpu_time_ms = launch_euclidean_batch_cpu(h_t1.data(), h_t2.data(), cpu_results.data(), num_t, n);
    float gpu_time_ms = 0;
    launch_euclidean_batch_gpu(h_t1.data(), h_t2.data(), gpu_results.data(), num_t, n, gpu_time_ms);
    
    std::cout << "\n[ Euclidean Distance Experiment ]" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CPU Execution Time : " << cpu_time_ms << " ms" << std::endl;
    std::cout << "GPU Execution Time : " << gpu_time_ms << " ms" << std::endl;
    std::cout << "SPEEDUP: " << cpu_time_ms / gpu_time_ms << "x" << std::endl;
    bool pass = true;
    for (int i = 0; i < num_t; i++) {
		if (abs(cpu_results[i] - gpu_results[i]) > 1e-3) {
            pass = false;
            std::cout << cpu_results[i] << "--" << gpu_results[i] << std::endl;
            break;
        }
    }
    std::cout << "Integrity Check: " << (pass ? "PASS " : "FAIL ") << std::endl;
    std::cout << cpu_results[10] << "--" << gpu_results[10] << std::endl;
}

void test_euclidean_2() {
    const int num_t = 1200;
    const int n = 1200;
    std::vector<Point> h_t1(num_t * n);
    std::vector<Point> h_t2(num_t * n);
    std::vector<float> gpu_results(num_t);
    std::vector<float> cpu_results(num_t);
    /*generate_random_points(h_t1);
    generate_random_points(h_t2);*/
    generate_base_trajectories(h_t1, num_t, n);
    float noise_max = 2.0f;
    generate_similar_trajectories_nm(h_t1, h_t2, num_t, n, n, noise_max);

    float cpu_time_ms = launch_euclidean_batch_cpu_rtree(h_t1.data(), h_t2.data(), cpu_results.data(), num_t, n);
    float gpu_time_ms = 0;
    launch_euclidean_batch_gpu_rtree_exact(h_t1.data(), h_t2.data(), gpu_results.data(), num_t, n, gpu_time_ms);

    std::cout << "\n[ Euclidean Distance Experiment ]" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CPU Execution Time : " << cpu_time_ms << " ms" << std::endl;
    std::cout << "GPU Execution Time : " << gpu_time_ms << " ms" << std::endl;
    std::cout << "SPEEDUP: " << cpu_time_ms / gpu_time_ms << "x" << std::endl;
    bool pass = true;
    for (int i = 0; i < num_t; i++) {
        if (abs(cpu_results[i] - gpu_results[i]) > 1e-3) {
            pass = false;
            std::cout << cpu_results[i] << "--" << gpu_results[i] << std::endl;
            break;
        }
    }
    std::cout << "Integrity Check: " << (pass ? "PASS " : "FAIL ") << std::endl;
    std::cout << cpu_results[10] << "--" << gpu_results[10] << std::endl;

}

void test_hausdorff() {
    const int num_t = 1000;  
    const int n = 1900;     
    const int m = 2000;
    std::vector<Point> h_t1(num_t * n);
    std::vector<Point> h_t2(num_t * m);
    std::vector<float> gpu_results(num_t);
    std::vector<float> cpu_results(num_t);
    /*generate_random_points(h_t1);
    generate_random_points(h_t2);*/
    generate_base_trajectories(h_t1, num_t, n);
    float noise_max = 2.0f;
    generate_similar_trajectories_nm(h_t1, h_t2, num_t, n, m, noise_max);
    
    float cpu_time_ms_grid = launch_hausdorff_batch_cpu(h_t1.data(), h_t2.data(), cpu_results.data(), num_t, n, m);
    float gpu_time_ms = 0;
    launch_hausdorff_batch_gpu(h_t1.data(), h_t2.data(), gpu_results.data(), num_t, n, m, gpu_time_ms);
    
    std::cout << "\n[ Hausdorff Distance Experiment ]" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CPU Time: " << cpu_time_ms_grid << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "SPEEDUP: " << cpu_time_ms_grid / gpu_time_ms << "x" << std::endl;

    bool pass = true;
    for (int i = 0; i < num_t; i++) {
        if (abs(cpu_results[i] - gpu_results[i]) > 1e-5) {
            pass = false;
            std::cout << cpu_results[i] << "--" << gpu_results[i] << std::endl;
            break;
        }
    }
    std::cout << "Integrity Check: " << (pass ? "PASS " : "FAIL ") << std::endl;
    std::cout <<cpu_results[10] << "--" << gpu_results[10] << std::endl;
}

void test_dtw() {
    const int num_t = 400; 
    const int n = 800;    
    const int m = 1000;
    std::vector<Point> h_t1(num_t * n);
    std::vector<Point> h_t2(num_t * m);
    std::vector<float> gpu_results(num_t);
    std::vector<float> cpu_results(num_t);
    /*generate_random_points(h_t1);
    generate_random_points(h_t2);*/
    generate_base_trajectories(h_t1, num_t, n);
    float noise_max = 2.0f;
    generate_similar_trajectories_nm(h_t1, h_t2, num_t, n, m, noise_max);

    float cpu_time = launch_dtw_batch_cpu(h_t1.data(), h_t2.data(), cpu_results.data(), num_t, n, m);
    float gpu_time = 0;
    launch_dtw_batch_gpu(h_t1.data(), h_t2.data(), gpu_results.data(), num_t, n, m, gpu_time);
    
    std::cout << "\n[ DTW Distance Experiment ]" << std::endl;
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;;
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;
    std::cout << "SPEEDUP: " << cpu_time / gpu_time << "x" << std::endl;

    bool pass = true;
    for (int i = 0; i < num_t; i++) {
        if (abs(cpu_results[i] - gpu_results[i]) > 1e-2) {
            pass = false;
            std::cout << cpu_results[i] << "--" << gpu_results[i] << std::endl;
            break;
        }
    }
    
    std::cout << "Integrity Check: " << (pass ? "PASS " : "FAIL ") << std::endl;
    std::cout << cpu_results[10] << "--" << gpu_results[10] << std::endl;
}


void test_lcss() {
    const int num_t = 1000; 
    const int n = 1000;       
    const int m = 1200;
    const float epsilon = 0.5f; 
    std::vector<Point> h_t1(num_t * n);
    std::vector<Point> h_t2(num_t * m);
    std::vector<float> gpu_results(num_t);
    std::vector<float> cpu_results(num_t);
    /*generate_random_points(h_t1);
    generate_random_points(h_t2);*/
    generate_base_trajectories(h_t1, num_t, n);
    float noise_max = 2.0f;
    generate_similar_trajectories_nm(h_t1, h_t2, num_t, n, m, noise_max);

    float cpu_time = launch_lcss_batch_cpu(h_t1.data(), h_t2.data(), cpu_results.data(), num_t, n,m, epsilon);
    float gpu_time = 0;
    launch_lcss_batch_gpu_wavefront(h_t1.data(), h_t2.data(), gpu_results.data(), num_t, n, m, epsilon, gpu_time);

    std::cout << "\n[ LCSS Similarity Experiment ]" << std::endl;
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;
	std::cout << "SPEEDUP: " << cpu_time / gpu_time << "x" << std::endl;

    bool pass = true;
    for (int i = 0; i < num_t; i++) {
        if (abs(cpu_results[i] - gpu_results[i]) > 1e-5 ) {
            pass = false;
            std::cout << cpu_results[i] << "--" << gpu_results[i] << std::endl;
            break;
        }
    }
    std::cout << "Integrity Check: " << (pass ? "PASS " : "FAIL ") << std::endl;
    std::cout << cpu_results[10] << "--" << gpu_results[10] << std::endl;

}

void test_frechet() {
    const int num_t = 400;
    const int n = 900;
    const int m = 1100;
    std::vector<Point> h_t1(num_t * n);
    std::vector<Point> h_t2(num_t * m);
    std::vector<float> gpu_results_wavefront(num_t);
    std::vector<float> cpu_results(num_t);
    /*generate_random_points(h_t1);
    generate_random_points(h_t2);*/
    generate_base_trajectories(h_t1, num_t, n);
    float noise_max = 2.0f;
    generate_similar_trajectories_nm(h_t1, h_t2, num_t, n, m, noise_max);
    //传入参数前对轨迹长度进行判断，h1是较短的一条，n<=m
    float cpu_time = launch_frechet_batch_cpu(h_t1.data(), h_t2.data(), cpu_results.data(), num_t, n , m);
    float gpu_time_wavefront = 0;
    launch_frechet_batch_gpu_wavefront(h_t1.data(), h_t2.data(), gpu_results_wavefront.data(), num_t, n , m , gpu_time_wavefront);
    
    std::cout << "\n[ Fréchet Distance Experiment ]" << std::endl;
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time_wavefront << " ms" << std::endl;
    std::cout << "SPEEDUP: " << cpu_time / gpu_time_wavefront << "x" << std::endl;

    bool pass = true;
    for (int i = 0; i < num_t; i++) {
        if (abs(cpu_results[i] - gpu_results_wavefront[i]) > 1e-5) {
            pass = false;
            std::cout << cpu_results[i] << "--" << gpu_results_wavefront[i] << std::endl;
            break;
        }
    }
    std::cout << "Integrity Check: " << (pass ? "PASS " : "FAIL ") << std::endl;
    std::cout << cpu_results[10] << "--" << gpu_results_wavefront[10] << std::endl;
 
}

int main() {
    srand(time(NULL));

    //test_euclidean();
    test_euclidean_2();
    test_hausdorff();
    /*test_dtw();
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    test_lcss();
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    test_frechet();*/
    

    return 0;
}
//反对角线的矩阵是如下格式：从左上到右下
//  j
//i 1 1 1 1 1
//  1 1 1 1 1
//  1 1 1 1 1
//  1 1 1 1 1
//  1 1 1 1 1