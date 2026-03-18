#include <iostream>
#include <vector>
#include <iomanip>
#include "euclidean.cuh"
#include "hausdorff.cuh"
#include "dtw.cuh"
#include "lcss.cuh"
#include "frechet.cuh"

void generate_random_points(std::vector<Point>& host_vec) {

    for (auto& p : host_vec) {
        p.x = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0f));
        p.y = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0f));
    }
}

void test_euclidean() {
    const int num_t = 1000;
    const int n = 1000;
    std::vector<Point> h_t1(num_t * n);
    std::vector<Point> h_t2(num_t * n);
    std::vector<float> gpu_results(num_t);
    std::vector<float> cpu_results(num_t);
    generate_random_points(h_t1);
    generate_random_points(h_t2);
   
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
            break;
        }
    }
    std::cout << "Integrity Check: " << (pass ? "PASS " : "FAIL ") << std::endl;

    std::cout << cpu_results[10] << "--" << gpu_results[10] << std::endl;

}

void test_hausdorff() {
    const int num_t = 1000;  
    const int n = 2000;      
    std::vector<Point> h_t1(num_t * n);
    std::vector<Point> h_t2(num_t * n);
    std::vector<float> gpu_results(num_t);
    std::vector<float> cpu_results(num_t);
    
    generate_random_points(h_t1);
    generate_random_points(h_t2);
    
    float cpu_time_ms_grid = launch_hausdorff_batch_cpu(h_t1.data(), h_t2.data(), cpu_results.data(), num_t, n);
    float gpu_time_ms = 0;
    launch_hausdorff_batch_gpu(h_t1.data(), h_t2.data(), gpu_results.data(), num_t, n, gpu_time_ms);
    
    std::cout << "\n[ Hausdorff Distance Experiment ]" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CPU Time: " << cpu_time_ms_grid << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "SPEEDUP: " << cpu_time_ms_grid / gpu_time_ms << "x" << std::endl;

    bool pass = true;
    for (int i = 0; i < num_t; i++) {
        if (abs(cpu_results[i] - gpu_results[i]) > 1e-5) {
            pass = false;
            break;
        }
    }
    std::cout << "Integrity Check: " << (pass ? "PASS " : "FAIL ") << std::endl;
    std::cout <<cpu_results[10] << "--" << gpu_results[10] << std::endl;
}

void test_dtw() {
    const int num_t = 800; 
    const int n = 800;    
    std::vector<Point> h_t1(num_t * n);
    std::vector<Point> h_t2(num_t * n);
    std::vector<float> gpu_results(num_t);
    std::vector<float> cpu_results(num_t);
    generate_random_points(h_t1);
    generate_random_points(h_t2);

    float cpu_time = launch_dtw_batch_cpu(h_t1.data(), h_t2.data(), cpu_results.data(), num_t, n);
    float gpu_time = 0;
    launch_dtw_batch_gpu(h_t1.data(), h_t2.data(), gpu_results.data(), num_t, n, gpu_time);
    
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
    const float epsilon = 0.5f; 
    std::vector<Point> h_t1(num_t * n);
    std::vector<Point> h_t2(num_t * n);
    std::vector<float> gpu_results(num_t);
    std::vector<float> cpu_results(num_t);
    generate_random_points(h_t1);
    generate_random_points(h_t2);

    float cpu_time = launch_lcss_batch_cpu(h_t1.data(), h_t2.data(), cpu_results.data(), num_t, n, epsilon);
    float gpu_time = 0;
    launch_lcss_batch_gpu_wavefront(h_t1.data(), h_t2.data(), gpu_results.data(), num_t, n, epsilon, gpu_time);

    std::cout << "\n[ LCSS Similarity Experiment ]" << std::endl;
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;
	std::cout << "SPEEDUP: " << cpu_time / gpu_time << "x" << std::endl;

    bool pass = true;
    for (int i = 0; i < num_t; i++) {
        if (abs(cpu_results[i] - gpu_results[i]) > 1e-5 ) {
            pass = false;
            break;
        }
    }
    std::cout << "Integrity Check: " << (pass ? "PASS " : "FAIL ") << std::endl;
    std::cout << cpu_results[10] << "--" << gpu_results[10] << std::endl;

}

void test_frechet() {
    const int num_t = 1000;
    const int n = 1000;
    std::vector<Point> h_t1(num_t * n);
    std::vector<Point> h_t2(num_t * n);
    std::vector<float> gpu_results_wavefront(num_t);
    std::vector<float> cpu_results(num_t);
    generate_random_points(h_t1);
    generate_random_points(h_t2);

    float cpu_time = launch_frechet_batch_cpu(h_t1.data(), h_t2.data(), cpu_results.data(), num_t, n);
    float gpu_time_wavefront = 0;
    launch_frechet_batch_gpu_wavefront(h_t1.data(), h_t2.data(), gpu_results_wavefront.data(), num_t, n, gpu_time_wavefront);
    
    std::cout << "\n[ Fréchet Distance Experiment ]" << std::endl;
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_time_wavefront << " ms" << std::endl;
    std::cout << "SPEEDUP: " << cpu_time / gpu_time_wavefront << "x" << std::endl;

    bool pass = true;
    for (int i = 0; i < num_t; i++) {
        if (abs(cpu_results[i] - gpu_results_wavefront[i]) > 1e-5) {
            pass = false;
            break;
        }
    }
    std::cout << "Integrity Check: " << (pass ? "PASS " : "FAIL ") << std::endl;
    std::cout << cpu_results[10] << "--" << gpu_results_wavefront[10] << std::endl;
 
}

int main() {
    srand(time(NULL));

    //test_euclidean();
    test_hausdorff();
    //test_dtw();
    //test_lcss();
    //test_frechet();
    
    ;

    return 0;
}