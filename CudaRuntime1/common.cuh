#ifndef COMMON_CUH
#define COMMON_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 轨迹点结构体
struct Point {
    float x, y;
};

// 专业的错误检查宏
#define CHECK(call)                                   \
do {                                                  \
    const cudaError_t error = call;                   \
    if (error != cudaSuccess) {                       \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                      \
    }                                                 \
} while(0)

#endif