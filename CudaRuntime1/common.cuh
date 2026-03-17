#ifndef COMMON_CUH
#define COMMON_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


struct Point {
    float x, y;
};


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