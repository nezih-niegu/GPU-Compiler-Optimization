// --- Auto-injected helpers (run_pipeline.sh) ---
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// constexpr dims (no macros)
static constexpr int N = 100;
static constexpr int K = 50;
static constexpr int M = 200;

static inline void init_random_float(float* x, int n, uint32_t seed=123456789u) {
  uint32_t s = seed;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; // xorshift32
    x[i] = (s & 0x00FFFFFF) / float(0x01000000); // [0,1)
  }
}

static inline void print_summary(const float* c, int n) {
  double sum = 0.0, sumabs = 0.0;
  int take = n < 10 ? n : 10;
  for (int i = 0; i < n; ++i) { double v=c[i]; sum += v; sumabs += (v<0?-v:v); }
  std::printf("[result] C size=%d\n", n);
  std::printf("[result] C first %d: ", take);
  for (int i = 0; i < take; ++i) std::printf("%g%s", c[i], (i+1==take?"\n":" "));
  std::printf("[result] checksum sum=%0.6f sumabs=%0.6f\n", sum, sumabs);
}
/* Auto-generated CUDA code for tensor operations */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define THREADS_PER_BLOCK 256

__global__ void matmul_kernel_4(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    int size = 1024 * 1024; /* Example size */
    
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));
    
    /* Launch kernels */
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matmul_kernel_4<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    
    cudaDeviceSynchronize();
    
    /* Free memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}

/* End of generated CUDA code */
