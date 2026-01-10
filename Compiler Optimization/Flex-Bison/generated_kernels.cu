/* Auto-generated CUDA code for tensor operations */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define THREADS_PER_BLOCK 256

__global__ void matmul_kernel_5(float *A, float *B, float *C, int M, int N, int K) {
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
    matmul_kernel_5<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    
    cudaDeviceSynchronize();
    
    /* Free memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}

/* End of generated CUDA code */
