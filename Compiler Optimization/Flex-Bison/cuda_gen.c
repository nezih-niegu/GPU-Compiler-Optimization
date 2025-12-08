#include "cuda_gen.h"
#include <string.h>

void generate_cuda_code(TensorGraph *graph, const char *output_file) {
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", output_file);
        return;
    }
    
    generate_kernel_header(fp);
    
    /* Generate kernels for each operation */
    for (int i = 0; i < graph->num_nodes; i++) {
        GraphNode *node = graph->nodes[i];
        if (!node) continue;
        
        switch (node->op_type) {
            case OP_ADD:
                generate_elementwise_kernel(fp, node, "+");
                break;
            case OP_MUL:
                generate_elementwise_kernel(fp, node, "*");
                break;
            case OP_MATMUL:
                generate_matmul_kernel(fp, node);
                break;
            case OP_REDUCE:
                generate_reduce_kernel(fp, node);
                break;
            case OP_TRANSPOSE:
                generate_transpose_kernel(fp, node);
                break;
            default:
                break;
        }
    }
    
    generate_main_function(fp, graph);
    generate_kernel_footer(fp);
    
    fclose(fp);
}

void generate_kernel_header(FILE *fp) {
    fprintf(fp, "/* Auto-generated CUDA code for tensor operations */\n");
    fprintf(fp, "#include <cuda_runtime.h>\n");
    fprintf(fp, "#include <stdio.h>\n");
    fprintf(fp, "#include <stdlib.h>\n\n");
    fprintf(fp, "#define BLOCK_SIZE 16\n");
    fprintf(fp, "#define THREADS_PER_BLOCK 256\n\n");
}

void generate_kernel_footer(FILE *fp) {
    fprintf(fp, "\n/* End of generated CUDA code */\n");
}

void generate_matmul_kernel(FILE *fp, GraphNode *node) {
    fprintf(fp, "__global__ void matmul_kernel_%d(float *A, float *B, float *C, int M, int N, int K) {\n", 
            node->node_id);
    fprintf(fp, "    int row = blockIdx.y * blockDim.y + threadIdx.y;\n");
    fprintf(fp, "    int col = blockIdx.x * blockDim.x + threadIdx.x;\n");
    fprintf(fp, "    \n");
    fprintf(fp, "    if (row < M && col < N) {\n");
    fprintf(fp, "        float sum = 0.0f;\n");
    fprintf(fp, "        for (int k = 0; k < K; k++) {\n");
    fprintf(fp, "            sum += A[row * K + k] * B[k * N + col];\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        C[row * N + col] = sum;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "}\n\n");
}

void generate_elementwise_kernel(FILE *fp, GraphNode *node, const char *op) {
    const char *op_name = (strcmp(op, "+") == 0) ? "add" : "mul";
    fprintf(fp, "__global__ void %s_kernel_%d(float *A, float *B, float *C, int size) {\n", 
            op_name, node->node_id);
    fprintf(fp, "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
    fprintf(fp, "    if (idx < size) {\n");
    fprintf(fp, "        C[idx] = A[idx] %s B[idx];\n", op);
    fprintf(fp, "    }\n");
    fprintf(fp, "}\n\n");
}

void generate_reduce_kernel(FILE *fp, GraphNode *node) {
    fprintf(fp, "__global__ void reduce_kernel_%d(float *input, float *output, int size) {\n", 
            node->node_id);
    fprintf(fp, "    extern __shared__ float sdata[];\n");
    fprintf(fp, "    int tid = threadIdx.x;\n");
    fprintf(fp, "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n");
    fprintf(fp, "    sdata[tid] = (i < size) ? input[i] : 0.0f;\n");
    fprintf(fp, "    __syncthreads();\n");
    fprintf(fp, "    \n");
    fprintf(fp, "    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n");
    fprintf(fp, "        if (tid < s) {\n");
    fprintf(fp, "            sdata[tid] += sdata[tid + s];\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        __syncthreads();\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    \n");
    fprintf(fp, "    if (tid == 0) output[blockIdx.x] = sdata[0];\n");
    fprintf(fp, "}\n\n");
}

void generate_transpose_kernel(FILE *fp, GraphNode *node) {
    fprintf(fp, "__global__ void transpose_kernel_%d(float *input, float *output, int rows, int cols) {\n", 
            node->node_id);
    fprintf(fp, "    int row = blockIdx.y * blockDim.y + threadIdx.y;\n");
    fprintf(fp, "    int col = blockIdx.x * blockDim.x + threadIdx.x;\n");
    fprintf(fp, "    \n");
    fprintf(fp, "    if (row < rows && col < cols) {\n");
    fprintf(fp, "        output[col * rows + row] = input[row * cols + col];\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "}\n\n");
}

void generate_main_function(FILE *fp, TensorGraph *graph) {
    fprintf(fp, "int main() {\n");
    fprintf(fp, "    /* Allocate device memory */\n");
    fprintf(fp, "    float *d_A, *d_B, *d_C;\n");
    fprintf(fp, "    int size = 1024 * 1024; /* Example size */\n");
    fprintf(fp, "    \n");
    fprintf(fp, "    cudaMalloc(&d_A, size * sizeof(float));\n");
    fprintf(fp, "    cudaMalloc(&d_B, size * sizeof(float));\n");
    fprintf(fp, "    cudaMalloc(&d_C, size * sizeof(float));\n");
    fprintf(fp, "    \n");
    fprintf(fp, "    /* Launch kernels */\n");
    
    for (int i = 0; i < graph->num_nodes; i++) {
        GraphNode *node = graph->nodes[i];
        if (!node) continue;
        
        switch (node->op_type) {
            case OP_MATMUL:
                fprintf(fp, "    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);\n");
                fprintf(fp, "    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);\n");
                fprintf(fp, "    matmul_kernel_%d<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);\n", node->node_id);
                break;
            case OP_ADD:
            case OP_MUL:
                fprintf(fp, "    int numBlocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;\n");
                fprintf(fp, "    %s_kernel_%d<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, size);\n", 
                        (node->op_type == OP_ADD) ? "add" : "mul", node->node_id);
                break;
            default:
                break;
        }
    }
    
    fprintf(fp, "    \n");
    fprintf(fp, "    cudaDeviceSynchronize();\n");
    fprintf(fp, "    \n");
    fprintf(fp, "    /* Free memory */\n");
    fprintf(fp, "    cudaFree(d_A);\n");
    fprintf(fp, "    cudaFree(d_B);\n");
    fprintf(fp, "    cudaFree(d_C);\n");
    fprintf(fp, "    \n");
    fprintf(fp, "    return 0;\n");
    fprintf(fp, "}\n");
}

const char* get_cuda_type_string(void) {
    return "float";
}

