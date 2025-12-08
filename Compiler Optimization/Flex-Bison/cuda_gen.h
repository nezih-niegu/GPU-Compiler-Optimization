#ifndef CUDA_GEN_H
#define CUDA_GEN_H

#include "tensor_graph.h"
#include <stdio.h>

/* CUDA code generation */
void generate_cuda_code(TensorGraph *graph, const char *output_file);
void generate_kernel_header(FILE *fp);
void generate_kernel_footer(FILE *fp);
void generate_matmul_kernel(FILE *fp, GraphNode *node);
void generate_elementwise_kernel(FILE *fp, GraphNode *node, const char *op);
void generate_reduce_kernel(FILE *fp, GraphNode *node);
void generate_transpose_kernel(FILE *fp, GraphNode *node);
void generate_main_function(FILE *fp, TensorGraph *graph);
const char* get_cuda_type_string(void);

#endif /* CUDA_GEN_H */

