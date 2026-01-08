#ifndef VALIDATOR_H
#define VALIDATOR_H

#include "tensor_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Validation result structure */
typedef struct ValidationResult {
    int passed;              /* 1 if validation passed, 0 otherwise */
    char *test_name;        /* Name of the test */
    char *error_message;    /* Error message if validation failed */
    int expected_nodes;     /* Expected number of nodes in graph */
    int actual_nodes;       /* Actual number of nodes */
    int expected_ops;       /* Expected number of operations */
    int actual_ops;         /* Actual number of operations */
} ValidationResult;

/* Validation functions */
ValidationResult* validate_syntax(const char *test_file);
ValidationResult* validate_tensor_dimensions(TensorGraph *graph);
ValidationResult* validate_operation_compatibility(TensorGraph *graph);
ValidationResult* validate_matmul_dimensions(GraphNode *node);
ValidationResult* validate_elementwise_dimensions(GraphNode *node);
ValidationResult* validate_transpose_dimensions(GraphNode *node);
ValidationResult* validate_reduce_dimensions(GraphNode *node);
ValidationResult* validate_graph_structure(TensorGraph *graph);

/* Helper functions */
void free_validation_result(ValidationResult *result);
void print_validation_result(ValidationResult *result);
int validate_cuda_code_generated(const char *cuda_file);
int validate_llvm_ir_generated(const char *llvm_file);

/* Test suite runner */
int run_validation_suite(const char *test_dir);
int validate_single_test(const char *test_file);

#endif /* VALIDATOR_H */

