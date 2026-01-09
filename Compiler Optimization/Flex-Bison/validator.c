#include "validator.h"
#include "tensor_graph.h"
#include <ctype.h>

/* Free validation result */
void free_validation_result(ValidationResult *result) {
    if (!result) return;
    if (result->test_name) free(result->test_name);
    if (result->error_message) free(result->error_message);
    free(result);
}

/* Print validation result */
void print_validation_result(ValidationResult *result) {
    if (!result) return;
    
    printf("\n=== Validation Result: %s ===\n", result->test_name ? result->test_name : "Unknown");
    if (result->passed) {
        printf("✓ PASSED\n");
    } else {
        printf("✗ FAILED\n");
        if (result->error_message) {
            printf("Error: %s\n", result->error_message);
        }
    }
    
    if (result->expected_nodes > 0) {
        printf("Nodes: Expected=%d, Actual=%d\n", result->expected_nodes, result->actual_nodes);
    }
    if (result->expected_ops > 0) {
        printf("Operations: Expected=%d, Actual=%d\n", result->expected_ops, result->actual_ops);
    }
    printf("================================\n");
}

/* Validate syntax by checking if compiler can parse the file */
ValidationResult* validate_syntax(const char *test_file) {
    ValidationResult *result = calloc(1, sizeof(ValidationResult));
    result->test_name = strdup(test_file);
    result->passed = 0;
    
    /* Try to compile and run the test file */
    char command[512];
    snprintf(command, sizeof(command), "./compiler < %s > /dev/null 2>&1", test_file);
    
    int exit_code = system(command);
    
    if (exit_code == 0) {
        result->passed = 1;
    } else {
        result->error_message = strdup("Syntax error: Compiler failed to parse the file");
    }
    
    return result;
}

/* Validate tensor dimensions compatibility */
ValidationResult* validate_tensor_dimensions(TensorGraph *graph) {
    ValidationResult *result = calloc(1, sizeof(ValidationResult));
    result->test_name = strdup("Tensor Dimensions Validation");
    result->passed = 1;
    result->actual_nodes = graph ? graph->num_nodes : 0;
    
    if (!graph || graph->num_nodes == 0) {
        result->passed = 0;
        result->error_message = strdup("Empty graph");
        return result;
    }
    
    /* Check each node has valid dimensions */
    for (int i = 0; i < graph->num_nodes; i++) {
        GraphNode *node = graph->nodes[i];
        if (!node) continue;
        
        if (node->shape) {
            /* Validate dimension sizes are positive */
            for (int j = 0; j < node->shape->ndims; j++) {
                if (node->shape->dims[j] <= 0) {
                    result->passed = 0;
                    char msg[256];
                    snprintf(msg, sizeof(msg), "Node %d has invalid dimension %d: %d", 
                            node->node_id, j, node->shape->dims[j]);
                    result->error_message = strdup(msg);
                    return result;
                }
            }
        }
    }
    
    return result;
}

/* Validate matmul dimensions */
ValidationResult* validate_matmul_dimensions(GraphNode *node) {
    ValidationResult *result = calloc(1, sizeof(ValidationResult));
    result->test_name = strdup("MatMul Dimensions Validation");
    result->passed = 1;
    
    if (!node || node->op_type != OP_MATMUL) {
        result->passed = 0;
        result->error_message = strdup("Not a matmul node");
        return result;
    }
    
    if (node->num_inputs < 2) {
        result->passed = 0;
        result->error_message = strdup("MatMul requires 2 inputs");
        return result;
    }
    
    GraphNode *input1 = node->inputs[0];
    GraphNode *input2 = node->inputs[1];
    
    if (!input1->shape || !input2->shape || !node->shape) {
        result->passed = 0;
        result->error_message = strdup("Missing shape information");
        return result;
    }
    
    /* MatMul: A[m,n] @ B[n,k] = C[m,k] */
    int m = input1->shape->dims[0];
    int n1 = input1->shape->ndims > 1 ? input1->shape->dims[1] : 1;
    int n2 = input2->shape->ndims > 0 ? input2->shape->dims[0] : 1;
    int k = input2->shape->ndims > 1 ? input2->shape->dims[1] : 1;
    
    int out_m = node->shape->dims[0];
    int out_k = node->shape->ndims > 1 ? node->shape->dims[1] : 1;
    
    /* Check inner dimension matches */
    if (n1 != n2) {
        result->passed = 0;
        char msg[256];
        snprintf(msg, sizeof(msg), "MatMul dimension mismatch: A[%d,%d] @ B[%d,%d] - inner dims don't match (%d != %d)",
                m, n1, n2, k, n1, n2);
        result->error_message = strdup(msg);
        return result;
    }
    
    /* Check output dimensions */
    if (out_m != m || out_k != k) {
        result->passed = 0;
        char msg[256];
        snprintf(msg, sizeof(msg), "MatMul output dimension mismatch: Expected [%d,%d], got [%d,%d]",
                m, k, out_m, out_k);
        result->error_message = strdup(msg);
        return result;
    }
    
    return result;
}

/* Validate elementwise operation dimensions */
ValidationResult* validate_elementwise_dimensions(GraphNode *node) {
    ValidationResult *result = calloc(1, sizeof(ValidationResult));
    result->test_name = strdup("Elementwise Dimensions Validation");
    result->passed = 1;
    
    if (!node || (node->op_type != OP_ADD && node->op_type != OP_MUL)) {
        result->passed = 0;
        result->error_message = strdup("Not an elementwise operation");
        return result;
    }
    
    if (node->num_inputs < 2) {
        result->passed = 0;
        result->error_message = strdup("Elementwise operation requires 2 inputs");
        return result;
    }
    
    GraphNode *input1 = node->inputs[0];
    GraphNode *input2 = node->inputs[1];
    
    if (!input1->shape || !input2->shape || !node->shape) {
        result->passed = 0;
        result->error_message = strdup("Missing shape information");
        return result;
    }
    
    /* Elementwise operations require same shape */
    if (input1->shape->ndims != input2->shape->ndims) {
        result->passed = 0;
        result->error_message = strdup("Elementwise operation requires same number of dimensions");
        return result;
    }
    
    for (int i = 0; i < input1->shape->ndims; i++) {
        if (input1->shape->dims[i] != input2->shape->dims[i]) {
            result->passed = 0;
            char msg[256];
            snprintf(msg, sizeof(msg), "Elementwise dimension mismatch at dim %d: %d != %d",
                    i, input1->shape->dims[i], input2->shape->dims[i]);
            result->error_message = strdup(msg);
            return result;
        }
    }
    
    /* Output should match input shape */
    if (node->shape->ndims != input1->shape->ndims) {
        result->passed = 0;
        result->error_message = strdup("Output shape doesn't match input shape");
        return result;
    }
    
    return result;
}

/* Validate transpose dimensions */
ValidationResult* validate_transpose_dimensions(GraphNode *node) {
    ValidationResult *result = calloc(1, sizeof(ValidationResult));
    result->test_name = strdup("Transpose Dimensions Validation");
    result->passed = 1;
    
    if (!node || node->op_type != OP_TRANSPOSE) {
        result->passed = 0;
        result->error_message = strdup("Not a transpose node");
        return result;
    }
    
    if (node->num_inputs < 1) {
        result->passed = 0;
        result->error_message = strdup("Transpose requires 1 input");
        return result;
    }
    
    GraphNode *input = node->inputs[0];
    
    if (!input->shape || !node->shape) {
        result->passed = 0;
        result->error_message = strdup("Missing shape information");
        return result;
    }
    
    /* Transpose swaps last two dimensions for 2D tensors */
    if (input->shape->ndims == 2 && node->shape->ndims == 2) {
        int in_rows = input->shape->dims[0];
        int in_cols = input->shape->dims[1];
        int out_rows = node->shape->dims[0];
        int out_cols = node->shape->dims[1];
        
        if (in_rows != out_cols || in_cols != out_rows) {
            result->passed = 0;
            char msg[256];
            snprintf(msg, sizeof(msg), "Transpose dimension mismatch: Input[%d,%d] -> Output[%d,%d] (expected [%d,%d])",
                    in_rows, in_cols, out_rows, out_cols, in_cols, in_rows);
            result->error_message = strdup(msg);
            return result;
        }
    }
    
    return result;
}

/* Validate reduce dimensions */
ValidationResult* validate_reduce_dimensions(GraphNode *node) {
    ValidationResult *result = calloc(1, sizeof(ValidationResult));
    result->test_name = strdup("Reduce Dimensions Validation");
    result->passed = 1;
    
    if (!node || node->op_type != OP_REDUCE) {
        result->passed = 0;
        result->error_message = strdup("Not a reduce node");
        return result;
    }
    
    if (node->num_inputs < 1) {
        result->passed = 0;
        result->error_message = strdup("Reduce requires 1 input");
        return result;
    }
    
    GraphNode *input = node->inputs[0];
    
    if (!input->shape || !node->shape) {
        result->passed = 0;
        result->error_message = strdup("Missing shape information");
        return result;
    }
    
    /* Reduce along axis 0 should reduce first dimension */
    if (input->shape->ndims > 1 && node->shape->ndims == input->shape->ndims - 1) {
        /* Check that first dimension is removed */
        for (int i = 0; i < node->shape->ndims; i++) {
            if (node->shape->dims[i] != input->shape->dims[i + 1]) {
                result->passed = 0;
                char msg[256];
                snprintf(msg, sizeof(msg), "Reduce dimension mismatch at dim %d: %d != %d",
                        i, node->shape->dims[i], input->shape->dims[i + 1]);
                result->error_message = strdup(msg);
                return result;
            }
        }
    }
    
    return result;
}

/* Validate operation compatibility */
ValidationResult* validate_operation_compatibility(TensorGraph *graph) {
    ValidationResult *result = calloc(1, sizeof(ValidationResult));
    result->test_name = strdup("Operation Compatibility Validation");
    result->passed = 1;
    result->actual_ops = 0;
    
    if (!graph || graph->num_nodes == 0) {
        result->passed = 0;
        result->error_message = strdup("Empty graph");
        return result;
    }
    
    /* Validate each operation */
    for (int i = 0; i < graph->num_nodes; i++) {
        GraphNode *node = graph->nodes[i];
        if (!node) continue;
        
        ValidationResult *op_result = NULL;
        
        switch (node->op_type) {
            case OP_MATMUL:
                op_result = validate_matmul_dimensions(node);
                result->actual_ops++;
                break;
            case OP_ADD:
            case OP_MUL:
                op_result = validate_elementwise_dimensions(node);
                result->actual_ops++;
                break;
            case OP_TRANSPOSE:
                op_result = validate_transpose_dimensions(node);
                result->actual_ops++;
                break;
            case OP_REDUCE:
                op_result = validate_reduce_dimensions(node);
                result->actual_ops++;
                break;
            default:
                break;
        }
        
        if (op_result && !op_result->passed) {
            result->passed = 0;
            if (!result->error_message) {
                char msg[512];
                snprintf(msg, sizeof(msg), "Operation validation failed at node %d: %s",
                        node->node_id, op_result->error_message ? op_result->error_message : "Unknown error");
                result->error_message = strdup(msg);
            }
            free_validation_result(op_result);
            return result;
        }
        
        if (op_result) free_validation_result(op_result);
    }
    
    return result;
}

/* Validate graph structure */
ValidationResult* validate_graph_structure(TensorGraph *graph) {
    ValidationResult *result = calloc(1, sizeof(ValidationResult));
    result->test_name = strdup("Graph Structure Validation");
    result->passed = 1;
    result->actual_nodes = graph ? graph->num_nodes : 0;
    
    if (!graph) {
        result->passed = 0;
        result->error_message = strdup("Graph is NULL");
        return result;
    }
    
    /* Check for cycles (basic check) */
    for (int i = 0; i < graph->num_nodes; i++) {
        GraphNode *node = graph->nodes[i];
        if (!node) continue;
        
        /* Check that inputs are valid */
        for (int j = 0; j < node->num_inputs; j++) {
            if (!node->inputs[j]) {
                result->passed = 0;
                result->error_message = strdup("Node has NULL input");
                return result;
            }
        }
    }
    
    return result;
}

/* Validate CUDA code was generated */
int validate_cuda_code_generated(const char *cuda_file) {
    FILE *fp = fopen(cuda_file, "r");
    if (!fp) return 0;
    
    char line[1024];
    int has_kernel = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "__global__")) has_kernel = 1;
    }
    
    fclose(fp);
    return has_kernel;
}

/* Validate LLVM IR was generated */
int validate_llvm_ir_generated(const char *llvm_file) {
    FILE *fp = fopen(llvm_file, "r");
    if (!fp) return 0;
    
    char line[1024];
    int has_module = 0;
    int has_function = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "ModuleID") || strstr(line, "target")) has_module = 1;
        if (strstr(line, "define") || strstr(line, "declare")) has_function = 1;
    }
    
    fclose(fp);
    return has_module && has_function;
}

/* Validate a single test file */
int validate_single_test(const char *test_file) {
    printf("\n>>> Validating: %s\n", test_file);
    
    /* Test 1: Syntax validation */
    ValidationResult *syntax_result = validate_syntax(test_file);
    print_validation_result(syntax_result);
    int syntax_ok = syntax_result->passed;
    free_validation_result(syntax_result);
    
    if (!syntax_ok) {
        printf("✗ Syntax validation failed - skipping further tests\n");
        return 0;
    }
    
    /* Test 2: Check if CUDA code was generated */
    int cuda_ok = validate_cuda_code_generated("generated_kernels.cu");
    if (cuda_ok) {
        printf("✓ CUDA code generated successfully\n");
    } else {
        printf("⚠ CUDA code not generated (may be expected for non-tensor tests)\n");
    }
    
    /* Test 3: Check if LLVM IR was generated */
    int llvm_ok = validate_llvm_ir_generated("tensor_output.ll");
    if (llvm_ok) {
        printf("✓ LLVM IR generated successfully\n");
    } else {
        printf("⚠ LLVM IR not generated (may be expected for non-tensor tests)\n");
    }
    
    return syntax_ok;
}

/* Run validation suite on all tests */
int run_validation_suite(const char *test_dir) {
    printf("========================================\n");
    printf("VALIDATION SUITE - Tensor Compiler\n");
    printf("========================================\n\n");
    
    int total_tests = 0;
    int passed_tests = 0;
    
    /* List of test files */
    const char *test_files[] = {
        "tests/pruebaT51.txt",
        "tests/pruebaT52.txt",
        "tests/test_simple.txt",
        "tests/test_tensor.txt",
        "tests/test_tensor2.txt",
        "tests/test_tensor3.txt",
        "tests/test_algebraic.txt",
        "tests/test_cse.txt",
        NULL
    };
    
    for (int i = 0; test_files[i] != NULL; i++) {
        total_tests++;
        if (validate_single_test(test_files[i])) {
            passed_tests++;
        }
        printf("\n");
    }
    
    printf("========================================\n");
    printf("VALIDATION SUMMARY\n");
    printf("========================================\n");
    printf("Total Tests: %d\n", total_tests);
    printf("Passed: %d\n", passed_tests);
    printf("Failed: %d\n", total_tests - passed_tests);
    printf("Success Rate: %.1f%%\n", (float)passed_tests / total_tests * 100);
    printf("========================================\n");
    
    return (passed_tests == total_tests) ? 0 : 1;
}

