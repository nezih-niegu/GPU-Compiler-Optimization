#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor_graph.h"

/* Optimization strategies */
typedef enum OptStrategy {
    OPT_FUSE_OPS,        /* Fuse consecutive operations */
    OPT_COMMON_SUBEXPR,  /* Common subexpression elimination */
    OPT_LOOP_FUSION,     /* Fuse loops when possible */
    OPT_MEMORY_LAYOUT,   /* Optimize memory layout */
    OPT_ALGEBRAIC        /* Algebraic optimizations */
} OptStrategy;

/* Optimization result */
typedef struct OptResult {
    TensorGraph *optimized_graph;
    int memory_reduction;      /* Percentage reduction in memory accesses */
    int compute_reduction;    /* Percentage reduction in compute */
    int total_ops_before;
    int total_ops_after;
} OptResult;

/* Function declarations */
OptResult* optimize_graph(TensorGraph *graph, OptStrategy *strategies, int num_strategies);
void free_opt_result(OptResult *result);
TensorGraph* fuse_operations(TensorGraph *graph);
TensorGraph* eliminate_common_subexpressions(TensorGraph *graph);
TensorGraph* optimize_memory_layout(TensorGraph *graph);
void print_optimization_report(OptResult *result);

#endif /* OPTIMIZER_H */

