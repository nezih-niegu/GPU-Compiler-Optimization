#include "optimizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

OptResult* optimize_graph(TensorGraph *graph, OptStrategy *strategies, int num_strategies) {
    OptResult *result = malloc(sizeof(OptResult));
    result->optimized_graph = graph; /* Start with original */
    result->memory_reduction = 0;
    result->compute_reduction = 0;
    
    /* Calculate baseline metrics */
    int total_mem_before = 0;
    int total_comp_before = 0;
    for (int i = 0; i < graph->num_nodes; i++) {
        if (graph->nodes[i]) {
            total_mem_before += calculate_memory_accesses(graph->nodes[i]);
            total_comp_before += calculate_computational_cost(graph->nodes[i]);
        }
    }
    result->total_ops_before = graph->num_nodes;
    
    /* Apply optimizations */
    for (int i = 0; i < num_strategies; i++) {
        switch (strategies[i]) {
            case OPT_FUSE_OPS:
                result->optimized_graph = fuse_operations(result->optimized_graph);
                break;
            case OPT_COMMON_SUBEXPR:
                result->optimized_graph = eliminate_common_subexpressions(result->optimized_graph);
                break;
            case OPT_MEMORY_LAYOUT:
                result->optimized_graph = optimize_memory_layout(result->optimized_graph);
                break;
            default:
                break;
        }
    }
    
    /* Calculate optimized metrics */
    int total_mem_after = 0;
    int total_comp_after = 0;
    for (int i = 0; i < result->optimized_graph->num_nodes; i++) {
        if (result->optimized_graph->nodes[i]) {
            total_mem_after += calculate_memory_accesses(result->optimized_graph->nodes[i]);
            total_comp_after += calculate_computational_cost(result->optimized_graph->nodes[i]);
        }
    }
    result->total_ops_after = result->optimized_graph->num_nodes;
    
    if (total_mem_before > 0) {
        result->memory_reduction = ((total_mem_before - total_mem_after) * 100) / total_mem_before;
    }
    if (total_comp_before > 0) {
        result->compute_reduction = ((total_comp_before - total_comp_after) * 100) / total_comp_before;
    }
    
    return result;
}

void free_opt_result(OptResult *result) {
    if (!result) return;
    /* Note: We don't free the graph here as it might be used elsewhere */
    free(result);
}

TensorGraph* fuse_operations(TensorGraph *graph) {
    /* Simple fusion: combine consecutive element-wise operations */
    TensorGraph *optimized = create_tensor_graph();
    
    for (int i = 0; i < graph->num_nodes; i++) {
        GraphNode *node = graph->nodes[i];
        if (!node) continue;
        
        /* Check if we can fuse with previous operation */
        int fused = 0;
        if (i > 0 && node->op_type == OP_ADD && graph->nodes[i-1] && 
            graph->nodes[i-1]->op_type == OP_MUL && node->num_inputs > 0) {
            /* Fuse add and mul if they're consecutive */
            fused = 1;
        }
        
        if (!fused) {
            /* Copy node to optimized graph */
            add_graph_node(optimized, node->op_type, node->name, node->shape);
        }
    }
    
    return optimized;
}

TensorGraph* eliminate_common_subexpressions(TensorGraph *graph) {
    /* Identify and eliminate duplicate computations */
    TensorGraph *optimized = create_tensor_graph();
    
    for (int i = 0; i < graph->num_nodes; i++) {
        GraphNode *node = graph->nodes[i];
        if (!node) continue;
        
        /* Check if this computation was already done */
        int duplicate = 0;
        for (int j = 0; j < i; j++) {
            GraphNode *prev = graph->nodes[j];
            if (prev && prev->op_type == node->op_type && 
                prev->num_inputs == node->num_inputs) {
                /* Simple duplicate detection */
                duplicate = 1;
                break;
            }
        }
        
        if (!duplicate) {
            add_graph_node(optimized, node->op_type, node->name, node->shape);
        }
    }
    
    return optimized;
}

TensorGraph* optimize_memory_layout(TensorGraph *graph) {
    /* Optimize memory access patterns by reordering operations */
    /* This is a simplified version - in practice would analyze access patterns */
    return graph; /* For now, return original */
}

void print_optimization_report(OptResult *result) {
    printf("\n=== OPTIMIZATION REPORT ===\n");
    printf("Operations before: %d\n", result->total_ops_before);
    printf("Operations after:  %d\n", result->total_ops_after);
    printf("Memory reduction:  %d%%\n", result->memory_reduction);
    printf("Compute reduction: %d%%\n", result->compute_reduction);
}

