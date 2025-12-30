#ifndef LLVM_LOWERING_H
#define LLVM_LOWERING_H

#include "tensor_graph.h"
#include <stdio.h>

/**
 * LLVM IR Lowering Module
 *
 * Translates TensorGraph (internal IR) to LLVM IR (.ll files)
 * Uses LLVM's textual IR format for analysis and optimization
 *
 * Design Philosophy:
 * - Minimal lowering: map tensor ops to LLVM function calls
 * - Preserve structure for analysis passes
 * - Emit metadata for optimization hints
 * - Focus on correctness and clarity over performance
 */

/* Lowering context */
typedef struct LLVMLoweringContext
{
    FILE *output;         /* Output .ll file */
    TensorGraph *graph;   /* Input tensor graph */
    int register_counter; /* SSA register naming */
    int label_counter;    /* Basic block labels */
    int function_id;      /* Unique function IDs */
} LLVMLoweringContext;

/* Main lowering functions */
LLVMLoweringContext *create_lowering_context(TensorGraph *graph, const char *output_file);
void free_lowering_context(LLVMLoweringContext *ctx);
int lower_graph_to_llvm_ir(TensorGraph *graph, const char *output_file);

/* IR emission functions */
void emit_llvm_header(LLVMLoweringContext *ctx);
void emit_llvm_footer(LLVMLoweringContext *ctx);
void emit_tensor_declaration(LLVMLoweringContext *ctx, GraphNode *node);
void emit_tensor_allocation(LLVMLoweringContext *ctx, GraphNode *node);
void emit_operation(LLVMLoweringContext *ctx, GraphNode *node);
void emit_main_function(LLVMLoweringContext *ctx);

/* Operation-specific emission */
void emit_matmul_op(LLVMLoweringContext *ctx, GraphNode *node);
void emit_elementwise_op(LLVMLoweringContext *ctx, GraphNode *node, const char *op);
void emit_transpose_op(LLVMLoweringContext *ctx, GraphNode *node);
void emit_reduce_op(LLVMLoweringContext *ctx, GraphNode *node);

/* Helper functions */
int get_next_register(LLVMLoweringContext *ctx);
int get_next_label(LLVMLoweringContext *ctx);
const char *get_llvm_type_string(TensorDim *shape);
void emit_metadata(LLVMLoweringContext *ctx, const char *key, const char *value);
void emit_analysis_annotations(LLVMLoweringContext *ctx, GraphNode *node);

/* Verification */
int verify_llvm_ir(const char *ll_file);

#endif /* LLVM_LOWERING_H */
