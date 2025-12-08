#ifndef TENSOR_GRAPH_H
#define TENSOR_GRAPH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tensor dimension information */
typedef struct TensorDim {
    int *dims;      /* Array of dimension sizes */
    int ndims;      /* Number of dimensions */
    int total_size; /* Total elements */
} TensorDim;

/* Operation types in the graph */
typedef enum OpType {
    OP_ADD,
    OP_MUL,
    OP_MATMUL,
    OP_TRANSPOSE,
    OP_REDUCE,
    OP_RESHAPE,
    OP_ASSIGN,
    OP_IDENTITY
} OpType;

/* Graph node representing a tensor operation */
typedef struct GraphNode {
    int node_id;
    OpType op_type;
    char *name;              /* Variable/tensor name */
    TensorDim *shape;        /* Output shape */
    struct GraphNode **inputs; /* Input nodes */
    int num_inputs;
    float *data;             /* For constant tensors */
    int is_constant;
    int visited;             /* For graph traversal */
    struct GraphNode *next;  /* For linked list */
} GraphNode;

/* Graph structure */
typedef struct TensorGraph {
    GraphNode **nodes;
    int num_nodes;
    int capacity;
    GraphNode *outputs;      /* Linked list of output nodes */
} TensorGraph;

/* Function declarations */
TensorGraph* create_tensor_graph(void);
void free_tensor_graph(TensorGraph *graph);
GraphNode* add_graph_node(TensorGraph *graph, OpType op_type, char *name, TensorDim *shape);
void add_edge(GraphNode *from, GraphNode *to);
TensorDim* create_tensor_dim(int *dims, int ndims);
void free_tensor_dim(TensorDim *td);
void print_graph(TensorGraph *graph);
int calculate_memory_accesses(GraphNode *node);
int calculate_computational_cost(GraphNode *node);

#endif /* TENSOR_GRAPH_H */

