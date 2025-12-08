#include "tensor_graph.h"

#define INITIAL_CAPACITY 64

TensorGraph* create_tensor_graph(void) {
    TensorGraph *graph = malloc(sizeof(TensorGraph));
    graph->capacity = INITIAL_CAPACITY;
    graph->num_nodes = 0;
    graph->nodes = calloc(graph->capacity, sizeof(GraphNode*));
    graph->outputs = NULL;
    return graph;
}

void free_tensor_graph(TensorGraph *graph) {
    if (!graph) return;
    for (int i = 0; i < graph->num_nodes; i++) {
        GraphNode *node = graph->nodes[i];
        if (node) {
            if (node->name) free(node->name);
            if (node->shape) free_tensor_dim(node->shape);
            if (node->inputs) free(node->inputs);
            if (node->data) free(node->data);
            free(node);
        }
    }
    free(graph->nodes);
    free(graph);
}

TensorDim* create_tensor_dim(int *dims, int ndims) {
    TensorDim *td = malloc(sizeof(TensorDim));
    td->ndims = ndims;
    td->dims = malloc(ndims * sizeof(int));
    td->total_size = 1;
    for (int i = 0; i < ndims; i++) {
        td->dims[i] = dims[i];
        td->total_size *= dims[i];
    }
    return td;
}

void free_tensor_dim(TensorDim *td) {
    if (!td) return;
    if (td->dims) free(td->dims);
    free(td);
}

GraphNode* add_graph_node(TensorGraph *graph, OpType op_type, char *name, TensorDim *shape) {
    if (graph->num_nodes >= graph->capacity) {
        graph->capacity *= 2;
        graph->nodes = realloc(graph->nodes, graph->capacity * sizeof(GraphNode*));
    }
    
    GraphNode *node = malloc(sizeof(GraphNode));
    node->node_id = graph->num_nodes++;
    node->op_type = op_type;
    node->name = name ? strdup(name) : NULL;
    node->shape = shape;
    node->inputs = NULL;
    node->num_inputs = 0;
    node->data = NULL;
    node->is_constant = 0;
    node->visited = 0;
    node->next = NULL;
    
    graph->nodes[node->node_id] = node;
    return node;
}

void add_edge(GraphNode *from, GraphNode *to) {
    if (!from || !to) return;
    to->num_inputs++;
    to->inputs = realloc(to->inputs, to->num_inputs * sizeof(GraphNode*));
    to->inputs[to->num_inputs - 1] = from;
}

void print_graph(TensorGraph *graph) {
    printf("\n=== TENSOR OPERATION GRAPH ===\n");
    for (int i = 0; i < graph->num_nodes; i++) {
        GraphNode *node = graph->nodes[i];
        if (!node) continue;
        
        const char *op_names[] = {
            "ADD", "MUL", "MATMUL", "TRANSPOSE", "REDUCE", 
            "RESHAPE", "ASSIGN", "IDENTITY"
        };
        
        printf("Node %d: %s", node->node_id, op_names[node->op_type]);
        if (node->name) printf(" [%s]", node->name);
        if (node->shape) {
            printf(" Shape: [");
            for (int j = 0; j < node->shape->ndims; j++) {
                printf("%d", node->shape->dims[j]);
                if (j < node->shape->ndims - 1) printf(", ");
            }
            printf("]");
        }
        if (node->num_inputs > 0) {
            printf(" <- Inputs: ");
            for (int j = 0; j < node->num_inputs; j++) {
                printf("%d ", node->inputs[j]->node_id);
            }
        }
        printf("\n");
    }
}

int calculate_memory_accesses(GraphNode *node) {
    if (!node || !node->shape) return 0;
    
    int accesses = node->shape->total_size; /* Output accesses */
    
    /* Add input accesses */
    for (int i = 0; i < node->num_inputs; i++) {
        if (node->inputs[i] && node->inputs[i]->shape) {
            accesses += node->inputs[i]->shape->total_size;
        }
    }
    
    return accesses;
}

int calculate_computational_cost(GraphNode *node) {
    if (!node || !node->shape) return 0;
    
    int cost = 0;
    switch (node->op_type) {
        case OP_ADD:
        case OP_MUL:
            cost = node->shape->total_size;
            break;
        case OP_MATMUL:
            /* Matrix multiplication: O(n*m*k) */
            if (node->num_inputs >= 2 && node->inputs[0]->shape && node->inputs[1]->shape) {
                int n = node->inputs[0]->shape->dims[0];
                int m = node->inputs[0]->shape->dims[1];
                int k = node->inputs[1]->shape->dims[1];
                cost = n * m * k;
            }
            break;
        case OP_TRANSPOSE:
            cost = node->shape->total_size; /* Just memory movement */
            break;
        case OP_REDUCE:
            cost = node->shape->total_size * 2; /* Read + reduce */
            break;
        case OP_RESHAPE:
            cost = node->shape->total_size; /* Memory movement */
            break;
        default:
            cost = node->shape->total_size;
    }
    
    return cost;
}

