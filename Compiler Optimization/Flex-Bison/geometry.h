#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "tensor_graph.h"

/* Point in n-dimensional space */
typedef struct Point {
    int *coords;  /* Coordinates */
    int dim;      /* Dimension */
} Point;

/* Hyper-rectangle representing iteration space */
typedef struct IterationSpace {
    Point *lower;  /* Lower bounds */
    Point *upper;  /* Upper bounds */
    int dim;       /* Dimension */
} IterationSpace;

/* Function declarations */
IterationSpace* create_iteration_space(int *lower, int *upper, int dim);
void free_iteration_space(IterationSpace *space);
int is_point_in_space(Point *p, IterationSpace *space);
IterationSpace* intersect_spaces(IterationSpace *a, IterationSpace *b);
IterationSpace* union_spaces(IterationSpace *a, IterationSpace *b);
int calculate_space_volume(IterationSpace *space);
void print_iteration_space(IterationSpace *space);
IterationSpace* tensor_to_iteration_space(TensorDim *tensor);
Point* create_point(int *coords, int dim);
void free_point(Point *p);

#endif /* GEOMETRY_H */

