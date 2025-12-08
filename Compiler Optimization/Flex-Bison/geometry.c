#include "geometry.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Point* create_point(int *coords, int dim) {
    Point *p = malloc(sizeof(Point));
    p->dim = dim;
    p->coords = malloc(dim * sizeof(int));
    memcpy(p->coords, coords, dim * sizeof(int));
    return p;
}

void free_point(Point *p) {
    if (!p) return;
    if (p->coords) free(p->coords);
    free(p);
}

IterationSpace* create_iteration_space(int *lower, int *upper, int dim) {
    IterationSpace *space = malloc(sizeof(IterationSpace));
    space->dim = dim;
    space->lower = create_point(lower, dim);
    space->upper = create_point(upper, dim);
    return space;
}

void free_iteration_space(IterationSpace *space) {
    if (!space) return;
    if (space->lower) free_point(space->lower);
    if (space->upper) free_point(space->upper);
    free(space);
}

int is_point_in_space(Point *p, IterationSpace *space) {
    if (!p || !space || p->dim != space->dim) return 0;
    
    for (int i = 0; i < p->dim; i++) {
        if (p->coords[i] < space->lower->coords[i] || 
            p->coords[i] > space->upper->coords[i]) {
            return 0;
        }
    }
    return 1;
}

IterationSpace* intersect_spaces(IterationSpace *a, IterationSpace *b) {
    if (!a || !b || a->dim != b->dim) return NULL;
    
    int *lower = malloc(a->dim * sizeof(int));
    int *upper = malloc(a->dim * sizeof(int));
    
    for (int i = 0; i < a->dim; i++) {
        lower[i] = (a->lower->coords[i] > b->lower->coords[i]) ? 
                   a->lower->coords[i] : b->lower->coords[i];
        upper[i] = (a->upper->coords[i] < b->upper->coords[i]) ? 
                   a->upper->coords[i] : b->upper->coords[i];
        
        if (lower[i] > upper[i]) {
            free(lower);
            free(upper);
            return NULL; /* No intersection */
        }
    }
    
    IterationSpace *result = create_iteration_space(lower, upper, a->dim);
    free(lower);
    free(upper);
    return result;
}

IterationSpace* union_spaces(IterationSpace *a, IterationSpace *b) {
    if (!a || !b || a->dim != b->dim) return NULL;
    
    int *lower = malloc(a->dim * sizeof(int));
    int *upper = malloc(a->dim * sizeof(int));
    
    for (int i = 0; i < a->dim; i++) {
        lower[i] = (a->lower->coords[i] < b->lower->coords[i]) ? 
                   a->lower->coords[i] : b->lower->coords[i];
        upper[i] = (a->upper->coords[i] > b->upper->coords[i]) ? 
                   a->upper->coords[i] : b->upper->coords[i];
    }
    
    IterationSpace *result = create_iteration_space(lower, upper, a->dim);
    free(lower);
    free(upper);
    return result;
}

int calculate_space_volume(IterationSpace *space) {
    if (!space) return 0;
    
    int volume = 1;
    for (int i = 0; i < space->dim; i++) {
        int size = space->upper->coords[i] - space->lower->coords[i] + 1;
        volume *= size;
    }
    return volume;
}

void print_iteration_space(IterationSpace *space) {
    if (!space) return;
    
    printf("Iteration Space [");
    for (int i = 0; i < space->dim; i++) {
        printf("[%d, %d]", space->lower->coords[i], space->upper->coords[i]);
        if (i < space->dim - 1) printf(", ");
    }
    printf("] Volume: %d\n", calculate_space_volume(space));
}

IterationSpace* tensor_to_iteration_space(TensorDim *tensor) {
    if (!tensor) return NULL;
    
    int *lower = calloc(tensor->ndims, sizeof(int));
    int *upper = malloc(tensor->ndims * sizeof(int));
    
    for (int i = 0; i < tensor->ndims; i++) {
        upper[i] = tensor->dims[i] - 1;
    }
    
    IterationSpace *space = create_iteration_space(lower, upper, tensor->ndims);
    free(lower);
    free(upper);
    return space;
}

