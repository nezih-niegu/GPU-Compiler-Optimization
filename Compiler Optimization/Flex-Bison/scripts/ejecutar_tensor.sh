#!/bin/bash
# Script para ejecutar el compilador con operaciones tensoriales
# Uso: ./ejecutar_tensor.sh

echo "program tensor_test
tensor A[100,50];
tensor B[50,200];
begin
C := A @matmul B;
D := A + A;
E := @transpose(A);
F := @reduce(A, 0);
end" | ./compiler

