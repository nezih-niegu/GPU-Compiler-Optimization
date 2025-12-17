#!/bin/bash
# Script para ejecutar el compilador con operaciones tensoriales
# Uso: ./ejecutar.sh

printf "program tensor_test\ntensor A[100,50];\ntensor B[50,200];\nbegin\nC := A @matmul B;\nD := A + A;\nE := @transpose(A);\nF := @reduce(A, 0);\nend\n" | ./compiler

