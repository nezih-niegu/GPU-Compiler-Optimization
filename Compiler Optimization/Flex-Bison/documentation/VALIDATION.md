# Quick Validation Guide

## Quick Start

### 1. Compile the Project
```bash
cd "Compiler Optimization/Flex-Bison"
make
```

### 2. Run Full Validation
```bash
./run_validation.sh
```

This script will run all tests and generate a complete report.

### 3. Validate an Individual Test
```bash
# Validate syntax
./compiler < tests/test_tensor.txt

# Check generated code
ls -la generated_kernels.cu tensor_output.ll
```

---

## What is Validated?

### Syntax
- The compiler can parse the code without errors
- Tokens recognized correctly
- Valid program structure

### ✅ Mathematical Operations
- **MatMul**: Compatible dimensions (A[m,n] @ B[n,k] = C[m,k])
- **Addition/Multiplication**: Same dimensions (A[m,n] + B[m,n] = C[m,n])
- **Transpose**: Inverted dimensions (A[m,n] → B[n,m])
- **Reduction**: Dimension correctly eliminated

### ✅ Tensor Dimensions
- All dimensions are positive
- Compatibility between operations
- Correct shape propagation

### ✅ Generated Code
- **CUDA**: File `generated_kernels.cu` exists and contains kernels
- **LLVM IR**: File `tensor_output.ll` exists and is valid

---

## Validation Examples

### Successful Test
```bash
$ ./compiler < tests/test_tensor.txt

✓ Valid syntax
✓ Operation graph created (20 nodes)
✓ CUDA code generated (4 kernels)
✓ LLVM IR generated
```

### Test with Error
```bash
$ ./compiler < tests/test_error.txt

✗ Syntax error: unexpected token 'tensor' at line 3
```

---