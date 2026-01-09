#!/bin/bash

# Script for Validation of the Tensor Compiler
# This script runs a comprehensive validation suite

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "VALIDATION OF THE TENSOR COMPILER"
echo "=========================================="
echo ""

# Verifying that we are in the correct directory
if [ ! -f "./compiler" ]; then
    echo -e "${RED}✗ Error: Compiler is not compiled${NC}"
    echo "Run 'make' first"
    exit 1
fi

# Note: validator.c is compiled as part of the main compiler via Makefile
# It's a library module, not a standalone executable

# Function to validate a test
validate_test() {
    local test_file=$1
    local test_name=$(basename "$test_file")
    
    echo "----------------------------------------"
    echo "Test: $test_name"
    echo "----------------------------------------"
    
    # Clean files generated previously
    rm -f generated_kernels.cu tensor_output.ll
    
    # Execute compiler
    if ./compiler < "$test_file" > /tmp/compiler_output.txt 2>&1; then
        echo -e "${GREEN}✓ Valid syntax${NC}"

        # Verify CUDA generation (if tensor operations are involved)
        if grep -q "tensor" "$test_file"; then
            if [ -f "generated_kernels.cu" ]; then
                local kernel_count=$(grep -c "__global__" generated_kernels.cu 2>/dev/null || echo "0")
                kernel_count=$(echo "$kernel_count" | tr -d '\n\r' | head -1)
                if [ -z "$kernel_count" ] || [ "$kernel_count" = "" ]; then
                    kernel_count=0
                fi
                if [ "$kernel_count" -gt 0 ] 2>/dev/null; then
                    echo -e "${GREEN}✓ CUDA code generated ($kernel_count kernels)${NC}"
                else
                    echo -e "${YELLOW}⚠ CUDA code generated but no kernels found${NC}"
                fi
            else
                echo -e "${YELLOW}⚠ CUDA code not generated${NC}"
            fi
            
            # Verify LLVM IR generation
            if [ -f "tensor_output.ll" ]; then
                if grep -q "define\|declare" tensor_output.ll 2>/dev/null; then
                    echo -e "${GREEN}✓ LLVM IR generated${NC}"
                else
                    echo -e "${YELLOW}⚠ LLVM IR generated but empty${NC}"
                fi
            else
                echo -e "${YELLOW}⚠ LLVM IR not generated${NC}"
            fi
        fi
        
        return 0
    else
        echo -e "${RED}✗ Syntax error${NC}"
        echo "Compiler output:"
        cat /tmp/compiler_output.txt | head -5
        return 1
    fi
}

# Function to validate mathematical operations
validate_operations() {
    local test_file=$1
    
    # Only validate if there are actual tensor operations, not just declarations
    if ! grep -q "@matmul\|@transpose\|@reduce\|@reshape" "$test_file"; then
        return 0  
    fi
    
    echo "Validating operations..."
    
    # Verify that the generated CUDA code is syntactically correct
    if [ -f "generated_kernels.cu" ]; then
        # Verify that it has basic structures
        if grep -q "__global__" generated_kernels.cu && \
           grep -q "#include" generated_kernels.cu; then
            echo -e "${GREEN}✓ Valid CUDA structure${NC}"
        else
            echo -e "${RED}✗ Invalid CUDA structure${NC}"
            return 1
        fi
   
        if grep -q "@matmul" "$test_file"; then
            if grep -q "matmul_kernel" generated_kernels.cu; then
                echo -e "${GREEN}✓ Matrix multiplication kernel found${NC}"
            else
                echo -e "${YELLOW}⚠ Matrix multiplication kernel not found${NC}"
            fi
        fi
        
        if grep -q "@transpose" "$test_file"; then
            if grep -q "transpose_kernel" generated_kernels.cu; then
                echo -e "${GREEN}✓ Transpose kernel found${NC}"
            else
                echo -e "${YELLOW}⚠ Transpose kernel not found${NC}"
            fi
        fi
        
        if grep -q "@reduce" "$test_file"; then
            if grep -q "reduce_kernel" generated_kernels.cu; then
                echo -e "${GREEN}✓ Reduce kernel found${NC}"
            else
                echo -e "${YELLOW}⚠ Reduce kernel not found${NC}"
            fi
        fi
    fi
    
    return 0
}

# List of tests to run
TESTS=(
    "tests/pruebaT51.txt"
    "tests/pruebaT52.txt"
    "tests/pruebaT53.txt"
    "tests/pruebaT54.txt"
    "tests/pruebaT55.txt"
    "tests/simple_test.txt"
    "tests/test_algebraic.txt"
    "tests/test_cse.txt"
    "tests/test_dce.txt"
    "tests/test_fusion.txt"
    "tests/test_memory.txt"
    "tests/test_simple.txt"
    "tests/test_tensor.txt"
    "tests/test_tensor2.txt"
    "tests/test_tensor3.txt"
)

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Execute validation
echo "Starting test validation..."
echo ""

for test_file in "${TESTS[@]}"; do
    if [ ! -f "$test_file" ]; then
        echo -e "${YELLOW}⚠ Test not found: $test_file${NC}"
        continue
    fi
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if validate_test "$test_file"; then
        validate_operations "$test_file"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "${GREEN}✓ Test PASSED${NC}"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo -e "${RED}✗ Test FAILED${NC}"
    fi
    
    echo ""
done

# Summary
echo "=========================================="
echo "VALIDATION SUMMARY"
echo "=========================================="
echo "Total tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"

if [ $TOTAL_TESTS -gt 0 ]; then
    SUCCESS_RATE=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)
    echo "Success rate: ${SUCCESS_RATE}%"
fi

echo "=========================================="

# Exit with appropriate status
if [ $FAILED_TESTS -eq 0 ]; then
    exit 0
else
    exit 1
fi
