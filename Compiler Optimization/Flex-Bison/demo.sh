#!/bin/bash
# demo.sh - Quick LLVM Integration Demo
# Demonstrates the dual backend architecture (CUDA + LLVM IR)

set -e

echo "========================================"
echo "   LLVM Integration Demo"
echo "   Tensor Compiler - Dual Backend"
echo "========================================"
echo ""

# Step 1: Compile
echo "Step 1: Compiling tensor program..."
echo "   Input: tests/test_tensor.txt"
./compiler < tests/test_tensor.txt > /dev/null 2>&1
echo "   ✓ Generated: generated_kernels.cu (CUDA backend)"
echo "   ✓ Generated: tensor_output.ll (LLVM IR backend)"
echo ""

# Step 2: Show LLVM IR structure
echo "Step 2: LLVM IR Preview (first 25 lines):"
echo "----------------------------------------"
head -25 tensor_output.ll
echo "..."
echo ""

# Step 3: Verify IR validity
echo "Step 3: Verifying IR validity..."
source llvm_config.sh > /dev/null 2>&1
if "$LLVM_AS" < tensor_output.ll > /dev/null 2>&1; then
    echo "   ✓ LLVM IR is valid (passed llvm-as verification)"
else
    echo "   ✗ LLVM IR verification failed"
    exit 1
fi
echo ""

# Step 4: Run analysis pipeline
echo "Step 4: Running LLVM optimization pipeline..."
echo "   Stages: DCE → CSE → InstCombine → Mem2Reg → O3 → CFG"
./llvm_analyze.sh tensor_output.ll llvm_output > /dev/null 2>&1
echo "   ✓ Analysis complete (9 stages executed)"
echo ""

# Step 5: Show optimization results
echo "Step 5: Optimization Results:"
echo "----------------------------------------"
cat llvm_output/tensor_output_summary.txt
echo ""

# Step 6: List generated files
echo "Step 6: Generated Analysis Files:"
echo "----------------------------------------"
echo "LLVM IR files:"
ls -lh llvm_output/*.ll 2>/dev/null | awk '{printf "   %-35s %8s\n", $9, $5}'
echo ""
echo "Metrics and reports:"
ls -lh llvm_output/*.txt 2>/dev/null | awk '{printf "   %-35s %8s\n", $9, $5}'
echo ""
echo "CFG visualization:"
ls -lh llvm_output/*.dot 2>/dev/null | awk '{printf "   %-35s %8s\n", $9, $5}'
echo ""

# Step 7: Show key metrics
echo "Step 7: Key Metrics Summary:"
echo "----------------------------------------"
if [ -f "llvm_output/tensor_output_O0_metrics.txt" ]; then
    echo "Baseline metrics:"
    grep -E "(Instructions|Function calls|Tensor allocations)" llvm_output/tensor_output_O0_metrics.txt | sed 's/^/   /'
fi
echo ""

# Step 8: Compare IR sizes
echo "Step 8: IR Size Comparison:"
echo "----------------------------------------"
ORIGINAL_SIZE=$(wc -c < tensor_output.ll)
O3_SIZE=$(wc -c < llvm_output/tensor_output_O3.ll 2>/dev/null || echo "0")
printf "   Original IR:  %8d bytes\n" $ORIGINAL_SIZE
printf "   Optimized IR: %8d bytes\n" $O3_SIZE
if [ $O3_SIZE -gt 0 ]; then
    REDUCTION=$((100 - (O3_SIZE * 100 / ORIGINAL_SIZE)))
    printf "   Reduction:    %7d%%\n" $REDUCTION
fi
echo ""

echo "========================================"
echo "   Demo Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  • View detailed IR:     less tensor_output.ll"
echo "  • Compare optimizations: diff tensor_output.ll llvm_output/tensor_output_O3.ll"
echo "  • View CFG graph:       dot -Tpng llvm_output/.main.dot -o cfg.png"
echo "  • Run other tests:      ./compiler < tests/test_dce.txt"
echo "  • Full analysis:        cat llvm_output/tensor_output_analysis.log"
echo ""
