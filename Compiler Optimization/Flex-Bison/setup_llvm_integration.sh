#!/bin/bash
# LLVM Integration Quick Start
# This script helps you integrate LLVM into your tensor compiler

echo "========================================="
echo "LLVM Integration for Tensor Compiler"
echo "========================================="
echo ""

# Step 1: Verify LLVM installation
echo "[Step 1/5] Verifying LLVM installation..."
source llvm_config.sh
if [ $? -ne 0 ]; then
    echo "✗ LLVM configuration failed"
    echo "  Please edit llvm_config.sh to point to your LLVM installation"
    exit 1
fi
echo "✓ LLVM tools found"
echo ""

# Step 2: Build compiler with LLVM support
echo "[Step 2/5] Building compiler with LLVM support..."
make clean
make
if [ $? -ne 0 ]; then
    echo "✗ Build failed"
    exit 1
fi
echo "✓ Compiler built successfully"
echo ""

# Step 3: Add LLVM lowering to parser
echo "[Step 3/5] Checking parser integration..."
if grep -q "llvm_lowering.h" finalAssignment.y; then
    echo "✓ Parser already includes LLVM lowering"
else
    echo "⚠ Parser needs manual integration"
    echo ""
    echo "Add to finalAssignment.y header section (after #include \"cuda_gen.h\"):"
    echo "────────────────────────────────────────"
    echo '#include "llvm_lowering.h"'
    echo "────────────────────────────────────────"
    echo ""
    echo "Add to start_program action (after generate_cuda_code line):"
    echo "────────────────────────────────────────"
    echo '        /* Generate LLVM IR */'
    echo '        if (lower_graph_to_llvm_ir(tensorGraph, "tensor_output.ll") == 0) {'
    echo '            printf("\nLLVM IR generated in: tensor_output.ll\n");'
    echo '        } else {'
    echo '            fprintf(stderr, "Warning: LLVM IR generation failed\n");'
    echo '        }'
    echo "────────────────────────────────────────"
    echo ""
    read -p "Press Enter after you've made these changes..."
fi
echo ""

# Step 4: Run simple test
echo "[Step 4/5] Running simple test..."
if [ ! -f "tests/simple_test.txt" ]; then
    echo "✗ Test file not found: tests/simple_test.txt"
    exit 1
fi

./compiler < tests/simple_test.txt > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "✗ Compiler failed on test"
    exit 1
fi

if [ ! -f "tensor_output.ll" ]; then
    echo "✗ No LLVM IR generated"
    echo "  Make sure you added the LLVM lowering call to finalAssignment.y"
    exit 1
fi

echo "✓ LLVM IR generated: tensor_output.ll"
echo ""

# Step 5: Verify and analyze IR
echo "[Step 5/5] Verifying and analyzing IR..."
./llvm_analyze.sh tensor_output.ll llvm_output/
if [ $? -ne 0 ]; then
    echo "✗ Analysis failed"
    exit 1
fi

echo "✓ Analysis complete"
echo ""

# Success!
echo "========================================="
echo "✓ LLVM Integration Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Review generated IR:    less tensor_output.ll"
echo "2. View analysis results:  cat llvm_output/tensor_output_summary.txt"
echo "3. Run all tests:          ./run_llvm_tests.sh"
echo "4. Read documentation:     less ../../../LLVM_INTEGRATION.md"
echo ""
echo "Quick reference:"
echo "• Generate IR:    ./compiler < tests/test_dce.txt"
echo "• Analyze IR:     ./llvm_analyze.sh tensor_output.ll"
echo "• Run all tests:  ./run_llvm_tests.sh"
echo ""
