#!/bin/bash
# Run all LLVM integration tests
# Tests the 5 core test cases from TESTS_AND_BENCHMARKS.md

set -e

echo "========================================"
echo "LLVM Integration Test Suite"
echo "========================================"
echo ""

# Source LLVM configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/llvm_config.sh"

# Create output directory
OUTPUT_ROOT="llvm_test_results"
rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

# Test cases
TESTS=(
    "test_dce:Dead Code Elimination"
    "test_cse:Common Subexpression Elimination"
    "test_algebraic:Algebraic Simplification"
    "test_fusion:Operation Fusion"
    "test_memory:Memory Layout Optimization"
)

RESULTS_FILE="$OUTPUT_ROOT/test_summary.txt"
echo "LLVM Integration Test Results" > "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Track results
PASSED=0
FAILED=0

for test_case in "${TESTS[@]}"; do
    TEST_NAME="${test_case%%:*}"
    TEST_DESC="${test_case##*:}"
    
    echo "----------------------------------------"
    echo "Test: $TEST_DESC ($TEST_NAME)"
    echo "----------------------------------------"
    
    TEST_DIR="$OUTPUT_ROOT/$TEST_NAME"
    mkdir -p "$TEST_DIR"
    
    # Run compiler
    echo "  [1/3] Generating LLVM IR..."
    if ./compiler < "tests/${TEST_NAME}.txt" > "$TEST_DIR/compiler_output.log" 2>&1; then
        echo "  ✓ Compiler succeeded"
    else
        echo "  ✗ Compiler failed"
        echo "$TEST_NAME: FAILED (compiler error)" >> "$RESULTS_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Check if IR was generated
    if [ ! -f "tensor_output.ll" ]; then
        echo "  ✗ No LLVM IR generated"
        echo "$TEST_NAME: FAILED (no IR)" >> "$RESULTS_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    mv tensor_output.ll "$TEST_DIR/"
    
    # Verify IR
    echo "  [2/3] Verifying LLVM IR..."
    if "$LLVM_AS" < "$TEST_DIR/tensor_output.ll" > /dev/null 2>&1; then
        echo "  ✓ IR is valid"
    else
        echo "  ✗ Invalid LLVM IR"
        echo "$TEST_NAME: FAILED (invalid IR)" >> "$RESULTS_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Run analysis
    echo "  [3/3] Running LLVM analysis..."
    if ./llvm_analyze.sh "$TEST_DIR/tensor_output.ll" "$TEST_DIR" > /dev/null 2>&1; then
        echo "  ✓ Analysis complete"
        
        # Extract key metrics
        if [ -f "$TEST_DIR/tensor_output_summary.txt" ]; then
            INSTR_O0=$(grep "Baseline (O0)" "$TEST_DIR/tensor_output_summary.txt" | awk '{print $3}')
            INSTR_O3=$(grep "After -O3" "$TEST_DIR/tensor_output_summary.txt" | awk '{print $3}')
            REDUCTION=$(grep "After -O3" "$TEST_DIR/tensor_output_summary.txt" | grep -o '[0-9.]*%' || echo "N/A")
            
            echo "  Instructions: $INSTR_O0 → $INSTR_O3 (reduction: $REDUCTION)"
            echo "$TEST_NAME: PASSED - Reduction: $REDUCTION" >> "$RESULTS_FILE"
            PASSED=$((PASSED + 1))
        else
            echo "  ⚠ No summary generated"
            echo "$TEST_NAME: PASSED (no summary)" >> "$RESULTS_FILE"
            PASSED=$((PASSED + 1))
        fi
    else
        echo "  ✗ Analysis failed"
        echo "$TEST_NAME: FAILED (analysis error)" >> "$RESULTS_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    echo ""
done

# Final summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Passed: $PASSED / $((PASSED + FAILED))"
echo "Failed: $FAILED / $((PASSED + FAILED))"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Detailed outputs in: $OUTPUT_ROOT/"
echo ""

# Append summary to results file
echo "" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "Summary: $PASSED passed, $FAILED failed" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"

# Exit with failure if any tests failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
