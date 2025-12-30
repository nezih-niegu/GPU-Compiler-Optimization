#!/bin/bash
# LLVM Analysis Pipeline
# Runs LLVM optimization passes and extracts analysis metrics

set -e

# Source LLVM configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/llvm_config.sh"

# Input/output files
INPUT_LL="$1"
OUTPUT_DIR="${2:-.}"

if [ -z "$INPUT_LL" ]; then
    echo "Usage: $0 <input.ll> [output_dir]"
    echo ""
    echo "Runs LLVM analysis passes on generated IR and extracts metrics"
    exit 1
fi

if [ ! -f "$INPUT_LL" ]; then
    echo "Error: Input file not found: $INPUT_LL"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

BASENAME=$(basename "$INPUT_LL" .ll)
LOG_FILE="$OUTPUT_DIR/${BASENAME}_analysis.log"

echo "========================================" | tee "$LOG_FILE"
echo "LLVM Analysis Pipeline" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Input:  $INPUT_LL" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 1: Verify Input IR
# ========================================
echo "[1/9] Verifying input IR..." | tee -a "$LOG_FILE"
if ! "$LLVM_AS" < "$INPUT_LL" > /dev/null 2>&1; then
    echo "✗ ERROR: Invalid LLVM IR" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✓ IR is valid" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 2: Baseline Metrics (Unoptimized)
# ========================================
echo "[2/9] Extracting baseline metrics..." | tee -a "$LOG_FILE"
BASELINE_METRICS="$OUTPUT_DIR/${BASENAME}_O0_metrics.txt"

echo "=== Baseline (Unoptimized) Metrics ===" > "$BASELINE_METRICS"
echo "" >> "$BASELINE_METRICS"

# Instruction count
INSTR_COUNT_O0=$(grep -c '^\s*%' "$INPUT_LL" || echo 0)
echo "Instructions: $INSTR_COUNT_O0" >> "$BASELINE_METRICS"

# Load/Store count
LOAD_COUNT_O0=$(grep -c 'load ' "$INPUT_LL" 2>/dev/null || echo 0)
LOAD_COUNT_O0=$(echo "$LOAD_COUNT_O0" | tr -d '\n' | tr -d ' ')
STORE_COUNT_O0=$(grep -c 'store ' "$INPUT_LL" 2>/dev/null || echo 0)
STORE_COUNT_O0=$(echo "$STORE_COUNT_O0" | tr -d '\n' | tr -d ' ')
echo "Load operations: $LOAD_COUNT_O0" >> "$BASELINE_METRICS"
echo "Store operations: $STORE_COUNT_O0" >> "$BASELINE_METRICS"
TOTAL_MEM=$((LOAD_COUNT_O0 + STORE_COUNT_O0))
echo "Total memory ops: $TOTAL_MEM" >> "$BASELINE_METRICS"

# Function call count
CALL_COUNT_O0=$(grep -c 'call ' "$INPUT_LL" || echo 0)
echo "Function calls: $CALL_COUNT_O0" >> "$BASELINE_METRICS"

# Alloc count (tensor allocations)
ALLOC_COUNT_O0=$(grep -c '@tensor_alloc' "$INPUT_LL" || echo 0)
echo "Tensor allocations: $ALLOC_COUNT_O0" >> "$BASELINE_METRICS"

cat "$BASELINE_METRICS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 3: Dead Code Elimination (DCE)
# ========================================
echo "[3/9] Applying Dead Code Elimination..." | tee -a "$LOG_FILE"
DCE_LL="$OUTPUT_DIR/${BASENAME}_dce.ll"
"$LLVM_OPT" -passes=dce -S "$INPUT_LL" -o "$DCE_LL" 2>&1 | tee -a "$LOG_FILE"

INSTR_COUNT_DCE=$(grep -c '^\s*%' "$DCE_LL" || echo 0)
DCE_REDUCTION=$((INSTR_COUNT_O0 - INSTR_COUNT_DCE))
echo "✓ Instructions after DCE: $INSTR_COUNT_DCE (removed: $DCE_REDUCTION)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 4: Common Subexpression Elimination (CSE)
# ========================================
echo "[4/9] Applying Early CSE..." | tee -a "$LOG_FILE"
CSE_LL="$OUTPUT_DIR/${BASENAME}_cse.ll"
"$LLVM_OPT" -passes=early-cse -S "$DCE_LL" -o "$CSE_LL" 2>&1 | tee -a "$LOG_FILE"

INSTR_COUNT_CSE=$(grep -c '^\s*%' "$CSE_LL" || echo 0)
CSE_REDUCTION=$((INSTR_COUNT_DCE - INSTR_COUNT_CSE))
echo "✓ Instructions after CSE: $INSTR_COUNT_CSE (removed: $CSE_REDUCTION)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 5: Instruction Combining (Algebraic)
# ========================================
echo "[5/9] Applying Instruction Combining..." | tee -a "$LOG_FILE"
INSTCOMBINE_LL="$OUTPUT_DIR/${BASENAME}_instcombine.ll"
"$LLVM_OPT" -passes=instcombine -S "$CSE_LL" -o "$INSTCOMBINE_LL" 2>&1 | tee -a "$LOG_FILE"

INSTR_COUNT_IC=$(grep -c '^\s*%' "$INSTCOMBINE_LL" || echo 0)
IC_REDUCTION=$((INSTR_COUNT_CSE - INSTR_COUNT_IC))
echo "✓ Instructions after instcombine: $INSTR_COUNT_IC (removed: $IC_REDUCTION)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 6: Memory-to-Register Promotion
# ========================================
echo "[6/9] Applying Memory-to-Register Promotion..." | tee -a "$LOG_FILE"
MEM2REG_LL="$OUTPUT_DIR/${BASENAME}_mem2reg.ll"
"$LLVM_OPT" -passes=mem2reg -S "$INSTCOMBINE_LL" -o "$MEM2REG_LL" 2>&1 | tee -a "$LOG_FILE"

LOAD_COUNT_M2R=$(grep -c 'load ' "$MEM2REG_LL" 2>/dev/null || echo 0)
LOAD_COUNT_M2R=$(echo "$LOAD_COUNT_M2R" | tr -d '\n' | tr -d ' ')
STORE_COUNT_M2R=$(grep -c 'store ' "$MEM2REG_LL" 2>/dev/null || echo 0)
STORE_COUNT_M2R=$(echo "$STORE_COUNT_M2R" | tr -d '\n' | tr -d ' ')
MEMORY_REDUCTION=$((LOAD_COUNT_O0 + STORE_COUNT_O0 - LOAD_COUNT_M2R - STORE_COUNT_M2R))
TOTAL_MEM_M2R=$((LOAD_COUNT_M2R + STORE_COUNT_M2R))
echo "✓ Memory ops after mem2reg: $TOTAL_MEM_M2R (removed: $MEMORY_REDUCTION)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 7: Full -O3 Optimization
# ========================================
echo "[7/9] Applying full -O3 optimization..." | tee -a "$LOG_FILE"
O3_LL="$OUTPUT_DIR/${BASENAME}_O3.ll"
"$LLVM_OPT" -O3 -S "$INPUT_LL" -o "$O3_LL" 2>&1 | tee -a "$LOG_FILE"

INSTR_COUNT_O3=$(grep -c '^\s*%' "$O3_LL" || echo 0)
O3_REDUCTION=$((INSTR_COUNT_O0 - INSTR_COUNT_O3))
O3_PERCENT=$(awk "BEGIN {printf \"%.1f\", 100.0 * $O3_REDUCTION / $INSTR_COUNT_O0}")
echo "✓ Instructions after -O3: $INSTR_COUNT_O3 (removed: $O3_REDUCTION, ${O3_PERCENT}%)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 8: Control Flow Graph Analysis
# ========================================
echo "[8/9] Generating Control Flow Graph..." | tee -a "$LOG_FILE"
CFG_DOT="$OUTPUT_DIR/${BASENAME}_cfg.dot"
"$LLVM_OPT" -passes=dot-cfg -disable-output "$O3_LL" 2>&1 | tee -a "$LOG_FILE"

# Move generated .dot files
if ls .*.dot >/dev/null 2>&1; then
    mv .*.dot "$OUTPUT_DIR/" 2>/dev/null || true
    echo "✓ CFG generated in $OUTPUT_DIR" | tee -a "$LOG_FILE"
else
    echo "⚠ No CFG generated (no functions with multiple blocks)" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 9: Generate Summary Report
# ========================================
echo "[9/9] Generating summary report..." | tee -a "$LOG_FILE"
SUMMARY="$OUTPUT_DIR/${BASENAME}_summary.txt"

cat > "$SUMMARY" << EOF
========================================
LLVM Analysis Summary
========================================
Input: $INPUT_LL
Date: $(date)

OPTIMIZATION METRICS
────────────────────────────────────────
Stage                Instructions    Reduction
────────────────────────────────────────
Baseline (O0)        $INSTR_COUNT_O0          -
After DCE            $INSTR_COUNT_DCE          $DCE_REDUCTION
After CSE            $INSTR_COUNT_CSE          $CSE_REDUCTION
After InstCombine    $INSTR_COUNT_IC          $IC_REDUCTION
After -O3            $INSTR_COUNT_O3          $O3_REDUCTION (${O3_PERCENT}%)
────────────────────────────────────────

MEMORY OPERATIONS
────────────────────────────────────────
                     Baseline        After mem2reg
Load operations      $LOAD_COUNT_O0             $LOAD_COUNT_M2R
Store operations     $STORE_COUNT_O0             $STORE_COUNT_M2R
Total                $((LOAD_COUNT_O0 + STORE_COUNT_O0))             $((LOAD_COUNT_M2R + STORE_COUNT_M2R))
Reduction: $MEMORY_REDUCTION operations
────────────────────────────────────────

FUNCTION CALLS
────────────────────────────────────────
Total calls:         $CALL_COUNT_O0
Tensor allocations:  $ALLOC_COUNT_O0
────────────────────────────────────────

GENERATED FILES
────────────────────────────────────────
• ${BASENAME}_dce.ll           - After Dead Code Elimination
• ${BASENAME}_cse.ll           - After Common Subexpr Elimination
• ${BASENAME}_instcombine.ll   - After Instruction Combining
• ${BASENAME}_mem2reg.ll       - After Memory-to-Register
• ${BASENAME}_O3.ll            - Full -O3 optimization
• ${BASENAME}_summary.txt      - This summary
• ${BASENAME}_analysis.log     - Full analysis log
────────────────────────────────────────

LLVM TOOLS USED
────────────────────────────────────────
clang: $LLVM_CLANG
opt:   $LLVM_OPT
llc:   $LLVM_LLC
────────────────────────────────────────
EOF

cat "$SUMMARY" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "✓ Analysis complete!" | tee -a "$LOG_FILE"
echo "  Summary: $SUMMARY" | tee -a "$LOG_FILE"
echo "  Log:     $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
