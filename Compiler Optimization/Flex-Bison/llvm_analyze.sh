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

# Start overall timer
TOTAL_START=$(date +%s.%N)

echo "========================================" | tee "$LOG_FILE"
echo "LLVM Analysis Pipeline" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Input:  $INPUT_LL" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 1: Verify Input IR
# ========================================
STAGE1_START=$(date +%s.%N)
echo "[1/9] Verifying input IR..." | tee -a "$LOG_FILE"
if ! "$LLVM_AS" < "$INPUT_LL" > /dev/null 2>&1; then
    echo "✗ ERROR: Invalid LLVM IR" | tee -a "$LOG_FILE"
    exit 1
fi
STAGE1_END=$(date +%s.%N)
STAGE1_TIME=$(awk "BEGIN {printf \"%.3f\", $STAGE1_END - $STAGE1_START}")
echo "✓ IR is valid (${STAGE1_TIME}s)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 2: Baseline Metrics (Unoptimized)
# ========================================
STAGE2_START=$(date +%s.%N)
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
STAGE2_END=$(date +%s.%N)
STAGE2_TIME=$(awk "BEGIN {printf \"%.3f\", $STAGE2_END - $STAGE2_START}")
echo "✓ Baseline metrics extracted (${STAGE2_TIME}s)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 3: Dead Code Elimination (DCE)
# ========================================
STAGE3_START=$(date +%s.%N)
echo "[3/9] Applying Dead Code Elimination..." | tee -a "$LOG_FILE"
DCE_LL="$OUTPUT_DIR/${BASENAME}_dce.ll"
"$LLVM_OPT" -passes=dce -S "$INPUT_LL" -o "$DCE_LL" 2>&1 | tee -a "$LOG_FILE"

INSTR_COUNT_DCE=$(grep -c '^\s*%' "$DCE_LL" || echo 0)
DCE_REDUCTION=$((INSTR_COUNT_O0 - INSTR_COUNT_DCE))
STAGE3_END=$(date +%s.%N)
STAGE3_TIME=$(awk "BEGIN {printf \"%.3f\", $STAGE3_END - $STAGE3_START}")
echo "✓ Instructions after DCE: $INSTR_COUNT_DCE (removed: $DCE_REDUCTION) [${STAGE3_TIME}s]" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 4: Common Subexpression Elimination (CSE)
# ========================================
STAGE4_START=$(date +%s.%N)
echo "[4/9] Applying Early CSE..." | tee -a "$LOG_FILE"
CSE_LL="$OUTPUT_DIR/${BASENAME}_cse.ll"
"$LLVM_OPT" -passes=early-cse -S "$DCE_LL" -o "$CSE_LL" 2>&1 | tee -a "$LOG_FILE"

INSTR_COUNT_CSE=$(grep -c '^\s*%' "$CSE_LL" || echo 0)
CSE_REDUCTION=$((INSTR_COUNT_DCE - INSTR_COUNT_CSE))
STAGE4_END=$(date +%s.%N)
STAGE4_TIME=$(awk "BEGIN {printf \"%.3f\", $STAGE4_END - $STAGE4_START}")
echo "✓ Instructions after CSE: $INSTR_COUNT_CSE (removed: $CSE_REDUCTION) [${STAGE4_TIME}s]" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 5: Instruction Combining (Algebraic)
# ========================================
STAGE5_START=$(date +%s.%N)
echo "[5/9] Applying Instruction Combining..." | tee -a "$LOG_FILE"
INSTCOMBINE_LL="$OUTPUT_DIR/${BASENAME}_instcombine.ll"
"$LLVM_OPT" -passes=instcombine -S "$CSE_LL" -o "$INSTCOMBINE_LL" 2>&1 | tee -a "$LOG_FILE"

INSTR_COUNT_IC=$(grep -c '^\s*%' "$INSTCOMBINE_LL" || echo 0)
IC_REDUCTION=$((INSTR_COUNT_CSE - INSTR_COUNT_IC))
STAGE5_END=$(date +%s.%N)
STAGE5_TIME=$(awk "BEGIN {printf \"%.3f\", $STAGE5_END - $STAGE5_START}")
echo "✓ Instructions after instcombine: $INSTR_COUNT_IC (removed: $IC_REDUCTION) [${STAGE5_TIME}s]" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 6: Memory-to-Register Promotion
# ========================================
STAGE6_START=$(date +%s.%N)
echo "[6/9] Applying Memory-to-Register Promotion..." | tee -a "$LOG_FILE"
MEM2REG_LL="$OUTPUT_DIR/${BASENAME}_mem2reg.ll"
"$LLVM_OPT" -passes=mem2reg -S "$INSTCOMBINE_LL" -o "$MEM2REG_LL" 2>&1 | tee -a "$LOG_FILE"

LOAD_COUNT_M2R=$(grep -c 'load ' "$MEM2REG_LL" 2>/dev/null || echo 0)
LOAD_COUNT_M2R=$(echo "$LOAD_COUNT_M2R" | tr -d '\n' | tr -d ' ')
STORE_COUNT_M2R=$(grep -c 'store ' "$MEM2REG_LL" 2>/dev/null || echo 0)
STORE_COUNT_M2R=$(echo "$STORE_COUNT_M2R" | tr -d '\n' | tr -d ' ')
MEMORY_REDUCTION=$((LOAD_COUNT_O0 + STORE_COUNT_O0 - LOAD_COUNT_M2R - STORE_COUNT_M2R))
TOTAL_MEM_M2R=$((LOAD_COUNT_M2R + STORE_COUNT_M2R))
STAGE6_END=$(date +%s.%N)
STAGE6_TIME=$(awk "BEGIN {printf \"%.3f\", $STAGE6_END - $STAGE6_START}")
echo "✓ Memory ops after mem2reg: $TOTAL_MEM_M2R (removed: $MEMORY_REDUCTION) [${STAGE6_TIME}s]" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 7: Full -O3 Optimization
# ========================================
STAGE7_START=$(date +%s.%N)
echo "[7/9] Applying full -O3 optimization..." | tee -a "$LOG_FILE"
O3_LL="$OUTPUT_DIR/${BASENAME}_O3.ll"
"$LLVM_OPT" -O3 -S "$INPUT_LL" -o "$O3_LL" 2>&1 | tee -a "$LOG_FILE"

INSTR_COUNT_O3=$(grep -c '^\s*%' "$O3_LL" || echo 0)
O3_REDUCTION=$((INSTR_COUNT_O0 - INSTR_COUNT_O3))
O3_PERCENT=$(awk "BEGIN {printf \"%.1f\", 100.0 * $O3_REDUCTION / $INSTR_COUNT_O0}")
STAGE7_END=$(date +%s.%N)
STAGE7_TIME=$(awk "BEGIN {printf \"%.3f\", $STAGE7_END - $STAGE7_START}")
echo "✓ Instructions after -O3: $INSTR_COUNT_O3 (removed: $O3_REDUCTION, ${O3_PERCENT}%) [${STAGE7_TIME}s]" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 8: Control Flow Graph Analysis
# ========================================
STAGE8_START=$(date +%s.%N)
echo "[8/9] Generating Control Flow Graph..." | tee -a "$LOG_FILE"
CFG_DOT="$OUTPUT_DIR/${BASENAME}_cfg.dot"
"$LLVM_OPT" -passes=dot-cfg -disable-output "$O3_LL" 2>&1 | tee -a "$LOG_FILE"

# Move generated .dot files
if ls .*.dot >/dev/null 2>&1; then
    mv .*.dot "$OUTPUT_DIR/" 2>/dev/null || true
    STAGE8_END=$(date +%s.%N)
    STAGE8_TIME=$(awk "BEGIN {printf \"%.3f\", $STAGE8_END - $STAGE8_START}")
    echo "✓ CFG generated in $OUTPUT_DIR [${STAGE8_TIME}s]" | tee -a "$LOG_FILE"
else
    STAGE8_END=$(date +%s.%N)
    STAGE8_TIME=$(awk "BEGIN {printf \"%.3f\", $STAGE8_END - $STAGE8_START}")
    echo "⚠ No CFG generated (no functions with multiple blocks) [${STAGE8_TIME}s]" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

# ========================================
# Stage 9: Generate Summary Report
# ========================================
STAGE9_START=$(date +%s.%N)
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
Stage                Instructions    Reduction    Time (s)
────────────────────────────────────────
Baseline (O0)        $INSTR_COUNT_O0          -            -
After DCE            $INSTR_COUNT_DCE          $DCE_REDUCTION            $STAGE3_TIME
After CSE            $INSTR_COUNT_CSE          $CSE_REDUCTION            $STAGE4_TIME
After InstCombine    $INSTR_COUNT_IC          $IC_REDUCTION            $STAGE5_TIME
After -O3            $INSTR_COUNT_O3          $O3_REDUCTION (${O3_PERCENT}%)    $STAGE7_TIME
────────────────────────────────────────

MEMORY OPERATIONS
────────────────────────────────────────
                     Baseline        After mem2reg    Time (s)
Load operations      $LOAD_COUNT_O0             $LOAD_COUNT_M2R               $STAGE6_TIME
Store operations     $STORE_COUNT_O0             $STORE_COUNT_M2R               
Total                $((LOAD_COUNT_O0 + STORE_COUNT_O0))             $((LOAD_COUNT_M2R + STORE_COUNT_M2R))               
Reduction: $MEMORY_REDUCTION operations
────────────────────────────────────────

FUNCTION CALLS
────────────────────────────────────────
Total calls:         $CALL_COUNT_O0
Tensor allocations:  $ALLOC_COUNT_O0
────────────────────────────────────────

PHASE TIMING
────────────────────────────────────────
Phase                                    Time (s)
────────────────────────────────────────
[1/9] Verify Input IR                   $STAGE1_TIME
[2/9] Extract Baseline Metrics          $STAGE2_TIME
[3/9] Dead Code Elimination             $STAGE3_TIME
[4/9] Common Subexpression Elimination  $STAGE4_TIME
[5/9] Instruction Combining             $STAGE5_TIME
[6/9] Memory-to-Register Promotion      $STAGE6_TIME
[7/9] Full -O3 Optimization             $STAGE7_TIME
[8/9] Control Flow Graph Generation     $STAGE8_TIME
[9/9] Summary Generation                (current)
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
STAGE9_END=$(date +%s.%N)
STAGE9_TIME=$(awk "BEGIN {printf \"%.3f\", $STAGE9_END - $STAGE9_START}")
echo "" | tee -a "$LOG_FILE"

# Calculate total time
TOTAL_END=$(date +%s.%N)
TOTAL_TIME=$(awk "BEGIN {printf \"%.3f\", $TOTAL_END - $TOTAL_START}")

echo "========================================" | tee -a "$LOG_FILE"
echo "✓ Analysis complete!" | tee -a "$LOG_FILE"
echo "  Total time: ${TOTAL_TIME}s" | tee -a "$LOG_FILE"
echo "  Summary: $SUMMARY" | tee -a "$LOG_FILE"
echo "  Log:     $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
