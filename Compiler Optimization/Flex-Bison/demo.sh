#!/bin/bash
# demo.sh - LLVM Integration Demo (TIMED + TABLE + ROBUST + CFG-OPTIONAL)

set -euo pipefail

# ----------------------------
# Timing helpers + table data
# ----------------------------
now_ns() { date +%s%N; }
fmt_ms() {
  local ns="$1"
  local ms=$((ns / 1000000))
  local s=$((ms / 1000))
  local rem=$((ms % 1000))
  printf "%d.%03ds" "$s" "$rem"
}

TOTAL_NS=0
declare -a STEP_NAMES=()
declare -a STEP_DURS=()

step_start() { STEP_T0="$(now_ns)"; }
step_end() {
  local name="$1"
  local t1; t1="$(now_ns)"
  local dt=$((t1 - STEP_T0))
  STEP_NAMES+=("$name")
  STEP_DURS+=("$dt")
  TOTAL_NS=$((TOTAL_NS + dt))
  echo "   ⏱  Step time: $(fmt_ms "$dt")"
  echo ""
}

RUN_T0="$(now_ns)"

echo "========================================"
echo "   LLVM Integration Demo"
echo "   Tensor Compiler - Dual Backend"
echo "========================================"
echo ""

INPUT_FILE="${1:-tests/test_tensor.txt}"

COMPILER="./compiler"
LLVM_IR_OUT="tensor_output.ll"
CUDA_OUT="generated_kernels.cu"

LOG_DIR="demo_logs"
COMPILE_OUT="$LOG_DIR/compile.stdout.log"
COMPILE_ERR="$LOG_DIR/compile.stderr.log"

COMPILE_TIMEOUT_SECS=20
VERIFY_TIMEOUT_SECS=10

mkdir -p "$LOG_DIR"

# ----------------------------
# Step 1: Compile
# ----------------------------
echo "Step 1: Compiling tensor program..."
echo "   Input: $INPUT_FILE"
step_start

if [ ! -e "$COMPILER" ]; then
  echo "   ✗ Compiler not found: $COMPILER"
  exit 1
fi
if [ ! -x "$COMPILER" ]; then
  echo "   ✗ Compiler is not executable: $COMPILER"
  echo "     Try: chmod +x $COMPILER"
  exit 1
fi
if [ ! -f "$INPUT_FILE" ]; then
  echo "   ✗ Input file not found: $INPUT_FILE"
  exit 1
fi

if file "$COMPILER" | grep -qiE "shell script|text"; then
  echo "   ✗ $COMPILER does not look like a compiled binary (it's a script/text file)."
  file "$COMPILER" | sed 's/^/       /'
  exit 1
fi

set +e
timeout "${COMPILE_TIMEOUT_SECS}s" "$COMPILER" < "$INPUT_FILE" > "$COMPILE_OUT" 2> "$COMPILE_ERR"
rc=$?
set -e

if [ "$rc" -ne 0 ]; then
  if [ "$rc" -eq 124 ]; then
    echo "   ✗ Compile step timed out after ${COMPILE_TIMEOUT_SECS}s (likely a hang)"
  else
    echo "   ✗ Compile step failed (exit code: $rc)"
  fi
  echo ""
  echo "   --- stderr (last 120 lines) ---"
  tail -120 "$COMPILE_ERR" 2>/dev/null || true
  echo ""
  echo "   --- stdout (last 120 lines) ---"
  tail -120 "$COMPILE_OUT" 2>/dev/null || true
  echo ""
  echo "   Full logs:"
  echo "     $COMPILE_ERR"
  echo "     $COMPILE_OUT"
  exit "$rc"
fi

missing=0
if [ -f "$CUDA_OUT" ]; then
  echo "   ✓ Generated: $CUDA_OUT (CUDA backend)"
else
  echo "   ✗ Expected CUDA output missing: $CUDA_OUT"
  missing=1
fi

if [ -f "$LLVM_IR_OUT" ]; then
  echo "   ✓ Generated: $LLVM_IR_OUT (LLVM IR backend)"
else
  echo "   ✗ Expected LLVM IR output missing: $LLVM_IR_OUT"
  missing=1
fi

if [ "$missing" -ne 0 ]; then
  echo ""
  echo "   --- stderr (last 120 lines) ---"
  tail -120 "$COMPILE_ERR" 2>/dev/null || true
  echo ""
  echo "   --- stdout (last 120 lines) ---"
  tail -120 "$COMPILE_OUT" 2>/dev/null || true
  echo ""
  exit 1
fi

step_end "1) Compile tensor program"

# ----------------------------
# Step 2: IR Preview
# ----------------------------
echo "Step 2: LLVM IR Preview (first 25 lines):"
echo "----------------------------------------"
step_start
head -25 "$LLVM_IR_OUT"
echo "..."
step_end "2) LLVM IR preview"

# ----------------------------
# Step 3: Verify IR validity
# ----------------------------
echo "Step 3: Verifying IR validity..."
step_start

# shellcheck disable=SC1091
source ./llvm_config.sh

: "${LLVM_AS:=}"
if [ -z "$LLVM_AS" ]; then
  echo "   ✗ LLVM_AS is empty after sourcing llvm_config.sh"
  exit 1
fi

echo "   Using llvm-as: $LLVM_AS"

set +e
timeout "${VERIFY_TIMEOUT_SECS}s" "$LLVM_AS" < "$LLVM_IR_OUT" > /dev/null 2> "$LOG_DIR/llvm-as.stderr.log"
vrc=$?
set -e

if [ "$vrc" -eq 0 ]; then
  echo "   ✓ LLVM IR is valid (passed llvm-as verification)"
elif [ "$vrc" -eq 124 ]; then
  echo "   ✗ llvm-as timed out after ${VERIFY_TIMEOUT_SECS}s"
  tail -120 "$LOG_DIR/llvm-as.stderr.log" 2>/dev/null || true
  exit 1
else
  echo "   ✗ LLVM IR verification failed (exit code: $vrc)"
  tail -120 "$LOG_DIR/llvm-as.stderr.log" 2>/dev/null || true
  exit 1
fi

step_end "3) Verify IR validity"

# ----------------------------
# Step 4: Optimization pipeline
# ----------------------------
echo "Step 4: Running LLVM optimization pipeline..."
echo "   Stages: DCE → CSE → InstCombine → Mem2Reg → O3 → CFG"
step_start
./llvm_analyze.sh "$LLVM_IR_OUT" llvm_output > /dev/null 2>&1
echo "   ✓ Analysis complete (9 stages executed)"
step_end "4) Run optimization pipeline"

# ----------------------------
# Step 5: Optimization results
# ----------------------------
echo "Step 5: Optimization Results:"
echo "----------------------------------------"
step_start
cat llvm_output/tensor_output_summary.txt
step_end "5) Show optimization results"

# ----------------------------
# Step 6: List generated files (CFG optional)
# ----------------------------
echo "Step 6: Generated Analysis Files:"
echo "----------------------------------------"
step_start

echo "LLVM IR files:"
if ls llvm_output/*.ll >/dev/null 2>&1; then
  ls -lh llvm_output/*.ll | awk '{printf "   %-35s %8s\n", $9, $5}'
else
  echo "   (No .ll files found in llvm_output/)"
fi
echo ""

echo "Metrics and reports:"
if ls llvm_output/*.txt >/dev/null 2>&1; then
  ls -lh llvm_output/*.txt | awk '{printf "   %-35s %8s\n", $9, $5}'
else
  echo "   (No .txt files found in llvm_output/)"
fi
echo ""

echo "CFG visualization:"
if ls llvm_output/*.dot >/dev/null 2>&1; then
  ls -lh llvm_output/*.dot | awk '{printf "   %-35s %8s\n", $9, $5}'
else
  echo "   (No CFG .dot files generated)"
  echo "   Tip: ensure llvm_analyze.sh actually emits .dot via opt -dot-cfg / -dot-cfg-only and writes into llvm_output/"
fi

step_end "6) List generated analysis files"

# ----------------------------
# Step 7: Key metrics
# ----------------------------
echo "Step 7: Key Metrics Summary:"
echo "----------------------------------------"
step_start
if [ -f "llvm_output/tensor_output_O0_metrics.txt" ]; then
  echo "Baseline metrics:"
  grep -E "(Instructions|Function calls|Tensor allocations)" llvm_output/tensor_output_O0_metrics.txt | sed 's/^/   /' || true
else
  echo "   (No baseline metrics file found: llvm_output/tensor_output_O0_metrics.txt)"
fi
step_end "7) Key metrics summary"

# ----------------------------
# Step 8: IR size comparison
# ----------------------------
echo "Step 8: IR Size Comparison:"
echo "----------------------------------------"
step_start
ORIGINAL_SIZE=$(wc -c < "$LLVM_IR_OUT")
O3_SIZE=$(wc -c < llvm_output/tensor_output_O3.ll 2>/dev/null || echo "0")

printf "   Original IR:  %8d bytes\n" "$ORIGINAL_SIZE"
printf "   Optimized IR: %8d bytes\n" "$O3_SIZE"

if [ "$O3_SIZE" -gt 0 ]; then
  REDUCTION=$((100 - (O3_SIZE * 100 / ORIGINAL_SIZE)))
  printf "   Reduction:    %7d%%\n" "$REDUCTION"
fi
step_end "8) IR size comparison"

# ----------------------------
# Timing summary table
# ----------------------------
RUN_T1="$(now_ns)"
WALL_NS=$((RUN_T1 - RUN_T0))

echo "Timing Summary:"
echo "----------------------------------------"
printf "  %-38s %12s\n" "Step" "Time"
printf "  %-38s %12s\n" "--------------------------------------" "------------"
for i in "${!STEP_NAMES[@]}"; do
  printf "  %-38s %12s\n" "${STEP_NAMES[$i]}" "$(fmt_ms "${STEP_DURS[$i]}")"
done
printf "  %-38s %12s\n" "--------------------------------------" "------------"
printf "  %-38s %12s\n" "Sum of step times" "$(fmt_ms "$TOTAL_NS")"
printf "  %-38s %12s\n" "Wall-clock time"   "$(fmt_ms "$WALL_NS")"
echo ""

echo "========================================"
echo "   Demo Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  • View detailed IR:      less $LLVM_IR_OUT"
echo "  • Compare optimizations: diff $LLVM_IR_OUT llvm_output/tensor_output_O3.ll"
echo "  • View CFG graph:        dot -Tpng llvm_output/.main.dot -o cfg.png"
echo "  • Run other tests:       $COMPILER < tests/test_dce.txt"
echo "  • Full analysis:         cat llvm_output/tensor_output_analysis.log"
echo ""
echo "Logs (compile step):"
echo "  • stdout: $COMPILE_OUT"
echo "  • stderr: $COMPILE_ERR"
echo ""
