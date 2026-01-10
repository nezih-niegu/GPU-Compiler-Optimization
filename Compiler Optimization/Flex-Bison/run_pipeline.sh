#!/bin/bash
# run_pipeline.sh - Unified Validation + LLVM Demo Pipeline + CSV timings

set -euo pipefail

# ----------------------------
# Colors
# ----------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ----------------------------
# Timing helpers
# ----------------------------
now_ns() { date +%s%N; }
ns_to_ms() { awk -v ns="$1" 'BEGIN{printf "%.3f", ns/1000000.0}'; }

# ----------------------------
# Config
# ----------------------------
COMPILER="./compiler"
LLVM_IR_OUT="tensor_output.ll"
CUDA_OUT="generated_kernels.cu"

LOG_DIR="pipeline_logs"
LLVM_DIR="llvm_output"
CSV_OUT="validation_times.csv"

COMPILE_TIMEOUT_SECS="${COMPILE_TIMEOUT_SECS:-20}"
VERIFY_TIMEOUT_SECS="${VERIFY_TIMEOUT_SECS:-10}"

# Control flags (override via env)
#   SKIP_LLVM_VERIFY=1   -> skip llvm-as verification
#   SKIP_LLVM_ANALYZE=1  -> skip llvm_analyze.sh stage
SKIP_LLVM_VERIFY="${SKIP_LLVM_VERIFY:-0}"
SKIP_LLVM_ANALYZE="${SKIP_LLVM_ANALYZE:-0}"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "UNIFIED VALIDATION PIPELINE"
echo "Tensor Compiler: Compile + (LLVM Verify) + (LLVM Analyze)"
echo "=========================================="
echo ""

# ----------------------------
# Sanity checks
# ----------------------------
if [ ! -e "$COMPILER" ]; then
  echo -e "${RED}✗ Error: compiler not found: $COMPILER${NC}"
  echo "Run 'make' first."
  exit 1
fi
if [ ! -x "$COMPILER" ]; then
  echo -e "${RED}✗ Error: compiler is not executable: $COMPILER${NC}"
  echo "Try: chmod +x $COMPILER"
  exit 1
fi
if file "$COMPILER" | grep -qiE "shell script|text"; then
  echo -e "${RED}✗ Error: $COMPILER does not look like a compiled binary (it's a script/text file).${NC}"
  file "$COMPILER" | sed 's/^/   /'
  exit 1
fi

# llvm-as location (optional)
LLVM_AS=""
if [ "$SKIP_LLVM_VERIFY" -eq 0 ]; then
  if [ -f "./llvm_config.sh" ]; then
    # shellcheck disable=SC1091
    source ./llvm_config.sh
    LLVM_AS="${LLVM_AS:-}"
  fi

  if [ -z "$LLVM_AS" ]; then
    echo -e "${YELLOW}⚠ llvm-as verification enabled but LLVM_AS is not set (llvm_config.sh missing or empty).${NC}"
    echo -e "${YELLOW}  -> llvm-as verification will be skipped.${NC}"
    SKIP_LLVM_VERIFY=1
  fi
fi

# llvm_analyze.sh (optional)
if [ "$SKIP_LLVM_ANALYZE" -eq 0 ]; then
  if [ ! -x "./llvm_analyze.sh" ]; then
    echo -e "${YELLOW}⚠ llvm_analyze.sh not found or not executable.${NC}"
    echo -e "${YELLOW}  -> LLVM analysis will be skipped.${NC}"
    SKIP_LLVM_ANALYZE=1
  fi
fi

# ----------------------------
# Tests
# ----------------------------
TESTS=(
  "tests/bench_test_1.txt"
  "tests/bench_test_2.txt"
  "tests/bench_test_3.txt"
)

# ----------------------------
# CSV header
# ----------------------------
cat > "$CSV_OUT" <<EOF
test,status,compile_ms,verify_ms,analyze_ms,total_ms,kernels,ir_bytes,o3_bytes,ir_reduction_pct
EOF

# ----------------------------
# Counters
# ----------------------------
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# ----------------------------
# Helpers
# ----------------------------
has_tensor_ops() {
  local f="$1"
  grep -q "@matmul\|@transpose\|@reduce\|@reshape" "$f"
}

count_kernels() {
  local f="$1"
  if [ -f "$f" ]; then
    grep -c "__global__" "$f" 2>/dev/null || echo "0"
  else
    echo "0"
  fi
}

safe_wc_bytes() {
  local f="$1"
  if [ -f "$f" ]; then
    wc -c < "$f" | tr -d ' \n\r'
  else
    echo "0"
  fi
}

compute_reduction_pct() {
  local orig="$1"
  local opt="$2"
  if [ "$orig" -le 0 ] || [ "$opt" -le 0 ]; then
    echo ""
    return
  fi
  awk -v o="$orig" -v p="$opt" 'BEGIN{printf "%.2f", (100.0 - (p*100.0/o))}'
}

# ----------------------------
# Main loop
# ----------------------------
echo "Starting unified validation..."
echo ""

for test_file in "${TESTS[@]}"; do
  if [ ! -f "$test_file" ]; then
    echo -e "${YELLOW}⚠ Test not found: $test_file${NC}"
    continue
  fi

  TOTAL_TESTS=$((TOTAL_TESTS + 1))
  test_name="$(basename "$test_file")"

  echo "----------------------------------------"
  echo "Test: $test_name"
  echo "----------------------------------------"

  # Clean previous artifacts (keep llvm_output/ history per-test by moving it)
  rm -f "$CUDA_OUT" "$LLVM_IR_OUT"
  rm -rf "$LLVM_DIR"
  mkdir -p "$LLVM_DIR"

  # Per-test logs
  COMPILE_OUT="$LOG_DIR/${test_name}.compile.stdout.log"
  COMPILE_ERR="$LOG_DIR/${test_name}.compile.stderr.log"
  LLVMAS_ERR="$LOG_DIR/${test_name}.llvm-as.stderr.log"
  ANALYZE_LOG="$LOG_DIR/${test_name}.llvm_analyze.log"

  status="PASS"

  # ----------------------------
  # Step A: Compile
  # ----------------------------
  t0="$(now_ns)"
  set +e
  timeout "${COMPILE_TIMEOUT_SECS}s" "$COMPILER" < "$test_file" > "$COMPILE_OUT" 2> "$COMPILE_ERR"
  rc=$?
  set -e
  t1="$(now_ns)"
  compile_ns=$((t1 - t0))
  compile_ms="$(ns_to_ms "$compile_ns")"

  if [ "$rc" -ne 0 ]; then
    status="FAIL"
    if [ "$rc" -eq 124 ]; then
      echo -e "${RED}✗ Compile timed out after ${COMPILE_TIMEOUT_SECS}s${NC}"
    else
      echo -e "${RED}✗ Compile failed (exit code: $rc)${NC}"
    fi
    echo "   --- stderr (last 40 lines) ---"
    tail -40 "$COMPILE_ERR" 2>/dev/null || true
    echo ""
  else
    echo -e "${GREEN}✓ Valid syntax (compiled)${NC}"
  fi

  # Artifact checks (only meaningful if compile succeeded)
  kernels="0"
  ir_bytes="0"
  o3_bytes="0"
  reduction_pct=""

  if [ "$status" = "PASS" ]; then
    if [ -f "$CUDA_OUT" ]; then
      kernels="$(count_kernels "$CUDA_OUT" | tr -d '\n\r')"
      if [ "${kernels:-0}" -gt 0 ] 2>/dev/null; then
        echo -e "${GREEN}✓ CUDA code generated ($kernels kernels)${NC}"
      else
        echo -e "${YELLOW}⚠ CUDA code generated but no kernels found${NC}"
      fi
    else
      # only warn if the test likely expects tensor ops
      if has_tensor_ops "$test_file"; then
        echo -e "${YELLOW}⚠ CUDA output missing: $CUDA_OUT${NC}"
      else
        echo -e "${YELLOW}⚠ No CUDA output (non-tensor or declarations-only test)${NC}"
      fi
    fi

    if [ -f "$LLVM_IR_OUT" ]; then
      if grep -q "define\|declare" "$LLVM_IR_OUT" 2>/dev/null; then
        echo -e "${GREEN}✓ LLVM IR generated${NC}"
      else
        echo -e "${YELLOW}⚠ LLVM IR generated but looks empty${NC}"
      fi
    else
      if has_tensor_ops "$test_file"; then
        echo -e "${YELLOW}⚠ LLVM IR output missing: $LLVM_IR_OUT${NC}"
      else
        echo -e "${YELLOW}⚠ No LLVM IR output (non-tensor or declarations-only test)${NC}"
      fi
    fi
  fi

  # ----------------------------
  # Step B: llvm-as verify (optional)
  # ----------------------------
  verify_ms=""
  if [ "$status" = "PASS" ] && [ "$SKIP_LLVM_VERIFY" -eq 0 ] && [ -f "$LLVM_IR_OUT" ]; then
    echo "   Verifying LLVM IR with llvm-as..."
    v0="$(now_ns)"
    set +e
    timeout "${VERIFY_TIMEOUT_SECS}s" "$LLVM_AS" < "$LLVM_IR_OUT" > /dev/null 2> "$LLVMAS_ERR"
    vrc=$?
    set -e
    v1="$(now_ns)"
    verify_ns=$((v1 - v0))
    verify_ms="$(ns_to_ms "$verify_ns")"

    if [ "$vrc" -eq 0 ]; then
      echo -e "   ${GREEN}✓ llvm-as verification passed${NC}"
    elif [ "$vrc" -eq 124 ]; then
      echo -e "   ${RED}✗ llvm-as timed out after ${VERIFY_TIMEOUT_SECS}s${NC}"
      status="FAIL"
      tail -40 "$LLVMAS_ERR" 2>/dev/null || true
    else
      echo -e "   ${RED}✗ llvm-as verification failed (exit code: $vrc)${NC}"
      status="FAIL"
      tail -40 "$LLVMAS_ERR" 2>/dev/null || true
    fi
  fi

  # ----------------------------
  # Step C: llvm_analyze pipeline (optional)
  # ----------------------------
  analyze_ms=""
  if [ "$status" = "PASS" ] && [ "$SKIP_LLVM_ANALYZE" -eq 0 ] && [ -f "$LLVM_IR_OUT" ]; then
    echo "   Running LLVM optimization pipeline (llvm_analyze.sh)..."
    a0="$(now_ns)"
    set +e
    ./llvm_analyze.sh "$LLVM_IR_OUT" "$LLVM_DIR" > "$ANALYZE_LOG" 2>&1
    arc=$?
    set -e
    a1="$(now_ns)"
    analyze_ns=$((a1 - a0))
    analyze_ms="$(ns_to_ms "$analyze_ns")"

    if [ "$arc" -ne 0 ]; then
      echo -e "   ${RED}✗ llvm_analyze.sh failed (exit code: $arc)${NC}"
      status="FAIL"
      tail -60 "$ANALYZE_LOG" 2>/dev/null || true
    else
      echo -e "   ${GREEN}✓ Analysis complete${NC}"
    fi
  fi

  # ----------------------------
  # Metrics for CSV
  # ----------------------------
  ir_bytes="$(safe_wc_bytes "$LLVM_IR_OUT")"

  # if analysis ran, try to read O3 file (your llvm_analyze.sh writes tensor_output_O3.ll in llvm_output/)
  if [ -f "$LLVM_DIR/tensor_output_O3.ll" ]; then
    o3_bytes="$(safe_wc_bytes "$LLVM_DIR/tensor_output_O3.ll")"
    reduction_pct="$(compute_reduction_pct "$ir_bytes" "$o3_bytes")"
  fi

  # Total time = compile + verify + analyze (only those present)
  # (we keep ms strings; compute numeric total with awk)
  total_ms="$(awk -v c="${compile_ms:-0}" -v v="${verify_ms:-0}" -v a="${analyze_ms:-0}" 'BEGIN{printf "%.3f", c+v+a}')"

  # ----------------------------
  # Print per-test summary
  # ----------------------------
  if [ "$status" = "PASS" ]; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
    echo -e "${GREEN}✓ Test PASSED${NC}"
  else
    FAILED_TESTS=$((FAILED_TESTS + 1))
    echo -e "${RED}✗ Test FAILED${NC}"
  fi

  echo "   Timings: compile=${compile_ms}s verify=${verify_ms:-NA}s analyze=${analyze_ms:-NA}s total=${total_ms}s"
  echo ""

  # ----------------------------
  # CSV row
  # ----------------------------
  # Note: keep empty cells for missing verify/analyze times
  echo "${test_name},${status},${compile_ms},${verify_ms},${analyze_ms},${total_ms},${kernels},${ir_bytes},${o3_bytes},${reduction_pct}" >> "$CSV_OUT"
done

# ----------------------------
# Summary
# ----------------------------
echo "=========================================="
echo "VALIDATION SUMMARY"
echo "=========================================="
echo "Total tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"

if [ "$TOTAL_TESTS" -gt 0 ]; then
  success_rate="$(awk -v p="$PASSED_TESTS" -v t="$TOTAL_TESTS" 'BEGIN{printf "%.1f", (p*100.0)/t}')"
  echo "Success rate: ${success_rate}%"
fi

echo "=========================================="
echo "CSV timings written to: $CSV_OUT"
echo "Logs directory:         $LOG_DIR"
echo "=========================================="

# Exit code
if [ "$FAILED_TESTS" -eq 0 ]; then
  exit 0
else
  exit 1
fi

