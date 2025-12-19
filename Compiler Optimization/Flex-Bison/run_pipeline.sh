#!/usr/bin/env bash
set -euo pipefail

ARCH="${ARCH:-sm_70}"
COMPILER_BIN="${COMPILER_BIN:-./compiler}"
GEN_CU="${GEN_CU:-generated_kernels.cu}"
PATCHED_CU="${PATCHED_CU:-generated_kernels.patched.cu}"
OUT_BIN="${OUT_BIN:-generated_kernels.out}"

INPUT_FILE="${1:-}"
if [[ -z "${INPUT_FILE}" ]]; then
  echo "Usage: $0 path/to/test_file.txt"
  exit 1
fi
if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "Error: input file not found: ${INPUT_FILE}"
  exit 1
fi

echo "==> Running make"
make

echo "==> Running ${COMPILER_BIN} < ${INPUT_FILE}"
COMP_OUT_FILE="$(mktemp)"
trap 'rm -f "${COMP_OUT_FILE}"' EXIT

set +e
"${COMPILER_BIN}" < "${INPUT_FILE}" > "${COMP_OUT_FILE}" 2>&1
COMP_RC=$?
set -e

cat "${COMP_OUT_FILE}"

if [[ ${COMP_RC} -ne 0 ]]; then
  if [[ ${COMP_RC} -eq 139 ]]; then
    echo "Error: compiler crashed with SIGSEGV (exit 139)."
  else
    echo "Error: compiler returned nonzero exit code: ${COMP_RC}"
  fi
  exit ${COMP_RC}
fi

echo "==> Parsing N,K,M from last MATMUL node in compiler output"

read -r N K M <<EOF
$(awk '
  function get_node_id(line,   t) {
    t=line; sub(/^Node[ \t]+/, "", t); sub(/:.*/, "", t); return t
  }
  function get_shape(line,   s, a) {
    if (match(line, /Shape:[ \t]*\[[0-9]+[ \t]*,[ \t]*[0-9]+\]/)) {
      s = substr(line, RSTART, RLENGTH)
      gsub(/[^0-9,]/, "", s); split(s, a, ",")
      return a[1] " " a[2]
    }
    return ""
  }
  function get_inputs(line,   s) {
    if (match(line, /Inputs:[ \t]*[0-9]+[ \t]+[0-9]+/)) {
      s = substr(line, RSTART, RLENGTH)
      sub(/^Inputs:[ \t]*/, "", s); gsub(/[ \t]+/, " ", s)
      return s
    }
    return ""
  }

  /^Node[ \t]+[0-9]+:/ {
    node = get_node_id($0)
    shp = get_shape($0); if (shp != "") shapes[node] = shp
    if ($0 ~ /MATMUL/) {
      ins = get_inputs($0)
      if (ins != "") { split(ins, tmp, " "); last_left=tmp[1]; last_right=tmp[2] }
    }
  }

  END {
    if (last_left == "" || last_right == "") { print ""; exit }
    split(shapes[last_left], L, " "); split(shapes[last_right], R, " ")
    N=L[1]; K=L[2]; M=R[2]
    if (N=="" || K=="" || M=="") { print ""; exit }
    print N, K, M
  }
' "${COMP_OUT_FILE}")
EOF

if [[ -z "${N:-}" || -z "${K:-}" || -z "${M:-}" ]]; then
  echo "Error: could not parse N/K/M from graph MATMUL node."
  exit 1
fi

echo "==> Using dims from last MATMUL: N=${N}, K=${K}, M=${M}"

echo "==> Patching ${GEN_CU} (constexpr dims + random init + redeclare fix + printing)"

if [[ ! -f "${GEN_CU}" ]]; then
  echo "Error: ${GEN_CU} not found. Did your compiler generate it?"
  exit 1
fi

# Detect which common host variable names exist in generated code
HAS_hA=0; HAS_Ah=0; HAS_hostA=0
HAS_hB=0; HAS_Bh=0; HAS_hostB=0
OUTVAR=""

grep -Eq 'float[[:space:]]*\*[[:space:]]*h_A\b'    "${GEN_CU}" && HAS_hA=1
grep -Eq 'float[[:space:]]*\*[[:space:]]*A_h\b'    "${GEN_CU}" && HAS_Ah=1
grep -Eq 'float[[:space:]]*\*[[:space:]]*hostA\b'  "${GEN_CU}" && HAS_hostA=1

grep -Eq 'float[[:space:]]*\*[[:space:]]*h_B\b'    "${GEN_CU}" && HAS_hB=1
grep -Eq 'float[[:space:]]*\*[[:space:]]*B_h\b'    "${GEN_CU}" && HAS_Bh=1
grep -Eq 'float[[:space:]]*\*[[:space:]]*hostB\b'  "${GEN_CU}" && HAS_hostB=1

# Pick an output host buffer name (first match)
if grep -Eq 'float[[:space:]]*\*[[:space:]]*h_C\b' "${GEN_CU}"; then
  OUTVAR="h_C"
elif grep -Eq 'float[[:space:]]*\*[[:space:]]*C_h\b' "${GEN_CU}"; then
  OUTVAR="C_h"
elif grep -Eq 'float[[:space:]]*\*[[:space:]]*hostC\b' "${GEN_CU}"; then
  OUTVAR="hostC"
else
  OUTVAR=""
fi

TMP_HEADER="$(mktemp)"
TMP_STAGE="$(mktemp)"
trap 'rm -f "${TMP_HEADER}" "${TMP_STAGE}"' EXIT

cat > "${TMP_HEADER}" <<EOF
// --- Auto-injected helpers (run_pipeline.sh) ---
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// constexpr dims (no macros)
static constexpr int N = ${N};
static constexpr int K = ${K};
static constexpr int M = ${M};

static inline void init_random_float(float* x, int n, uint32_t seed=123456789u) {
  uint32_t s = seed;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; // xorshift32
    x[i] = (s & 0x00FFFFFF) / float(0x01000000); // [0,1)
  }
}

static inline void print_summary(const float* c, int n) {
  double sum = 0.0, sumabs = 0.0;
  int take = n < 10 ? n : 10;
  for (int i = 0; i < n; ++i) { double v=c[i]; sum += v; sumabs += (v<0?-v:v); }
  std::printf("[result] C size=%d\\n", n);
  std::printf("[result] C first %d: ", take);
  for (int i = 0; i < take; ++i) std::printf("%g%s", c[i], (i+1==take?"\\n":" "));
  std::printf("[result] checksum sum=%0.6f sumabs=%0.6f\\n", sum, sumabs);
}
EOF

# 1) prepend header
cat "${TMP_HEADER}" "${GEN_CU}" > "${TMP_STAGE}"

# 2) fix duplicate blockSize/gridSize + inject random init + inject printing after last D2H memcpy
awk -v HAS_hA="${HAS_hA}" -v HAS_Ah="${HAS_Ah}" -v HAS_hostA="${HAS_hostA}" \
    -v HAS_hB="${HAS_hB}" -v HAS_Bh="${HAS_Bh}" -v HAS_hostB="${HAS_hostB}" \
    -v OUTVAR="${OUTVAR}" '
  BEGIN { seen_block=0; seen_grid=0; d2h_total=0; }

  {
    # count total DeviceToHost copies (for later injection)
    if ($0 ~ /cudaMemcpy/ && $0 ~ /cudaMemcpyDeviceToHost/) d2h_total++
    lines[NR]=$0
  }

  END {
    d2h_seen=0
    for (i=1; i<=NR; i++) {
      line = lines[i]

      # redeclare fix
      if (line ~ /(^|[[:space:]])dim3[[:space:]]+blockSize[[:space:]]*\(/) {
        if (seen_block++) sub(/dim3[[:space:]]+blockSize[[:space:]]*\(/, "blockSize = dim3(", line)
      }
      if (line ~ /(^|[[:space:]])dim3[[:space:]]+gridSize[[:space:]]*\(/) {
        if (seen_grid++) sub(/dim3[[:space:]]+gridSize[[:space:]]*\(/, "gridSize = dim3(", line)
      }

      print line

      # random init right after host allocations (only if those names exist)
      if (HAS_hA==1 && line ~ /float[[:space:]]*\*[[:space:]]*h_A[[:space:]]*=/) {
        print "  init_random_float(h_A, N*K);"
        print "  std::printf(\"[run_pipeline] init random A (%d)\\n\", N*K);"
      }
      if (HAS_Ah==1 && line ~ /float[[:space:]]*\*[[:space:]]*A_h[[:space:]]*=/) {
        print "  init_random_float(A_h, N*K);"
        print "  std::printf(\"[run_pipeline] init random A (%d)\\n\", N*K);"
      }
      if (HAS_hostA==1 && line ~ /float[[:space:]]*\*[[:space:]]*hostA[[:space:]]*=/) {
        print "  init_random_float(hostA, N*K);"
        print "  std::printf(\"[run_pipeline] init random A (%d)\\n\", N*K);"
      }

      if (HAS_hB==1 && line ~ /float[[:space:]]*\*[[:space:]]*h_B[[:space:]]*=/) {
        print "  init_random_float(h_B, K*M);"
        print "  std::printf(\"[run_pipeline] init random B (%d)\\n\", K*M);"
      }
      if (HAS_Bh==1 && line ~ /float[[:space:]]*\*[[:space:]]*B_h[[:space:]]*=/) {
        print "  init_random_float(B_h, K*M);"
        print "  std::printf(\"[run_pipeline] init random B (%d)\\n\", K*M);"
      }
      if (HAS_hostB==1 && line ~ /float[[:space:]]*\*[[:space:]]*hostB[[:space:]]*=/) {
        print "  init_random_float(hostB, K*M);"
        print "  std::printf(\"[run_pipeline] init random B (%d)\\n\", K*M);"
      }

      # print after the last D2H memcpy
      if (line ~ /cudaMemcpy/ && line ~ /cudaMemcpyDeviceToHost/) {
        d2h_seen++
        if (d2h_seen == d2h_total && OUTVAR != "") {
          print "  // --- auto-print result (run_pipeline.sh) ---"
          print "  print_summary(" OUTVAR ", N*M);"
        }
      }
    }
  }
' "${TMP_STAGE}" > "${PATCHED_CU}"

echo "==> Compiling ${PATCHED_CU} with nvcc (arch=${ARCH})"
nvcc -arch="${ARCH}" -O3 "${PATCHED_CU}" -o "${OUT_BIN}"

echo "==> Build OK: ./${OUT_BIN}"
echo "==> Running ./${OUT_BIN}"
./"${OUT_BIN}"
echo "==> Run completed successfully."
