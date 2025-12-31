# LLVM Integration Guide
**Tensor Compiler → LLVM IR → Analysis & Optimization**

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Usage Workflow](#usage-workflow)
5. [LLVM Analysis Pipeline](#llvm-analysis-pipeline)
6. [Test Cases](#test-cases)
7. [Interpreting Results](#interpreting-results)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This integration enables the tensor compiler to emit **LLVM IR** (`.ll` files) instead of directly generating CUDA code. LLVM's powerful analysis and optimization infrastructure is then used to:

1. **Validate** IR correctness
2. **Optimize** tensor operations (DCE, CSE, algebraic simplification, memory optimization)
3. **Analyze** performance characteristics (instruction count, memory operations, control flow)
4. **Generate metrics** for test cases and benchmarks

### Key Benefits

- ✅ **Industry-standard optimizations** (LLVM's proven optimization passes)
- ✅ **Reproducible analysis** (deterministic IR-level metrics)
- ✅ **No GPU required** (all analysis at IR level)
- ✅ **Educational clarity** (human-readable IR for debugging)
- ✅ **Extensible** (easy to add new LLVM passes)

---

## Architecture

### Compilation Pipeline

```
┌──────────────┐
│ Tensor Source│
│   (.txt)     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Flex + Bison │  ← Frontend
│   (Parser)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  TensorGraph │  ← Internal IR
│   (DAG)      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│LLVM Lowering │  ← NEW MODULE
│ llvm_lowering│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LLVM IR     │  ← Textual IR (.ll)
│  (.ll file)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ LLVM Analysis│  ← Analysis Pipeline
│  Pipeline    │     (opt, llvm-dis)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Metrics    │  ← Test Data
│  & Reports   │
└──────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **Lowering Module** | `llvm_lowering.c/h` | Translates TensorGraph → LLVM IR |
| **Analysis Script** | `llvm_analyze.sh` | Runs LLVM passes, extracts metrics |
| **Configuration** | `llvm_config.sh` | LLVM tool paths |
| **Integration** | Modified `Makefile` | Build & test targets |

---

## Installation & Setup

### Prerequisites

- **LLVM installed** at `~/Documents/GitHub/llvm-project/build`
- **Flex & Bison** (already present)
- **GCC** (for compiling the frontend)

### Setup Steps

1. **Verify LLVM Installation**

```bash
cd "Compiler Optimization/Flex-Bison"
source llvm_config.sh
```

Expected output:
```
✓ LLVM tools configured:
  LLVM_CLANG: /Users/.../llvm-project/build/bin/clang
  LLVM_OPT:   /Users/.../llvm-project/build/bin/opt
  ...
```

2. **Build Compiler with LLVM Support**

```bash
make clean
make
```

This compiles:
- Frontend (Flex/Bison)
- TensorGraph module
- **NEW:** LLVM lowering module

3. **Verify Build**

```bash
ls -la compiler llvm_lowering.o
```

---

## Usage Workflow

### Basic Usage: Generate LLVM IR

```bash
# Run compiler with tensor source
./compiler < tests/simple_test.txt

# Output: tensor_output.ll (LLVM IR file)
```

### Full Workflow: Generate + Analyze

```bash
# Option 1: Use Makefile target
make llvm-test

# Option 2: Manual workflow
./compiler < tests/test_dce.txt        # Generate IR
./llvm_analyze.sh tensor_output.ll     # Analyze IR
```

### Example: Test Case 1 (DCE)

```bash
# 1. Generate LLVM IR from DCE test
./compiler < tests/test_dce.txt

# 2. Run analysis pipeline
./llvm_analyze.sh tensor_output.ll llvm_output/

# 3. View results
cat llvm_output/tensor_output_summary.txt
```

Expected output:
```
========================================
LLVM Analysis Summary
========================================

OPTIMIZATION METRICS
────────────────────────────────────────
Stage                Instructions    Reduction    Time (s)
────────────────────────────────────────
Baseline (O0)        45          -            -
After DCE            36          9            0.023
After CSE            34          2            0.018
After InstCombine    30          4            0.021
After -O3            28          17 (37.8%)   0.156
────────────────────────────────────────

PHASE TIMING
────────────────────────────────────────
Phase                                    Time (s)
────────────────────────────────────────
[1/9] Verify Input IR                   0.012
[2/9] Extract Baseline Metrics          0.008
[3/9] Dead Code Elimination             0.023
[4/9] Common Subexpression Elimination  0.018
[5/9] Instruction Combining             0.021
[6/9] Memory-to-Register Promotion      0.015
[7/9] Full -O3 Optimization             0.156
[8/9] Control Flow Graph Generation     0.019
[9/9] Summary Generation                (current)
────────────────────────────────────────

Total pipeline time: 0.289s
```

---

## LLVM Analysis Pipeline

The `llvm_analyze.sh` script runs a sequence of LLVM optimization passes:

### Pipeline Stages

| Stage | LLVM Pass | Purpose | Metrics Extracted |
|-------|-----------|---------|-------------------|
| **1. Verify** | `llvm-as` | Check IR validity | ✓/✗ + Time (s) |
| **2. Baseline** | - | Count instructions | Instruction count, memory ops + Time (s) |
| **3. DCE** | `-passes=dce` | Remove dead code | Instructions removed + Time (s) |
| **4. CSE** | `-passes=early-cse` | Eliminate redundancy | Duplicates merged + Time (s) |
| **5. InstCombine** | `-passes=instcombine` | Algebraic simplification | Identity ops removed + Time (s) |
| **6. Mem2Reg** | `-passes=mem2reg` | Promote to registers | Load/store reduction + Time (s) |
| **7. O3** | `-O3` | Full optimization | Overall reduction % + Time (s) |
| **8. CFG** | `-passes=dot-cfg` | Control flow graph | .dot files + Time (s) |
| **9. Summary** | - | Generate report | Summary.txt + Time (s) |

### Output Files

After running `llvm_analyze.sh tensor_output.ll llvm_output/`:

```
llvm_output/
├── tensor_output_O0_metrics.txt      # Baseline metrics
├── tensor_output_dce.ll              # After DCE
├── tensor_output_cse.ll              # After CSE
├── tensor_output_instcombine.ll      # After instcombine
├── tensor_output_mem2reg.ll          # After mem2reg
├── tensor_output_O3.ll               # Full optimization
├── tensor_output_summary.txt         # Summary report
├── tensor_output_analysis.log        # Full log
└── *.dot                             # CFG files (if generated)
```

---

## Test Cases

### Available Tests (from TESTS_AND_BENCHMARKS.md)

| Test | File | Focus | Expected Outcome |
|------|------|-------|------------------|
| **TC-1** | `tests/test_dce.txt` | Dead Code Elimination | 25% node reduction |
| **TC-2** | `tests/test_cse.txt` | Common Subexpr Elimination | 50% computation reduction |
| **TC-3** | `tests/test_algebraic.txt` | Algebraic Simplification | Identity ops removed |
| **TC-4** | `tests/test_fusion.txt` | Operation Fusion | Kernel count reduced 67% |
| **TC-5** | `tests/test_memory.txt` | Memory Layout | 33% memory op reduction |

### Running Test Cases

```bash
# Test 1: Dead Code Elimination
./compiler < tests/test_dce.txt
./llvm_analyze.sh tensor_output.ll llvm_output_dce/
cat llvm_output_dce/tensor_output_summary.txt

# Test 2: CSE
./compiler < tests/test_cse.txt
./llvm_analyze.sh tensor_output.ll llvm_output_cse/
cat llvm_output_cse/tensor_output_summary.txt

# ... (repeat for TC-3, TC-4, TC-5)
```

### Creating New Tests

1. **Write tensor source** (`.txt` file):
```
program test_example
tensor A[10,10];
tensor B[10,10];
tensor C[10,10];
begin
  C := A @matmul B;
end
```

2. **Generate IR**:
```bash
./compiler < tests/test_example.txt
```

3. **Analyze**:
```bash
./llvm_analyze.sh tensor_output.ll llvm_output_example/
```

4. **Interpret metrics** (see next section)

---

## Interpreting Results

### Key Metrics

#### 1. **Instruction Count Reduction**

```
Baseline (O0)        45
After -O3            28
Reduction:           17 (37.8%)
```

**Interpretation**:
- **Good**: 25-40% reduction (comparable to LLVM benchmarks)
- **Excellent**: >40% reduction
- **Concerning**: <10% reduction (may indicate missed optimizations)

#### 2. **Memory Operations**

```
                     Baseline    After mem2reg
Load operations      12          8
Store operations     8           5
Total                20          13
Reduction:           7 operations (35%)
```

**Interpretation**:
- Fewer load/store = better register usage
- Target: 30-50% reduction
- Memory-bound workloads benefit most

#### 3. **Dead Code Elimination**

```
After DCE: 36 (removed: 9)
```

**Interpretation**:
- Removed instructions = unused computations
- Higher removal = more wasted work in original code

#### 4. **CSE Effectiveness**

```
After CSE: 34 (removed: 2)
```

**Interpretation**:
- Merged duplicates = redundant computations
- Low removal = good (no unnecessary duplication)
- High removal = opportunity for frontend improvement

### Comparing with LLVM Benchmarks

From academic research (see TESTS_AND_BENCHMARKS.md):

| Optimization | Expected Reduction | Your Result | Status |
|--------------|-------------------|-------------|---------|
| DCE | 10-20% | ? | ✓/✗ |
| CSE | 15-30% | ? | ✓/✗ |
| Algebraic | 5-15% | ? | ✓/✗ |
| Overall (O3) | 27-33% | ? | ✓/✗ |

**Validation Criteria**:
- Within **±5%** of LLVM = ✓ Excellent
- Within **±10%** = ✓ Good
- >15% difference = Investigate

---

## Troubleshooting

### Common Issues

#### 1. **LLVM Tools Not Found**

```
Error: LLVM tool not found: .../clang
```

**Solution**:
```bash
# Edit llvm_config.sh
vim llvm_config.sh

# Update LLVM_BUILD_DIR to match your installation
LLVM_BUILD_DIR="$HOME/Documents/GitHub/llvm-project/build"
```

#### 2. **Invalid LLVM IR Generated**

```
✗ ERROR: Invalid LLVM IR
```

**Solution**:
```bash
# Manually verify IR
llvm-as < tensor_output.ll 2>&1 | head -20

# Check for syntax errors in llvm_lowering.c
# Common issues:
# - Missing closing braces in functions
# - Incorrect register numbering
# - Invalid type strings
```

#### 3. **No Metrics Extracted**

```
Instructions: 0
```

**Solution**:
- Ensure compiler actually generated tensor_output.ll
- Check file is not empty: `wc -l tensor_output.ll`
- Verify IR structure: `grep "define i32 @main" tensor_output.ll`

#### 4. **Analysis Script Fails**

```
./llvm_analyze.sh: command not found
```

**Solution**:
```bash
chmod +x llvm_analyze.sh llvm_config.sh
```

### Debugging Tips

1. **Inspect Generated IR**:
```bash
less tensor_output.ll
# Look for:
# - "define i32 @main()" function
# - "call" instructions for tensor operations
# - Proper SSA register naming (%0, %1, ...)
```

2. **Test Individual LLVM Passes**:
```bash
source llvm_config.sh
$LLVM_OPT -passes=dce -S tensor_output.ll -o test_dce.ll
$LLVM_OPT -passes=early-cse -S test_dce.ll -o test_cse.ll
```

3. **Compare with Reference**:
```bash
# Generate reference LLVM IR
$LLVM_CLANG -O0 -S -emit-llvm llvm_reference/test_dce.c -o reference.ll

# Compare structures
diff tensor_output.ll reference.ll
```

---

## Advanced Usage

### Custom Analysis Pass

Add custom LLVM pass to pipeline:

```bash
# Edit llvm_analyze.sh, add before Stage 9:
echo "[Custom] Running scalar evolution analysis..."
"$LLVM_OPT" -passes=analyze -scalar-evolution "$O3_LL" 2>&1 | tee -a "$LOG_FILE"
```

### Batch Testing

```bash
#!/bin/bash
for test in tests/test_*.txt; do
    echo "Testing $test..."
    ./compiler < "$test"
    ./llvm_analyze.sh tensor_output.ll "llvm_output_$(basename $test .txt)/"
done
```

### Visualization

Generate and view CFG:

```bash
# Generate CFG dot files
$LLVM_OPT -passes=dot-cfg -disable-output tensor_output.ll

# Convert to PDF (requires graphviz)
dot -Tpdf .main.dot -o cfg_main.pdf
open cfg_main.pdf
```

---

## Integration with Existing Workflow

### Compatibility with `run_pipeline.sh`

The original pipeline:
```bash
make → ./compiler → awk → nvcc → execute
```

New LLVM-based pipeline:
```bash
make → ./compiler → LLVM IR → opt → analysis
```

**Both can coexist**:
- CUDA generation: Use `generate_cuda_code()` (existing)
- LLVM IR generation: Use `lower_graph_to_llvm_ir()` (new)

### Switching Modes

Modify compiler main (see next section) to choose:
```c
// Mode 1: CUDA generation (original)
generate_cuda_code(tensorGraph, "generated_kernels.cu");

// Mode 2: LLVM IR generation (new)
lower_graph_to_llvm_ir(tensorGraph, "tensor_output.ll");
```

---

## Next Steps

1. **Implement test cases** from TESTS_AND_BENCHMARKS.md
2. **Run analysis pipeline** on all 5 test cases
3. **Compare metrics** with expected LLVM reductions
4. **Document findings** in test results
5. **Iterate on lowering** if metrics don't match expectations

**Success Criteria**:
- ✓ All test IRs pass `llvm-as` verification
- ✓ DCE removes 10-25% instructions
- ✓ CSE removes 15-30% redundancy
- ✓ Overall O3 achieves 27-33% reduction

---

## References

- **LLVM Documentation**: https://llvm.org/docs/
- **LLVM IR Language Reference**: https://llvm.org/docs/LangRef.html
- **Optimization Pass Reference**: https://llvm.org/docs/Passes.html
- **New Pass Manager**: https://llvm.org/docs/NewPassManager.html
- **Project Design Doc**: `LLVM_DESIGN.md`
- **Test Specifications**: `TESTS_AND_BENCHMARKS.md`

---

**End of LLVM Integration Guide**
