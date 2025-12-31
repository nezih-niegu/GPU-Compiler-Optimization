# LLVM Documentation
**GPU Compiler Optimization: Complete LLVM Integration Guide**

## Executive Summary

This document provides comprehensive documentation for the LLVM integration in the tensor compiler, including architecture design, usage instructions, and test cases. The compiler generates both CUDA code and LLVM IR for analysis, enabling industry-standard optimization validation without requiring GPU execution.

**Core IR**: `TensorGraph` (DAG-based)  
**Targets**: CUDA Code + LLVM IR (parallel generation)  
**Approach**: Graph transformations with polyhedral analysis + LLVM-based optimization  
**Key Features**: 9-stage analysis pipeline with timing measurements, comprehensive metrics extraction

---

## Table of Contents

### Part I: Design & Architecture
1. [IR and Component Mapping](#1-ir-and-component-mapping)
2. [Compilation Pipeline](#2-compilation-pipeline)
3. [Optimization Passes](#3-optimization-passes)
4. [Pass Ordering](#4-pass-ordering)
5. [Architecture Diagrams](#5-architecture-diagrams)

### Part II: Integration & Usage
6. [Installation & Setup](#6-installation--setup)
7. [Usage Workflow](#7-usage-workflow)
8. [LLVM Analysis Pipeline](#8-llvm-analysis-pipeline)
9. [Interpreting Results](#9-interpreting-results)
10. [Troubleshooting](#10-troubleshooting)

### Part III: Testing & Benchmarks
11. [Test Cases](#11-test-cases)
12. [Benchmarks](#12-benchmarks)
13. [LLVM Commands Reference](#13-llvm-commands-reference)

---

# PART I: DESIGN & ARCHITECTURE

## 1. IR and Component Mapping

### 1.1 Primary IR: TensorGraph

**TensorGraph** is the canonical IR with DAG structure enabling:
- Language-independent representation
- Data-flow analysis and transformations
- SSA-like properties (unique node values)
- Shape-aware type system (`TensorDim`)

**Supporting IRs:**
- `IterationSpace`: Polyhedral model for loop bounds
- `ARST`: High-level AST (frontend)

### 1.2 Component Mapping

| **Project Component** | **LLVM Equivalent** | **Functionality** | **Files** |
|-----------------------|---------------------|-------------------|-----------|
| **Lexer (Flex)** | `clang::Lexer` | Tokenization of source code | `finalAssignment.l` |
| **Parser (Bison)** | `clang::Parser` | Syntactic analysis, AST construction | `finalAssignment.y` |
| **AST** | `clang::AST` | High-level language representation | `struct arst` |
| **Symbol Table** | `clang::Sema` / `SymbolTable` | Semantic analysis, name resolution | `HashTable` in parser |
| **TensorGraph (DAG)** | **LLVM IR** | **Primary intermediate representation** | `tensor_graph.h/c` |
| **GraphNode** | `llvm::Instruction` | Single operation in IR | `struct GraphNode` |
| **OpType Enum** | `llvm::Opcode` | Operation type enumeration | `enum OpType` |
| **TensorDim** | `llvm::Type` | Type system (shapes/dimensions) | `struct TensorDim` |
| **Optimizer Module** | `llvm::PassManager` | Orchestrates optimization passes | `optimizer.h/c` |
| **OptStrategy Enum** | `llvm::Pass` subclasses | Individual optimization passes | `enum OptStrategy` |
| **Operation Fusion** | `InstCombine` / Loop Fusion Pass | Combine operations to reduce overhead | `fuse_operations()` |
| **Common Subexpression Elimination** | `GVN` (Global Value Numbering) | Eliminate redundant computations | `eliminate_common_subexpressions()` |
| **Memory Layout Optimization** | `MemorySSA` / `LoopOptimization` | Optimize memory access patterns | `optimize_memory_layout()` |
| **Geometry Module** | Polyhedral Analysis (`Polly`) | Iteration space analysis, loop bounds | `geometry.h/c` |
| **IterationSpace** | Polyhedral Sets (ISL) | Mathematical loop representation | `struct IterationSpace` |
| **LLVM Lowering Module** | `clang::CodeGen` (IRGen) | TensorGraph → LLVM IR translation | `llvm_lowering.h/c` |
| **LLVM Analysis Pipeline** | `opt` pass pipeline | Run optimization passes, extract metrics + timing | `llvm_analyze.sh` |
| **LLVM Configuration** | Build system | LLVM tool paths and setup | `llvm_config.sh` |
| **CUDA Code Generator** | `llvm::CodeGen` / Backend | Target code emission | `cuda_gen.h/c` |
| **Kernel Generation** | Machine Code Emission | Generate target-specific kernels | `generate_*_kernel()` |
| **Compiler Driver** | `clang` driver | Orchestrates entire pipeline | `main()` in parser / `run_pipeline.sh` |
| **OptResult** | `llvm::AnalysisManager` | Collect and report optimization metrics | `struct OptResult` |

### 2.2 Detailed Mapping Justifications

#### TensorGraph ↔ LLVM IR
- **Structural Similarity**: Both are graph-based with explicit data dependencies
- **SSA Property**: Each GraphNode represents a unique value (read-once semantics)
- **Multi-Level**: Can represent operations at varying abstraction levels
- **Transformation Target**: Primary structure for optimization passes

#### Optimizer ↔ PassManager
- **Pass Orchestration**: Both sequence and apply multiple optimization strategies
- **Metrics Collection**: Track optimization effectiveness
- **Graph Preservation**: Maintain IR validity throughout transformations

#### Geometry Module ↔ Polly (Polyhedral Framework)
- **Mathematical Foundations**: Both use polyhedral models for loop analysis
- **Dependence Analysis**: Geometric operations enable dependency detection
- **Loop Transformation**: Enables advanced loop optimizations (tiling, fusion, interchange)

---

## 3. Compilation Pipeline Design

### 3.1 LLVM-Style Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                        STAGE 1: FRONTEND                        │
├─────────────────────────────────────────────────────────────────┤
│  Source Code (.txt) → Lexer → Parser → AST + Symbol Table       │
│  Analogous to: clang → Clang AST                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 2: IR GENERATION (Lowering)              │
├─────────────────────────────────────────────────────────────────┤
│  AST → TensorGraph Construction (IR Generation)                  │
│  + Geometry Module: Tensor → IterationSpace                      │
│  Analogous to: Clang AST → LLVM IR (IRGen)                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 3: IR ANALYSIS & VERIFICATION             │
| **OptResult** | `llvm::AnalysisManager` | Collect and report optimization metrics | `struct OptResult` |

## 2. Compilation Pipeline

### 2.1 Six-Stage Pipeline
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 5: BACKEND CODE GENERATION                    │
├─────────────────────────────────────────────────────────────────┤
│  TensorGraph → Dual Backend:                                     │
│                                                                  │
│  Path A: CUDA Backend                                            │
│  - Instruction Selection (GraphNode → CUDA operations)           │
│  - Register Allocation (thread/block mapping)                    │
│  - Scheduling (kernel launch configuration)                      │
│  - Code Emission (generate .cu file)                             │
│                                                                  │
│  Path B: LLVM Backend (NEW)                                      │
│  - LLVM IR Lowering (GraphNode → LLVM instructions)              │
│  - External function declarations (@tensor_alloc, @tensor_matmul)│
│  - SSA-form register naming                                      │
│  - Emit .ll file (textual LLVM IR)                               │
│                                                                  │
│  Analogous to: LLVM CodeGen (SelectionDAG → MachineInstr → Asm) │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 6: TARGET CODE OUTPUT                    │
├─────────────────────────────────────────────────────────────────┤
│  Output A: generated_kernels.cu (CUDA source code)               │
│  Output B: tensor_output.ll (LLVM IR)                            │
│                                                                  │
│  Post-Processing (LLVM path):                                    │
│  - llvm_analyze.sh: Run optimization passes (DCE, CSE, etc.)     │
│  - Extract metrics: instruction count, memory ops, function calls│
│  - Measure timing: per-phase execution time + total pipeline time│
│  - Generate reports: summary.txt, analysis.log                   │
│                                                                  │
│  Analogous to: LLVM assembly output (.s) or object files (.o)   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Stage-by-Stage Breakdown

#### **Stage 1: Frontend (Lexical & Syntactic Analysis)**

**Purpose**: Convert textual source code into structured representation

**Components**:
- **Lexer** (`finalAssignment.l`): Pattern matching, tokenization
- **Parser** (`finalAssignment.y`): Grammar enforcement, AST construction
- **Symbol Table**: Variable tracking, scope management

**Input**: Tensor operation source code
```
program tensor_test
tensor A[100,50];
tensor B[50,200];
begin
C := A @matmul B;
end
```

**Output**: AST + Symbol Table

**LLVM Equivalent**: `clang -fsyntax-only` (frontend without codegen)

---

#### **Stage 2: IR Generation (AST → TensorGraph Lowering)**

**Purpose**: Convert high-level AST to optimizable IR

**Process**:
1. Traverse AST nodes representing tensor operations
2. Create `GraphNode` for each operation
3. Establish data dependencies (edges between nodes)
4. Compute shape information (`TensorDim`)
5. Generate iteration spaces for each tensor (geometry module)

**Example Transformation**:
```
AST: C := A @matmul B

TensorGraph:
  Node 0: IDENTITY [A] Shape: [100, 50]
  Node 1: IDENTITY [B] Shape: [50, 200]
  Node 2: MATMUL [C] Shape: [100, 200] ← Inputs: 0, 1
```

**Code Location**: `finalAssignment.y` (semantic actions), `tensor_graph.c` (graph construction)

**LLVM Equivalent**: Clang's IR generation (`CodeGenFunction`)

---

#### **Stage 3: IR Analysis & Verification**

**Purpose**: Analyze IR for correctness and collect baseline metrics

**Analysis Passes**:

1. **Dependency Analysis**
   - Verify DAG property (no cycles)
   - Build def-use chains
   - Identify program outputs

2. **Type/Shape Checking**
   - Validate dimension compatibility (e.g., matmul requires compatible shapes)
   - Ensure shape propagation correctness

3. **Iteration Space Analysis**
   - Convert tensors to geometric representations
   - Calculate iteration bounds
   - Detect potential out-of-bounds accesses

4. **Cost Modeling**
   - Calculate memory access count (`calculate_memory_accesses()`)
   - Calculate computational cost (`calculate_computational_cost()`)
   - Establish baseline for optimization metrics

**LLVM Equivalent**: Analysis passes like `DominatorTree`, `LoopInfo`, `ScalarEvolution`

---

#### **Stage 4: Middle-End Optimization (The Core Pass Pipeline)**

**Purpose**: Transform IR to reduce cost while preserving semantics

*Detailed in Section 4 and 5*

---

#### **Stage 5: Backend Code Generation**

**Purpose**: Translate optimized IR to target-specific code (dual backends)

**Path A: CUDA Backend (Original)**

**Process**:
1. **Instruction Selection**: Map GraphNode operations to CUDA patterns
   - `OP_MATMUL` → tiled matrix multiplication kernel
   - `OP_ADD/MUL` → element-wise parallel kernel
   - `OP_REDUCE` → reduction with shared memory

2. **Resource Allocation**:
   - Determine grid/block dimensions
   - Allocate shared memory
   - Map threads to data elements

3. **Code Emission**:
   - Generate kernel function signatures
   - Emit CUDA kernel bodies
   - Generate host-side memory management
   - Create kernel launch calls

**Example Output**:
```cuda
__global__ void matmul_kernel_2(float *A, float *B, float *C, 
                                 int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**Code Location**: `cuda_gen.c`

---

**Path B: LLVM IR Backend (NEW)**

**Purpose**: Generate LLVM IR for analysis and optimization validation

**Process**:
1. **IR Lowering**: Translate TensorGraph to LLVM IR textual format
   - Map GraphNode operations to external function calls
   - Generate SSA-form register names (%0, %1, %2, ...)
   - Emit module header with target triple and data layout

2. **Function Declarations**: Declare external tensor operations
   ```llvm
   declare float* @tensor_alloc(i32) nounwind
   declare void @tensor_matmul(float*, float*, float*, i32, i32, i32) nounwind
   declare void @tensor_add(float*, float*, float*, i32) nounwind
   ```

3. **Main Function Generation**:
   - Allocate tensors (exclude reduce ops - they return scalars)
   - Emit operation calls
   - Free allocated tensors
   - Return 0

4. **Metadata Emission**: Add operation annotations
   - Memory access patterns (sequential, strided)
   - Compute/memory operation counts
   - Arithmetic intensity hints

**Example Output**:
```llvm
define i32 @main() {
entry:
  ; Allocations
  %0 = call float* @tensor_alloc(i32 5000)  ; A [100x50]
  %1 = call float* @tensor_alloc(i32 10000) ; B [50x200]
  
  ; Operations
  %2 = call float* @tensor_alloc(i32 20000) ; Result [100x200]
  call void @tensor_matmul(float* %2, float* %0, float* %1, 
                           i32 100, i32 50, i32 200)
  
  ; Cleanup
  call void @tensor_free(float* %0)
  call void @tensor_free(float* %1)
  call void @tensor_free(float* %2)
  ret i32 0
}
```

**Design Rationale**:
- **External Function Calls**: Simplifies IR generation, maintains abstraction
- **Trade-off**: Limited LLVM optimization opportunities (0% reduction expected for call-only patterns)
- **Benefit**: Clear, analyzable IR structure for instruction counting and validation

**Code Location**: `llvm_lowering.c/h`

**LLVM Equivalent**: LLVM Backend (SelectionDAG, instruction scheduling, register allocation)

---

#### **Stage 6: Target Code Output**

**Purpose**: Write final code to files and optionally analyze

**Output A: CUDA Path**
- **File**: `generated_kernels.cu`
- **Post-Compilation**: Execute via `nvcc` (NVIDIA CUDA Compiler)
- **Result**: GPU-executable binary

**Output B: LLVM IR Path (NEW)**
- **File**: `tensor_output.ll` (textual LLVM IR)
- **Verification**: `llvm-as tensor_output.ll` (check IR validity)
- **Analysis Pipeline**: `llvm_analyze.sh` script runs:
  1. **Baseline metrics**: Count instructions, memory ops, function calls
  2. **DCE**: Dead code elimination (`opt -passes=dce`)
  3. **CSE**: Common subexpression elimination (`opt -passes=early-cse`)
  4. **InstCombine**: Algebraic simplification (`opt -passes=instcombine`)
  5. **Mem2Reg**: Memory-to-register promotion (`opt -passes=mem2reg`)
  6. **O3**: Full optimization pipeline (`opt -O3`)
  7. **CFG**: Control flow graph generation (`opt -passes=dot-cfg`)
  8. **Summary**: Generate analysis report

**Analysis Output**:
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
└── .main.dot                         # CFG visualization
```

**Metrics Extracted**:
- Instruction count (before/after optimization)
- Memory operations (loads/stores)
- Function calls (tensor operations)
- Optimization reduction percentage

**LLVM Equivalent**: Assembly emission (`.s` files) or object code (`.o`)

---

## 4. Optimization Pass Design

### 4.1 Selected LLVM-Style Optimization Passes

Based on the tensor computation domain and existing infrastructure, the following passes are selected:

| # | **Pass Name** | **LLVM Analogue** | **Purpose** | **Complexity** |
|---|---------------|-------------------|-------------|----------------|
| 1 | Dead Code Elimination (DCE) | `DCEPass` | Remove unused computations | O(n) |
| 2 | Common Subexpression Elimination (CSE) | `GVN`, `EarlyCSE` | Eliminate duplicate computations | O(n²) → O(n) |
| 3 | Algebraic Simplification | `InstCombine` | Simplify algebraic expressions | O(n) |
| 4 | Operation Fusion | Loop Fusion, Kernel Fusion | Merge kernels to reduce overhead | O(n) |
| 5 | Memory Layout Optimization | `MemorySSA`, Cache Optimization | Improve spatial/temporal locality | O(n log n) |
| 6 | Loop-Invariant Code Motion (LICM) | `LICM` | Hoist loop-invariant computations | O(n × depth) |

### 3.2 Pass Summaries

**Pass 1: DCE** - Remove unused operations via mark-sweep  
**Pass 2: CSE** - Hash-based duplicate detection (O(n))  
**Pass 3: Algebraic** - Pattern matching (`A+0→A`, `A*1→A`, `transpose²→identity`)  
**Pass 4: Fusion** - Merge producer-consumer chains (reduces kernel launches)  
**Pass 5: Memory Layout** - Locality-aware topological sort, coalescing detection  
**Pass 6: LICM** - Hoist invariants to host computation

## 4. Pass Ordering

```
┌──────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION PASS PIPELINE                 │
│                        (Middle-End)                           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  [Entry: Unoptimized TensorGraph from IR Generation]         │
│                            │                                  │
│                            ▼                                  │
│  ┌──────────────────────────────────────────────────┐        │
│  │ PASS 1: Dead Code Elimination (DCE)              │        │
│  │ Purpose: Remove unused nodes                     │        │
│  │ Rationale: Clean up before expensive analysis    │        │
│  │ Dependencies: None (can always run first)        │        │
│  └────────────────────┬─────────────────────────────┘        │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────┐        │
│  │ PASS 2: Common Subexpression Elimination (CSE)   │        │
│  │ Purpose: Eliminate redundant computations        │        │
│  │ Rationale: Reduce node count early               │        │
│  │ Dependencies: Requires DCE (avoid dead CSE)      │        │
│  └────────────────────┬─────────────────────────────┘        │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────┐        │
│  │ PASS 3: Algebraic Simplification                 │        │
│  │ Purpose: Simplify expressions, remove identities │        │
│  │ Rationale: Expose fusion opportunities           │        │
│  │ Dependencies: CSE (avoid simplifying duplicates) │        │
│  └────────────────────┬─────────────────────────────┘        │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────┐        │
│  │ PASS 4: Operation Fusion                         │        │
│  │ Purpose: Merge operations into fused kernels     │        │
│  │ Rationale: Maximize kernel efficiency            │        │
│  │ Dependencies: Algebraic (needs simplified graph) │        │
│  └────────────────────┬─────────────────────────────┘        │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────┐        │
│  │ PASS 5: Memory Layout Optimization               │        │
│  │ Purpose: Optimize memory access patterns         │        │
│  │ Rationale: Improve cache/coalescing              │        │
│  │ Dependencies: Fusion (see final access patterns) │        │
│  └────────────────────┬─────────────────────────────┘        │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────┐        │
│  │ PASS 6: Loop-Invariant Code Motion (LICM)        │        │
│  │ Purpose: Hoist invariant computations            │        │
│  │ Rationale: Final cleanup, enable constant prop   │        │
│  │ Dependencies: All previous (needs final form)    │        │
│  └────────────────────┬─────────────────────────────┘        │
│                       │                                       │
│                       ▼                                       │
│  [Exit: Optimized TensorGraph → Backend Code Generation]     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Pass Ordering Rationale

#### **Why This Specific Order?**

The ordering follows fundamental compiler optimization principles:

**Principle 1: Clean Before Analyze**
- DCE runs first to avoid wasting effort on dead code
- Smaller graphs → faster subsequent passes

**Principle 2: Redundancy Before Transformation**
- CSE before fusion prevents fusing duplicate code
- CSE before algebraic simplification avoids redundant pattern matching

**Principle 3: Simplify Before Combine**
- Algebraic simplification exposes fusion opportunities
- Example: `(A + 0) * B` → `A * B` (simplified) → fuse with next op

**Principle 4: Combine Before Layout**
- Fusion changes memory access patterns
- Layout optimization must see final fused kernels to make correct decisions
- Example: Two separate kernels might have good layouts individually, but fused kernel might need different layout

**Principle 5: Global Before Local**
- High-level transformations (fusion, CSE) before low-level (LICM)
- Hoisting should operate on simplified, fused operations

**Principle 6: Iterative Refinement**
- Pipeline can be repeated until convergence (no more changes)
- Similar to LLVM's `-O3` which runs multiple iterations

#### **Alternative Orderings Considered (and Rejected)**

❌ **Fusion before CSE**:
- Problem: Might fuse redundant subgraphs, losing CSE opportunity
- Example: Fuse `(A+B)` into kernel K1 and identical `(A+B)` into kernel K2, missing that they're duplicates

❌ **LICM before Fusion**:
- Problem: LICM might hoist operations that would have been fused
- Splits code that should stay together

❌ **Algebraic after Fusion**:
- Problem: Fusion hides simplification opportunities
- Harder to pattern-match inside fused kernels

### 5.3 Pass Interaction Matrix

| **Pass** | **Enables** | **Enabled By** | **Can Conflict With** |
|----------|-------------|----------------|----------------------|
| DCE | All passes (smaller graph) | - | - |
| CSE | Algebraic (simpler patterns) | DCE (no dead code) | - |
| Algebraic | Fusion (more opportunities) | CSE (no duplicates) | - |
| Fusion | Layout (final patterns) | Algebraic (simplified) | LICM (if reversed) |
| Layout | Code quality | Fusion (stable ops) | - |
| LICM | Constant propagation | All (clean graph) | Fusion (if reversed) |

### 5.4 Iterative Optimization

**Multi-Pass Strategy** (like LLVM's `-O3`):

```python
def optimize_iterative(graph):
    changed = True
    iteration = 0
    max_iterations = 5  # Prevent infinite loops
    
    while changed and iteration < max_iterations:
        changed = False
        
        changed |= dead_code_elimination(graph)
        changed |= cse_pass(graph)
        changed |= algebraic_simplification(graph)
        changed |= operation_fusion(graph)
        changed |= memory_layout_optimization(graph)
        changed |= licm_pass(graph)
        
        iteration += 1
    
    return graph
```

**Why Iterate?**
- Early passes may expose opportunities for later passes
- Later passes may create opportunities for early passes
- Example: Fusion creates new nodes → might have dead code → run DCE again

**Convergence**: Typically converges in 2-3 iterations for tensor graphs

---

## 6. Pipeline Architecture Diagram

### 6.1 Complete End-to-End Pipeline

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                      LLVM-STYLE COMPILATION PIPELINE                      ║
║              Tensor Operations Compiler (GPU-Compiler-Optimization)        ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT: Source Code (Tensor Operations Language)                        │
│  File: test_tensor.txt                                                  │
│                                                                          │
│  program tensor_test                                                    │
│  tensor A[100,50];                                                      │
│  tensor B[50,200];                                                      │
│  begin                                                                  │
│    C := A @matmul B;                                                    │
│  end                                                                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃          FRONTEND (clang equivalent)            ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┛
                                 │
                ┌────────────────▼──────────────┐
                │  Lexer (finalAssignment.l)    │
                │  Tokenize → Keywords, Ops     │
                └────────────────┬──────────────┘
                                 │
                                 │ Token Stream
                                 │
                ┌────────────────▼──────────────┐
                │  Parser (finalAssignment.y)   │
                │  - Build AST                  │
                │  - Symbol Table Management    │
                └────────────────┬──────────────┘
                                 │
                                 │ AST + Symbol Table
                                 │
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃       IR GENERATION (IRGen equivalent)          ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┛
                                 │
                ┌────────────────▼──────────────┐
                │  TensorGraph Builder          │
                │  - AST → GraphNode            │
                │  - Establish dependencies     │
                │  - Compute TensorDim          │
                └────────────────┬──────────────┘
                                 │
                                 │ Parallel Path
                                 │
                ┌────────────────▼──────────────┐
                │  Geometry Module              │
                │  - Tensor → IterationSpace    │
                │  - Bounds Analysis            │
                └────────────────┬──────────────┘
                                 │
                                 │ TensorGraph (IR)
                                 │
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃      IR ANALYSIS (Analysis Passes)              ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┛
                                 │
                ┌────────────────▼──────────────┐
                │  Graph Analysis               │
                │  - Verify DAG property        │
                │  - Build use-def chains       │
                │  - Shape/type checking        │
                │  - Calculate baseline metrics │
                └────────────────┬──────────────┘
                                 │
                                 │ Analyzed IR
                                 │
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃     OPTIMIZATION PASSES (opt -O3 equivalent)     ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┛
                                 │
                ┌────────────────▼──────────────┐
                │  [1] Dead Code Elimination    │
                │      Remove unused ops        │
                └────────────────┬──────────────┘
                                 │
                ┌────────────────▼──────────────┐
                │  [2] Common Subexpr Elim.     │
                │      Hash-based deduplication │
                └────────────────┬──────────────┘
                                 │
                ┌────────────────▼──────────────┐
                │  [3] Algebraic Simplification │
                │      Pattern matching         │
                └────────────────┬──────────────┘
                                 │
                ┌────────────────▼──────────────┐
                │  [4] Operation Fusion         │
                │      Kernel merging           │
                └────────────────┬──────────────┘
                                 │
                ┌────────────────▼──────────────┐
                │  [5] Memory Layout Opt.       │
                │      Cache/coalescing         │
                └────────────────┬──────────────┘
                                 │
                ┌────────────────▼──────────────┐
                │  [6] Loop-Invariant Code      │
                │      Motion (LICM)            │
                └────────────────┬──────────────┘
                                 │
                                 │ Optimized TensorGraph
                                 │
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃      BACKEND CODE GENERATION (CodeGen)          ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┛
                                 │
                ┌────────────────▼──────────────┐
                │  Instruction Selection        │
                │  GraphNode → CUDA patterns    │
                └────────────────┬──────────────┘
                                 │
                ┌────────────────▼──────────────┐
                │  Resource Allocation          │
                │  - Grid/Block sizing          │
                │  - Shared memory allocation   │
                │  - Thread mapping             │
                └────────────────┬──────────────┘
                                 │
                ┌────────────────▼──────────────┐
                │  Code Emission                │
                │  - Generate kernel functions  │
                │  - Generate host code         │
                │  - Generate main() driver     │
                └────────────────┬──────────────┘
                                 │
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│  OUTPUT: generated_kernels.cu (CUDA Source Code)                   │
│                                                                     │
│  __global__ void matmul_kernel_2(float *A, float *B, float *C,     │
│                                   int M, int N, int K) { ... }     │
│  int main() { /* kernel launches */ }                              │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ Post-Compilation
                                 │
                     ┌───────────▼───────────┐
                     │  nvcc (NVIDIA)        │
                     │  PTX → Binary         │
                     └───────────┬───────────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │  GPU Execution │
                        └────────────────┘
```

**Data Flow**: Source → Tokens → AST → TensorGraph IR → Optimized IR → CUDA → Binary

### 5.2 Component Interactions

- **Lexer/Parser** → AST + Symbol Table
- **TensorGraph Builder** → IR from AST
- **Geometry Module** → IterationSpace analysis
- **Optimizer** → 6-pass transformations (DCE→CSE→Algebraic→Fusion→Layout→LICM)
- **CUDA CodeGen** → Kernel emission
- **Output** → `generated_kernels.cu`

## 6. Summary

### 6.1 Key Design Decisions

- **IR**: TensorGraph (DAG with SSA-like properties)
- **Passes**: 6 ordered transformations (20-50% reduction typical)
- **Ordering**: Clean→Redundancy→Simplify→Combine→Layout→Hoist
- **Target**: CUDA (GPU-optimized code generation)

### 6.2 Comparison to Compilers

| Aspect | This Project | LLVM | Related Work |
|--------|--------------|------|--------------||
| **IR** | DAG (TensorGraph) + LLVM IR | CFG + SSA | Halide, TVM, XLA |
| **Domain** | Tensor ops | General | ML/Image-specific |
| **Backend** | CUDA + LLVM IR (dual) | Machine code | Domain-specific |
| **Polyhedral** | Integrated | Polly (optional) | Pluto, PPCG |
| **Optimization** | IR-level analysis | ~30% (-O3) | Comparable |
| **Analysis** | LLVM passes (opt) | Built-in | Framework-specific |

### 6.3 Design Strengths

✅ Clean separation (frontend/middle/backend)  
✅ Extensible pass architecture  
✅ Measurable metrics (nodes, memory, kernels)  
✅ DAG ensures correctness  
✅ Integrated polyhedral analysis

### 6.4 Implementation Roadmap (Design Only)

**Phase 1**: PassManager infrastructure  
**Phase 2**: Core passes (DCE, CSE, Algebraic)  
**Phase 3**: Advanced passes (Fusion, Layout, LICM)  
**Phase 4**: Integration and iteration  
**Phase 5**: Testing and validation

**Estimated**: ~5000 LOC, 4-6 weeks

---

# PART II: INTEGRATION & USAGE

---

## 6. Installation & Setup
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

---

# PART II: INTEGRATION & USAGE

---

## 6. Overview & Benefits

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

---

## 7. Installation & Setup

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

---

## 8. Usage Workflow

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

---

## 9. LLVM Analysis Pipeline

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

---

## 10. Interpreting Results

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

---

## 11. Test Cases & Validation

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

---

## 12. Troubleshooting

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
---

# PART III: TESTING & BENCHMARKS

---

## 13. Test Cases and Benchmarks
**LLVM-Style Pipeline Validation**

### 13.1 Pipeline Mapping

### 13.1 Pipeline Mapping

#### Shell Pipeline to LLVM

| **Shell Stage** | **LLVM Equivalent** | **Purpose** | **Validation Focus** |
|-----------------|---------------------|-------------|---------------------|
| `make` | `clang` | Lexical/Syntax | Parse correctness |
| `./compiler` | `clang -emit-llvm` | IR Generation | TensorGraph construction |
| Optimizer | `opt` | IR Optimization | Transformations |
| `cuda_gen.c` | `llc` | Code Generation | Kernel emission |
| `nvcc` | `as` + `ld` | Target compilation | Binary |
| `./binary` | Executable | Execution | Validation |

**Key Analogy**: `./compiler` ≈ `clang -O0 -emit-llvm` + `opt -O3` + `llc`

### 13.2 Test Cases

### 2.1 Five Core Tests

| Test | Pass | LLVM Analogue | Validation |
|------|------|---------------|------------|
| TC-1 | DCE | `-dce` | Unused ops removed |
| TC-2 | CSE | `-early-cse`, `-gvn` | Duplicates merged |
| TC-3 | Algebraic | `-instcombine` | Identities eliminated |
| TC-4 | Fusion | Polly | Kernels fused |
| TC-5 | Memory | `-mem2reg`, `-sroa` | Load/store minimized |

### 2.2 TC-1: Dead Code Elimination

**Input:**
```
program test_dce
tensor A[10,10];
tensor B[10,10];
tensor C[10,10];
tensor D[10,10];
begin
  C := A @matmul B;    # Used
  D := A @matmul B;    # Dead: not used
end
```

**Expected**: 3 nodes (A, B, C), D eliminated

**LLVM Reference:**
```c
void compute(float* A, float* B, float* C) {
  float sum1 = 0.0f;
  float sum2 = 0.0f;  // Dead
  for (int i = 0; i < 100; i++) sum1 += A[i] * B[i];
  for (int i = 0; i < 100; i++) sum2 += A[i] * B[i];  // Unused
}
```

**Commands:**
```bash
clang -O0 -emit-llvm -S test_dce.c -o test_dce_O0.ll
opt -dce -S test_dce_O0.ll -o test_dce_dce.ll
grep "sum2" test_dce_dce.ll  # Empty = success
```

**Validation:**
```bash
./compiler < tests/test_dce.txt | tee log
grep "^Node" log | wc -l  # Count nodes
grep "\[D\]" log  # Empty = D removed
```

**Metrics:**
| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Node Count | 4 | 3 | 25% |
| MATMUL Ops | 2 | 1 | 50% |
| Memory | 4 tensors | 3 tensors | 25% |

### 2.3 TC-2: Common Subexpression Elimination

**Input:**
```
program test_cse
tensor A[10,10];
tensor B[10,10];
tensor C[10,10];
tensor D[10,10];
tensor E[10,10];
begin
  C := A @matmul B;    # First computation
  D := A @matmul B;    # Duplicate (CSE candidate)
  E := C + D;          # D is duplicate of C
end
```

**Expected**: Single matmul, E uses result twice

**LLVM Reference:**
```c
float compute_cse(float* A, float* B) {
  float x = A[0] * B[0];
  float y = A[0] * B[0];  // Duplicate
  return x + y;
}
```

**Commands:**
```bash
clang -O0 -emit-llvm -S test_cse.c -o O0.ll
opt -early-cse -S O0.ll -o cse.ll
grep "mul float" O0.ll | wc -l  # 2
grep "mul float" cse.ll | wc -l  # 1
```

**Metrics:**
| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Node Count | 5 | 4 | 20% |
| MATMUL Ops | 2 | 1 | 50% |
| Memory Access | 2×(N×K+K×M) | 1×(N×K+K×M) | 50% |
| Compute | 2×N×K×M | 1×N×K×M | 50% |

### 2.4 TC-3: Algebraic Simplification

**Input:**
```
program test_algebraic
tensor A[10,10];
tensor Zero[10,10];      # Zero tensor
tensor One[10,10];       # Identity (all 1s)
tensor C[10,10];
tensor D[10,10];
tensor E[10,10];
begin
  C := A + Zero;         # Identity: A + 0 = A
  D := A * One;          # Identity: A * 1 = A
  E := C * D;            # Simplifies to A * A
end
```

**Expected**: Identity ops removed (A+0→A, A*1→A)

**LLVM Reference:**
```c
float algebraic(float x) {
  float y = x + 0.0f;  // Identity
  float z = x * 1.0f;  // Identity
  return y * z;
}
```

**Commands:**
```bash
clang -O0 -emit-llvm -S test_algebraic.c -o O0.ll
opt -instcombine -S O0.ll -o opt.ll
grep -E "(fadd|fmul)" O0.ll | wc -l  # Higher
grep -E "(fadd|fmul)" opt.ll | wc -l  # Lower
grep "fadd.*0.000000e+00" opt.ll  # Empty
```

**Metrics:**
| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Node Count | 7 | 4 | 43% |
| ADD Ops | 1 | 0 | 100% |
| MUL Ops | 2 | 1 | 50% |
| Identity Ops | 2 | 0 | 100% |

### 2.5 TC-4: Operation Fusion

**Input:**
```
program test_fusion
tensor A[1000,1000];
tensor B[1000,1000];
tensor E[1000,1000];
begin
  E := ((A + B) * A) + B;  # 3 operations fusible
end
```

**Expected**: 1 fused kernel vs 3 separate, no intermediate buffers

**LLVM Reference**: Loop fusion with Polly
```bash
clang -O1 -emit-llvm -S test_fusion.c -o O1.ll
opt -polly-opt-fusion=max -polly-codegen O1.ll -o fused.ll
grep -c "for.body" O1.ll    # 3 loops
grep -c "for.body" fused.ll  # 1 loop
```

**Metrics:**
| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Kernel Count | 3 | 1 | 67% |
| Memory Writes | 3 | 1 | 67% |
| Memory Reads | 5 | 2 | 60% |

### 2.6 TC-5: Memory Layout

**Input:**
```
program test_memory
tensor A[1000,500];
tensor B[1000,500];
tensor C[1000,500];
begin
  C := A + B;              # Sequential access
  D := @transpose(A);      # Pattern change
end
```

**Expected**: Coalescing detection, locality-aware scheduling

**LLVM Reference**: mem2reg + SROA
```bash
clang -O1 -emit-llvm -S test_memory.c -o O1.ll
opt -mem2reg -sroa -S O1.ll -o opt.ll
grep -c "load\|store" O1.ll   # Higher
grep -c "load\|store" opt.ll  # Lower
```

**Metrics:**
| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Load/Store Count | 6+ | 4 | ~33% |
| Memory Ops Locality | Low | High | Better cache |

### 13.3 Benchmarks

| BM-1 | Mixed Operations | 20+ ops with redundancy | DCE, CSE, Algebraic |
| BM-2 | Chain Optimization | 10 sequential matmuls | CSE, Fusion, LICM |
| BM-3 | Polyhedral Model | Nested loops (5 deep) | Loop opt, LICM |

**Validation**: Compare IR metrics (node count, memory ops) before/after O0→O3.

### 3.2 BM-1: Mixed Operations

**Input:**
```
begin
  # 20+ operations mixing matmuls, adds, multiplies
  C := A @matmul B;
  D := A @matmul B;    # Duplicate for CSE
  E := C + D;
  # ... (more operations)
end
```

**Metrics:**
| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Node Count | 20+ | 8-12 | ~50% |
| MATMUL Ops | 10 | 5-6 | ~50% |
| Kernel Count | 20+ | 5-8 | ~65% |
| IR Size | Baseline | -40% | Smaller |

### 3.3 BM-2: Chain Optimization

**Input:**
```
begin
  C := A @matmul B;
  D := C @matmul A;
  E := D @matmul B;
  # ... (10 sequential matmuls)
  J := I @matmul A;
end
```

**Metrics:**
| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| MATMUL Ops | 10 | 10 | 0% (all needed) |
| Memory Buffers | 9 | 2-3 | ~70% |
| Compute Reuse | None | High | CSE effective |

### 3.4 BM-3: Polyhedral Model

**Input:**
```
begin
  # Simulate 5-level nested loops
  # Complex nested tensor operations
end
```

**Metrics:**
| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Iteration Space | N⁵ | Reduced | LICM effective |
| Memory Pattern | Scattered | Coalesced | Better locality |

### 13.4 Directory Structure

```
Compiler Optimization/
├── tests/
│   ├── test_dce.txt
│   ├── test_cse.txt
│   ├── test_algebraic.txt
│   ├── test_fusion.txt
│   └── test_memory.txt
├── benchmarks/
│   ├── benchmark_mixed.txt
│   ├── benchmark_chain.txt
│   └── benchmark_poly.txt
├── validation/
│   ├── run_tests.sh
│   ├── extract_metrics.sh
│   └── compare_llvm.sh
└── llvm_reference/
    ├── test_dce.c
    ├── test_cse.c
    └── ...
```

### 13.5 LLVM Commands Reference

### Quick Reference

| Pass | LLVM Command | Purpose |
|------|--------------|---------|
| DCE | `opt -dce` | Remove dead code |
| CSE | `opt -early-cse` or `-gvn` | Eliminate redundancy |
| Algebraic | `opt -instcombine` | Simplify operations |
| Fusion | Polly | `opt -polly-opt-fusion=max -polly-codegen` |
| Memory | mem2reg, SROA | `opt -mem2reg -sroa` |

### Example Workflow

```bash
# 1. Generate unoptimized IR
clang -O0 -emit-llvm -S test.c -o O0.ll

# 2. Apply full optimization
opt -O3 O0.ll -o O3.bc
llvm-dis O3.bc -o O3.ll

# 3. Compare metrics
grep -c "^\s*%" O0.ll  # Instruction count
grep -c "^\s*%" O3.ll

# 4. Analyze transformations
llvm-diff O0.ll O3.ll
```

---

**End of Test Cases and Benchmarks**

**Input:**
```
begin
  # Simulate 5-level nested loops
  # Complex nested tensor operations
end
```


### Quick Reference
│       ├── tests/                    # Existing test files
│       │   ├── pruebaT51.txt
│       │   ├── test_tensor.txt
│       │   └── ...
│       └── documentation/
│
├── tests/                            # NEW: Organized test suite
│   ├── unit/                         # Unit tests for individual passes
│   │   ├── test_dce.txt              # Dead code elimination
│   │   ├── test_cse.txt              # Common subexpression elimination
│   │   ├── test_algebraic.txt        # Algebraic simplification
│   │   ├── test_fusion.txt           # Operation fusion
│   │   ├── test_memory.txt           # Memory layout optimization
│   │   └── expected/                 # Expected outputs
│   │       ├── test_dce_graph.txt
│   │       ├── test_cse_graph.txt
│   │       └── ...
│   │
│   ├── integration/                  # Multi-pass integration tests
│   │   ├── test_pipeline_simple.txt
│   │   ├── test_pipeline_complex.txt
│   │   └── expected/
│   │
│   ├── regression/                   # Prevent optimization bugs
│   │   ├── test_cse_correctness.txt
│   │   ├── test_fusion_correctness.txt
│   │   └── golden/                   # Golden reference outputs
│   │
│   └── llvm_reference/               # LLVM equivalent tests
│       ├── test_dce.c
│       ├── test_cse.c
│       ├── test_algebraic.c
│       ├── Makefile                  # Build LLVM IR
│       └── README.md                 # LLVM commands
│
├── benchmarks/                       # NEW: Performance benchmarks
│   ├── bench_simple.txt              # Small test (10×10)
│   ├── bench_complex.txt             # Medium test (100×100)
│   ├── bench_pipeline.txt            # Large test (1000×1000)
│   ├── bench_mlp.txt                 # ML workload (neural net)
│   ├── bench_convolution.txt         # Conv workload
│   ├── results/                      # Benchmark results
│   │   ├── metrics_simple.json
│   │   ├── metrics_complex.json
│   │   ├── metrics_pipeline.json
│   │   └── summary.csv
│   └── scripts/                      # Automation
│       ├── run_all_benchmarks.sh
│       ├── collect_metrics.py
│       └── generate_report.py
│
├── validation/                       # NEW: Correctness validation
│   ├── semantic_checks/              # Verify optimization correctness
│   │   ├── check_cse_equivalence.py
│   │   ├── check_fusion_correctness.py
│   │   └── verify_graph_validity.py
│   ├── ir_comparison/                # Compare IRs
│   │   ├── compare_tensor_llvm.py
│   │   └── diff_graphs.py
│   └── test_runner.sh                # Automated test execution
│
├── LLVM_DESIGN.md                    # Existing: Pipeline design
├── TESTS_AND_BENCHMARKS.md           # THIS DOCUMENT
└── README.md                         # Update with testing info
```

### 4.1 Directory Purposes

| Directory | Purpose | Contents |
|-----------|---------|----------|
| `tests/unit/` | Test individual optimization passes | Minimal test cases for each pass |
| `tests/integration/` | Test complete pipeline | Multi-pass scenarios |
| `tests/regression/` | Prevent bugs | Known-good outputs (golden files) |
| `tests/llvm_reference/` | LLVM comparison baselines | C code + LLVM IR generation |
| `benchmarks/` | Performance measurement | Representative workloads |
| `benchmarks/results/` | Historical data | Track optimization improvements |
| `validation/` | Correctness tools | Scripts to verify semantic equivalence |

### 4.2 Test File Naming Convention

```
test_<pass>_<scenario>.txt

Examples:
- test_dce_basic.txt
- test_cse_matmul.txt
- test_fusion_elementwise.txt
- test_algebraic_identity.txt
```

### 4.3 Benchmark File Naming Convention

```
bench_<workload>_<size>.txt

Examples:
- bench_mlp_small.txt      (10×10 matrices)
- bench_mlp_medium.txt     (100×100 matrices)
- bench_mlp_large.txt      (1000×1000 matrices)
- bench_convolution_2d.txt
```

---

## 5. LLVM Reference Commands

### 5.1 Standard LLVM Workflow

#### **Step 1: Generate Unoptimized IR**
```bash
clang -O0 -emit-llvm -S input.c -o output_O0.ll
```

#### **Step 2: Apply Individual Passes**
```bash
# Dead Code Elimination
opt -dce -S input.ll -o output_dce.ll

# Common Subexpression Elimination
opt -early-cse -S input.ll -o output_cse.ll

# Instruction Combining (Algebraic)
opt -instcombine -S input.ll -o output_combine.ll

# Global Value Numbering (Advanced CSE)
opt -gvn -S input.ll -o output_gvn.ll

# Memory to Register Promotion
opt -mem2reg -S input.ll -o output_mem2reg.ll

# Loop optimizations
opt -loop-simplify -loop-unroll -S input.ll -o output_loop.ll
```

#### **Step 3: Apply Full Optimization Pipeline**
```bash
# Equivalent to -O3
opt -O3 input.ll -o output_O3.ll

# Custom pass pipeline
opt -mem2reg -early-cse -instcombine -gvn -dce -S input.ll -o output_custom.ll
```

#### **Step 4: Analyze IR**
```bash
# Disassemble binary IR to readable form
llvm-dis output.bc -o output.ll

# Count instructions
grep -c '^\s*%' output.ll

# Count loads/stores
grep 'load\|store' output.ll | wc -l

# View CFG (control flow graph)
opt -dot-cfg input.ll
dot -Tpng .main.dot -o cfg.png

# View call graph
opt -dot-callgraph input.ll
```

### 5.2 Comparison Commands

#### **Differential Analysis**
```bash
# Compare two IR files
llvm-diff before.ll after.ll

# Show optimization report
opt -O3 -pass-remarks=.* input.ll -o output.ll 2>&1 | grep "remark"

# Count specific instruction types
llvm-dis output.bc | grep "fadd\|fmul\|load\|store" | sort | uniq -c
```

### 5.3 Metrics Extraction Script

```bash
#!/bin/bash
# extract_llvm_metrics.sh

IR_FILE=$1
START_TIME=$(date +%s.%N)

echo "=== LLVM IR Metrics for $IR_FILE ==="

# Total instructions
TOTAL_INST=$(grep -c '^\s*%' "$IR_FILE")
echo "Total Instructions: $TOTAL_INST"

# Arithmetic operations
FADD=$(grep -c 'fadd' "$IR_FILE")
FMUL=$(grep -c 'fmul' "$IR_FILE")
echo "Arithmetic: $FADD adds, $FMUL multiplies"

# Timing
END_TIME=$(date +%s.%N)
ELAPSED=$(awk "BEGIN {printf \"%.3f\", $END_TIME - $START_TIME}")
echo "Analysis time: ${ELAPSED}s"

# Memory operations
LOADS=$(grep -c 'load' "$IR_FILE")
STORES=$(grep -c 'store' "$IR_FILE")
echo "Memory: $LOADS loads, $STORES stores"

# Control flow
BRANCHES=$(grep -c 'br' "$IR_FILE")
CALLS=$(grep -c 'call' "$IR_FILE")
echo "Control: $BRANCHES branches, $CALLS calls"

# Function count
FUNCTIONS=$(grep -c '^define' "$IR_FILE")
echo "Functions: $FUNCTIONS"

# Basic blocks
BASIC_BLOCKS=$(grep -c '^\w\+:' "$IR_FILE")
echo "Basic Blocks: $BASIC_BLOCKS"

# File size
SIZE=$(wc -c < "$IR_FILE")
echo "IR Size: $SIZE bytes"
```

---

## 6. Validation Methodology

### 6.1 Correctness Validation

#### **Principle: Semantic Equivalence**

An optimization is **correct** if:
```
∀ inputs. output_optimized(inputs) = output_unoptimized(inputs)
```

#### **Validation Approach**

**1. Graph Structure Validation**
```bash
# Check DAG property (no cycles)
python3 validation/semantic_checks/verify_graph_validity.py \
    tests/unit/test_cse.txt

# Output:
# ✓ Graph is acyclic (valid DAG)
# ✓ All nodes reachable from roots
# ✓ No undefined references
```

**2. Numerical Validation** (using LLVM)
```bash
# Generate executables
clang -O0 test.c -o test_O0
clang -O3 test.c -o test_O3

# Run with same inputs
echo "1.0 2.0 3.0" | ./test_O0 > output_O0.txt
echo "1.0 2.0 3.0" | ./test_O3 > output_O3.txt

# Compare outputs (should be identical)
diff output_O0.txt output_O3.txt
```

**3. Tensor Compiler Validation**
```bash
# Run test case
./compiler < tests/unit/test_cse.txt 2>&1 | tee test_output.log

# Extract final graph
grep "^Node" test_output.log > final_graph.txt

# Compare with expected
diff final_graph.txt tests/unit/expected/test_cse_graph.txt
```

### 6.2 Performance Validation

#### **Metric Correlation Analysis**

Validate that proxy metrics correlate with real performance:

```python
# validation/metric_correlation.py
import numpy as np
from scipy.stats import pearsonr

# Simulated data (would be from actual runs)
kernel_count = [10, 8, 6, 4, 3]        # Optimization progression
runtime_ms = [45.2, 38.1, 29.5, 21.3, 18.7]

# Calculate correlation
r, p_value = pearsonr(kernel_count, runtime_ms)
print(f"Correlation: r={r:.3f}, p={p_value:.4f}")
# Expected: r > 0.9 (strong correlation)

# Same for memory accesses
memory_accesses = [1000000, 750000, 500000, 300000, 250000]
r2, p2 = pearsonr(memory_accesses, runtime_ms)
print(f"Memory correlation: r={r2:.3f}, p={p2:.4f}")
```

### 6.3 Automated Test Runner

```bash
#!/bin/bash
# validation/test_runner.sh

echo "=== Running Test Suite ==="

PASSED=0
FAILED=0

# Run all unit tests
for test in tests/unit/test_*.txt; do
    echo "Running $test..."
    
    # Run compiler
    ./compiler < "$test" > temp_output.log 2>&1
    
    # Extract metrics
    NODES=$(grep "Operations after:" temp_output.log | awk '{print $3}')
    
    # Check against expected
    EXPECTED_FILE="tests/unit/expected/$(basename $test .txt)_nodes.txt"
    if [ -f "$EXPECTED_FILE" ]; then
        EXPECTED=$(cat "$EXPECTED_FILE")
        if [ "$NODES" == "$EXPECTED" ]; then
            echo "  ✓ PASS"
            ((PASSED++))
        else
            echo "  ✗ FAIL: Expected $EXPECTED, got $NODES"
            ((FAILED++))
        fi
    else
        echo "  ⚠ SKIP: No expected file"
    fi
done

echo ""
echo "=== Test Results ==="
echo "Passed: $PASSED"
echo "Failed: $FAILED"

exit $FAILED
```

### 6.4 Regression Testing

**Purpose**: Ensure optimizations don't break existing functionality.

```bash
# Run before making changes
./validation/test_runner.sh > baseline_results.txt

# Make changes to optimizer...

# Run after changes
./validation/test_runner.sh > new_results.txt

# Compare
diff baseline_results.txt new_results.txt
```

**Golden Files**: Store known-good outputs
```
tests/regression/golden/
├── test_cse_output.txt
├── test_fusion_output.txt
└── test_pipeline_output.txt
```

---

## 7. Integration with Existing Pipeline

### 7.1 Mapping Shell Pipeline to Tests

| Shell Command | Test Focus | Test Cases |
|---------------|------------|------------|
| `make` | Compilation correctness | Build system tests (not covered here) |
| `./compiler < input.txt` | Full pipeline | All test cases in `tests/` |
| `awk '...' output.log` | IR analysis | Metric extraction tests |
| `nvcc -O3 *.cu` | Backend compilation | Code generation tests |
| `./binary` | Execution correctness | Semantic validation |

### 7.2 Enhanced Pipeline with Testing

```bash
#!/bin/bash
# run_pipeline_with_tests.sh

INPUT_FILE=$1
TEST_MODE=${2:-"none"}  # none, validate, benchmark

# Stage 1: Compile
make

# Stage 2: Run compiler with metric collection
if [ "$TEST_MODE" == "benchmark" ]; then
    echo "==> Running in BENCHMARK mode"
    ./compiler < "$INPUT_FILE" 2>&1 | tee compiler_output.log
    
    # Extract metrics
    python3 benchmarks/scripts/collect_metrics.py compiler_output.log \
        > benchmarks/results/metrics_$(basename "$INPUT_FILE" .txt).json
else
    ./compiler < "$INPUT_FILE" 2>&1 | tee compiler_output.log
fi

# Stage 3: Validate correctness
if [ "$TEST_MODE" == "validate" ]; then
    echo "==> Validating graph structure"
    python3 validation/semantic_checks/verify_graph_validity.py \
        compiler_output.log
fi

# Stage 4-6: Continue with normal pipeline
# (existing code from run_pipeline.sh)
```

### 7.3 Continuous Integration

```yaml
# .github/workflows/test.yml (if using GitHub Actions)
name: Compiler Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y flex bison gcc make
      
      - name: Build compiler
        run: cd "Compiler Optimization/Flex-Bison" && make
      
      - name: Run unit tests
        run: bash validation/test_runner.sh
      
      - name: Run benchmarks
        run: bash benchmarks/scripts/run_all_benchmarks.sh
      
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: benchmarks/results/
```

---

## 8. Conclusion

### 8.1 Summary of Contributions

This document provides:

1. **5 Concrete Test Cases** mapping to LLVM optimization passes
   - DCE, CSE, Algebraic, Fusion, Memory Layout

2. **3 Comprehensive Benchmarks** with IR-level metrics
   - Simple, Complex, Pipeline scenarios
   - Before/After comparison tables

3. **LLVM Reference Framework** for validation
   - Equivalent C code for each test
   - LLVM command sequences
   - Metric extraction scripts

4. **Organized Directory Structure** for maintainability
   - `tests/unit/`, `tests/integration/`, `tests/regression/`
   - `benchmarks/`, `validation/`

5. **Validation Methodology** ensuring correctness
   - Semantic equivalence checks
   - Regression testing
   - Automated test runners

### 8.2 Why This Approach Works

**Academic Rigor**: Grounded in compiler theory and empirical studies

**Practical Validity**: Uses industry-standard LLVM as reference

**No GPU Required**: IR-level metrics are validated proxies for performance

**Maintainable**: Clear structure, documented commands, automated scripts

### 8.3 Expected Outcomes

Running these tests should demonstrate:

- **Correctness**: All optimizations preserve semantics (diff = 0 on outputs)
- **Effectiveness**: 20-50% reduction in IR size, memory ops, kernel count
- **LLVM Parity**: Similar optimization levels to `clang -O3`
- **Regression Safety**: Golden files prevent optimization bugs

### 8.4 Future Extensions

**Potential Additions**:

1. **Auto-Tuning Tests**: Validate parameter selection (block size, etc.)
2. **Polyhedral Tests**: Advanced loop transformations (tiling, skewing)
3. **Numerical Precision Tests**: Validate floating-point optimizations
4. **Scalability Tests**: Large graphs (10,000+ nodes)
5. **Comparative Analysis**: Benchmark against TVM, XLA, Halide

### 8.5 Alignment with LLVM Design

This testing framework directly validates the **LLVM_DESIGN.md** pipeline:

| LLVM Design Stage | Test Coverage |
|-------------------|---------------|
| Frontend (Lexer/Parser) | Implicit (tests parse correctly) |
| IR Generation | All tests validate TensorGraph |
| Pass 1: DCE | TC-1 (test_dce.txt) |
| Pass 2: CSE | TC-2 (test_cse.txt) |
| Pass 3: Algebraic | TC-3 (test_algebraic.txt) |
| Pass 4: Fusion | TC-4 (test_fusion.txt) |
| Pass 5: Memory Layout | TC-5 (test_memory.txt) |
| Pass 6: LICM | (Covered in bench_pipeline) |
| Backend (CodeGen) | Kernel count metrics |

---

## Appendix A: Quick Start Guide

### A.1 Running a Single Test

```bash
cd "Compiler Optimization/Flex-Bison"

# Build compiler
make

# Run test case
./compiler < ../../tests/unit/test_cse.txt 2>&1 | tee test_output.log

# Check results
grep "Operations after:" test_output.log
grep "Memory reduction:" test_output.log
```

### A.2 Running LLVM Comparison

```bash
cd tests/llvm_reference

# Generate IR
make test_cse_ir

# Compare metrics
bash compare_metrics.sh test_cse
```

### A.3 Running Full Benchmark Suite

```bash
# From repository root
bash benchmarks/scripts/run_all_benchmarks.sh

# View results
cat benchmarks/results/summary.csv
```

---

## Appendix B: Metric Extraction Scripts

### B.1 TensorGraph Metrics Extractor

```python
#!/usr/bin/env python3
# benchmarks/scripts/collect_metrics.py

import sys
import re
import json

def extract_metrics(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract node count
    match = re.search(r'Operations before:\s*(\d+)', content)
    if match:
        metrics['nodes_before'] = int(match.group(1))
    
    match = re.search(r'Operations after:\s*(\d+)', content)
    if match:
        metrics['nodes_after'] = int(match.group(1))
    
    # Extract optimization percentages
    match = re.search(r'Memory reduction:\s*(\d+)%', content)
    if match:
        metrics['memory_reduction_pct'] = int(match.group(1))
    
    match = re.search(r'Compute reduction:\s*(\d+)%', content)
    if match:
        metrics['compute_reduction_pct'] = int(match.group(1))
    
    # Count operation types
    metrics['matmul_ops'] = len(re.findall(r'Node \d+: MATMUL', content))
    metrics['add_ops'] = len(re.findall(r'Node \d+: ADD', content))
    metrics['mul_ops'] = len(re.findall(r'Node \d+: MUL', content))
    
    return metrics

if __name__ == '__main__':
    metrics = extract_metrics(sys.argv[1])
    print(json.dumps(metrics, indent=2))
```

### B.2 LLVM Metrics Extractor

```bash
#!/bin/bash
# tests/llvm_reference/extract_metrics.sh

IR_FILE=$1

echo "{"
echo "  \"total_instructions\": $(grep -c '^\s*%' "$IR_FILE"),"
echo "  \"load_ops\": $(grep -c 'load' "$IR_FILE"),"
echo "  \"store_ops\": $(grep -c 'store' "$IR_FILE"),"
echo "  \"fadd_ops\": $(grep -c 'fadd' "$IR_FILE"),"
echo "  \"fmul_ops\": $(grep -c 'fmul' "$IR_FILE"),"
echo "  \"functions\": $(grep -c '^define' "$IR_FILE"),"
echo "  \"size_bytes\": $(wc -c < "$IR_FILE")"
echo "}"
```

---

## References

1. **LLVM Documentation**: LLVM Language Reference Manual, https://llvm.org/docs/LangRef.html
2. **Compiler Optimization**: Cooper & Torczon (2011), "Engineering a Compiler", 2nd Ed.
3. **GPU Performance**: Kirk & Hwu (2016), "Programming Massively Parallel Processors"
4. **Roofline Model**: Williams et al. (2009), "Roofline: An Insightful Visual Performance Model"
5. **Polyhedral Compilation**: Bondhugula et al. (2008), "A Practical Automatic Polyhedral Parallelizer"
6. **CSE/GVN**: Alpern et al. (1988), "Detecting Equality of Variables in Programs"
7. **Loop Fusion**: McKinley et al. (1996), "Improving Data Locality with Loop Transformations"
8. **GPU Memory**: Hong & Kim (2009), "An Analytical Model for GPU Architectures"

---

**Document Version**: 1.0  
**Author**: Compiler Testing & Validation Team  
**Date**: December 21, 2025  
**Project**: GPU-Compiler-Optimization (LLVM-Design Branch)  
**Related Documents**: LLVM_DESIGN.md  
**Status**: Design Complete - Ready for Implementation of Test Infrastructure
