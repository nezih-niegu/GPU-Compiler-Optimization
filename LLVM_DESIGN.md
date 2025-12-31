# LLVM-Style Compilation Pipeline Design
**GPU Compiler Optimization: Tensor Operations Framework**

## Executive Summary

This document maps the tensor-based compiler framework to LLVM concepts, establishing a systematic multi-pass optimization architecture.

**Core IR**: `TensorGraph` (DAG-based)  
**Targets**: CUDA Code + LLVM IR (parallel generation)  
**Approach**: Graph transformations with polyhedral analysis + LLVM-based optimization

## Table of Contents

1. [IR and Component Mapping](#1-ir-and-component-mapping)
2. [Compilation Pipeline](#2-compilation-pipeline)
3. [Optimization Passes](#3-optimization-passes)
4. [Pass Ordering](#4-pass-ordering)
5. [Architecture Diagrams](#5-architecture-diagrams)

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