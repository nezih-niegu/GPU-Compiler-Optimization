# LLVM Integration - Code Modifications Required

This document describes the manual code changes needed to integrate LLVM IR generation into the compiler.

## Required Changes

### 1. Modify `finalAssignment.y` (Parser)

**Location**: Line ~10 (after existing includes)

**Add include directive**:
```c
#include "types.h"
#include "tensor_graph.h"
#include "optimizer.h"
#include "cuda_gen.h"
#include "geometry.h"
#include "llvm_lowering.h"  /* ← ADD THIS LINE */
```

**Location**: Line ~119 (in `start_program` action, after `generate_cuda_code`)

**Add LLVM IR generation**:
```c
        /* Generate CUDA code */
        generate_cuda_code(tensorGraph, "generated_kernels.cu");
        printf("\nCUDA code generated in: generated_kernels.cu\n");
        
        /* ↓ ADD THESE LINES ↓ */
        /* Generate LLVM IR for analysis */
        if (lower_graph_to_llvm_ir(tensorGraph, "tensor_output.ll") == 0) {
            printf("LLVM IR generated in: tensor_output.ll\n");
            printf("Run './llvm_analyze.sh tensor_output.ll' for analysis\n");
        } else {
            fprintf(stderr, "Warning: LLVM IR generation failed\n");
        }
        /* ↑ END OF NEW CODE ↑ */
        
        free_opt_result(opt_result);
```

### 2. Rebuild Compiler

After making the above changes:

```bash
cd "Compiler Optimization/Flex-Bison"
make clean
make
```

Expected output:
```
bison -d finalAssignment.y
flex finalAssignment.l
gcc -Wall -g -c tensor_graph.c -o tensor_graph.o
gcc -Wall -g -c optimizer.c -o optimizer.o
gcc -Wall -g -c cuda_gen.c -o cuda_gen.o
gcc -Wall -g -c geometry.c -o geometry.o
gcc -Wall -g -c llvm_lowering.c -o llvm_lowering.o  ← NEW
gcc -Wall -g -c lex.yy.c -o lex.yy.o
gcc -Wall -g -c finalAssignment.tab.c -o finalAssignment.tab.o
gcc -Wall -g -o compiler lex.yy.o finalAssignment.tab.o tensor_graph.o optimizer.o cuda_gen.o geometry.o llvm_lowering.o -lm
```

### 3. Test Integration

```bash
# Run simple test
./compiler < tests/simple_test.txt

# Expected output should include:
# "LLVM IR generated in: tensor_output.ll"

# Verify IR file exists
ls -la tensor_output.ll

# Run analysis
./llvm_analyze.sh tensor_output.ll
```

## Alternative: Quick Setup Script

Use the automated setup script:

```bash
./setup_llvm_integration.sh
```

This script will:
1. Verify LLVM installation
2. Build the compiler
3. Prompt for manual code changes (if needed)
4. Run a test to verify integration
5. Generate and analyze sample IR

## Troubleshooting

### "undefined reference to `lower_graph_to_llvm_ir`"

**Problem**: Linker can't find LLVM lowering functions

**Solution**: Make sure `llvm_lowering.c` is compiled:
```bash
gcc -Wall -g -c llvm_lowering.c -o llvm_lowering.o
```

And linked into the final binary (check Makefile `C_SRCS`).

### No IR file generated

**Problem**: `tensor_output.ll` not created

**Solution**: 
1. Check if you added the `lower_graph_to_llvm_ir()` call
2. Check compiler output for error messages
3. Verify tensorGraph is not NULL
4. Run with stderr visible: `./compiler < tests/simple_test.txt 2>&1`

### Invalid LLVM IR

**Problem**: `llvm-as` reports syntax errors

**Solution**:
1. Check `tensor_output.ll` manually
2. Common issues:
   - Missing closing braces in functions
   - Incorrect register numbering
   - Invalid type strings
3. Compare with reference: `llvm-clang -O0 -S -emit-llvm test.c -o reference.ll`

## Verification Checklist

- [ ] `llvm_lowering.h` included in `finalAssignment.y`
- [ ] `lower_graph_to_llvm_ir()` called in parser action
- [ ] Compiler builds without errors
- [ ] `llvm_lowering.o` in object file list
- [ ] Test generates `tensor_output.ll`
- [ ] IR passes `llvm-as` verification
- [ ] Analysis script runs successfully

## Next Steps

Once integration is complete:

1. **Run full test suite**: `./run_llvm_tests.sh`
2. **Review test results**: `cat llvm_test_results/test_summary.txt`
3. **Study generated IR**: `less tensor_output.ll`
4. **Analyze optimizations**: Review `llvm_output/*_summary.txt` files
5. **Document findings**: Add results to test documentation

## Complete Integration Example

Here's the complete modified section of `finalAssignment.y`:

```c
start_program:
    prog {
      /* (*) symbolTable has already been created in main() before yyparse() */
      rootAST = $1;
      execAST(rootAST);
      printf("\n=== Execution tree by levels ===\n");
      printTreeLevels(rootAST);
      print_table(symbolTable);
      
      /* Process tensor operations if graph exists */
      if (tensorGraph && tensorGraph->num_nodes > 0) {
        print_graph(tensorGraph);
        
        /* Apply optimizations */
        OptStrategy strategies[] = {OPT_FUSE_OPS, OPT_COMMON_SUBEXPR, OPT_MEMORY_LAYOUT};
        OptResult *opt_result = optimize_graph(tensorGraph, strategies, 3);
        print_optimization_report(opt_result);
        
        /* Generate CUDA code (original backend) */
        generate_cuda_code(tensorGraph, "generated_kernels.cu");
        printf("\nCUDA code generated in: generated_kernels.cu\n");
        
        /* Generate LLVM IR (new backend for analysis) */
        if (lower_graph_to_llvm_ir(tensorGraph, "tensor_output.ll") == 0) {
            printf("LLVM IR generated in: tensor_output.ll\n");
            printf("Run './llvm_analyze.sh tensor_output.ll' for detailed analysis\n");
        } else {
            fprintf(stderr, "Warning: LLVM IR generation failed\n");
        }
        
        free_opt_result(opt_result);
      }
      
      if (tensorGraph) free_tensor_graph(tensorGraph);
      free_table(symbolTable);
    }
;
```

## Reference Files

- **LLVM Integration Guide**: `../../../LLVM_INTEGRATION.md`
- **Design Document**: `../../../LLVM_DESIGN.md`
- **Test Specifications**: `../../../TESTS_AND_BENCHMARKS.md`
- **LLVM Lowering Header**: `llvm_lowering.h`
- **LLVM Lowering Implementation**: `llvm_lowering.c`
- **Analysis Script**: `llvm_analyze.sh`
- **Configuration**: `llvm_config.sh`
