#!/bin/bash
# LLVM Configuration for Tensor Compiler
# This script sets up paths to your local LLVM installation

# LLVM installation directory (adjust if your build directory name differs)
LLVM_BUILD_DIR="$HOME/Documents/GitHub/llvm-project/build"

# Check if directory exists
if [ ! -d "$LLVM_BUILD_DIR" ]; then
    echo "ERROR: LLVM build directory not found: $LLVM_BUILD_DIR"
    echo "Please update LLVM_BUILD_DIR in llvm_config.sh"
    exit 1
fi

# Export LLVM tool paths
export LLVM_BIN="$LLVM_BUILD_DIR/bin"
export LLVM_CLANG="$LLVM_BIN/clang"
export LLVM_OPT="$LLVM_BIN/opt"
export LLVM_LLC="$LLVM_BIN/llc"
export LLVM_DIS="$LLVM_BIN/llvm-dis"
export LLVM_AS="$LLVM_BIN/llvm-as"
export LLVM_LLI="$LLVM_BIN/lli"
export LLVM_LINK="$LLVM_BIN/llvm-link"

# Verify tools exist
for tool in clang opt llc llvm-dis llvm-as; do
    if [ ! -x "$LLVM_BIN/$tool" ]; then
        echo "ERROR: LLVM tool not found: $LLVM_BIN/$tool"
        exit 1
    fi
done

echo "✓ LLVM tools configured:"
echo "  LLVM_CLANG: $LLVM_CLANG"
echo "  LLVM_OPT:   $LLVM_OPT"
echo "  LLVM_LLC:   $LLVM_LLC"
echo "  LLVM_DIS:   $LLVM_DIS"

# Add LLVM bin to PATH for convenience
export PATH="$LLVM_BIN:$PATH"
