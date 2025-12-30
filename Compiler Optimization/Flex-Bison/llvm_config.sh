#!/usr/bin/env bash
# llvm_config.sh - LLVM Configuration for Tensor Compiler (SOURCE-SAFE)
#
# Usage:
#   source ./llvm_config.sh            # prefer local build if present, else system LLVM
#   LLVM_BUILD_DIR=/custom/build source ./llvm_config.sh
#   LLVM_PREFER_SYSTEM=1 source ./llvm_config.sh   # force system LLVM
#
# Exports:
#   LLVM_BIN LLVM_CLANG LLVM_OPT LLVM_LLC LLVM_DIS LLVM_AS LLVM_LLI LLVM_LINK
#   and prepends LLVM_BIN to PATH

set -u  # (don't use -e here; we want to handle errors gracefully when sourced)

# Detect whether this file is being sourced
_is_sourced() {
  # bash: return true if sourced
  [[ "${BASH_SOURCE[0]}" != "${0}" ]]
}

# Exit or return depending on sourced/executed
_die() {
  echo "ERROR: $*" >&2
  if _is_sourced; then
    return 1
  else
    exit 1
  fi
}

_warn() { echo "WARN: $*" >&2; }

# Prefer system LLVM if requested
LLVM_PREFER_SYSTEM="${LLVM_PREFER_SYSTEM:-0}"

# Default local build dir (user can override via env var)
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$HOME/Documents/GitHub/llvm-project/build}"
LLVM_BIN=""

# Helper: configure from a given bin dir
_configure_from_bin() {
  local bin="$1"

  export LLVM_BIN="$bin"
  export LLVM_CLANG="$LLVM_BIN/clang"
  export LLVM_OPT="$LLVM_BIN/opt"
  export LLVM_LLC="$LLVM_BIN/llc"
  export LLVM_DIS="$LLVM_BIN/llvm-dis"
  export LLVM_AS="$LLVM_BIN/llvm-as"
  export LLVM_LLI="$LLVM_BIN/lli"
  export LLVM_LINK="$LLVM_BIN/llvm-link"

  # Verify a minimal set (expand if you need more)
  local missing=0
  for tool in clang opt llc llvm-dis llvm-as; do
    if [ ! -x "$LLVM_BIN/$tool" ]; then
      _warn "Missing tool: $LLVM_BIN/$tool"
      missing=1
    fi
  done

  if [ "$missing" -ne 0 ]; then
    return 1
  fi

  # Prepend to PATH (avoid duplicating)
  case ":$PATH:" in
    *":$LLVM_BIN:"*) : ;;
    *) export PATH="$LLVM_BIN:$PATH" ;;
  esac

  return 0
}

# Helper: configure from system PATH tools
_configure_from_system() {
  local t
  for t in llvm-as opt llvm-dis llc clang; do
    command -v "$t" >/dev/null 2>&1 || return 1
  done

  export LLVM_AS="$(command -v llvm-as)"
  export LLVM_OPT="$(command -v opt)"
  export LLVM_DIS="$(command -v llvm-dis)"
  export LLVM_LLC="$(command -v llc)"
  export LLVM_CLANG="$(command -v clang)"

  # Optional tools (may not exist)
  export LLVM_LLI="$(command -v lli 2>/dev/null || true)"
  export LLVM_LINK="$(command -v llvm-link 2>/dev/null || true)"

  export LLVM_BIN="$(dirname "$LLVM_AS")"

  case ":$PATH:" in
    *":$LLVM_BIN:"*) : ;;
    *) export PATH="$LLVM_BIN:$PATH" ;;
  esac

  return 0
}

# ----------------------------
# Main selection logic
# ----------------------------
if [ "$LLVM_PREFER_SYSTEM" = "1" ]; then
  if _configure_from_system; then
    echo "✓ LLVM tools configured (system): $LLVM_BIN"
  else
    _die "System LLVM tools not found in PATH. Install e.g.: sudo apt-get install llvm clang"
  fi
else
  # Try local build first
  if [ -d "$LLVM_BUILD_DIR/bin" ] && _configure_from_bin "$LLVM_BUILD_DIR/bin"; then
    echo "✓ LLVM tools configured (local build): $LLVM_BUILD_DIR/bin"
  else
    # Fallback to system
    if _configure_from_system; then
      echo "✓ LLVM tools configured (system fallback): $LLVM_BIN"
      echo "  (Local build not found or incomplete at: $LLVM_BUILD_DIR/bin)"
    else
      _die "No usable LLVM toolchain found. Fix LLVM_BUILD_DIR or install system LLVM."
    fi
  fi
fi
