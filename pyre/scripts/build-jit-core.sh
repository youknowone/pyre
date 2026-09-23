#!/bin/sh
# Fast core for JIT-quality loops. Drops the native-library modules behind
# the `full` feature (`_socket`, `_ssl`, `_ctypes`, `_cffi_backend`, `mmap`,
# `select`) and does not extract `pyre-module` or wasm layout sidecars.
# The product `build/llbc` set is left alone.
#
#   pyre/scripts/build-jit-core.sh
#   pyre/scripts/build-jit-core.sh --check   # cargo check, no release codegen
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
MODE=release
if [ "${1:-}" = "--check" ]; then
    MODE=check
fi
export PYRE_JIT_CORE=1
export LLBC_LAYOUT_TARGETS=
export LLBC_DEST="$ROOT/build/llbc-jit-core"
python3 pyre/scripts/extract-llbc.py majit-rlib pyre-object pyre-interpreter pyre-jit
LLBC="$LLBC_DEST"
export MAJIT_MIR_FRONTEND_LLBC="$LLBC/majit-rlib.ullbc:$LLBC/pyre-object.ullbc:$LLBC/pyre-interpreter.ullbc:$LLBC/pyre-jit.ullbc"
export PYRE_LLBC_SKIP_FINGERPRINT_CHECK=1
if [ "$MODE" = check ]; then
    exec cargo check -p pyrex --bin pyre-dynasm --no-default-features --features dynasm,mimalloc
fi
exec cargo build --release -p pyrex --bin pyre-dynasm --no-default-features --features dynasm,mimalloc
