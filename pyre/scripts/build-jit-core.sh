#!/bin/sh
# Fast core for JIT-quality loops. Leaves `pyre-module` out of the binary and
# out of the extraction; the other artefacts are the product `build/llbc` set.
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
python3 pyre/scripts/extract-llbc.py majit-rlib pyre-object pyre-interpreter pyre-jit
# Without `pyre-module`, `pyre-jit-trace`'s prepass neither links nor hashes
# `pyre-module` and does not read its artefact.
if [ "$MODE" = check ]; then
    exec cargo check -p pyrex --bin pyre-dynasm --no-default-features --features dynasm,mimalloc
fi
exec cargo build --release -p pyrex --bin pyre-dynasm --no-default-features --features dynasm,mimalloc
