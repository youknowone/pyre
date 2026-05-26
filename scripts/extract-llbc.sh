#!/usr/bin/env bash
# extract-llbc.sh — run Charon on the JIT-consumed crates and drop
# `.ullbc` artefacts into ./build/llbc/.
#
# The Charon migration described in issue #97 lowers MIR-derived IR
# (`.ullbc`) into pyre's `FunctionGraph`. This script is the producer
# of those `.ullbc` files; the consumer is `majit-charon-reader` plus
# whatever the Step 3 driver wires up.
#
# Usage:
#   scripts/extract-llbc.sh                  # extract all JIT-consumed crates
#   scripts/extract-llbc.sh pyre-object      # extract one crate
#   scripts/extract-llbc.sh corpus           # extract the charon-spike corpus
#   LLBC_DEST=./out scripts/extract-llbc.sh  # override output dir
#
# Notes:
#   - Charon invokes `cargo build` internally under its pinned nightly
#     toolchain. The first run downloads / installs that toolchain.
#   - `pyre-interpreter` requires a JIT backend feature to compile.
#     We default to `cranelift`; override with CARGO_FEATURES=dynasm.
#   - Outputs are NOT committed (see /build/ in .gitignore). Re-run
#     this script after source changes; Cargo's incremental cache
#     keeps re-runs cheap.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
charon_bin="$repo_root/build/charon/charon"

if [[ ! -x "$charon_bin" ]]; then
    echo "extract-llbc.sh: charon not installed at $charon_bin" >&2
    echo "  run: scripts/install-charon.sh" >&2
    exit 1
fi

LLBC_DEST="${LLBC_DEST:-$repo_root/build/llbc}"
mkdir -p "$LLBC_DEST"

# bash 3.2 (macOS default) has no associative arrays — use a case
# statement instead.  `crate_info <name>` echoes "<path>|<cargo flags>"
# or empty if the name is unknown.
crate_info() {
    case "$1" in
        corpus)
            echo "$repo_root/majit/charon-spike/corpus|"
            ;;
        pyre-object)
            echo "$repo_root/pyre/pyre-object|"
            ;;
        pyre-module)
            # pyre-module re-exports pyre-interpreter's JIT backend feature.
            echo "$repo_root/pyre/pyre-module|--features pyre-interpreter/${CARGO_FEATURES:-cranelift}"
            ;;
        pyre-interpreter)
            echo "$repo_root/pyre/pyre-interpreter|--features ${CARGO_FEATURES:-cranelift}"
            ;;
        pyre-jit)
            # pyre-jit hosts PyreBlackholeAllocator + the Drop-impl
            # guards (JitSuppressionGuard / GuardCompilingScope /
            # TestJitParamsGuard) the AST extract audit lists as
            # uncovered.  Extracting it closes the residual gap so
            # extract_trait_impls / extract_inherent_impl_methods stop
            # emitting `graph: None` placeholders for those entries.
            echo "$repo_root/pyre/pyre-jit|--features ${CARGO_FEATURES:-cranelift}"
            ;;
        *)
            echo ""
            ;;
    esac
}

ALL_CRATES="corpus pyre-object pyre-module pyre-interpreter pyre-jit"

if [[ "$#" -eq 0 ]]; then
    targets="$ALL_CRATES"
else
    targets="$*"
fi

for crate in $targets; do
    info="$(crate_info "$crate")"
    if [[ -z "$info" ]]; then
        echo "extract-llbc.sh: unknown crate '$crate'" >&2
        echo "  known: $ALL_CRATES" >&2
        exit 1
    fi
    path="${info%%|*}"
    flags="${info#*|}"

    if [[ ! -d "$path" ]]; then
        echo "extract-llbc.sh: missing crate dir for '$crate' at $path" >&2
        exit 1
    fi

    dest="$LLBC_DEST/${crate}.ullbc"
    echo "=== extracting $crate -> $dest ==="

    pushd "$path" > /dev/null
    # `--ullbc` = basic-block CFG form (the analog of CPython bytecode);
    # `--dest-file` overrides the default `<crate>.{ull,ll}bc` placement.
    if [[ -n "$flags" ]]; then
        "$charon_bin" cargo --ullbc --dest-file "$dest" -- $flags
    else
        "$charon_bin" cargo --ullbc --dest-file "$dest"
    fi
    popd > /dev/null

    size="$(du -h "$dest" | cut -f1)"
    echo "    wrote $dest ($size)"
done

echo
echo "all extractions complete. artefacts under: $LLBC_DEST"
