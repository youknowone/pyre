#!/usr/bin/env bash
# extract-llbc.sh — run Charon on the JIT-consumed crates and drop
# `.ullbc` artefacts into ./build/llbc/.
#
# The Charon-extracted-MIR front-end lowers MIR-derived IR (`.ullbc`)
# into pyre's `FunctionGraph`. This script is the producer of those
# `.ullbc` files; the consumer is `majit-charon-reader`.
#
# Usage:
#   scripts/extract-llbc.sh                  # extract all JIT-consumed crates
#   scripts/extract-llbc.sh pyre-object      # extract one crate
#   scripts/extract-llbc.sh corpus           # extract the Charon fixture corpus
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
source "$repo_root/scripts/charon-msvc-env.sh"
charon_prepend_msvc_link
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) charon_exe="charon.exe" ;;
    *)                    charon_exe="charon" ;;
esac
charon_bin="$repo_root/build/charon/$charon_exe"

if [[ ! -x "$charon_bin" ]]; then
    echo "extract-llbc.sh: charon not installed at $charon_bin" >&2
    echo "  run: scripts/install-charon.sh" >&2
    exit 1
fi

LLBC_DEST="${LLBC_DEST:-$repo_root/build/llbc}"
case "$LLBC_DEST" in
    /*) ;;
    *) LLBC_DEST="$repo_root/$LLBC_DEST" ;;
esac
mkdir -p "$LLBC_DEST"

# bash 3.2 (macOS default) has no associative arrays — use a case
# statement instead.  `crate_info <name>` echoes "<path>|<cargo flags>"
# or empty if the name is unknown.
crate_info() {
    case "$1" in
        corpus)
            echo "$repo_root/majit/charon-corpus|"
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
            # TestJitParamsGuard), which have no JIT-traceable graph.
            # Extraction is supported for experimentation,
            # but the production build does NOT discover this artefact
            # (see `auto_discover_workspace_llbc_paths` in
            # majit-translate/src/lib.rs): those entries are semantically
            # residual runtime calls, so their `graph: None` placeholder
            # is the correct representation and the build stays
            # environment-invariant on the canonical pair regardless of
            # whether `pyre-jit.ullbc` is present.
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

# `cfg_select!` skew workaround.
#
# Charon's pinned release embeds rustc nightly-2026-02-22, where
# `core::cfg_select!` is still feature-gated (E0658).  rustpython's
# `host_env` crate (a transitive dep of pyre-interpreter / pyre-jit)
# calls `cfg_select!` without a `#![feature(cfg_select)]` gate because
# the macro is stable in the workspace's own toolchain (stable 1.95.0,
# 2026-04-14).  Under charon's older nightly the gate is missing, so the
# extraction `cargo build` fails to compile host_env.
#
# Inject `#![feature(cfg_select)]` into every crate compiled during
# extraction via `-Zcrate-attr` (needs `RUSTC_BOOTSTRAP=1` to allow `-Z`
# on the pinned nightly).  This affects ONLY the charon extraction build
# — never the production / stable build — and is a no-op for crates that
# don't use the macro.  Remove once the charon pin advances to a nightly
# where `cfg_select` is stable.
#
# Host/target graph split.  Charon always passes an explicit
# `--target <host-triple>` so it can instrument the target crates while
# leaving build scripts / proc-macros (the host graph) untouched.  Cargo
# applies `RUSTFLAGS` only to the TARGET graph, so the crate-attr reaches
# the target-side host_env but NOT the copy a build script drags into the
# HOST graph (built under `target/debug/deps`, no flag) — that one still
# fails E0658.  pyre-jit's dependency graph pulls host_env into the host
# graph; pyre-interpreter's does not, which is why only pyre-jit hits
# this.
# Inject the same crate-attr into the host graph via cargo's `[host]`
# rustflags table (`-Zhost-config`, which also requires
# `-Ztarget-applies-to-host` and `target-applies-to-host=false`).  Passed
# as `--config` CLI args + `CARGO_UNSTABLE_*` env so no `.cargo/config.toml`
# is written and the stable build stays untouched.
export RUSTC_BOOTSTRAP=1
charon_crate_attr="-Zcrate-attr=feature(cfg_select)"
if [[ -n "${RUSTFLAGS:-}" ]]; then
    export RUSTFLAGS="$RUSTFLAGS $charon_crate_attr"
else
    export RUSTFLAGS="$charon_crate_attr"
fi
export CARGO_UNSTABLE_HOST_CONFIG=true
export CARGO_UNSTABLE_TARGET_APPLIES_TO_HOST=true
charon_host_config=(
    --config target-applies-to-host=false
    --config "host.rustflags=[\"$charon_crate_attr\"]"
)

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
    # `charon_host_config` (the `[host]` rustflags injection above) is
    # forwarded to the inner `cargo build` after `--`.
    if [[ -n "$flags" ]]; then
        "$charon_bin" cargo --ullbc --dest-file "$dest" -- $flags "${charon_host_config[@]}"
    else
        "$charon_bin" cargo --ullbc --dest-file "$dest" -- "${charon_host_config[@]}"
    fi
    popd > /dev/null

    size="$(du -h "$dest" | cut -f1)"
    echo "    wrote $dest ($size)"
done

echo
echo "all extractions complete. artefacts under: $LLBC_DEST"
