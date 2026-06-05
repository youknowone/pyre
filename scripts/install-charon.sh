#!/usr/bin/env bash
# install-charon.sh — fetch the pinned Charon release into a shared cache.
#
# Charon is the rustc-driver tool that extracts ULLBC (MIR-derived IR)
# from a Rust crate. The Charon migration described in issue #97 uses
# it as the front-end for the JIT lowering pipeline.
#
# Pin policy:
#   - Charon releases are nightly-only ("prerelease: true" on every tag).
#     We pick a specific date and update it deliberately.
#   - The release binary embeds the rustc nightly date it was built
#     against; rustup auto-installs that toolchain on first run.
#   - Downstream stable-Rust consumers never touch the pinned nightly.
#
# Usage:
#   scripts/install-charon.sh                   # install to ../.pyre-build/charon/<platform>
#   PYRE_SHARED_BUILD=/tmp/pyre-build scripts/install-charon.sh
#   CHARON_DEST=/usr/local/bin scripts/install-charon.sh
#   CHARON_VERSION=nightly-2026.05.20 scripts/install-charon.sh
#
# After install:
#   ../.pyre-build/charon/<platform>/charon version
#   ../.pyre-build/charon/<platform>/charon toolchain-path
#
# Update procedure (when bumping the pin):
#   1. Edit CHARON_VERSION_DEFAULT below to the new tag.
#   2. Re-run this script (delete the old shared cache entry first if needed).
#   3. Re-extract corpus: cd majit/charon-corpus && \
#        ../../scripts/extract-llbc.sh corpus
#   4. Run the reader/frontend regression tests that consume corpus.ullbc.
#   5. If the diff is benign (schema-format only), update the corpus notes;
#      otherwise debug the reader or MIR frontend.

set -euo pipefail

CHARON_VERSION_DEFAULT="nightly-2026.05.29"
CHARON_VERSION="${CHARON_VERSION:-$CHARON_VERSION_DEFAULT}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
repo_parent="$(dirname "$repo_root")"
PYRE_SHARED_BUILD="${PYRE_SHARED_BUILD:-$repo_parent/.pyre-build}"
source "$repo_root/scripts/charon-msvc-env.sh"

# Detect platform. Charon publishes prebuilt assets for Linux/macOS only;
# on Windows there is no release asset, so we build from source at the
# pinned tag. Set CHARON_FROM_SOURCE=1 to force a source build elsewhere.
uname_s="$(uname -s)"
uname_m="$(uname -m)"
exe=""
from_source="${CHARON_FROM_SOURCE:-0}"
case "$uname_s-$uname_m" in
    Darwin-arm64)  asset="charon-macos-aarch64.tar.gz"; platform_key="darwin-arm64" ;;
    Darwin-x86_64) asset="charon-macos-x86_64.tar.gz";  platform_key="darwin-x86_64" ;;
    Linux-aarch64) asset="charon-linux-aarch64.tar.gz";  platform_key="linux-aarch64" ;;
    Linux-x86_64)  asset="charon-linux-x86_64.tar.gz";   platform_key="linux-x86_64" ;;
    MINGW*|MSYS*|CYGWIN*)
        from_source=1
        exe=".exe"
        platform_key="windows"
        ;;
    *)
        echo "install-charon.sh: unsupported platform $uname_s-$uname_m" >&2
        echo "  prebuilt: darwin-aarch64, darwin-x86_64, linux-aarch64, linux-x86_64" >&2
        echo "  set CHARON_FROM_SOURCE=1 to build from source instead" >&2
        exit 1
        ;;
esac

CHARON_DEST="${CHARON_DEST:-$PYRE_SHARED_BUILD/charon/$platform_key}"

# Skip re-install if the installed binary already matches the pinned
# version. `charon version` prints the cargo version (e.g. 0.1.196),
# not the nightly tag, so we cache the tag in a sidecar file.
stamp="$CHARON_DEST/.installed-version"
if [[ -x "$CHARON_DEST/charon$exe" && -f "$stamp" ]]; then
    cur="$(cat "$stamp")"
    if [[ "$cur" == "$CHARON_VERSION" ]]; then
        echo "charon $CHARON_VERSION already installed at $CHARON_DEST"
        exit 0
    fi
    echo "charon at $CHARON_DEST is $cur; replacing with $CHARON_VERSION"
fi

mkdir -p "$CHARON_DEST"

if [[ "$from_source" == 1 ]]; then
    # Build charon from source at the pinned tag. Charon's rust-toolchain
    # pins the nightly it needs; rustup auto-installs it on the first build.
    charon_src="${CHARON_SRC:-$PYRE_SHARED_BUILD/charon-src/$platform_key}"
    if [[ -d "$charon_src/.git" ]]; then
        echo "updating $charon_src to $CHARON_VERSION"
        git -C "$charon_src" fetch --depth 1 origin "refs/tags/$CHARON_VERSION:refs/tags/$CHARON_VERSION"
        git -C "$charon_src" checkout -q "$CHARON_VERSION"
    else
        echo "cloning charon $CHARON_VERSION into $charon_src"
        git clone --depth 1 --branch "$CHARON_VERSION" \
            https://github.com/AeneasVerif/charon.git "$charon_src"
    fi
    echo "building charon (cargo build --release; first run installs the pinned nightly)"
    charon_prepend_msvc_link
    ( cd "$charon_src/charon" && cargo build --release )
    cp "$charon_src/charon/target/release/charon$exe" \
       "$charon_src/charon/target/release/charon-driver$exe" "$CHARON_DEST/"
else
    url="https://github.com/AeneasVerif/charon/releases/download/$CHARON_VERSION/$asset"
    echo "fetching $url"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    curl -fL --progress-bar -o "$tmp/$asset" "$url"
    tar -C "$tmp" -xzf "$tmp/$asset"
    # Charon archives contain `charon` + `charon-driver` at the archive root.
    mv "$tmp/charon" "$tmp/charon-driver" "$CHARON_DEST/"
fi

echo "$CHARON_VERSION" > "$stamp"

echo
echo "installed: $CHARON_DEST/charon$exe"
"$CHARON_DEST/charon$exe" version || true
echo
echo "next: trigger the rustc nightly install (one-time, ~1 minute):"
echo "  $CHARON_DEST/charon$exe toolchain-path"
