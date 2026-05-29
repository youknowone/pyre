#!/usr/bin/env bash
# install-charon.sh — fetch the pinned Charon release into ./build/charon/
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
#   scripts/install-charon.sh                   # install to ./build/charon
#   CHARON_DEST=/usr/local/bin scripts/install-charon.sh
#   CHARON_VERSION=nightly-2026.05.20 scripts/install-charon.sh
#
# After install:
#   ./build/charon/charon version
#   ./build/charon/charon toolchain-path     # triggers nightly install
#
# Update procedure (when bumping the pin):
#   1. Edit CHARON_VERSION_DEFAULT below to the new tag.
#   2. Re-run this script with CHARON_DEST=./build/charon (delete old first).
#   3. Re-extract corpus: cd majit/charon-spike/corpus && \
#        ../../../build/charon/charon cargo --ullbc --dest-file ../corpus.ullbc
#   4. cd majit/charon-spike/prototype && ./compare.sh
#   5. If the diff is benign (schema-format only), regenerate the expected/
#      snapshots; otherwise debug.

set -euo pipefail

CHARON_VERSION_DEFAULT="nightly-2026.05.29"
CHARON_VERSION="${CHARON_VERSION:-$CHARON_VERSION_DEFAULT}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
CHARON_DEST="${CHARON_DEST:-$repo_root/build/charon}"

# Detect platform tag for the prebuilt asset.
uname_s="$(uname -s)"
uname_m="$(uname -m)"
case "$uname_s-$uname_m" in
    Darwin-arm64)  asset="charon-macos-aarch64.tar.gz" ;;
    Darwin-x86_64) asset="charon-macos-x86_64.tar.gz" ;;
    Linux-aarch64) asset="charon-linux-aarch64.tar.gz" ;;
    Linux-x86_64)  asset="charon-linux-aarch64.tar.gz" ;;  # only aarch64 published; fail clearly
    *)
        echo "install-charon.sh: unsupported platform $uname_s-$uname_m" >&2
        echo "  charon releases publish: darwin-aarch64, darwin-x86_64, linux-aarch64" >&2
        exit 1
        ;;
esac

if [[ "$uname_s-$uname_m" == "Linux-x86_64" ]]; then
    echo "install-charon.sh: WARNING — charon does not publish a linux-x86_64 binary" >&2
    echo "  Build from source: https://github.com/AeneasVerif/charon#installation" >&2
    exit 2
fi

# Skip re-download if the installed binary already matches the pinned
# version. `charon version` prints the cargo version (e.g. 0.1.196),
# not the nightly tag, so we cache the tag in a sidecar file.
stamp="$CHARON_DEST/.installed-version"
if [[ -x "$CHARON_DEST/charon" && -f "$stamp" ]]; then
    cur="$(cat "$stamp")"
    if [[ "$cur" == "$CHARON_VERSION" ]]; then
        echo "charon $CHARON_VERSION already installed at $CHARON_DEST"
        exit 0
    fi
    echo "charon at $CHARON_DEST is $cur; replacing with $CHARON_VERSION"
fi

mkdir -p "$CHARON_DEST"

url="https://github.com/AeneasVerif/charon/releases/download/$CHARON_VERSION/$asset"
echo "fetching $url"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fL --progress-bar -o "$tmp/$asset" "$url"

tar -C "$tmp" -xzf "$tmp/$asset"
# Charon archives contain `charon` + `charon-driver` at the archive root.
mv "$tmp/charon" "$tmp/charon-driver" "$CHARON_DEST/"
echo "$CHARON_VERSION" > "$stamp"

echo
echo "installed: $CHARON_DEST/charon"
"$CHARON_DEST/charon" version || true
echo
echo "next: trigger the rustc nightly install (one-time, ~1 minute):"
echo "  $CHARON_DEST/charon toolchain-path"
