#!/usr/bin/env python3
"""pyre driver for the Charon ULLBC extraction engine.

Declares the pyre crate table and delegates to the neutral engine in
`<repo-root>/scripts/llbc_extract.py`. Artefacts land under
`<repo-root>/build/llbc`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from llbc_extract import CrateSpec, run_cli  # noqa: E402


# `pyre-object` / `pyre-interpreter` exclude `majit-translate` from their
# fingerprint: their `.ullbc` holds zero references to it, so a pure edit there
# reuses the cached artefact. `pyre-jit` is deliberately ABSENT — its `.ullbc`
# embeds majit-translate *type* layouts (translator::rtyper::lltypesystem,
# model, flowspace::model, codewriter, tool::algo, …), so a type change there
# must re-extract it.
SPECS: dict[str, CrateSpec] = {
    # `corpus` lives outside the crate graph the metadata walk sees, so its
    # fingerprint inputs are explicit pathspecs.
    "corpus": CrateSpec(
        name="corpus",
        crate_dir=ROOT / "majit" / "charon-corpus",
        output_name="corpus.ullbc",
        fingerprint_pathspecs=[
            "majit/charon-corpus/Cargo.toml",
            "majit/charon-corpus/src/",
        ],
        # A reader fixture, not a build input: nothing consumes its layouts
        # for a cross target.
        layout_targets=(),
    ),
    "pyre-object": CrateSpec(
        name="pyre-object",
        crate_dir=ROOT / "pyre" / "pyre-object",
        output_name="pyre-object.ullbc",
        excluded_deps={"majit-translate"},
    ),
    "pyre-module": CrateSpec(
        name="pyre-module",
        crate_dir=ROOT / "pyre" / "pyre-module",
        output_name="pyre-module.ullbc",
        cargo_args=["--features", "pyre-interpreter/{features}"],
    ),
    "pyre-interpreter": CrateSpec(
        name="pyre-interpreter",
        crate_dir=ROOT / "pyre" / "pyre-interpreter",
        output_name="pyre-interpreter.ullbc",
        cargo_args=["--features", "{features}"],
        excluded_deps={"majit-translate"},
    ),
    "pyre-jit": CrateSpec(
        name="pyre-jit",
        crate_dir=ROOT / "pyre" / "pyre-jit",
        output_name="pyre-jit.ullbc",
        cargo_args=["--features", "{features}"],
        # No layout sidecar. A cross-target pass has to pass cargo
        # `--target`, and cargo then stops applying `RUSTFLAGS` to host
        # units — including `pyre-jit-trace`'s build script, which
        # build-depends on `pyre-interpreter` and so drags in
        # `rustpython-host_env`. That crate uses `cfg_select!`, still
        # unstable on Charon's pinned nightly, and the
        # `-Zcrate-attr=feature(cfg_select)` that enables it can only reach
        # host units through `-Zhost-config`, which cargo panics on when
        # `--target` is set. The stock build never hits this: its toolchain
        # has `cfg_select` stable.
        #
        # The gap this leaves is the 588 layout-carrying types declared
        # only here — `jit::{flow,codewriter,flatten,regalloc}`,
        # `majit_*`, and closure environments, i.e. the compiler's own
        # data structures rather than the object model traced bytecode
        # reads. Every runtime type reached through a descr comes from
        # pyre-object or pyre-interpreter, both of which do get sidecars.
        layout_targets=(),
    ),
}

DEFAULT_CRATES = ["pyre-object", "pyre-interpreter", "pyre-jit"]

# Targets, besides the extraction host, that get a layout sidecar. The
# wasm32 build reads the same `build/llbc` set as the native build, and its
# pointers are 4 bytes wide: without its own field offsets every descr field
# past the first pointer names the wrong bytes.
LAYOUT_TARGETS = ("wasm32-unknown-unknown",)

# The wasm32 compiler pass needs the same `getrandom` backend selection the
# wasm build itself uses (`check.py`'s `WASM_RUSTFLAGS`) — the default
# backend refuses to build for `wasm32-unknown-unknown`.
LAYOUT_TARGET_RUSTFLAGS = '--cfg getrandom_backend="custom"'

# Base fingerprint inputs shared by every pyre crate: workspace manifests plus
# the extraction tooling itself, so an edit to the engine/driver busts caches.
BASE_PATHSPECS = [
    "Cargo.lock",
    "Cargo.toml",
    "scripts/llbc_extract.py",
    "scripts/extract-llbc.py",
    "pyre/scripts/extract-llbc.py",
    "scripts/install-charon.py",
]


def main() -> None:
    run_cli(
        SPECS,
        DEFAULT_CRATES,
        root=ROOT,
        out_dir=ROOT / "build" / "llbc",
        base_pathspecs=BASE_PATHSPECS,
        metadata_feature_crates=("pyre-interpreter", "pyre-jit"),
        layout_targets=LAYOUT_TARGETS,
        layout_target_rustflags=LAYOUT_TARGET_RUSTFLAGS,
    )


if __name__ == "__main__":
    main()
