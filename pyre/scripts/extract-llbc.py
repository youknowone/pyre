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
    ),
}

DEFAULT_CRATES = ["pyre-object", "pyre-interpreter", "pyre-jit"]

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
    )


if __name__ == "__main__":
    main()
