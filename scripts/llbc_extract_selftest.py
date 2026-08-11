#!/usr/bin/env python3
"""Check `include_closure` against every `include*!` argument spelling.

`fingerprint_inputs` decides which bytes the LLBC staleness stamp — and the CI
cache key built from it — covers. An argument spelling that walks off the end
of that set does not fail anything: the artefact simply reads as fresh while
the crate compiles something else. Nothing downstream can notice, so the check
has to live here.

Runs on a synthetic tree in a temp dir: no cargo, no git, well under a second.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent


def load_engine():
    """Import the engine by path; it is a library the drivers import, not a package."""
    spec = importlib.util.spec_from_file_location(
        "llbc_extract", ROOT / "llbc_extract.py"
    )
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves annotations through `sys.modules[cls.__module__]`.
    sys.modules["llbc_extract"] = module
    spec.loader.exec_module(module)
    return module


def run(engine) -> int:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    outside = pathlib.Path(tempfile.mkdtemp()).resolve()
    try:
        (root / "src").mkdir()
        (root / "data").mkdir()
        (root / "data" / "rules.txt").write_text("rules")
        (outside / "far.txt").write_text("far")

        # FOLD: the file joins the digest. PASS: nothing to add, and nothing
        # unresolved was passed over. REFUSE: extraction stops.
        cases = [
            ("literal outside the pathspecs", 'include_str!("../data/rules.txt");', "FOLD data/rules.txt"),
            ("literal already enumerated", 'include_str!("lib.rs");', "PASS"),
            ("nonexistent path", 'include_str!("../nope.txt");', "PASS"),
            ("concat!(env!(OUT_DIR))", 'include!(concat!(env!("OUT_DIR"), "/g.rs"));', "PASS"),
            ("macro_rules! parameter", "include_str!($appfile);", "PASS"),
            ("prose about the macro", "// how include!() behaves", "PASS"),
            ("raw literal", 'include_str!(r"../data/rules.txt");', "REFUSE"),
            ("literal carrying an escape", 'include_str!("../data\\\\rules.txt");', "REFUSE"),
            ("path through a constant", "include_str!(RULES_PATH);", "REFUSE"),
        ]

        # One `..` per component of `root/src` below the filesystem anchor.
        chain = "../" * len(root.parts) + (outside / "far.txt").relative_to(
            outside.anchor
        ).as_posix()
        cases.append(("`..` chain leaving the repo", f'include_str!("{chain}");', "REFUSE"))

        try:
            (root / "src" / "link.txt").symlink_to(outside / "far.txt")
            cases.append(("symlink leaving the repo", 'include_str!("link.txt");', "REFUSE"))
        except OSError as exc:
            # Windows grants symlink creation only under Developer Mode.
            print(f"skip: symlink case ({exc})")

        failures = 0
        for name, body, expected in cases:
            (root / "src" / "lib.rs").write_text(body + "\n")
            try:
                extra = engine.include_closure(root, {pathlib.Path("src/lib.rs")})
                sorted_extra = sorted(path.as_posix() for path in extra)
                actual = "FOLD " + ",".join(sorted_extra) if extra else "PASS"
            except SystemExit:
                actual = "REFUSE"
            if actual == expected:
                print(f"ok   {name}: {actual}")
            else:
                failures += 1
                print(f"FAIL {name}: expected {expected}, got {actual}")
        return failures
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)


def run_fingerprint(engine) -> int:
    """Exercise the complete source key against a synthetic Git working tree."""
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    try:
        (root / "src").mkdir()
        (root / "scripts").mkdir()
        (root / "Cargo.toml").write_text("[package]\nname='probe'\nversion='0.0.0'\n")
        (root / ".gitignore").write_text("src/ignored.rs\n")
        (root / "src" / "lib.rs").write_text("pub fn answer() -> i64 { 42 }\n")
        tool = root / "scripts" / "tool.py"
        tool.write_text("# implementation detail\n")
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "add", ".gitignore", "Cargo.toml", "src/lib.rs", "scripts/tool.py"],
            cwd=root,
            check=True,
        )

        spec = engine.CrateSpec(
            name="probe",
            crate_dir=root,
            output_name="probe.ullbc",
            fingerprint_pathspecs=["src/"],
            layout_targets=(),
        )

        def make_engine(*, extraction_abi="probe-1", rustflags=""):
            return engine.Engine(
                specs={"probe": spec},
                default_crates=["probe"],
                root=root,
                out_dir=root / "out",
                base_pathspecs=["Cargo.toml"],
                charon_root=root,
                cargo_features="",
                extraction_abi=extraction_abi,
                layout_target_rustflags=rustflags,
            )

        def fingerprint(eng):
            return engine.source_fingerprint(eng, ["probe"], "")

        failures = 0

        def expect(name, condition):
            nonlocal failures
            if condition:
                print(f"ok   {name}")
            else:
                failures += 1
                print(f"FAIL {name}")

        eng = make_engine()
        base = fingerprint(eng)

        tool.write_text("# diagnostics and comments changed\n")
        expect("tool implementation is outside the source key", fingerprint(eng) == base)

        untracked = root / "src" / "new.rs"
        untracked.write_text("pub fn new_item() {}\n")
        before_stage = fingerprint(eng)
        expect("untracked source moves the source key", before_stage != base)
        subprocess.run(["git", "add", "src/new.rs"], cwd=root, check=True)
        expect("staging identical bytes preserves the source key", fingerprint(eng) == before_stage)

        ignored = root / "src" / "ignored.rs"
        ignored.write_text("build output\n")
        expect("ignored file stays outside the source key", fingerprint(eng) == before_stage)

        expect(
            "extraction ABI moves the cache and staleness key",
            fingerprint(make_engine(extraction_abi="probe-2")) != before_stage,
        )
        expect(
            "effective layout configuration moves the key",
            fingerprint(make_engine(rustflags="--cfg changed")) != before_stage,
        )
        return failures
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    engine = load_engine()
    failures = run(engine) + run_fingerprint(engine)
    if failures:
        raise SystemExit(f"llbc_extract_selftest: {failures} failed")
    print("llbc_extract_selftest: all passed")


if __name__ == "__main__":
    main()
