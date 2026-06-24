#!/usr/bin/env python3
"""Report RPython/PyPy module-name parity gaps in the Rust port.

This is an audit helper, not a waiver list.  It normalizes package entry
points (`__init__.py` in Python, `mod.rs`/`lib.rs` in Rust) so the report
focuses on real module names rather than language-specific filesystem
conventions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModulePair:
    label: str
    python_dir: Path
    rust_dir: Path


DEFAULT_PAIRS = [
    ModulePair(
        "rpython/annotator",
        Path("rpython/annotator"),
        Path("majit/majit-translate/src/annotator"),
    ),
    ModulePair(
        "rpython/config",
        Path("rpython/config"),
        Path("majit/majit-translate/src/config"),
    ),
    ModulePair(
        "rpython/flowspace",
        Path("rpython/flowspace"),
        Path("majit/majit-translate/src/flowspace"),
    ),
    ModulePair(
        "rpython/jit/codewriter",
        Path("rpython/jit/codewriter"),
        Path("majit/majit-translate/src/jit_codewriter"),
    ),
    ModulePair(
        "rpython/jit/metainterp",
        Path("rpython/jit/metainterp"),
        Path("majit/majit-metainterp/src"),
    ),
    ModulePair(
        "rpython/jit/metainterp/ruleopt",
        Path("rpython/jit/metainterp/ruleopt"),
        Path("majit/majit-metainterp/src/ruleopt"),
    ),
    ModulePair(
        "rpython/jit/metainterp/optimizeopt",
        Path("rpython/jit/metainterp/optimizeopt"),
        Path("majit/majit-metainterp/src/optimizeopt"),
    ),
    ModulePair(
        "rpython/rtyper",
        Path("rpython/rtyper"),
        Path("majit/majit-translate/src/translator/rtyper"),
    ),
    ModulePair(
        "rpython/rtyper/lltypesystem",
        Path("rpython/rtyper/lltypesystem"),
        Path("majit/majit-translate/src/translator/rtyper/lltypesystem"),
    ),
    ModulePair(
        "rpython/rtyper/lltypesystem/module",
        Path("rpython/rtyper/lltypesystem/module"),
        Path("majit/majit-translate/src/translator/rtyper/lltypesystem/module"),
    ),
    ModulePair(
        "rpython/rtyper/tool",
        Path("rpython/rtyper/tool"),
        Path("majit/majit-translate/src/translator/rtyper/tool"),
    ),
    ModulePair(
        "rpython/tool/algo",
        Path("rpython/tool/algo"),
        Path("majit/majit-translate/src/tool/algo"),
    ),
    ModulePair(
        "rpython/translator",
        Path("rpython/translator"),
        Path("majit/majit-translate/src/translator"),
    ),
]

DEFAULT_EXCLUDES = {"test", "__pycache__"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def python_modules(path: Path, excludes: set[str]) -> set[str]:
    modules = set()
    for child in path.iterdir():
        if child.name in excludes:
            continue
        if child.is_file() and child.suffix == ".py":
            modules.add("mod" if child.stem == "__init__" else child.stem)
        elif child.is_dir() and (child / "__init__.py").is_file():
            modules.add(child.name)
    return modules


def rust_file_module_name(child: Path) -> str:
    if child.stem == "lib":
        return "mod"
    return child.stem


def rust_modules(path: Path, excludes: set[str]) -> set[str]:
    modules = set()
    for child in path.iterdir():
        if child.name in excludes:
            continue
        if child.is_file() and child.suffix == ".rs":
            modules.add(rust_file_module_name(child))
        elif child.is_dir() and (child / "mod.rs").is_file():
            modules.add(child.name)
    return modules


def compare_pair(root: Path, pair: ModulePair, excludes: set[str]) -> dict[str, object]:
    python_dir = root / pair.python_dir
    rust_dir = root / pair.rust_dir
    if not python_dir.is_dir():
        raise SystemExit(f"missing Python directory: {pair.python_dir}")
    if not rust_dir.is_dir():
        raise SystemExit(f"missing Rust directory: {pair.rust_dir}")

    py_modules = python_modules(python_dir, excludes)
    rs_modules = rust_modules(rust_dir, excludes)
    missing = sorted(py_modules - rs_modules)
    extra = sorted(rs_modules - py_modules)
    matched = sorted(py_modules & rs_modules)
    return {
        "label": pair.label,
        "python_dir": pair.python_dir.as_posix(),
        "rust_dir": pair.rust_dir.as_posix(),
        "matched": matched,
        "missing": missing,
        "extra": extra,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare immediate RPython/PyPy module names with their Rust "
            "port directories."
        )
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="include Python test packages in module comparison",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of text",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any missing or extra module is found",
    )
    return parser.parse_args(argv)


def print_text(results: list[dict[str, object]]) -> None:
    for result in results:
        print(f"## {result['label']} -> {result['rust_dir']}")
        missing = result["missing"]
        extra = result["extra"]
        if missing:
            print("missing: " + ", ".join(missing))
        else:
            print("missing: <none>")
        if extra:
            print("extra: " + ", ".join(extra))
        else:
            print("extra: <none>")
        print()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = repo_root()
    excludes = set(DEFAULT_EXCLUDES)
    if args.include_tests:
        excludes.discard("test")

    results = [compare_pair(root, pair, excludes) for pair in DEFAULT_PAIRS]
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print_text(results)

    has_gap = any(result["missing"] or result["extra"] for result in results)
    return 1 if args.strict and has_gap else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
