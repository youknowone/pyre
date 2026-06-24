#!/usr/bin/env python3
"""Report RPython/PyPy module-name parity gaps in the Rust port.

This is an audit helper for actionable module-name gaps.  It normalizes package
entry points (`__init__.py` in Python, `mod.rs`/`lib.rs` in Rust) so the report
focuses on real module names rather than language-specific filesystem
conventions.  Pyre-local Rust boundaries and permanently-unused PyPy layers
are reported separately as ignored entries, with reasons, so they do not drive
blind ports of code pyre will not use.

With `--symbols`, the helper also compares top-level Python class names with
top-level Rust public type names, and top-level Python function names with
top-level Rust public function names, for already-matched modules.  Thin Rust
reexport wrappers are classified separately so shared implementation crates
such as `majit_ir` and `majit_trace` do not turn into false positives.
"""

from __future__ import annotations

import argparse
import json
import re
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
        Path("majit/majit-translate/src/codewriter"),
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
PACKAGE_ENTRY = "mod"

INTENTIONAL_MISSING: dict[str, dict[str, str]] = {
    "rpython/rtyper/lltypesystem": {
        "ll2ctypes": "permanently unused: pyre never simulates lltype programs through ctypes",
        "llarena": "permanently unused: pyre does not port RPython moving-GC arena simulation",
    },
    "rpython/rtyper/tool": {
        "rffi_platform": "permanently unused: pyre uses Rust/Charon layouts instead of C probing",
    },
    "rpython/translator": {
        "c": "permanently unused: pyre must not grow a local translator/c backend tree",
        "exceptiontransform": "represented in Rust Result/? lowering, not a standalone module",
    },
}

INTENTIONAL_EXTRA: dict[str, dict[str, str]] = {
    "rpython/jit/codewriter": {
        "annotation_state": "local Rust boundary for temporary ValueType/SomeValue projection",
        "insns": "local stable byte table derived from assembler.py's dynamic insns table",
        "jtransform_opname": "local transducer for rtyped helper graphs into jtransform shape",
        "jtransform_shadow": "env-gated diagnostic, never production path",
        "transform_profile": "env-gated drain profiler with no upstream runtime effect",
        "type_state": "local concretetype projection boundary during rtyper cutover",
    },
    "rpython/jit/metainterp": {
        "call_descr": "runtime call-descr boundary for codewriter/backend descriptor surfaces",
        "io_buffer": "compiled-loop stdout buffer; RPython interpreter writes directly",
        "jit": "runtime half of rpython/rlib/jit.py; translator half lives under rlib",
        "jit_state": "Rust trait abstraction for interpreter state",
        "jitcode": "runtime ABI boundary around canonical translate-side jitcode.py port",
        "parity": "test-only trace comparison utilities",
        "recorder": "runtime Trace boundary around opencoder/history recording roles",
        "trace_ctx": "Rust tracing context split across history/compile roles",
    },
    "rpython/rtyper": {
        "cutover": "transitional bridge between legacy and orthodox graph paths",
        "flowspace_adapter": "transitional bridge from pyre graph model to flowspace graph model",
        "legacy_annotator": "temporary legacy graph adapter for cutover",
        "legacy_resolve": "temporary legacy call resolution adapter for cutover",
        "pairtype": "Rust carrier for rtyper-side __extend__(pairtype(...)) blocks",
        "pyre_call_registry": "symbolic FunctionPath registration in place of CPython callable identity",
        "unit_variant_fold": "Rust unit-variant PBC pre-folding before jtransform",
    },
    "rpython/translator": {
        "backend": "intentional non-c module for minimal CBuilder-shaped driver shells",
        "rtyper": "crate-local nesting; upstream rtyper remains compared separately",
        "targetspec": "typed carrier for driver.py from_targetspec's open Python dict",
    },
}


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


def python_module_path(path: Path, module: str) -> Path:
    if module == PACKAGE_ENTRY:
        return path / "__init__.py"
    file_path = path / f"{module}.py"
    if file_path.is_file():
        return file_path
    return path / module / "__init__.py"


def rust_module_path(path: Path, module: str) -> Path:
    if module == PACKAGE_ENTRY:
        lib_path = path / "lib.rs"
        if lib_path.is_file():
            return lib_path
        return path / "mod.rs"
    file_path = path / f"{module}.rs"
    if file_path.is_file():
        return file_path
    return path / module / "mod.rs"


PYTHON_TOP_LEVEL_SYMBOL = re.compile(r"^(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b")


def python_top_level_symbols(path: Path) -> dict[str, set[str]]:
    symbols = {"types": set(), "functions": set()}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PYTHON_TOP_LEVEL_SYMBOL.match(line)
        if not match:
            continue
        name = match.group(1)
        if name.startswith("_"):
            continue
        if line.startswith("class "):
            symbols["types"].add(name)
        else:
            symbols["functions"].add(name)
    return symbols


RUST_PUB_ITEM = re.compile(
    r"^pub\s+(?:unsafe\s+)?(?:extern\s+(?:\"[^\"]+\"\s+)?)?"
    r"(struct|enum|trait|type|fn)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
RUST_PUB_REEXPORT = re.compile(r"^pub\s+use\s+")
RUST_ITEM_START = re.compile(
    r"^(?:pub\s+)?(?:unsafe\s+)?(?:extern\s+(?:\"[^\"]+\"\s+)?)?"
    r"(struct|enum|trait|type|fn|const|static|impl|mod)\b"
)


def _strip_rust_line(line: str) -> str:
    line = line.strip()
    if line.startswith("//"):
        return ""
    if line.startswith("#["):
        return ""
    return line


def rust_top_level_symbols(path: Path) -> tuple[dict[str, set[str]], bool]:
    symbols = {"types": set(), "functions": set()}
    has_pub_reexport = False
    has_direct_item = False
    depth = 0
    in_block_comment = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line
        if in_block_comment:
            if "*/" in line:
                line = line.split("*/", 1)[1]
                in_block_comment = False
            else:
                continue
        while "/*" in line:
            before, after = line.split("/*", 1)
            if "*/" in after:
                after = after.split("*/", 1)[1]
                line = before + after
            else:
                line = before
                in_block_comment = True
                break

        candidate = _strip_rust_line(line)
        if depth == 0 and candidate:
            pub_match = RUST_PUB_ITEM.match(candidate)
            if pub_match:
                kind = pub_match.group(1)
                bucket = "functions" if kind == "fn" else "types"
                symbols[bucket].add(pub_match.group(2))
                has_direct_item = True
            elif RUST_PUB_REEXPORT.match(candidate):
                has_pub_reexport = True
            elif RUST_ITEM_START.match(candidate):
                has_direct_item = True

        depth += line.count("{") - line.count("}")
        if depth < 0:
            depth = 0

    return symbols, has_pub_reexport and not has_direct_item


def compare_symbols_for_pair(
    root: Path, pair: ModulePair, matched: list[str]
) -> list[dict[str, object]]:
    python_dir = root / pair.python_dir
    rust_dir = root / pair.rust_dir
    results = []

    for module in matched:
        if module == PACKAGE_ENTRY:
            continue
        py_path = python_module_path(python_dir, module)
        rs_path = rust_module_path(rust_dir, module)
        if not py_path.is_file() or not rs_path.is_file():
            continue

        py_symbols = python_top_level_symbols(py_path)
        rs_symbols, is_reexport = rust_top_level_symbols(rs_path)
        result = {
            "module": module,
            "python_path": py_path.relative_to(root).as_posix(),
            "rust_path": rs_path.relative_to(root).as_posix(),
            "types": {
                "matched": sorted(py_symbols["types"] & rs_symbols["types"]),
                "missing": sorted(py_symbols["types"] - rs_symbols["types"]),
                "extra": sorted(rs_symbols["types"] - py_symbols["types"]),
            },
            "functions": {
                "matched": sorted(py_symbols["functions"] & rs_symbols["functions"]),
                "missing": sorted(py_symbols["functions"] - rs_symbols["functions"]),
                "extra": sorted(rs_symbols["functions"] - py_symbols["functions"]),
            },
            "skipped_reexport": is_reexport,
        }
        results.append(result)
    return results


def compare_pair(root: Path, pair: ModulePair, excludes: set[str]) -> dict[str, object]:
    python_dir = root / pair.python_dir
    rust_dir = root / pair.rust_dir
    if not python_dir.is_dir():
        raise SystemExit(f"missing Python directory: {pair.python_dir}")
    if not rust_dir.is_dir():
        raise SystemExit(f"missing Rust directory: {pair.rust_dir}")

    py_modules = python_modules(python_dir, excludes)
    rs_modules = rust_modules(rust_dir, excludes)
    raw_missing = py_modules - rs_modules
    raw_extra = rs_modules - py_modules
    ignored_missing = {
        name: reason
        for name, reason in INTENTIONAL_MISSING.get(pair.label, {}).items()
        if name in raw_missing
    }
    ignored_extra = {
        name: reason
        for name, reason in INTENTIONAL_EXTRA.get(pair.label, {}).items()
        if name in raw_extra
    }
    missing = sorted(raw_missing - ignored_missing.keys())
    extra = sorted(raw_extra - ignored_extra.keys())
    matched = sorted(py_modules & rs_modules)
    return {
        "label": pair.label,
        "python_dir": pair.python_dir.as_posix(),
        "rust_dir": pair.rust_dir.as_posix(),
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "ignored_missing": dict(sorted(ignored_missing.items())),
        "ignored_extra": dict(sorted(ignored_extra.items())),
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
    parser.add_argument(
        "--symbols",
        action="store_true",
        help="also compare top-level class/function names with Rust pub item names",
    )
    parser.add_argument(
        "--strict-symbols",
        action="store_true",
        help="exit non-zero when --symbols finds any non-reexport symbol gap",
    )
    return parser.parse_args(argv)


def print_text(results: list[dict[str, object]], show_symbols: bool) -> None:
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
        ignored_missing = result["ignored_missing"]
        ignored_extra = result["ignored_extra"]
        if ignored_missing:
            print(
                "ignored missing: "
                + "; ".join(f"{name} ({reason})" for name, reason in ignored_missing.items())
            )
        if ignored_extra:
            print(
                "ignored extra: "
                + "; ".join(f"{name} ({reason})" for name, reason in ignored_extra.items())
            )
        if show_symbols:
            symbol_results = result["symbols"]
            symbol_gaps = [
                item
                for item in symbol_results
                if item["types"]["missing"]
                or item["types"]["extra"]
                or item["functions"]["missing"]
                or item["functions"]["extra"]
                or item["skipped_reexport"]
            ]
            if not symbol_gaps:
                print("symbols: <none>")
            else:
                print("symbols:")
                for item in symbol_gaps:
                    if item["skipped_reexport"]:
                        print(
                            f"  {item['module']}: skipped reexport wrapper "
                            f"({item['rust_path']})"
                        )
                    else:
                        details = []
                        if item["types"]["missing"]:
                            details.append(
                                "missing types " + ", ".join(item["types"]["missing"])
                            )
                        if item["types"]["extra"]:
                            details.append(
                                "extra types " + ", ".join(item["types"]["extra"])
                            )
                        if item["functions"]["missing"]:
                            details.append(
                                "missing functions "
                                + ", ".join(item["functions"]["missing"])
                            )
                        if item["functions"]["extra"]:
                            details.append(
                                "extra functions "
                                + ", ".join(item["functions"]["extra"])
                            )
                        print(f"  {item['module']}: " + "; ".join(details))
        print()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = repo_root()
    excludes = set(DEFAULT_EXCLUDES)
    if args.include_tests:
        excludes.discard("test")

    results = [compare_pair(root, pair, excludes) for pair in DEFAULT_PAIRS]
    if args.symbols:
        for pair, result in zip(DEFAULT_PAIRS, results):
            result["symbols"] = compare_symbols_for_pair(root, pair, result["matched"])
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print_text(results, args.symbols)

    has_gap = any(result["missing"] or result["extra"] for result in results)
    has_symbol_gap = False
    if args.symbols:
        has_symbol_gap = any(
            (
                item["types"]["missing"]
                or item["types"]["extra"]
                or item["functions"]["missing"]
                or item["functions"]["extra"]
            )
            and not item["skipped_reexport"]
            for result in results
            for item in result["symbols"]
        )
    if args.strict and has_gap:
        return 1
    if args.strict_symbols and has_symbol_gap:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
