#!/usr/bin/env python3
"""Reject pyre-owned identifiers and filenames in the majit subtree."""

from pathlib import Path
import os
import re
import sys


ROOT = Path(__file__).resolve().parent.parent
MAJIT = ROOT / "majit"
TOKEN = re.compile(r'//|/\*|r(#{0,255})"|"|[A-Za-z_][A-Za-z0-9_]*')


def is_runtime_owned(name: str) -> bool:
    # Match the project's conventional lower-, upper-, and CamelCase spellings
    # without misclassifying Python/RPython names such as `PyResult` and
    # `GetRpyReferentsFn`.
    return "pyre" in name or "Pyre" in name or "PYRE" in name


def code_identifiers(source: str):
    i = 0
    while i < len(source):
        match = TOKEN.search(source, i)
        if match is None:
            return
        token = match.group(0)
        i = match.end()
        if token == "//":
            newline = source.find("\n", i)
            i = len(source) if newline < 0 else newline + 1
        elif token == "/*":
            depth = 1
            while depth:
                opening = source.find("/*", i)
                closing = source.find("*/", i)
                if closing < 0:
                    return
                if 0 <= opening < closing:
                    depth += 1
                    i = opening + 2
                else:
                    depth -= 1
                    i = closing + 2
        elif token == '"':
            while True:
                closing = source.find('"', i)
                if closing < 0:
                    return
                escapes = 0
                cursor = closing - 1
                while cursor >= 0 and source[cursor] == "\\":
                    escapes += 1
                    cursor -= 1
                i = closing + 1
                if escapes % 2 == 0:
                    break
        elif token.startswith("r") and token.endswith('"'):
            closing = '"' + "#" * len(match.group(1))
            end = source.find(closing, i)
            if end < 0:
                return
            i = end + len(closing)
        else:
            yield token


def main() -> int:
    failures = []
    for directory, child_dirs, filenames in os.walk(MAJIT):
        child_dirs[:] = sorted(name for name in child_dirs if name not in {"target", ".git"})
        base = Path(directory)
        for filename in sorted(filenames):
            path = base / filename
            if any(is_runtime_owned(part) for part in path.relative_to(MAJIT).parts):
                failures.append(f"runtime-owned name in path: {path.relative_to(ROOT)}")
            if path.suffix != ".rs":
                continue
            for name in code_identifiers(path.read_text(encoding="utf-8")):
                if is_runtime_owned(name):
                    failures.append(
                        f"runtime-owned Rust identifier {name!r} in {path.relative_to(ROOT)}"
                    )
    if failures:
        print("majit boundary check failed:", file=sys.stderr)
        print("\n".join(f"  {failure}" for failure in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
