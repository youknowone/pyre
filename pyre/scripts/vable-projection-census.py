#!/usr/bin/env python3
"""Census of non-dereferenced virtualizable-field projections in the LLBC.

`FrameBox::new` takes its frame **by value**, so its virtualizable-field
projections read `(frame_1).pycode` rather than `((*self_1)).pycode`.  The
codewriter relies on that: `FieldDescriptor::base_is_local_aggregate` treats a
projection whose base is a non-dereferenced local aggregate as not reaching a
live virtualizable, which is what keeps the frame-construction stores off the
`getfield_vable_*` / `setfield_vable_*` path.

Changing the constructor to allocate first and initialise through a pointer
would turn every one of those projections into a deref and silently withdraw
the suppression — nothing fails at the point of the edit, and it surfaces much
later as a corpus-build abort.  This script is the tripwire for that.

It counts **projections**, both reads and writes.  A reads-only count passes
when the write side regresses, which reads as coverage while providing none:
an unsuppressed `setfield_vable_*` against a stack aggregate that carries no
`vable_token` is the more dangerous of the two directions.

Input is the Charon-extracted LLBC, i.e. the codewriter's *input* — no corpus
build is involved.  Usage:

    pyre/scripts/vable-projection-census.py build/llbc/*.ullbc

A pre-rendered `charon pretty-print` dump is accepted in place of a `.ullbc`,
which is what makes a negative control cheap to run.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VABLE_SPEC = ROOT / "pyre" / "pyre-jit-trace" / "src" / "virtualizable_spec.rs"

# The only function permitted to project a virtualizable field off a
# non-dereferenced base, and how many such projections it has.
EXPECTED = {"pyre_interpreter::pyframe::{FrameBox}::new": 12}

FN_RE = re.compile(r"^(?:pub )?fn ([^(<]+)")


def vable_fields() -> list[str]:
    """Field names from `virtualizable_spec.rs` — the single source of truth.

    Parsed rather than duplicated so that adding a virtualizable field cannot
    quietly fall outside the census.
    """
    text = VABLE_SPEC.read_text()
    names: list[str] = []
    for const in ("PYFRAME_VABLE_FIELDS", "PYFRAME_VABLE_ARRAYS"):
        m = re.search(const + r"[^=]*=\s*&\[(.*?)\];", text, re.S)
        if not m:
            raise SystemExit(f"{VABLE_SPEC}: cannot find {const}")
        names += re.findall(r'\("([A-Za-z_][A-Za-z0-9_]*)",', m.group(1))
    if not names:
        raise SystemExit(f"{VABLE_SPEC}: parsed no field names")
    return names


def charon_bin() -> Path:
    shared = Path(
        os.environ.get("PYRE_SHARED_BUILD", ROOT.parent / ".pyre-build")
    )
    machine = "aarch64" if platform.machine() in ("arm64", "aarch64") else "x86_64"
    key = {
        ("Darwin", "aarch64"): "darwin-arm64",
        ("Darwin", "x86_64"): "darwin-x86_64",
        ("Linux", "aarch64"): "linux-aarch64",
        ("Linux", "x86_64"): "linux-x86_64",
    }.get((platform.system(), machine))
    if key is None:
        raise SystemExit(f"unsupported platform {platform.system()}/{machine}")
    path = Path(os.environ.get("CHARON_DEST", shared / "charon" / key)) / "charon"
    if not path.exists():
        raise SystemExit(f"charon not installed at {path}\n  run: scripts/install-charon.py")
    return path


def render(path: Path) -> str:
    """Pretty-print a `.ullbc`; pass a text dump through unchanged."""
    with path.open("rb") as fh:
        if not fh.read(1).startswith(b"{"):
            return path.read_text(errors="replace")
    return subprocess.run(
        [str(charon_bin()), "pretty-print", str(path)],
        capture_output=True,
        check=True,
    ).stdout.decode(errors="replace")


BARE_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def base_of(line: str, close: int) -> str | None:
    """Text inside the parenthesised base ending at `close`, or None.

    `close` indexes the `)` immediately before `.field`; walk backwards
    matching parens so that a nested base like `((*(_1).0))` is read whole
    rather than truncated at its first inner paren.
    """
    depth = 0
    for i in range(close, -1, -1):
        if line[i] == ")":
            depth += 1
        elif line[i] == "(":
            depth -= 1
            if depth == 0:
                return line[i + 1 : close]
    return None


def census(dump: str, fields: list[str]) -> tuple[dict[str, int], list[str]]:
    """Per-function count of projections whose base is not a deref.

    `(frame_1).f` is non-deref; anything reaching through a `*` is a deref.
    A base of neither shape is returned as unclassified rather than assumed
    harmless — a silently miscounting tripwire is worse than none.
    """
    alt = "|".join(re.escape(f) for f in fields)
    any_proj = re.compile(r"\)\.(?:" + alt + r")\b")

    counts: dict[str, int] = defaultdict(int)
    unclassified: list[str] = []
    fn = "<toplevel>"
    for line in dump.split("\n"):
        m = FN_RE.match(line)
        if m:
            fn = m.group(1).strip()
            continue
        if line.lstrip().startswith("//"):
            continue
        for hit in any_proj.finditer(line):
            base = base_of(line, hit.start())
            if base is not None and BARE_IDENT.fullmatch(base):
                counts[fn] += 1
            elif base is not None and "*" in base:
                pass
            else:
                unclassified.append(f"{fn}: {line.strip()}")
    return counts, unclassified


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path)
    args = ap.parse_args()

    fields = vable_fields()
    print(f"virtualizable fields ({len(fields)}): {' '.join(fields)}\n")

    total: dict[str, int] = defaultdict(int)
    unclassified: list[str] = []
    for path in args.inputs:
        counts, bad = census(render(path), fields)
        for fn, n in counts.items():
            total[fn] += n
        unclassified += bad
        print(f"  {path.name}: {sum(counts.values())} non-deref projections")

    failures: list[str] = []
    print("\nnon-deref projections by function:")
    for fn in sorted(set(total) | set(EXPECTED)):
        got, want = total.get(fn, 0), EXPECTED.get(fn, 0)
        mark = "ok" if got == want else "FAIL"
        if got != want:
            failures.append(f"{fn}: expected {want}, found {got}")
        print(f"  [{mark}] {fn}: {got} (expected {want})")

    for line in unclassified:
        failures.append(f"unclassified projection shape — {line}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
