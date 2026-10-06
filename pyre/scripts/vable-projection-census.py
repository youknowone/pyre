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
import importlib.util
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
# non-dereferenced base.  It takes the frame by value, so every
# `frame.<field>` is a projection off a local aggregate rather than off a
# dereference.
#
# A set, not a count.  How many times that constructor happens to touch a
# virtualizable field moves with refactoring that has nothing to do with the
# protocol — reworking its GC-root bracket took it from 12 to 10 — so a pinned
# number fails on unrelated work and says nothing when it passes.  What
# defeats the virtualizable lowering is a *new* function acquiring such a
# projection, and that is what this gates.
ALLOWED = {"pyre_interpreter::pyframe::{FrameBox}::new"}

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
    pin_path = ROOT / "scripts" / "install-charon.py"
    spec = importlib.util.spec_from_file_location("install_charon", pin_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {pin_path}")
    pin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pin)
    version = os.environ.get("CHARON_VERSION", pin.CHARON_VERSION_DEFAULT)
    path = Path(
        os.environ.get("CHARON_DEST", pin.default_charon_dest(shared, key, version))
    ) / "charon"
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
# Charon pretty-print embeds the source of `assert!(self.field)` in
# `panic(const "assertion failed: … self.field")`. That is a string, not a
# projection; matching it attributes a non-deref to the function that only
# reads `(*self).field`.
# Charon pretty-print embeds `assert!(index < self.valuestackdepth)` as
# `panic(const "assertion failed: index < self.valuestackdepth")`.  The
# field name inside that string is not a projection; 10.04 started
# spelling the message this way and the census must not count it.
STRING_LIT = re.compile(r'"(?:\\.|[^"\\])*"')


def blank_string_lits(line: str) -> str:
    """Replace double-quoted literals with spaces, keeping columns."""
    return STRING_LIT.sub(lambda m: " " * len(m.group(0)), line)


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


def projection_base(line: str, dot: int) -> tuple[str, bool] | None:
    """The base of the projection whose `.field` starts at `dot`.

    Returns `(base, parenthesised)`.  Charon prints a projection off a
    dereference with its base in parentheses, `(*self_1).pycode`, and one off
    a local as a bare path, `frame.pycode` (older releases parenthesised that
    too, `(frame_1).pycode`).  A bare path may itself be a chain of field
    projections, `frame.inner.pycode`, whose root is then the base; a chain
    rooted at a parenthesised base, `(*self).inner.pycode`, takes that base.
    None when the text before `dot` is neither shape.
    """
    i = dot
    while i > 0 and (line[i - 1].isalnum() or line[i - 1] in "_."):
        i -= 1
    chain = line[i:dot]
    if i > 0 and line[i - 1] == ")" and (chain == "" or chain.startswith(".")):
        base = base_of(line, i - 1)
        return None if base is None else (base, True)
    root = chain.split(".", 1)[0]
    if BARE_IDENT.fullmatch(root):
        return root, False
    return None


def census(dump: str, fields: list[str]) -> tuple[dict[str, int], list[str]]:
    """Per-function count of projections whose base is not a deref.

    `frame_1.f` / `(frame_1).f` is non-deref; anything reaching through a `*`
    is a deref.  Quoted string literals are not projections — Charon 10.04
    pretty-print puts `assert!(self.field)` into a `panic(const "...")`
    message.  A base of neither shape is returned as unclassified rather
    than assumed harmless — a silently miscounting tripwire is worse than none.
    """
    alt = "|".join(re.escape(f) for f in fields)
    any_proj = re.compile(r"(?<=[\w)])\.(?:" + alt + r")\b")

    counts: dict[str, int] = defaultdict(int)
    unclassified: list[str] = []
    fn = "<toplevel>"
    for raw in dump.split("\n"):
        m = FN_RE.match(raw)
        if m:
            fn = m.group(1).strip()
            continue
        if raw.lstrip().startswith("//"):
            continue
        scan = blank_string_lits(raw)
        cut = scan.find("//")
        if cut != -1:
            scan = scan[:cut]
        for hit in any_proj.finditer(scan):
            found = projection_base(scan, hit.start())
            if found is None:
                unclassified.append(f"{fn}: {raw.strip()}")
                continue
            base, parenthesised = found
            if BARE_IDENT.fullmatch(base):
                counts[fn] += 1
            elif parenthesised and "*" in base:
                pass
            else:
                unclassified.append(f"{fn}: {raw.strip()}")
    return counts, unclassified


def _panic_string_is_not_a_projection() -> None:
    dump = """
pub fn settopvalue(self: &mut PyFrame)
{
        _5 = copy (*self).valuestackdepth;
        _15 = panic(const "assertion failed: index < self.valuestackdepth");
        _16 = copy (*self).valuestackdepth; // frame.valuestackdepth
}
"""
    counts, unclassified = census(dump, ["valuestackdepth"])
    assert unclassified == [], unclassified
    assert "settopvalue" not in counts, counts


def main() -> int:
    _panic_string_is_not_a_projection()
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
    for fn in sorted(total):
        mark = "ok" if fn in ALLOWED else "FAIL"
        if fn not in ALLOWED:
            failures.append(
                f"{fn}: {total[fn]} non-dereferenced virtualizable-field "
                "projection(s); the virtualizable lowering is suppressed for "
                "these, so only the by-value frame constructor may have them"
            )
        print(f"  [{mark}] {fn}: {total[fn]}")

    # A census that matches nothing must not read as a clean tree: if the
    # render or the projection pattern stops resolving, every count silently
    # goes to zero and the gate passes while checking nothing.  The by-value
    # constructor is the standing witness that it is still looking.
    if not total:
        failures.append(
            "no non-dereferenced projections found at all — the census is no "
            "longer matching the shape rather than the tree being clean; if "
            "the frame constructor genuinely stopped taking the frame by "
            "value, update ALLOWED"
        )

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
