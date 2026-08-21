#!/usr/bin/env python3
"""Rewrite `upstream.py:LINE symbol` citations to `upstream.py symbol`.

A line number rots silently: the vendored tree moves and the citation keeps
reading as authoritative. A symbol does not rot, and `rg` can check it. This
tool deletes the number from every citation that ALREADY names a symbol the
resolved upstream file really defines, so nothing is invented and nothing that
was checkable stops being checkable.

A citation that instead QUOTES an upstream statement is pinned by the quote, so
its number is dropped too -- but only after the quote is found to occur exactly
once in the resolved file, which establishes the target without consulting the
number at all.

It refuses every other shape. A citation with neither a symbol nor a locating
quote beside it, or one whose basename resolves to several upstream files that
all define the named symbol, needs a human to name the right symbol -- guessing
it from the cited line would launder a possibly-drifted number into an
assertive claim. `--report` lists what was refused and why.

Only citations inside `//` comments are touched, so a string literal that
happens to hold a path keeps its number.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

UPSTREAM_ROOTS = ("rpython", "pypy", "lib-python")
SEARCH_ROOTS = ("majit", "pyre")

BASENAME = r"[a-z_][a-z0-9_]*\.py"
RG_PREFILTER = BASENAME + r":[0-9]+"
PY_PATH = r"(?:[A-Za-z0-9_.-]+/)*" + BASENAME

# The number span. `,\d+` has no space allowed: a real list is written
# `graphlib.py:102,107,113`, whereas ``foo.py:12`, 3 callers`` is prose whose
# `, 3` a comma-with-space would swallow.
CITE = re.compile(rf"({PY_PATH}):(\d+(?:\s*-\s*\d+|,\d+)*)")

DEF_RE = re.compile(r"^(\s*)(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
# Module-level binding: `NAME = ...` / `NAME: T = ...`, excluding `==`.
CONST_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=[^=]")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# A backticked span long enough to be a statement rather than a bare word. Its
# uniqueness in the file is what pins the citation, so short spans are useless:
# `n` occurs everywhere and would pin nothing.
QUOTE = re.compile(r"`([^`]{6,120})`")


def quote_candidates(text):
    """Backticked spans that can pin a citation on their own.

    A comment often puts the number and the statement it points at inside one
    span -- ``error.py:548 assert len(strings) == len(formats) + 1`` -- and it
    is the statement alone that occurs in the upstream file. Yield the
    remainder after a leading citation as well as the span itself, so that
    shape pins like any other quote.
    """
    for q in QUOTE.findall(text):
        q = q.strip()
        if q:
            yield q
        m = CITE.match(q)
        if m:
            rest = q[m.end():].strip()
            if len(rest) >= 6:
                yield rest

# How much of the line around the citation may supply the symbol. Bounded so a
# long line cannot donate an unrelated identifier from its far end.
CONTEXT = 90


def build_index(root: Path):
    """basename -> [(relpath, defined names)], and relpath -> the same entry."""
    index, by_path = defaultdict(list), {}
    for base in UPSTREAM_ROOTS:
        base_dir = root / base
        if not base_dir.is_dir():
            continue
        for p in base_dir.rglob("*.py"):
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            names = set()
            for line in text.splitlines():
                if m := DEF_RE.match(line):
                    names.add(m.group(3))
                elif m := CONST_RE.match(line):
                    names.add(m.group(1))
            rel = str(p.relative_to(root))
            entry = (rel, names, text)
            index[p.name].append(entry)
            by_path[rel] = entry
    if not by_path:
        sys.exit(
            f"error: no upstream .py files under {root}/{{{','.join(UPSTREAM_ROOTS)}}}.\n"
            "Refusing to run: every citation would resolve to nothing and the "
            "tool would report a clean tree it never checked."
        )
    return index, by_path


def candidates(pyfile, index, by_path):
    """Files a citation's path could name. A cited path must match at a
    directory boundary, or `pypy/somefoo/bar.py` would answer `foo/bar.py`."""
    if "/" in pyfile:
        return [v for k, v in by_path.items() if k == pyfile or k.endswith("/" + pyfile)]
    return list(index.get(pyfile, []))


def scan(root: Path, search_roots):
    proc = subprocess.run(
        ["rg", "-l", "--type", "rust", RG_PREFILTER, *search_roots],
        cwd=root, capture_output=True, text=True,
    )
    if proc.returncode not in (0, 1):
        sys.exit(f"error: rg failed ({proc.returncode}): {proc.stderr.strip()}")
    files = [f for f in proc.stdout.splitlines() if f]
    if not files:
        sys.exit(
            f"error: scanning {', '.join(search_roots)} found no citations at all.\n"
            "That is a misconfigured search, not a clean tree."
        )
    return files


def rewrite_line(line, index, by_path, stats, refused):
    """Return the line with every droppable citation's number removed."""
    comment = line.find("//")
    out, last, changed = [], 0, False
    for m in CITE.finditer(line):
        stats["total"] += 1
        if comment < 0 or m.start() < comment:
            stats["refused_not_in_comment"] += 1
            refused["not_in_comment"].append(m.group(0))
            continue
        cands = candidates(m.group(1), index, by_path)
        if not cands:
            stats["refused_file_not_found"] += 1
            refused["file_not_found"].append(m.group(0))
            continue
        # Only the comment may supply the symbol. A trailing comment sits on a
        # line of Rust, and that code's own identifiers -- a string literal
        # naming a dunder, say -- must not be read as an upstream symbol.
        left = max(comment + 2, m.start() - CONTEXT)
        ctx = line[left:m.start()] + " " + line[m.end():m.end() + CONTEXT]
        idents = set(IDENT.findall(ctx))
        matched = [rel for rel, names, _text in cands if idents & names]
        by_symbol = bool(matched)
        if not matched:
            # No symbol -- but a quote occurring exactly once in the file
            # locates the citation on its own, so the number adds nothing.
            quotes = list(quote_candidates(line[comment:]))
            pinned = [rel for rel, _names, body in cands
                      if any(body.count(q) == 1 for q in quotes)]
            if len(pinned) != 1:
                stats["refused_no_symbol"] += 1
                refused["no_symbol"].append(m.group(0))
                continue
            matched = pinned
        if len(matched) > 1:
            stats["refused_symbol_in_several_files"] += 1
            refused["symbol_in_several_files"].append(m.group(0))
            continue
        # Counted only past the last refusal, or a citation refused for
        # resolving to several files would also be tallied as dropped.
        stats["dropped_named_symbol" if by_symbol else "dropped_pinned_by_quote"] += 1
        out.append(line[last:m.start()])
        out.append(m.group(1))  # the path, without `:NNN`
        last = m.end()
        changed = True
        stats["dropped"] += 1
    if not changed:
        return line
    out.append(line[last:])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--search-root", action="append", dest="search_roots")
    ap.add_argument("--apply", action="store_true", help="write files (default: report only)")
    ap.add_argument("--report", action="store_true", help="print refusal samples")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files (bisecting aid)")
    args = ap.parse_args()

    root = args.root.resolve()
    search_roots = args.search_roots or list(SEARCH_ROOTS)
    index, by_path = build_index(root)
    files = scan(root, search_roots)
    if args.limit:
        files = files[:args.limit]

    stats, refused = Counter(), defaultdict(list)
    touched = []
    for rel in files:
        p = root / rel
        original = p.read_text()
        lines = original.splitlines(keepends=True)
        new = [rewrite_line(l, index, by_path, stats, refused) for l in lines]
        text = "".join(new)
        if text != original:
            touched.append((p, text))

    # A file is written whole or not at all: a partial write would leave a
    # citation set nobody can reason about if a later file raises.
    if args.apply:
        for p, text in touched:
            p.write_text(text)

    print(f"citations seen        {stats['total']:6d}")
    print(f"numbers dropped       {stats['dropped']:6d}")
    print(f"  named a symbol      {stats['dropped_named_symbol']:6d}")
    print(f"  pinned by a quote   {stats['dropped_pinned_by_quote']:6d}")
    for k in ("refused_no_symbol", "refused_symbol_in_several_files",
              "refused_file_not_found", "refused_not_in_comment"):
        print(f"{k:22s}{stats[k]:6d}")
    total = stats["dropped"] + sum(stats[k] for k in stats if k.startswith("refused_"))
    assert stats["dropped"] == stats["dropped_named_symbol"] + stats["dropped_pinned_by_quote"]
    assert stats["total"] == total, f"partition invariant: {stats['total']} != {total}"
    print("partition invariant   OK")
    print(f"files changed         {len(touched):6d}  ({'written' if args.apply else 'dry run'})")

    if args.report:
        for k, v in refused.items():
            print(f"\n--- refused: {k} ({len(v)}) ---")
            for s in v[:15]:
                print("   ", s)


main()
