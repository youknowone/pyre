#!/usr/bin/env python3
"""Reject newly written `upstream.py:LINE` citations.

`AGENTS.md` (Porting discipline) says to cite upstream by symbol: a line number
rots silently as the vendored tree moves, and a rotted citation still reads as
authoritative. Existing numbers are left alone -- this only looks at lines a
change ADDS, so the rule applies from here on rather than retroactively.

Modes:
  (default)      the staged diff, for a pre-commit hook
  --base REF     everything REF..HEAD adds, for CI on a pull request

A line that genuinely needs a number -- an unnamed arm inside a long function,
a module-level comment with no symbol at all -- keeps it by carrying
`allow-line-citation` in the same comment, which also tells the next reader the
number was a deliberate choice.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

CITE = re.compile(r"(?:[A-Za-z0-9_.-]+/)*[a-z_][a-z0-9_]*\.py:\d+")
ESCAPE = "allow-line-citation"


def added_lines(base):
    """(path, hunk line number, text) for every line the diff adds."""
    cmd = ["git", "diff", "-U0"]
    cmd += [base, "--"] if base else ["--cached", "--"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"error: {' '.join(cmd)} failed: {proc.stderr.strip()}")
    path, lineno = None, 0
    for line in proc.stdout.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            lineno = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            yield path, lineno, line[1:]
            lineno += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", help="diff against this ref instead of the index")
    ap.add_argument("files", nargs="*", help="ignored; pre-commit passes them")
    args = ap.parse_args()

    bad = []
    for path, lineno, text in added_lines(args.base):
        if not path or not path.endswith(".rs"):
            continue
        comment = text.find("//")
        if comment < 0 or ESCAPE in text[comment:]:
            continue
        for m in CITE.finditer(text):
            if m.start() > comment:
                bad.append((path, lineno, m.group(0), text.strip()))

    if not bad:
        return 0
    print("Upstream citations must name a symbol, not a line number "
          "(AGENTS.md, Porting discipline).\n")
    for path, lineno, cite, text in bad:
        print(f"  {path}:{lineno}: {cite}")
        print(f"      {text[:100]}")
    print(f"\n{len(bad)} new line-number citation(s).")
    print("Drop the `:LINE` and name the symbol instead — the enclosing "
          "`def`/`class` when the claim is about a statement inside one.")
    print(f"Where no symbol pins the claim, keep the number and add `{ESCAPE}` "
          "to the comment, which also records that it was a choice.")
    print("`scripts/drop-line-citations.py --apply` handles the citations whose "
          "symbol is already beside them.")
    return 1


sys.exit(main())
