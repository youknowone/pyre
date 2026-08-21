#!/usr/bin/env python3
"""Reject newly written `upstream.py:LINE` citations.

`AGENTS.md` (Porting discipline) says to cite upstream by symbol: a line number
rots silently as the vendored tree moves, and a rotted citation still reads as
authoritative. Existing numbers are left alone -- this only looks at lines a
change ADDS, so the rule applies from here on rather than retroactively.

Modes:
  (default)      the staged diff, for a pre-commit hook
  --base REF     everything REF..HEAD adds, for CI on a pull request
  --annotate     report as GitHub annotations and exit 0

The commit hook fails: that is the moment the line is being written and the
cheapest one at which to fix it. CI annotates instead, so a branch that
predates the hook -- or a commit made with `--no-verify` -- still shows the
citation on the diff without walling work that is already in flight.

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

def keep_output_printable():
    """Report through the console's codec without dying on it.

    The comments these scripts quote carry em dashes, and a Windows box whose
    code page is not UTF-8 cannot spell one: piping the report into anything
    raises `UnicodeEncodeError` before a single finding reaches the reader.
    `backslashreplace` spells those characters out instead, so the report
    survives a console that cannot render it.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


CITE = re.compile(r"(?:[A-Za-z0-9_.-]+/)*[a-z_][a-z0-9_]*\.py:\d+")
ESCAPE = "allow-line-citation"


def added_lines(base):
    """(path, hunk line number, text) for every line the diff adds."""
    cmd = ["git", "diff", "-U0"]
    cmd += [base, "--"] if base else ["--cached", "--"]
    # Decode as UTF-8 rather than letting `text=True` pick the locale codec:
    # the diff carries this tree's own bytes, and a Windows box whose ANSI code
    # page is not UTF-8 fails on the first em dash.
    proc = subprocess.run(cmd, capture_output=True,
                          encoding="utf-8", errors="replace")
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
    ap.add_argument("--annotate", action="store_true",
                    help="emit GitHub annotations and exit 0 instead of failing")
    ap.add_argument("files", nargs="*", help="ignored; pre-commit passes them")
    args = ap.parse_args()
    keep_output_printable()

    bad = []
    added = rust_added = 0
    files = set()
    for path, lineno, text in added_lines(args.base):
        added += 1
        if not path or not path.endswith(".rs"):
            continue
        files.add(path)
        rust_added += 1
        comment = text.find("//")
        if comment < 0 or ESCAPE in text[comment:]:
            continue
        for m in CITE.finditer(text):
            if m.start() > comment:
                bad.append((path, lineno, m.group(0), text.strip()))

    # Printed whichever way this ends. A gate whose pass is indistinguishable
    # from a gate that read an empty range is a gate nobody can trust: the
    # population belongs in the log next to the verdict.
    scanned = (f"scanned {rust_added} added Rust line(s) in {len(files)} file(s), "
               f"of {added} added line(s) in range")

    if not bad:
        print(f"{scanned}; no new line-number citations.")
        return 0

    if args.annotate:
        # `::warning file=,line=::` puts the marker on the line itself in the
        # PR's Files-changed view. A message carries no raw newline.
        for path, lineno, cite, _text in bad:
            print(f"::warning file={path},line={lineno},"
                  f"title=Cite upstream by symbol::`{cite}` names a line number. "
                  "Drop the `:LINE` and name the symbol, or add "
                  f"`{ESCAPE}` to record that the number was deliberate.")
        print(f"{scanned}; {len(bad)} new line-number citation(s), "
              "see the annotations above.")
        return 0

    print("Upstream citations must name a symbol, not a line number "
          "(AGENTS.md, Porting discipline).\n")
    for path, lineno, cite, text in bad:
        print(f"  {path}:{lineno}: {cite}")
        print(f"      {text[:100]}")
    print(f"\n{scanned}.")
    print(f"{len(bad)} new line-number citation(s).")
    print("Drop the `:LINE` and name the symbol instead — the enclosing "
          "`def`/`class` when the claim is about a statement inside one.")
    print(f"Where no symbol pins the claim, keep the number and add `{ESCAPE}` "
          "to the comment, which also records that it was a choice.")
    print("`scripts/drop-line-citations.py --apply` handles the citations whose "
          "symbol is already beside them.")
    return 1


sys.exit(main())
