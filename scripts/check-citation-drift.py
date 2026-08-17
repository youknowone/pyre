#!/usr/bin/env python3
"""Report stale line-number citations into the vendored RPython/PyPy sources.

A citation of ``file.py:line symbol`` is checkable when ``symbol`` names a
function or class in the resolved file. The tool reports citations whose line
now belongs to a different definition. It does not validate prose about
upstream behavior, call-site citations, or names that are not definitions.

The report partitions checkable and unmeasured citations so a zero cannot be
mistaken for complete coverage. This is a reporting tool; it exits nonzero only
when its own population or partition invariant fails.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Share the basename pattern between the `rg` prefilter and parser so files
# containing digits cannot be accepted by one and dropped by the other.
BASENAME = r"[a-z_][a-z0-9_]*\.py"
RG_PREFILTER = BASENAME + r":[0-9]+"
PY_PATH = r"(?:[A-Za-z0-9_.-]+/)*" + BASENAME

# LOCATOR is a non-overlapping population control. CITE shares its prefix but
# parses more text, so equal counts prove the parser did not consume a following
# citation accidentally.
LOCATOR = re.compile(rf"({PY_PATH}):(\d+)(?:-\d+)?")

# The separator admits adjacent code spans. The negative lookahead prevents a
# broadened separator from consuming the basename of the next citation as the
# current citation's symbol. `--self-test` exercises that interaction.
SEPARATOR = r"[\s`'\"*:\-]*"
SYMBOL = r"(?:([A-Za-z_][A-Za-z0-9_]*)\b(?!\.py))?"
CITE = re.compile(rf"({PY_PATH}):(\d+)(?:-\d+)?" + SEPARATOR + SYMBOL)

# A deliberately broken parser used as the population-control self-test.
SEPARATOR_OVERCORRECTION = r"[\s`'\"*:,()\-]*"
SYMBOL_NO_LOOKAHEAD = r"(?:([A-Za-z_][A-Za-z0-9_]*)\b)?"

UPSTREAM_ROOTS = ("rpython", "pypy", "lib-python")
SEARCH_ROOTS = ("majit", "pyre")

DEF_RE = re.compile(r"^(\s*)(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")


def build_symbol_index(root: Path, upstream_roots):
    """basename -> [(path, defs, total)], and relpath -> (path, defs, total)."""
    index = defaultdict(list)
    by_path = {}
    for base in upstream_roots:
        base_dir = root / base
        if not base_dir.is_dir():
            continue
        for p in base_dir.rglob("*.py"):
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            defs = []
            for i, line in enumerate(text.splitlines(), 1):
                m = DEF_RE.match(line)
                if m:
                    defs.append((i, len(m.group(1)), m.group(2), m.group(3)))
            entry = (p, defs, len(text.splitlines()))
            index[p.name].append(entry)
            by_path[str(p.relative_to(root))] = entry
    return index, by_path


def enclosing_symbols(defs, total, lineno):
    """Every def/class whose body plausibly contains `lineno`, innermost first.

    A def at indent d owns lines until the next def/class at indent <= d, so a
    method inside a class yields [method, class].  Returns None when the line is
    outside the file, which is a distinct and louder defect than drift.
    """
    if lineno < 1 or lineno > total:
        return None
    out = []
    for idx, (ln, indent, _kind, name) in enumerate(defs):
        if ln > lineno:
            break
        end = total
        for ln2, indent2, _, _ in defs[idx + 1:]:
            if indent2 <= indent:
                end = ln2 - 1
                break
        if ln <= lineno <= end:
            out.append(name)
    out.reverse()
    return out


def scan_tree(root: Path, search_roots):
    """Every line under `search_roots` that contains an upstream citation."""
    missing = [r for r in search_roots if not (root / r).exists()]
    if missing:
        sys.exit(f"error: search root(s) not found under {root}: {', '.join(missing)}")
    proc = subprocess.run(
        ["rg", "-n", "--no-heading", "--type", "rust", RG_PREFILTER, *search_roots],
        cwd=root, capture_output=True, text=True,
    )
    # rg exits 1 for "no matches", which for this tool is itself a red flag: the
    # tree is known to carry tens of thousands of citations, so an empty result
    # means the search was misconfigured, not that the tree is clean.
    if proc.returncode not in (0, 1):
        sys.exit(f"error: rg failed ({proc.returncode}): {proc.stderr.strip()}")
    rows = []
    for row in proc.stdout.splitlines():
        try:
            path, lno, text = row.split(":", 2)
        except ValueError:
            continue
        rows.append((path, lno, text))
    return rows


def classify(rows, index, by_path):
    stats = defaultdict(int)
    drifted = []
    out_of_range = []

    for path, lno, text in rows:
        for m in CITE.finditer(text):
            pyfile, cited_line, symbol = m.group(1), int(m.group(2)), m.group(3)
            stats["total_citations"] += 1
            if not symbol:
                stats["no_symbol_named"] += 1
                continue
            if "/" in pyfile:
                # the citation NAMES its corpus -- resolve by path suffix, never
                # by basename, or a same-named file in another corpus wins.
                hits = [v for k, v in by_path.items() if k.endswith(pyfile)]
                if len(hits) > 1:
                    stats["ambiguous_path"] += 1
                    continue
                if not hits:
                    stats["cited_path_not_found"] += 1
                    continue
                entry = hits[0]
            else:
                cands = index.get(pyfile)
                if not cands:
                    stats["upstream_file_not_found"] += 1
                    continue
                if len(cands) > 1:
                    stats["ambiguous_basename"] += 1
                    continue
                entry = cands[0]
            _p, defs, total = entry
            encl = enclosing_symbols(defs, total, cited_line)
            if encl is None:
                stats["out_of_range"] += 1
                out_of_range.append((path, lno, pyfile, cited_line, symbol, total))
                continue
            if symbol in encl:
                stats["ok"] += 1
                continue
            # Only a def/class elsewhere in the same file gives us an assertion
            # to falsify. Other trailing words may describe call sites or prose.
            names = {n for _, _, _, n in defs}
            if symbol not in names:
                stats["symbol_is_prose"] += 1
                continue
            stats["drifted"] += 1
            true_lines = [ln for ln, _, _, n in defs if n == symbol]
            dist = min(abs(t - cited_line) for t in true_lines)
            drifted.append((path, lno, pyfile, cited_line, symbol,
                            encl[0] if encl else "<module level>", dist))
    return stats, drifted, out_of_range


def tier_of(dist):
    if dist > 100:
        return "A"
    if dist > 20:
        return "B"
    if dist > 5:
        return "C"
    return "D"


TIER_LABEL = {
    "A": "A >100 lines",
    "B": "B 21-100",
    "C": "C 6-20",
    "D": "D <=5 (adjacent def)",
}


def check_population(rows):
    """CITE must find exactly as many citations as the non-overlapping LOCATOR.

    This is the guard on parser edits.  It is a within-run comparison rather
    than a recorded number precisely so that adding citations to the tree -- the
    normal thing to do -- cannot make it stale and get it rekeyed blindly.
    """
    located = sum(len(LOCATOR.findall(text)) for _, _, text in rows)
    parsed = sum(len(CITE.findall(text)) for _, _, text in rows)
    return located, parsed


def self_test(rows):
    """Prove the population guard catches a known parser mutation."""
    located, parsed = check_population(rows)

    def count(sep, sym):
        r = re.compile(rf"({PY_PATH}):(\d+)(?:-\d+)?" + sep + sym)
        return sum(len(r.findall(text)) for _, _, text in rows)

    wide_pair = count(SEPARATOR_OVERCORRECTION, SYMBOL_NO_LOOKAHEAD)
    no_look = count(SEPARATOR, SYMBOL_NO_LOOKAHEAD)
    wide_only = count(SEPARATOR_OVERCORRECTION, SYMBOL)

    print(f"locator population (ground truth)     {located:,}")
    print(f"shipped parser                        {parsed:,}"
          f"   (must EQUAL locator)")
    print(f"mutation: wide separator + no lookahead {wide_pair:>8,}"
          f"   (must be LESS -- this is the guard's control)")
    print(f"mutation: no lookahead alone          {no_look:,}"
          f"   (latent today: equal is expected)")
    print(f"mutation: wide separator alone        {wide_only:,}"
          f"   (latent today: equal is expected)")

    ok = parsed == located and wide_pair < located
    if parsed != located:
        print(f"\nFAIL: the shipped parser already loses "
              f"{located - parsed:,} citations.")
    if wide_pair >= located:
        print("\nFAIL: the guard is DEAD -- the control mutation no longer "
              "trips it, so this control no longer proves anything and the "
              "population assertion is unvalidated.")
    if ok:
        print(f"\nPASS: guard is live -- the control mutation eats "
              f"{located - wide_pair:,} citations and the assertion catches it.")
    return 0 if ok else 1


def main():
    default_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", type=Path, default=default_root,
                    help="repository root (default: parent of scripts/)")
    ap.add_argument("--search-root", action="append", dest="search_roots",
                    metavar="DIR",
                    help=f"tree to scan (repeatable; default: {', '.join(SEARCH_ROOTS)})")
    ap.add_argument("--upstream-root", action="append", dest="upstream_roots",
                    metavar="DIR",
                    help=f"upstream corpus (repeatable; default: {', '.join(UPSTREAM_ROOTS)})")
    ap.add_argument("--claims", type=int, default=25, metavar="N",
                    help="distinct drifted claims to list in detail (default: 25, 0 for all)")
    ap.add_argument("--summary", action="store_true",
                    help="print the headline and partition only")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the population guard is live, then exit")
    args = ap.parse_args()

    root = args.root.resolve()
    search_roots = args.search_roots or list(SEARCH_ROOTS)
    upstream_roots = args.upstream_roots or list(UPSTREAM_ROOTS)

    rows = scan_tree(root, search_roots)
    if args.self_test:
        return self_test(rows)

    located, parsed = check_population(rows)
    if parsed != located:
        sys.exit(
            f"POPULATION INVARIANT FAILED: the parser found {parsed:,} citations "
            f"where the non-overlapping locator found {located:,}.\n"
            f"A match is running into the next citation and consuming it, so "
            f"{located - parsed:,} citations are being lost silently -- every "
            f"count below would be a fraction of an unknown whole. Fix the "
            f"parser before reading any number from this tool."
        )

    index, by_path = build_symbol_index(root, upstream_roots)
    stats, drifted, out_of_range = classify(rows, index, by_path)

    checkable = stats["ok"] + stats["drifted"] + stats["out_of_range"]
    unverifiable = (stats["no_symbol_named"] + stats["symbol_is_prose"]
                    + stats["ambiguous_basename"] + stats["ambiguous_path"]
                    + stats["upstream_file_not_found"]
                    + stats["cited_path_not_found"])
    total = checkable + unverifiable
    if total != stats["total_citations"]:
        sys.exit(
            f"PARTITION INVARIANT FAILED: {total:,} != "
            f"{stats['total_citations']:,} -- a bucket is missing or "
            f"double-counted, so the headline percentage is unusable."
        )

    tiers = defaultdict(int)
    for *_rest, dist in drifted:
        tiers[tier_of(dist)] += 1
    tier_str = " ".join(f"{t}:{tiers[t]}" for t in "ABCD")
    rate = 100.0 * (stats["drifted"] + stats["out_of_range"]) / checkable if checkable else 0.0

    print(f"checked {checkable:,} of {stats['total_citations']:,} claims")
    print(f"  {unverifiable:,} not checkable (name no symbol / ambiguous "
          f"basename / symbol is prose)")
    print(f"drifted {stats['drifted']:,} ({rate:.1f}%) -- {tier_str} -- "
          f"past EOF {stats['out_of_range']:,}")
    print("NOTE: this finds stale line numbers. It cannot find a comment that "
          "describes")
    print("      control flow upstream does not have.")
    print(f"NOTE: the {unverifiable:,} not-checkable citations are UNMEASURED, "
          f"not clean. The only")
    print("      hand-audited sample of that bucket was 10 of 10 drifted.")
    print("WARNING: much of the pyjitpl.py drift is a uniform +24, which looks "
          "like one")
    print("      upstream insertion and therefore like a one-line sed. It is "
          "not. A blanket")
    print("      +24 breaks every citation whose target sits ABOVE the "
          "insertion point,")
    print("      including the ones that are correct today. Repoint per site.")

    print("\n=== PARTITION (must sum to total) ===")
    print(f"  CHECKABLE                {checkable:,}")
    print(f"    correct                  {stats['ok']:,}")
    print(f"    DRIFTED                  {stats['drifted']:,}")
    print(f"    out of range (past EOF)  {stats['out_of_range']:,}")
    print(f"  UNVERIFIABLE (unmeasured){unverifiable:>8,}")
    print(f"    names no symbol          {stats['no_symbol_named']:,}")
    print(f"    trailing word not a def  {stats['symbol_is_prose']:,}")
    print(f"    ambiguous basename       {stats['ambiguous_basename']:,}")
    print(f"    ambiguous path           {stats['ambiguous_path']:,}")
    print(f"    upstream file not found  {stats['upstream_file_not_found']:,}")
    print(f"    cited path not found     {stats['cited_path_not_found']:,}")
    print(f"  SUM                      {total:,}")

    print("\n=== DRIFT SEVERITY (distance from cited line to the named symbol) ===")
    for t in "ABCD":
        print(f"  {TIER_LABEL[t]:24s} {tiers[t]:,}")
    print("  D is mostly an adjacent def and is not listed below.")

    if args.summary:
        return 0

    by_claim = defaultdict(list)
    for path, lno, pyfile, cited, sym, actual, dist in drifted:
        if dist > 5:
            by_claim[(pyfile, cited, sym, actual, dist)].append(f"{path}:{lno}")
    ordered = sorted(by_claim.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]))
    shown = ordered if args.claims == 0 else ordered[:args.claims]
    print(f"\n=== DRIFTED CLAIMS: cited line is inside a DIFFERENT function ===")
    print(f"({len(ordered):,} distinct claims above tier D; showing {len(shown):,})")
    for (pyfile, cited, sym, actual, dist), sites in shown:
        print(f"\n{pyfile}:{cited} claims `{sym}` -> inside `{actual}`"
              f"  (symbol is {dist} lines away)  [{len(sites)} site(s)]")
        for s in sites[:6]:
            print(f"    {s}")
        if len(sites) > 6:
            print(f"    ... +{len(sites) - 6} more")

    if out_of_range:
        print("\n=== OUT OF RANGE: cited line exceeds the upstream file ===")
        for path, lno, pyfile, cited, sym, total_lines in out_of_range:
            print(f"  {path}:{lno}  {pyfile}:{cited} "
                  f"(file has {total_lines} lines) `{sym}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
