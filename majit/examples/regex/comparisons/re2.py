"""Google re2 on the same regex and the same input -- the "Google re2" row of
"A JIT for Regular Expression Matching" (2010), which reported 550,000 chars/s
on a 2.26 GHz Core 2 Duo.

re2 is the interesting foreign row, and the only one of the seven that is
algorithmically the same *kind* of thing as the marked matcher: it is an
automaton engine, not a backtracker, so it makes a bounded amount of work per
input character and reads the whole string.  Unlike the `re` row below it, a
chars/s figure from re2 IS comparable in kind.

re2 is not installed on the machine this harness was written on and nothing is
installed by this script.  When the module is missing this prints
`re2 unavailable: <the one pip command>` and exits 0, so `run.sh` can carry on
and the table shows the row as absent rather than as zero.

Usage: python3 re2.py <length> <repeats> [n]
Prints one line on stdout: `re2 <chars_per_second>`, or `re2 unavailable: ...`.
Per-round detail goes to stderr as `round <i> <chars_per_second>`.
"""

import os
import sys
import time

INSTALL = "pip install google-re2"

# This file is called `re2.py`, so its own directory is the first thing on
# `sys.path` and `import re2` finds *itself*.  That is not a hypothetical: it
# imports cleanly, and the failure only surfaces later as
# `module 're2' has no attribute 'compile'` -- a bogus reason for a skipped row
# that looks like a broken binding.  Drop this directory before importing, so
# the import means what it says.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

try:
    import re2
except ImportError:
    sys.stdout.write("re2 unavailable: %s\n" % INSTALL)
    sys.exit(0)

MASK64 = (1 << 64) - 1


def nonmatching(length, n, seed=42):
    """Byte for byte the generator in `regex.rs::nonmatching`; see marked.py."""
    out = bytearray(length)
    a, b = ord("a"), ord("b")
    for i in range(length):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & MASK64
        out[i] = a if ((seed >> 33) & 1) == 0 else b
    d = n + 1
    for i in range(length - d):
        if out[i] == a and out[i + d] == a:
            out[i + d] = b
    return bytes(out)


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & MASK64
    return h


def verify(length, n):
    """The input half of the ports' `--verify` line.  This engine has no marks
    and no node tree, so it can only attest to the bytes -- but that is the part
    that has to match, because a row measured over different bytes is not a row
    in the same table."""
    s = nonmatching(length, n)
    sys.stdout.write(
        "verify engine=re2 input_fnv1a=%016x head=%s tail=%s\n"
        % (fnv1a64(s), s[:64].decode(), s[-64:].decode())
    )
    return 0


def main(argv):
    if len(argv) >= 3 and argv[1] == "--verify":
        return verify(int(argv[2]), int(argv[3]) if len(argv) > 3 else 20)
    if len(argv) < 3:
        sys.stderr.write(
            "usage: re2.py <length> <repeats> [n]\n"
            "       re2.py --verify <length> [n]\n"
        )
        return 2
    length = int(argv[1])
    repeats = int(argv[2])
    n = int(argv[3]) if len(argv) > 3 else 20
    if length <= 0 or repeats <= 0:
        sys.stderr.write("length and repeats must both be positive\n")
        return 2

    # Compiled outside the timed region, for the same reason as in re_module.py.
    pat = re2.compile(("(a|b)*a(a|b){%d}a(a|b)*" % n).encode())
    s = nonmatching(length, n)

    # `fullmatch` is the operation the marked matcher performs -- "is the whole
    # string in the language?".  google-re2 mirrors the `re` API; if a future
    # binding drops `fullmatch` this is the line to look at rather than
    # silently switching to `match`, which asks a different question.
    if pat.fullmatch(s) is not None:
        sys.stderr.write("the benchmark input matched: it is supposed NOT to\n")
        return 1

    pat.fullmatch(s)  # untimed round

    rates = []
    sink = 0
    for r in range(repeats):
        t0 = time.perf_counter()
        m = pat.fullmatch(s)
        t1 = time.perf_counter()
        sink += 1 if m is not None else 0
        rate = length / (t1 - t0)
        rates.append(rate)
        sys.stderr.write("round %d %.0f\n" % (r + 1, rate))
    if sink != 0:
        sys.stderr.write("a timed round reported a match; the number is not valid\n")
        return 1

    median = sorted(rates)[len(rates) // 2]
    sys.stderr.write(
        "detail engine=re2 op=fullmatch input_fnv1a=%016x sink=%d\n" % (fnv1a64(s), sink)
    )
    sys.stdout.write("re2 %.0f\n" % median)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
