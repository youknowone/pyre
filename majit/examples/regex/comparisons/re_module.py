"""CPython's `re` on the same regex and the same input -- the "CPython re
module" row of "A JIT for Regular Expression Matching" (2010), which reported
2,500,000 chars/s on a 2.26 GHz Core 2 Duo.

*** THIS ROW IS NOT COMPARABLE WITH THE OTHERS. ***

`re` is a backtracking engine.  The marked matcher is not: it makes exactly one
`shift` per input character, visiting all 93 nodes each time, and it therefore
reads every character exactly once whatever the answer is.  A backtracking
engine chooses its own amount of work -- it may read a character many times, or
it may decide a non-match early and never read the tail at all.  Two numbers in
chars/s that measure different amounts of work per character are not a
comparison, and this row is here only to forestall the reader who reaches for
it.  It is a measurement of a *different algorithm* on the *same input*.

Which `re` call, and why `fullmatch`:

  * `fullmatch` is the operation the marked matcher performs -- "is the whole
    string in the language?" -- so it is the only semantically equal one.
  * `search` would be the wrong row twice over.  It answers a different
    question, and on this pattern it is quadratic: measured here at four
    lengths (the `search_scaling` line below), the chars/s HALVES every time the
    length doubles.  At 2^20 characters that is hours.  The quadratic blowup is
    the leading `(a|b)*` being retried from every start position, which is
    exactly the backtracking the marked algorithm exists to avoid.

How much of the input does `fullmatch` actually consume?  Two measurements,
both printed on stderr, because the honest answer is not "all of it by
definition":

  1. `rate_flatness` -- chars/s measured over a sweep of lengths spanning 32x.
     If the engine bailed out after a bounded prefix, the chars/s would rise
     roughly in proportion to the length.  A flat rate means the time is linear
     in the length, i.e. the work is proportional to the whole input.  The
     number printed is max/min over the sweep; 1.0 is perfectly flat.
  2. `reaches_last_char` -- a matching pair of `a`s is planted at
     `[len-(n+2), len-1]`, so the only way the pattern can match is by reading
     the final byte.  If `fullmatch` answers True, the engine read to the end.

Neither turns this into a comparable row.  They only establish that the number
is a per-character rate over the same input, rather than a constant.

Usage: python3 re_module.py <length> <repeats> [n]
Prints one line on stdout: `re <chars_per_second>` (median of the rounds).
Per-round detail goes to stderr as `round <i> <chars_per_second>`.
"""

import re
import sys
import time

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


def pattern_source(n):
    return "(a|b)*a(a|b){%d}a(a|b)*" % n


def verify(length, n):
    """The input half of the ports' `--verify` line.  This engine has no marks
    and no node tree, so it can only attest to the bytes -- but that is the part
    that has to match, because a row measured over different bytes is not a row
    in the same table."""
    s = nonmatching(length, n)
    sys.stdout.write(
        "verify engine=re input_fnv1a=%016x head=%s tail=%s\n"
        % (fnv1a64(s), s[:64].decode(), s[-64:].decode())
    )
    return 0


def main(argv):
    if len(argv) >= 3 and argv[1] == "--verify":
        return verify(int(argv[2]), int(argv[3]) if len(argv) > 3 else 20)
    if len(argv) < 3:
        sys.stderr.write(
            "usage: re_module.py <length> <repeats> [n]\n"
            "       re_module.py --verify <length> [n]\n"
        )
        return 2
    length = int(argv[1])
    repeats = int(argv[2])
    n = int(argv[3]) if len(argv) > 3 else 20
    if length <= 0 or repeats <= 0:
        sys.stderr.write("length and repeats must both be positive\n")
        return 2

    # Compiled outside the timed region: the marked matcher builds its tree
    # outside its timed region too, and a compile charged to one row and not the
    # other would be the timing bug this file is otherwise arguing against.
    pat = re.compile(pattern_source(n).encode())
    s = nonmatching(length, n)

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

    # --- evidence 1: is the rate flat across lengths, or does it rise?
    #
    # The prefixes are slices, not fresh generations.  Both halves of the
    # generator are prefix-stable -- the LCG's first L states do not depend on
    # how many follow, and the fixup writes `s[i+d]` from `s[i]`, an index it has
    # already finalised -- so `nonmatching(L) == nonmatching(BIG)[:L]`.  That is
    # asserted rather than assumed, because if it were false this sweep would be
    # comparing different strings and the flatness would mean nothing.
    assert nonmatching(min(1024, length), n) == s[: min(1024, length)]

    def stable_rate(buf):
        """chars/s, least-noise-first: repeat until 30 ms have accumulated, and
        take the best of three such batches.  Benchmark noise is one sided --
        a preemption or a page fault can only make a run slower -- so at the
        short end of the sweep a single call measures the scheduler, and a
        sweep whose short lengths are noise cannot say anything about whether
        the long ones are linear."""
        best = 0.0
        for _ in range(3):
            calls, elapsed = 0, 0.0
            while elapsed < 0.030 or calls < 3:
                t0 = time.perf_counter()
                pat.fullmatch(buf)
                elapsed += time.perf_counter() - t0
                calls += 1
            best = max(best, calls * len(buf) / elapsed)
        return best

    sweep = []
    L = length
    for _ in range(6):
        if L < 1024:
            break
        sweep.append((L, stable_rate(s[:L])))
        L //= 2
    flat = max(r for _, r in sweep) / min(r for _, r in sweep) if sweep else float("nan")
    sys.stderr.write(
        "consumed sweep=%s rate_flatness=%.2f (1.0 == perfectly flat; an engine "
        "that stopped after a bounded prefix would show the rate RISING with "
        "length)\n"
        % (";".join("%d:%.0f" % (l, r) for l, r in sweep), flat)
    )

    # --- evidence 2: does it read the final byte?
    planted = bytearray(s)
    d = n + 1
    if length > d + 1:
        planted[length - 1 - d] = ord("a")
        planted[length - 1] = ord("a")
        reaches = pat.fullmatch(bytes(planted)) is not None
    else:
        reaches = None
    sys.stderr.write("consumed reaches_last_char=%s\n" % reaches)

    # --- why `search` is not the row: measured, not asserted.
    scaling = []
    for sl in (1000, 2000, 4000):
        if sl > length:
            break
        t0 = time.perf_counter()
        pat.search(s[:sl])
        t1 = time.perf_counter()
        scaling.append((sl, sl / (t1 - t0)))
    sys.stderr.write(
        "search_scaling %s  (chars/s falling as length rises == quadratic)\n"
        % ";".join("%d:%.0f" % (l, r) for l, r in scaling)
    )

    sys.stderr.write(
        "detail engine=re op=fullmatch pattern=%s input_fnv1a=%016x sink=%d impl=%s %s "
        "NOT_COMPARABLE=backtracking\n"
        % (pattern_source(n), fnv1a64(s), sink, sys.implementation.name,
           ".".join(str(v) for v in sys.version_info[:3]))
    )
    sys.stdout.write("re %.0f\n" % median)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
