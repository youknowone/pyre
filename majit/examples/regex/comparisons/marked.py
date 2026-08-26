"""The marked-regex matcher in pure Python -- the "pure Python" row of
"A JIT for Regular Expression Matching" (2010), which reported 12,200 chars/s
on a 2.26 GHz Core 2 Duo.

The shape is part 1's, "An Efficient and Elegant Regular Expression Matcher in
Python": a class per node kind, a `shift(c, mark)` method that delegates to a
per-kind `_shift` and stores the mark it computed, and `empty` set once in
`__init__` rather than recomputed.  Marks live in the nodes, so `Sequence` can
read `self.left.marked` -- the mark left by the PREVIOUS character -- and that
read is the whole reason the algorithm needs no backtracking.

The operators are part 2's: `&` and `|` over ints, not `and` and `or` over
bools.  Part 2 makes that substitution so the body carries no short-circuit
branch, and this file follows it for one reason beyond fidelity -- the crate's
`src/interp.rs` and the C++ and Java ports here all use `&` / `|`, and a row
that used the short-circuiting operators would be measuring a strictly smaller
amount of work than the rows it sits next to.  The `and` / `or` variant is
available as `--short-circuit` for anyone who wants to price the substitution;
it is not what the reported row runs.

Two ways this row could lie, and what is done about each:

  * The matcher exits early.  It cannot: `match_` loops over the whole input
    with no `break`, and the answer is checked to be "no match" -- a
    non-matching input is the point, because then no early exit hides the
    per-character cost.  A match makes the program exit non-zero.
  * The number is really an import or a build cost.  Only `match_` is timed;
    the tree and the 1 MiB input are built before the clock starts, and one
    untimed round runs first.

`marked.py --verify <length> [n]` is the correctness gate: it prints a line
that every port must print identically -- the input digest, the answers to a
fixed battery, and a digest of all 93 marks left after scanning the benchmark
input.  See the `verify` section below for why each part is there.  It catches
the third way this row could lie: a matcher that lost an arm and answers "no
match" to everything would pass the check above and post a number.

Usage: python3 marked.py <length> <repeats> [n] [--short-circuit]
Prints one line on stdout: `python <chars_per_second>` (median of the rounds).
Per-round detail goes to stderr as `round <i> <chars_per_second>`.
"""

import sys
import time

# ---------------------------------------------------------------- the matcher

class Regex(object):
    """Base: owns the one mutable bit and the `shift` that stores it."""

    def __init__(self):
        self.marked = 0

    def shift(self, c, mark):
        marked = self._shift(c, mark)
        self.marked = marked
        return marked

    def reset(self):
        self.marked = 0


class Char(Regex):
    def __init__(self, c):
        Regex.__init__(self)
        self.c = c
        self.empty = 0

    def _shift(self, c, mark):
        return mark & (c == self.c)


class Epsilon(Regex):
    def __init__(self):
        Regex.__init__(self)
        self.empty = 1

    def _shift(self, c, mark):
        return 0


class Alternative(Regex):
    def __init__(self, left, right):
        Regex.__init__(self)
        self.left = left
        self.right = right
        self.empty = left.empty | right.empty

    def _shift(self, c, mark):
        marked_left = self.left.shift(c, mark)
        marked_right = self.right.shift(c, mark)
        return marked_left | marked_right

    def reset(self):
        self.marked = 0
        self.left.reset()
        self.right.reset()


class Repetition(Regex):
    def __init__(self, re):
        Regex.__init__(self)
        self.re = re
        self.empty = 1

    def _shift(self, c, mark):
        return self.re.shift(c, mark | self.marked)

    def reset(self):
        self.marked = 0
        self.re.reset()


class Sequence(Regex):
    def __init__(self, left, right):
        Regex.__init__(self)
        self.left = left
        self.right = right
        self.empty = left.empty & right.empty

    def _shift(self, c, mark):
        # The left mark from the PREVIOUS character is what enters the right
        # side, so read it before `shift` overwrites it.
        old_marked_left = self.left.marked
        marked_left = self.left.shift(c, mark)
        marked_right = self.right.shift(c, old_marked_left | (mark & self.left.empty))
        return (marked_left & self.right.empty) | marked_right

    def reset(self):
        self.marked = 0
        self.left.reset()
        self.right.reset()


# Part 1's operators, kept only so the substitution can be priced.  These are
# the `and` / `or` bodies, monkeypatched over the classes above by
# `--short-circuit`; everything else, including `empty`, is unchanged.
def _short_circuit_bodies():
    def char_shift(self, c, mark):
        return mark and c == self.c

    def alt_shift(self, c, mark):
        marked_left = self.left.shift(c, mark)
        marked_right = self.right.shift(c, mark)
        return marked_left or marked_right

    def rep_shift(self, c, mark):
        return self.re.shift(c, mark or self.marked)

    def seq_shift(self, c, mark):
        old_marked_left = self.left.marked
        marked_left = self.left.shift(c, mark)
        marked_right = self.right.shift(c, old_marked_left or (mark and self.left.empty))
        return marked_left and self.right.empty or marked_right

    Char._shift = char_shift
    Alternative._shift = alt_shift
    Repetition._shift = rep_shift
    Sequence._shift = seq_shift


def match_(re, s):
    """Shift one mark in from the left for `s[0]`, then shift the marks already
    inside the tree along for every remaining character.  No `break`: every
    character is looked at, whatever the answer turns out to be."""
    if not s:
        return re.empty
    result = re.shift(s[0], 1)
    for c in s[1:]:
        result = re.shift(c, 0)
    re.reset()
    return result


# ------------------------------------------------------------------ the tree

def ab():
    """`a|b`.  A fresh pair every call: marks live in the nodes, so an instance
    may never be shared -- the twenty `(a|b)` groups of `(a|b){20}` are twenty
    distinct objects."""
    return Alternative(Char(ord("a")), Char(ord("b")))


def build_balanced(xs, lo, hi):
    """Balanced association, matching `regex.rs::build_balanced`: left half is
    `xs[lo:mid]`, right half is `xs[mid:hi]`."""
    if hi - lo == 1:
        return xs[lo]
    mid = lo + (hi - lo) // 2
    return Sequence(build_balanced(xs, lo, mid), build_balanced(xs, mid, hi))


def bench_regex(n):
    """`(a|b)*a(a|b){n}a(a|b)*`, the benchmark regex of the post."""
    parts = [Repetition(ab()), Char(ord("a"))]
    parts.extend(ab() for _ in range(n))
    parts.append(Char(ord("a")))
    parts.append(Repetition(ab()))
    return build_balanced(parts, 0, len(parts))


def fnv1a64(data):
    """A digest of the input, so `run.sh` can prove every port scanned the same
    bytes.  Any hash would do; FNV-1a is four lines in all four languages."""
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & MASK64
    return h


def count_nodes(node):
    c = 1
    for attr in ("left", "right", "re"):
        child = getattr(node, attr, None)
        if child is not None:
            c += count_nodes(child)
    return c


# ------------------------------------------------------------------ the input

MASK64 = (1 << 64) - 1


def nonmatching(length, n, seed=42):
    """A random `a`/`b` string forced NOT to match `(a|b)*a(a|b){n}a(a|b)*`.

    Byte for byte the generator in `regex.rs::nonmatching`: the same LCG
    constants, masked to 64 bits because Python ints do not wrap, the same bit
    picked out of the state, and the same left-to-right fixup.  That regex
    matches iff some pair of `a`s sits exactly `n + 1` apart, and a random
    string almost surely has one, so clearing those pairs is what makes a
    non-matching input."""
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


# ------------------------------------------------------------------- verify
#
# `--verify` is the correctness gate, and it is built so that the four ports
# can be checked against EACH OTHER and against the crate's `interp::matches`,
# not merely each against its own copy of an expectation.  It prints one line,
# and every port must print the same one:
#
#   verify nodes=.. input_fnv1a=.. head=.. tail=.. answers=.. marks=..
#
#   * `input_fnv1a`, `head`, `tail` -- the bytes the port generated.  A port
#     whose LCG is off by one wrapping multiply still produces a plausible
#     `a`/`b` string, and a chars/s number taken over different bytes is not a
#     row in the same table.
#   * `answers` -- one bit per case of a fixed battery: the 28 vectors of
#     `regex.rs::vectors()`, then four cases at the benchmark's own scale (the
#     non-matching input, and the same input with a matching pair of `a`s
#     planted at the front, in the middle, and hard against the end).  The
#     planted cases are the ones that would catch a matcher that lost an arm
#     and answers "no" to everything -- which a non-matching benchmark alone
#     cannot catch -- and the last one only matches if the final byte was read.
#   * `marks` -- a digest of all 93 `marked` bits left in the tree after
#     scanning the benchmark input, taken BEFORE `reset` clears them.  This is
#     the strong one: it compares the whole state of the computation after a
#     million characters, so two ports agreeing on it are doing the same work
#     and not merely arriving at the same boolean.
#
# The 28 vectors are also checked against their expected answers here, so a
# port that agrees with the others but disagrees with `regex.rs` still fails.

def _re_abc():  # `abc`
    return Sequence(Sequence(Char(ord("a")), Char(ord("b"))), Char(ord("c")))


def _re_abc_alt():  # `a|b|c`
    return Alternative(Alternative(Char(ord("a")), Char(ord("b"))), Char(ord("c")))


def _re_tricky():  # `((abc)*|(abcd))(d|e)`
    return Sequence(
        Alternative(Repetition(_re_abc()), Sequence(_re_abc(), Char(ord("d")))),
        Alternative(Char(ord("d")), Char(ord("e"))),
    )


# `regex.rs::vectors()`, in its order.
VECTORS = [
    (_re_abc_alt, "a", True), (_re_abc_alt, "b", True),
    (_re_abc_alt, "c", True), (_re_abc_alt, "d", False),
    (_re_abc_alt, "", False), (_re_abc_alt, "ab", False),
    (lambda: Repetition(_re_abc_alt()), "abcbac", True),
    (lambda: Repetition(_re_abc_alt()), "", True),
    (lambda: Repetition(_re_abc_alt()), "abd", False),
    (lambda: Repetition(_re_abc_alt()), "a", True),
    (_re_abc, "abc", True), (_re_abc, "abcd", False),
    (_re_abc, "ab", False), (_re_abc, "", False),
    (_re_tricky, "abcabcabcd", True), (_re_tricky, "abcd", True),
    (_re_tricky, "abcde", True), (_re_tricky, "abcdf", False),
    (_re_tricky, "abcabcd", True), (_re_tricky, "d", True),
    (_re_tricky, "e", True), (Epsilon, "", True), (Epsilon, "a", False),
    (lambda: bench_regex(2), "aaaa", True),
    (lambda: bench_regex(2), "abba", True),
    (lambda: bench_regex(2), "aba", False),
    (lambda: bench_regex(2), "babbab", True),
    (lambda: bench_regex(2), "bbbb", False),
]


def children(node):
    """Pre-order children, in the same slot order as the `NodeRec` graph:
    `left` then `right`, with `Repetition`'s single child in the `left` slot."""
    if isinstance(node, Repetition):
        return (node.re,)
    if isinstance(node, (Alternative, Sequence)):
        return (node.left, node.right)
    return ()


def marks_digest(node):
    """FNV-1a over every node's `marked`, pre-order.  Order is part of the
    digest, so two ports must also agree on the tree's shape."""
    h = 0xCBF29CE484222325
    stack = [node]
    while stack:
        cur = stack.pop()
        h = ((h ^ (cur.marked & 0xFF)) * 0x100000001B3) & MASK64
        for child in reversed(children(cur)):
            stack.append(child)
    return h


def scan_no_reset(re_, s):
    """`match_` without the `reset`, so the marks survive to be digested."""
    if not s:
        return re_.empty
    result = re_.shift(s[0], 1)
    for c in s[1:]:
        result = re_.shift(c, 0)
    return result


def plant(s, i, n):
    """Two `a`s exactly `n + 1` apart is the whole regex."""
    out = bytearray(s)
    out[i] = ord("a")
    out[i + n + 1] = ord("a")
    return bytes(out)


def verify(length, n):
    bad = 0
    answers = 0
    bit = 0
    for build, s, want in VECTORS:
        got = bool(match_(build(), s.encode()))
        if got:
            answers |= 1 << bit
        bit += 1
        if got != want:
            bad += 1
            sys.stderr.write("verify FAIL vector %d: input %r got %s want %s\n"
                             % (bit - 1, s, got, want))

    # The four benchmark-scale cases.  `plant` positions: the front, the middle,
    # and hard against the end -- the last one can only match if the matcher
    # read the final byte.
    s = nonmatching(length, n)
    root = bench_regex(n)
    for candidate in (s, plant(s, 0, n), plant(s, length // 2, n),
                      plant(s, length - 1 - (n + 1), n)):
        if bool(match_(root, candidate)):
            answers |= 1 << bit
        bit += 1

    # The marks after the benchmark input, before `reset` clears them.
    scan_no_reset(root, s)
    marks = marks_digest(root)
    root.reset()

    sys.stderr.write("verify vectors %d/%d\n" % (len(VECTORS) - bad, len(VECTORS)))
    sys.stdout.write(
        "verify nodes=%d input_fnv1a=%016x head=%s tail=%s answers=%016x marks=%016x\n"
        % (count_nodes(root), fnv1a64(s), s[:64].decode(), s[-64:].decode(), answers, marks)
    )
    return 0 if bad == 0 else 1


# -------------------------------------------------------------------- driver

def main(argv):
    args = [x for x in argv[1:] if not x.startswith("--")]
    flags = {x for x in argv[1:] if x.startswith("--")}
    if "--verify" in flags:
        if len(args) < 1:
            sys.stderr.write("usage: marked.py --verify <length> [n]\n")
            return 2
        return verify(int(args[0]), int(args[1]) if len(args) > 1 else 20)
    if len(args) < 2:
        sys.stderr.write(
            "usage: marked.py <length> <repeats> [n] [--short-circuit]\n"
            "       marked.py --verify <length> [n]\n"
        )
        return 2
    length = int(args[0])
    repeats = int(args[1])
    n = int(args[2]) if len(args) > 2 else 20
    if length <= 0 or repeats <= 0:
        sys.stderr.write("length and repeats must both be positive\n")
        return 2
    if "--short-circuit" in flags:
        _short_circuit_bodies()

    root = bench_regex(n)
    nodes = count_nodes(root)
    s = nonmatching(length, n)

    if match_(root, s):
        sys.stderr.write(
            "the benchmark input matched: it is supposed NOT to, and a matching "
            "input lets the scan stop early\n"
        )
        return 1

    match_(root, s)  # untimed round

    rates = []
    sink = 0
    for r in range(repeats):
        t0 = time.perf_counter()
        hit = match_(root, s)
        t1 = time.perf_counter()
        sink += 1 if hit else 0
        rate = length / (t1 - t0)
        rates.append(rate)
        sys.stderr.write("round %d %.0f\n" % (r + 1, rate))
    if sink != 0:
        sys.stderr.write("a timed round reported a match; the number is not valid\n")
        return 1

    median = sorted(rates)[len(rates) // 2]
    variant = "and/or" if "--short-circuit" in flags else "&/|"
    sys.stderr.write(
        "detail nodes=%d variant=%s node_visits_per_s=%.0f sink=%d "
        "input_fnv1a=%016x impl=%s %s\n"
        % (nodes, variant, median * nodes, sink, fnv1a64(s),
           sys.implementation.name, ".".join(str(v) for v in sys.version_info[:3]))
    )
    sys.stdout.write("python %.0f\n" % median)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
