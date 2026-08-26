# `comparisons/` — the foreign-language rows

"A JIT for Regular Expression Matching" (PyPy, 2010) ends with a table of seven
implementations of the same marked-regex matcher on the same benchmark. The
crate above this directory has three of them: pure Python on CPython, an
ahead-of-time-compiled matcher (Rust `--release` standing in for the post's
RPython-translated-to-C), and the JIT. This directory is the rest of the table,
so the example is not asking anyone to take the missing rows on trust.

| the post's row | 2010 figure | here |
|---|---:|---|
| pure Python | 12,200 chars/s | `marked.py` |
| Google re2 | 550,000 | `re2.py` — not installed on this machine, see below |
| RPython translated to C | 720,000 | the crate's `interp.rs` row, via `cargo run` |
| C++ (Sebastian Fischer) | 750,000 | `marked.cpp` |
| Java (Baltasar Trancon y Widemann) | 1,920,000 | `Marked.java` |
| CPython `re` module | 2,500,000 | `re_module.py` — **not comparable**, see below |
| RPython + JIT | 16,500,000 | the crate's `jit_interp.rs` row, via `cargo run` |

The 2010 column is from an Intel Core 2 Duo P8400 at 2.26 GHz. It is quoted for
shape, never as a comparison: sixteen years of hardware sit between it and any
number this directory prints, and dividing one by the other means nothing. The
quantity that *does* travel across machines is a ratio taken within one run —
the post's own JIT-over-no-JIT is 16,500,000 / 720,000 = 22.9x — which is why
the crate reports its ratio and this directory reports rows all measured in the
same run on the same machine.

## Run it

```sh
majit/examples/regex/comparisons/run.sh
```

Runnable from any directory. Defaults are the post's benchmark: 1,048,576
characters, 5 timed runs per row, `n = 20`. Override as
`run.sh <length> <repeats> <n>`. `KEEP=1 run.sh` leaves the scratch directory in
place and prints where it is.

`run.sh` compiles what this machine can compile into a `mktemp -d` scratch
directory — nothing is ever written inside the repository — **verifies every row
before timing it**, runs each row that can run, prints a reason for each row that
cannot, and reports per row:

* **min / median / max** over the timed runs, because one run of a benchmark is
  not a measurement; and
* the **1-minute load average before and after that row**, with a `!` and a
  `PROVISIONAL` banner on anything taken above `MAXLOAD`, because a number
  without its load is not a measurement either. See *Measurement discipline*
  below.

Nothing is installed. A missing toolchain is a skipped row with a reason and a
command, never an automatic `brew install`. Environment knobs: `MAXLOAD` (the
load above which a row is stamped), `PYTHON`, `KEEP`.

The two Rust rows are not built here; they need the crate:

```sh
cargo run -p regex --release --no-default-features --features dynasm
```

## What makes these rows comparable

Every port is a transcription of the crate's `src/interp.rs`, not a rewrite:

* the same five node kinds and the same `NodeRec` field set;
* the same `shift` arms in the same order, including `Sequence` reading
  `left.marked` — the mark from the **previous** character — before the
  recursive call overwrites it;
* `empty` computed once while the tree is built, never recomputed per character;
* `&` and `|`, never `&&` and `||`. Part 2 of the blog series makes that
  substitution so the loop body carries no short-circuit branch, and a port that
  used the short-circuiting operators would be doing strictly less work than the
  rows beside it. (`marked.py --short-circuit` restores `and`/`or` if you want
  to price the substitution; it is not what the reported row runs.);
* the tree built **balanced**, exactly as `regex.rs::build_balanced` builds it —
  association does not change the language, only the depth, and a differently
  associated tree is a different memory layout for the same answers;
* the same input: the LCG of `regex.rs::nonmatching` with the same constants and
  the same left-to-right fixup, which clears every pair of `a`s exactly `n + 1`
  apart and is what makes the input **not** match.

A non-matching input is the whole point. The regex matches iff some such pair
exists, a random string almost surely has one, and against a matching input a
matcher may stop early — so the per-character cost would be hidden.

Correctness is gated, not assumed. Every port has a `--verify <length> [n]`
mode, `run.sh` runs it before timing anything, and **a row that fails the gate is
never timed**. The gate is built on agreement between ports rather than
self-assessment, because a port that grades itself can only catch the mistakes
its author anticipated. Each port prints one line, and all of them must be
identical:

```
verify nodes=93 input_fnv1a=… head=… tail=… answers=… marks=…
```

* **`input_fnv1a`, `head`, `tail`** — the bytes the port actually generated: an
  FNV-1a-64 digest plus the first and last 64 characters. A port whose LCG is off
  by one wrapping multiply still produces a perfectly plausible `a`/`b` string,
  and a chars/s number taken over different bytes is not a row in the same table.
  Java's `long` is signed where the generator's is not, so this is what settles
  whether `>>>` and the wrapping multiply really reproduced the u64 arithmetic.
* **`answers`** — one bit per case of a fixed battery: the 28 vectors of
  `regex.rs::vectors()`, then four cases at the benchmark's own scale — the
  non-matching input, and the same input with a matching pair of `a`s planted at
  the front, in the middle, and hard against the end. The planted cases are what
  catch a matcher that lost an arm and answers "no match" to everything, which a
  non-matching benchmark alone cannot catch; the last one can only match if the
  final byte was read. The 28 vectors are additionally checked against their
  expected answers, so a port that agrees with the others but disagrees with
  `regex.rs` still fails.
* **`marks`** — a digest of all 93 `marked` bits left in the tree after scanning
  the benchmark input, taken *before* `reset` clears them, in a fixed pre-order.
  This is the strong one. It compares the whole state of the computation after a
  million characters, so two ports agreeing on it are doing the same work rather
  than merely arriving at the same boolean.

`re` and `re2` have neither marks nor a node tree, so they attest only to the
three input fields — which is the part that has to match for them to be rows in
the same table at all.

The gate has been shown to fail when it should, which is the only thing that
makes a passing gate mean anything. Two negative tests, both run against a
scratch copy:

| break | what the gate did |
|---|---|
| `Sequence` reads `0` instead of `left.marked` — the previous character's mark | `verify FAIL vector 26`, 20/28, the row dropped before timing |
| the tree built left-associated instead of balanced — **same language, same 93 nodes, same input, identical `answers`** | `MARKED PORTS DISAGREE`, caught by `marks` alone; nothing was timed |

The second is the one that justifies the `marks` field: every other check passed.

Two further checks run after timing:

* **The node-count control.** chars/s alone cannot say whether an optimizer
  deleted the tree walk. This can: `n = 2` is 21 nodes against `n = 20`'s 93, so
  the same matcher over the same length must come out several times faster per
  character. The ideal ratio is 93 / 21 = 4.4x; `run.sh` renders a verdict at
  2.0x, because on a loaded machine the ratio is noisy but "the walk is in the
  timed loop" is not. A ratio near 1 would mean the row is not measuring the
  marked algorithm at all. It is printed for every marked-matcher row (not for
  `re` or `re2`, which are not this algorithm).
* **The compiled code itself.** For the C++ row this was checked directly rather
  than inferred. At `-O2` on this machine `shift` is a real 60-instruction
  function: `ldrb` at offsets 0, 1, 2 and 3 of `Node` (`kind`, `ch`, `empty`,
  `marked`), `ldr` at offsets 8 and 16 (`left`, `right` — so every tree edge is
  still a pointer load, as it must be for a tree built at runtime), exactly one
  `strb` to offset 3, and five recursive `bl`s to itself — `Alternative` twice,
  `Repetition` once, `Sequence` twice. Nothing was deleted and nothing was
  folded. To reproduce:

  ```sh
  clang++ -O2 -std=c++17 -o /tmp/marked marked.cpp
  objdump -d /tmp/marked | awk '/<__ZL5shiftP4Nodexx>:/{f=1} f&&/^$/{exit} f'
  ```

Each program also refuses to report at all if a timed round comes back "match",
and accumulates its answers into a `sink` that it prints.

The `cpp` row and the crate's Rust `interp` row are the same algorithm through
two different optimizing compilers, so on the same machine under the same load
they should land within a few percent of each other. Measured back to back on
this machine they did (2.70M vs 2.70M, 2.77M vs 2.98M, 3.15M vs 2.82M chars/s
under an identical load). If one of them ever runs away from the other, the fast
one deleted work the slow one did.

## Measurement discipline

**`run.sh` records the 1-minute load average before and after every timed row and
prints it beside that row.** Any row taken above `MAXLOAD` (default 4.0) is
stamped `!` and the table carries a `PROVISIONAL` banner saying the row is a
lower bound, not a measurement.

This is not decoration. Measured here, same binary, same 2^20-character input,
minutes apart, with the load average sampled at each run:

| load average (1-minute) | C++ row, min – max over the rounds |
|---:|---:|
| 8.75 | 8,211,173 – 9,132,379 chars/s |
| 8.13 | 7,266,570 – 7,714,346 chars/s |
| ~33 | 2,695,067 – 3,145,059 chars/s |

A 3.2x distortion with nothing whatsoever changed in the program. A harness that
prints 2.7M without saying the machine was at 33 is exactly the defect this
guards against, and the guard belongs in the harness rather than in a human
remembering.

(An earlier draft of this table put the fast row at "load ~2". That was wrong and
is worth saying rather than quietly fixing: the load average for that run was
never sampled — the harness did not print it yet — and "~2" was reconstructed
from memory afterwards. The rows above are the ones where the number and its load
came out of the same run. No row in this repository should carry a load that was
not read at the time.)

**And the load average is necessary but not sufficient.** Also measured here,
same binary, same 2^20-character input, minutes apart, with the 1-minute load
average sitting at about 8 on both occasions:

| when | C++ row, min – max over the rounds |
|---|---|
| after an idle-ish stretch (9 rounds) | 8,211,173 – 9,132,379 chars/s |
| immediately after 11 s of solid CPU burn on another core (5 rounds) | 6,287,687 – 7,083,453 chars/s |

Same load average, 30% apart. Whatever the second one is — thermal headroom,
a frequency step, the scheduler putting the run on an efficiency core — the load
average did not see it, so the `!` stamp catches a machine that is busy but not
a machine that is merely warm. Treat two rows taken minutes apart as comparable
only to about ±20% on this hardware, and treat a ratio taken *within* one
process as the quantity that actually travels.

`MAXLOAD = 4.0` is **chosen, not measured**. The only two loads these rows have
been observed at are ~2 (undistorted) and ~33 (3.2x low); the knee between them
was never measured, because measuring it means deliberately loading a machine
other people are building on. `run.sh`'s header carries the loop that would
calibrate it on an idle machine.

**No chars/s figures are recorded in this file.** Every number taken so far was
taken on a machine at a load average between 16 and 33, and writing a provisional
number into a README is how a provisional number becomes a quoted one. Run the
harness on an idle machine and read the table it prints.

## Why the `re` row is not comparable

`re_module.py` is in the table because a reader will reach for it, and it says so
in its own output. It is a measurement of a **different algorithm** on the same
input.

The marked matcher makes exactly one `shift` per input character, visits all 93
nodes each time, and therefore reads every character exactly once whatever the
answer is. `re` is a backtracking engine: it chooses its own amount of work, may
read a character many times, and may decide a non-match early. Two chars/s
figures over different amounts of work per character are not a comparison.

### How much of the input does `re` consume?

This is the question that decides whether the post's headline comparison means
anything, so here is exactly what could and could not be established.

**What was established.** Two measurements, both printed by `re_module.py` on
every run:

* `rate_flatness` — chars/s over a length sweep spanning 32x, each point the best
  of three batches of at least 30 ms (single un-warmed calls at the short end
  measure the scheduler, not the engine). Measured at 2^20 characters the sweep
  is flat to within **1.12x** across a 32x range of lengths. An engine that
  abandoned the non-match after a bounded prefix would show chars/s **rising**
  roughly in proportion to length — a 32x rise, not 1.12x. So the time is linear
  in the input: the work is proportional to the whole string, and `re` does not
  bail early on *this* pattern and *this* input.
* `reaches_last_char` — a matching pair of `a`s is planted at
  `[len-(n+2), len-1]`, so the pattern can only match by reading the final byte.
  It answers `True`. The engine reaches the end of the input.

Those two together are why the `re` row is not simply thrown out.

**What was NOT established, and cannot be from outside the engine.** Neither
measurement recovers *how many times* `re` touched each character. The marked
matcher's per-character work is exactly 93 node visits, known by construction.
`re`'s is unknown: `sre` exposes no step counter, `re.DEBUG` prints the compiled
program rather than a trace, and `sys.settrace` does not reach C. A pattern like
this one can revisit a position many times over. So "linear in the length" is not
"the same work per character", and

> **the `re` row is a caveat, not a ratio.** It is the same input and a different
> algorithm doing an unmeasured multiple of the work per character. Dividing it
> by, or into, any other row in this table produces a number that means nothing.

The one thing that would settle it is an instrumented build of `_sre` that counts
opcode dispatches per input position; nothing installed here can do that.

One consequence worth stating: the row uses `fullmatch`, which is the operation
the marked matcher performs — "is the whole string in the language?". `search`
would be wrong twice over. It answers a different question, and on this pattern
it is quadratic — `re_module.py` measures that rather than asserting it
(`search_scaling`), and the chars/s roughly halves every time the length doubles.
That is the leading `(a|b)*` being retried from every start position, which is
precisely the backtracking the marked algorithm exists to avoid, and it is why
`search` at 2^20 characters would take hours.

## Why re2 is missing

Google re2 is not installed on the machine this harness was written on: there is
no `libre2`, no re2 headers, and no importable `re2` Python module, and this
harness installs nothing. `re2.py` prints

```
re2 unavailable: pip install google-re2
```

and exits 0, so the row shows as absent rather than as a zero. That one command
is the whole fix; rerun `run.sh` afterwards and the row fills itself in.

re2 is the interesting missing row, and the only one of the seven that is
algorithmically the same *kind* of thing as the marked matcher — an automaton
engine, bounded work per character, whole string read. Its chars/s would be
comparable in kind, where `re`'s is not.

(`re2.py` is careful about one trap worth knowing: a file named `re2.py` puts its
own directory first on `sys.path`, so `import re2` finds *itself*. It imports
cleanly and fails much later as `module 're2' has no attribute 'compile'` — a
bogus reason for a skipped row that reads like a broken binding. The file drops
its own directory from `sys.path` before importing.)

## Java: written, never compiled, never run

**No number should be quoted from `Marked.java` until it has run once.** The
machine this harness was written on has no JDK. `/usr/bin/java` and
`/usr/bin/javac` exist, but they are the macOS stubs — `command -v` finds them
and `/usr/libexec/java_home` finds nothing behind them — and a filesystem sweep
for any bundled JVM (Xcode, Android tooling, an app-embedded runtime, SDKMAN,
Homebrew) found none. Nothing is installed by this harness, so the Java row is
**unexecuted**: the file has never been through a compiler, and its self-check
has never run.

What *was* established without a JVM, and what was not:

* The generator's constants and shift operator were extracted mechanically from
  all six sources and compared, not eyeballed: every one carries
  `6364136223846793005` and `1442695040888963407`, and Java is the only one
  spelling the shift `>>> 33` rather than `>> 33`, which is what it must be for
  a signed `long` to reproduce an unsigned shift.
* Java's `long` semantics — 64-bit two's-complement wrapping multiply and add,
  logical `>>>` — were **simulated** and reproduce the Rust generator's digest
  exactly at 64, 4096 and 1,048,576 characters. That is a simulation of Java's
  arithmetic, not an execution of this Java file, and it says nothing about
  whether the file compiles.
* Everything else about the Java row is unverified.

`run.sh` is built so the first run proves it rather than trusting it: `javac` is
tested for being a real compiler and not a stub, `--verify` runs before any
timing, and the row is skipped unless its line matches the other ports character
for character. One command changes this:

```sh
brew install --cask temurin
```

`Marked.java` gives the JVM `WARMUP_ROUNDS = 5` untimed rounds before the first
timed one. HotSpot decides what to compile from invocation and back-edge
counters: the first pass through `matches` runs interpreted, the long inner loop
is then replaced on-stack, and only after `matches` and `shift` have been entered
enough times does the whole nest get compiled with a settled profile. Timing
round 1 would report that transition rather than the steady state. Five rounds at
2^20 characters is on the order of a hundred million `shift` invocations, far
past every tier threshold — and the per-round rates are printed so a reader can
confirm they have stopped climbing instead of taking it on trust. If the timed
rounds are still rising, raise the constant: the number was warm-up.
