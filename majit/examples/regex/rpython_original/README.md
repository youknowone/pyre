# `rpython_original/` — the post's matcher, through RPython's own JIT

This is the comparison the example owes. "A JIT for Regular Expression Matching"
(PyPy, 2010) describes an RPython program and the trace RPython's JIT produces
for it; the crate above reimplements that program in majit. The only way to say
whether the reimplementation is faithful is to run the original and diff the two
traces — which is what this directory does, against the RPython checkout that is
already at the root of this repository.

## Run it

```sh
pypy majit/examples/regex/rpython_original/runner.py 20 4096
```

Runnable from any directory, and nothing has to be installed or exported:
`runner.py` walks up from its own path to find the `rpython/` checkout at this
repository's root and puts it on `sys.path` itself.

**It must be `pypy`, not `python3`.** RPython's toolchain is Python 2 and these
files are written in it (any Python 2 with the checkout importable works).
Running it under Python 3 prints the `pypy` command line to use and exits 2,
rather than failing with a `print`-statement `SyntaxError` that names the wrong
problem.

Arguments are `n` (the regex's `{n}` repetition count, default 20) and the input
length (default 4096). `--listing` additionally prints the peeled body op by op.

Nothing is translated and nothing is installed. `runner.py` uses `LLJitMixin`,
the in-process harness RPython's own JIT tests use: real metainterpreter, real
optimizer, LLGraph backend. The loop that comes out is the trace RPython would
compile, without a multi-hour translation.

## Reading its output

```text
=== result: 0 (0 == did not match, which is the benchmark) ===
=== 1 loop(s)/bridge(s) compiled ===

--- trace 0: 306 ops total, 153 in the peeled body ---
  debug_merge_point            2
  getfield_gc_i               24
  guard_false                 21
  ...
```

* **`result: 0`** — the matcher's answer, and it is supposed to be 0. The
  benchmark input is generated *not* to match, because a matcher may stop early
  on a match and then the per-character cost is hidden. A `1` here would mean
  the generator drifted, not that something got faster.
* **`1 loop(s)/bridge(s) compiled`** — the whole scan is one trace. The regex
  is the JitDriver's green, so every character shares one green key. More than
  one would mean the tree stopped being constant to the tracer.
* **`306 ops total, 153 in the peeled body`** — a compiled loop is preamble plus
  peeled body. The preamble runs once; the **peeled body is what runs per input
  character**, and it is the only half worth grading. Reading 306 would charge
  the body for reads the preamble hoisted out.
* **the census** — the peeled body by op name. Four lines carry the post's
  claims:
  * **`getfield_gc_r` absent (0)** — not one pointer read. The tree has 92
    edges and the trace follows none of them: they folded against the constant
    regex. This is the headline.
  * **`setfield_gc 93`** — one mark stored per node, for a 93-node tree. So the
    whole tree is in the loop; nothing was deleted to get the pointer reads to
    zero.
  * **`int_eq 2`** — two comparisons for 46 `Char` nodes. That is the subset
    construction, performed by the tracer: a node whose incoming mark the
    optimizer proved constant zero has nothing to compare, so its comparison is
    not in the loop at all.
  * **`guard_true` + `guard_false` (27)** — the short-circuit branches. The
    post's source is `and`/`or`, which really do stop early, so the tracer turns
    each into a guard. This is the number that identifies which majit portal is
    the faithful port.

None of that says whether **majit** reproduced it — for that the numbers have
to sit next to majit's. They do, below, and the comparison is asserted rather
than eyeballed.

## What is here

* **`marked.py`** — the post's matcher, in RPython. The class hierarchy
  (`Regex`, `Char`, `Epsilon`, `Binary`, `Alternative`, `Repetition`,
  `Sequence`), `_immutable_fields_ = ['empty']` with `marked` left mutable, and
  the JitDriver whose green is the regex and whose reds are the position and the
  accumulated result. `shift` is written with `and`/`or`, which is how the post
  writes it.
* **`fixture.py`** — the benchmark: `bench_regex(n)` building
  `(a|b)*a(a|b){n}a(a|b)*` balanced, and `nonmatching(length, n, seed)` — the
  same LCG, the same constants, the same left-to-right fixup as the crate's
  `regex.rs`.
* **`runner.py`** — `meta_interp` plus the census.
* **`marked_masking.py`**, **`fixture_masking.py`** — the same two modules with
  part 2's `&`/`|` in place of part 1's `and`/`or`. GENERATED: `make_masking.py`
  writes them from `marked.py` / `fixture.py` and refuses if any of its five
  substitutions no longer matches, so the pair stays an A/B of the operators and
  of nothing else. `make_masking.py --check` exits non-zero when they are stale;
  it is plain text processing and runs under Python 2 or 3.
* **`target.py`**, **`target_masking.py`** — RPython translation targets, one
  per spelling. These are the only thing on this side that can answer how fast
  a trace *runs*; see below.

## The input is pinned on both sides

A trace census taken over different bytes is not a comparison. `fixture.py`
carries

```python
NONMATCHING_4096_FNV1A = r_uint(0xd9f7f62ad250969e)
```

and `regex.rs` carries the same constant as `NONMATCHING_4096_FNV1A`, each
asserted by a test on its own side. The two matchers scan the same 4096 bytes.

The LCG is spelled with `r_uint` rather than Python's arbitrary-precision ints,
because RPython's annotator rejects a prebuilt `(1 << 64) - 1` mask outright —
and `r_uint` is in any case the truer match for Rust's `wrapping_mul`.

## The result

Peeled body — everything from the last `label` on, which is what runs per input
character. Two tree sizes, because a census that does not move with the tree is
not measuring the tree:

| `n = 20`, 93 nodes | RPython JIT | majit `shortcircuit` | majit `jit_interp` |
|---|---:|---:|---:|
| `getfield_gc_r` | 0 | 0 | 0 |
| `getfield_gc_i` | 24 | 24 | 24 |
| `setfield_gc` | 93 | 93 | 93 |
| `int_eq` | 2 | 2 | 2 |
| guards | 27 | 26 | 1 |
| total | 153 | 176 | 194 |

| `n = 2`, 21 nodes | RPython JIT | majit `shortcircuit` | majit `jit_interp` |
|---|---:|---:|---:|
| `getfield_gc_r` | 0 | 0 | 0 |
| `getfield_gc_i` | 6 | 6 | 6 |
| `setfield_gc` | 21 | 21 | 21 |
| `int_eq` | 2 | 2 | 2 |
| guards | 9 | **9** | 1 |
| total | 45 | 51 | 50 |

Every structural count is exact. Not one pointer read survives on either side —
the 92 edges of the tree walk are gone, folded against the green regex. 93
stores is one mark per node, so the whole tree is in the loop and nothing was
deleted. And `2 int_eq` for 46 `Char` nodes is the subset construction itself,
performed identically by both tracers: a node whose incoming mark the optimizer
proved constant zero has nothing to compare, so its comparison is not in the
loop at all.

The loop tails are the same ops in the same order — `int_add`, `setfield_gc`,
`int_lt`, `guard_true`, `jump` — with RPython additionally carrying a
`debug_merge_point`, which costs nothing.

The 21-node row is the control: the store count tracks the node count on all
three, so the census is measuring the tree and not a constant. It also lands
the guard counts exactly on top of each other, 9 against 9 — at this size the
branching portal is not merely close to RPython's shape, it *is* it.

## This comparison is a gate, not a paragraph

Both tables are asserted, not just written down:

```sh
cargo test -p regex --no-default-features --features dynasm \
    -- --nocapture the_peeled_body_matches_the_rpython_original
```

It prints the three columns side by side and fails if they part. RPython's
column is a recorded constant inside the test — a Rust test cannot call a
Python 2 — with the `runner.py` command to re-derive it written beside it.

The four structural counts are asserted **exactly**, on both portals and at
both sizes, because those are the post's structural claims. majit's own two
numbers — the branching body's guard count and its total — are asserted exactly
as well, against recorded values rather than a band around RPython's.

### The gate has been shown to fail

A passing gate means nothing until it has been watched to fail. Three breaks,
each applied and then reverted:

| break | what the gate did |
|---|---|
| the portal's `promote(root)` removed, so the tree stops being constant to the tracer | `getfield_gc_r` 0 → **84**; the structural assertion fired |
| `Sequence` reads `left.marked` *after* the recursive shift instead of before | `getfield_gc_i` 24 → **1**; caught here and by six other tests, since it also changes the answer |
| the `Char` arm masks (`mark & (ch == c)`) instead of branching — **same answers, same specialization, same node count** | guards 26 → **28**, total 176 → **180** |

The third is why the guard assertion is not a band. Written first as "within
one of RPython's 27", it admitted 26, 27 *and* 28 — so the entire suite passed
with a `Char` arm silently switched to the spelling the other portal is
supposed to own, which is the exact difference this example exists to measure.
Recording majit's own numbers closed it.

## What this settled about the crate's two portals

The post's source writes `shift` with `and`/`or`; part 2 of the series then
remarks that `&`/`|` is the better spelling because it keeps short-circuit
branches out of the loop body. The crate has both. Until this comparison ran,
its documentation had them backwards.

RPython's trace carries 27 guards (`guard_true` 6 + `guard_false` 21). The
crate's `shortcircuit.rs` carries 26; `jit_interp.rs` carries 1. So
**`shortcircuit.rs` is the faithful port of the post's own source**, and
`jit_interp.rs` is the adapted variant the remark asks for — an A/B of the
remark rather than a second copy of the post. Both module docs now say so.

## Where the 23 remaining ops come from

153 against 176 is the port's integer typing, and it accounts for every one of
the 23.

RPython carries the marks as `Bool`. A `Bool` local *is* a branch condition, so
`flatten.py` emits a plain `goto_if_not` and `pyjitpl.py opimpl_goto_if_not`
hands that box straight to `generate_guard` — no truth test — and a `Bool` field
is stored as it stands, with no mask.

Our `NodeRec.marked` is a `u8` and `shift` carries marks as `i64`, so both
conversions become real operations:

* `if mark != 0` is an `IntIsTrue` — **24**, exactly one per guard whose
  condition is not already a comparison (26 guards, less the loop exit's
  `IntLt` and one `SetfieldGc`-fed guard); and
* `n.marked = m as u8` is an `IntAnd(m, 255)` — **2** survive, on the two live
  `Char` marks.

24 + 2, less RPython's one extra `guard_false` and its two zero-cost
`debug_merge_point`s, is 23.

This is not a place majit falls short of RPython. `rewrite.py
optimize_INT_IS_TRUE` folds the test only when the argument's bounds are
`is_bool()`, and a one-byte unsigned field read is [0, 255] — upstream's own
`FieldDescr.get_integer_min` / `get_integer_max` answer the same for
`lltype.Bool`, which is `FLAG_UNSIGNED` at size 1. The fold arm is ported and
fires elsewhere; it has nothing to fire on here. The two `IntAnd` are the same
story from the other end: `autogenintrules.py`'s `and_x_c_in_range` would remove
`int_and(x, 255)` for an `x` bounded by [0, 1], but the `IntEq` producing that
`x` is a postponed pure op forced out *after* its consumer was optimized, so
`make_bool` had not run on it yet. Upstream never meets the pattern, because it
never masks a `Bool`.

To read the two bodies side by side:

```sh
pypy majit/examples/regex/rpython_original/runner.py 20 4096 --listing
REGEX_LISTING=1 cargo test -p regex --no-default-features --features dynasm \
    -- --nocapture the_branching_body_trades_ops_for_guards
```

The two listings are formatted differently — RPython's prints op names, majit's
prints the whole operation — so they are read together, not `diff`ed.

## Speed, measured on both sides

The census above is the trace-shape comparison and it is exact. This is the
other half: how fast the two implementations actually scan the same 1,048,576
characters against the same 93-node tree.

`meta_interp` cannot answer it — `runner.py` runs the LLGraph backend, which
executes traces in an interpreter, so timing it would measure the harness.
Only a translated binary runs compiled traces natively, which is what
`target.py` and `target_masking.py` are for:

```sh
pypy <repo root>/rpython/bin/rpython --opt=jit target.py   # RPython + JIT
pypy <repo root>/rpython/bin/rpython --opt=2   target.py   # translated to C
```

This README used to say that no such measurement had been taken and that each
translation was a multi-hour build. Both halves were wrong: `--opt=jit` took
**204s** and `--opt=2` **41s** here, so all four binaries — two spellings times
two optimization levels — are a five-minute build.

### The measurement

Three rounds, each running every binary once in the same loop so a load spike
moves all six rows together rather than one of them. Each row is the median of
five timed runs after one untimed warm-up, over 1,048,576 characters at `n =
20`; the table is the median of the three rounds. Machine at 1-minute load 19
to 26 throughout.

| 1,048,576 chars, 93 nodes | majit | RPython `--opt=jit` | RPython `--opt=2` (C) |
|---|---:|---:|---:|
| `&`/`\|` — `jit_interp.rs` / `target_masking.py` | **42,548,721** | **42,729,349** | 6,319,695 |
| `and`/`or` — `shortcircuit.rs` / `target.py` | 135,690 | 555,383 | 4,514,372 |

Two readings, and they are different readings.

**On the spelling the post reports, the two JITs are the same speed.** The
post's own numbers come from the adapted `&`/`|` matcher — part 2 says so
outright — and on that row majit reads 1.00x of RPython. Its headline ratio is
the same too: 42,548,721 / 4,514,372 = **9.4x** against RPython's own
42,729,349 / 4,514,372 = **9.5x**, both against the post's 16,500,000 / 720,000
= 22.9x on 2010 hardware.

**On the unadapted spelling both JITs lose to their own C, and majit loses
harder.** RPython's `and`/`or` JIT is **8.2x slower than RPython translated to
C** (555,383 against 4,514,372). So "if you don't change the `and` and `or`
... it's not particularly fast" is not a majit artifact and not a Rust
artifact — it is a property of the spelling, and upstream pays it in the same
direction and the same order of magnitude. majit pays it 4.1x harder than
upstream does, which is the one gap in this table that is majit's own.

`PYPYLOG=jit-summary:-` on the `and`/`or` build at 262,144 characters says
where upstream's time goes: 1 loop, **1185 bridges**, `Tracing 0.145s` +
`Backend 0.107s` out of `TOTAL 0.847s`. Even upstream spends 30% of that run
compiling, and 1185 is a rate rather than a ceiling — `trace_eagerness` is 200,
so a pass of `n` characters grows at most about `n / 200` bridges however many
distinct mark patterns the tree has. majit is rate-limited by the same
parameter and grows 84 over 32,768 characters.

### One caveat about the majit column

`matches()` builds a fresh `JitDriver` per call, so each of majit's five timed
runs re-records and re-compiles; `target_masking.py` calls a module-level
`jitdriver` and pays that once, in the warm-up. It shows: across the three
rounds RPython's four rows move by 8% and majit's masking row moves by 2.5x
(42.5M, 16.9M, 45.2M), which is compile-time variance rather than scan-time
variance. The median is the row above; the low round is what a contended
compile does to it. Every majit portal in the tree builds its driver per call,
so this is a majit convention rather than something this example chose.

### The majit-only ratio

The post's headline ratio is also measured on the majit side alone, in one
process, and that one runs in the normal suite:

```sh
cargo test -p regex --release --no-default-features --features dynasm \
    -- --nocapture the_jit_is_worth_several_times
```

```text
[perf] 1048576 chars, majit JIT : 36530817 chars/s (min 28390215, max 37312110)
[perf] 1048576 chars, no JIT    :  6700337 chars/s (min  5680768, max  7246276)
[perf] majit JIT / no JIT = 5.5x   (the post's own: 16,500,000 / 720,000 = 22.9x)
```

That ratio is smaller than the table's 9.4x because its denominator is
`interp.rs` rather than RPython's C — and the table says those two are within
10% of each other (majit's no-JIT row reads 4,175,067 against `target.py`'s
4,514,372), so the two denominators are comparable and the two ratios are
measuring the same thing at different lengths and loads. `--release` matters: a
debug run reads about 9.0x because the denominator is unoptimized, and the test
prints a banner saying so rather than letting that number be quoted.

Cranelift reads 2.6x in the same conditions. The two backends agree op for op
on every census above and differ only here — this row includes the one
recording each call pays for, and cranelift compiles slower than dynasm.
