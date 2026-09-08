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
=== 1 loop(s), 9 bridge(s), 0 aborted (trace_eagerness=200) ===

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
* **`1 loop(s)`** — one compiled loop for the whole scan. The regex is the
  JitDriver's green, so every character shares one green key; a second loop
  would mean the tree stopped being constant to the tracer. Bridges are counted
  separately and a nonzero count is not that signal — see below.
* **`9 bridge(s)`, `trace_eagerness=200`** — the branching body's guards do
  fail, about once per input character, and a guard that has failed 200 times
  earns a bridge. Both numbers are load-bearing and both were wrong here until
  measured: `stats.get_all_loops()` is fed by `send_loop_to_backend`'s
  `add_new_loop` (compile.py:550) and by nothing else, so it never counted a
  bridge, and `warmspot.py:112` pins `set_param_trace_eagerness(2)` "for
  tests", at which this run compiles **1407** bridges rather than 9. `runner.py`
  now reads `stats.compiled_count`, which counts both (compile.py:552,
  compile.py:604), and pins the eagerness to the `rlib/jit.py` PARAMETERS
  default of 200 that a translated PyPy and majit both use.
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
| total | 153 | 150 | 194 |

| `n = 2`, 21 nodes | RPython JIT | majit `shortcircuit` | majit `jit_interp` |
|---|---:|---:|---:|
| `getfield_gc_r` | 0 | 0 | 0 |
| `getfield_gc_i` | 6 | 6 | 6 |
| `setfield_gc` | 21 | 21 | 21 |
| `int_eq` | 2 | 2 | 2 |
| guards | 9 | **9** | 1 |
| total | 45 | 43 | 50 |

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

## The bridges are RPython's too

The census above grades the loop body. It says nothing about what happens when
that body's guards *fail*, and on the branching portal they fail constantly —
about once per input character, at every length, never settling. Bridges
accumulate and the rate never converges. Read alone that is the signature of a
bridge cascade: bridges that never rejoin the loop, so every one spawns the
next.

It is not a majit defect. RPython's JIT does the same thing on the same body:

| `n = 20`, `trace_eagerness = 200` | RPython | majit |
|---|---:|---:|
| loops, 4096 chars | 1 | 1 |
| bridges, 4096 chars | 9 | 10 |
| deopts per character, 4096 chars | 0.9990 | 0.9961 |
| bridges, 16384 chars | 41 | 38 |
| deopts per character, 16384 chars | 0.9998 | 0.9973 |
| distinct guards that ever failed, 16384 chars | 388 | 344 |

Within a run majit attaches 38 bridges over 16384 characters and a bridged
guard then fails **zero** more times — it is patched into its bridge and never
returns to the frontend. So the residual rate is not bridges failing to take.
It is the guards that have not earned one yet, and their shape is the point:
306 unbridged coordinates, 8735 failures between them, **median 10** and a
maximum of 190. Only 8 are anywhere near the 200 the counter wants; 147 fired
fewer than ten times.

That is a tree fanning out faster than the counter can close it, not a queue
draining. Each bridge is a new trace carrying its own guards, those guards
carry the next ones, and the input ends long before the frontier is covered.
RPython's 388 distinct failing guards against 41 bridges is the same shape at
the same scale. The body has 27 guards; neither JIT saturates what is behind
them.

Two measurement bugs on this side had to be fixed before the comparison could
be made at all, and both of them made majit look broken when it was not:

* `runner.py` printed `len(stats.get_all_loops())` under the label
  "loop(s)/bridge(s) compiled". That list is appended by
  `send_loop_to_backend` (compile.py:550) and by nothing else, so it never held
  a bridge. It reported **1** for a run that compiled nine, which reads as
  "RPython needs no bridges here" — the opposite of what happens.
* The harness ran at `trace_eagerness = 2`, pinned by `warmspot.py:112` "for
  tests", against the `rlib/jit.py` PARAMETERS default of 200 that a translated
  PyPy and majit both use. At 2 this run compiles **1407** bridges over 4096
  characters. Compared against majit's 10 that reads as majit refusing to
  bridge — again the opposite.

`runner.py` now reads `stats.compiled_count`, which counts loops and bridges
both (compile.py:552, compile.py:604), and pins the eagerness to 200.

The translated build had been saying this all along, further down this file: a
`PYPYLOG=jit-summary` run of the `and`/`or` target reports 1 loop and **1185
bridges** over 262,144 characters — one per 221 — against the in-process
runner's claim of none at all. Two measurements of the same program disagreeing
by 1185 was the thing to chase, and it was not chased, because one of them was
printed under a label that made it look like an answer.

This table is asserted too, by `the_branching_portal_bridges_like_the_rpython_original`
in `shortcircuit.rs`, with RPython's column recorded beside the command that
re-derives it. The band is deliberately loose — half to double on the bridge
count — because which guards cross the 200th failure before the input ends is a
boundary effect, not behaviour. The deopt rate is the tight one. Both halves
have been watched to fail:

| break | what the gate did |
|---|---|
| the masking portal substituted for the branching one — 0 bridges, 1 deopt | the rate assertion fired: `deopted 0.0002 times per character ... RPython deopts 0.9990` |
| RPython's recorded bridge count moved 9 → 100 | the band assertion fired: `majit grew 10 bridge(s) ... where RPython grows 100` |

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
| the portal's `root` ref green removed, so the tree stops being constant to the tracer | `getfield_gc_r` 0 → **84**; the structural assertion fired |
| `Sequence` reads `left.marked` *after* the recursive shift instead of before | `getfield_gc_i` 24 → **1**; caught here and by six other tests, since it also changes the answer |
| the `Char` arm masks (`mark & (ch == c)`) instead of branching — **same answers, same specialization, same node count** | guards 26 → **28**, total 150 → **154** |

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

## Why the totals differ by three

153 against 150 is fully accounted for: RPython's listing contains two
zero-cost `debug_merge_point` operations that majit's IR does not represent,
and RPython records one additional guard.

RPython carries the marks as `Bool`. A `Bool` local *is* a branch condition, so
`flatten.py` emits a plain `goto_if_not` and `pyjitpl.py opimpl_goto_if_not`
hands that box straight to `generate_guard` — no truth test — and a `Bool` field
is stored as it stands, with no mask.

`NodeRec.empty`, `NodeRec.marked`, and the traced `shift` now preserve that
Bool source type. They still travel through majit's integer register bank and
one-byte integer field descriptors, exactly as `lltype.Bool` does, but the
former 24 `IntIsTrue` and two narrowing `IntAnd` operations are gone.

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
harder.** RPython's `and`/`or` JIT is **8.1x slower than RPython translated to
C** (555,383 against 4,514,372). So "if you don't change the `and` and `or`
... it's not particularly fast" is not a majit artifact and not a Rust
artifact — it is a property of the spelling, and upstream pays it in the same
direction and the same order of magnitude. majit pays it 4.1x harder than
upstream does, which is the one gap in this table that is majit's own — read
that figure with the re-measurement below, which puts it at the top of a
3.0-4.1x range.

`PYPYLOG=jit-summary:-` on the `and`/`or` build at 262,144 characters says
where upstream's time goes: 1 loop, **1185 bridges**, `Tracing 0.145s` +
`Backend 0.107s` out of `TOTAL 0.847s`. Even upstream spends 30% of that run
compiling, and 1185 is a rate rather than a ceiling — `trace_eagerness` is 200,
so a pass of `n` characters grows at most about `n / 200` bridges however many
distinct mark patterns the tree has. majit is rate-limited by the same
parameter and grows 84 over 32,768 characters.

#### Re-measured 2026-09-01: the ratio is 3.0-4.1x, and the absolutes do not travel

The table above was one sitting. Six same-round pairs taken on another --
three at 1-minute load 4.0-4.6 and three at 5.9-9.2, every pair being the two
binaries run back to back so one load spike moves both -- put the `and`/`or`
ratio at **3.02, 3.22, 3.45, 3.56, 3.58 and 4.11**, median about 3.5x. The
table's 4.09x (555,383 / 135,690) is the top of that range, so whatever the
allocation work bought is inside this instrument's resolution rather than
clearly above it. Claiming a speedup from it would not be supportable.

What the six rounds do settle is that **the absolute columns are machine
state, not results.** Between the two sittings majit's `and`/`or` row moved
135,690 -> 181,717-330,438 and RPython's moved 555,383 -> 548,352-1,269,859:
both about 2x, in the same direction, from nothing but the machine. Inside the
second sitting the *same* RPython binary read 1,184,427 and then 663,658 a few
minutes later as load went 4.0 to 9.2 -- 1.78x on one arm with nothing
changed. Read every chars/s figure in this file as a ratio against the number
printed beside it in the same round, never against a number from another day.

Three things did reproduce. RPython's two `--opt=2` columns come back within
1.11x and 1.13x of what is tabulated, and its JIT is 6.7x slower than its own
C on this spelling against the 8.1x recorded -- so "both JITs lose to their
own C, and majit loses harder" survives. The masking row read 0.88x, 1.61x and
1.08x of RPython, but RPython's own masking column swung 2.02x across those
same rounds, so that round resolves nothing finer than "consistent with the
parity the table reports". And the allocation census, which does not depend on
the machine at all, reproduces to the digit: 11.4 allocations and 1,360.4
bytes per input character. That is why the census is the instrument this gap
is graded with.

#### Re-measured 2026-09-03: 2.3-2.5x, after the per-bridge retention fixes

The RSS runs in this subsection used the macOS system allocator, a release
`dynasm` build with `alloc-census` and `fast-alloc` both off, and
`/usr/bin/time -l`'s `maximum resident set size`.  The measured arm was run as
`PYRE_REGEX_LENGTHS=1048576 PYRE_REGEX_ROWS=2 target/release/regex`; bridges
were on unless the control explicitly set `MAJIT_NO_BRIDGE=1`.  The 139 MiB
quoted in the PR summary was the intermediate reading immediately after the
RWX-arena change.  The 124 MiB below is the later reading after the remaining
descriptor/cell retention fixes, so it is the final same-configuration number
rather than a conflicting measurement.

With bridges on, majit's heap grew with the input -- 1,138 MiB peak RSS at
1,048,576 characters against 15 MiB with `MAJIT_NO_BRIDGE=1` -- and the
`and`/`or` row followed it: 10.5x slower per character at 1M than at 4K.
Three causes, each a majit-only retention with no upstream counterpart: a
`CompiledTrace` kept per bridge in the frontend (`send_bridge_to_backend`
keeps nothing), a page per compiled block where `asmmemmgr.py` packs blocks
into 1 MiB RWX mappings, and cranelift's per-descr bridge cells plus a
recovery layout rebuilt per deopt. With those gone the peak is 124 MiB at 1M
and the row no longer moves with length (448K / 473K / 416K chars/s at 4K /
64K / 1M in one sitting).

Three same-round pairs at 1,048,576 characters, `n = 20`, 1-minute load 4.1
to 4.7, the two binaries back to back and each row the median of five timed
runs:

| round | RPython `--opt=jit` | majit | ratio |
|---|---:|---:|---:|
| 1 | 1,075,379 | 459,039 | 2.34x |
| 2 | 1,092,345 | 463,868 | 2.35x |
| 3 | 1,127,526 | 460,087 | 2.45x |

The 3.0-4.1x above becomes 2.3-2.5x, at the same length and against a binary
translated the same afternoon. At that revision, with bridges off majit read
750K in the same sitting, so the compiled bridges cost 1.5x over running none,
and the deopt that every character takes ran 124 blackhole operations here
against RPython's 82. The 2026-09-04 follow-up below closes that operation-count
gap; bridge construction remains the structural difference.

### Where majit's share of that gap goes

#### Allocator control corrected 2026-09-05

`fast-alloc` used to select mimalloc only when `alloc-census` was also enabled.
A clean `--features dynasm,fast-alloc` timing build therefore still used System.
The feature now selects mimalloc with or without the census; it remains opt-in.
This fixes the measurement control, not the recorder representation.

Three clean System/mimalloc/mimalloc/System rounds at 262,144 characters,
`PYRE_REGEX_ROWS=2`, each process running the usual five timed samples, were
collected with `/usr/bin/time -l`.

The saved pair used the recorder at `bdf4abd130e`, before the accompanying
GC/byte-materializer repairs, and differed only in allocator selection. This
is an allocator-only control, not a timing of the completed recorder migration.

| allocator | whole-process retired instructions, median of six processes | range |
|---|---:|---:|
| System | 51,391,087,211 | 51,301,743,543–51,465,158,457 |
| mimalloc | 47,970,429,131 | 47,916,263,321–48,018,761,845 |

These are `/usr/bin/time -l` process totals, including setup and warm-up,
**not** allocations or JIT-only instruction counts. The reduction is 6.7%.
Both arms retained the 150-op body and the `0 / 24 / 93 / 2` structural counts.
All twelve processes reported 3,612 bridges and 293,782 compiled bridge ops.
Concurrent builds made wall time unusable: the six process medians ranged
125,163–275,993 chars/s for System and 131,786–284,267 for mimalloc. No wall-clock
speedup or RPython parity is established by this experiment. The live recorder
still uses `Rc<Op>` and structured snapshots; switching allocators does not
close that structural gap.

To reproduce the controls, build and save each executable separately:

```sh
cargo build -p regex --release --no-default-features --features dynasm
cargo build -p regex --release --no-default-features --features dynasm,fast-alloc
# Run each saved executable in alternating forward/reverse order:
PYRE_REGEX_LENGTHS=262144 PYRE_REGEX_ROWS=2 /usr/bin/time -l <saved-executable>
```

The subsequent GC/materializer repairs and direct live-frame guard capture
were measured against that saved mimalloc executable with the same clean
build flags, length and three ABBA rounds. Both arms used mimalloc; neither
enabled `alloc-census`. This comparison includes all those repairs, so it
does not isolate the removal of the old full-state failarg-list builder.

| mimalloc build | whole-process retired instructions, median of six processes | range |
|---|---:|---:|
| saved allocator control | 47,941,578,060 | 47,899,039,519–48,055,675,906 |
| GC/materializer repairs + direct guard capture | 47,906,746,857 | 47,880,392,131–47,942,807,794 |

The median difference is only 0.073%, with overlapping ranges: **no meaningful
performance improvement is established**. The process-median throughput ranges
were 227,697–294,767 and 255,293–289,188 chars/s respectively. All twelve
processes again had the same 150-op body, four structural counts, 3,612 bridges
and 293,782 bridge ops. Removing the redundant list API is a structural repair,
not evidence that the live byte-recorder migration or the `and/or` gap is done.

#### Recorder and resume costs

Three theories about that gap were measured and two of them were wrong, so the
numbers matter more than the reasoning:

**Not the bridge cascade.** The branching portal deopts about once per character
and never converges, but so does RPython — see "The bridges are RPython's too"
above. Refuted.

**Not the blackhole byte-interpreting callees.** That was recorded as majit's
big deviation, at 2142 blackhole ops per character against 39. Re-measured at
that revision with
`MAJIT_BH_DEBUG=1 MAJIT_GUARDLOG=1` at 1024 characters, and with RPython's own
dump counted the same way (`bh: (\w+)` lines over `runner.py 20 1024`), both
halves of that were wrong:

| per deopt | RPython | majit |
|---|---:|---:|
| blackhole ops | **82.0** | **124.3** |
| native callee call | 3.38 `residual_call_ir_i` | 3.41 `inline_call` |
| register copies | 3.09 `int_copy` | 36.08 (`move_i` + `move_i_c`) |

The native callee call is at parity. `[bh-setpos]` and `[bh-frame]` are both
33,222, so not one `inline_call` seats an interpreted frame — every one takes
the native arm. The blackhole is 1.5x, not 55x, and 1.5x cannot produce a 3-4x.

The register-copy row was a real deviation. The proc-macro lowerer allocated
every temporary monotonically, while upstream runs
`rpython/tool/algo/regalloc.py` before `CodeWriter.make_jitcode` flattens the
graph. The proc-macro path now builds the same per-bank interference graph and
uses the ported `tool/algo/color.py::DependencyGraph` before liveness encoding.
On this exact `shift` JitCode the working-register footprint falls from **36
int / 26 ref** to **4 int / 3 ref**; the blackhole therefore has fewer slots to
initialize and fewer renamings to execute. The native-entry test pins those
counts so monotonic allocation cannot return unnoticed.

#### 2026-09-04: the missing coalescing half

The footprint result did not mean the register allocator was complete. RPython
`tool/algo/regalloc.py::RegAllocator.coalesce_variables` runs between building
the interference graph and coloring it, then
`flatten.py::GraphFlattener.insert_renamings` omits a link copy when source and
target received the same color. The proc-macro adapter skipped both operations;
it also reserved the whole ABI-input prefix from temporaries instead of doing
`GraphFlattener.enforce_input_args`' post-color swaps. In the flattened adapter,
branch-join `Move{I,R,F}` operations are the link source/target pairs. They are
now coalesced in reverse control-flow order, input colors are swapped into the
ABI prefix afterwards, and identity moves are omitted.

The same 1,024-character logs, normalized by actual deopts, now read:

| per deopt | RPython | majit before coalescing | majit after |
|---|---:|---:|---:|
| all blackhole ops | 82.20 | 80.73 | **65.39** |
| register copies | 3.09 `int_copy` | 17.55 | **2.21** |

The earlier 124.3 total included the pre-coloring implementation; after the
first coloring pass it had already fallen to 80.73, but that fact had not been
re-measured. Coalescing removes another 15.34 `move_i` operations per deopt.
The total is now lower than RPython rather than higher, so blackhole opcode
count is no longer an explanation for the remaining speed gap.

Retired-instruction slope over 262,144 → 1,048,576 characters, five timed runs
per point, moved from the branch baseline's 28,274 to **27,464 instructions per
character**. That is a real but much smaller 2.9% change: most remaining
instructions are still spent constructing and optimizing bridges, not running
the copies. Against the freshly translated RPython's 11,354 the current ratio
is **2.42x**.

Repeated non-null ConstPtr operands now share the trace's upstream-shaped ref
cache; null uses the reserved inline ref-zero representation. Its same-tree A/B
at 65,536 characters was 19.0 / 2,650 → 9.9 allocations / 1,747 bytes per
character. Two ownership copies in bridge preparation were then removed: a
tentative trace retains its existing `Rc<Op>` handles across `cut` instead of
deep-cloning every operation, and snapshot maps read the live snapshot slice
instead of cloning every frame/box vector first. Those bring the same census to
**9.7 allocations / 1,649 bytes per character**. The peeled trace remains
exactly `0 / 24 / 93 / 2`.

The remaining boundary is literal rather than diagnostic: the live recorder is
still `recorder::Trace` (`Vec<Rc<Op>>` plus structured snapshots), while the
already-ported `opencoder::TraceRecordBuffer` flat operation and snapshot
buffers are not yet the `TraceCtx` owner. Completing that field swap, then
letting its `TraceIterator` be the sole operation materializer, is the remaining
RPython-shaped route to remove the per-bridge allocator traffic.

**It is bridge compilation, and allocation traffic inside it.** `/usr/bin/sample`
over a 262,144-character run, counting only the `shortcircuit::mainloop`
subtree (5,993 samples):

| | samples | share |
|---|---:|---:|
| jitdriver back-edge → MetaInterp: record + optimize | 2,633 | **44%** |
| ↳ `Optimizer::optimize_bridge` | 1,267 | 21% |
| ↳ ↳ `emit_operation` → `store_final_boxes_in_guard` → `ResumeDataLoopMemo::number` | 221 | 4% |
| `BlackholeInterpreter::resume_mainloop` | 876 | 15% |

Upstream's own `PYPYLOG=jit-summary` on this spelling reports `Tracing 0.145s` +
`Backend 0.107s` of `TOTAL 0.847s` — **30%** — while compiling **1185** bridges
over 262,144 characters. majit compiles about **half** as many per character
(38 over 16,384) and spends **more** of its time doing it, so its per-bridge
cost is roughly 3x upstream's.

What that time is made of, from the same profile's flat histogram and from
`--features alloc-census`: `malloc`/`free` symbols are the largest single bucket
(~2,800 of 17,146 samples), and `majit_ir::resoperation::Op::clone` was the
fifth hottest leaf. RPython builds its trace in the opencoder's flat buffer and
its resume data in a nursery; majit still builds both out of `Rc<Op>` and
per-guard vectors. This pass removes one common-case tax: upstream stores
`GuardResOp._fail_args` only on guards, so the unified Rust `Op` no longer
embeds three inline `Operand` slots in every ordinary operation. `Op` allocations
fall from 280 to 240 bytes: that field measures 72 bytes as a
`SmallVec<[Operand; 3]>` against 32 as a `Vec` header. The remaining allocation traffic is still the
largest majit-only cost; it is now smaller, not gone.

### Driver ownership is now the same on both sides

`marked.py` owns one module-level `JitDriver`; the old Rust wrapper built a new
driver inside every `matches()` call and discarded its compiled cell on return.
Each Rust portal now owns a persistent `Matcher { root, driver }`, with `root`
passed as a ref green instead of kept red and manually promoted. The benchmark
constructs both matchers once before the length sweep, so the untimed warm-up
pays for the loop and later calls reuse it, exactly like `target.py`.

Keeping two portals alive exposed another ownership defect: canonical liveness
was stored in one thread-local publication slot, so whichever matcher was
constructed last could make the other decode a same-numbered pc against the
wrong JitCode. The payload already lived on each driver; the ordinary
hot-counter trace entry simply failed to republish it (the force and bridge
entries did). It now does, and the two-live-portal regression test exercises
branching → masking → branching in one thread.

### Grading a change when the machine will not hold still

The tables above were taken on a quiet machine. On a busy one this row cannot
be timed at all: an interleaved A/B of two binaries built from the same tree
minutes apart read a **4.9x spread between rounds on the same arm**, which
swallows any change worth making. Repeats do not help — the noise is in the
machine, not in the sample.

`--features alloc-census` swaps in a counting `#[global_allocator]` and prints,
under every timed row, what that row allocated per input character. That number
does not depend on the machine: the same binary over the same input allocates
the same bytes every time. It is also the right unit for this particular gap,
because the costs in question **are** allocation — a jitframe `alloc_zeroed`ed
and freed per guard failure, blackhole frames boxed per resume, position tables
sized by a monotonic counter. Two of those three are fixed below, and the
counter is what graded both.

```sh
cargo run -p regex --release --no-default-features --features dynasm,alloc-census
```

It is off by default because a `#[global_allocator]` is process-wide, so its
counters would otherwise sit inside the timed rows this file reports.

What it says about the three rows, at 1,048,576 characters:

| per input character | allocs | bytes | of which zeroed |
|---|---:|---:|---:|
| Rust interp, no JIT | 0.0 | 0 | 0 |
| majit `&`/`\|` — `jit_interp.rs` | 0.0 | 4 | 0.1 |
| majit `and`/`or` — `shortcircuit.rs` | **11.4** | **1,360** | 7.8 |

The masking row and plain matcher round to zero allocations per character. The
branching row still calls the process allocator about **eleven times per input
character**. That is the remaining gap against RPython's own `--opt=jit`, in a
unit that can be measured while the machine is doing something else — every
character deopts, and majit's deopt round trip still owns vectors and `Rc<Op>`s
that upstream takes from a nursery or encodes into flat storage.

The current 1 MiB row is the combined result of persistent driver ownership,
pre-flatten helper coloring, the smaller ordinary `Op`, and the deopt-storage
fixes below: 11.4 allocations and 1,360 bytes per character, versus the
previously recorded 52.1 and 6,520. Fixed tracing and initial bridge compilation
are divided by fewer characters at short lengths, so the slope is expected and
explicit.

#### The third: preserve ownership across a deopt

Four Rust ownership conversions were allocating where the upstream object was
already stable:

* `BlackholeInterpBuilder.acquire_interp` returned an unboxed interpreter and
  `release_interp` boxed it again. The intrusive free list now carries the same
  `Box`, matching the object identity of upstream's pooled
  `BlackholeInterpreter`.
* Runtime field bytecodes rebuilt an optimizer descriptor from the blackhole
  descriptor on every execution. `RuntimeBhDescr` now resolves and retains the
  canonical optimizer descriptor once, after field-parent patching.
* constant queries converted a constant operand through a fresh `Rc<Op>` merely
  to ask `is_const` or obtain its value. `OptBoxEnv` and `force_box` now answer
  directly from the resolved operand, and `TraceIterator` preserves an already
  decoded constant.
* `bhimpl_jit_merge_point` allocated six new `Vec` backings and a fresh
  144-byte `ContinueRunningNormallyArgs` box on every deopt. The pooled
  blackhole frame now retains that box together with the six cleared backing
  capacities: the handoff crosses `DispatchError` and `JitException` as a
  `Box`, so neither enum is widened by a variant only a deopt reaches, and the
  same box returns to the frame once the payload has been consumed.

At 65,536 characters these changes move the measured row from 45.6 allocations
and 7,517 bytes per character to 21.6 and 2,891 (−52.6% and −61.5%). At the
published 1 MiB length, against the immediately preceding 19.9 / 1,900 row,
they read 11.4 / 1,360. The peeled trace remains exactly
`0 / 24 / 93 / 2` (`getfield_gc_r / getfield_gc_i / setfield_gc / int_eq`).

This does not turn the allocator census into a speed measurement. A clean
non-census release run after the change read 102,076 chars/s, but its five runs
spanned 44,677 to 128,912; that spread cannot establish either a speedup or a
regression against the older 135,690 row. The allocation reduction is the
result this instrument can establish.

#### The first one it caught

`OpTypeIndex`'s two position arrays were `vec![NO_POS; max_raw + 1]` indexed by
the bare `OpRef` raw. Raws come from a monotonic counter and a bridge mints its
own at `[parent_high_water..)`, so the array was sized by everything the trace
family had compiled so far. Instrumented at the 500th build: 187 entries
spanning 282 raws, in 17,332 slots.

The allocation census shows it as a per-character cost that **grows with the
input**, which a per-character cost cannot legitimately do:

| `and`/`or` row, bytes per character | 4,096 | 65,536 | 1,048,576 |
|---|---:|---:|---:|
| before | 11,687 | 9,750 | **19,559** |
| after | 11,604 | 8,661 | **6,948** |

`allocs/char` and `frees/char` were unchanged to one decimal place (54.0 and
51.9 at 1M on both, this row's figures before the next fix below) — the same vectors are still allocated, they are simply
sized by the trace instead of by the family. Before the fix the row's
per-character allocation doubles between 65,536 and 1,048,576 characters; after
it, it falls, which is what a per-character cost does. Wall clock could not
resolve the change on the machine of the day; this did.

#### The second: the jitframe itself

`execute_token` allocated the entry JITFRAME with `alloc_zeroed` off the GC
heap and freed it on the way out, once per compiled entry — so once per guard
failure on this row. A block obtained that way carries no header the collector
recognises, which is why it also needed a process-global `IndexSet` to be
traced at all (`shadow_stack::register_libc_jitframe`) and a second one to stay
rooted while the deadframe was being read (`libc_deadframe::LIVE_DEADFRAMES`).
Upstream has one mechanism and no registry: `llmodel.py malloc_jitframe` is
`jitframe.JITFRAME.allocate`, i.e. `lltype.malloc(JITFRAME, depth)`, and the
deadframe is an ordinary local that the translated stack map roots.

`runner::alloc_jitframe` now takes the frame from the nursery under the
published JITFRAME type id — the same place `dynasm_nursery_slowpath_jitframe`
already took the CALL_ASSEMBLER callee frames — and the deadframe holds it in a
root slot, re-reading the address on every access because a collection in that
window moves it. Nothing is registered and nothing is freed.

The instrument that reads it is not this crate's census but the metainterp's:

```sh
cargo test -p majit-metainterp --features dynasm --test allocs_per_compiled_entry
```

```text
  per call                 4.000      (was 5.000)
```

One allocation per warm compiled entry, gone, and the two backends agree again:
that file used to pin 4 for cranelift and 5 for dynasm, and the single row
between them was this frame.

#### …which needed this crate to have a collector at all

Both numbers above are conditional on one, and this crate had none — no
`majit/examples/` crate did. That is not a difference from RPython that was
open to it: **a JIT-enabled RPython build cannot be translated without a GC.**
`--gc=none` and `--gc=ref` are real translation options
(`translationoption.py:65-82`), but they select a `gctransformer` of
`"none"`/`"ref"`, and `gc.py:653-662 get_ll_description` looks up
`GcLLDescr_<gctransformer>` in a module that defines only `GcLLDescr_boehm`
(`gc.py:151`) and `GcLLDescr_framework` (`gc.py:313`) — anything else raises
`NotImplementedError("GC transformer %r not supported by the JIT backend")`.
Both surviving descrs inherit `malloc_jitframe` from the base class
(`gc.py:132-135`), where it is `lltype.malloc(JITFRAME,
frame_info.jfi_frame_depth)` (`jitframe.py:50`) — a GC allocation, and there is
no arm that puts a JITFRAME anywhere else. Running majit's side without a
collector put it on a path no translatable configuration takes. `src/gc.rs`
installs MiniMark in `main`, which is `pyre-jit`'s `init_gc_subsystem` with the
pyre-specific parts removed.

Installing it was not enough on its own, and the census is what said so: the
row did not move. The frame was coming from `alloc_nursery_no_collect_typed`,
so once the nursery filled every frame after that spilled to old-gen through
`rawmalloc`, one process allocation each and nothing ever reclaiming them. Upstream's `malloc_jitframe` is an ordinary
`lltype.malloc` and collects when the nursery is full; it can, because the
arguments it stores into the frame afterwards (`llmodel.py:306-315`) are RPython
locals the translated stack map roots across the allocation.

`execute_token` now says the same thing in majit's spelling: the `Ref` arguments
go on the shadow stack, the allocation is allowed to collect, and the arguments
are read back from the slots a collection would have rewritten. The frames are a
nursery bump again and the dead ones are reclaimed.

| `and`/`or` row, per character | allocs | bytes |
|---|---:|---:|
| off-GC frame, no collector | 54.0 | 6,948 |
| collector installed, no-collect frame | 54.0 | 7,000 |
| collector installed, collecting frame | **52.1** | **6,520** |
| persistent driver + compact helper/Op | **19.9** | **1,900** |
| current: deopt ownership preserved too | **11.4** | **1,360** |

The middle row is why the census is worth having: it is the fix applied and not
working, and nothing else in the run says so. The final row combines the later
structural changes described above and uses the RPython-orthodox persistent
driver across the warm-up and five timed matches.

### The majit-only ratio

The post's headline ratio is also measured on the majit side alone, in one
process, and that one runs in the normal suite:

```sh
cargo test -p regex --release --no-default-features --features dynasm \
    -- --nocapture the_jit_is_worth_several_times
```

```text
[perf] 1048576 chars, majit JIT : 39773276 chars/s (min 34615751, max 40783960)
[perf] 1048576 chars, no JIT    :  6302331 chars/s (min  5512454, max  6937078)
[perf] majit JIT / no JIT = 6.3x   (the post's own: 16,500,000 / 720,000 = 22.9x)
```

That ratio is smaller than the table's 9.4x, and most of the difference is in
the denominators: this one is `interp.rs` through `rustc -O`, while the table's
is `target.py` through RPython's C backend. The numerators are the same
quantity measured under different machine load. `--release` matters: a debug
run inflates the ratio because the
denominator is unoptimized, and the test prints a banner saying so rather than
letting that number be quoted.

Cranelift and dynasm agree op for op on every census above. Their remaining
wall-clock difference is backend work — especially bridge compilation — not a
different traced program or repeated reconstruction of the portal driver.
