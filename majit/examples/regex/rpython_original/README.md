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
other half: how fast the two implementations scan the same input against the
same 93-node tree.

`meta_interp` cannot answer it — `runner.py` runs the LLGraph backend, which
executes traces in an interpreter, so timing it would measure the harness.
Only a translated binary runs compiled traces natively, which is what
`target.py` and `target_masking.py` are for:

```sh
pypy <repo root>/rpython/bin/rpython --opt=jit target.py   # RPython + JIT
pypy <repo root>/rpython/bin/rpython --opt=2   target.py   # translated to C
```

`--opt=jit` takes about 204 s and `--opt=2` about 41 s, so all four binaries —
two spellings times two optimization levels — are a five-minute build. The
`--opt=jit` output lands as `target-c` (RPython's default name); `PYPYLOG=jit-summary:-`
on it prints a JIT summary, which is how to tell the two apart.

### The instrument

Wall clock cannot grade this row on a shared machine: an interleaved A/B of two
binaries built from the same tree minutes apart has read a 4.9x spread between
rounds on the same arm, and absolute chars/s moves 2x with load. Two
instruments do hold still:

* **Retired instructions per character, as a slope.** `/usr/bin/time -l`
  reports `instructions retired` for the whole process, so `PYRE_REGEX_ROWS=2`
  deselects the other rows and the counter charges the `and`/`or` row alone.
  Each side is run at 262,144 and 1,048,576 characters, five timed runs per
  point after one untimed warm-up; the slope
  `(instr@1M − instr@256K) / (5 × 786,432)` cancels process setup, warm-up and
  the initial compile. The two binaries run back to back inside a round and
  the ratio is taken **inside that round only**. Before quoting a ratio
  against an older one, check the denominator arm: RPython's own slope must
  reproduce its recorded **11,354** instr/char within about 0.5%. When it
  does, the two sittings are the same instrument; when it does not, they are
  not comparable.
* **The allocation census.** `--features alloc-census` swaps in a counting
  `#[global_allocator]` and prints, under every timed row, what that row
  allocated per input character. It does not depend on the machine: the same
  binary over the same input allocates the same bytes every time, to the
  printed digit. It is off by default because a global allocator is
  process-wide and would sit inside the timed rows.

```sh
cargo run -p regex --release --no-default-features --features dynasm,alloc-census
```

### Where the two portals stand

**`&`/`|`, the spelling the post reports, is at parity.** The post's own
numbers come from the adapted matcher — part 2 says so — and on that row majit
reads 1.00x of RPython's `--opt=jit` in the same round (42.5M against 42.7M
chars/s on a quiet machine). Its headline ratio against RPython's C is the
same too: 9.4x against RPython's own 9.5x, both against the post's
16,500,000 / 720,000 = 22.9x on 2010 hardware. Nothing on this row is open.

**`and`/`or`, the post's own source, is where majit is slower.** Both JITs lose
to their own C on this spelling — RPython's `--opt=jit` runs 6.7–8x slower
than RPython's `--opt=2` — so "if you don't change the `and` and `or` ... it's
not particularly fast" is a property of the spelling, paid upstream in the
same direction. majit's own share is what is measured here. Measured
2026-09-23, three rounds, leading side alternated, 1-minute load 4.6–5.6:

| round | RPython `--opt=jit`, instr/char | majit, instr/char | ratio |
|---|---:|---:|---:|
| 1 | 11,370 | 20,865 | 1.835 |
| 2 | 11,372 | 20,855 | 1.834 |
| 3 | 11,358 | 20,874 | 1.838 |

The denominator arm reads 0.16% from its recorded 11,354, so this is the same
instrument as every earlier slope. Wall clock in the same rounds: majit
666K–708K chars/s against RPython 972K–1,086K. The allocation census reads
**1.3 allocations / 396.5 B per character** at 65,536 and 1.2 / 337.4 at
262,144.

How the ratio has moved, each row a same-round measurement:

| date | instrument | ratio |
|---|---|---:|
| 2026-09-03 | wall clock, three same-round pairs at 1,048,576 | 2.34–2.45 |
| 2026-09-04 | instruction slope, 27,464 vs 11,354 | 2.42 |
| 2026-09-19 | instruction slope, 22,011 vs 11,414 | 1.93 |
| 2026-09-23 | instruction slope, 20,865 vs 11,370 | **1.835** |

Bridge counts and deopt rates are unchanged throughout and match RPython's
(see "The bridges are RPython's too"); the peeled body stays exactly
`0 / 24 / 93 / 2`.

### Where the time goes now

`/usr/bin/sample` over the benchmark closure's subtree at 1,048,576
characters, 1,424 samples, 2026-09-23:

| | share |
|---|---:|
| bridge compilation (`bridge_from_guard_resume_position` subtree) | 30.5% |
| ↳ `Optimizer::optimize_bridge` | 14.5% |
| ↳ ↳ `optimize_with_constants_and_inputs_at` (the propagate loop) | 12.1% |
| blackhole resume (`resume_mainloop` subtree) | 20.8% |
| resume-data decode, flat self time in `majit_metainterp::resume` | 18.8% |
| `_tlv_get_addr` (thread-local access), flat | 2.9% |
| `malloc`/`free`, flat | 2.2% |
| `Rc` drop glue, `memcpy`/`memmove`, hash probing, flat | 2.0% / 2.0% / 0.8% |

The flat leaders inside the decode bucket are `BlackholeInterpreter::run_inner`
8.0%, `resume::blackhole_from_resumedata` 3.7%, and
`ResumeDataDirectReader`'s `decode_ref` 2.9% / `consume_one_section` 1.9% /
`decode_int` 1.6% / `next_int` 1.5%.

No single bucket owns the gap. Blackhole dispatch, the optimizer's propagate
loop and the resume walk each run about 1.8–2x their upstream counterpart, and
every uniform tax (refcounts, allocator, copies, thread-locals, hashing) sums
to 10.2% — removing all of it entirely would read 1.65x. Upstream's own
`PYPYLOG=jit-summary` on this spelling spends 30% of its run compiling
(`Tracing 0.145s` + `Backend 0.107s` of `TOTAL 0.847s` at 262,144 characters,
1185 bridges), so the bridge work itself is not the deviation; its per-bridge
cost is.

### Candidates refuted by measurement

Each of these was proposed as *the* majit-only cost and measured. They are
listed so they are not re-opened.

| candidate | measured | verdict |
|---|---|---|
| bridge cascade / per-character deopts | RPython deopts 0.9990 per character and grows 1185 bridges at 262,144; majit 0.9961 and about half as many per character | RPython's too |
| blackhole op count per deopt | 65.4 majit vs 82.2 RPython after coloring + coalescing | fewer than upstream |
| register copies per deopt | 2.21 majit vs 3.09 RPython `int_copy` | fewer than upstream |
| mimalloc | −6.7% process instructions, no wall-clock gain | measurement control only |
| GC / byte-materializer repairs | 0.073% instruction difference | not a lever |
| `malloc`/`free` as the largest bucket | 16% (2026-09-04) → 2.2% (2026-09-23) after the ownership fixes below | closed |
| `resop_refs` / `find_producer_op` positional lookup | 0.6% subtree, 0.3% self | not a lever |
| RSS growing with input | 1,138 MiB → 124 MiB at 1,048,576; row no longer moves with length | closed |

### Structural fixes that hold, and what pins them

Each of these closed a majit-only divergence from the upstream shape. They
are listed with the test or instrument that would report their return.

* **Persistent driver ownership.** `marked.py` owns one module-level
  `JitDriver`; each Rust portal now owns a persistent `Matcher { root, driver }`
  with `root` a ref green. Canonical liveness is republished by the hot-counter
  entry as well as the force and bridge entries, so two live portals cannot
  decode a same-numbered pc against the wrong JitCode. Pinned by the
  two-live-portal regression test.
* **Register allocation before flattening.** The proc-macro lowerer now runs
  the ported `tool/algo/regalloc.py` interference graph, `color.py`
  `DependencyGraph` coloring, `coalesce_variables`, and
  `GraphFlattener.enforce_input_args`' post-color swaps, omitting identity
  moves as `insert_renamings` does. The `shift` JitCode's working footprint
  fell from 36 int / 26 ref to 4 / 3. Pinned by the native-entry test's
  register counts.
* **`Bool` marks stay `Bool`.** `NodeRec.empty`/`marked` and the traced
  `shift` preserve the source type, so `goto_if_not` hands the box straight to
  `generate_guard` as `opimpl_goto_if_not` does; the former 24 `IntIsTrue` and
  two `IntAnd` are gone. Pinned by the peeled-body gate.
* **The jitframe comes from the nursery.** `execute_token` used to
  `alloc_zeroed` an off-heap frame per compiled entry and register it in two
  process-global sets to be traced at all; `llmodel.py` `malloc_jitframe` is
  `lltype.malloc(JITFRAME, depth)` and the deadframe is a stack-map-rooted
  local. `runner::alloc_jitframe` now takes the frame from the nursery under
  the JITFRAME type id, is allowed to collect, and reads its `Ref` arguments
  back through shadow-stack slots. Pinned by
  `cargo test -p majit-metainterp --features dynasm --test allocs_per_compiled_entry`
  at `per call 4.000`, the same on both backends.
* **The example has a collector.** A JIT-enabled RPython build cannot be
  translated without one: `get_ll_description` accepts only
  `GcLLDescr_boehm` and `GcLLDescr_framework`, and both inherit the GC
  `malloc_jitframe`. `src/gc.rs` installs MiniMark in `main`. The census is
  what showed the install alone was not enough — frames spilled to old-gen
  through `rawmalloc` until the allocation was allowed to collect.
* **Deopt ownership preserved.** The pooled `BlackholeInterpreter` stays
  boxed across acquire/release; `RuntimeBhDescr` resolves its optimizer descr
  once; constant queries answer from the operand instead of a fresh `Rc<Op>`;
  `bhimpl_jit_merge_point`'s six `Vec` backings and its
  `ContinueRunningNormallyArgs` box are retained by the pooled frame. Ordinary
  `Op` no longer embeds three inline fail-arg slots (280 → 240 B). `OpTypeIndex`
  is sized by the trace, not by the whole family's raw counter. Repeated
  non-null `ConstPtr` operands share the trace's ref cache. Together these took
  the census from 54.0 allocations / 6,948 B per character to 1.3 / 396.5.
* **Per-bridge retention removed.** No `CompiledTrace` kept per bridge
  (`send_bridge_to_backend` keeps nothing), compiled blocks packed into shared
  RWX mappings as `asmmemmgr.py` does, and cranelift's per-descr bridge cells
  and per-deopt recovery layout gone. Peak RSS at 1,048,576 characters is
  124 MiB and the row's chars/s no longer depends on length.

### What remains

The remaining deficit is spread across the host runtime rather than sitting in
one bucket, so the work is a queue of upstream-shape divergences, each named
by the upstream symbol pyre should match. In order:

1. **`bridge_from_guard_resume_position` materializes liveness.**
   `read_frame_liveness_reg_indices` builds three `Vec<u32>` per resumed frame
   and the caller re-walks them; `jitcode.py` `enumerate_vars` visits each
   index once through a callback and stores nothing, and
   `ResumeDataBoxReader.consume_boxes` consumes one value per callback. pyre
   already has `enumerate_vars`; the bridge walk does not use it.
2. **`BH_PROBE_PHASE`.** Two thread-local resolves per `BlackholeInterpreter::run`,
   not gated on the probe flag; `blackhole.py` has no such phase label.
3. **`ResumeStorage::virtual_infos`.** The `OnceLock` documented as "built on
   the first resume and shared" has no callers; `virtual_info_from_rd` rebuilds
   a `VirtualInfo` per failure where `_prepare_virtuals` assigns
   `rd_virtuals` and never rebuilds. Empty on this row, so a Python-side item.
4. **Identity-override compare per live ref** in `next_ref_for_resume_slot`;
   `ResumeDataDirectReader.next_ref` decodes and writes.
5. **The recorder owner.** During recording `attach_byte_buffer` writes the
   ported `opencoder::TraceRecordBuffer`, but `TraceCtx.recorder` is still
   `recorder::Trace`: a `FrontendSlot` per op, and `into_tree_loop` /
   `materialize_into_ops` rebuild a `Vec<OpRc>` of 240-byte `Op`s in front of
   the optimizer, where upstream's optimizer walks the buffer's
   `TraceIterator` alone. This is not a field swap: `OpRef` is a per-op unique
   index while `TraceRecordBuffer._index` counts only box-producing ops, so
   the map keys in compile, blackhole, optimizeopt and pyjitpl move to that
   convention together. It is the largest item and goes last.

A/B for each item is the instruction slope above, on a baseline binary built
from the same tree before the edit, with stdout and `MAJIT_STATS` loop/bridge
counts required identical.

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

That ratio is smaller than the 9.4x above, and most of the difference is in
the denominators: this one is `interp.rs` through `rustc -O`, while the 9.4x
is against `target.py` through RPython's C backend. The numerators are the
same quantity measured under different machine load. `--release` matters: a
debug run inflates the ratio because the denominator is unoptimized, and the
test prints a banner saying so rather than letting that number be quoted.

Cranelift and dynasm agree op for op on every census above. Their remaining
wall-clock difference is backend work — especially bridge compilation — not a
different traced program or repeated reconstruction of the portal driver.
