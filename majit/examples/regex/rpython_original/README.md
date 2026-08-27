# `rpython_original/` — the post's matcher, through RPython's own JIT

This is the comparison the example owes. "A JIT for Regular Expression Matching"
(PyPy, 2010) describes an RPython program and the trace RPython's JIT produces
for it; the crate above reimplements that program in majit. The only way to say
whether the reimplementation is faithful is to run the original and diff the two
traces — which is what this directory does, against the RPython checkout that is
already at the root of this repository.

## Run it

```sh
cd majit/examples/regex/rpython_original
PYTHONPATH=/path/to/pyre pypy runner.py 20 4096
```

`PYTHONPATH` is the repository root, because that is where `rpython/` lives.
The interpreter must be a Python 2 (`pypy`, or a `python2` with the RPython
checkout importable) — RPython's own toolchain is Python 2, and these files are
written in it.

Arguments are `n` (the regex's `{n}` repetition count, default 20) and the input
length (default 4096). `--listing` additionally prints the peeled body op by op.

Nothing is translated and nothing is installed. `runner.py` uses `LLJitMixin`,
the in-process harness RPython's own JIT tests use: real metainterpreter, real
optimizer, LLGraph backend. The loop that comes out is the trace RPython would
compile, without a multi-hour translation.

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
character — at `n = 20`, 93 nodes:

| op | RPython JIT | majit `shortcircuit` | majit `jit_interp` |
|---|---:|---:|---:|
| `getfield_gc_r` | 0 | 0 | 0 |
| `getfield_gc_i` | 24 | 24 | 24 |
| `setfield_gc` | 93 | 93 | 93 |
| `int_eq` | 2 | 2 | 2 |
| `guard_true` | 6 | 6 | 1 |
| `guard_false` | 21 | 20 | 0 |
| total | 153 | 176 | 194 |

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

At `n = 2` (21 nodes) RPython gives `setfield_gc 21`, `int_eq 2`,
`getfield_gc_r 0`: the store count tracks the node count, which is the control
saying the census is measuring the tree and not a constant.

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

## The 23 ops that differ

153 against 176 is one spelling difference, not a missing optimization: majit
emits an `IntIsTrue` ahead of each guard where an RPython guard takes its
condition directly. To read them side by side:

```sh
PYTHONPATH=/path/to/pyre pypy runner.py 20 4096 --listing
REGEX_LISTING=1 cargo test -p regex --no-default-features --features dynasm \
    -- --nocapture the_branching_body_trades_ops_for_guards
```

The two listings are formatted differently — RPython's prints op names, majit's
prints the whole operation — so they are read together, not `diff`ed.

## What this is not

It is not a speed comparison. `meta_interp` runs the optimizer and the LLGraph
backend, which executes traces in an interpreter; timing it would measure the
harness. Speed lives in `../comparisons/`, and the quantity that travels there
is a ratio taken within one run, never an absolute chars/s across machines.
