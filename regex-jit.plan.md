# majit/examples/regex — reproducing PyPy's "A JIT for Regular Expression Matching"

Reproduce, on majit, the experiment from Carl Friedrich Bolz-Tereick's 2010 posts
[An Efficient and Elegant Regular Expression Matcher in Python][part1] (part 1)
and [A JIT for Regular Expression Matching][part2] (part 2): take a marked-regex
matcher, treat it as an interpreter whose *program* is the regular expression,
and let the JIT generator turn it into a DFA.

[part1]: https://pypy.org/posts/2010/05/efficient-and-elegant-regular-2727904462179540436.html
[part2]: https://pypy.org/posts/2010/06/jit-for-regular-expression-matching-3877859053629057968.html

The original sources lived on codespeak and are gone. Both posts carry the full
matcher in their body, and that body is the porting source. Nothing equivalent
exists in this checkout — `rpython/jit/tl/grep.py` is an `xxx` stub.

## What the experiment actually claims

The matcher walks a regex *tree*; every node carries one mutable `marked` bit,
and one `shift` per input character propagates marks left to right. Part 2 makes
the regex tree the JitDriver's single green
(`jitdriver = jit.JitDriver(reds=["i", "result", "s"], greens=["re"])`), declares
every other field `_immutable_fields_`, and replaces `and`/`or` with the
non-short-circuiting `&`/`|` so the loop body has no conditionals at all.

The payoff is the trace: the node pointers become `ConstPtr` constants, the
method calls and the tree traversal vanish, and what is left is character
compares, `&`, `|`, and stores into the `marked` field of *constant addresses*.
That straight-line body is one DFA state transition — the tracing JIT performed
subset construction as a side effect of specializing on a green.

**The claim is not "a JIT is fast".** It is "choose the right green and the
tracer builds the DFA for you". Measured here on 2026-08-25, on this machine,
with the post's own benchmark regex `(a|b)*a(a|b){20}a(a|b)*` over a
non-matching random `a`/`b` string:

| | plain (`and`/`or`) | `&`/`\|` variant |
|---|---:|---:|
| CPython 3.14.2 | 148,986 chars/s | 145,454 chars/s |
| PyPy 7.3.20 (20k chars, cold) | 74,162 | 304,057 |
| PyPy 7.3.20 (1M chars, warm) | 208,755 | 335,325 |

PyPy's *general* Python JIT, running the same algorithm, reaches 335k chars/s.
The post's 16.5M came from making the regex a green. That gap — roughly 50× — is
the thing this example exists to reproduce. The 2010 absolute numbers
(pure Python 12,200; RPython→C 720,000; RPython+JIT 16,500,000) are 2010 hardware
and are quoted for context only, never as a comparison.

The `&`/`|` substitution is worth 1.6–2.8× even at the Python level, which
independently confirms the post's remark about short-circuiting.

### Fixture facts

* `(a|b)*a(a|b){20}a(a|b)*` is **93 nodes** (Sequence 23, Repetition 2,
  Alternative 22, Char 46). At `n = 2` it is 21 nodes, at `n = 5` it is 33.
* Built left-associated the tree is **depth 26**; built balanced it is **depth 8**.
  The two associations accept the same language — verified against 1560 random
  strings and, at `n = 2`, exhaustively over every `a`/`b` string up to length 11.
  Association is therefore a free lever that does not change the algorithm.
* **Node instances may never be shared.** Each node owns its own mark, so the
  twenty `(a|b)` groups in `(a|b){20}` must be twenty distinct instances. No `Rc`.
* `DEFAULT_TRACE_LIMIT` is 6000 (`trace_ctx.rs`, mirroring RPython
  `rlib/jit.py`'s `trace_limit`). At roughly 6–10 ops per node that caps the
  compilable regex at a few hundred nodes. Overflow is `AbortReason::TooLong`
  plus segmenting, not a refusal.

A verbatim transcription of both variants of the matcher, 28 correctness vectors
(56 checks, since each runs against both variants), and the non-matching-input
generator are the porting source for the Rust side.

## Verified state of majit

Every claim below was read out of this tree, not assumed.

1. **A matchless mainloop is a compile error.** `generate_trace_fn`
   (`majit-macros/src/jit_interp/codegen_trace.rs`) opens with
   `let Some(match_expr) = find_dispatch_match(...) else { return
   syn::Error::new_spanned(func, "could not find opcode dispatch match") ... }`.
   `collect_matches_in_expr` in the same file recurses through
   `Expr::Match | While | Loop | Block | If` and has **no `Expr::ForLoop` arm**,
   so a `for c in s` portal loop is invisible to both the match finder and
   `find_dispatch_loop_body`. `Expr::ForLoop` *is* handled inside arm bodies
   (`lower_for_loop`, reached from `lower_stmt.rs`), where it unrolls literal
   ranges at expansion time. The refusal is specific to the portal loop.

2. **A Rust `enum` cannot be the traced representation.** `lower_match_stmt`
   (`jitcode_lower/lower_control.rs`) requires a `BindingKind::Int` discriminant,
   and `extract_pat_value_tokens` (`jitcode_lower/helpers.rs`) accepts
   `Pat::Lit | Pat::Path | Pat::Ident | Pat::Or` and ends `_ => None`, so
   `Node::Char(c)` — a `Pat::TupleStruct` — cannot be a dispatch arm.
   `#[jit_struct]` refuses enums outright.

3. **Greens must be single-segment idents bound at portal entry.**
   `resolve_greens` (`jitcode_lower/dispatch.rs`) panics otherwise, citing
   `jtransform.py`'s `decode_hp_hint_args`; dotted greens are a named follow-up
   task. The only unconditional portal-entry bindings are `program` (Ref, r0) and
   `pc` (Int, i0) in `lower_dispatch_body`, plus body-locals bound before the
   merge point. `greens = [<state field>]` panics. All thirteen in-tree examples
   spell `greens = [pc, program]`.

4. **Structural field loads can never constant-fold on this path.** The only
   fold site is `OptHeap::optimize_getfield`
   (`majit-metainterp/src/optimizeopt/heap.rs`), gated
   `if descr.is_always_pure() && ctx.get_constant_box(...)`, and
   `is_always_pure()` is `self.is_immutable && !self.is_quasi_immutable`
   (`majit-ir/src/descr.rs`). All three field-descr mint sites in
   `majit-metainterp/src/jitcode/assembler.rs` write `is_immutable: false,
   is_quasi_immutable: false`. `TraceCtx::try_const_fold_pure_field` — the
   record-time equivalent — exists with **zero callers**.

5. **A recursive `#[jit_inline]` helper has no representation.** The Inline arm
   of `Lowerer::lower_call_value` (`jitcode_lower/lower_value.rs`) emits
   `let __sub_jitcode = #builder_path(__asm);`, and `inline_builder_path`
   rewrites that path to the callee's own
   `__majit_inline_jitcode_<name>_with_asm` — a call that sits inside the
   caller's own `_with_asm` function. `lower_if_stmt` flattens both branches into
   one linear build-time stream, so a base case cannot stop a self-call. There is
   no cycle guard anywhere under `jitcode_lower/`. Nobody has observed the
   failure, so do not write "stack overflow" into a commit message without
   running it.

6. Ref greens themselves are **not** unexercised: every example's `program` is a
   Ref green (`impl<T: ?Sized> GreenAsI64 for &T` yields `GreenType::Ref`),
   `emit_promote_greens` emits `ref_guard_value`, and the walker's
   `BC_REF_GUARD_VALUE` arm rebinds the register to `ctx.const_ref(concrete)`.
   What has no callers is the tagged spelling `greens = [x: ref]` and a
   ref-only green list.

### The decisive find

`#[majit_macros::jit_immutable_fields("pools", "size?", "digits[*]")]` **already
exists**, in RPython's exact spelling — `?` for quasi-immutable, `[*]` for
arrays. Its consumers exist too: `BhFieldSpec.is_immutable` on the translate
side, `is_always_pure()`, `optimize_getfield`'s fold, `constant_fold`.

Only the middle is missing. The marker is harvested by
`majit-translate`'s LLBC front end (`harvest_immutable_fields_from_llbcs`,
feeding `layout.rs`'s `immutable_fields_by_struct`) — pyre's Charon path — and
the proc-macro path's `Assembler::register_struct_layout(size, type_id,
is_gc_managed, headerless, fields: &[(offset, is_ref, name, size, signed)])`
tuple has no immutability slot at all.

So the hint the blog post uses is present in majit and dropped by one front end.

Likewise for recursion: RPython's `CodeWriter.make_jitcodes` walks
`callcontrol.enum_pending_graphs()` and emits **one JitCode per graph**. It does
not splice. majit's *trace-time* machinery is already that shape —
`add_sub_jitcode` registers the callee and
`push_inline_frame((sub_idx, pc), u32::MAX)` inlines it while tracing. Only the
build step deviates.

Both fixes are therefore "connect what is already there", not new machinery.

## Stage 0 — probe first (throwaway)

No fixture in this repo proves that a `#[jit_interp]` machine which walks heap
nodes compiles a loop at all. Every `ref_fields` / `ref(T)` fixture under
`majit-metainterp/tests/` asserts structure — jitcode bytes, descr registration,
`struct_layout_conflicts() == vec![]` — and **none reads a compile counter**;
every fixture that gates on `COMPILES > 0` (tinyframe, tl, tla, tlc, tlr,
i64env) is integer-and-virt-array only. That precondition is unproven, and
Stages 1–3 all rest on it.

Add ~40 throwaway lines to an example that already has a liveness gate
(`majit/examples/tla`): a `#[repr(C)]` three-node chain built before the loop,
`state_fields = { ..., root: ref(NodeRec) }`, `ref_fields` / `int_fields`
declarations, and one existing arm given a body that reads a child pointer,
promotes it, reads and writes `marked` through it.

Read three things off one run:

1. Does `[jit] degraded dispatch arm:` appear for that arm? If yes, the ref-field
   vocabulary does not reach an arm body and the whole approach changes.
2. Does `COMPILES` still move, and does `LoopBodyShape::closes_a_loop()` hold?
3. In the optimized-trace dump, is the `setfield_gc_i` operand a `ConstPtr` or a
   `getfield` result, and how many `GuardValue`s survive the peel?

Run it under `--features dynasm` **and** `--features cranelift`. This workspace
has a recorded 463/463-green-on-dynasm, red-on-cranelift case, and CI's cranelift
leg is package-scoped so it never builds examples.

Delete the probe afterwards; its output is an answer, not code we keep.

## Stage 1 — thread `#[jit_immutable_fields]` through the proc-macro path

Widen the `register_struct_layout` field tuple with the immutable rank and carry
it into the minted descrs, so `is_always_pure()` can answer true. The consuming
half needs no change.

Read the declaration by harvesting the existing `#[jit_immutable_fields]`
attribute rather than adding a new `#[jit_interp]` key: one declaration serves
both front ends, and it is the exact analogue of RPython's `rclass.py` reading
`cls._immutable_fields_` off the class.

Effect on the trace, and why it cascades: the root is a green, hence a
`ConstPtr`; `root.left` folds to a constant; that makes the base of
`(root.left).left` constant, so it folds too. The whole walk collapses, and
`marked` — deliberately absent from the declaration — remains a `setfield_gc_i`
into a constant address. That is the post's listing.

The post shows the *optimized* IR, so optimize-time folding is the correct
target. Reviving `try_const_fold_pure_field` for a record-time fold is optional
polish, not a requirement.

## Stage 2 — make a recursive `#[jit_inline]` helper representable

Memoize helper JitCode construction on the helper's canonical path, so a
self-call resolves to the already-registered sub-jitcode index instead of
re-entering the builder. Depth is then bounded at trace time by the existing
frame machinery rather than at build time.

There is no cheap way around this. A static depth chain `shift_d0..shift_dK`
explodes **exponentially**: `Alternative` and `Sequence` each call `shift`
twice, so K levels of build-time splicing produce 2^K copies of the body — 256
at the balanced tree's depth 8, hopeless at the left-associated depth 26.

If Stage 2 proves deeper than expected, the fallback is an explicit worklist walk
in the arm body with the stack in `state_fields = { stack: [int; virt], sp: int }`.
Meta-tracing unrolls that loop naturally, so the *trace* stays faithful; what is
lost is the resemblance to the post's five-line `shift`. Note that loop-carried
values must live in `state.*`: `lower_while_loop` snapshots and restores
bindings, and `lower_local_reassign` allocates a fresh register on rebind.

## Stage 3 — the example

```
majit/examples/regex/
├── Cargo.toml            # features dynasm/cranelift; deps majit-{ir,macros,metainterp}
├── src/regex.rs          # enum Node (authoring API) + lower() -> NodeRec
├── src/interp.rs         # the plain matcher over NodeRec — the baseline row
├── src/jit_interp.rs     # the #[jit_interp] portal + the #[jit_inline] shift
└── src/main.rs           # chars/s benchmark, interp vs JIT
```

plus `"majit/examples/regex"` in the workspace `members` list.

The authoring surface keeps part 1's structure:

```rust
pub enum Node {
    Char(u8), Epsilon,
    Alternative(Box<Node>, Box<Node>),
    Sequence(Box<Node>, Box<Node>),
    Repetition(Box<Node>),
}
```

`Node::lower()` produces the traced representation once, before the JIT sees
anything:

```rust
#[repr(C)]
#[majit_macros::jit_immutable_fields("kind", "ch", "empty", "left", "right")]
pub struct NodeRec {
    kind: u8, ch: u8, empty: u8, marked: u8,
    left: *mut NodeRec, right: *mut NodeRec,
}
```

`empty` is part 1's `Regex.empty`, computed during lowering; `marked` is the one
mutable bit and is deliberately absent from the immutability declaration. That
attribute line *is* the post's `_immutable_fields_`.

Lowering the enum by hand is not a concession — RPython has no enums, and its
rtyper lowers a class hierarchy to exactly this tagged-struct shape. `match kind`
is the match dispatch, on an `Int` discriminant the lowerer accepts.

The tree is `Box::leak`ed. Every traced `ConstPtr` must stay valid for the life
of the process; this is a **Rust lifetime obligation, not a GC one** —
`majit_gc::can_move` answers false with no backend installed and
`LlModel::protect_speculative_field` returns early when
`!majit_gc::supports_guard_gc_type()`.

`shift` is part 1's recursion, over integers, with `&`/`|`:

```rust
CHAR    => mark & ((ch(n) == c) as i64)
EPSILON => 0
ALT     => shift(left(n), c, mark) | shift(right(n), c, mark)
REP     => shift(left(n), c, mark | marked(n))
SEQ     => { let oml = marked(left(n));
             let ml  = shift(left(n),  c, mark);
             let mr  = shift(right(n), c, oml | (mark & empty(left(n))));
             (ml & empty(right(n))) | mr }
```

`#[jit_inline]` parameters may only be `usize` / `i64` (`is_supported_ref_type`
accepts `usize` and `Type::Ptr`; `is_supported_int_cast` has no `bool`), which
dovetails with the post's own move to `&`/`|` on integers.

The portal is a one-instruction interpreter. The post says so itself: "the
matcher works by running exactly one loop as many times as the input string is
long, irrespective of the program ... within the loop there are no conditions at
all". So `pc` is degenerate and pinned at 0, and `greens = [pc, program]` — the
canonical shape — carries the regex root as `program`.

The first character is shifted in before the loop, exactly as the post's JIT
version does (`result = re.shift(s[0], 1); i = 1`), which leaves `mark` constant
at 0 inside the loop and the body free of conditionals.

**Open point to settle during implementation.** The post (per Armin Rigo's
correction in its comments) requires `can_enter_jit` before `jit_merge_point`
with nothing in between; every majit example writes `jit_merge_point!` first and
then `if pc == 0 { can_enter_jit!(...) }`. Follow the macro's contract and state
the inversion in the example's header.

## Stage 4 — gates

1. **Correctness.** The 28 transcribed vectors, plus a differential test of
   `interp` against `jit` over random strings.
2. **JIT liveness, not "it compiled".** `lower_dispatch_chain` returns its
   default label *silently* when no `Int` binding named `opcode` exists, and
   `dispatch_lower_ok` still reads true — a silently inert JIT tier would pass
   every other assertion in the file. Gate on
   `assert_no_degraded_dispatch_arms(...)`, `COMPILES > 0`,
   `LoopBodyShape::closes_a_loop()`, and an `ops_after` pinned op by op.
3. **Trace shape — the actual claim.** On the optimized trace for `(a|b)*`:
   zero `getfield_gc_r` (the walk is gone), every `setfield_gc_i` base a
   `ConstPtr`, zero `guard_*` on the tree. "We reproduced the post" must be an
   assertion, not a screenshot.
4. Both backends, per Stage 0.

## Stage 5 — the benchmark table

`main.rs` reports chars/s for `interp` and `jit` on the post's regex with a
non-matching random input, and prints the 2010 figures alongside, labelled as
2010 hardware.

| row | status |
|---|---|
| CPython 3.14 pure Python | measured, 148,986 |
| PyPy 7.3.20, general Python JIT | measured, 335,325 |
| Rust interp over `NodeRec`, no JIT | to measure |
| majit JIT | the experiment |
| CPython `re` | optional reference |

The example's header must state which claim its number supports, and must say
that the marks live **in the nodes**. A `[int; virt]` mark bank would give
strictly better code — virtualizable entries resolve with no memory ops at all —
but it would be a different claim, and reporting it as "we reproduced the post"
would be a mis-report.

## Scope boundaries

* Regex size is bounded by the 6000-op trace limit. `n = 20` is 93 nodes, well
  inside it.
* `lower_call_value` reads `config.call_returns` in only the `ResidualRef` and
  `NurseryAllocRef` arms; every `*Wrapped` arm, including the
  `ElidableRef*Wrapped` family, falls through to a tail returning
  `struct_type: None`. That closes the elidable-accessor route to constant node
  addresses. It is **not** on this plan's critical path — Stage 1 uses immutable
  fields instead — so record it as a separate finding; the fix is a ~5-line hoist
  of that lookup into the shared tail.
* `#[jit_interp]` requires a `match`. A one-instruction portal satisfies it with
  a single-arm `match opcode`. Note the shape tax in the example header and file
  "let a matchless portal lower" as a separate follow-up.

## Rejected alternatives

* **Flat `[i64]` arena walked by literal-range sweeps.** Needs no majit change at
  all — `lower_env_array_read` accepts `program[<any Int expression>]` when
  `pc_is_green`, and the walker already folds an all-constant
  `BC_GETARRAYITEM_GC_I` at record time. Rejected because it pre-flattens the
  tree the post's whole point is about, and because two topologically ordered
  sweeps are a different algorithm from `shift`. Kept in reserve.
* **`ref(T)` state root with a `promote` per edge.** Also needs no majit change,
  and the mark stores do land on constant addresses. Rejected as the deliverable
  because it leaves one `GuardValue` per tree edge per input character where the
  post has none. Retained as Stage 0's probe shape.
* **`Box<dyn Regex>` trait objects.** Closest to part 1's class hierarchy, but
  the lowerer has no vocabulary for dynamic dispatch, and the enum-to-tagged-
  struct lowering is what RPython's rtyper does anyway.
