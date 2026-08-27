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

## Stage 0 — probe — **DONE, and it clears the way**

Ran in an isolated worktree against `majit/examples/tla`: a `#[repr(C)]`
three-node chain leaked before the loop, `state_fields = { …, root: ref(NodeRec),
acc: int }`, `ref_fields = { NodeRec::left => NodeRec, NodeRec::right => NodeRec }`,
`int_fields = { NodeRec::kind => u8, NodeRec::ch => u8, NodeRec::marked => u8 }`,
and the `ADD` arm given a body that reads a child pointer, promotes it, then
reads and writes `marked` through it. Nothing kept; the probe was deleted.

**1. The arm does not degrade.** `outcome=Inlined`, no
`[jit] degraded dispatch arm:` line, and the example's own
`degraded_dispatch_arms()` assertion still passes. The ref-field vocabulary
reaches a `#[jit_interp]` arm body. The headline worry is refuted.

**2. The tier stays alive and the compiled code is right.** 1 loop compiled
before and after (30 → 39 ops), `closes_a_loop()` holds, suite 17/17 green.
Warm and cold agree on the same binary at N=401: `result=401 marked=1 acc=200`,
`compiles=0` cold vs `compiles=1` warm.

**3. The store's base is a `ConstPtr` — the post's shape.**

```
SetfieldGc(ptr(0x107720c20), v290)
  descr=<SimpleFieldDescr { name: "marked", offset: 2, field_size: 1,
                            field_type: Int, flag: Unsigned }>
```

The `int_fields` declaration is what gives the descr the real sub-word width;
`left` mints `offset: 8, field_size: 8, Ref, Pointer`.

**4. Zero `GuardValue` survive into the loop body.** Both promotes sit in the
peeled preamble; the body carries only the two exit `GuardTrue`s. The
`GetfieldGcR`, the `ref_guard_value` and the `GetfieldGcI` reload are all gone
from the body.

**5. Both backends agree op-for-op** — 39 ops, 6 guards, identical v-numbering
on dynasm and cranelift; only the leaked pointer literal differs. 17/17 green on
each.

### The control that changes Stage 1's justification

Deleting **only** the `promote(l)` still lowers and still compiles (38 ops), but
the store's base becomes a loop input arg:

```
v22 = GetfieldGcR(v3) descr=left      ← hoisted to the preamble
SetfieldGc(v22, v288) descr=marked    ← body: base is the getfield result
```

So loop-invariance alone hoists the read; **the promote is what makes the base a
constant**, and it pays a `ref_guard_value` for it. That is the difference
between this probe and the post. The post's walk is constant with *no* guard,
because `left`/`right` are declared immutable and `optimize_getfield` folds them
outright. At 93 nodes a promote-per-level would emit ~93 `ref_guard_value`s;
immutable fields emit none. Stage 1 is therefore not polish — it is what
separates a guarded re-derivation from the post's subset construction.

### Two corrections to this document's own premises

- **"…and **none** reads a compile counter" was false.**
  `majit-metainterp/tests/jit_interp_array_field_state_ref_base.rs` declares
  `state_fields = { stack: ref(ElemStack), … }` plus `array_fields`, drives 200
  iterations, and asserts `COMPILES > before` *and* `GetfieldGcR > 0`,
  `SetarrayitemGc > 0`, `GetarrayitemGcI > 0`, plus warm-vs-cold agreement. 3/3
  pass. The `ref(T)`-base path was already gated; only `ref_fields` node chasing
  was not.
- **`promote` is not needed to make the walk lower.** The no-promote control
  lowers and compiles fine. It is needed only to constant-fold the base.

### The instrument to use

`MAJIT_MACRO_DEBUG=1` — proc-macro time. It prints
`[majit-macro] dispatch arm <NAME> pattern=… inlined=… outcome=Inlined|Rejected`
and, on a refusal, the offending statement verbatim:
`[majit-macro] lower_stmt unsupported (local): let m = l.marked;`. That names the
*statement*; the runtime `[jit] degraded dispatch arm:` line (gated on
`MAJIT_LOG`) only names the arm. There is also a free coverage check —
`[jit] unconsulted field declaration: TlaState declares NodeRec::ch, unused`.


## Stage 1 — thread `#[jit_immutable_fields]` through the proc-macro path

Every link of the chain below was read at its definition; only the two marked
**MISSING** do not exist.

```
#[jit_immutable_fields("left", "right", "kind", "ch", "empty")]
  └─ emits  const _immutable_fields_NodeRec: &str = "left,right,kind,ch,empty"   [exists]
  └─ emits  NodeRec::__MAJIT_IMMUTABLE_FIELDS                              [MISSING]
       └─ lower_vable.rs: 9 × register_struct_layout(…, IMMUT)             [MISSING]
            └─ field_specs_from_layout — ImmutableRank::parse              [MISSING]
                 └─ BhFieldSpec.is_immutable / .is_quasi_immutable         [exists]
                      └─ SimpleFieldDescr.is_immutable                     [exists]
                           └─ is_always_pure()                             [exists]
                                └─ OptHeap::optimize_getfield fold         [exists]
```

`is_always_pure` is `descr.rs`:

```rust
fn is_always_pure(&self) -> bool {
    self.is_immutable && !self.is_quasi_immutable
}
```

**The symbol-availability problem, and the fix.** Generated code must name the
declaration for *every* struct it registers a layout for, including structs that
never opted in — and a struct without the attribute has no such symbol, so an
unconditional reference is a compile error. Solved with an inherent associated
const shadowing a blanket-trait default, verified by rustc:

```rust
pub trait MajitImmutableFields { const __MAJIT_IMMUTABLE_FIELDS: &'static str = ""; }
impl<T: ?Sized> MajitImmutableFields for T {}
// #[jit_immutable_fields] additionally emits, next to the untouched struct:
impl NodeRec { #[doc(hidden)] pub const __MAJIT_IMMUTABLE_FIELDS: &'static str = "left,right,kind,ch,empty"; }
```

`NodeRec::__MAJIT_IMMUTABLE_FIELDS` → `"left,right,kind,ch,empty"`;
`Plain::__MAJIT_IMMUTABLE_FIELDS` → `""`. The inherent const wins, so the
generated call site is uniform and nothing opting out breaks.

This keeps the existing free `_immutable_fields_<Struct>` const untouched —
`harvest_immutable_fields_from_llbcs` reads it and must keep working. One
declaration, two front ends, exactly as `rclass.py InstanceRepr` reads
`cls._immutable_fields_` once.

**The suffix grammar stays in one place.** `majit-metainterp` already depends on
`majit-translate` (`Cargo.toml`), so `ImmutableRank::parse` — which handles
`?[*]` → `[*]` → `?` → plain — is called directly rather than re-implemented.

**The one non-obvious edit.** Threading into `BhFieldSpec` is not enough: the
*parented* field-descr mint hardcodes `is_immutable: false` while already doing
the very lookup that would answer it —

```rust
let slot = field_slot_in(&parent_spec.all_fielddescrs, field_name, offset);
let (field_size, field_flag, is_field_signed) = slot
    .map(|idx| { let f = &parent_spec.all_fielddescrs[idx];
                 (f.field_size, f.field_flag, f.is_field_signed) })
    .unwrap_or(…);
…
is_immutable: false,        // ← carry f.is_immutable from the same slot
is_quasi_immutable: false,  // ← and f.is_quasi_immutable
```

Half of this plumbing is already built: `add_scalar_field_descr_with_immutability`
exists, and its doc comment cites `_immutable_fields_` by name. Only the parented
path was never connected.

**Why it cascades.** The root is a green, hence a `ConstPtr`; `root.left` is an
immutable read off a constant base, so `optimize_getfield` folds it to a
constant; that makes `(root.left).left`'s base constant, so it folds too. The
whole 93-node walk collapses with **no guards**. `marked` — deliberately absent
from the declaration — stays a `setfield_gc_i` into a constant address. That is
the post's listing.

The post shows the *optimized* IR, so optimize-time folding is the correct
target. Reviving `try_const_fold_pure_field` for a record-time fold is optional
polish, not a requirement.

## Stage 2 — make a recursive `#[jit_inline]` helper representable

The defect is one generated line. `#[jit_inline]` lowering emits a plain Rust
call to the helper's builder (`lower_value.rs`, and twice in `lower_stmt.rs`):

```rust
let __sub_jitcode = #builder_path(__asm);
let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
```

If `shift` is `#[jit_inline]` and its body calls `shift`, then
`__majit_jitcode_shift(asm)` calls `__majit_jitcode_shift(asm)` — unbounded
recursion **at jitcode-build time**, i.e. a stack overflow during warmup, before
any tracing happens.

`add_sub_jitcode_arc` is a thin `push_descr_entry(RuntimeBhDescr::JitCode(..))`
with no memo, so the fix is: key a memo on the builder's function pointer,
*reserve* the descr slot before lowering the body, and fill it after. A self-call
then finds the reserved index and emits `inline_call_*` against it. Depth is
bounded at trace time by the existing frame machinery, which is where RPython
bounds it too.

`recursive_portal_call!` / `recursive_entry` is **not** this. That pair is for a
recursive call that re-enters the *portal* through the JIT driver; `shift` is an
ordinary helper the tracer must see through.

There is no cheap way around it. A static depth chain `shift_d0..shift_dK`
explodes **exponentially**: `Alternative` and `Sequence` each call `shift` twice,
so K levels of build-time splicing produce 2^K copies — 256 at the balanced
tree's depth 8, hopeless at the left-associated depth 26.

If Stage 2 proves deeper than expected, the fallback is an explicit worklist walk
in the arm body with the stack in `state_fields = { stack: [int; virt], sp: int }`.
Meta-tracing unrolls that loop naturally, so the *trace* stays faithful; what is
lost is the resemblance to the post's five-line `shift`. Note that loop-carried
values must live in `state.*`: `lower_while_loop` snapshots and restores
bindings, and `lower_local_reassign` allocates a fresh register on rebind.

### Off the critical path: `*Wrapped` ref policies drop `call_returns`

The Stage 0 probe surfaced a third gap, confirmed by a single-variable control.
A helper under any `*_wrapped` ref policy cannot carry a pointee type, so the
arm degrades:

```
[majit-macro] lower_stmt unsupported (local): let m = l.marked;
[jit] degraded dispatch arm: TlaState::ADD lowered to an abort stub
```

`lower_call_value` reads `config.call_returns` only in its `ResidualRef` and
`NurseryAllocRef` early-return arms; the whole `*Wrapped` block falls through to
a shared tail that returns `struct_type: None`. Swapping *only* the policy to
`residual_ref` — same function, same `call_returns`, same body — lowers fine.
And there is no non-wrapped `elidable_ref*` policy at all, so the workaround
costs a `CallR` + `GuardNoException` + re-promote every iteration (43 ops / 9
guards vs 39 / 6). The concrete path already handles it —
`RefFieldRewriter::call_return_type` consults `call_returns` policy-blind — so
the crate *builds* and only the JIT half is silently lost.

**This example does not need it**: `shift` reaches children by direct
`ref_fields` access (`n.left`), not through a helper returning a ref. Recorded
here as a finding to file, not as work in this plan.


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

All rows below were taken on this machine, same regex (`n = 20`, 93 nodes),
same non-matching input, `&`/`|` variant.

| row | chars/s | status |
|---|---|---|
| CPython 3.14.2, pure Python | 145,454 | measured |
| PyPy 7.3.20, general Python JIT, warm | 335,325 | measured |
| Rust interp over `NodeRec`, no JIT | 1,705,377 – 2,582,170 | measured |
| majit JIT | — | **the experiment** |
| CPython `re` | 5,409,486 – 13,281,692 | measured, **not comparable** |

The `re` row is not a baseline and must be labelled so in `main.rs`: `re` may
bail out of a non-match early, while the marked matcher always scans the whole
string. It is included only to forestall the reader who reaches for it.

The Rust-interp row is a range because it moves with the input; report the
harness's own number next to the JIT number from the same run, not this figure.

The example's header must state which claim its number supports, and must say
that the marks live **in the nodes**. A `[int; virt]` mark bank would give
strictly better code — virtualizable entries resolve with no memory ops at all —
but it would be a different claim, and reporting it as "we reproduced the post"
would be a mis-report.

## Scope boundaries

* Regex size is bounded by the 6000-op trace limit. `n = 20` is 93 nodes, well
  inside it.
* ~~`lower_call_value` reads `config.call_returns` in only the `ResidualRef` and
  `NurseryAllocRef` arms; every `*Wrapped` arm falls through to a tail returning
  `struct_type: None`.~~ **Closed, and now on `main`**: `lower_value.rs`'s shared
  tail attaches `declared_return_struct(func)` for a `BindingKind::Ref` result,
  so every `*_ref_wrapped` policy carries the same `struct_type` the two
  early-returning arms did. It was off this plan's critical path — Stage 1 uses
  immutable fields — and it was fixed anyway, because a ref binding with no
  `struct_type` degrades its WHOLE dispatch arm to an abort stub while the
  concrete path keeps returning the right answer.
* `#[jit_interp]` requires a `match` — `codegen_trace.rs` errors with `could not
  find opcode dispatch match` without one — so a portal whose loop is a plain
  `while` has to invent a program for it. **Noted** in `jit_interp.rs`'s module
  doc; "let a matchless portal lower" stands as a majit follow-up, and nothing
  in this plan is blocked on it. Still open.

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

---

# Implementation plan

> **For agentic workers:** each task ends with an independently testable
> deliverable and its own commit. Steps use `- [ ]` for tracking. This plan and
> the design above travel together — read both.

**Goal:** reproduce the post's result in `majit/examples/regex`, fixing the two
majit gaps that block it rather than measuring around them.

**Architecture:** an `enum Node` authoring tree is lowered once to a leaked
`#[repr(C)] NodeRec` graph; a `#[jit_interp]` portal drives one `shift` per
input character; `shift` is a recursive `#[jit_inline]` helper reaching children
through `ref_fields`. `left`/`right`/`kind`/`ch`/`empty` are declared immutable so the
tree walk constant-folds; `marked` is not, so it stays a store.

**Tech stack:** Rust 2021, `majit-macros` proc macros, `majit-metainterp`,
dynasm + cranelift backends.

## Global constraints

* Cite RPython/PyPy upstream **by symbol**, never `file:line`. Comments must not
  say "As in CPython" / "CPython:".
* Every trace-shape claim is verified on **both** `--features dynasm` and
  `--features cranelift`. A one-backend verification is not a verification.
* No `push`. No branch creation or switching. Commit on the current branch.
* Commit messages state what changed, factually; trailer `Assisted-by: Claude`.
* Generated code must compile warning-clean: the `use … as _` this plan
  introduces needs `#[allow(unused_imports)]`, since the import is genuinely
  unused whenever the inherent const wins.

---

## Task 1: `__MAJIT_IMMUTABLE_FIELDS` — a declaration the proc-macro path can read

**Files:**
- Modify: `majit/majit-metainterp/src/lib.rs` (trait + blanket impl, re-exported)
- Modify: `majit/majit-macros/src/lib.rs` (`jit_immutable_fields` emission)
- Test: `majit/majit-metainterp/tests/immutable_fields_declaration.rs` (create)

**Interfaces:**
- Produces: `majit_metainterp::MajitImmutableFields` with
  `const __MAJIT_IMMUTABLE_FIELDS: &'static str = ""`, blanket-implemented for
  all `T: ?Sized`; and an inherent const of the same name on every struct
  carrying `#[jit_immutable_fields(..)]`.
- Consumed by: Task 4's generated `register_struct_layout` call sites.

- [ ] **Step 1: Write the failing test**

```rust
//! One declaration, two front ends: `#[jit_immutable_fields]` must be readable
//! by name from generated code, for a struct that carries it and for one that
//! does not.

use majit_metainterp::MajitImmutableFields as _;

#[majit_macros::jit_immutable_fields("left", "right", "kind", "ch")]
#[repr(C)]
struct Declared {
    kind: u8,
    ch: u8,
    marked: u8,
    left: usize,
    right: usize,
}

#[repr(C)]
struct OptedOut {
    x: usize,
}

#[test]
fn a_declaring_struct_exposes_its_ranks_by_inherent_const() {
    assert_eq!(
        <Declared>::__MAJIT_IMMUTABLE_FIELDS,
        "left,right,kind,ch,empty",
        "the inherent const must shadow the blanket-trait default",
    );
}

#[test]
fn a_struct_that_never_declared_falls_back_to_empty() {
    assert_eq!(<OptedOut>::__MAJIT_IMMUTABLE_FIELDS, "");
}

#[test]
fn the_free_const_the_llbc_front_end_harvests_still_exists() {
    // `harvest_immutable_fields_from_llbcs` reads this one; widening the
    // attribute must not disturb it.
    assert_eq!(_immutable_fields_Declared, "left,right,kind,ch,empty");
}
```

- [ ] **Step 2: Run it and watch it fail**

```
cargo test -p majit-metainterp --no-default-features --features dynasm --test immutable_fields_declaration
```
Expected: `error[E0599]: no associated item named __MAJIT_IMMUTABLE_FIELDS`.

- [ ] **Step 3: Add the trait and blanket impl**

In `majit/majit-metainterp/src/lib.rs`:

```rust
/// The `_immutable_fields_` declaration, readable by name from generated code.
///
/// `rclass.py InstanceRepr` reads `cls._immutable_fields_` off the class and
/// every consumer goes through that one read. The proc-macro front end has no
/// class to read, so the declaration is published as an inherent associated
/// const and this blanket default answers for every struct that never declared
/// one — generated code can then name it unconditionally.
pub trait MajitImmutableFields {
    const __MAJIT_IMMUTABLE_FIELDS: &'static str = "";
}

impl<T: ?Sized> MajitImmutableFields for T {}
```

- [ ] **Step 4: Emit the inherent const from the attribute**

In `majit/majit-macros/src/lib.rs`, extend `jit_immutable_fields`'s `quote!` —
keeping the existing free const untouched:

```rust
let ident = &item_struct.ident;
let (impl_generics, ty_generics, where_clause) = item_struct.generics.split_for_impl();
quote! {
    #item_struct
    #[doc(hidden)]
    #[allow(non_upper_case_globals, dead_code)]
    #vis const #const_name: &'static str = #joined;
    #[allow(non_upper_case_globals)]
    impl #impl_generics #ident #ty_generics #where_clause {
        /// The struct's `_immutable_fields_` ranks, shadowing
        /// `MajitImmutableFields`'s empty default so a generated
        /// `<T>::__MAJIT_IMMUTABLE_FIELDS` reads this one.
        #[doc(hidden)]
        pub const __MAJIT_IMMUTABLE_FIELDS: &'static str = #joined;
    }
}
```

- [ ] **Step 5: Run the test**

Same command. Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add majit/majit-metainterp/src/lib.rs majit/majit-macros/src/lib.rs \
        majit/majit-metainterp/tests/immutable_fields_declaration.rs
git commit -m "majit: publish jit_immutable_fields as an inherent associated const

Assisted-by: Claude"
```

---

## Task 2: carry the ranks into `BhFieldSpec`

**Files:**
- Modify: `majit/majit-metainterp/src/jitcode/assembler.rs`
  (`register_struct_layout`, `field_specs_from_layout`)
- Modify: `majit/majit-metainterp/src/jitcode/assembler.rs` — the two internal
  callers that pass a layout
- Modify: `majit/majit-metainterp/tests/struct_layout_conflict.rs` — 6 call
  sites gain the new argument
- Test: `majit/majit-metainterp/tests/immutable_fields_layout.rs` (create)

**Interfaces:**
- Consumes: Task 1's const (as a plain `&str` at the call site).
- Produces: `register_struct_layout(size, type_id, is_gc_managed, headerless,
  fields, immutable_fields: &str)`.

- [ ] **Step 1: Write the failing test**

```rust
//! `_immutable_fields_` must survive the emit-site layout table.

use majit_metainterp::JitCodeBuilder;

const TID: u64 = 0x494D_4D55_5401;

#[test]
fn declared_ranks_land_on_the_field_specs() {
    let mut builder = JitCodeBuilder::new();
    builder.register_struct_layout(
        24,
        TID,
        false,
        false,
        &[
            (0, false, "kind", 1, false),
            (8, true, "left", 8, false),
            (16, false, "version", 8, true),
        ],
        "kind,left,version?",
    );
    let spec = builder
        .struct_size_spec(TID)
        .expect("the layout was just registered");

    let by_name = |n: &str| {
        spec.all_fielddescrs
            .iter()
            .find(|f| f.name == n)
            .unwrap_or_else(|| panic!("{n} missing from the spec"))
    };

    assert!(by_name("kind").is_immutable);
    assert!(by_name("left").is_immutable);
    // The two flags are disjoint. `ImmutableRanking.pure` is false for the
    // quasi ranks — a quasi-immutable read is pinned by a guard, not by the
    // descr's pure flag — and the LLBC front end writes the same pair.
    assert!(by_name("version").is_quasi_immutable);
    assert!(!by_name("version").is_immutable);
}

#[test]
fn an_undeclared_field_stays_mutable() {
    let mut builder = JitCodeBuilder::new();
    builder.register_struct_layout(
        24,
        TID + 1,
        false,
        false,
        &[(0, false, "kind", 1, false), (2, false, "marked", 1, false)],
        "kind",
    );
    let spec = builder.struct_size_spec(TID + 1).unwrap();
    let marked = spec.all_fielddescrs.iter().find(|f| f.name == "marked").unwrap();
    assert!(!marked.is_immutable, "marked is the mutable state; it must not fold");
    assert!(!marked.is_quasi_immutable);
}
```

- [ ] **Step 2: Run it and watch it fail**

```
cargo test -p majit-metainterp --no-default-features --features dynasm --test immutable_fields_layout
```
Expected: `error[E0061]: this function takes 5 arguments but 6 were supplied`.

- [ ] **Step 3: Thread the parameter**

`register_struct_layout` gains `immutable_fields: &str` as its last parameter and
forwards it to `field_specs_from_layout`. That function parses once, outside the
per-field map:

```rust
fn field_specs_from_layout(
    fields: &[(usize, bool, &str, usize, bool)],
    immutable_fields: &str,
) -> Vec<BhFieldSpec> {
    // `rclass.py` reads `_immutable_fields_` once per class, not once per
    // field. `ImmutableRank::parse` owns the `?` / `[*]` suffix grammar and
    // stays the only place that spells it.
    let ranks: Vec<(String, majit_translate::model::ImmutableRank)> = immutable_fields
        .split(',')
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .map(majit_translate::model::ImmutableRank::parse)
        .collect();
    let rank_of = |name: &str| {
        ranks
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, r)| *r)
    };
    // … existing body, with the two hardcoded fields replaced by:
    //     is_immutable: rank_of(name).map(|r| r.is_immutable()).unwrap_or(false),
    //     is_quasi_immutable: rank_of(name).map(|r| r.is_quasi_immutable()).unwrap_or(false),
}
```

Also add the accessor the test reads:

```rust
/// The merged layout registered for `type_id`, or `None` if no emit site
/// registered one.
pub fn struct_size_spec(&self, type_id: u64) -> Option<&BhSizeSpec> {
    self.struct_size_specs.get(&type_id)
}
```

Update the two internal callers in the same file to pass `""`, and the six call
sites in `tests/struct_layout_conflict.rs` likewise — those tests are about
offset disagreement and declare nothing.

- [ ] **Step 4: Run both test files**

```
cargo test -p majit-metainterp --no-default-features --features dynasm \
  --test immutable_fields_layout --test struct_layout_conflict
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add majit/majit-metainterp/src/jitcode/assembler.rs \
        majit/majit-metainterp/tests/struct_layout_conflict.rs \
        majit/majit-metainterp/tests/immutable_fields_layout.rs
git commit -m "majit: read _immutable_fields_ ranks into the emit-site field specs

Assisted-by: Claude"
```

---

## Task 3: carry the ranks from the spec into the minted field descr

**Files:**
- Modify: `majit/majit-metainterp/src/jitcode/assembler.rs`
  (the parented field-descr mint, `CanonicalBhDescr::Field { is_immutable: false, … }`)
- Test: extend `majit/majit-metainterp/tests/immutable_fields_layout.rs`

**Interfaces:**
- Consumes: Task 2's `BhFieldSpec.is_immutable` / `.is_quasi_immutable`.
- Produces: `SimpleFieldDescr::is_always_pure() == true` for a declared field.

Task 2 alone is inert: the parented mint hardcodes `false` while already
resolving the very slot that answers it.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn a_declared_field_mints_an_always_pure_descr() {
    let mut builder = JitCodeBuilder::new();
    builder.register_struct_layout(
        24, TID + 2, false, false,
        &[(8, true, "left", 8, false), (2, false, "marked", 1, false)],
        "left",
    );
    let left = builder.field_descr_for(TID + 2, "left").expect("declared above");
    let marked = builder.field_descr_for(TID + 2, "marked").expect("declared above");

    assert!(left.is_always_pure(), "an immutable read is what folds the tree walk");
    assert!(!marked.is_always_pure(), "the mark is the mutable state");
}
```

- [ ] **Step 2: Run it and watch it fail**

```
cargo test -p majit-metainterp --no-default-features --features dynasm \
  --test immutable_fields_layout a_declared_field_mints
```
Expected: FAIL — `left.is_always_pure()` is false.

- [ ] **Step 3: Carry the ranks off the slot already looked up**

In the parented mint, extend the existing `slot.map(..)` rather than adding a
second lookup:

```rust
let (field_size, field_flag, is_field_signed, is_immutable, is_quasi_immutable) = slot
    .map(|idx| {
        let f = &parent_spec.all_fielddescrs[idx];
        (f.field_size, f.field_flag, f.is_field_signed, f.is_immutable, f.is_quasi_immutable)
    })
    .unwrap_or((scalar_size(field_type), field_flag, is_field_signed, false, false));
```

and pass `is_immutable` / `is_quasi_immutable` into `CanonicalBhDescr::Field`
instead of the two `false` literals. The miss case stays mutable: a field the
layout does not name has no declaration to honour.

Add `field_descr_for(type_id, name)` returning the minted
`Arc<SimpleFieldDescr>`, so the test reads the descr rather than the spec.

- [ ] **Step 4: Run it**

Same command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add majit/majit-metainterp/src/jitcode/assembler.rs \
        majit/majit-metainterp/tests/immutable_fields_layout.rs
git commit -m "majit: mint parented field descrs with the layout's immutability ranks

Assisted-by: Claude"
```

---

## Task 4: make `#[jit_interp]`-generated layout calls pass the declaration

**Files:**
- Modify: `majit/majit-macros/src/jit_interp/jitcode_lower/lower_vable.rs`
  — the 9 `register_struct_layout` emissions
- Test: `majit/majit-metainterp/tests/jit_interp_immutable_field_folds.rs` (create)

**Interfaces:**
- Consumes: Task 1's const, Task 3's descr.
- Produces: an end-to-end fold — a `#[jit_interp]` machine reading a declared
  ref field off a constant base emits no `GetfieldGcR` in the optimized body.

- [ ] **Step 1: Write the failing test**

A two-node chain reached from a `ref(T)` state root, walked one level per
iteration, with `left` declared immutable and no `promote` anywhere:

```rust
#[test]
fn a_declared_ref_field_folds_instead_of_guarding() {
    let before = COMPILES.load(Ordering::Relaxed);
    run(4_000);
    assert!(COMPILES.load(Ordering::Relaxed) > before, "no loop compiled — nothing to read");

    let body = optimized_loop_body().expect("a compiled loop");
    assert_eq!(
        body.iter().filter(|op| op.opcode() == OpCode::GetfieldGcR).count(), 0,
        "an immutable read off a constant base must fold, not reload: {body:#?}",
    );
    assert_eq!(
        body.iter().filter(|op| op.opcode() == OpCode::GuardValue).count(), 0,
        "folding must not be paid for with a promote guard: {body:#?}",
    );
    assert!(
        body.iter().any(|op| op.opcode() == OpCode::SetfieldGc),
        "the mutable mark must still be stored: {body:#?}",
    );
}
```

- [ ] **Step 2: Run it and watch it fail**

```
cargo test -p majit-metainterp --no-default-features --features dynasm \
  --test jit_interp_immutable_field_folds
```
Expected: FAIL on the `GetfieldGcR == 0` assertion — the read is still a reload.

- [ ] **Step 3: Pass the declaration at every emission**

Each of the 9 sites already spells `#struct_path`. Append one argument:

```rust
__builder.register_struct_layout(
    ::core::mem::size_of::<#struct_path>(),
    #tid,
    #gc_managed,
    #headerless,
    &[ /* unchanged */ ],
    {
        #[allow(unused_imports)]
        use majit_metainterp::MajitImmutableFields as _;
        <#struct_path>::__MAJIT_IMMUTABLE_FIELDS
    },
);
```

The `use … as _` must be inside the argument's block and carry
`#[allow(unused_imports)]`: whenever the inherent const wins, the import is
genuinely unused and CI builds warning-clean. A qualified
`<T as MajitImmutableFields>::…` would be wrong — it always selects the blanket
default and never sees the declaration.

- [ ] **Step 4: Run it on both backends**

```
cargo test -p majit-metainterp --no-default-features --features dynasm   --test jit_interp_immutable_field_folds
cargo test -p majit-metainterp --no-default-features --features cranelift --test jit_interp_immutable_field_folds
```
Expected: PASS on both. A pass on one only is not a pass.

- [ ] **Step 5: Guard the rest of the workspace**

```
cargo test --all --no-default-features --features dynasm,cpyext
```
Expected: no new failures. Fields that were mutable stay mutable — only structs
carrying the attribute change behaviour.

- [ ] **Step 6: Commit**

```bash
git add majit/majit-macros/src/jit_interp/jitcode_lower/lower_vable.rs \
        majit/majit-metainterp/tests/jit_interp_immutable_field_folds.rs
git commit -m "majit: pass the struct's immutability declaration from jit_interp emit sites

Assisted-by: Claude"
```

---

## Task 5: let a `#[jit_inline]` helper call itself

**Files:**
- Modify: `majit/majit-metainterp/src/jitcode/assembler.rs`
  (`add_sub_jitcode`, `add_sub_jitcode_arc`, new reserve/fill pair + memo)
- Modify: `majit/majit-macros/src/jit_interp/jitcode_lower/lower_value.rs`
  (the `INT_INLINE` arm and its sibling at the other inline emission)
- Modify: `majit/majit-macros/src/jit_interp/jitcode_lower/lower_stmt.rs`
  (two inline emissions)
- Test: `majit/majit-metainterp/tests/jit_inline_self_recursive.rs` (create)

**Interfaces:**
- Produces: `Assembler::sub_jitcode_for_builder(builder: usize) -> Option<u16>`
  and `Assembler::reserve_sub_jitcode(builder: usize) -> u16` /
  `Assembler::fill_sub_jitcode(idx: u16, jitcode: Arc<JitCode>)`.
- Consumed by: Task 7's `shift`.

Today the generated code is `let __sub_jitcode = #builder_path(__asm);` — a plain
Rust self-call, so a recursive helper overflows the stack while the jitcode is
being *built*, before any tracing.

- [ ] **Step 1: Write the failing test**

```rust
//! A helper that calls itself must be representable: RPython emits one JitCode
//! per graph and links them, it does not splice bodies.

#[majit_macros::jit_inline]
fn sum_chain(n: usize, acc: i64) -> i64 {
    // Walks a leaked chain; recursion depth is bounded by the data.
    if n == 0 { acc } else { sum_chain(n - 1, acc + n as i64) }
}

#[test]
fn a_self_recursive_inline_helper_builds_one_sub_jitcode() {
    // Reaching this line at all is the first assertion: before the memo, the
    // builder recursed until the stack was gone.
    let before = COMPILES.load(Ordering::Relaxed);
    let got = run(4_000);
    assert_eq!(got, expected(), "the compiled answer must match the interpreter");
    assert!(COMPILES.load(Ordering::Relaxed) > before);
}
```

- [ ] **Step 2: Run it and watch it fail**

```
cargo test -p majit-metainterp --no-default-features --features dynasm \
  --test jit_inline_self_recursive
```
Expected: the process aborts with `thread ... has overflowed its stack`.

- [ ] **Step 3: Add the memo and the reserve/fill pair**

`add_sub_jitcode_arc` is a thin `push_descr_entry(RuntimeBhDescr::JitCode(..))`
with no memo. Add a `HashMap<usize, u16>` keyed on the builder's function-pointer
address, plus:

```rust
/// The sub-jitcode index already reserved for `builder`, if this assembler has
/// begun lowering it.
///
/// `codewriter.py CodeWriter.make_jitcodes` mints one JitCode per graph and
/// links callees by index, so a graph that calls itself resolves to the index
/// it is already being built under. Splicing the body instead — which is what a
/// plain builder self-call does — does not terminate.
pub fn sub_jitcode_for_builder(&self, builder: usize) -> Option<u16> { … }

/// Claim a descr slot for `builder` before its body is lowered, so a self-call
/// inside that body finds an index to link against.
pub fn reserve_sub_jitcode(&mut self, builder: usize) -> u16 { … }

/// Publish the finished jitcode into the slot `reserve_sub_jitcode` claimed.
pub fn fill_sub_jitcode(&mut self, idx: u16, jitcode: std::sync::Arc<JitCode>) { … }
```

The reserved entry must be a placeholder the descr pool can hold; fill it before
`try_finish`, and have `try_finish` decline the JitCode if any reservation was
never filled — an unfilled slot is a linking bug, and declining is how this
builder reports one (it runs at runtime and cannot panic).

- [ ] **Step 4: Emit through the memo**

At all four inline emissions, replace the unconditional build:

```rust
let __builder_fn: fn(&mut majit_metainterp::Assembler) -> majit_metainterp::JitCode =
    unsafe { std::mem::transmute(__inline_builder) };
let __key = __builder_fn as usize;
let __sub_idx = match __asm.sub_jitcode_for_builder(__key) {
    Some(idx) => idx,
    None => {
        let __idx = __asm.reserve_sub_jitcode(__key);
        let __sub_jitcode = __builder_fn(__asm);
        __asm.fill_sub_jitcode(__idx, std::sync::Arc::new(__sub_jitcode));
        __idx
    }
};
```

`trailing_return_info` is currently read off the freshly built jitcode; on the
memo-hit path read it off the stored `Arc` instead, so the return kind is still
checked for a self-call.

- [ ] **Step 5: Run it on both backends**

```
cargo test -p majit-metainterp --no-default-features --features dynasm   --test jit_inline_self_recursive
cargo test -p majit-metainterp --no-default-features --features cranelift --test jit_inline_self_recursive
```
Expected: PASS on both.

- [ ] **Step 6: Guard the workspace, then commit**

```
cargo test --all --no-default-features --features dynasm,cpyext
```

```bash
git add majit/majit-metainterp/src/jitcode/assembler.rs \
        majit/majit-macros/src/jit_interp/jitcode_lower/lower_value.rs \
        majit/majit-macros/src/jit_interp/jitcode_lower/lower_stmt.rs \
        majit/majit-metainterp/tests/jit_inline_self_recursive.rs
git commit -m "majit: memoize inline-helper jitcode construction so a helper can call itself

Assisted-by: Claude"
```

---

## Task 6: the matcher, without the JIT

**Files:**
- Create: `majit/examples/regex/Cargo.toml`
- Create: `majit/examples/regex/src/regex.rs` (`enum Node`, `lower`, builders)
- Create: `majit/examples/regex/src/interp.rs` (`shift`, `matches`, `reset`)
- Modify: `Cargo.toml` (workspace members)

Source is already written and rustc-verified in the scratchpad
(`rs/regex_check.rs`): 28/28 correctness vectors, 93 nodes at `n = 20`, balanced
depth 8 / left-associated depth 26, 0 disagreements over 1560 strings.

**Interfaces:**
- Produces: `regex::{Node, lower, bench_regex, bench_regex_left, nonmatching}`,
  `interp::{shift, matches, reset}`, `NodeRec` with
  `#[jit_immutable_fields("left", "right", "kind", "ch", "empty")]`.
- Consumed by: Tasks 7–9.

- [ ] **Step 1: Port the verified transcription**

`Cargo.toml` mirrors `majit/examples/tla/Cargo.toml` exactly — `publish = false`,
features `cranelift` / `dynasm`, deps `majit-ir`, `majit-metainterp`,
`majit-macros`. Add `"majit/examples/regex"` to the workspace `members`.

The module header cites the post by title, and states the two things a reader
must not have to guess: the marks live **in the nodes**, and `empty` is computed
once at lowering time.

- [ ] **Step 2: Bring the 28 vectors over as `#[test]`s**

Table-driven, one `#[test]` walking `(pattern, input, expected)`. Include the two
that corrected an earlier wrong expectation: `((abc)*|(abcd))(d|e)` matches
`"abcde"` (via `abcd` + `e`) and does **not** match `"abcdf"`.

- [ ] **Step 3: Run**

```
cargo test -p regex --no-default-features --features dynasm
```
Expected: all vectors pass.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml majit/examples/regex/
git commit -m "majit/examples/regex: marked-regex matcher over a lowered node graph

Assisted-by: Claude"
```

---

## Task 7: the `#[jit_interp]` portal and the recursive `shift`

**Files:**
- Create: `majit/examples/regex/src/jit_interp.rs`
- Modify: `majit/examples/regex/src/main.rs` (module wiring)

**Interfaces:**
- Consumes: Tasks 1–5 (the two majit fixes), Task 6's `NodeRec`.

- [ ] **Step 1: Declare the machine**

`state_fields = { root: ref(NodeRec), input: [int; …], pos: int, result: int }`,
`ref_fields = { NodeRec::left => NodeRec, NodeRec::right => NodeRec }`,
`int_fields = { NodeRec::kind => u8, NodeRec::ch => u8, NodeRec::empty => u8,
NodeRec::marked => u8 }`. Greens are the regex identity and the pc; reds are the
position and the mark state. `#[jit_interp]` requires a `match`, so the portal is
a single-arm `match` over a one-instruction opcode — note that shape tax in the
file header.

- [ ] **Step 2: Write `shift` as a recursive `#[jit_inline]` helper**

Structurally the same five arms as `interp.rs`'s `shift`, so the two can be read
side by side.

- [ ] **Step 3: Verify it compiles a loop, warm equals cold**

```
cargo test -p regex --no-default-features --features dynasm
cargo test -p regex --no-default-features --features cranelift
```

- [ ] **Step 4: Commit**

```bash
git add majit/examples/regex/src/
git commit -m "majit/examples/regex: jit_interp portal driving a recursive shift

Assisted-by: Claude"
```

---

## Task 8: gate the trace shape — the post's actual claim

**Files:**
- Modify: `majit/examples/regex/src/jit_interp.rs` (`#[cfg(test)]` section)

A speed number is not the claim. The claim is that the tracer performs subset
construction, and that is a statement about the optimized trace.

- [ ] **Step 1: Assert the shape**

- `COMPILES > 0` and `closes_a_loop()`.
- Warm and cold agree on the same binary, over the full 28-vector table.
- Zero `GetfieldGcR` in the optimized loop body — the walk folded.
- Zero `GuardValue` in the body — it folded without being promoted.
- `SetfieldGc` count equals the number of nodes whose mark can change, and every
  such store's base is a `ConstPtr`.
- No `[jit] degraded dispatch arm:` line, and no
  `[jit] unconsulted field declaration:` line.

- [ ] **Step 2: Run on both backends, and record the op counts in the header**

- [ ] **Step 3: Commit**

```bash
git add majit/examples/regex/src/jit_interp.rs
git commit -m "majit/examples/regex: gate the optimized trace shape on both backends

Assisted-by: Claude"
```

---

## Task 9: the benchmark table

**Files:**
- Modify: `majit/examples/regex/src/main.rs`

- [ ] **Step 1: Report interp and JIT chars/s from one run**

Print the measured rows from Stage 5 alongside, each labelled with what it is.
The `re` row must carry its non-comparability inline — `re` can bail out of a
non-match early; the marked matcher always scans the whole string. Print the
2010 figures labelled as 2010 hardware.

- [ ] **Step 2: Run and fill Stage 5's `majit JIT` row in this document**

- [ ] **Step 3: Commit**

```bash
git add majit/examples/regex/src/main.rs regex-jit.plan.md
git commit -m "majit/examples/regex: benchmark harness and the measured table

Assisted-by: Claude"
```

---

## Self-review

* **Spec coverage.** Stage 0 → done, no task. Stage 1 → Tasks 1–4. Stage 2 →
  Task 5. Stage 3 → Tasks 6–7. Stage 4 → Task 8. Stage 5 → Task 9. The
  `*Wrapped` / `call_returns` gap and the matchless-portal tax are recorded as
  findings to file, explicitly out of scope.
* **Placeholders.** None: every code step carries the code, every test step
  carries the command and the expected result.
* **Type consistency.** `register_struct_layout`'s sixth parameter is
  `immutable_fields: &str` in Tasks 2 and 4. `ImmutableRank::parse(&str) ->
  (String, ImmutableRank)` is used only in Task 2. `__MAJIT_IMMUTABLE_FIELDS` is
  `&'static str` in Tasks 1 and 4. The reserve/fill pair is spelled identically
  in Task 5's steps 3 and 4.
* **Ordering.** Tasks 1→2→3→4 are strictly sequential (each is inert without its
  predecessor). Task 5 is independent of 1–4 and may run in parallel. Task 6 is
  independent of all majit work. Task 7 depends on 4, 5 and 6.

---

# Execution log

## Tasks 1–4 — landed

**Task 1** (`cb4b242d799`) — `MajitImmutableFields` trait + blanket impl in
`majit-metainterp`, and `#[jit_immutable_fields]` additionally emitting the
inherent `__MAJIT_IMMUTABLE_FIELDS` const. 5 tests.

**Tasks 2–4 landed as one commit, not three.** The plan sequenced them
2 → 3 → 4, but widening `register_struct_layout`'s signature breaks every caller
in the same build, and the proc macro is one of those callers — so the parameter,
the macro emission and the callers had to move together or the workspace would
not compile between commits. Task 3's descr-mint change is separable and is
reflected in its own test module.

**The call sites were wider than the plan said.** Beyond the 9 emissions in
`lower_vable.rs`, the arity change reached:

* `new_struct` — a 10th emission, in `lower_value.rs`'s `lower_struct_value`.
  Given the declaration too, so that a `new_struct` registering a struct's layout
  cannot pre-empt the declaring site.
* `newlist_clear` — an internal caller synthesizing a `list` header. Passes `""`:
  the JIT synthesizes that struct and there is no declaring type to read.
* 8 call sites in `tests/struct_layout_conflict.rs`, 3 `register_struct_layout` and
  3 `field_specs_from_layout` sites in `assembler.rs`'s own unit tests, and 4
  `new_struct` sites in `pyjitpl/dispatch.rs`'s unit tests.

**One design change against the plan.** `register_struct_layout` now applies the
declaration to an already-registered spec *before* its "every offset known" early
return, and only ever raises a rank. Without that, a struct whose layout was first
registered by a site carrying no declaration would keep `is_immutable: false`
forever — emit sites lower in an arbitrary order, so the declaration must not
depend on arriving first. Two tests pin both directions.

**A wrong assertion in this plan, corrected.** Task 2's snippet asserted
`is_immutable` for a `?` entry. It is false: the ranks are disjoint —
`ImmutableRanking.pure` excludes the quasi variants, and the LLBC front end writes
the same pair from `ImmutableRank::is_immutable` / `is_quasi_immutable`. Note
`StructFieldLayout::is_immutable()` on the translate side uses a *different* rule
(`rank.is_some()`), so "immutable" is not one predicate repo-wide.

## Task 6 — landed (`aa5b55f91af`)

`majit/examples/regex` with `regex.rs` / `interp.rs` / `main.rs`, 11 tests,
asserting the fixture facts (93 nodes; depth 8 balanced, 26 left-associated).

`empty` was added to the declaration afterwards. It was omitted from this plan's
own field list, and that was an error: `shift`'s `Sequence` arm reads both
children's `empty`, so leaving it undeclared would survive the fold as one reload
per node — exactly the thing the experiment is supposed to eliminate.

## Task 5 — landed, and the plan's framing of it was wrong

The plan called this "memoize helper JitCode construction". That is not
sufficient. `add_sub_jitcode` pushes `RuntimeBhDescr::JitCode(Arc<JitCode>)` into
the **current builder's own** descr pool, so direct self-recursion needs a JitCode
whose pool holds an `Arc` to itself — a reference cycle, not a caching problem.
RPython does not hit it because `CodeWriter.make_jitcodes` / `get_jitcode` mint an
empty JitCode **shell** per graph before any body is written; majit's
`JitCodeBuilder` produces a `JitCode` by value at the end.

What landed is that shell, spelled in Rust as `Arc::new_cyclic`: it hands the
body a `Weak` to the `Arc` that does not exist yet, which is exactly what
`get_jitcode` registering before writing buys upstream.

* `#[jit_inline]` now emits a `_shared` entry point beside `_with_asm`.
  `_shared` is `CallControl.get_jitcode`: it consults an assembler-wide memo
  keyed by the address of a per-helper `AtomicU8` static, registers the helper's
  identity through `begin_inline_jitcode` before assembling, and publishes with
  `finish_inline_jitcode`. `_with_asm` still assembles unconditionally and keeps
  its signature.
* `RuntimeBhDescr` gained `JitCodeBackEdge(Weak<JitCode>)`, and
  `as_jitcode_owned()` resolves either variant. `as_jitcode()` stays narrowed to
  owning edges — `jitdriver.rs build_jitcode_registry` asserts each sub-jitcode
  it reaches is unnumbered, and a back edge is the same jitcode reached twice.
  The three resolvers that answer "what does this `j` operand execute"
  (`pyjitpl/dispatch.rs`'s `BC_INLINE_CALL`, `blackhole.rs`'s
  `read_inline_call_jitcode` and `handler_inline_call_pyre_nested`) went to
  `as_jitcode_owned`. The `None` window is confined to the helper's own
  assembly, which never runs code.
* The self-edge is a `Weak`, so a recursive helper is not a cycle of owning
  references that never drops.

`tests/jit_inline_self_recursive.rs` is the gate: a `sum_chain` that calls
itself, driven by a `#[jit_interp]` portal. Reaching the first assertion is the
first result — before this the process died in `dispatch_sum` with a build-time
stack overflow. The measured loop unrolls the whole 4-link walk with no call of
any kind, identical on both backends.

**A pre-existing test had to change with it.** `_shared`'s memo means two arms
calling one helper now hold the same `Arc`, and `add_sub_jitcode_arc` deduplicates
it — `assembler.py Assembler._encode_descr` memoises every descr through
`_descr_dict`, and upstream's `JitCode` **is** an `AbstractDescr`, so upstream
gives those two arms one index too. `jit_interp_discarded_inline_result.rs`'s
`the_discarded_call_is_spliced_into_a_jitcode_body` counted descr entries and
expected two. Counting cannot answer that question under a memo — one entry is
what both arms splicing and only the binding arm splicing both look like — so the
fixture grew a second portal, `dispatch_discard_only`, whose only caller of
`pop_stack` is the discarding arm. The memo itself is now pinned separately.

**`MAX_INLINE_DEPTH = 10` is not the ceiling this plan feared.** The depth-26
left-associated tree traces to the same 194-op body as the depth-8 balanced one.

## Tasks 7–9 — landed

`majit/examples/regex/src/jit_interp.rs`: the recursive `#[jit_inline] shift`,
the `#[jit_interp]` portal, 5 tests, and `main.rs`'s benchmark.

**Three spellings the plan did not have.**

* Sub-word fields need explicit casts on both sides. `int_fields` declares the
  real width (`NodeRec::kind => u8`), and the concrete path keeps the Rust type,
  so a read is `n.kind as i64` and the store is `n.marked = m as u8`. Both cast
  forms lower.
* **A value-position `match` on constants was silently mis-lowered, and it is
  fixed rather than worked around.** `lower_match_value` read its arms with
  `extract_pat_literals`, which takes int literals only; a constant path parses
  as `Pat::Ident`, and the classifier read every such arm as the *catch-all
  binding*. Four constant arms therefore produced no guarded arms and a
  `default_arm` set to the last of them, so the jitcode computed that one arm
  for every discriminant while the concrete interpreter — real Rust — stayed
  right. Only a warm-versus-cold answer could see it.

  The fix reads arms with `extract_pat_value_tokens`, the same function the
  sibling `lower_dispatch_chain` already used, and treats a `Pat::Ident` as a
  binding only when its name holds a lower-case letter — `non_upper_case_globals`
  is what makes that decidable at proc-macro time. `shift` now spells its arms
  `KIND_CHAR` / `KIND_SEQUENCE` / … and traces to the same 194-op body.
  `tests/jit_interp_match_const_patterns.rs` is the gate.
* **The input must not be an `[int; virt]` state field.** It was, and at
  `1 << 20` characters the run died in `resumecode.py Writer.append_int` —
  `append_int: value 1048577 out of i16 range`. A virtualizable array's elements
  ride every guard's `vable_array` resume section, so the section is O(input
  length) and the tagged short overflows past 32767. The fix is the
  non-virtualizable vocabulary: a `ref(Input)` state field plus
  `array_fields = { Input::data => u8 }`, which also drops the `u8 → i64`
  widening pass the `Vec<i64>` needed. That is the same distinction
  `ArrayFieldEntry`'s doc comment draws, read the right way round.

**The trace.** The peeled loop body — what runs per input character — measured
on `(a|b)*a(a|b){20}a(a|b)*`, 93 nodes, identical on dynasm and cranelift and
identical for both associations:

```text
194 ops: 0 getfield_gc_r, 24 getfield_gc_i, 93 setfield_gc,
         2 int_eq, 0 guard_value, 1 getarrayitem_gc_i
```

Zero pointer reads: the tree walk's 92 edges are gone. 93 stores is one mark per
node, so the whole tree is in the trace. The root promote and the input buffer's
base pointer are loop invariant and sit in the preamble. `2 int_eq` for 46 `Char`
nodes is the subset construction — a node whose incoming mark the optimizer
proved constant zero has no comparison left to make.

## Stage 5's table, filled

Release build, 1,048,576-character non-matching input, five runs:

| row | chars/s |
|---|---|
| Rust interp over `NodeRec`, no JIT | 8,712,202 – 9,263,315 |
| **majit JIT, regex promoted** | **46,710,465 – 47,751,626** |
| ratio | **5.0 – 5.5x** |

The Rust-interp row supersedes this document's earlier 1,705,377 – 2,582,170.
That figure came from the scratchpad probe built with a bare `rustc`, i.e.
unoptimized; the row above is `--release` and is the one the JIT number is
divided by.

The comparable quantity in the post is its own JIT-over-no-JIT ratio,
16,500,000 / 720,000 = 22.9x. Ours is 5.0 – 5.5x against a matcher that
`rustc -O` had already optimized, where the post's denominator was RPython
translated to C. The absolute rows are not comparable across the two machines
and sixteen years, and `main.rs` says so where it prints them.

## The filed findings, closed (`f47e7c32fbb`, `4bfc01c34aa`)

Both were recorded above as out of scope for the example. Neither is any more.

**`call_returns` through a wrapped ref policy.** The tail of `lower_call_value`
now asks `call_returns` when the result is a ref, and the three lookups share
one `declared_return_struct`. Measured on `jit_interp_call_returns_ref.rs`,
which is the first fixture in this corpus to use `call_returns` or any ref call
policy at all — which is why the gap survived. The subject arm went through
`residual_ref_wrapped` and the control through `residual_ref`, same callee
shape, same declaration:

| | before | after |
|---|---|---|
| `OP_PLAIN` (control) | lowered | lowered |
| `OP_WRAPPED` (subject) | abort stub | lowered |
| `BC_GETFIELD_GC_I` in the portal's bodies | 1 | 2 |
| the answer | 14 | 14 |

The answer is the point: the concrete path reads the same declaration through
`RefFieldRewriter`, so the degraded arm still computed the right sum and only
the arm census could see the loss. The plan's estimate — "a ~5-line hoist of
that lookup into the shared tail" — was right.

**A `ref(T)` state field bound to a local.** Found while writing the fixture
above, and two-sided rather than silent. `state.sel.value` lowers as one
expression because that path resolves the pointee from the `ref(T)`
declaration itself; `let n = state.sel; n.value` reached neither walk's copy of
it. `RefFieldRewriter` recorded a local ref only for a ref-kind field of a ref
base or a `call_returns` call, so the concrete arm did not compile at all
(`usize` has no fields); the JIT read discarded the struct path
`state_ref_scalars` already holds. One line each, gated by
`jit_interp_ref_local_binding.rs`: with only the concrete side fixed, the
local-binding arm degraded to a stub and the bodies carried 1 `BC_GETFIELD_GC_I`
where 2 were due.

`lower_state_field_assign` still discards that path deliberately — it consumes
a ref, it does not produce one.

## Three more, found by pulling on the dispatch lowerer (0825)

Each was found the same way: write the fixture the corpus never had, then read
the instrument rather than the answer.

**The dispatch default arm's edge.** `lower_dispatch_chain` sent every
unmatched opcode to `default_label`, bound at the portal function's trailing
return. That is the `_ => break` reading, and it is what 55 of this corpus's
default arms plus 30 ending in a divergent macro mean — so nothing here
witnessed the other two spellings. `_ => {}` means "go round again" and
`_ => { .. }` means "run this"; both got the return, the walk reported a
finished frame, and the portal answered partway through. Neither
`degraded_dispatch_arms()` nor the arm census saw it: from the lowerer's side
nothing had refused, the arm was simply never emitted.

| default arm | cold | warm |
|---|---|---|
| `_ => {}` | 1200 | 27 |
| `_ => { acc += 100 }` | 41200 | 827 |

Fixed by classifying the arm and routing the edge three ways — the return
label, the loop-start label, or a label bound at the arm itself, lowered as an
arm with no opcode test in front of it. `tests/jit_interp_default_arm_edge.rs`
covers all three plus the same body under `switch_dispatch = true`, which had
no coverage at all: the one machine here that sets it writes `_ => break`.

braininterp and dualtape were the machines to check, since both write
`_ => { pc += 1; }`. Neither changed: `pc += 1` off a constant-folded green is
carryable, and their degraded set stays exactly `["b'['", "b']'"]`, whose
`pc = target` from `find_matching_open` is the shape that genuinely cannot be.

**Which `match` is the dispatch.** `find_dispatch_match` answered "the one with
the most arms", over every match anywhere in the portal function. Arm count is
not a property of being the dispatch. A five-arm `let x = match ..` beside a
three-opcode dispatch takes its place, `classify_arms` reads that match's arms
as the opcodes, and the two pre-dispatch walkers — which locate their loop by
asking which one holds the dispatch match — find no loop and lower nothing.
Measured: cold 1200, warm 24, compiling clean with no degraded arm and no
warning. Fixed by searching the portal loop's body after its merge point, which
is the region the rest of the machinery already walks; the whole-function search
stays as a fallback for shapes outside that rule.
`tests/jit_interp_dispatch_match_choice.rs` runs one interpreter with the setup
match in four positions and reads the arm census as well as the answer.

**A macro invocation is opaque to the unknown-local walk.** `lower_local`'s last
resort emits an initialiser it could not lower verbatim into the `__builder`
block and loads it as a constant. The guard on that contract —
`expr_references_unknown_local` — reads a bare lowercase identifier as a user
local and refuses. It had no `Expr::Macro` arm, so a macro fell to the catch-all
and reported no reference, which it cannot know: the tokens are opaque to it.
`let bump = pick!(sel);` ahead of the merge point therefore put
`match sel { .. }` inside `__dispatch_jitcode_*`, whose parameters are the
assembler and the driver index, and the build failed at a line where `sel`
plainly is in scope. The sibling `stmt_contains_call` already scores a macro as
a hit for the same reason. Gated by `tests/jit_interp_macro_init_is_opaque.rs`,
whose compiling at all is the gate.

## The corpus's degraded arms, swept (0825)

`MAJIT_LOG=1` over the 13 examples plus `regex`, deduped: 13 degraded arms over
8 machines, five shapes. Nothing left in it is a majit defect.

| shape | arms | verdict |
|---|---|---|
| `find_matching_open` / a `while` scan over `program` | braininterp, dualtape `b'['` `b']'` | data-dependent, genuinely unlowerable |
| `pc += 8` after `i64::from_le_bytes(..)` | tiny2, tiny3 | the deliberate green-writeback refusal |
| `storage_roll(state.stack.as_mut_ptr() as usize, ..)` | tl, tlc `ROLL` | a raw-pointer argument has no lowering |
| `state.regs = vec![0; n]` | tlr `ALLOCATE` | allocation, outside the vocabulary |
| `state.stack[i] = inputarg` | tl, tlc, spcount `PUSHARG` | below |

`PUSHARG` looks like the `call_returns` shape — a `reds` key that this whole
corpus uses zero times, beside three machines whose comments call `inputarg`
"red by nature" — and is not. `red_schema` only *types* a binding that already
exists; a declared red absent from the body panics, matching
`decode_hp_hint_args`. It creates nothing. `inputarg` is a portal parameter, so
it is an unknown local and the store refuses fail-closed, which tl's own test
pins as `RefusalKind::UnlowerableStmt`. Carrying a portal parameter as a trace
red is a feature — a binding at entry, a place in the merge-point payload, and
the runtime plumbing to pass it — and a `state_fields` int is the workaround.

`_ => return X` on the default edge was probed and left alone. `default_label`
is bound at a return of the portal's *trailing expression*, so an explicit
`return SENTINEL` there would answer `state.acc` instead. No arm in this corpus
spells it, and the probe passes: the arm ends the portal, so it runs at most
once per entry and is never hot enough to be traced. `_ => break` is correct on
that edge by construction, because a break falls to the same trailing
expression.

Still open from the plan: "let a matchless portal lower".

## The fourth majit defect: `_forwarded` held a `Weak` (0826, `1e97e4ba14d`)

`Forwarded::Op` / `Forwarded::InputArg` carried a `Weak`, so reaching a
forwarding target depended on *some other* holder keeping it alive —
`OptContext`'s `resop_refs`, `phase1_emit_ops` or `new_operations`. A target
held by none of them was dropped mid-optimization; `get_box_replacement` then
took its documented "dropped target: terminate at `cur`" early return and
returned an operand whose own slot was still `Forwarded::Op`, which
`getnullness` asserts cannot happen:

```text
chain walker terminal: fw=Op(upgrades=false) resolved_tp=Int
orig_tp=Int resolved_is_same_as_orig=true
```

`upgrades=false` with `resolved_is_same_as_orig=true` is the whole diagnosis:
the arm fires, the target is gone, and the walk made **zero hops**. Get it by
replacing the `unreachable!` with a panic printing `w.upgrade().is_some()` and
`resolved.same_box(op)`; neither is visible from the message alone.

Making the dead-`Weak` arms fall through like `Forwarded::None` removes the
crash and **answers wrong** — `((abc)*|(abcd))(d|e)` on `"abcabcabcd"` gives
`cold=true warm=false` on dynasm. The drop is the defect; the `unreachable!`
is its second face.

`resoperation.py set_forwarded` is a plain attribute assignment, and the target
is also reachable from the trace `operations` list, so upstream's slot is a
strong reference. The port now is too. No new cycle class: the chain walker
already loops with no visited set, so a cycle would hang today, and
`Forwarded::Info(OpInfo)` already carries strong edges. About fifteen comments
across `optimizeopt/{mod,optimizer,rewrite}.rs` documented those registries as
forwarding keep-alives; they are producer-lookup and partial-trace stores, and
are restated rather than deleted.

Reproducing it needs the example, not a reduction. The trigger is an `int_or`
of two `int_is_true` results reaching `optimize_int_is_true` — in
`shortcircuit.rs`'s `Repetition` arm, `let inner_mark = (mark != 0) as i64 |
(n.marked as i64 != 0) as i64;`. A hand-written fixture with the same shape
(self-recursive `#[jit_inline]` over a tree, a mutable field, a promoted root,
that `int_or` as the recursive call's argument) does **not** reproduce, because
the trigger depends on which ops the optimizer eliminates.

So the gate is the invariant, not the route:
`majit-ir`'s `operand::tests::a_forwarded_target_survives_every_other_reference_being_dropped`
walks a two-hop chain after dropping every other reference to both targets.
Under the `Weak` slot it lands back on the head; measured red before, green
after.

After the fix that spelling compiles and agrees with the interpreter in the
`Repetition` arm alone, in the `Sequence` arm alone, and in all four arms at
once, on both backends — so the separately reported dynasm
`RegisterManager.loc: box IntOp(N) not found` from `consider_binop_j2` is gone
with it and was never a second defect.

## Why the branching portal got no bridge, and the fix (0826)

The short-circuit A/B's headline was that the branching body fails a guard on
essentially every character and no bridge is ever built. `MAJIT_BRIDGE_DEBUG=1`
named every one of the 19 bridge attempts, and they were all the same give-up:

```text
[bridgeB] multi-frame resume (7 frames) — only the root frame is seedable,
          giving up on the bridge
```

13 at 7 frames, 3 at 4, 2 at 6, 1 at 5 — the frame count is the recursion depth
of the inlined `shift` at the failing guard. `start_bridge_tracing` refused
whenever a state-field driver's resume data spanned more than one frame,
because `setup_bridge_sym` seeds the root frame only. That applied to any
`#[jit_interp]` machine whose hot loop calls an inlined recursive helper.

**Fixed.** `resume.py rebuild_from_resumedata` is the shape:

```python
while not resumereader.done_reading():
    jitcode_pos, pc = resumereader.read_jitcode_pos_pc()
    jitcode = metainterp.staticdata.jitcodes[jitcode_pos]
    f = metainterp.newframe(jitcode)
    f.setup_resume_at_op(pc)
    resumereader.consume_boxes(f.get_current_position_info(),
                               f.registers_i, f.registers_r, f.registers_f)
```

One real `MIFrame` per section, each at its own pc with its own registers,
appended in stream order — which is caller-first, because `opencoder.py
SnapshotIterator.__init__` reversed on the writer side — so the walk ends up
resuming in the innermost frame with its callers underneath it. majit now does
exactly that. Each section resolves its jitcode through the flat registry
(`resume.py`'s `jitcodes[jitcode_pos]`, `MetaInterp::jitcodes`) — the same
registry the blackhole rebuild already used — and reads its own liveness at its
own pc, so nothing is parent-relative.

Two fields a caller frame needs are deliberately NOT in the resume stream, and
recovering them is the whole difficulty. A frame below the top is suspended
inside a `BC_INLINE_CALL`, and `opencoder.py _ensure_parent_resumedata` reads a
parent's liveness with `in_a_call=True`, which blanks the result register
first, because nothing has written it yet. RPython recovers it from the
bytecode instead — `pyjitpl.py MIFrame.make_result_of_lastop` takes the
register from `ord(self.bytecode[self.pc - 1])` and its kind from
`self.jitcode._resulttypes[self.pc]`. majit's grouped `BC_INLINE_CALL` encoding
puts three optional return slots at the end rather than one register byte, so
"read the byte before pc" becomes `JitCode::inline_call_ending_at`: decode the
instruction forwards from the only start that can produce an instruction ending
at that pc, solving for `num_args` off the width formula and requiring the
match to be unique, the encoded argument count to be the assumed one, and the
recorded `_resulttypes` entry to agree with the slots decoded. Everything that
does not check out is a decline, which lands back on the blackhole — the arm
that was already serving these guards.

The decline sites all log under `MAJIT_BRIDGE_DEBUG=1`, following
`read_frame_liveness_reg_indices`'s own soft-decline policy, so "no bridge" is
never silent again.

One thing found by running it: the callee of a self-recursive `#[jit_inline]`
helper resolves through a **`Weak`** descr (`Arc::new_cyclic`), so
`RuntimeBhDescr::as_jitcode` misses exactly the recursion this walk exists to
resume through. `BC_INLINE_CALL` itself uses `as_jitcode_owned`; the rebuild
now does too. With `as_jitcode` it declined all 10 attempts at depth 1.

Measured on the same 4096-character run, both backends identically:

| | before | after |
|---|---:|---:|
| guard failures | 4090 | 4080 |
| bridges compiled | 0 | 10 |
| traces aborted | 19 (all bridge give-ups) | 0 |
| trace ids seen failing | 1 | 1–11 |

The deopt rate barely moves, and that is the honest result rather than a
disappointment: a bridge covers one more path through the mark pattern, the
bridge then exits at a guard of its own, and the next bridge grows off that.

Ten is a rate, not a ceiling, and eight times the input is what tells those
apart — `the_bridge_count_is_set_by_trace_eagerness_not_by_a_ceiling` measures
it, identically on both backends:

| input | guard failures | bridges |
|---:|---:|---:|
| 4,096 | 4,080 | 10 |
| 32,768 | 32,676 | 84 |

8.0x the failures, 8.4x the bridges, and 0.997 failures per character at both
lengths. The masking portal — the same input, the same tree, the counters now
wired on that side too — answers **1 guard failure, 0 bridges** over the same
4096 characters, which is what makes 4080 a verdict rather than a large number.

`trace_eagerness` is 200, so one guard must fail 200 times to be worth
a bridge and a pass of `n` characters can grow at most about `n / 200` of them
however many mark patterns the tree has. Bridging is not converging on the
pattern and stopping; it is trailing it at a fixed rate the deopt outruns.
What changed is that the post's "you get a lot of assembler code generated" now
actually happens here.

Two things were wrong in the tally on the way to that answer, and both are
fixed. The bridge give-ups called `abort_trace` without staging a reason, so
the `abort_trace` fallback counted every one of them into the
nothing-was-staged slot — the slot whose whole purpose is to separate the
unclassified aborts from the bridge give-up that shares `ABORT_BRIDGE`'s
integer. `compile.py giveup()` raises `SwitchToBlackhole(Counters.ABORT_BRIDGE)`,
so both sites now stage it, under a named `AbortReason::Bridge` that maps to
the same integer; the hook payload does not move. The `resume data could not be
rebuilt` site also had no log line at all, where its sibling had one. Neither
change has a fixture: they are diagnostic accuracy, with no behavioural
surface, and the shape that reaches them is the example itself.

## The copy that would have miscompiled, and the decline that replaced it (0827)

The multi-frame rebuild above shipped with one block that did not belong to it.
For every non-root frame it copied the ROOT frame's reserved identity-slot
registers over the callee's, justified by "only the root frame's identity slots
are meaningful and deopt re-derives the rest from the single reconstructed root
virtualizable". A parity review flagged it; four independent readers took it
apart. Both halves of the justification are false:

* **Nothing re-derives them.** `resume.py write_an_int` writes each section's
  decoded values — the blanked `Const(0)` placeholders included — straight into
  that frame's own register bank. The only root-to-callee propagation deopt
  performs is on the `virtualizable_ptr` / `virtualizable_info` **fields**,
  never the register file. So a compiled bridge would have seen the root's
  values where the blackhole exiting the same guard sees zero.
* **The root's registers are not a substitute.** The trim's own note records
  the counterexample already: a dualtape sub-frame carries the virtualizable
  identity in int reg 2, while the root's int reg 2 is the scalar `pb` under
  the canonical layout. Copying would have fed `pb` in where the callee reads
  an identity, and that identity is used as an array index.
* **Upstream gives no cover.** `rebuild_from_resumedata` fills every frame from
  its own section and consumes the vable once, before the loop.

The range is empty for all 99 `#[jit_interp]` symbols but one, and that one
(`dualtape`) compiles nothing, so the block was dead — but dead code that would
be wrong if reached is a trap, not a nullity. It is replaced by a decline in
`bridge_from_guard_resume_position`: `resume.frames.len() > 1 &&
S::reserved_int_identity_range().is_some()` returns `None`, which falls through
to `compile.py:711 resume_in_blackhole`.

Two things about the shape, both of which a first attempt got wrong:

* **The decline must be at the driver, not inside the walk.** Returning
  `TraceAction::Abort` from `trace_jitcode_at_resume_framestack` is not a
  decline: the walk publishes its handoff only after `run_to_end`, so an early
  return leaves both handoffs empty, and the caller resumes at the loop header
  with no blackhole at all. `None` at the driver is the decline, and it is the
  upstream shape — `pyjitpl.py _handle_guard_failure` reaches `compile.py
  giveup()` once the frames are known.
* **The range must come from the sym, not the layout.** `JitState::Sym` carries
  no `JitCodeSym` bound, and the split-unaware `StateFieldLayout` reports a
  non-empty range for regex too — gating on it would refuse every multi-frame
  bridge and erase the whole 0→10. Hence the `JitState::reserved_int_identity_range`
  hook, `None` by default, overridden by the macro from the same
  `split_dispatch` gate that raises the allocation floor.

Both sides are gated, because a gate on only one is a gate on nothing:
`this_portal_reserves_no_identity_slots_so_the_decline_is_inert` pins regex at
`None` (if it ever reported a range, every bridge count in that module would
silently go to zero), and
`split_dispatch_reserves_the_identity_range_the_bridge_walk_refuses` pins
dualtape at `Some`.

Still open, and now recorded at `get_list_of_active_snapshot_boxes`: the
blanking itself. A blackhole resuming inside a split arm reads `Const(0)`
through `handler_load_state_field_di`. The deopt side needs a real
re-derivation before any resume path is entitled to re-fill those slots — and
if one is written it must read the reconstructed vable list the way
`pyjitpl.py rebuild_state_after_failure` does, not frame 0's registers.

## The post's experiments, measured (0826)

Everything below is one machine — Apple M4, 10 cores, macOS 26.5.2, dynasm,
`--release` — on `(a|b)*a(a|b){20}a(a|b)*` (93 nodes) over a 1,048,576-character
non-matching input, median of 5. The 1-minute load average is quoted with every
row because it has to be: the same C++ binary reads 8.3M chars/s at load ~10 and
2.7M at load ~33, and no row here was taken on an idle machine.

| the post's row | 2010 figure | here | load |
|---|---:|---:|---:|
| pure Python | 12,200 | **119,025** | 9.98→7.86 |
| RPython → C | 720,000 | Rust `interp.rs`, `rustc -O`: **7,380,362** | 7.7→8.2 |
| C++ (Fischer) | 750,000 | `clang++ -O2`: **8,475,048** | 9.98 |
| CPython `re` | 2,500,000 | **12,399,816**, and NOT comparable | 7.86→7.55 |
| RPython + JIT | 16,500,000 | majit JIT: **37,494,169** | 7.7→8.2 |

The two independently-compiled ports of the same algorithm land within 15% of
each other (C++ 8.48M, Rust 7.38M), which is the check that neither optimizer
deleted the walk. The harness also runs the node-count control on both: `n = 2`
(21 nodes) over `n = 20` (93) gives 5.00x for C++ and 4.38x for Python against
the ideal 93/21 = 4.4x, so the tree walk really is inside the timed loop.

**JIT over the same matcher compiled ahead of time: 5.1x.** The post's own
comparable quantity is 16,500,000 / 720,000 = 22.9x. Ours divides by a matcher
`rustc -O` had already optimized well; the absolute rows are sixteen years and
two instruction sets apart and only the ratios travel.

The `re` row is a caveat, not a ratio: `re` is a backtracking engine doing an
unmeasured multiple of the work per character. What `comparisons/` does
establish about it is that it is linear in the input on this pattern
(`rate_flatness` 1.16 over a 32x length sweep, where an engine bailing after a
bounded prefix would show 32x) and that it reads the final byte
(`reaches_last_char`), which is why the row is kept at all rather than thrown
out.

The post's remaining two rows — Google re2 and Java — are out of scope. Neither
is installed here, this harness installs nothing, and a row nothing ever ran is
an invitation to quote a number that does not exist, so their files were deleted
rather than kept. What the example owes is the comparison against RPython, and
that is `rpython_original/`, below.

The post's other experiments:

* **The trace itself.** `(a|b)*`, four nodes, is the listing the post prints in
  full, and `jit_interp::tests::the_four_node_tree_reproduces_the_posts_listing`
  asserts our peeled body against it as an ordered sequence, not a census: 15
  ops, same ops in the same order, on both backends. Ours is that listing minus
  two reads — the repetition's mark, which the preamble reads once and the back
  edge then carries in a register, and `len(s0)`, a red state field and so a
  loop argument. 0 `getfield_gc_r`, 0 `getfield_gc_i`, 0 `guard_value`.
* **Varying lengths.** 4,096 / 65,536 / 1,048,576, and the JIT row is *not* a
  per-character rate across them: 1,822,199 → 19,916,228 → 37,494,169, a 20.6x
  spread, with `1.0 loops compiled per run` at every length. `matches` builds
  its `JitDriver` inside `mainloop`, so every call records and compiles again;
  read the short row as matcher-plus-compiler and the long one as the compiled
  loop's own.
* **`&&`/`||` instead of `&`/`|`.** Measured, and the post's claim holds by a
  different mechanism than it states — see the module doc of `shortcircuit.rs`
  and the bridge section above. Re-measured after the bridge fix, twice, at
  1M characters: masking / branching 1706x and 1848x, no-JIT / branching 378x
  and 405x, against 2128x and 399x before the fix. All three absolute rows fell
  by about the same factor (2.2-2.7x) between those runs because the machine
  went from load 8 to load 34-83; the ratios, taken inside one process, did
  not. The branching portal stays about 400x slower than the same algorithm
  with no JIT under it, which is the verdict the module doc reports.

## The comparison against RPython itself (0827)

The measurements above are chars/s on one machine; they cannot say whether the
majit portals are the post's program. Only the original can. `rpython_original/`
holds the post's matcher written in RPython — the class hierarchy,
`_immutable_fields_ = ['empty']` with `marked` mutable, and a JitDriver whose
green is the regex — and `runner.py` puts it through `LLJitMixin`, the in-process
harness RPython's own JIT tests use: real metainterpreter, real optimizer,
LLGraph backend. The loop that comes out is the trace RPython would compile,
without a translation. The RPython checkout is already at this repository's root,
so nothing had to be installed.

```sh
cd majit/examples/regex/rpython_original
PYTHONPATH=<repo root> pypy runner.py 20 4096
```

Both sides scan the same bytes, pinned rather than assumed: `fixture.py` carries
`NONMATCHING_4096_FNV1A = r_uint(0xd9f7f62ad250969e)` and `regex.rs` carries the
same constant, each asserted by a test on its own side.

Peeled body — from the last `label` on, what runs per input character — at 93
nodes:

| op | RPython JIT | majit `shortcircuit` | majit `jit_interp` |
|---|---:|---:|---:|
| `getfield_gc_r` | 0 | 0 | 0 |
| `getfield_gc_i` | 24 | 24 | 24 |
| `setfield_gc` | 93 | 93 | 93 |
| `int_eq` | 2 | 2 | 2 |
| `guard_true` | 6 | 6 | 1 |
| `guard_false` | 21 | 20 | 0 |
| total | 153 | 176 | 194 |

Every structural count is exact, on both portals: no pointer read survives, one
mark store per node, and 2 comparisons for 46 `Char` nodes — the subset
construction, performed identically by both tracers. The loop tails are the same
ops in the same order (`int_add`, `setfield_gc`, `int_lt`, `guard_true`, `jump`;
RPython additionally carries a zero-cost `debug_merge_point`). At `n = 2` (21
nodes) RPython gives `setfield_gc 21`, `int_eq 2`, `getfield_gc_r 0`, so the
census tracks the tree rather than a constant.

**This corrected the crate's attribution, which had the two portals backwards.**
The post's source writes `shift` with `and`/`or`; part 2 then remarks that
`&`/`|` is the better spelling. RPython's trace carries 27 guards, `shortcircuit`
26, `jit_interp` 1 — so `shortcircuit.rs` is the faithful port of the post's own
source and `jit_interp.rs` is the adapted variant, the A/B of that remark. The
docs had called `jit_interp` "the post's portal" and `shortcircuit` "the
control". Both module docs, `main.rs`'s row names, and this document now say
which is which.

The 153-vs-176 gap is one spelling difference, not a missing optimization: majit
emits `IntIsTrue` ahead of each guard where an RPython guard takes its condition
directly. `runner.py --listing` and `REGEX_LISTING=1` on the census test print
the two peeled bodies op by op for reading side by side.

Two RPython-side traps worth recording. `get_stats` lives in
`rpython.jit.metainterp.warmspot`, not in `history`. And the annotator rejects a
prebuilt `(1 << 64) - 1` mask outright (`seeing a prebuilt long`), so the LCG is
spelled with `r_uint` — which is in any case the truer match for Rust's
`wrapping_mul`.
