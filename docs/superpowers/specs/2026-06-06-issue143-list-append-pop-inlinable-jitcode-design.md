# Issue #143 — Make `list.append`/`list.pop` genuinely inlinable (PyPy-orthodox)

Date: 2026-06-06
Branch: `pyre-7`
Issue: https://github.com/youknowone/pyre/issues/143

> **SUPERSEDED IN PART — see "VERIFIED INVESTIGATION (2026-06-06)" at the bottom.**
> A multi-agent investigation (with adversarial verification + direct re-reads)
> corrected this doc's central premise. Summary of corrections:
> 1. The production forward tracer is the hand-written **Python-bytecode walker**;
>    it does NOT trace into Rust jitcodes. "Remove `dont_look_inside` → the walker
>    inlines the fast path" is FALSE.
> 2. `guess_call_kind` classifies `jit_list_append` as `CallKind::Builtin`
>    (residual) because it is in `LIST_APPEND_TARGETS` — the builtin check
>    PRECEDES the candidate-graph check (`call.rs:3086`). Removing
>    `dont_look_inside` alone keeps it residual; it must ALSO be removed from the
>    oopspec/builtin targets to become a `Regular` candidate.
> 3. The orthodox resume ("resize guard inside a real callee frame") requires
>    NEW infrastructure pyre lacks: **the blackhole cannot resume into a helper
>    (Rust) jitcode** — it only resumes at outer Python opcode boundaries
>    (`jitcode_dispatch.rs:3740-3756`); the multi-frame snapshot
>    (`capture_snapshot_for_last_guard_multi_frame_with_vable_vref`) is a named
>    placeholder. This is the load-bearing cost and the real #73 resume story.
>    The "resume is free from the pipeline" claim below is WRONG.
> 4. `Skip` ≠ untranslated: the dual-gate `Skip` arm types a graph via the
>    LEGACY rtyper (still produces a jitcode). The Match=0 / 926-skip cascade is
>    the #73 canonical-flip / legacy-retirement concern — ORTHOGONAL to whether
>    list.append translates. So translation is likely NOT the blocker.
> 5. The user chose the FULL #73 convergence path (walker → translated-interpreter
>    jitcode tracing) so the fast path also comes from real tracing.

## Problem

When `list.append(x)` / `list.pop()` are folded into inline IR in a compiled
trace (strategy guard + `getfield(len)` + resize-guard + `setarrayitem` +
`setfield(len+1)`), a growing list eventually fails the resize-guard at runtime.
On that guard failure the blackhole interpreter re-executes the Python `CALL`
bytecode of the method call and reconstructs the callable as `NULL`, raising
`TypeError: call on null callable`.

Reproduction:

```python
def main():
    xs = []
    i = 0
    while i < 5000:
        xs.append(i)
        i = i + 1
    print(len(xs), xs[0], xs[4999])
main()
```

## Root cause

The fold is hand-emitted (`generated_list_append_by_strategy`) directly in the
**outer Python frame** while tracing the `CALL` opcode. So the resize-guard's
resume PC is the `CALL` PC, and the metainterp framestack at guard-capture is
just `[python_frame]`. On guard failure the blackhole resumes at the `CALL` and
re-executes the method-shape call — but the codewriter's static blackhole call
convention cannot reconstruct a method-shape callable from the deep/shallow
stack slots, so the callable reads `NULL`.

CPython-shaped workarounds (replaying the `CALL` operands into the guard
snapshot, materializing the bound method as a constant, reordering the
codewriter `Call`/`LoadAttr` arms) all bottom out at this wall — they all still
require the blackhole to re-execute a method-shape `CALL`.

The current branch WIP (`004ca99cf4`) is exactly this workaround
(`push_call_replay_stack*` + a `load_method` arm that materializes the unbound
function as a `Const`). It is the path the issue rejects.

## How PyPy does it (the orthodox target)

PyPy traces **into** RPython. `list.append` is RPython (`W_ListObject.append`
→ strategy → `_ll_list_resize_ge` → store), and the codewriter generates a
jitcode for it at translation time. When the Python `CALL` is traced, the
metainterp **inlines** the call: `MIFrame.perform_call` pushes a real `MIFrame`
for the callee jitcode and traces into it, recording `getfield(len)`, the
resize check (→ `int_lt` + `guard_true`), `setarrayitem`, `setfield(len+1)`.

At the resize guard, `capture_resumedata` walks the **live framestack**
`[…, python_frame, …, w_list_append_frame, _ll_list_resize_ge_frame]` and
snapshots each frame's `(jitcode, pc, live regs)`. The frame chain in the
resume descriptor falls out automatically because the frames are real.

On guard failure the blackhole rebuilds the chain, runs the innermost callee
from the guard's realloc branch (`_ll_list_resize_hint_really`, which is
`@jit.dont_look_inside` → a residual call), does the store, returns up through
`ll_append`, and finally lands the Python frame at the instruction **after**
the `CALL` with the result already pending. The `CALL` is never re-executed,
so `callable = NULL` is structurally impossible.

## Pyre after #121: "RPython" is Rust

`#121` replaced the JIT front-end with a **Charon-extracted MIR front-end**. The
build requires `build/llbc/{pyre-object,pyre-interpreter}.ullbc`, i.e. the
majit-translate pipeline turns **Rust** (via Charon LLBC) into jitcodes. So
pyre's "RPython" is Rust, and the metainterp inlines Rust helper calls by
default (`policy.look_inside_graph` / `guess_call_kind`, faithful ports of
RPython `JitPolicy`). `#[majit_macros::dont_look_inside]` is the faithful port
of `@jit.dont_look_inside` and opts a call **out** of inlining (residual).

`jit_list_append` was marked `#[dont_look_inside]` in commit `f6e042f3ab`
("helper annotations — rlib/jit.py decorator parity"), a blanket annotation
sweep — **not** because inlining was attempted and failed.

Critically, the Rust list code is favorably structured:

- `ListStrategy` is a `#[repr(u8)]` enum, so strategy dispatch is a `match`,
  **not** `dyn`-trait dispatch (avoids the documented "no common base"
  trait-dispatch wall).
- `object_push` has the exact hot/cold split PyPy traces:

  ```rust
  unsafe fn object_push(&mut self, value: PyObjectRef) {
      if self.length == self.object_items_capacity() {   // → resize GUARD
          self.object_grow(self.length + 1);              // cold realloc → residual
      }
      let base = items_block_items_base(self.items);     // getfield items
      *base.add(self.length) = value;                     // setarrayitem
      self.length += 1;                                   // setfield len+1
  }
  ```

- The IR `generated_list_append_by_strategy` hand-emits today is **byte-for-byte
  the IR real tracing into `w_list_append` would produce** — so the compiled
  fast-path trace is unchanged by going orthodox; only its *provenance* and the
  *resume descriptor* change.

## Chosen approach: make `list.append`/`list.pop` genuinely inlinable

Remove `#[dont_look_inside]` from the `jit_list_append`/`jit_list_pop` boundary
helpers and let the metainterp inline the call, tracing
`jit_list_append → w_list_append → object_push / IntArray::push / FloatArray::push`.
Keep the cold paths (`object_grow`/`grow_list_items_block`/realloc,
`switch_to_*_strategy`) as `#[dont_look_inside]` residual calls — exactly like
PyPy's `_ll_list_resize_hint_really`.

Result:
- The resize check becomes a real guard inside the `w_list_append`/`object_push`
  callee jitcode frame. `capture_resumedata` snapshots
  `[python_frame @ CALL-fallthrough, w_list_append_callee @ resize-guard]`.
- On guard failure the blackhole runs the callee from the realloc branch
  (residual `object_grow`), does the store + length++, returns, and the Python
  frame continues at the `CALL` fallthrough. No `CALL` re-execution; no null
  callable.
- Resume liveness / `pc_map` / per-frame live-value counts come **free** from
  the normal codewriter pipeline (no hand-built jitcode metadata).

### Why orthodox

- Single source of truth: `w_list_append` — no hand-emitted IR that can diverge
  from the interpreter's real semantics.
- Generalizes: pop / insert / extend / other builtins get the same treatment by
  removing their `dont_look_inside` boundary, not by writing per-op folds.
- Resume correctness is structural (a real jitcode frame), not bolted on.

### Removed at the end

- `generated_list_append_by_strategy` / `generated_list_pop_by_strategy` and the
  `guard_append_without_resize` / `guard_pop_without_resize_le` helpers
  (`majit/majit-translate/src/codegen.rs`).
- The list-method arms in `call_callable_value` and the list arm in
  `load_method` (`pyre/pyre-jit-trace/src/trace_opcode.rs`).
- The WIP workaround: `push_call_replay_stack` /
  `push_call_replay_stack_self_in_args` / `pop_call_replay_stack*` and the
  `load_method` Const-materialization arm.

## Gating risk → Phase 0 spike

The only unknown is whether the Charon → majit-translate pipeline ingests
`w_list_append`'s hot-path subtree as inlinable graphs (raw-pointer ops
`items_block_items_base` / `base.add`, `IntArray`/`FloatArray` push, capacity
header read, `is_plain_int1` / `plain_int_w`) and whether the metainterp
produces a clean trace. The translator frontier can hit walls (untranslatable
ops needing IR lowerings or `dont_look_inside` boundaries pushed inward).

**Phase 0** is a throwaway spike:
1. Remove `dont_look_inside` from `jit_list_append`.
2. Mark `object_grow` / `grow_list_items_block` / `switch_to_*_strategy` as
   `dont_look_inside`.
3. Remove the `list_append` fold dispatch so the call stays a plain inlinable
   call (no hand-emitted fold).
4. Build, run the reproduction + `bench/list_pop_append.py` on dynasm, observe
   where the translator or tracer breaks.

Outcomes:
- (a) Traces clean → proceed to full implementation; the spike is the skeleton.
- (b) Specific untranslatable ops → add IR lowerings or push `dont_look_inside`
  boundaries inward (still orthodox), iterate.
- (c) Deep cascade wall → report honestly; choose between investing in the
  translator frontier vs. landing the A1 fallback as interim.

## A1 fallback (documented safety net)

If Phase 0 hits a (c) wall, A1 produces the **identical compiled output**:
hand-emit the fold (keep `generated_list_append_by_strategy`) + synthesize a
build-time slow-path callee jitcode (`residual_call(jit_list_append, list,
value); return None`) and inject a synthetic top `SnapshotFrame` so the resize
guard snapshots `[outer @ fallthrough, synthetic-callee @ slow-path]`. Cost:
hand-built `JitCodeBody` + a skeleton `PyJitCode` metadata (`pc_map`,
`stack_base`, liveness offsets) registered by `jitcode_index` — the
machinery exists (`JitCode::new` + `set_body`, `get_jitcode_by_index`) and a
working `dispatch_loop` test template exists, but resume needs the metadata
wired manually. Kept as fallback only; the tracked goal stays the orthodox path.

## Phasing

- **Phase 0** — feasibility spike (de-risk). Throwaway.
- **Phase 1** — `append` inlinable end-to-end: reproduction + `list_pop_append`
  append side green on dynasm and cranelift.
- **Phase 2** — `pop` inlinable: add a `jit_list_pop` boundary helper (none
  exists today), pop shrink guard via real tracing.
- **Phase 3** — remove the hand-emitted fold, all list-method special-casing,
  and the WIP workaround. Verify `check.py` 41/41 both backends + perf gate
  (`list_pop_append`, `synth/list_append_pop`).

## Acceptance criteria (from the issue)

- `bench/list_pop_append.py` and `bench/synth/list_append_pop.py` pass on both
  dynasm and cranelift with the append/pop fold enabled (not interpreter
  fallback).
- A program growing a list across ≥1 reallocation in a hot loop runs correctly
  on both backends.
- The resize-guard failure resumes through the blackhole frame chain without
  re-executing the `CALL`; no `call on null callable`.
- No regression in the test suite or benchmark perf gate.

## Key code references

- `pyre/pyre-object/src/listobject.rs` — `jit_list_append` (`dont_look_inside`),
  `w_list_append`, `object_push`/`object_grow`, `ListStrategy` enum.
- `pyre/pyre-object/src/{int_array,float_array,object_array}.rs` — typed array
  push / spare_capacity / grow.
- `majit/majit-translate/src/jit_codewriter/call.rs` — `CALL_DESCRIPTOR_TABLE`
  (`LIST_APPEND_TARGETS`), `guess_call_kind`, `policy.look_inside_graph`.
- `majit/majit-translate/src/codegen.rs` — `generated_list_append_by_strategy`,
  `generated_list_pop_by_strategy`, `guard_append_without_resize`.
- `pyre/pyre-jit-trace/src/trace_opcode.rs` — `call_callable_value` list arms,
  `load_method` list arm, `list_append_value`/`list_pop_value`, the WIP
  `push_call_replay_stack*`.
- `pyre/pyre-jit/src/call_jit.rs` — `bh_call_fn_impl_with_frame`, the
  `resolve_jitcode` closure, `blackhole_from_resumedata`.
- Build: `scripts/install-charon.sh`, `scripts/extract-llbc.sh`,
  `pyre/check.py`.

---

# VERIFIED INVESTIGATION (2026-06-06)

A multi-agent workflow (4 parallel readers + synthesis + 3 adversarial
verifiers) plus direct re-reads established the following. Empirical anchors:
the bug reproduces (`while i<5000: xs.append(i)` → `TypeError: call on null
callable` on baseline dynasm); `bench/list_pop_append.py` passes only because
append↔pop keeps len≈5 (no realloc). (One verifier wrongly claimed the bug was
"already fixed" by the WIP replay-stack — refuted by the live crash.)

## Architecture (verified)

- **Production forward tracer = hand-written Python-bytecode walker**
  (`trace_opcode.rs` / `metainterp.rs` / `jitcode_dispatch.rs`). The top driver
  dispatches each Python opcode via the trait interpreter and diverts only
  allow-listed opcodes into the codewriter walker
  (`trace_opcode.rs:6931 production_walker_handles(&instruction) && !in_inline_frame`).
  The walker DOES record real IR and DOES recurse into sub-jitcodes via
  `inline_call_*` (`jitcode_dispatch.rs:4526+`) — but only for the opcodes it
  handles. **`Call{argc}` is NOT walker-handled** (only `CallKw`/`CallFunctionEx`
  are, `trace_opcode.rs:7752`); `xs.append(i)` is handled by hand-wired
  trait-dispatch arms in `call_callable_value` (`trace_opcode.rs:5356/5457`) →
  `list_append_value` (`:5103`) → `generated_list_append_by_strategy`
  (`codegen.rs:2127`) hand-fold emitted in the OUTER Python frame.

- **Blackhole resume wall (load-bearing)** — `jitcode_dispatch.rs:3740-3756`,
  verbatim: pyre's blackhole "only knows how to run *pyjitcode* bytecode (Python
  bytecode), not helper jitcodes … any walker-emitted guard … must resume to the
  *outer* Python opcode boundary — that is the only resume point pyre's blackhole
  can re-enter." The multi-frame successor
  `capture_snapshot_for_last_guard_multi_frame_with_vable_vref` is named as
  future work ("WalkContext must grow a parent-Python-frame chain"). **Building
  this — multi-frame snapshot + blackhole resume INTO a helper/Rust jitcode — is
  the campaign's core deliverable and is exactly the #73 resume convergence.**

- **`guess_call_kind` builtin precedence** — `call.rs:3071-3098`: the
  `builtin_targets` check (`:3086`) precedes `graphs_from(op).is_none()`
  (`:3093`). So an oopspec/`*_TARGETS` member is `CallKind::Builtin` (residual)
  even when it is also a candidate graph. To inline `jit_list_append` it must be
  removed from `LIST_APPEND_TARGETS` AND lose `dont_look_inside`.

- **`Skip` ≠ untranslated** — `codewriter.rs:324-399`: the dual-gate `Skip` arm
  falls back to `legacy_annotator::annotate` + `legacy_resolve::resolve_types`
  (still commits a jitcode). `Match` = the canonical real-rtyper path. The
  Match=0 / ~926-skip cascade is the #73 canonical-flip / legacy-retirement
  metric — orthogonal to whether the list subtree translates. The only true
  panic is `Err` ("PYRE_RTYPER real-path failure", `codewriter.rs:400`): a new
  unported shape on the canonical path that is not whitelisted as known-unported.

## Goal-directed campaign (full #73 path, chosen)

The smallest slice that puts the resize guard inside a REAL `jit_list_append`
callee frame with correct resume:

- **REUSE**: walker IR recorder + `inline_call_*` recursion; the hand-wired
  list devirtualization (receiver=list + method=append behind a strategy guard);
  `LIST_APPEND_TARGETS` registration (as the recognition hook).
- **MUST-BUILD (core)**: blackhole multi-frame resume into a Rust callee jitcode
  + the multi-frame snapshot. This is the #73 resume story; multi-week.
- **MUST-BUILD (routing)**: make `jit_list_append` a `Regular` candidate (drop
  it from `LIST_APPEND_TARGETS` + drop `dont_look_inside`) AND drive inline
  tracing of its jitcode from the trait-dispatch list arm (or add `Call{argc}`
  to the walker), replacing the hand-fold.
- **PROBE (one spike-only unknown)**: does the `jit_list_append → w_list_append
  → object_push / IntArray::push` subtree translate (via Match or legacy Skip)
  without an `Err` real-path panic? (In progress.)

## Scope verdict (honest)

Genuinely a multi-week campaign. Even the narrow list-only slice is gated on
building the blackhole resume-into-callee infrastructure (NOT free). Sequence:
PROBE (resolve translation) → build resume infra (the bulk) → rewire routing →
verify both backends + perf. The full "move the entire forward driver onto
translated-interpreter jitcode tracing" is the larger umbrella.

## Milestones

- **M0 PROBE**: confirm the list subtree translates (no `Err` panic); observe
  Skip(legacy)/Match per graph via `PYRE_RTYPER_VERBOSE=1`. Throwaway edits.
- **M1 forward trace-into-callee**: `jit_list_append` Regular candidate + trait
  arm drives inline tracing; observe (MAJIT_LOG) the fold + resize guard emitted
  inside the `jit_list_append` callee frame. (Resume still broken here.)
- **M2 resume infra**: multi-frame snapshot + blackhole resume into the callee
  jitcode; `repro143.py` runs correctly, dynasm.
- **M3 cranelift parity + pop**: same on cranelift; add `jit_list_pop` path;
  `list_pop_append` / `synth/list_append_pop` green both backends.
- **M4 cleanup**: remove the hand-fold + list special-casing + WIP workaround;
  `check.py` 41/41 both backends + perf gate.

## CORE BUILD STATUS (2026-06-06, verdict-corrected)

A 3-agent adversarial build-plan workflow + direct re-reads refined the model
above. The user chose to build the **core resume infra first**.

### Corrected resume model (supersedes the "Blackhole resume wall" item)

- The blackhole resume **engine** `blackhole_from_resumedata` (resume.rs:6775)
  is ALREADY jitcode-agnostic. Proof: the passing test
  `blackhole_from_resumedata_accepts_runtime_jitcode_without_canonical_pair`
  (resume.rs:4935) drives it with a CodeObject-less hand-built jitcode. The
  chain loop (resume.rs:6824-6845) resolves every frame uniformly via the
  caller `resolve_jitcode` closure + `consume_one_section`; it does NOT branch
  on Python-vs-helper.
- The "NAMED PLACEHOLDER" wall is the **walker's** multi-frame helper. But
  `list.append` does NOT use the walker — its resize `GuardTrue` is emitted by
  `MIFrame::generate_guard` (trace_opcode.rs:3567) on the **metainterp** path
  (codegen.rs:1669). So CORE extends `generate_guard` →
  `build_framestack_snapshot` (trace_opcode.rs:3817) → `blackhole_from_resumedata`,
  NOT a walker placeholder.
- **interpret() can't trace a Rust callee.** `PyreMetaInterp::interpret`
  (metainterp.rs:88-99) decodes `top.jitcode` as a `W_CodeObject` (Python
  bytecode) via `cf.next_instr()`. So we CANNOT push a real Rust-callee MIFrame
  and let interpret() trace it. The fast path stays the hand-fold
  (`generated_list_append_by_strategy`); CORE only needs to **synthesize the
  callee SnapshotFrame at the resize guard** so RESUME enters the helper.

### Verdict-converged load-bearing gaps

1. Top-level `xs.append(i)` has EMPTY `self.parent_frames` → `generate_guard`
   takes the single-frame fallback (trace_opcode.rs:3660-3675) with
   `resume_pc = orgpc` (the CALL) → re-exec CALL = the null-callable bug. #143
   MUST make the snapshot `[callee=helper@guard-pc, parent=outer-Python@post-CALL-fallthrough, …self.parent_frames]`.
   (The HEAD `push_call_replay_stack` WIP does NOT push/synthesize a frame — it
   only re-stages operands → still re-execs the CALL → wrong.)
2. The callee frame needs a valid `jitcode_index` (`build_framestack_snapshot`
   reads `(*self.sym().jitcode).index`, trace_opcode.rs:3826).
3. cranelift x86_64 GC-root crash: the callee's extra `[list,value]` refs land
   in blackhole `registers_r` and become GC roots → recurring typeid OOB at
   trace.rs:820 (see memory `cranelift-linux-gc-typeid-crash`). OPEN — only
   ubuntu cranelift CI can validate; gate cranelift to baseline until cleared.

### KEY INSIGHT — identity pc_map closes gap 2 with NO production resolve change

`resolve_jitcode` (call_jit.rs:1318) maps `resume_jitcode_pc_for(pc)` =
`pc_map.get(pc)`. Register the helper with an **identity pc_map** (`pc_map[i]==i`)
so a jitcode byte-offset resumes to itself. And `jitcode_for` /
`portal_bridge_jitcode_for` (state.rs:360/332) already assign
`index = jitcodes.len()` at RUNTIME (not just startup) — so runtime index
registration is one new method, not an architecture change. (feasibility
verdict's "index registration UNIMPLEMENTED" was overstated.)

### M-core milestones (refines M1–M4 above)

- **M-core1 — LANDED (uncommitted, tested).** `MetaInterpStaticData::
  register_runtime_helper_jitcode` (state.rs, after :394; null `w_code`, identity
  pc_map, `#[allow(dead_code)]` until M-core2 wires it). Test
  `register_runtime_helper_jitcode_resolves_by_index_with_identity_pc`
  (indirectcalltargets_tests) proves production `pyjitcode_for_jitcode_index(idx)`
  resolves the helper and `resume_jitcode_pc_for(off)==off`. PASSES under
  `cargo test -p pyre-jit-trace --no-default-features --features dynasm`.
- **M-core2 — LANDED (uncommitted, tested; 2026-06-06, verdict-revised).**
  Three methods on `impl MIFrame` (trace_opcode.rs), all dead-code until M-core3:
  (1) `build_framestack_frames(ctx, lead) -> Vec<SnapshotFrame>` extracted from
  `build_framestack_snapshot` (append `self.parent_frames` to an innermost-first
  `lead`, then `reverse()` to the outermost-first `recorder.rs:56` contract); the
  existing `build_framestack_snapshot` now delegates to it (`lead=[top]`). (2)
  `build_synthetic_callee_frames(ctx, helper_jitcode_index, helper_pc, boxes,
  result_stack_idx, result_type)` — helper top frame with boxes built UNSLICED
  (no `NUM_SCALAR_INPUTARGS` header; types via `value_type` per box → list=Ref,
  unboxed int value=Int), self demoted to first parent at `resume_pc=
  fallthrough_pc` through `materialize_parent_snapshot_state` (borrow-safe: it
  reads nothing from `self`, rebuilds an MIFrame from the raw `sym` ptr → ≤1
  `&mut PyreSym` live), then `self.parent_frames`; returns outermost-first
  `[...parents, self, helper]`. (3) `build_framestack_snapshot_with_synthetic_callee`
  wraps (2) + one-shot vable/vref from the REAL Python `sym`. Test
  `build_synthetic_callee_frames_synthesizes_self_caller_and_helper_top` asserts
  `frames==[self@fallthrough(empty boxes), helper@guard([Box(list,Ref),
  Box(value,Int)])]`. Full `pyre-jit-trace` suite green (239 passed).
  **3-lens adversarial verify resolved into the code:** (a) the demoted self's
  pending result slot is a SIGNATURE PARAMETER (`result_stack_idx`/`result_type`)
  — it MUST be `Some` in production so `get_list_of_active_boxes`'s `in_a_call`
  null-out (trace_opcode.rs:1538) clears the undefined call result instead of
  capturing a stale box; the unit test passes `None` (valid empty-liveness case).
  (b) the unit test uses an IDENTITY pc_map `(0..6)` so `resume_jitcode_pc_for(
  fallthrough_pc=3)` resolves (a partial pc_map would `.expect`-panic at
  trace_opcode.rs:1231 — the bug all 3 verifiers caught in the draft). (c)
  frame order is OUTERMOST-FIRST (`frames[0]=self`, `frames[1]=helper`), refuting
  two finders that claimed `frames[0]=helper`.
  **Deferred to M-core3 (documented in method (3)'s doc comment):** the
  `orgpc` swap + vable scalar save/restore that `capture_resumedata`
  (trace_opcode.rs:3742-3771) performs around the snapshot must wrap the M-core3
  call site (`list_of_boxes_virtualizable` derives the vable last_instr/vsd from
  `self.orgpc`); the value box's SnapshotTagged kind must match the kind the
  M-core1 helper jitcode declares at `guard_pc` (Ref vs Int mismatch mis-numbers
  the box on resume); and method (3)'s vable/vref tail is unit-untested (it would
  SEGV on a skeleton helper's null code — needs a real PyFrame or seeded
  `virtualizable_array_lengths`).
- **M-core3 — wire + real helper body. VERIFIED PLAN (2026-06-06, workflow
  `wzunyzc9c`: 6 findings + synth `feasible-with-changes` + 3 adversarial
  verdicts, all source-grounded).** The whole stack exists; no architecture
  change. Implement in committable increments:
  - **Helper body (the resumable callee).** The blackhole CAN execute a
    hand-built runtime-helper jitcode that does a residual call: wired handlers
    `residual_call_r_v/iRd` … `residual_call_irf_*` (blackhole.rs:6819-6828);
    `read_descr` reads `bh.jitcode.exec.descrs[idx]` (blackhole.rs:5066, fallback
    `bh.descrs`); `setup_return_value_r` threads a returning callee's result into
    `caller.registers_r[caller.code[caller.position-1]]` (blackhole.rs:677-688)
    and `resume_mainloop`/`run_forever` (blackhole.rs:727,2180) drive the chain.
    Build the body with the **high-level `JitCodeBuilder`** (majit-metainterp
    jitcode/assembler.rs), NOT raw bytes: `b.live(...)` (resume-entry liveness
    marker, live_r=[0,1] = list,value) → `b.residual_call_void_canonical_typed_args(
    jit_list_append as i64, &[JitCallArg{reg:0,Ref}, JitCallArg{reg:1,Ref}],
    BhCallDescr::from_arg_classes("rr".into(), 'v', default_effect_info()))`
    (:2072) → `b.ref_return_const(w_none() as i64)` (:1690) → `b.finish()`.
    **CORRECTNESS FIX (found vs the draft):** `jit_list_append` returns `0`
    always (listobject.rs:1107, side-effect-only via `w_list_append`); the
    existing emit treats it VOID (helpers.rs:524). `None` in pyre is `w_none()`
    (a real pointer), **NOT** `0`/null — so the helper must do a VOID append then
    `ref_return_const(w_none())`, never reinterpret the `0` as `GcRef(0)`. Wrap
    `finish()` in `PyJitCode::from_parts` with IDENTITY pc_map `(0..len)` + null
    `w_code`, register via M-core1 `register_runtime_helper_jitcode`.
  - **Increment 1 — LANDED (tested, both backends).** `build_list_append_resize_helper_payload()`
    (state.rs, after `register_runtime_helper_jitcode`) + a unit test
    `list_append_resize_helper_appends_and_returns_none_through_blackhole` that
    drives it through the REAL `build_pyre_production_bh_builder()`
    (jitcode_runtime.rs:503) blackhole as a single bottommost frame — builds a
    `W_ListObject` filled to the `IntArray` inline boundary (`INT_ARRAY_INLINE_CAP`=8,
    spare 0 → next append MUST realloc, asserted via `w_list_can_append_without_realloc`),
    `setposition(payload,0)`, `setarg_r(0,list)`/`setarg_r(1,value)`,
    `run_forever(&mut builder, bh, 0)`, asserts `DoneWithThisFrameRef(w_none())` +
    `w_list_len` grew by 1. Proves the residual call EXECUTES end-to-end (the one
    gap the resume.rs:4935 test leaves — it proves the chain BUILDS, not that
    opcodes RUN). check.py-suite unaffected; 233/233 lib tests both backends.
    **Plan deviations discovered + resolved while grounding:**
    (a) `default_effect_info()` → used `EffectInfo::MOST_GENERAL` (what
    `BhCallDescr::default()` carries; conservative, and `bh_call_v` ignores
    extra_info anyway — only `arg_classes` drives dispatch). (b) The "builder
    needs the global insns table" note was WRONG: `write_insn` resolves via the
    STATIC `wellknown_bh_insns()` table (insns.rs), so building needs NO prior
    init; the production-bh-builder init is only for the blackhole's `op_*` cache
    + cpu wiring at EXECUTE time. (c) `live(&mut asm,…)` needs an `Assembler` —
    used a throwaway local `Assembler::new()`; safe because `handler_live` only
    SKIPS the 2-byte offset (never derefs it) and `run` roots all of `registers_r`
    via `push_bh_regs`, so the local intern is inert at resume + GC-covered.
    (d) Test gated `#[cfg(any(feature="dynasm",feature="cranelift"))]`: without a
    backend `builder.cpu` is `None` and the residual handler's `cpu()` panics.
  - **Increment 2 — LANDED (tested, both backends).** 2-frame chain test
    `list_append_resize_helper_threads_none_into_caller_result_register`
    (stub-caller + real helper) via `run_forever`, proving `setup_return_value_r`
    threads `w_none()` into the caller's `code[position-1]` register. Stub caller
    jitcode = `[dest_reg=0, BC_REF_RETURN, r0]` resumed at position 1
    (`code[0]`=the post-CALL result-register byte); chain linked
    `helper.nextblackholeinterp = Some(caller)`, helper runs first, threads None
    into caller r0, caller `ref_return r0` → `DoneWithThisFrameRef(w_none())`.
    Isolates the return convention WITHOUT pyre's real Python CALL layout (the
    real fallthrough layout is the increment-6 proof). 234/234 lib tests both
    backends. NOTE: only re-confirms the GENERIC `setup_return_value_r` mechanism
    (already used by every inline call) — the genuinely unproven part (that the
    real Python jitcode's fallthrough `code[position-1]` IS the append's result
    slot) is still increment 6's job; see LOAD-BEARING UNPROVEN below.
  - **Increment 3 — memoized helper index — LANDED (tested, both backends).**
    `ensure_list_append_resize_helper_index()` (state.rs free fn) →
    `MetaInterpStaticData::ensure_list_append_resize_helper_jitcode` registers the
    payload once and caches the index in a NEW SD field
    `list_append_resize_helper_index: Option<i32>` (NOT a thread_local Cell — the
    field is reset-safe: it lives/dies with the `jitcodes` table, and the index
    stays valid because `set_jitcodes_from_make_result` only updates
    W_CodeObject-keyed entries [helper carries null `w_code`] and appends, never
    shifting an existing index). The "build BEFORE the METAINTERP_SD borrow /
    intern_liveness nested-borrow" caveat is MOOT — the inc.1 builder uses a
    throwaway local `Assembler`, touching no thread-local, so the build is safe
    even inside `borrow_mut`. Test
    `ensure_list_append_resize_helper_index_is_stable_and_resolves`: second call
    returns same index; resolves to a populated non-skeleton null-`w_code` payload
    with identity pc_map. 235/235 lib tests both backends.
  - **Increment 4-5 (tracer wiring) — REMAINING:** `capture_resumedata_with_synthetic_callee`
    on MIFrame (saves/restores orgpc+vable_last_instr+vable_valuestackdepth like
    capture_resumedata trace_opcode.rs:3742-3771, sets orgpc=fallthrough before the
    build); `generate_guard_with_synthetic_callee` (record_guard_typed BEFORE
    set_last_guard_resume_position; fail_arg_types in snapshot box order
    [helper,self,parents]); then codegen.rs:1646 — add `value: OpRef` to
    `guard_append_without_resize`, pass `value` from generated_list_append_by_strategy
    (:2143), and route to the synthetic guard ONLY when `value_type(value)==Ref`
    (else single-frame fallback). `result_stack_idx` = COMPUTE from
    `sym().valuestackdepth`/`nlocals` (do NOT hardcode `Some(0)`); `result_type=Ref`.
  - **Increment 6 (gate):** `repro143.py` (`while i<5000: xs.append(i)`) stops
    crashing on dynasm; check.py green both backends.
  - **LOAD-BEARING UNPROVEN (all 3 verifiers):** the `setup_return_value_r`
    `code[position-1]` result-register convention is verified for normal inline
    calls but UNPROVEN for a folded-method-CALL fallthrough in pyre's real Python
    jitcode. Increment 2 tests the mechanism in isolation; Increment 6 is the real
    proof. If wrong, set the self-frame resume pc to the exact post-call
    result-register offset instead of generic `fallthrough_pc`.
  - **cranelift = M-core4:** if the extra `[list,value]` callee refs regress
    cranelift x86_64 GC roots, gate the synthetic path to dynasm
    (`cfg!(feature="cranelift") -> single-frame fallback`, nbody-entry-bridge
    style) and defer.
- **M-core4 — cranelift gate** (gap 3). Validate on ubuntu cranelift CI; keep
  cranelift on baseline (re-exec-CALL) until the GC-root path is proven safe.
