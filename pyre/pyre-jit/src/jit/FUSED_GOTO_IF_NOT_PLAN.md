# Plan: codewriter-level compare+branch fusion (Option A)

## Goal

Close the structural parity gap exposed after A4: fuse `CompareOp` +
`PopJumpIf{False,True}` at the pyre codewriter (jtransform.py:196
`optimize_goto_if_not` equivalent) instead of only at the tracer dispatch
layer. After this plan, both tracer (`try_fused_compare_goto_if_not` in
`pyre/pyre-jit-trace/src/trace_opcode.rs`) and runtime codewriter
(`pyre/pyre-jit/src/jit/codewriter.rs`) emit the same fused shape, and
the blackhole consumes a matching `goto_if_not_<op>_<type>/iiL|ffL|rrL`
bytecode.

## Current state of the world

- **RPython evidence**
  - `rpython/jit/codewriter/jtransform.py:196-234` `optimize_goto_if_not`:
    folds `v = int_gt(x,y); exitswitch = v` into
    `exitswitch = ('int_gt', x, y, '-live-before')`.
  - `rpython/jit/metainterp/pyjitpl.py:510-556`:
    `opimpl_goto_if_not_int_lt` etc. issue a single jitcode op that
    emits `INT_LT` + `GUARD_TRUE/FALSE` via `opimpl_goto_if_not`.
  - `rpython/jit/metainterp/blackhole.py:864-920`: matching
    `bhimpl_goto_if_not_int_lt` / `_int_le` / `_int_eq` / `_int_ne` /
    `_int_gt` / `_int_ge`, plus `_float_*` and `_ptr_*` variants.

- **pyre today**
  - Tracer: `try_fused_compare_goto_if_not`
    (`pyre/pyre-jit-trace/src/trace_opcode.rs`) already fuses at
    dispatch — emits IntLt/FloatLt + GuardTrue/False in one call.
  - Runtime codewriter:
    `pyre/pyre-jit/src/jit/codewriter.rs:660-725` still emits
    `CompareOp` → `compare_fn_idx` (returns a Bool Ref) + separate
    `PopJumpIf*` → `truth_fn_idx` + `branch_reg_zero`. Two opcodes, one
    temporary Bool object, one extra call per compare.
  - Blackhole: `handler_goto_if_not_int_{lt,le,eq,ne,gt,ge}` and
    `_float_*`, `_ptr_*` already implemented
    (`majit/majit-metainterp/src/blackhole.rs:4453-4818,5563-5700`).
    They are registered via `wire_handler` against the
    orthodox `dispatch_loop` (test-only path). The production
    `dispatch_one` switch does **not** yet know about them — no
    `BC_GOTO_IF_NOT_*` constants, no switch arms, no wellknown_bh_insns
    entries.

## Critical files

| File | Lines / symbols | Role |
|---|---|---|
| `majit/majit-metainterp/src/jitcode/mod.rs` | `BC_*` block at top; `wellknown_bh_insns` (~110) | Add new BC_* constants + opname/argcodes entries |
| `majit/majit-metainterp/src/jitcode/assembler.rs` | `pub fn branch_reg_zero` (355) | Add `goto_if_not_int_{lt,le,eq,ne,gt,ge}`, `_float_*`, `_ptr_*` emitters |
| `majit/majit-metainterp/src/blackhole.rs` | `dispatch_one` (1633), `BC_BRANCH_REG_ZERO` arm (1742) | Add `BC_GOTO_IF_NOT_*` arms; inline the int/float/ptr compare + branch |
| `pyre/pyre-jit/src/jit/codewriter.rs` | `CompareOp` arm (661), `PopJumpIfFalse` (685), `PopJumpIfTrue` (705) | Peek next non-trivia; emit fused `goto_if_not_*` when followed by PopJumpIf and concrete type match |
| `pyre/pyre-jit/src/jit/codewriter.rs` | trivia helpers (`skip_caches`) | Reuse existing `skip_caches` / `jump_target_forward` |

## Argcode convention

Follow `assembler.py:162-196`:
- `iiL` = int reg + int reg + label (goto_if_not_int_*)
- `ffL` = float reg + float reg + label (goto_if_not_float_*)
- `rrL` = ref reg + ref reg + label (goto_if_not_ptr_eq / _ne)
- `iL` = single int + label (goto_if_not, goto_if_not_int_is_{true,zero})
- `rL` = single ref + label (goto_if_not_ptr_{iszero,nonzero})

This plan covers the `iiL` int family first; `ffL`, `rrL`, `iL`/`rL`
follow the same shape and land in the same commit group.

## Commit plan (small, each green on `pyre/check.sh`)

### Commit 1 — BC_* constants + assembler emitters

1. `majit/majit-metainterp/src/jitcode/mod.rs`:
   - Add 14 new `BC_*` constants (6 int, 6 float, 2 ptr).
   - Extend `wellknown_bh_insns()` with
     `goto_if_not_int_lt/iiL` …, `goto_if_not_float_lt/ffL` …,
     `goto_if_not_ptr_eq/rrL`, `goto_if_not_ptr_ne/rrL`.
2. `majit/majit-metainterp/src/jitcode/assembler.rs`:
   - Emit helpers:
     ```rust
     pub fn goto_if_not_int_lt(&mut self, a: u16, b: u16, label: u16) {
         self.touch_reg(a); self.touch_reg(b);
         self.push_u8(BC_GOTO_IF_NOT_INT_LT);
         self.push_u16(a); self.push_u16(b);
         self.push_label_ref(label);
     }
     ```
     and analogues for the rest of the int/float/ptr family.

No production path uses them yet — pure additive commit. `cargo build
-p pyrex --features cranelift` + `--features dynasm` + `pyre/check.sh`
must stay 14/14.

### Commit 2 — blackhole dispatch_one arms

`majit/majit-metainterp/src/blackhole.rs` `dispatch_one` (1633):

For each new `BC_GOTO_IF_NOT_<OP>_<TYPE>` arm, decode
`a_idx: u16`, `b_idx: u16`, `target: u16`, read the typed register
banks, and follow the RPython bhimpl semantics
(`blackhole.py:872-911`):

```rust
BC_GOTO_IF_NOT_INT_LT => {
    let a_idx = self.next_u16() as usize;
    let b_idx = self.next_u16() as usize;
    let target = self.next_u16() as usize;
    if self.registers_i[a_idx] >= self.registers_i[b_idx] {
        self.position = target;
    }
}
```

(`a < b` returns `pc` — fall through; else jump to `target`.)

Float and ptr follow the same shape using `registers_f` and
`registers_r` respectively. `_ptr_eq` compares pointers directly.

Again pure additive: the new arms are unreachable until codewriter
emits them. Verify no other bytecode numbers collided.

### Commit 3 — pyre codewriter fused emission

`pyre/pyre-jit/src/jit/codewriter.rs:660-725`:

1. Introduce a helper `peek_post_compare_branch(code, num_instrs,
py_pc)` that returns `Option<(branch_py_pc, jump_if_true, target_py_pc)>`
by walking past trivia (CACHE, EXTENDED_ARG, TO_BOOL) — this mirrors
`trace_opcode.rs::next_instruction_consumes_comparison_truth`.
2. Rewrite the `Instruction::CompareOp { opname }` arm:
   - If peek succeeds and both operands have a statically-known int/float
     type (available via the existing type channel used by
     `call_int_typed` dispatch, or conservatively "always emit the
     fused form; blackhole handles type mismatch via the generic
     bhimpl"), emit `goto_if_not_int_{opname}` via the new
     `JitCodeBuilder` helper and advance the outer loop past the
     PopJumpIf (and any intervening trivia).
   - Otherwise, fall back to the existing separate
     CompareOp + truth + branch path.
3. Reuse the existing `jump_target_forward` / `skip_caches` helpers for
   target resolution.

No new state fields are required — the fused-or-not decision is
determined entirely from the peek.

The separate arms (`Instruction::PopJumpIfFalse`, `PopJumpIfTrue`) stay
unchanged; they are still needed for the standalone case (e.g.
PopJumpIf right after a non-compare, or after TO_BOOL of a non-compare
value).

### Commit 4 — wire orthodox handlers + sanity harness

1. `majit/majit-metainterp/src/blackhole.rs`: no code change required,
   but confirm that `wire_handler("goto_if_not_int_lt/iiL", …)` (line
   5563) now succeeds (returns `true`) because Commit 1 added the
   opname to `wellknown_bh_insns()`. If there is an existing assertion
   relying on the old return value, update it.
2. Optional: add a focused integration test under
   `pyre/pyre-jit-trace/tests/` or an existing bh test that emits a
   short jitcode using the new emitter and runs `dispatch_loop` over
   it. (Kept optional because `pyre/check.sh` already covers the
   fused path via bench + integration.)

### Commit 5 — clean-up sweep

- Remove the comment reference at `codewriter.rs:660` that calls the
  separate emit "jtransform.py: rewrite_op_int_lt" once the fused
  emission is the primary path. Replace with
  `jtransform.py: optimize_goto_if_not` citation.
- Grep for any remaining "compare_fn_idx" / "truth_fn_idx" usage in
  the compare path; make sure the fallback (non-fuseable) still wires
  them.

## Verification per commit

1. `cargo build -p pyrex --features cranelift` + `cargo build -p pyrex
--features dynasm` — no errors, no warnings relative to baseline.
2. `bash pyre/check.sh` — `ALL PASSED: dynasm 14/14` +
   `ALL PASSED: cranelift 14/14`.
3. Diff a MAJIT_LOG trace of fib_recursive against the pre-plan
   baseline — the ResOp sequence must be bit-identical (same IntLt,
   same GuardTrue/False with same resume_pc); only the jitcode
   bytecode stream changes.

## Scope limits

- **Not** porting `int_is_true`, `int_is_zero`, `ptr_iszero`,
  `ptr_nonzero` fused forms — these fire on bare LoadFast + PopJumpIf
  without a preceding CompareOp and are a separate optimisation target.
- **Not** touching pyre tracer (`try_fused_compare_goto_if_not`
  stays; it is the tracer-layer analog of the codewriter fusion).
- **Not** touching `MAJIT_LOG` or metainterp ResOp encoding — the ops
  are unchanged, only the jitcode around them.

## Known risks

- Type witness at codewriter time: `CompareOp` arm currently does not
  know concrete Int vs Float vs Ref up front. Two mitigation options:
  (a) always emit the most general `goto_if_not` (single-bool branch)
  even after fusion, which gives up the `iiL` fast-path; (b) infer
  type from existing dispatch hints (e.g. the `compare_fn_idx` chosen
  already depends on operand kind). Prefer (b); fall back to (a) if
  untyped paths remain.
- wire_handler assertion: any code that checks `wire_handler` returns
  false for "not yet wired" opnames needs to be updated after
  Commit 1; audit before committing.
