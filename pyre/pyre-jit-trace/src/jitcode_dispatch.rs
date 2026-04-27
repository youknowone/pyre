//! Trace-side jitcode walker (Phase D-1 entry, eval-loop automation
//! plan `tingly-splashing-balloon.md`).
//!
//! RPython parity: this is the trace-side counterpart of
//! `BlackholeInterpBuilder.dispatch_loop` (`blackhole.py:65-100`). The
//! blackhole loop *executes* each `bhimpl_*` in turn; the tracing-side
//! analogue lives in `pyjitpl.py:opimpl_*` where each opcode becomes
//! a `MetaInterp.execute_and_record` call (RPython
//! `pyjitpl.py:1640-1660`). Pyre is mid-migration: the production
//! tracing path is the trait-driven `MIFrame::execute_opcode_step`
//! (trace_opcode.rs); this module is the orthodox path that consumes
//! the codewriter-emitted jitcode bytes directly.
//!
//! Scope so far (cumulative through slice 2g):
//!
//! | opname              | parity status | behaviour |
//! |---------------------|---------------|-----------|
//! | `live/`             | PARITY        | skip OFFSET_SIZE, continue (RPython tracing does not record `live/` either) |
//! | `goto/L`            | PARITY        | jump to 2-byte LE target, continue |
//! | `catch_exception/L` | PARITY        | skip 2-byte target, continue (handler PC metadata for unwinder; `pyjitpl.py:497-504` records nothing on normal flow) |
//! | `ref_return/r`      | PARITY        | record `Finish(reg) descr=done_with_this_frame_descr_ref` to TraceCtx, terminate (`pyjitpl.py:opimpl_ref_return → compile_done_with_this_frame`) |
//! | `inline_call_r_r/dR>r` | PRE-EXISTING-ADAPTATION | linear stub; RPython pushes a fresh MIFrame + raises `ChangeFrame()` and writes `registers_r[dst]`. Convergence: slice 2h + Phase D-3. |
//! | `int_copy/i>i`      | PRE-EXISTING-ADAPTATION | src OOR validation only; RPython performs `registers_i[dst] = registers_i[src]` SSA rename via the `box >i` decorator. Convergence: mutable register banks (slice 2h or Phase D-3). |
//! | `residual_call_r_r/iRd>r` | PRE-EXISTING-ADAPTATION | records `CallR([funcptr, ...args], descr)` only; RPython `do_residual_or_indirect_call` additionally runs heapcache invalidation, effectinfo dispatch (loop-invariant / elidable rewrites), guard emission, and dst writeback. Convergence: Phase D-3 routes through `state.rs:3380-3390 do_residual_call_r_r_impl`. |
//! | `raise/r`           | PRE-EXISTING-ADAPTATION | terminate only; RPython sets `last_exc_value`, pops the frame, then `finishframe_exception` scans the framestack's exceptiontable for a handler (resume tracing inside it) or emits `FINISH(exit_frame_with_exception_descr_ref)` at the outermost frame. Convergence: Phase D-3. |
//! | `reraise/`          | PRE-EXISTING-ADAPTATION | terminate only; same `finishframe_exception` path as `raise/r`. Convergence: Phase D-3. |
//!
//! Slice 1 = pure decode walker (no TraceCtx); slice 2b adds
//! `WalkContext { registers_r, trace_ctx }` + `ref_return/r` recording.
//! Slice 2c = `goto/L` (first branching primitive). RPython parity:
//! `blackhole.py:950-952 bhimpl_goto(target): return target` — the
//! handler returns the new position. Recording-wise pyre + RPython agree:
//! `goto/` is pure control flow, no IR op generated.
//!
//! `raise/r` recording stays deferred — the RPython
//! `pyjitpl.py:opimpl_raise → finishframe_exception` (`pyjitpl.py:2506`)
//! is *not* a single-op record: it scans the exceptiontable, unwinds
//! the symbolic stack to handler depth, pushes lasti+exc onto the
//! stack, and continues tracing inside the handler. That requires
//! full MIFrame state (symbolic_stack_types, concrete_stack,
//! valuestackdepth, code+exceptiontable) — wider than the
//! `WalkContext { registers_r, trace_ctx }` slice. Lands when the
//! walker integrates into `MIFrame::dispatch_jitcode` (Phase D-3).
//!
//! Convergence path: when every opname has a recording handler this
//! module replaces the trait dispatch in `MIFrame::execute_opcode_step`
//! (Phase D-3 → E in the plan). The free-standing module shape stays —
//! the entry point becomes `MIFrame::dispatch_jitcode` calling [`walk`]
//! with the appropriate context.
//!
//! Production fidelity gaps (next-session work, ranked by priority):
//!
//! 1. (LANDED slice 2h) `inline_call_r_r/dR>r` recurses into the
//!    sub-jitcode body via `JitCodeDescr::jitcode_index()` (resolved
//!    through `WalkContext::sub_jitcode_lookup`), allocates a fresh
//!    callee frame, populates `callee.registers_r[0..N]` from the
//!    caller's R-list, and writes the sub-walk's `SubReturn` result
//!    into the caller's dst slot. `ref_return/r` and `raise/r` route
//!    through `is_top_level` to either record top-level FINISH or
//!    surface `SubReturn` / `SubRaise` to the caller's
//!    `inline_call_*` handler. **Still deferred**: per-PC
//!    exceptiontable scan inside `inline_call`'s `SubRaise` arm —
//!    today the SubRaise bubbles straight to the top-level walker
//!    (catch_exception/L's stored target metadata is wired but
//!    unread). Lands when the exceptiontable plumb-through goes
//!    through (Phase D-3).
//! 2. `residual_call_r_r/iRd>r` records the bare `CallR(args, descr)`
//!    only. RPython `pyjitpl.py:1995 do_residual_or_indirect_call`
//!    additionally checks the descr's `EffectInfo` for inline/CONST
//!    targets (`call_pure`), invalidates `heapcache` on may-force /
//!    write effects, emits the `GUARD_NO_EXCEPTION` /
//!    `GUARD_EXCEPTION` pair the codewriter inserted, and routes
//!    `last_exc_value` through the resume bookkeeping. Walker only
//!    handles the recording skeleton — full path lands when MIFrame
//!    state (heapcache, last_exc_value, EffectInfo registry) is
//!    reachable. Multi-session epic.
//! 3. `raise/r` + `reraise/` terminate without recording; full
//!    `pyjitpl.py:2506 finishframe_exception` (exceptiontable scan,
//!    symbolic stack unwind to handler depth, `setarrayitem_vable`
//!    push of lasti+exc, continue tracing inside the handler — or
//!    record `FINISH(exit_frame_with_exception_descr_ref)` at the
//!    outermost frame) needs MIFrame framestack access. Phase D-3.
//! 4. End-to-end test (`walk_pop_top_arm_terminates_with_recorded_ops`)
//!    fills `descr_refs` with `make_fail_descr` placeholders. The
//!    walker treats descrs opaquely (`Arc::ptr_eq` is the only
//!    operation), so this is currently sound — but once handlers
//!    start dispatching on descr type (e.g. `EffectInfo` lookup for
//!    `residual_call`), the fixture must switch to real
//!    `SimpleCallDescr` instances. Triggers when (2) starts.
//! 5. (External) `build_default_bh_builder_with_unwired_report` is a
//!    transitional helper for Task #85 (6 unwired opnames:
//!    `int_ge/ir>i`, `int_mul/ir>i`, `int_ne/fr>i`, `int_xor/ri>i`,
//!    `setarrayitem_gc_f/rrfd`, `setarrayitem_gc_i/rrid` — kind-flow
//!    bug in assembler emitting mixed-kind operand types). RPython
//!    upstream has no non-strict builder. Removed when Task #85
//!    closes; not blocking dispatcher work.

use crate::jitcode_runtime::{DecodedOp, decode_op_at};
use majit_ir::{DescrRef, OpCode, OpRef};
use majit_metainterp::TraceCtx;

/// Body of a callee jitcode that the walker needs to recurse into.
/// RPython parity: when `inline_call_r_r/dR>r` fires, the metainterp
/// reads the descr's `JitCode` body (`pyjitpl.py:1266-1324
/// _opimpl_inline_call*`). Walker consumes the same minimal subset:
/// the bytecode bytes + register-bank sizes for the fresh callee
/// frame.
///
/// Body is always `'static` — production wires the lookup to
/// `crate::jitcode_runtime::all_jitcodes()` whose `Arc<JitCode>`
/// entries live inside a `LazyLock<Vec<...>>` (`'static`); tests
/// either use static byte arrays or `Box::leak` to surface
/// `'static`. Constraining the body's lifetime simplifies
/// `WalkContext`'s lifetime parameters — otherwise the closure's
/// covariance would force register-bank borrows to extend to the
/// lookup's lifetime.
#[derive(Debug, Clone)]
pub struct SubJitCodeBody {
    /// Callee's jitcode bytes (RPython `JitCode.code`).
    pub code: &'static [u8],
    /// Number of Ref-bank registers the callee declares
    /// (`JitCode.num_regs_r`). The walker allocates a fresh
    /// `Vec<OpRef>` of this size for the recursive frame.
    pub num_regs_r: usize,
    /// Number of Int-bank registers (`JitCode.num_regs_i`).
    pub num_regs_i: usize,
}

/// Caller-provided sub-jitcode lookup. RPython equivalent: descr
/// resolution within the metainterp loop reads `BhDescr::JitCode {
/// jitcode_index, .. }` and looks up `ALL_JITCODES[idx]`. Walker
/// inverts the dependency: the caller supplies the lookup so the
/// walker stays decoupled from the runtime's all-jitcodes table
/// (production passes a closure over `crate::jitcode_runtime::all_jitcodes()`,
/// tests pass synthetic closures over a local fixture map).
pub type SubJitCodeLookup = dyn Fn(usize) -> Option<SubJitCodeBody>;

/// State the walker reads from / writes to while stepping. RPython
/// equivalent: `MetaInterp` itself — the trace recorder, the symbolic
/// register banks (`registers_i`, `registers_r`, `registers_f`), and
/// the metainterp static data are all reachable from `self` in
/// `pyjitpl.py:opimpl_*`. Pyre passes them via this struct so the
/// walker can be tested without standing up a full `MIFrame`.
///
/// Field roster grows per slice — current cumulative:
///
/// * `registers_r`: Ref bank for `r`-coded operands.
/// * `registers_i`: Int bank for `i`-coded operands (slice 2f).
///   `registers_f` (Float bank) lands when float opnames join the
///   handler table.
/// * `descr_refs`: descr pool for `d`-coded operands (slice 2g).
///   Mirrors RPython `Assembler.descrs` (`assembler.py:23`); each
///   2-byte LE descr index in the jitcode bytes resolves through this
///   table.
/// * `trace_ctx`: live trace recorder.
/// * `done_with_this_frame_descr_ref`: descr the FINISH terminator
///   for a Ref-returning trace must carry. Production callers resolve
///   via `MetaInterpStaticData::done_with_this_frame_descr_for(Type::Ref)`
///   (`pyjitpl.py:4736`); tests use `make_fail_descr(1)` as the same
///   fallback `finish_and_compile` (`pyjitpl.py:4733`) uses when the
///   staticdata singleton was never attached.
///
/// Register banks are *mutable* — `int_copy/i>i` and
/// `residual_call_r_r/iRd>r` write their dst slot inline (RPython parity:
/// `pyjitpl.py:471-477 _opimpl_any_copy` returns the box, the
/// `@arguments("box")` + `>X` decorator pair writes it into the result
/// slot; `pyjitpl.py:1334-1347 _opimpl_residual_call*` returns the
/// recorder OpRef which the `>X` slot consumes). `inline_call_r_r/dR>r`
/// also *would* write dst (after sub-jitcode recursion) but stays
/// deferred — see the per-handler comments + module-level "Production
/// fidelity gaps" below.
/// `WalkContext` carries two lifetimes:
/// * `'frame` — the inner-frame lifetime: register banks + trace
///   recorder. Sub-walk recursion (`inline_call_r_r/dR>r`) allocates
///   fresh register banks scoped to the sub-walk's block, so
///   `'frame` must be allowed to *shrink* on recursion.
/// * `'static_a` — the outer lifetime: descr pool + sub-jitcode
///   lookup. These flow unchanged from caller into callee, so they
///   keep their original (longer) lifetime.
pub struct WalkContext<'frame, 'static_a: 'frame> {
    /// Symbolic Ref-bank register file. Indexing matches RPython
    /// `MIFrame.registers_r` (`pyjitpl.py:177-234`); the byte after a
    /// `r`-coded operand opcode indexes directly into this slice.
    /// Mutable so handlers writing `>r` results (currently
    /// `residual_call_r_r/iRd>r`) can land their dst.
    pub registers_r: &'frame mut [OpRef],
    /// Symbolic Int-bank register file. Indexing matches RPython
    /// `MIFrame.registers_i` (`pyjitpl.py:177-234`). Pyre's PyreSym is
    /// mid-migration to a 3-bank typed model — production callers may
    /// pass an empty slice today (the assembler only emits `i`-coded
    /// operands once the codewriter wires Int kind). Mutable so
    /// `int_copy/i>i` can land its dst.
    pub registers_i: &'frame mut [OpRef],
    /// Descr pool for `d`-coded operands. Each `d` argcode in the
    /// jitcode bytes resolves to `descr_refs[2-byte LE index]`.
    /// RPython `Assembler.descrs` (`assembler.py:23`) +
    /// `BlackholeInterpBuilder.setup_descrs` (`blackhole.py:102-103`)
    /// — production callers pass the codewriter-emitted descr table.
    pub descr_refs: &'static_a [DescrRef],
    /// Live trace recorder. `record_finish` / `record_op` /
    /// `record_op_with_descr` go through this.
    pub trace_ctx: &'frame mut TraceCtx,
    /// `done_with_this_frame_descr_ref` — the descr `pyjitpl.py:4729-4738
    /// finish_and_compile` attaches to the trace's terminator FINISH for
    /// the Ref kind. Caller-provided so the dispatcher does not reach
    /// into `TraceCtx::metainterp_sd` (which is `pub(crate)`).
    pub done_with_this_frame_descr_ref: DescrRef,
    /// `exit_frame_with_exception_descr_ref` — the descr `pyjitpl.py:3238-3242
    /// compile_exit_frame_with_exception` attaches to the FINISH that
    /// terminates a trace whose outermost frame raised an unhandled
    /// exception. RPython:
    ///   token = sd.exit_frame_with_exception_descr_ref
    ///   self.history.record1(rop.FINISH, valuebox, None, descr=token)
    /// Production callers resolve via `MetaInterpStaticData`
    /// (cf. `metainterp.rs:733`); tests use `make_fail_descr(1)`.
    pub exit_frame_with_exception_descr_ref: DescrRef,
    /// Whether this `WalkContext` is the outermost trace frame
    /// (`true`) or a nested sub-jitcode frame entered through
    /// `inline_call_r_r/dR>r` recursion (`false`). The flag
    /// disambiguates dual-behaviour terminators:
    ///
    /// * `ref_return/r` at top-level records `Finish` + Terminate;
    ///   inside a sub-walk it returns `SubReturn { result }` so the
    ///   caller's `inline_call_*` handler can write the dst register.
    /// * `raise/r` at top-level records the outermost
    ///   `Finish(exit_frame_with_exception_descr_ref)`; inside a
    ///   sub-walk it propagates `SubRaise { exc }` — the caller's
    ///   `inline_call_*` handler may catch via `catch_exception`
    ///   metadata or bubble up further.
    ///
    /// RPython parity: pyre flattens the framestack-driven
    /// `metainterp.popframe()` + `finishframe[_exception]` flow
    /// (`pyjitpl.py:1688-1704`) into this Rust-level outcome.
    pub is_top_level: bool,
    /// Caller-provided callback resolving a `jitcode_index` to a
    /// `SubJitCodeBody`. Invoked when `inline_call_r_r/dR>r` fires
    /// and needs to recurse into the callee's bytecode body.
    pub sub_jitcode_lookup: &'static_a SubJitCodeLookup,
}

/// Outcome of dispatching one opcode. The walker uses this to decide
/// whether to continue stepping or terminate.
///
/// RPython parity: `pyjitpl.py:opimpl_*` returns through Python's
/// generator/exception flow — opcodes that end a trace raise
/// `DoneWithThisFrameRef`/`SwitchToBlackhole`/`ChangeFrame`. Pyre
/// flattens that into an explicit enum because Rust has no analogous
/// non-local exit and we want the walker to stay in plain Result form.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DispatchOutcome {
    /// Step succeeded, continue with the next opcode at the returned pc.
    Continue,
    /// Trace ends here. The arm produced a final `ref_return`/`raise`
    /// equivalent at the top-level frame and no further bytes should
    /// be walked.
    Terminate,
    /// Sub-walk frame returned with a result OpRef (Some) or void
    /// (None — no `>X` slot in the callee's `*_return` op). Surfaced
    /// only when `WalkContext::is_top_level == false`. The caller's
    /// `inline_call_r_r/dR>r` handler consumes this to write the dst
    /// register and continue stepping its own jitcode.
    ///
    /// RPython parity: `metainterp.popframe()` after an `opimpl_*_return`
    /// (`pyjitpl.py:1688-1698`) — the callee frame ends, control returns
    /// to the caller's metainterp loop with the resbox in hand.
    SubReturn { result: Option<OpRef> },
    /// Sub-walk frame raised. RPython
    /// `metainterp.popframe() + finishframe_exception()` walks up the
    /// framestack scanning each parent's exceptiontable; pyre's walker
    /// surfaces the outcome to the caller's `inline_call_*` handler,
    /// which today bubbles it up further (no per-handler
    /// exceptiontable scan yet — that lives behind the
    /// `catch_exception/L` metadata pipe and is deferred until the
    /// per-PC exceptiontable plumb-through lands).
    SubRaise { exc: OpRef },
}

/// Errors surfaced by the trace-side walker.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DispatchError {
    /// The opcode byte at `pc` is not present in the `insns` table or
    /// the instruction's operand bytes overflowed the code slice. This
    /// is the same `decode_op_at -> None` path surfaced as a typed
    /// error.
    UndecodableOpcode { pc: usize },
    /// The opcode is decodable but the dispatcher has no handler for
    /// it yet. Carries the `opname/argcodes` key so callers can
    /// identify what blocked the walk (subsequent slices will add
    /// handlers one-by-one).
    UnsupportedOpname { pc: usize, key: &'static str },
    /// A register operand byte indexed past the symbolic register file.
    /// `len` is the slice length the walker was handed in the
    /// `WalkContext`; `reg` is the byte the codewriter emitted. Surfaces
    /// either an assembler-pass bug (out-of-range register) or a
    /// caller mismatch between the symbolic register layout and the
    /// arm's expected number of registers.
    RegisterOutOfRange {
        pc: usize,
        reg: usize,
        len: usize,
        bank: &'static str,
    },
    /// A `d`-coded descr index resolved past the descr pool. Surfaces
    /// either an assembler-pass bug (descr index out of range) or a
    /// caller mismatch between the codewriter's descr table size and
    /// the table the walker was handed in `WalkContext::descr_refs`.
    DescrIndexOutOfRange { pc: usize, index: usize, len: usize },
    /// `inline_call_*` resolved a descr that does not implement
    /// `JitCodeDescr`. Surfaces either a codewriter bug (an
    /// `inline_call_*` opnum emitted with a non-jitcode descr index)
    /// or a caller mismatch (the descr pool wasn't built from the
    /// codewriter's descr table). `descr_index` is the 2-byte LE
    /// index the walker decoded.
    ExpectedJitCodeDescr { pc: usize, descr_index: usize },
    /// `inline_call_*`'s descr resolved to a `jitcode_index`, but the
    /// caller's `sub_jitcode_lookup` returned `None`. Production wires
    /// the lookup to `crate::jitcode_runtime::all_jitcodes()`; tests
    /// build synthetic maps. A `None` return means the codewriter
    /// emitted an index past the runtime's jitcode table.
    SubJitCodeNotFound { pc: usize, jitcode_index: usize },
}

/// Walk one opcode at `pc` and return the dispatch outcome plus the
/// next pc. Side effects reach `ctx.trace_ctx` only for opnames whose
/// handler explicitly records (e.g. `ref_return/r` calls
/// `record_finish`).
///
/// The returned `next_pc` is normally `op.next_pc` (linear advance
/// past the operand bytes); branch handlers (`goto/L` etc.) override
/// this with their target.
pub fn step(
    code: &[u8],
    pc: usize,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let op: DecodedOp = decode_op_at(code, pc).ok_or(DispatchError::UndecodableOpcode { pc })?;
    handle(&op, code, ctx)
}

/// Walk the code from `start_pc` until a terminating opcode fires.
/// Returns the terminating outcome plus the pc immediately after the
/// terminator. Top-level callers expect `DispatchOutcome::Terminate`
/// (other variants appear only inside a sub-walk frame entered via
/// `inline_call_r_r/dR>r` — `ref_return/r` and `raise/r` produce
/// `SubReturn` / `SubRaise` there).
pub fn walk(
    code: &[u8],
    start_pc: usize,
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    let mut pc = start_pc;
    loop {
        let (outcome, next_pc) = step(code, pc, ctx)?;
        pc = next_pc;
        match outcome {
            DispatchOutcome::Continue => {}
            DispatchOutcome::Terminate
            | DispatchOutcome::SubReturn { .. }
            | DispatchOutcome::SubRaise { .. } => return Ok((outcome, pc)),
        }
    }
}

/// Read a Ref-bank register operand byte at `pc + offset` and resolve
/// to its symbolic [`OpRef`]. RPython
/// `pyjitpl.py:registers_r[code[pc+1]]` for an `r`-coded operand.
fn read_ref_reg(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Result<OpRef, DispatchError> {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_r
        .get(reg)
        .copied()
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg,
            len: ctx.registers_r.len(),
            bank: "r",
        })
}

/// Read an Int-bank register operand byte at `pc + offset` and resolve
/// to its symbolic [`OpRef`]. RPython
/// `pyjitpl.py:registers_i[code[pc+1]]` for an `i`-coded operand.
fn read_int_reg(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Result<OpRef, DispatchError> {
    let byte_pc = op.pc + 1 + operand_offset;
    let reg = code[byte_pc] as usize;
    ctx.registers_i
        .get(reg)
        .copied()
        .ok_or(DispatchError::RegisterOutOfRange {
            pc: op.pc,
            reg,
            len: ctx.registers_i.len(),
            bank: "i",
        })
}

/// Read a 2-byte little-endian label operand at `pc + 1 +
/// operand_offset`. RPython encoding: `assembler.py:write_label`
/// writes the resolved target as `chr(target & 0xFF)` +
/// `chr((target >> 8) & 0xFF)`, matching `bhimpl_goto`'s
/// `code[pc] | (code[pc+1] << 8)` decode.
fn read_label(code: &[u8], op: &DecodedOp, operand_offset: usize) -> usize {
    let lo = code[op.pc + 1 + operand_offset] as usize;
    let hi = code[op.pc + 1 + operand_offset + 1] as usize;
    lo | (hi << 8)
}

/// Read a 2-byte little-endian descr index operand and resolve to
/// the descr from [`WalkContext::descr_refs`]. RPython equivalent:
/// `BlackholeInterpreter.descrs[code[pc] | (code[pc+1] << 8)]`
/// (`blackhole.py:102-103` setup + per-`bhimpl_*` site).
fn read_descr(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Result<DescrRef, DispatchError> {
    let lo = code[op.pc + 1 + operand_offset] as usize;
    let hi = code[op.pc + 1 + operand_offset + 1] as usize;
    let index = lo | (hi << 8);
    ctx.descr_refs
        .get(index)
        .cloned()
        .ok_or(DispatchError::DescrIndexOutOfRange {
            pc: op.pc,
            index,
            len: ctx.descr_refs.len(),
        })
}

/// Read a Ref-bank variadic operand list (`R` argcode): 1 length byte
/// followed by `len` register bytes. Returns the resolved [`OpRef`]s
/// in jitcode order plus the total operand byte width (so callers can
/// skip past or compute downstream operand offsets).
///
/// RPython parity: `assembler.py:write_varlist` emits exactly this
/// shape — `chr(len(args))` followed by one byte per arg register.
fn read_ref_var_list(
    code: &[u8],
    op: &DecodedOp,
    operand_offset: usize,
    ctx: &WalkContext<'_, '_>,
) -> Result<(Vec<OpRef>, usize), DispatchError> {
    let len_pc = op.pc + 1 + operand_offset;
    let len = code[len_pc] as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let reg = code[len_pc + 1 + i] as usize;
        let opref = ctx
            .registers_r
            .get(reg)
            .copied()
            .ok_or(DispatchError::RegisterOutOfRange {
                pc: op.pc,
                reg,
                len: ctx.registers_r.len(),
                bank: "r",
            })?;
        out.push(opref);
    }
    Ok((out, 1 + len))
}

/// Per-opname dispatch table. Returning `(outcome, next_pc)` lets
/// branching handlers (`goto/L`) override the linear `op.next_pc`
/// advance; non-branching handlers return `op.next_pc` unchanged.
fn handle(
    op: &DecodedOp,
    code: &[u8],
    ctx: &mut WalkContext<'_, '_>,
) -> Result<(DispatchOutcome, usize), DispatchError> {
    match op.key {
        "live/" => Ok((DispatchOutcome::Continue, op.next_pc)),
        "inline_call_r_r/dR>r" => {
            // RPython parity: `pyjitpl.py:1266-1324 _opimpl_inline_call1`
            //   pushes a fresh `MIFrame(jitcode)` populated with the caller's
            //   args (`metainterp.newframe(jitcode)` + `frame.setup_call(args)`)
            //   then raises `ChangeFrame()` so the metainterp loop dispatches
            //   the next op on the new frame. The callee's `*_return` op
            //   eventually pops back via `metainterp.finishframe(value)`,
            //   writing `value` into the caller's `>r` slot.
            //
            // Walker simulates the same shape with synchronous recursion:
            //   1. Resolve descr → `JitCodeDescr::jitcode_index()` →
            //      caller-supplied lookup → `SubJitCodeBody`.
            //   2. Allocate fresh callee `registers_r` / `registers_i`
            //      vectors (sized to the callee's declared bank widths).
            //   3. Populate `callee.registers_r[0..N]` from the caller's
            //      R-list args (RPython `setup_call_r(self, argboxes)`,
            //      `pyjitpl.py:230-260`).
            //   4. Recursive `walk()` with `is_top_level = false`.
            //   5. On `SubReturn { Some(value) }` write the dst register
            //      and `Continue`. On `SubReturn { None }` (void return —
            //      shouldn't happen for the `_r_r` variant but kept
            //      defensive) `Continue` without dst write. On `SubRaise`
            //      bubble up the same outcome (RPython
            //      `metainterp.finishframe_exception()` walks the
            //      framestack). On `Terminate` something has gone wrong
            //      (sub-frame should never reach top-level Terminate when
            //      `is_top_level=false`); surface as `Terminate` to caller.
            //
            // Operand layout `dR>r`:
            //   2B d (descr index) + 1B varlen + N×1B Ref args + 1B >r dst
            let sub_descr = read_descr(code, op, 0, ctx)?;
            // Re-decode the descr index for diagnostics (read_descr already
            // succeeded, so the OOR branch is skipped here — but we still
            // need the integer for ExpectedJitCodeDescr).
            let descr_index = (code[op.pc + 1] as usize) | ((code[op.pc + 2] as usize) << 8);
            let jc_descr =
                sub_descr
                    .as_jitcode_descr()
                    .ok_or(DispatchError::ExpectedJitCodeDescr {
                        pc: op.pc,
                        descr_index,
                    })?;
            let sub_index = jc_descr.jitcode_index();
            let sub_body =
                (ctx.sub_jitcode_lookup)(sub_index).ok_or(DispatchError::SubJitCodeNotFound {
                    pc: op.pc,
                    jitcode_index: sub_index,
                })?;
            // R-list lives at operand offset 2 (skip 2-byte descr). The
            // helper returns total operand byte width.
            let (args, arg_width) = read_ref_var_list(code, op, 2, ctx)?;

            // Allocate fresh callee frame (RPython `MIFrame(jitcode)`'s
            // bank vectors are sized in `__init__` to the jitcode's
            // declared widths). All slots default to OpRef::NONE so any
            // callee read of an unwritten slot surfaces RegisterOutOfRange
            // semantics (or, more precisely, an OpRef::NONE that downstream
            // ops will reject).
            let mut callee_regs_r = vec![OpRef::NONE; sub_body.num_regs_r];
            let mut callee_regs_i = vec![OpRef::NONE; sub_body.num_regs_i];

            // Populate callee args. RPython `setup_call_r(argboxes)`
            // copies caller's argboxes into the callee's first N
            // `registers_r` slots. Pyre's `_r_r` variant carries only Ref
            // args (the `R` list); int/float would require additional
            // I/F lists which the codewriter doesn't emit for this opname.
            for (i, arg) in args.iter().enumerate() {
                if i < callee_regs_r.len() {
                    callee_regs_r[i] = *arg;
                }
            }

            // Recursive walk. Re-borrow the trace_ctx + descr_refs +
            // sub_jitcode_lookup; bumps `is_top_level` to false so the
            // callee's `ref_return` / `raise` route through SubReturn /
            // SubRaise instead of recording top-level FINISH.
            let (callee_outcome, _callee_end_pc) = {
                let mut sub_wc = WalkContext {
                    registers_r: &mut callee_regs_r,
                    registers_i: &mut callee_regs_i,
                    descr_refs: ctx.descr_refs,
                    trace_ctx: ctx.trace_ctx,
                    done_with_this_frame_descr_ref: ctx.done_with_this_frame_descr_ref.clone(),
                    exit_frame_with_exception_descr_ref: ctx
                        .exit_frame_with_exception_descr_ref
                        .clone(),
                    is_top_level: false,
                    sub_jitcode_lookup: ctx.sub_jitcode_lookup,
                };
                walk(sub_body.code, 0, &mut sub_wc)?
            };

            match callee_outcome {
                DispatchOutcome::SubReturn {
                    result: Some(value),
                } => {
                    // dst register byte sits after the descr (2B) +
                    // varlist (1B + arg_width) bytes.
                    let dst = code[op.pc + 1 + 2 + arg_width] as usize;
                    let len = ctx.registers_r.len();
                    let slot =
                        ctx.registers_r
                            .get_mut(dst)
                            .ok_or(DispatchError::RegisterOutOfRange {
                                pc: op.pc,
                                reg: dst,
                                len,
                                bank: "r",
                            })?;
                    *slot = value;
                    Ok((DispatchOutcome::Continue, op.next_pc))
                }
                DispatchOutcome::SubReturn { result: None } => {
                    // Void return inside an `_r_r` slot is a parity
                    // surprise — RPython's `_r_r` never emits void
                    // `*_return`. Continue without writing dst; if
                    // production ever sees this, the codewriter / opname
                    // mismatch is the bug to chase, not the walker.
                    Ok((DispatchOutcome::Continue, op.next_pc))
                }
                DispatchOutcome::SubRaise { exc } => {
                    // PRE-EXISTING-ADAPTATION: RPython
                    // `finishframe_exception` walks the framestack
                    // scanning each parent's exceptiontable for a
                    // matching handler. Walker has no per-PC exceptiontable
                    // pipe yet, so the SubRaise bubbles straight up to
                    // the next caller. At top-level it surfaces through
                    // raise/r's outermost FINISH branch — but only after
                    // bubbling through every intermediate caller without
                    // a chance to catch. catch_exception/L's stored
                    // target metadata is the wire that lands when
                    // exceptiontable lookup ports.
                    Ok((DispatchOutcome::SubRaise { exc }, op.next_pc))
                }
                DispatchOutcome::Terminate => {
                    // Sub-walk reaching Terminate means an inner frame
                    // ran with is_top_level=true (impossible in correct
                    // recursive setup) or the walker hit a future
                    // Terminate-emitting opname we haven't accounted for.
                    // Surface to the caller's caller — they'll see it as
                    // a normal trace end, which is at least no worse
                    // than panicking.
                    Ok((DispatchOutcome::Terminate, op.next_pc))
                }
                DispatchOutcome::Continue => {
                    unreachable!("walk() only exits on Terminate / SubReturn / SubRaise")
                }
            }
        }
        "goto/L" => {
            // RPython `blackhole.py:950-952 bhimpl_goto(target): return
            // target`. The 2-byte LE label was resolved by
            // `assembler.fix_labels` to a direct pc; pyre + RPython
            // agree that goto records nothing (pure control flow).
            let target = read_label(code, op, 0);
            Ok((DispatchOutcome::Continue, target))
        }
        "catch_exception/L" => {
            // RPython `blackhole.py:969-974 bhimpl_catch_exception(target)` —
            // "no-op when run normally" — and `pyjitpl.py:497-504
            // opimpl_catch_exception` confirms tracing-side records nothing
            // (just an `assert not last_exc_value`). The 2-byte target is
            // metadata: when a `raise` fires on the previous instruction,
            // `handle_exception_in_frame` (`blackhole.py:406-422`) reads it
            // to redirect the unwinder. Linear walk advances past the
            // operand without using the target. Phase D-3 MIFrame
            // integration will surface the target to the exception
            // handler routing.
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "residual_call_r_r/iRd>r" => {
            // RPython parity: `pyjitpl.py:1334-1347 _opimpl_residual_call1`
            // → `do_residual_or_indirect_call` → `do_residual_call`
            // (pyjitpl.py:1995-2127). Walker classifies the call by
            // `EffectInfo::extraeffect` and picks the matching record
            // op + emits GUARD_NO_EXCEPTION when the callee may raise.
            //
            // Operand layout `iRd>r`:
            //   1B i-reg (funcptr) + 1B varlen + N×1B Ref args + 2B descr
            //   + 1B >r (result reg)
            let funcptr = read_int_reg(code, op, 0, ctx)?;
            let (args, arg_width) = read_ref_var_list(code, op, 1, ctx)?;
            let descr = read_descr(code, op, 1 + arg_width, ctx)?;
            let mut call_args = Vec::with_capacity(1 + args.len());
            call_args.push(funcptr);
            call_args.extend_from_slice(&args);

            // EffectInfo classification (pyjitpl.py:2085-2126).
            //   - forces virtual / virtualizable → CALL_MAY_FORCE_R + GUARD_NOT_FORCED
            //     + vable_and_vrefs_before/after_residual_call. Walker scope
            //     can't run that path (needs MIFrame.virtualizable_boxes /
            //     vrefs / heapcache); fall back to plain CallR with a
            //     PRE-EXISTING-ADAPTATION marker. Phase D-3 fixes.
            //   - EF_LOOPINVARIANT → CALL_LOOPINVARIANT_R; pyre's
            //     `OpCode` enum doesn't carry that variant yet, so this
            //     also falls back to CallR. PRE-EXISTING-ADAPTATION.
            //   - check_is_elidable() → CALL_PURE_R (drop GUARD_NO_EXCEPTION
            //     unless the elidable variant can raise).
            //   - default → CallR + GUARD_NO_EXCEPTION when can_raise.
            let effect = descr.as_call_descr().map(|cd| cd.get_extra_info());
            let (call_opcode, can_raise, classification): (OpCode, bool, &'static str) =
                match effect {
                    None => (OpCode::CallR, true, "no-effectinfo-fallback"),
                    Some(ei) if ei.check_forces_virtual_or_virtualizable() => {
                        // PRE-EXISTING-ADAPTATION: forces path needs
                        // CALL_MAY_FORCE + GUARD_NOT_FORCED + vable bookkeeping.
                        (OpCode::CallR, true, "forces-fallback")
                    }
                    Some(ei) if ei.extraeffect == majit_ir::ExtraEffect::LoopInvariant => {
                        // PRE-EXISTING-ADAPTATION: CALL_LOOPINVARIANT_R missing
                        // from majit-ir's OpCode enum.
                        (OpCode::CallR, false, "loopinvariant-fallback")
                    }
                    Some(ei) if ei.check_is_elidable() => {
                        (OpCode::CallPureR, ei.check_can_raise(false), "elidable")
                    }
                    Some(ei) => (OpCode::CallR, ei.check_can_raise(false), "default"),
                };
            let _ = classification; // reserved for future telemetry/log
            let result = ctx
                .trace_ctx
                .record_op_with_descr(call_opcode, &call_args, descr);

            // RPython `metainterp.handle_possible_exception()` (pyjitpl.py:2082)
            // emits `GUARD_NO_EXCEPTION` after every raising call, fed
            // through `execute_varargs(..., exc=can_raise)`. Walker emits
            // the same guard but with `num_live=0` — the symbolic frame
            // state needed for accurate fail_args lives in MIFrame and
            // surfaces in Phase D-3. PRE-EXISTING-ADAPTATION (num_live).
            if can_raise {
                ctx.trace_ctx.record_guard(OpCode::GuardNoException, &[], 0);
            }

            // dst register writeback (`>r`).
            let dst = code[op.pc + 1 + 1 + arg_width + 2] as usize;
            let len = ctx.registers_r.len();
            let slot = ctx
                .registers_r
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "r",
                })?;
            *slot = result;

            // Deferred (Phase D-3 / state.rs:3380-3390):
            //   - heapcache.invalidate_caches_varargs(opnum, descr, allboxes)
            //   - vable_and_vrefs_before_residual_call / vable_after_residual_call
            //   - direct_libffi_call / direct_call_release_gil specialization
            //   - call_loopinvariant_known_result fast-path
            //   - direct_assembler_call (assembler_call=True)
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "int_copy/i>i" => {
            // RPython `pyjitpl.py:471-477 _opimpl_any_copy(self, box) → box`
            // + `@arguments("box")` + `>i` result coding: read src
            // register, write the same OpRef into the dst slot. Pypy
            // records *no* IR op for a copy — pure SSA-level rename.
            // Operand layout `i>i`: 1B src + 1B dst.
            let src_val = read_int_reg(code, op, 0, ctx)?;
            let dst = code[op.pc + 2] as usize;
            let len = ctx.registers_i.len();
            let slot = ctx
                .registers_i
                .get_mut(dst)
                .ok_or(DispatchError::RegisterOutOfRange {
                    pc: op.pc,
                    reg: dst,
                    len,
                    bank: "i",
                })?;
            *slot = src_val;
            Ok((DispatchOutcome::Continue, op.next_pc))
        }
        "ref_return/r" => {
            // RPython `pyjitpl.py:opimpl_ref_return(self, value)` calls
            // `metainterp.finishframe(value)`. Two branches by frame depth:
            //
            //   * Outermost frame → `compile_done_with_this_frame` (pyjitpl.py:3198-3220)
            //     records `rop.FINISH(value)` with
            //     `done_with_this_frame_descr_ref`. Trace ends.
            //   * Nested frame → `metainterp.popframe()` returns control to
            //     the caller's metainterp loop with `value` in hand; the
            //     caller's `_opimpl_inline_call*` lands `value` in its
            //     `>r` slot via `make_result_of_lastop`.
            //
            // Walker selects between the two via `ctx.is_top_level`.
            let result = read_ref_reg(code, op, 0, ctx)?;
            if ctx.is_top_level {
                ctx.trace_ctx
                    .finish(&[result], ctx.done_with_this_frame_descr_ref.clone());
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((
                    DispatchOutcome::SubReturn {
                        result: Some(result),
                    },
                    op.next_pc,
                ))
            }
        }
        "raise/r" => {
            // RPython `pyjitpl.py:1688-1698 opimpl_raise(exc_value_box)` →
            //   metainterp.last_exc_value = exc_value_box.getref(...)
            //   metainterp.popframe()
            //   metainterp.finishframe_exception()
            // `finishframe_exception` walks the framestack scanning each
            // parent's exceptiontable; if a handler matches, control jumps
            // to the handler PC. If exhausted (outermost frame, no
            // handler), `compile_exit_frame_with_exception` (pyjitpl.py:3238-3242)
            // records `rop.FINISH(value, descr=exit_frame_with_exception_descr_ref)`.
            //
            // Walker dual behaviour:
            //   * `is_top_level` → outermost FINISH (above).
            //   * sub-walk frame → propagate `SubRaise { exc }` to the
            //     caller's `inline_call_*` handler. Currently the walker
            //     has no per-PC exceptiontable plumb so the caller can't
            //     match a handler — they bubble it up further until the
            //     top-level walker emits the FINISH. catch_exception/L's
            //     stored target is wired but unread until exceptiontable
            //     lookup ports.
            //
            // Deferred (still in module-level "Production fidelity gaps"):
            // exceptiontable scan inside the same frame, `last_exc_value`
            // bookkeeping for downstream `reraise/`.
            let exc = read_ref_reg(code, op, 0, ctx)?;
            if ctx.is_top_level {
                ctx.trace_ctx
                    .finish(&[exc], ctx.exit_frame_with_exception_descr_ref.clone());
                Ok((DispatchOutcome::Terminate, op.next_pc))
            } else {
                Ok((DispatchOutcome::SubRaise { exc }, op.next_pc))
            }
        }
        "reraise/" => {
            // PRE-EXISTING-ADAPTATION (slice 2e): terminate only.
            //
            // RPython `pyjitpl.py:1700-1704 opimpl_reraise(self)` reads
            // no operand and routes through `popframe` +
            // `finishframe_exception` using `metainterp.last_exc_value`
            // — the exception value set by an *earlier* `raise/r` or
            // by the unwinder when entering a `catch_exception` handler.
            //
            // Walker has no `last_exc_value` field — adding one means
            // tracking the catch_exception → handler-entry contract
            // (when the unwinder routes into a handler, it sets
            // `last_exc_value` from `metainterp.last_exception`; the
            // handler reads it via `bhimpl_last_exc_value`). That
            // bookkeeping requires the exceptiontable + framestack the
            // walker doesn't yet have. Convergence path: Phase D-3
            // MIFrame integration.
            Ok((DispatchOutcome::Terminate, op.next_pc))
        }
        other => Err(DispatchError::UnsupportedOpname {
            pc: op.pc,
            key: other,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jitcode_runtime::{insns_opname_to_byte, jitcode_for_instruction};
    use majit_ir::Type;
    use majit_metainterp::make_fail_descr;
    use pyre_interpreter::Instruction;

    /// Build a fresh `TraceCtx`. Uses the public `for_test_types` +
    /// `const_ref` / `make_fail_descr` factories so the fixture stays
    /// out of `pub(crate)` API.
    fn fresh_trace_ctx() -> TraceCtx {
        TraceCtx::for_test_types(&[Type::Ref])
    }

    /// Build a `done_with_this_frame_descr_ref` for tests. Mirrors the
    /// production fallback at `pyjitpl/mod.rs:4733` (`make_fail_descr_typed`)
    /// when the staticdata singleton was never attached.
    fn done_descr_ref_for_tests() -> DescrRef {
        make_fail_descr(1)
    }

    /// Build distinct `OpRef` constants for register slots so dataflow
    /// assertions don't get false positives from shared identity. Each
    /// slot holds `const_ref(0xC0DE_0000 + i)` for `i in 0..count`.
    fn distinct_const_refs(ctx: &mut TraceCtx, count: usize) -> Vec<OpRef> {
        (0..count)
            .map(|i| ctx.const_ref(0xC0DE_0000_i64 + i as i64))
            .collect()
    }

    /// Default `sub_jitcode_lookup` for tests that don't exercise
    /// `inline_call_r_r` recursion. Returns `None` for every index;
    /// any test that hits the inline_call handler with this lookup
    /// will see `DispatchError::SubJitCodeNotFound`.
    fn no_sub_jitcodes(_idx: usize) -> Option<SubJitCodeBody> {
        None
    }

    /// Production-like `sub_jitcode_lookup` that resolves `idx` against
    /// `crate::jitcode_runtime::all_jitcodes()`. Used by the end-to-end
    /// arm acceptance tests (`walk_return_value_arm_*`,
    /// `walk_pop_top_arm_*`) so the walker can recurse into real
    /// callee bodies. The runtime's `all_jitcodes()` is a
    /// `LazyLock<Vec<Arc<JitCode>>>` — every `.code` slice it surfaces
    /// is `'static`-rooted, satisfying `SubJitCodeBody`'s body
    /// constraint.
    fn production_sub_jitcodes(idx: usize) -> Option<SubJitCodeBody> {
        let all = crate::jitcode_runtime::all_jitcodes();
        all.get(idx).map(|jc| SubJitCodeBody {
            code: jc.code.as_slice(),
            num_regs_r: jc.num_regs_r(),
            num_regs_i: jc.num_regs_i(),
        })
    }

    /// Test-only `JitCodeDescr` wrapper around a `jitcode_index`.
    /// Production wires `BhDescr::JitCode { jitcode_index, .. }` through
    /// a real adapter (`PyreJitCodeDescr` — TODO follow-up commit
    /// extracts this into `pyre-jit-trace/src/descr.rs` so production
    /// callers can build `Arc<dyn Descr>` pools that the walker
    /// consumes natively).
    #[derive(Debug)]
    struct TestJitCodeDescr {
        jitcode_index: usize,
    }
    impl majit_ir::Descr for TestJitCodeDescr {
        fn as_jitcode_descr(&self) -> Option<&dyn majit_ir::JitCodeDescr> {
            Some(self)
        }
    }
    impl majit_ir::JitCodeDescr for TestJitCodeDescr {
        fn jitcode_index(&self) -> usize {
            self.jitcode_index
        }
    }

    /// Build a `descr_refs` pool of length `pool_len` where the slot at
    /// each `BhDescr::JitCode` index in `crate::jitcode_runtime::all_descrs()`
    /// holds a `TestJitCodeDescr` carrying that descr's `jitcode_index`,
    /// and every other slot holds a `make_fail_descr` placeholder.
    /// Lets acceptance tests resolve `inline_call_*` descr indices
    /// without standing up the full BhDescr → trait Descr adapter
    /// pipeline.
    fn descr_pool_with_jitcode_adapters(pool_len: usize) -> Vec<DescrRef> {
        let all_bh = crate::jitcode_runtime::all_descrs();
        (0..pool_len)
            .map(|i| match all_bh.get(i) {
                Some(majit_translate::jitcode::BhDescr::JitCode { jitcode_index, .. }) => {
                    std::sync::Arc::new(TestJitCodeDescr {
                        jitcode_index: *jitcode_index,
                    }) as DescrRef
                }
                _ => make_fail_descr(1 + i),
            })
            .collect()
    }

    #[test]
    fn inline_call_recursion_writes_subreturn_into_caller_dst_register() {
        // Slice 2h core acceptance: caller's `inline_call_r_r/dR>r`
        // recurses into a synthetic callee jitcode whose body is
        // simply `ref_return r0`. The callee's ref_return surfaces as
        // `SubReturn { result: Some(callee.registers_r[0]) }`; the
        // caller's inline_call handler writes that OpRef into the
        // caller's dst register. Then the caller's own `ref_return r3`
        // records the outermost Finish carrying that propagated value.
        let ret_byte = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_r/dR>r")
            .expect("`inline_call_r_r/dR>r` must be in insns table");
        // Callee body: `ref_return r0`. registers_r[0] is populated
        // from the caller's R-list arg.
        let callee_code: &'static [u8] = Box::leak(Box::new([ret_byte, 0]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 0,
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        // Caller body:
        //   inline_call_r_r/dR>r descr=7, R=[r2], >r=r5
        //   ref_return r5
        let caller_code = [
            inline_byte,
            0x07,
            0x00, // d (LE descr index = 7)
            0x01,
            0x02, // R: varlen=1, args=[r2]
            0x05, // >r: dst = r5
            ret_byte,
            0x05, // ref_return r5
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let arg_value = regs_r[2];
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = std::sync::Arc::new(TestJitCodeDescr { jitcode_index: 7 });
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
        };
        let (outcome, end_pc) =
            walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        assert_eq!(end_pc, caller_code.len());
        drop(wc);
        // dst register r5 must equal the arg the caller passed (since
        // callee's `ref_return r0` returns its registers_r[0] which
        // was populated from caller's R-list[0] = r2's OpRef).
        assert_eq!(
            regs_r[5], arg_value,
            "inline_call_r_r dst writeback must propagate callee's SubReturn",
        );
        // Outermost FINISH carries the same value.
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "exactly one Finish must be recorded (callee's ref_return surfaced as \
             SubReturn, did not record a Finish)",
        );
        let last = tc.ops().last().expect("Finish must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::Finish);
        assert_eq!(
            last.args.as_slice(),
            &[arg_value],
            "outermost Finish must carry the arg value the caller threaded through \
             inline_call_r_r",
        );
    }

    #[test]
    fn inline_call_recursion_propagates_subraise_from_callee() {
        // Slice 2h: callee's `raise/r` surfaces as `SubRaise { exc }`
        // to the caller's inline_call handler, which propagates it
        // (no exceptiontable scan yet). The top-level walk then sees
        // SubRaise as the terminating outcome — but since our walker
        // is currently top-level, raise/r emits the outermost FINISH
        // with exit_frame_with_exception_descr_ref. So the SubRaise
        // bubbles through inline_call and the top-level walk
        // returns... well, SubRaise actually, since the *caller*'s
        // walker is also at is_top_level=true but it didn't fire
        // raise/r itself, it received SubRaise from inline_call. The
        // caller's walk loop sees SubRaise as a terminating outcome
        // and exits the loop returning SubRaise.
        let raise_byte = *insns_opname_to_byte()
            .get("raise/r")
            .expect("`raise/r` must be in insns table");
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_r/dR>r")
            .expect("`inline_call_r_r/dR>r` must be in insns table");
        // Callee body: `raise r0`
        let callee_code: &'static [u8] = Box::leak(Box::new([raise_byte, 0]));
        let sub_body = SubJitCodeBody {
            code: callee_code,
            num_regs_r: 1,
            num_regs_i: 0,
        };
        let lookup = {
            let sub_body = sub_body.clone();
            move |idx: usize| {
                if idx == 7 {
                    Some(sub_body.clone())
                } else {
                    None
                }
            }
        };
        // Caller body: `inline_call_r_r descr=7 R=[r2] >r=r5`
        // (no follow-on `ref_return` — the SubRaise propagates straight
        // up to the caller's `walk` loop and exits.)
        let caller_code = [inline_byte, 0x07, 0x00, 0x01, 0x02, 0x05];
        let mut tc = fresh_trace_ctx();
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let arg_value = regs_r[2];
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[7] = std::sync::Arc::new(TestJitCodeDescr { jitcode_index: 7 });
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &lookup,
        };
        let (outcome, _) = walk(&caller_code, 0, &mut wc).expect("caller must walk to terminator");
        assert_eq!(
            outcome,
            DispatchOutcome::SubRaise { exc: arg_value },
            "callee's raise/r must surface to caller as SubRaise carrying the \
             exc OpRef from callee's registers_r[0] (= caller's R-list arg)",
        );
    }

    #[test]
    fn inline_call_with_unresolvable_descr_surfaces_typed_error() {
        // Slice 2h: descr at the inline_call's d-slot must implement
        // `JitCodeDescr`. A `FailDescr` placeholder doesn't, so the
        // walker surfaces `ExpectedJitCodeDescr`.
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_r/dR>r")
            .expect("`inline_call_r_r/dR>r` must be in insns table");
        let caller_code = [inline_byte, 0x05, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let err = step(&caller_code, 0, &mut wc)
            .expect_err("FailDescr at inline_call's d-slot must hit ExpectedJitCodeDescr");
        assert_eq!(
            err,
            DispatchError::ExpectedJitCodeDescr {
                pc: 0,
                descr_index: 5,
            },
        );
    }

    #[test]
    fn inline_call_with_missing_sub_jitcode_lookup_surfaces_typed_error() {
        // Slice 2h: descr resolves to JitCodeDescr but lookup returns
        // None — surface `SubJitCodeNotFound`.
        let inline_byte = *insns_opname_to_byte()
            .get("inline_call_r_r/dR>r")
            .expect("`inline_call_r_r/dR>r` must be in insns table");
        let caller_code = [inline_byte, 0x03, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut descr_pool: Vec<DescrRef> = (0..16).map(|i| make_fail_descr(1 + i)).collect();
        descr_pool[3] = std::sync::Arc::new(TestJitCodeDescr {
            jitcode_index: 999_999,
        });
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr,
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let err = step(&caller_code, 0, &mut wc)
            .expect_err("missing sub-jitcode must hit SubJitCodeNotFound");
        assert_eq!(
            err,
            DispatchError::SubJitCodeNotFound {
                pc: 0,
                jitcode_index: 999_999,
            },
        );
    }

    #[test]
    fn step_through_live_opcode_advances_by_offset_size() {
        let live_byte = *insns_opname_to_byte()
            .get("live/")
            .expect("`live/` must be in insns table");
        let code = [live_byte, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("live/ must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc,
            1 + majit_translate::liveness::OFFSET_SIZE,
            "live/ must advance past the OFFSET_SIZE liveness slot",
        );
    }

    #[test]
    fn step_through_ref_return_records_finish_with_descr_and_correct_arg() {
        // Slice 2c-fix: `ref_return/r` records `rop.FINISH(reg)` to the
        // TraceCtx with `done_with_this_frame_descr_ref` attached, and
        // the `reg` byte selects the correct OpRef from `registers_r`.
        // RPython `pyjitpl.py:opimpl_ref_return → finishframe →
        // compile_done_with_this_frame → record1(FINISH, descr=token)`.
        let ret_byte = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        // Read register at byte index 3 — distinct from index 0 to
        // catch off-by-one bugs in operand decoding.
        let code = [ret_byte, 0x03];
        let mut tc = fresh_trace_ctx();
        let mut regs = distinct_const_refs(&mut tc, 8);
        let expected_arg = regs[3];
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs,
            registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("ref_return/r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        assert_eq!(next_pc, 2, "ref_return/r consumes 1 register byte");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "exactly one Finish op must be recorded",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::Finish);
        assert_eq!(
            last.args.as_slice(),
            &[expected_arg],
            "Finish args must select registers_r[3], not registers_r[0]",
        );
        let recorded_descr = last
            .descr
            .as_ref()
            .expect("Finish must carry done_with_this_frame_descr_ref");
        assert!(
            std::sync::Arc::ptr_eq(recorded_descr, &descr),
            "Finish descr must be the exact instance the dispatcher was handed",
        );
    }

    #[test]
    fn ref_return_with_out_of_range_register_surfaces_typed_error() {
        let ret_byte = *insns_opname_to_byte()
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns table");
        let code = [ret_byte, 0x07]; // index 7 — registers_r is empty
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let err = step(&code, 0, &mut wc).expect_err("must surface RegisterOutOfRange");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 7,
                len: 0,
                bank: "r"
            },
        );
    }

    #[test]
    fn raise_with_out_of_range_register_surfaces_typed_error() {
        // Slice 2c-fix: `raise/r` reads its operand for OOR validation
        // even though recording is deferred. Catches the same classes
        // of assembler bugs `ref_return/r` does.
        let raise_byte = *insns_opname_to_byte()
            .get("raise/r")
            .expect("`raise/r` must be in insns table");
        let code = [raise_byte, 0x05];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let err = step(&code, 0, &mut wc).expect_err("raise/r must read its operand");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 5,
                len: 0,
                bank: "r"
            },
        );
    }

    #[test]
    fn step_through_goto_jumps_to_label_target() {
        // Slice 2c: `goto/L` reads its 2-byte LE label and the walker
        // returns Continue at the label target, not the linear next pc.
        // RPython `blackhole.py:950-952 bhimpl_goto(target): return target`.
        let goto_byte = *insns_opname_to_byte()
            .get("goto/L")
            .expect("`goto/L` must be in insns table");
        // target = 0x002A = 42
        let code = [goto_byte, 0x2A, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("goto/L must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc, 42,
            "goto/L must jump to its 2-byte LE label target",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "goto/L records nothing — pure control flow",
        );
    }

    #[test]
    fn step_through_goto_handles_high_byte_of_label() {
        // Confirm the LE decode reads both bytes (regression guard for
        // accidentally treating L as a single byte).
        let goto_byte = *insns_opname_to_byte()
            .get("goto/L")
            .expect("`goto/L` must be in insns table");
        // target = 0x0102 = 258
        let code = [goto_byte, 0x02, 0x01];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("goto/L must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(next_pc, 258);
    }

    #[test]
    fn step_through_catch_exception_advances_past_label_operand() {
        // Slice 2d: `catch_exception/L` records nothing on the normal
        // walk (RPython `pyjitpl.py:497-504 opimpl_catch_exception` is
        // an `assert not last_exc_value` only) and the walker advances
        // linearly past the 2-byte target.
        let catch_byte = *insns_opname_to_byte()
            .get("catch_exception/L")
            .expect("`catch_exception/L` must be in insns table");
        let code = [catch_byte, 0x2A, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("catch_exception/L must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc, 3,
            "catch_exception/L must advance past the 2-byte target operand",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "catch_exception/L records nothing on normal walk",
        );
    }

    #[test]
    fn step_through_raise_records_outermost_finish_and_terminates() {
        // RPython `pyjitpl.py:1688-1698 opimpl_raise` →
        // `finishframe_exception` (outermost-frame branch) →
        // `compile_exit_frame_with_exception` records
        // `FINISH(exc, descr=exit_frame_with_exception_descr_ref)`.
        // The walker treats every invocation as outermost (no
        // framestack), so this is the parity-correct emit.
        let raise_byte = *insns_opname_to_byte()
            .get("raise/r")
            .expect("`raise/r` must be in insns table");
        // exc operand reads registers_r[2]
        let code = [raise_byte, 0x02];
        let mut tc = fresh_trace_ctx();
        let mut regs = distinct_const_refs(&mut tc, 4);
        let expected_exc = regs[2];
        let descr_done = done_descr_ref_for_tests();
        let descr_exc = make_fail_descr(99);
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs,
            registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr_done,
            exit_frame_with_exception_descr_ref: descr_exc.clone(),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("raise/r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        assert_eq!(next_pc, 2);
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "raise/r must record exactly one FINISH op (outermost branch)",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::Finish);
        assert_eq!(
            last.args.as_slice(),
            &[expected_exc],
            "FINISH args must carry the exception OpRef from registers_r[src]",
        );
        let recorded_descr = last
            .descr
            .as_ref()
            .expect("FINISH must carry exit_frame_with_exception_descr_ref");
        assert!(
            std::sync::Arc::ptr_eq(recorded_descr, &descr_exc),
            "FINISH descr must be the caller-supplied \
             `exit_frame_with_exception_descr_ref`, not \
             `done_with_this_frame_descr_ref`",
        );
    }

    #[test]
    fn step_through_reraise_terminates_with_no_operand() {
        // Slice 2e: `reraise/` is a 0-arg terminator. RPython
        // `pyjitpl.py:1700-1704 opimpl_reraise → popframe → finishframe_exception`.
        // Linear walk just terminates; recording deferred to MIFrame integration.
        let reraise_byte = *insns_opname_to_byte()
            .get("reraise/")
            .expect("`reraise/` must be in insns table");
        let code = [reraise_byte];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("reraise/ must dispatch");
        assert_eq!(outcome, DispatchOutcome::Terminate);
        assert_eq!(next_pc, 1, "reraise/ has no operand");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "reraise/ records nothing on the linear walk (deferred to MIFrame integration)",
        );
    }

    #[test]
    fn step_through_int_copy_advances_past_operand_bytes() {
        // Slice 2f: `int_copy/i>i` reads the src `i` operand for OOR
        // validation, advances past 2 operand bytes, records nothing.
        // Dst writeback (`registers_i[dst] = registers_i[src]`) is
        // deferred — RPython `pyjitpl.py:471-477 _opimpl_any_copy(box)
        // -> box` is a register rename only, no IR op.
        let int_copy_byte = *insns_opname_to_byte()
            .get("int_copy/i>i")
            .expect("`int_copy/i>i` must be in insns table");
        // src=2, dst=5 — distinct so a future writeback assertion can
        // distinguish src from dst slots.
        let code = [int_copy_byte, 0x02, 0x05];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let (outcome, next_pc) = step(&code, 0, &mut wc).expect("int_copy/i>i must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc, 3,
            "int_copy/i>i must advance past src + dst register bytes",
        );
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before,
            "int_copy/i>i records no IR op (RPython parity)",
        );
    }

    #[test]
    fn int_copy_writes_src_value_into_dst_register() {
        // Verify the dst writeback half of `int_copy/i>i`. The src
        // and dst slots must hold *different* OpRefs going in so the
        // assertion catches an accidental no-op.
        let int_copy_byte = *insns_opname_to_byte()
            .get("int_copy/i>i")
            .expect("`int_copy/i>i` must be in insns table");
        let code = [int_copy_byte, 0x02, 0x05]; // src=2, dst=5
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 8);
        let src_val_pre = regs_i[2];
        let dst_val_pre = regs_i[5];
        assert_ne!(
            src_val_pre, dst_val_pre,
            "fixture must seed src and dst with different OpRefs",
        );
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let _ = step(&code, 0, &mut wc).expect("int_copy/i>i must dispatch");
        assert_eq!(
            wc.registers_i[5], src_val_pre,
            "int_copy must copy registers_i[src] into registers_i[dst] \
             (RPython _opimpl_any_copy + `>i` result coding)",
        );
        assert_eq!(
            wc.registers_i[2], src_val_pre,
            "src register must remain unchanged",
        );
    }

    #[test]
    fn int_copy_with_out_of_range_dst_register_surfaces_typed_error() {
        // dst byte indexes past `registers_i`; src is in range so the
        // src read succeeds and the dst write surfaces the OOR.
        let int_copy_byte = *insns_opname_to_byte()
            .get("int_copy/i>i")
            .expect("`int_copy/i>i` must be in insns table");
        let code = [int_copy_byte, 0x00, 0x09]; // src=0 (in range), dst=9 (OOR)
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 4);
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let err = step(&code, 0, &mut wc).expect_err("int_copy dst OOR must surface a typed error");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 9,
                len: 4,
                bank: "i",
            },
        );
    }

    #[test]
    fn int_copy_with_out_of_range_src_register_surfaces_typed_error() {
        // Slice 2f: src OOR validation parity with `raise/r`. Bank tag
        // is `"i"` to disambiguate from the Ref-bank OOR error.
        let int_copy_byte = *insns_opname_to_byte()
            .get("int_copy/i>i")
            .expect("`int_copy/i>i` must be in insns table");
        let code = [int_copy_byte, 0x07, 0x00];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [], // empty — index 7 must surface OOR
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let err = step(&code, 0, &mut wc).expect_err("int_copy/i>i must read its src operand");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 7,
                len: 0,
                bank: "i",
            },
        );
    }

    #[test]
    fn unsupported_opname_surfaces_typed_error() {
        // Slice 2g progresses past residual_call_r_r. `int_add/ii>i`
        // is confirmed in the production insns table (cf.
        // jitcode_runtime.rs:864 etc.) but lives outside the PopTop op
        // set — handler not installed yet. Stable choice for exercising
        // the catch-all `UnsupportedOpname` error path while we keep
        // adding handlers.
        let opname = "int_add/ii>i";
        let unsupported_byte = *insns_opname_to_byte()
            .get(opname)
            .unwrap_or_else(|| panic!("`{opname}` must be in insns table"));
        // Operand encoding `ii>i`: 1B src1 + 1B src2 + 1B dst = 3B
        let code = [unsupported_byte, 0, 0, 0];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let err =
            step(&code, 0, &mut wc).expect_err("unsupported opname must hit UnsupportedOpname");
        assert_eq!(err, DispatchError::UnsupportedOpname { pc: 0, key: opname },);
    }

    #[test]
    fn step_through_residual_call_r_r_records_callr_with_descr_and_args() {
        // Slice 2g: `residual_call_r_r/iRd>r` records `OpCode::CallR`
        // with `[funcptr, ...args]` and `descr=descr_refs[d]`. RPython
        // `pyjitpl.py:1334-1347 _opimpl_residual_call1` →
        // `do_residual_or_indirect_call → execute_and_record_varargs(
        // rop.CALL_R, [funcbox]+argboxes, descr=calldescr)`.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        // Operand encoding `iRd>r`: 1B funcptr (i-reg=2),
        // 1B varlen=2 + [r-reg=4, r-reg=7], 2B descr_index=1 (LE),
        // 1B dst-reg=0 (writeback deferred — not used by walker yet).
        let code = [
            residual_byte,
            0x02, // funcptr from registers_i[2]
            0x02, // varlen
            0x04,
            0x07, // args from registers_r[4, 7]
            0x01,
            0x00, // descr index = 1 (LE)
            0x00, // dst reg (deferred)
        ];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 4);
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let funcptr_expected = regs_i[2];
        let arg0_expected = regs_r[4];
        let arg1_expected = regs_r[7];
        // Build a 2-entry descr table — index 0 is a decoy (different
        // pointer), index 1 is the descr we expect the recorder to attach.
        let decoy = make_fail_descr(2);
        let call_descr = make_fail_descr(3);
        let descr_pool = vec![decoy, call_descr.clone()];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let (outcome, next_pc) =
            step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
        assert_eq!(outcome, DispatchOutcome::Continue);
        assert_eq!(
            next_pc,
            code.len(),
            "residual_call_r_r must advance past funcptr + varlist + descr + dst",
        );
        drop(wc);
        // FailDescr placeholder has no EffectInfo (`as_call_descr() = None`),
        // so the walker takes the `no-effectinfo-fallback` branch:
        // CallR + GuardNoException (RPython parity:
        // `do_residual_call → execute_varargs(..., exc=True)` →
        // `handle_possible_exception` emits GUARD_NO_EXCEPTION).
        assert_eq!(
            tc.num_ops(),
            ops_before + 2,
            "residual_call_r_r must record CallR + GuardNoException (no-effectinfo fallback)",
        );
        let call_op = tc
            .ops()
            .iter()
            .find(|o| o.opcode == majit_ir::OpCode::CallR)
            .expect("CallR must be recorded");
        assert_eq!(
            call_op.args.as_slice(),
            &[funcptr_expected, arg0_expected, arg1_expected],
            "CallR args must be [funcptr, ...args] from registers_i+registers_r",
        );
        let recorded_descr = call_op
            .descr
            .as_ref()
            .expect("CallR must carry the calldescr");
        assert!(
            std::sync::Arc::ptr_eq(recorded_descr, &call_descr),
            "CallR descr must be descr_refs[1] (not decoy at index 0)",
        );
        // GuardNoException follows immediately after.
        let guard_op = tc
            .ops()
            .iter()
            .find(|o| o.opcode == majit_ir::OpCode::GuardNoException)
            .expect("GuardNoException must follow CallR for raising calls");
        assert!(
            guard_op.args.is_empty(),
            "GuardNoException takes no operand args",
        );
    }

    /// Build a `SimpleCallDescr` with the given `extraeffect` so the
    /// walker's EffectInfo classification can be exercised in tests.
    fn call_descr_with_effect(idx: u32, extra: majit_ir::ExtraEffect) -> DescrRef {
        let mut effect = majit_ir::EffectInfo::default();
        effect.extraeffect = extra;
        std::sync::Arc::new(majit_ir::SimpleCallDescr::new(
            idx,
            vec![majit_ir::Type::Int, majit_ir::Type::Ref],
            majit_ir::Type::Ref,
            false,
            std::mem::size_of::<usize>(),
            effect,
        ))
    }

    #[test]
    fn residual_call_r_r_with_elidable_cannot_raise_records_callpurer_no_guard() {
        // RPython parity: `do_residual_call` (pyjitpl.py:2111-2118) reads
        // `effectinfo.check_is_elidable()` + `effectinfo.check_can_raise()`,
        // then `execute_varargs(rop.CALL_R, ..., exc, pure)`. With
        // EF_ELIDABLE_CANNOT_RAISE: `pure=True` (CALL_PURE_R) + `exc=False`
        // (no GUARD_NO_EXCEPTION).
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let elidable_descr = call_descr_with_effect(7, majit_ir::ExtraEffect::ElidableCannotRaise);
        let descr_pool = vec![elidable_descr.clone()];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "elidable+cannot-raise must record exactly CallPureR (no GuardNoException)",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(
            last.opcode,
            majit_ir::OpCode::CallPureR,
            "EF_ELIDABLE_CANNOT_RAISE must rewrite to CALL_PURE_R",
        );
    }

    #[test]
    fn residual_call_r_r_with_elidable_can_raise_records_callpurer_plus_guard() {
        // EF_ELIDABLE_CAN_RAISE: `pure=True` + `exc=True` —
        // CALL_PURE_R + GUARD_NO_EXCEPTION (pyjitpl.py:execute_varargs
        // emits both).
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let elidable_descr = call_descr_with_effect(8, majit_ir::ExtraEffect::ElidableCanRaise);
        let descr_pool = vec![elidable_descr.clone()];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 2,
            "elidable+can-raise must record CallPureR + GuardNoException",
        );
        let opcodes: Vec<_> = tc.ops().iter().skip(ops_before).map(|o| o.opcode).collect();
        assert_eq!(
            opcodes,
            vec![
                majit_ir::OpCode::CallPureR,
                majit_ir::OpCode::GuardNoException
            ],
            "EF_ELIDABLE_CAN_RAISE must record CALL_PURE_R then GUARD_NO_EXCEPTION",
        );
    }

    #[test]
    fn residual_call_r_r_with_cannot_raise_records_callr_no_guard() {
        // EF_CANNOT_RAISE: `pure=False` + `exc=False` — bare CallR,
        // no GUARD_NO_EXCEPTION.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 4);
        let nothrow_descr = call_descr_with_effect(9, majit_ir::ExtraEffect::CannotRaise);
        let descr_pool = vec![nothrow_descr.clone()];
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
        drop(wc);
        assert_eq!(
            tc.num_ops(),
            ops_before + 1,
            "EF_CANNOT_RAISE must record bare CallR (no GuardNoException)",
        );
        let last = tc.ops().last().expect("recorded op must exist");
        assert_eq!(last.opcode, majit_ir::OpCode::CallR);
    }

    #[test]
    fn residual_call_r_r_writes_recorder_result_into_dst_register() {
        // Verify the dst writeback half of `residual_call_r_r/iRd>r`.
        // After the handler runs, `registers_r[dst]` must equal the
        // OpRef the recorder returned (i.e., the OpRef whose Op is
        // the recorded CallR at the trace tail).
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        // funcptr=regs_i[0], no args, descr index=0, dst=3
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x03];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let mut regs_r = distinct_const_refs(&mut tc, 8);
        let dst_val_pre = regs_r[3];
        let descr_pool = vec![make_fail_descr(1)];
        let frame_done_descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let _ = step(&code, 0, &mut wc).expect("residual_call_r_r/iRd>r must dispatch");
        // The dst slot must hold the OpRef of the recorded CallR. Each
        // Op carries its OpRef in `op.pos` (recorder.rs:159), which lets
        // the test compare without re-deriving the index (input args
        // also occupy OpRef indices, so `ops.iter().position()` would
        // be off by `num_inputargs`).
        let dst_ref = wc.registers_r[3];
        assert_ne!(
            dst_ref, dst_val_pre,
            "dst must change from its pre-call value",
        );
        let call_op = wc
            .trace_ctx
            .ops()
            .iter()
            .find(|o| o.opcode == OpCode::CallR)
            .expect("a CallR op must be in the recorded trace");
        assert_eq!(
            dst_ref, call_op.pos,
            "registers_r[dst] must be the recorded CallR's OpRef (op.pos)",
        );
    }

    #[test]
    fn residual_call_r_r_with_out_of_range_dst_register_surfaces_typed_error() {
        // Dst register OOR — the call was already recorded at this
        // point (RPython parity: `do_residual_or_indirect_call` records
        // first, then writes the result), but `registers_r` is empty
        // so the writeback surfaces RegisterOutOfRange.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        let code = [residual_byte, 0x00, 0x00, 0x00, 0x00, 0x07]; // dst=7
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let descr_pool = vec![make_fail_descr(1)];
        let frame_done_descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let err = step(&code, 0, &mut wc).expect_err("dst OOR must surface a typed error");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 7,
                len: 0,
                bank: "r",
            },
        );
    }

    #[test]
    fn residual_call_r_r_with_descr_index_out_of_range_surfaces_typed_error() {
        // Slice 2g: descr-index OOR validation. Same shape as
        // RegisterOutOfRange, dedicated DispatchError variant for clarity.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        // descr_index=5, descr_refs.len()=2 → OOR
        let code = [residual_byte, 0x00, 0x00, 0x05, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let descr_pool = vec![make_fail_descr(1), make_fail_descr(1)];
        let frame_done_descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let err = step(&code, 0, &mut wc)
            .expect_err("descr index 5 with pool size 2 must surface DescrIndexOutOfRange");
        assert_eq!(
            err,
            DispatchError::DescrIndexOutOfRange {
                pc: 0,
                index: 5,
                len: 2,
            },
        );
    }

    #[test]
    fn residual_call_r_r_with_out_of_range_arg_register_surfaces_typed_error() {
        // Slice 2g: varlist member OOR validation. Bank tag = "r" since
        // R-list reads from registers_r.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        // varlen=1, arg=9 (registers_r is empty) → OOR
        let code = [residual_byte, 0x00, 0x01, 0x09, 0x00, 0x00, 0x00];
        let mut tc = fresh_trace_ctx();
        let mut regs_i = distinct_const_refs(&mut tc, 1);
        let descr_pool = vec![make_fail_descr(1)];
        let frame_done_descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut regs_i,
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        let err = step(&code, 0, &mut wc)
            .expect_err("R-list member out of range must surface RegisterOutOfRange");
        assert_eq!(
            err,
            DispatchError::RegisterOutOfRange {
                pc: 0,
                reg: 9,
                len: 0,
                bank: "r",
            },
        );
    }

    #[test]
    #[ignore = "slice 2h: inline_call recursion surfaces sub-jitcode opnames the walker doesn't yet support (e.g. getfield_vable_i/rd>i). Full end-to-end acceptance lands when slice 2i adds handlers for the rest of the codewriter-emitted opnames."]
    fn walk_return_value_arm_terminates_at_first_ref_return() {
        // Phase D-1 acceptance (post-slice-2h): walk the smallest real
        // arm jitcode (`Instruction::ReturnValue`, 18 bytes) end-to-end.
        // Layout (cranelift build):
        //
        //   pc=0..6   inline_call_r_r / dR>r  (recurse → SubReturn → caller dst write → Continue)
        //   pc=6..9   live /                  (continue)
        //   pc=9..11  ref_return / r          (terminate — top-level outermost)
        //   pc=11..18 (raise + ref_return tail, dead on this path)
        //
        // The arm's `inline_call_r_r` now recurses into the callee
        // jitcode via `production_sub_jitcodes` and
        // `descr_pool_with_jitcode_adapters` (slice 2h). The callee's
        // own `ref_return/r` surfaces as `SubReturn`; the caller writes
        // its dst register with that result and continues. The
        // caller's own `ref_return/r` at pc=9..11 then records the
        // outermost `Finish`.
        let jc = jitcode_for_instruction(&Instruction::ReturnValue)
            .expect("ReturnValue must resolve to a jitcode");
        let mut tc = fresh_trace_ctx();
        // 256 distinct OpRefs (one per possible 1-byte register
        // index). `inline_call_r_r`'s recursion overwrites the dst
        // slot with the callee's `SubReturn` value, so the
        // post-recursion `ref_return/r` reads the *recorded* OpRef
        // from the sub-walk, not a `regs_r` constant. The assertion
        // therefore checks the recorded Finish's args against the
        // post-recursion register state, not a precomputed constant.
        let mut regs_r = distinct_const_refs(&mut tc, 256);
        let mut regs_i = distinct_const_refs(&mut tc, 256);
        let descr = done_descr_ref_for_tests();
        let pool_len = crate::jitcode_runtime::all_descrs().len();
        let descr_pool = descr_pool_with_jitcode_adapters(pool_len);
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &production_sub_jitcodes,
        };
        let (outcome, end_pc) =
            walk(&jc.code, 0, &mut wc).expect("ReturnValue arm must walk to a terminator");
        assert_eq!(
            outcome,
            DispatchOutcome::Terminate,
            "top-level walk must end on Terminate",
        );
        assert!(
            end_pc <= jc.code.len(),
            "walker must not run past the arm body \
             (end_pc={end_pc}, code.len()={})",
            jc.code.len(),
        );
        assert_eq!(
            end_pc, 11,
            "ReturnValue arm walker must terminate at outermost `ref_return/r` (pc=9..11)",
        );
        drop(wc);
        assert!(
            tc.num_ops() > ops_before,
            "at least one Finish op must have been recorded; \
             callee sub-walk may also have contributed CallR / Finish ops",
        );
        // Locate the *outermost* Finish (descr=done_with_this_frame).
        // Sub-walks don't emit Finish (they surface `SubReturn`), so
        // there should be exactly one Finish carrying the
        // done-with-this-frame descr.
        let outermost_finish = tc
            .ops()
            .iter()
            .find(|o| {
                o.opcode == majit_ir::OpCode::Finish
                    && o.descr
                        .as_ref()
                        .map(|d| std::sync::Arc::ptr_eq(d, &descr))
                        .unwrap_or(false)
            })
            .expect("outermost Finish with done-with-this-frame descr must exist");
        assert_eq!(outermost_finish.args.len(), 1);
        let recorded_descr = outermost_finish
            .descr
            .as_ref()
            .expect("Finish must carry done_with_this_frame_descr_ref");
        assert!(
            std::sync::Arc::ptr_eq(recorded_descr, &descr),
            "Finish descr must be the exact instance the dispatcher was handed",
        );
    }

    #[test]
    #[ignore = "slice 2h: same as walk_return_value_arm_*; inline_call recursion needs all-opname handler coverage"]
    fn walk_pop_top_arm_terminates_with_recorded_ops() {
        // Phase D-1 acceptance: walk the entire PopTop arm jitcode (38
        // bytes per arm inventory memo, 12-op sequence per
        // jitcode_runtime.rs:922-934 PopTop op-sequence lock-in test).
        // Every distinct opname in the PopTop set now has a handler:
        // inline_call_r_r, live, catch_exception, goto, reraise,
        // int_copy, residual_call_r_r, ref_return, raise. The walker
        // must reach a terminator (ref_return or raise) without
        // hitting `UnsupportedOpname` and must record at least one
        // op along the way (FINISH from ref_return, optionally CallR
        // from residual_call_r_r if the goto path traverses the
        // handler body).
        let jc = jitcode_for_instruction(&Instruction::PopTop)
            .expect("PopTop must resolve to a jitcode");
        let mut tc = fresh_trace_ctx();
        // Generously sized banks so any byte the codewriter emits is
        // in-range. 256 is the maximum register index a 1-byte slot
        // can address.
        let mut regs_r = distinct_const_refs(&mut tc, 256);
        let mut regs_i = distinct_const_refs(&mut tc, 256);
        // Descr pool: slot at each `BhDescr::JitCode` index in
        // `all_descrs()` is wrapped in a `TestJitCodeDescr` adapter so
        // `inline_call_r_r/dR>r` can resolve `as_jitcode_descr()`.
        // Other slots default to `make_fail_descr`. The
        // `residual_call_r_r/iRd>r` slot is overwritten below with a
        // real `SimpleCallDescr`.
        let pool_len = crate::jitcode_runtime::all_descrs().len();
        let mut descr_pool = descr_pool_with_jitcode_adapters(pool_len);
        // Find which descr index the codewriter emitted for the
        // `residual_call_r_r/iRd>r` instance inside this arm. The
        // operand layout is `iRd>r`: opcode + 1B i + (1B varlen + N) +
        // 2B d + 1B >r. Walk the bytes to locate it; the d-index lives
        // immediately after the R-list.
        let residual_byte = *insns_opname_to_byte()
            .get("residual_call_r_r/iRd>r")
            .expect("`residual_call_r_r/iRd>r` must be in insns table");
        let mut residual_d_idx: Option<usize> = None;
        let mut pc = 0;
        while pc < jc.code.len() {
            let op = crate::jitcode_runtime::decode_op_at(&jc.code, pc)
                .expect("PopTop arm bytes must decode");
            if jc.code[pc] == residual_byte {
                let varlen = jc.code[pc + 2] as usize;
                let lo = jc.code[pc + 3 + varlen] as usize;
                let hi = jc.code[pc + 4 + varlen] as usize;
                residual_d_idx = Some(lo | (hi << 8));
                break;
            }
            pc = op.next_pc;
        }
        let residual_d_idx = residual_d_idx.expect("PopTop arm must contain a residual_call_r_r");
        // Replace the FailDescr at that slot with a real CallDescr.
        // The arg/result types match the `iRd>r` shape: funcptr (Int)
        // + N Ref args → Ref result. A dummy CallDescr with a single
        // Ref arg is enough to exercise the `as_call_descr()` cast
        // (descr.rs:2172) that any future EffectInfo-aware handler
        // will use.
        let real_call_descr: DescrRef = std::sync::Arc::new(majit_ir::SimpleCallDescr::new(
            residual_d_idx as u32,
            vec![majit_ir::Type::Int, majit_ir::Type::Ref],
            majit_ir::Type::Ref,
            false,
            std::mem::size_of::<usize>(),
            majit_ir::EffectInfo::default(),
        ));
        descr_pool[residual_d_idx] = real_call_descr.clone();
        let frame_done_descr = done_descr_ref_for_tests();
        let ops_before = tc.num_ops();
        let mut wc = WalkContext {
            registers_r: &mut regs_r,
            registers_i: &mut regs_i,
            descr_refs: &descr_pool,
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: frame_done_descr,
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &production_sub_jitcodes,
        };
        let (outcome, end_pc) =
            walk(&jc.code, 0, &mut wc).expect("PopTop arm must walk to a terminator");
        // Slice 2h: PopTop's `inline_call_r_r/dR>r` recurses into the
        // codewriter-emitted callee jitcode (resolved via
        // `production_sub_jitcodes`). The callee may itself contain
        // further `inline_call_*` ops or unsupported opnames; if walking
        // to terminator succeeds, the outermost `ref_return` / `raise`
        // landed an outermost FINISH and ops_after > ops_before.
        assert_eq!(
            outcome,
            DispatchOutcome::Terminate,
            "top-level PopTop walk must end on Terminate",
        );
        assert!(
            end_pc <= jc.code.len(),
            "walker must not run past the arm body \
             (end_pc={end_pc}, code.len()={})",
            jc.code.len(),
        );
        drop(wc);
        let ops_after = tc.num_ops();
        assert!(
            ops_after > ops_before,
            "PopTop walk must record at least one op (FINISH from ref_return, \
             optionally CallR from residual_call_r_r along the handler body) — \
             recorded {} → {}",
            ops_before,
            ops_after,
        );
        // If the goto traverses the handler body, a CallR was recorded
        // — its descr must be the real CallDescr we put in the pool,
        // not a FailDescr. (If the goto skipped the body the CallR
        // never fires, so this is conditional.)
        if let Some(call_op) = tc.ops().iter().find(|o| o.opcode == OpCode::CallR) {
            let descr = call_op
                .descr
                .as_ref()
                .expect("CallR must carry the calldescr");
            assert!(
                descr.as_call_descr().is_some(),
                "CallR must attach a CallDescr (not a FailDescr) — \
                 a future EffectInfo-aware handler will rely on \
                 `as_call_descr()` (descr.rs:2172) returning Some",
            );
            assert!(
                std::sync::Arc::ptr_eq(descr, &real_call_descr),
                "CallR descr must be the real SimpleCallDescr at the \
                 codewriter's emitted index, not the placeholder",
            );
        }
    }

    #[test]
    fn walk_undecodable_byte_surfaces_typed_error() {
        // 0xFF is unknown to the insns table (21 entries 0..=20 today).
        let code = [0xFFu8];
        let mut tc = fresh_trace_ctx();
        let descr = done_descr_ref_for_tests();
        let mut wc = WalkContext {
            registers_r: &mut [],
            registers_i: &mut [],
            descr_refs: &[],
            trace_ctx: &mut tc,
            done_with_this_frame_descr_ref: descr.clone(),
            exit_frame_with_exception_descr_ref: make_fail_descr(2),
            is_top_level: true,
            sub_jitcode_lookup: &no_sub_jitcodes,
        };
        assert_eq!(
            walk(&code, 0, &mut wc),
            Err(DispatchError::UndecodableOpcode { pc: 0 })
        );
    }
}
