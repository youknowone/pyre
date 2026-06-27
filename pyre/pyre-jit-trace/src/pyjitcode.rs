//! `PyJitCode`: pyre's per-CodeObject JitCode wrapper.
//!
//! RPython's `JitCode` (jitcode.py:9) is a single class that owns
//! both the bytecode (`code` / `constants_*` / `c_num_regs_*`) and
//! the per-graph metadata (`name`, `fnaddr`, `calldescr`,
//! `jitdriver_sd`). pyre still has a split runtime representation:
//!
//!   * `majit_metainterp::jitcode::JitCode` is the current runtime
//!     adapter bytecode container (`code`, `constants_*`, `num_regs_*`,
//!     plus pyre-only `exec.*` pools). It is not the canonical
//!     codewriter `majit_translate::jitcode::JitCode`.
//!   * `PyJitCode` (this struct) wraps that JitCode together with
//!     pyre-only translation metadata — `pc_map` (Python PC → byte
//!     offset), the runtime `w_code` wrapper, and register layout — that
//!     RPython does not need because RPython's bytecode PCs are already
//!     JitCode PCs.
//!
//! The struct lives in `pyre-jit-trace` (the lower crate) so that
//! both the codewriter (`pyre-jit::jit::codewriter`) and the
//! trace/blackhole runtime (this crate) can hold the same
//! `Arc<PyJitCode>` instances. RPython's `MetaInterpStaticData.jitcodes`
//! list and `CallControl.jitcodes` dict reference identical
//! `JitCode` Python objects via Python's reference semantics; pyre
//! mirrors the shared-identity part with `Arc<PyJitCode>`. The wrapped
//! runtime `JitCode` allocation is also kept stable when the codewriter
//! fills a shell, because `inline_call_*` descriptors hold the callee
//! `JitCode` object itself in the RPython model.
//!
//! ## Discriminator: 3-state mode mapping
//!
//! A `PyJitCode` is one of three modes, encoded across two flags:
//!
//! | mode             | `jitcode.code` | `metadata.pc_map` | predicate                |
//! |------------------|----------------|--------------------|--------------------------|
//! | Skeleton         | empty          | empty              | [`PyJitCode::is_skeleton`]       |
//! | PortalBridge     | non-empty      | empty              | [`PyJitCode::is_portal_bridge`]  |
//! | PerCodeObject    | non-empty      | non-empty          | [`PyJitCode::is_populated`]      |
//!
//! `code` and `pc_map` are independent because the portal-bridged
//! install ([`crate::canonical_bridge::install_portal_for`]) reuses
//! the canonical portal `JitCode.code` byte stream but skips the
//! per-Python-PC mapping (the portal dispatches via its own arms on
//! `pycode.instructions[pc]`). Drained CodeWriter installs do both:
//! fill real instructions into `code` and stamp `pc_map` to
//! `code.instructions.len()`. Skeletons have neither because they are
//! placeholder slots inserted by `CallControl::get_jitcode` before the
//! assembler drain runs.
//!
//! The fourth combination (`code` empty, `pc_map` non-empty) is not
//! produced by any production path; the predicates classify it as
//! neither Skeleton nor PortalBridge nor PerCodeObject. Test fixtures
//! that fabricate this combination (e.g. by calling [`PyJitCode::skeleton`]
//! and then pushing into `metadata.pc_map`) flow as PerCodeObject for
//! [`PyJitCode::is_populated`] purposes (the historical predicate
//! looks at `pc_map` only).
//!
//! Convergence path: RPython's single `JitCode` class has neither flag
//! to consult — `assembler.assemble` populates `code` in place and
//! per-PC mapping is implicit in the bytecode stream. pyre will lose
//! the dual-mode discrimination once the codewriter routes Python
//! bytecode through the canonical RPython codewriter pipeline (Phase
//! G.4.4+). Until then, the mode mapping above is the source of truth
//! for every reader that branches on install style.

use majit_metainterp::jitcode::JitCode as RuntimeJitCode;
use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};

/// Pyre-only metadata attached to a Python CodeObject's compiled JitCode.
///
/// RPython does not need these fields because its bytecode PCs are already
/// JitCode PCs. Pyre translates CPython bytecode to JitCode lazily, so the
/// translation maps live here instead of polluting either upstream's
/// canonical `JitCode` or pyre's eventual single-store replacement.
pub struct PyJitCodeMetadata {
    /// py_pc → jitcode byte offset. Named for RPython's `frame.pc →
    /// jitcode position` flow; the runtime side reads this to map
    /// the Python frame's `next_instr` to the JitCode entry point
    /// for blackhole resume / inline call tracing.
    pub pc_map: Vec<usize>,
    /// py_pc → jitcode byte offset of the post-`residual_call` `-live-`
    /// marker (the one immediately preceding the opcode's own
    /// `catch_exception`), `None` for PCs that do not make a residual
    /// call.  RPython keeps `frame.pc` at this position for
    /// `capture_resumedata(after_residual_call=True, resumepc=-1)`
    /// (`pyjitpl.py:2610-2624`); pyre stores Python PCs in the snapshot
    /// and translates through `pc_map`, so after-residual-call resume
    /// needs this second map to reach the call's own catch rather than
    /// the next opcode's start marker (`blackhole.py:396-410
    /// handle_exception_in_frame`).  Same length as `pc_map`.
    pub after_residual_call_resume_pc: Vec<Option<usize>>,
    /// py_pc → jitcode byte offset of the FIRST instruction the opcode
    /// emitted (`usize::MAX` for PCs that emit no jitcode of their own:
    /// trivia, folded ops).  `pc_map` resolves each PC to its nearest
    /// `-live-` marker at-or-before, so adjacent PCs share marker
    /// positions and the map is not invertible; the full-body walk needs
    /// the exact inverse (jitcode pc → containing Python opcode) for
    /// guard resume coordinates, which this table provides.  Same length
    /// as `pc_map`.
    pub first_jit_pc_by_py_pc: Vec<usize>,
    /// Value-stack depth at each Python PC, in slots above stack_base.
    pub depth_at_py_pc: Vec<u16>,
    /// Post-regalloc Ref-bank color of the portal jitdriver's first red
    /// argument (`frame`).  RPython parity: `pypy/module/pypyjit/
    /// interp_jit.py:67 reds = ['frame', 'ec']` declares the portal
    /// calling convention, and `JitDriverStaticData.red_args_indices`
    /// (`rpython/jit/metainterp/warmspot.py`) records the inputarg
    /// position of each red arg.  This field is the pyre equivalent —
    /// the snapshot serializer at
    /// `pyre-jit-trace::trace_opcode::get_list_of_active_boxes` uses it
    /// to map the live_r color back to the symbolic `sym.frame` OpRef.
    /// `u16::MAX` for portal-bridge installs that don't run the
    /// per-CodeObject regalloc (the snapshot helper sentinel-skips).
    pub portal_frame_reg: u16,
    /// Post-regalloc Ref-bank color of the portal jitdriver's second red
    /// argument (`ec`, `pypy/module/pypyjit/interp_jit.py:67`).
    /// Snapshot serializer maps this color to `sym.execution_context`.
    pub portal_ec_reg: u16,
    /// Whether the body was compiled with the PORTAL entry shape
    /// (`FrameInputs::Portal`: `[frame, ec]` red inputs + frame-vable
    /// locals prologue) — `jitdriver_sd_from_portal_graph(code)` was
    /// `Some` at compile time.  A body first compiled as a plain CALLEE
    /// (`FrameInputs::Frame`, discovered through another function's
    /// call) reads its params from caller-seeded registers and stays
    /// frozen once installed trace-side (resume data captured against
    /// it must stay consistent), so a later portal trace of the same
    /// code must NOT walk it: `run_perfn_walk` declines on
    /// `!built_as_portal` and the trait tracer compiles the function.
    pub built_as_portal: bool,
    /// Absolute start index of the operand stack in PyFrame.locals_cells_stack_w.
    pub stack_base: usize,
    /// Post-regalloc
    /// color of each Python-semantic stack slot.
    /// `stack_slot_color_map[d]` = `apply_rename(Kind::Ref, nlocals + d)`
    /// for `d in 0..code.max_stackdepth` (= CPython `co_stacksize`).
    /// The `+ nlocals` here is the register-space stack base used by
    /// the codewriter (`RegisterLayout::stack_base`, which
    /// `RegisterLayout::compute` sets to `nlocals as u16`), NOT the
    /// `stack_base` field above (which is the PyFrame absolute
    /// `varnames.len() + ncells`). Populated in `finalize_jitcode`
    /// after `apply_rename` runs; portal-bridge installs
    /// (`canonical_bridge::install_portal_for`) populate it as
    /// identity over the same range.
    ///
    /// Length invariant: `stack_slot_color_map.len() == code.max_stackdepth`,
    /// so the bridge fallback at `state.rs::setup_bridge_sym`
    /// (`stack_base + stack_slot_color_map.len()`) reconstructs the full
    /// runtime PyFrame allocation
    /// (`pyframe.rs:1576` `nlocals + ncells + max_stackdepth`). Earlier
    /// per-CodeObject installs sized this to
    /// `max_stack_depth_observed = max(depth_at_pc)` which under-sized
    /// the map when JIT-traced PCs did not reach the static peak; the
    /// `co_stacksize` invariant restores parity with the runtime.
    ///
    /// After the stack-slot input-arg pinning removal (regalloc.rs:448-466 +
    /// :527-535) the stack-slot input-arg pinning is gone, so this
    /// map is no longer the identity `[stack_base, stack_base+1, …]`
    /// — entries are whatever color `apply_rename` produced. Decoders
    /// (`state.rs`, `trace_opcode.rs`, `codewriter.rs`) must read
    /// through the map; they cannot assume the old `nlocals + d`
    /// invariant.
    ///
    /// Tail caveat: `regalloc::allocate_registers` only pins
    /// `[0..nlocals)` plus the portal red args; pre-indices
    /// `nlocals + d` for `d >= max(depth_at_py_pc)` never appear in
    /// any SSA op, so `apply_rename` falls through with identity and
    /// `stack_slot_color_map[d] == nlocals + d` by accident, not by
    /// post-regalloc decision. Today every consumer reads only the
    /// `[0..depth_at_py_pc[pc])` prefix for value recovery and uses
    /// `len()` solely for frame-allocation length matching, so the
    /// tail's identity-by-fallthrough is harmless. If a future
    /// consumer needs full-range colors as real post-regalloc values,
    /// extend `external` in `regalloc.rs:680-697` to cover
    /// `(nlocals..nlocals + max_stackdepth)` so `enforce_input_args`
    /// pins the tail too (parity with `flatten.py:88-100`).
    pub stack_slot_color_map: Vec<u16>,
    /// SSA-authoritative live_r epic slice 3a:
    /// post-regalloc color of each Python-semantic local slot.
    /// `pyre_color_for_semantic_local[i]` = `apply_rename(Kind::Ref, i)`
    /// for `i in 0..code.varnames.len()`. Populated in `finalize_jitcode`
    /// after `apply_rename` runs, parallel to `stack_slot_color_map`;
    /// portal-bridge installs (`canonical_bridge::install_portal_for`)
    /// populate it as identity over the same range.
    ///
    /// Length invariant: `pyre_color_for_semantic_local.len() == nlocals`,
    /// matching the locals prefix of the runtime PyFrame allocation.
    ///
    /// `enforce_input_args` (`flatten.py:88-100` parity) renumbers only the
    /// startblock inputargs (function args) into the dense `0..N` prefix per
    /// kind class — it does NOT pin body locals to identity, so post-#347/#348
    /// this map is no longer `[0, 1, ..., nlocals-1]`: a body local carries
    /// whatever color `apply_rename` produced. At a resume pc the freely-colored
    /// locals are recovered through the per-PC `pcdep_color_slots` map; this
    /// flat map is only the fallback base (matching the sibling
    /// `stack_slot_color_map` caveat above). `get_list_of_active_boxes` derives
    /// the semantic index from the color via this map for locals and
    /// `stack_slot_color_map` for stack slots.
    pub pyre_color_for_semantic_local: Vec<u16>,
    /// #348 Part (2): per-Python-PC color↔slot map for the live restorable
    /// Ref frame slots, the PC-dependent successor to the flat
    /// `pyre_color_for_semantic_local` / `stack_slot_color_map` inversion.
    /// Indexed by `py_pc`; each entry is the list of `(color, semantic_slot)`
    /// for the slots live and restorable at that PC, sorted by color. The
    /// semantic slot is the unified `locals_cells_stack_w` index (local `i`
    /// for `i < nlocals`, `nlocals + d` for operand-stack depth `d`).
    ///
    /// The flat maps record ONE color per slot for the whole jitcode (forced
    /// by the same-slot coalescing); this records each slot's TRUE per-program
    /// -point SSA color, the runtime analog of RPython's compile-time baked
    /// register operands. Empty when the producer did not populate it
    /// (portal-bridge identity installs, skeletons, or the flat-map path):
    /// runtime decoders fall back to the flat maps on an empty entry. When
    /// populated, the `-live-` markers carry the SAME per-PC colors (built by
    /// `filter_liveness_in_place` off this map), so encode/decode/`-live-`
    /// stay in one consistent color space.
    pub pcdep_color_slots: Vec<Vec<(u16, u16)>>,
    /// Per-Python-PC operand-stack Ref CONSTANTS (`(semantic_slot, raw_ref)`).
    /// `pcdep_color_slots` records live restorable Variables only; for the
    /// virtualizable ROOT frame the operand-stack constants are rematerialized
    /// from the value-stack resumedata's const pool, but an INLINED CALLEE
    /// frame has no virtualizable payload. `reconstruct_inline_recipe` reads
    /// this table to refill the registerless constant slots a guard resume
    /// leaves empty after the `pcdep_color_slots` color→slot inversion.
    /// Indexed by `py_pc`; empty for jitcodes with no inlined-callee resume.
    pub const_ref_slots_at_pc: Vec<Vec<(u16, i64)>>,
}

/// Compiled JitCode plus pyre-only metadata.
pub struct PyJitCodePayload {
    pub jitcode: std::sync::Arc<RuntimeJitCode>,
    pub metadata: PyJitCodeMetadata,
    /// pyre's graph identity for the cached jitcode slot.
    ///
    /// RPython indexes `CallControl.jitcodes` and `unfinished_graphs`
    /// directly by graph object. pyre still keys the public cache by
    /// `w_code` when available, but the cached object carries the raw
    /// CodeObject pointer so the queue can stay a bare graph list.
    pub code_ptr: *const pyre_interpreter::CodeObject,
    /// True if the jitcode contains an `abort` or `abort_permanent` opcode
    /// (unsupported bytecodes / emission-time bail-outs). Precomputed at
    /// compile time to avoid repeated bytecode scanning.
    pub has_abort: bool,
    /// Lazily-built per-fn walker descr pool
    /// (`state::sub_jitcode_descr_pool_for_code`): adapted `descr_refs`,
    /// raw `RuntimeBhDescr` slice, sub-jitcode lookup. Carried on the
    /// payload — not in a side table keyed by object identity — so a
    /// `replace_with` body refill drops the pool together with the
    /// `exec.descrs` it borrows from. RPython has no equivalent table:
    /// the active MIFrame's JitCode carries its descriptors directly.
    pub(crate) sub_descr_pool: std::cell::OnceCell<crate::state::SubDescrPool>,
}

/// Shared `PyJitCode` identity whose payload is filled in place.
///
/// Held by `Arc` so the same instance can sit in both
/// `MetaInterpStaticData.jitcodes` (the runtime list) and
/// `CallControl.jitcodes` (the codewriter dict) without duplicating
/// the bytecode buffer or metadata vectors. RPython's
/// `JitCode` references are shared the same way through Python's
/// refcount semantics.
///
/// RPython mutates the `JitCode` shell inserted by `call.py:168-170`
/// when `assembler.assemble(..., jitcode, ...)` runs in
/// `codewriter.py:67`. Pyre's assembler still returns a fresh payload,
/// so the outer `PyJitCode` uses interior mutability to preserve the
/// same object identity while filling the payload during the writer
/// drain. The inner runtime `JitCode` allocation is filled in place as
/// well, so any caller-side `RuntimeBhDescr::JitCode(Arc<JitCode>)`
/// created by a future orthodox `inline_call_*` rewrite keeps pointing
/// at the populated callee object after the drain.
///
/// Production mutation is confined to the single-threaded codewriter
/// publication path before runtime readers observe the populated object.
pub struct PyJitCode {
    payload: UnsafeCell<PyJitCodePayload>,
}

// SAFETY: `PyJitCode` payload replacement is restricted to the codewriter
// publication path, which runs under pyre's single-threaded JIT setup before
// the populated object is handed to runtime readers. Runtime-visible index
// stamping uses atomics on the inner `RuntimeJitCode`. The lazily-built
// `sub_descr_pool` `OnceCell` is initialized only from the trace-side walker
// (`state::sub_jitcode_descr_pool_for_code`), which runs on the single
// thread owning the thread-local `METAINTERP_SD` store.
unsafe impl Sync for PyJitCode {}

impl Deref for PyJitCode {
    type Target = PyJitCodePayload;

    fn deref(&self) -> &Self::Target {
        // SAFETY: shared readers only observe immutable payload references.
        unsafe { &*self.payload.get() }
    }
}

impl DerefMut for PyJitCode {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.payload.get_mut()
    }
}

/// `interp_jit.py:67 reds = ['frame', 'ec']` pre-regalloc slot derivation.
///
/// Returns `(portal_frame_reg, portal_ec_reg)` — the SSARepr Variable
/// indices the per-CodeObject codewriter emits for the portal red args.
/// The slot positions are `(nlocals + max_stackdepth + 11)` and
/// `(nlocals + max_stackdepth + 12)` respectively; slot `+10` was the
/// dedicated `null_ref_reg` PY_NULL holder before it was retired
/// it, and the portal reds kept their numerical positions so
/// layout-sensitive tests stay stable.
///
/// PyPy structural counterpart: `pypy/module/pypyjit/interp_jit.py:67
/// reds = ['frame', 'ec']` is the JitDriver declaration; `warmspot.py`
/// derives per-driver red arg slot indices from this list when wiring
/// up the trace's inputargs.  Pyre's codewriter pipeline lacks the
/// JitDriver greens/reds → register-slot derivation that
/// `warmspot.setup_jit` runs, so the slot positions are encoded here
/// as a shared formula instead — both `pyre-jit/src/jit/codewriter.rs`
/// `MetainterpCodeWriter::compile` and `canonical_bridge.rs`
/// `install_portal_for` route through this helper so they cannot drift.
///
#[inline]
pub fn portal_red_pre_regalloc_slots(nlocals: usize, max_stackdepth: usize) -> (u16, u16) {
    let portal_frame_reg = (nlocals + max_stackdepth + 11) as u16;
    let portal_ec_reg = (nlocals + max_stackdepth + 12) as u16;
    (portal_frame_reg, portal_ec_reg)
}

/// `#124` Approach B master switch (`PYRE_M3_JITCODE_PC`, default off).
///
/// When enabled, a kept-stack branch guard resumes at its carried direct
/// JitCode pc and the encoder collects the guard-pc live box set directly
/// rather than the resume-pc set patched by the positional kept-stack
/// heuristic.  Default off: for a depth-1 leading-`and` short-circuit
/// (`x = local and CONST`) the guard-pc box set holds the *taken*-path value
/// for the kept operand-stack slot, so the direct-pc path restores that value
/// on the not-taken arm and silently miscompiles (`flag and 11`: 2197063 vs
/// 1466663).  The positional heuristic recovers the not-taken kept value
/// correctly — depth-1 via `kept_stack_subst` and depth > 1 via the not-taken
/// edge's decoded `ref_copy` parallel moves (`#420`,
/// `jitcode_dispatch.rs decode_branch_trampoline_ref_moves`), the authoritative
/// kept-value source while M3 is off.  Set `PYRE_M3_JITCODE_PC=1` to opt back
/// into the direct-pc path for validating the future snapshot-before-guard fix
/// (#124/#281).
pub(crate) fn m3_jitcode_pc_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var_os("PYRE_M3_JITCODE_PC") {
        Some(v) => {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        }
        None => false,
    })
}

impl PyJitCode {
    pub fn new(payload: PyJitCodePayload) -> Self {
        Self {
            payload: UnsafeCell::new(payload),
        }
    }

    pub fn from_parts(
        jitcode: std::sync::Arc<RuntimeJitCode>,
        metadata: PyJitCodeMetadata,
        code_ptr: *const pyre_interpreter::CodeObject,
        has_abort: bool,
    ) -> Self {
        Self::new(PyJitCodePayload {
            jitcode,
            metadata,
            code_ptr,
            has_abort,
            sub_descr_pool: std::cell::OnceCell::new(),
        })
    }

    /// Fill the cached payload without changing the outer `PyJitCode`
    /// allocation or the inner runtime `JitCode` allocation, even if
    /// setup-time call descriptors have already cloned the inner
    /// `Arc<JitCode>` shell. This is pyre's Rust-side stand-in for RPython
    /// `assembler.assemble(..., jitcode, ...)` mutating the existing
    /// `JitCode` object from `CallControl.jitcodes[graph]`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee no runtime thread is currently reading
    /// the payload through any cloned `Arc<PyJitCode>` or cloned inner
    /// `Arc<JitCode>`. RPython relies
    /// on the implicit single-threaded semantics of the translation /
    /// codewriter setup phase — the JitCode shell is filled in place
    /// before any runtime reader observes it. Pyre cannot encode that
    /// invariant in the Rust type system without a heavyweight lock,
    /// so callers must check the precondition manually:
    ///
    /// * Only invoke this from the JIT setup / codewriter publication
    ///   path, before runtime tracing or blackhole resume can dispatch
    ///   on the same code.
    /// * In particular, do NOT call this to roll a populated payload
    ///   back to a skeleton — that breaks the "runtime reader never
    ///   observes a reset shell" invariant. Skeleton resets must
    ///   replace the outer `Arc` instead (see
    ///   `CallControl::reset_jitcode_skeleton`).
    pub unsafe fn replace_with(&self, next: PyJitCode) {
        let PyJitCodePayload {
            jitcode: next_jitcode,
            metadata,
            code_ptr,
            has_abort,
            sub_descr_pool,
        } = next.payload.into_inner();
        let next_jitcode = std::sync::Arc::try_unwrap(next_jitcode)
            .expect("freshly assembled PyJitCode must uniquely own its runtime JitCode");
        unsafe {
            let current = &mut *self.payload.get();
            // RPython's call descriptors keep the callee JitCode object itself.
            // During setup, an inline_call descr may therefore already point at
            // this shell before assembler.assemble() fills it. Rust's Arc cannot
            // express "shared for setup identity, exclusively mutated before
            // runtime publication", so we write through the stable allocation
            // under the setup-phase precondition documented above.
            // Drop any pool built against the body being replaced BEFORE
            // overwriting the inner JitCode: the pool borrows that body's
            // `exec.descrs`, and the new body starts with no pool (built
            // lazily on first walker inline of the refilled callee).
            current.sub_descr_pool = sub_descr_pool;
            let current_jitcode = std::sync::Arc::as_ptr(&current.jitcode) as *mut RuntimeJitCode;
            *current_jitcode = next_jitcode;
            current.metadata = metadata;
            current.code_ptr = code_ptr;
            current.has_abort = has_abort;
        }
    }

    /// Check if this jitcode has BC_ABORT opcodes.
    pub fn has_abort_opcode(&self) -> bool {
        self.has_abort
    }

    /// "Has `assembler.assemble` been run on this jitcode yet?" A
    /// freshly-constructed RPython `JitCode(name, fnaddr, calldescr,
    /// ...)` (jitcode.py:14, call.py:168) leaves `self.code` unset
    /// until `setup` (jitcode.py:22) is invoked by
    /// `assembler.assemble(ssarepr, jitcode, num_regs)`
    /// (codewriter.py:67); pyre's split wrapper uses `pc_map.is_empty()`
    /// as the same "still a shell" test.
    ///
    /// PerCodeObject mode in the discriminator table on the module
    /// doc.
    pub fn is_populated(&self) -> bool {
        !self.metadata.pc_map.is_empty()
    }

    /// Resolve a Python bytecode PC to the JitCode byte offset where
    /// blackhole resume / inline call tracing should restart execution.
    /// Returns `None` if `py_pc` falls outside the populated range
    /// (portal-bridge installs always return `None` because their
    /// `pc_map` is empty by construction).
    ///
    /// This is pyre's analog of `blackhole.py:1712 self.setposition(
    /// miframe.jitcode, miframe.pc)` where upstream stores the JitCode
    /// PC directly in resume data (`miframe.pc`); pyre's resume data
    /// stores the Python bytecode PC and translates here.  Centralizing
    /// the lookup makes the resume-data write-side an obvious migration
    /// target: once resume data stores `jitcode_pc` directly the
    /// translation step (and the `pc_map` it depends on) can retire.
    pub fn resume_jitcode_pc_for(&self, py_pc: usize) -> Option<usize> {
        self.metadata.pc_map.get(py_pc).copied()
    }

    /// JitCode byte offset of `py_pc`'s post-`residual_call` `-live-`
    /// (the marker preceding the opcode's own `catch_exception`), or
    /// `None` if `py_pc` makes no residual call.  After-residual-call
    /// guard resume (`blackhole.py:396-410 handle_exception_in_frame`)
    /// uses this instead of [`Self::resume_jitcode_pc_for`] so it lands
    /// on the call's own catch rather than the next opcode's start
    /// marker (`pc_map[next_pc]`).
    pub fn after_residual_call_resume_pc_for(&self, py_pc: usize) -> Option<usize> {
        self.metadata
            .after_residual_call_resume_pc
            .get(py_pc)
            .copied()
            .flatten()
    }

    /// Translate a resume-data pc word (as carried in rd_numb / RebuiltFrame)
    /// to a JitCode byte offset, honoring the after-residual-call marker:
    /// flagged words route through [`Self::after_residual_call_resume_pc_for`],
    /// plain words through [`Self::resume_jitcode_pc_for`].  Every decode-side
    /// py_pc→jitcode translation funnels through here so the marker is
    /// interpreted consistently.
    pub fn resolve_resume_pc(&self, raw_pc: i32) -> Option<usize> {
        let (py_pc, after_residual_call) = majit_ir::resumedata::decode_resume_pc(raw_pc);
        if py_pc < 0 {
            return None;
        }
        if after_residual_call {
            self.after_residual_call_resume_pc_for(py_pc as usize)
        } else {
            self.resume_jitcode_pc_for(py_pc as usize)
        }
    }

    /// `#124` Approach B resolver: translate a guard frame's resume
    /// coordinate, preferring the carried direct JitCode pc (`carried`,
    /// the rd_numb per-frame `jitcode_pc` word populated by M2) over the
    /// lossy `pc_map` translation of the stored Python pc.
    ///
    /// `resolve_resume_pc(raw_pc)` routes the Python pc through `pc_map`,
    /// which collapses every kept-operand-stack-across-branch state at one
    /// Python pc to a single JitCode offset — the precision loss `#124`
    /// fixes.  When `carried` names a valid startpoint, it IS the
    /// kept-stack-precise coordinate `setposition(jitcode, miframe.pc)`
    /// preserves upstream, so it is returned directly.
    ///
    /// The encoder (box collection) and decoder (box count + setposition)
    /// both funnel through this with identical `(raw_pc, carried, op_live)`,
    /// so the chosen offset — and hence the live-box layout — is symmetric
    /// by construction.  Gated by [`m3_jitcode_pc_enabled`]: with the flag
    /// off the carried word is ignored and behaviour is identical to
    /// [`Self::resolve_resume_pc`].
    ///
    /// The carried word is preferred only when it is a `-live-`-anchored
    /// coordinate ([`JitCode::can_decode_live_vars`]); a startpoint that is
    /// not so anchored (a synthesized specialization guard's `may_force`
    /// CALL op) falls through to the `pc_map` translation so
    /// `get_live_vars_info` never hits `_missing_liveness`.
    pub fn resolve_resume_pc_with_jitcode_pc(
        &self,
        raw_pc: i32,
        carried: i32,
        op_live: u8,
    ) -> Option<usize> {
        if m3_jitcode_pc_enabled() && carried != majit_ir::resumedata::NO_JITCODE_PC && carried >= 0
        {
            let jp = carried as usize;
            if self.jitcode.can_decode_live_vars(jp, op_live) {
                return Some(jp);
            }
        }
        self.resolve_resume_pc(raw_pc)
    }

    /// Skeleton slot inserted by [`Self::skeleton`] — neither `code`
    /// nor `pc_map` populated yet. See the discriminator table on
    /// the module doc.
    ///
    /// Strictly equivalent to `!is_populated() && !is_portal_bridge()`
    /// (DeMorgan-expanded: `pc_map.is_empty() && (code.is_empty() ||
    /// !pc_map.is_empty())` reduces to the conjunction below).
    /// Callers prefer this name over the negated-pair form because it
    /// names the third mode in the discriminator table directly.
    pub fn is_skeleton(&self) -> bool {
        self.jitcode.code.is_empty() && self.metadata.pc_map.is_empty()
    }

    /// Is this `PyJitCode` a portal-bridged install (G.3a
    /// `canonical_bridge::install_portal_for`)?
    ///
    /// Discriminator:
    ///   * `jitcode.code` non-empty (rules out `PyJitCode::skeleton`,
    ///     which clones `Arc::new(RuntimeJitCode::default())` whose
    ///     `code` is empty).
    ///   * `metadata.pc_map` empty (rules out drained CodeWriter
    ///     installs, whose setup-time drain populates `pc_map` to
    ///     `code.instructions.len()`).
    ///
    /// Used by readers that have to branch on portal-mode semantics —
    /// portal entry has no per-Python-PC `pc_map` because the portal
    /// jitcode dispatches on `pycode.instructions[pc]` at runtime via
    /// its own dispatch arms.  See
    /// `canonical_bridge::install_portal_for` for the full reader
    /// audit (G.3a).
    ///
    /// G.3b landed this discriminator for reader-audit probes. The
    /// orthodox redirect path now avoids binding portal-bridge payloads as
    /// `jd.mainjitcode`; production readers still branch on this predicate
    /// only for explicit bridge-probe installs.
    pub fn is_portal_bridge(&self) -> bool {
        !self.jitcode.code.is_empty() && self.metadata.pc_map.is_empty()
    }

    /// Empty `PyJitCode` slot inserted by `CallControl::get_jitcode`
    /// (call.py:168 `jitcode = JitCode(graph.name, fnaddr, calldescr, ...)`).
    ///
    /// In RPython the `JitCode` constructor returns a fresh object whose
    /// `code` / `descrs` / `liveness` arrays are all empty until
    /// `assembler.assemble(...)` populates them later in
    /// `make_jitcodes`'s drain loop (codewriter.py:80).  The skeleton
    /// gives the dict an entry with a stable identity so re-entrant
    /// `get_jitcode` calls can find an existing key without recompiling
    /// (call.py:155 `if graph in self.jitcodes: return`).
    pub fn skeleton(code_ptr: *const pyre_interpreter::CodeObject) -> Self {
        Self::from_parts(
            std::sync::Arc::new(RuntimeJitCode::default()),
            PyJitCodeMetadata {
                pc_map: Vec::new(),
                after_residual_call_resume_pc: Vec::new(),
                first_jit_pc_by_py_pc: Vec::new(),
                depth_at_py_pc: Vec::new(),
                // u16::MAX sentinel mirrors `canonical_bridge::install_portal_for`
                // (canonical_bridge.rs:165-166). Encoder/decoder readers in
                // `get_list_of_active_boxes`, `regalloc::external/input_indices`,
                // and `setup_bridge_sym::portal_red_regs_at` sentinel-skip both
                // values together. A real `0` here would alias every locals-
                // bank color 0 read and silently substitute `sym.frame` for
                // unrelated locals/stack slots.
                portal_frame_reg: u16::MAX,
                portal_ec_reg: u16::MAX,
                built_as_portal: false,
                stack_base: 0,
                stack_slot_color_map: Vec::new(),
                pyre_color_for_semantic_local: Vec::new(),
                pcdep_color_slots: Vec::new(),
                const_ref_slots_at_pc: Vec::new(),
            },
            code_ptr,
            false,
        )
    }
}
