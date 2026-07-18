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
//!     pyre-only translation metadata — the per-Python-PC resume tables
//!     (Python PC → byte offset), the runtime `w_code` wrapper, and
//!     register layout — that RPython does not need because RPython's
//!     bytecode PCs are already JitCode PCs.
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
//! ## Discriminator: two-state mode mapping
//!
//! A produced `PyJitCode` is one of two modes:
//!
//! | mode          | `jitcode.code` | `metadata.is_drained` | predicate                  |
//! |---------------|----------------|-----------------------|----------------------------|
//! | Skeleton      | empty          | false                 | [`PyJitCode::is_skeleton`] |
//! | PerCodeObject | non-empty      | true                  | [`PyJitCode::is_populated`] |
//!
//! `is_drained` tracks the setup-time drain (`codewriter.rs` `finalize_jitcode`
//! populates the per-PC maps); it is the sole mode-classification flag, so the
//! mode is independent of the translation tables.
//!
//! Drained CodeWriter installs fill real instructions into `code` and populate
//! the per-Python-PC tables. Skeletons have neither because they are placeholder slots inserted by
//! `CallControl::get_jitcode` before the assembler drain runs.
//!
//! Convergence path: RPython's single `JitCode` class has neither flag
//! to consult — `assembler.assemble` populates `code` in place and
//! per-PC mapping is implicit in the bytecode stream.

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
    /// Post-residual-call catch resume twin, split into the exact
    /// block-head-marker and predecessor op-start tiers used by
    /// `python_pc_for_jitcode_pc`. Empty for skeleton / fixture metadata.
    pub after_residual_call_resume_marker_by_jit_pc: Vec<(usize, Option<usize>)>,
    pub after_residual_call_resume_pred_by_jit_pc: Vec<(usize, Option<usize>)>,
    /// Number of Python instructions in the source CodeObject. Empty skeleton
    /// and fixture metadata carry zero.
    pub n_py_instrs: u32,
    /// Inverse of the derived marker resolution's block-head case: each distinct
    /// `-live-` marker byte offset that some PC resolves to → the first Python PC
    /// that resolves to it. Portal JitCodes emit live frame/global reads at
    /// control-flow entries, so the dense-map owner is the block entry.
    /// Sorted ascending by jitcode offset for binary search.  Replaces the
    /// former dense-map block-head scan in
    /// `python_pc_for_jitcode_pc` (a coordinate landing exactly on a marker is
    /// a block head — branch/catch target — and belongs to the first opcode
    /// resuming there). Empty for skeleton / fixture metadata.
    pub block_head_py_by_jit_pc: Vec<(usize, u32)>,
    /// JitCode byte-offset → containing Python PC floor boundary table. Each
    /// entry starts a piecewise-constant opcode-emission segment; the first
    /// entry is always `(0, 0)` for a drained install. This is only the
    /// predecessor op-start tier: exact block-head marker precedence remains in
    /// `block_head_py_by_jit_pc`. Empty for skeleton / fixture metadata.
    pub py_floor_by_jit_pc: Vec<(u32, u32)>,
    /// Trace-entry green py_pc → JitCode byte offset where tracing enters
    /// for that green. This is the restriction of resume-marker resolution to
    /// function entry and loop headers, built at codewrite time, not a general
    /// py_pc → jitcode inverse. At portal entry `pyjitpl.py:1567-1573` sets
    /// the live frame PC to the `jit_merge_point` origin before handling the
    /// loop header; pyre's whole-function JitCode has one corresponding entry
    /// per materialized trace-entry green. Sorted ascending by green py_pc for
    /// binary search; empty for skeleton / fixture metadata.
    pub merge_entry_by_green: Vec<(u32, u32)>,
    /// task#50 phase-1: predecessor-keyed jitcode-pc twin of `pcdep_color_slots`.
    /// Each entry `(off, colors)` maps a JitCode byte offset to the pcdep
    /// color→slot list of the py_pc that `python_pc_for_jitcode_pc(off)` returns
    /// (block-head marker precedence, else the predecessor op-start boundary).
    /// A PREDECESSOR binary search (largest offset ≤ jit_pc)
    /// then reproduces `pcdep_color_slots[python_pc_for_jitcode_pc(jit_pc)]` for
    /// the carried resume coordinates reaching the decode re-inversion at
    /// `bridge_semantic_maps_at_with_jitcode_pc`. Both sides are compile-time
    /// derivations of the same resume-marker / `first_jit` coordinates, so the
    /// equality holds by construction. Sorted ascending by offset; empty for
    /// skeleton / fixture.
    pub pcdep_by_jit_pc: Vec<(usize, Vec<(u8, u16, u16)>)>,
    /// task#50 phase-1: predecessor-keyed jitcode-pc twin of `depth_at_py_pc`,
    /// built alongside `pcdep_by_jit_pc` with the same `python_pc_for_jitcode_pc`
    /// resolution (marker precedence + op-start predecessor). Predecessor-covers
    /// op offsets, so it agrees with the depth read at the decode seam for every
    /// carried coordinate: it equals
    /// `depth_at_py_pc[python_pc_for_jitcode_pc(jit_pc)]` by construction.
    pub depth_pred_by_jit_pc: Vec<(usize, u16)>,
    /// task#50 #73-core: trivia-aware STATIC-liveness depth twin, split into the
    /// SAME two tiers as `python_pc_for_jitcode_pc` — an EXACT-match marker table
    /// (`depth_trivia_marker_by_jit_pc`, block-head precedence) and a PREDECESSOR
    /// op-start table (`depth_trivia_pred_by_jit_pc`, markers EXCLUDED). Each
    /// records `liveness_for(code).depth_at_py_pc()[skip_python_trivia_forward(py)]`
    /// for its resolving py — the value the ENCODE branch-resume depth readers
    /// compute AFTER their forward trivia-skip. A single merged predecessor table
    /// would mis-resolve an interior not-taken coordinate onto a marker byte that
    /// sits inside a preceding op's region, so the tiers stay separate. Empty for
    /// skeleton / fixture.
    pub depth_trivia_marker_by_jit_pc: Vec<(usize, Option<u16>)>,
    pub depth_trivia_pred_by_jit_pc: Vec<(usize, Option<u16>)>,
    /// Reproduces `pcdep_color_slots[skip_python_trivia_forward(
    /// python_pc_for_jitcode_pc(jit_pc))]`, resolved with the same exact-marker
    /// and predecessor-op-start tiers as the trivia-aware depth twin.
    pub pcdep_trivia_marker_by_jit_pc: Vec<(usize, Vec<(u8, u16, u16)>)>,
    /// Predecessor-op-start tier of the trivia-aware `pcdep_color_slots` twin.
    pub pcdep_trivia_pred_by_jit_pc: Vec<(usize, Vec<(u8, u16, u16)>)>,
    /// Const Ref operand-stack slots after Python trivia, resolved with the
    /// same exact-marker and predecessor-op-start tiers as the trivia-aware
    /// depth twin.
    pub const_ref_trivia_marker_by_jit_pc: Vec<(usize, Vec<(u16, i64)>)>,
    /// Predecessor-op-start tier of the trivia-aware const Ref slot twin.
    pub const_ref_trivia_pred_by_jit_pc: Vec<(usize, Vec<(u16, i64)>)>,
    /// #73-core: trivia-aware twin of `result_color_at_pc`, split into the
    /// SAME exact-marker / predecessor-op-start tiers as the depth-trivia
    /// twin. Each entry records
    /// `result_color_at_pc[skip_python_trivia_forward(py)]`, including an
    /// out-of-range trailing-trivia overshoot as `None`.
    pub result_color_trivia_marker_by_jit_pc: Vec<(usize, Option<u16>)>,
    pub result_color_trivia_pred_by_jit_pc: Vec<(usize, Option<u16>)>,
    /// task#73 S5 phase-0: resume-marker twin split into the SAME two tiers as
    /// `python_pc_for_jitcode_pc` — an EXACT-match marker table
    /// (`resume_marker_marker_by_jit_pc`, block-head precedence) and a
    /// PREDECESSOR op-start table (`resume_marker_pred_by_jit_pc`, markers
    /// EXCLUDED). Each records the block-head `-live-` resume-marker JitCode byte
    /// offset for its resolving py. A single merged predecessor table would mis-resolve an interior
    /// coordinate onto a marker byte inside a preceding op's emitted region, so
    /// the tiers stay separate. Empty for skeleton / fixture.
    pub resume_marker_marker_by_jit_pc: Vec<(usize, Option<usize>)>,
    pub resume_marker_pred_by_jit_pc: Vec<(usize, Option<usize>)>,
    /// task#73 S5 phase-2: after-residual fallthrough-marker twin with the
    /// same exact-marker / predecessor-op-start split as the resume-marker
    /// twin. Each value additionally applies the tracer's semantic
    /// fallthrough rule after skipping Python trivia. Empty for skeleton /
    /// fixture.
    pub after_residual_marker_marker_by_jit_pc: Vec<(usize, Option<usize>)>,
    pub after_residual_marker_pred_by_jit_pc: Vec<(usize, Option<usize>)>,
    /// Result color of the call-result operand slot at the after-residual
    /// fallthrough PC, keyed by JitCode byte offset with the same exact-marker
    /// / predecessor-op-start split as `after_residual_marker_*`. Value =
    /// `result_color_at_pc[semantic_fallthrough_pc(skip_python_trivia_forward(py))]`
    /// computed at codewrite time (`None` past the table end). Retires the
    /// runtime py-keyed result-color read in the inline-caller marker-miss
    /// fallback. Empty for skeleton / fixture.
    pub result_color_after_residual_marker_by_jit_pc: Vec<(usize, Option<u16>)>,
    pub result_color_after_residual_pred_by_jit_pc: Vec<(usize, Option<u16>)>,
    /// Whether codewriter register allocation assigned non-identity frame
    /// colors. Skeleton and portal metadata leave this false.
    pub has_color_map: bool,
    /// Post-regalloc Ref-bank color of the portal jitdriver's first red
    /// argument (`frame`).  RPython parity: `pypy/module/pypyjit/
    /// interp_jit.py:67 reds = ['frame', 'ec']` declares the portal
    /// calling convention, and `JitDriverStaticData.red_args_indices`
    /// (`rpython/jit/metainterp/warmspot.py`) records the inputarg
    /// position of each red arg.  This field is the pyre equivalent —
    /// the snapshot serializer at
    /// `pyre-jit-trace::trace_opcode::get_list_of_active_boxes` uses it
    /// to map the live_r color back to the symbolic `sym.frame` OpRef.
    /// `u16::MAX` for skeletons that have not run per-CodeObject regalloc.
    pub portal_frame_reg: u16,
    /// Post-regalloc Ref-bank color of the portal jitdriver's second red
    /// argument (`ec`, `pypy/module/pypyjit/interp_jit.py:67`).
    /// Snapshot serializer maps this color to `sym.execution_context`.
    pub portal_ec_reg: u16,
    /// Whether the body carries the PORTAL entry INPUT SHAPE
    /// (`FrameInputs::Portal`: `[frame, ec]` red inputs + frame-vable locals
    /// prologue) — the invariant for every drained per-code jitcode, so a
    /// later portal walk of any body is admitted.
    /// This records the input shape, NOT true-portal-ness (the
    /// `jit_merge_point` marker stays gated separately on
    /// `jitdriver_sd_from_portal_graph`).  `false` only for shapeless
    /// skeletons (`PyJitCodeMetadata::skeleton`), which `run_perfn_walk`
    /// declines to walk.
    pub built_as_portal: bool,
    /// Absolute start index of the operand stack in PyFrame.locals_cells_stack_w.
    pub stack_base: usize,
    /// Maximum operand-stack depth (`code.max_stackdepth` = CPython
    /// `co_stacksize`) for a compiled jitcode, `0` for non-compiled skeleton
    /// metadata. Carries the operand-stack dimension so the bridge
    /// frame-array sizing (`state.rs::setup_bridge_sym`) reconstructs the
    /// full runtime PyFrame allocation (`pyframe.rs:1576`
    /// `nlocals + ncells + max_stackdepth`). Sized to the static peak, not
    /// `max(depth_at_pc)` — JIT-traced PCs may not reach `co_stacksize`.
    pub max_stackdepth: usize,
    /// gh#73 S3.2: predecessor-keyed jitcode-pc const Ref operand-stack slots.
    /// Each entry `(off, slots)` maps a JitCode byte offset to the const
    /// operand-stack slot list of the py_pc that `python_pc_for_jitcode_pc(off)`
    /// returns (block-head marker precedence, else the predecessor op-start
    /// boundary at-or-before `off`). A PREDECESSOR binary search (largest
    /// offset ≤ jit_pc) returns the slots for a carried resume coordinate.
    /// Built in the same `by_off` loop as `pcdep_by_jit_pc`; empty for skeleton
    /// / fixture. Read on the decode-identity path of
    /// `const_ref_slots_at_pc_at`, mirroring the `pcdep_by_jit_pc` /
    /// `depth_pred_by_jit_pc` twins' production use.
    pub const_ref_slots_by_jit_pc: Vec<(usize, Vec<(u16, i64)>)>,
    /// True once `assembler.assemble`'s setup-time drain has run and stamped
    /// the per-Python-PC maps (`codewriter.rs` `finalize_jitcode`). The
    /// install-mode discriminator [`PyJitCode::is_populated`] reads this flag instead of testing
    /// the now-deleted dense translation table's population state, so the
    /// mode classification no longer depends on it. Set to `true` exactly
    /// where the drain populates the per-Python-PC maps (drained
    /// PerCodeObject installs); `false` for skeletons, which leave those maps empty.
    pub is_drained: bool,
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
/// as a shared formula used by `pyre-jit/src/jit/codewriter.rs`
/// `MetainterpCodeWriter::compile`.
///
#[inline]
pub fn portal_red_pre_regalloc_slots(nlocals: usize, max_stackdepth: usize) -> (u16, u16) {
    let portal_frame_reg = (nlocals + max_stackdepth + 11) as u16;
    let portal_ec_reg = (nlocals + max_stackdepth + 12) as u16;
    (portal_frame_reg, portal_ec_reg)
}

/// task#50 deletion-precondition: derive `pc_map[py_pc]` — the `-live-` resume
/// marker byte offset — on-demand from the two surviving per-Python-PC tables,
/// WITHOUT the retired dense `pc_map` Vec.
///
/// Two tiers reproduce the carry-forward rules that
/// `derive_pc_live_indices_from_sparse` (codewriter) baked into the dense map:
///
/// * A py_pc that emitted its own first op (`first_op != usize::MAX`) resolves
///   to the largest block-head marker at-or-before that first offset — no
///   marker lies strictly between a block head and the first op emitted inside
///   the block, so this equals the dense value.
/// * A py_pc that emitted no op of its own (`usize::MAX`: trivia / folded /
///   unconditional jump) inherits the resume marker the codewriter carried
///   forward to the NEXT real PC in the block (the trivia / next-op
///   forward-carry): scan ascending to the first py that emitted an op and
///   resolve THAT. This reproduces the common trivia-forward-carry; the
///   genuinely non-invertible residual (uncond-jump forward-carry to a
///   backward/forward jump TARGET, can-raise / branch re-keys keyed off the
///   stream position) diverges here and is captured in a build-time
///   carry-forward sidecar instead.
///
/// `None` when the tables are empty or
/// no at-or-after py emitted a real op.
pub fn derive_resume_marker(
    first_op_by_py_pc: &[usize],
    block_head_py_by_jit_pc: &[(usize, u32)],
    py_pc: usize,
) -> Option<usize> {
    if block_head_py_by_jit_pc.is_empty() {
        return None;
    }
    // Real-op offset for `py_pc`, else the first at-or-after py that emitted an
    // op (trivia / next-op forward-carry).
    let first_val = first_op_by_py_pc
        .get(py_pc..)?
        .iter()
        .copied()
        .find(|&v| v != usize::MAX)?;
    let search = block_head_py_by_jit_pc.binary_search_by_key(&first_val, |&(off, _)| off);
    let idx = match search {
        Ok(i) => i,
        Err(0) => return None,
        Err(i) => i - 1,
    };
    Some(block_head_py_by_jit_pc[idx].0)
}

/// Return the floor segment containing `jit_pc` in a codewriter-built JitCode
/// PC pivot. An empty table is deliberately distinguishable from the `(0, 0)`
/// fallback segment carried by every drained install.
pub fn floor_segment_for_jitcode_pc(
    py_floor_by_jit_pc: &[(u32, u32)],
    jit_pc: usize,
) -> Option<(usize, u32)> {
    let end = py_floor_by_jit_pc.partition_point(|&(off, _)| (off as usize) <= jit_pc);
    end.checked_sub(1).map(|idx| {
        (
            py_floor_by_jit_pc[idx].0 as usize,
            py_floor_by_jit_pc[idx].1,
        )
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
    /// (codewriter.py:67); pyre's split wrapper uses `metadata.is_drained`
    /// (the drain that populates the per-Python-PC maps) as the same
    /// "still a shell" test.
    ///
    /// PerCodeObject mode in the discriminator table on the module
    /// doc.
    pub fn is_populated(&self) -> bool {
        self.metadata.is_drained
    }

    /// Codewrite-time trace-entry offset for a function-entry or loop-header
    /// green py_pc.
    pub fn merge_entry_for(&self, py_pc: usize) -> Option<usize> {
        let table = &self.metadata.merge_entry_by_green;
        table
            .binary_search_by_key(&(py_pc as u32), |&(py, _)| py)
            .ok()
            .map(|i| table[i].1 as usize)
    }

    /// task#50 phase-1: predecessor index into a jitcode-pc twin — the entry
    /// with the largest offset at-or-before `jit_pc`, reproducing
    /// `python_pc_for_jitcode_pc`'s marker-then-first_jit resolution baked into
    /// the twin at build time. `None` when the table is empty (skeleton /
    /// fixture) or `jit_pc` precedes the first entry.
    fn predecessor_index(search: Result<usize, usize>) -> Option<usize> {
        match search {
            Ok(i) => Some(i),
            Err(0) => None,
            Err(i) => Some(i - 1),
        }
    }

    /// task#50 phase-1: pcdep color→slot list keyed directly by a JitCode byte
    /// offset via the `pcdep_by_jit_pc` predecessor twin. Equals
    /// `pcdep_color_slots[python_pc_for_jitcode_pc(jit_pc)]` by construction for
    /// a carried resume coordinate; `None` when the twin is empty (skeleton /
    /// fixture). Scaffolding for the decode-side pc_map re-inversion retirement.
    pub fn pcdep_for_jitcode_pc(&self, jit_pc: usize) -> Option<Vec<(u8, u16, u16)>> {
        let table = &self.metadata.pcdep_by_jit_pc;
        if table.is_empty() {
            return None;
        }
        let search = table.binary_search_by_key(&jit_pc, |&(off, _)| off);
        Self::predecessor_index(search).map(|i| table[i].1.clone())
    }

    /// task#50 phase-1: value-stack depth keyed by a JitCode byte offset via the
    /// `depth_pred_by_jit_pc` predecessor twin. Equals
    /// `depth_at_py_pc[python_pc_for_jitcode_pc(jit_pc)]` by construction for a
    /// carried resume coordinate; `None` when the twin is empty. Predecessor
    /// index (largest offset ≤ jit_pc).
    pub fn depth_for_jitcode_pc_pred(&self, jit_pc: usize) -> Option<u16> {
        let table = &self.metadata.depth_pred_by_jit_pc;
        if table.is_empty() {
            return None;
        }
        let search = table.binary_search_by_key(&jit_pc, |&(off, _)| off);
        Self::predecessor_index(search).map(|i| table[i].1)
    }

    /// gh#73 S3.2: const operand-stack slots keyed by a JitCode byte offset via
    /// the `const_ref_slots_by_jit_pc` predecessor twin. Returns the slots for
    /// a carried resume coordinate; `None` when the twin is empty (skeleton /
    /// fixture). Consumed on the decode-identity path of
    /// `const_ref_slots_at_pc_at`.
    pub fn const_ref_slots_for_jitcode_pc(&self, jit_pc: usize) -> Option<Vec<(u16, i64)>> {
        let table = &self.metadata.const_ref_slots_by_jit_pc;
        if table.is_empty() {
            return None;
        }
        let search = table.binary_search_by_key(&jit_pc, |&(off, _)| off);
        Self::predecessor_index(search).map(|i| table[i].1.clone())
    }

    /// task#50 #73-core: trivia-aware STATIC-liveness depth keyed by a JitCode
    /// byte offset, resolved with the SAME two tiers as
    /// `python_pc_for_jitcode_pc`: an EXACT marker match first (block-head
    /// precedence), else a PREDECESSOR scan of the op-start table (markers
    /// excluded). Equals
    /// `liveness_for(code).depth_at_py_pc()[skip_python_trivia_forward(python_pc_for_jitcode_pc(jit_pc))]`
    /// by construction for ANY jitcode coordinate — including an interior
    /// not-taken offset the branch-resume readers query (which a single merged
    /// predecessor table mis-resolves onto an interior marker) and a
    /// trailing-trivia overshoot past the last opcode (where the raw
    /// `depth_at_py_pc().get(py)` is `None`; the twin bakes the same `None`, not a
    /// spurious `0`). `None` when the twin is empty (skeleton /
    /// fixture) — distinguish that from an in-table `None` via
    /// [`Self::depth_trivia_populated`].
    pub fn depth_trivia_for_jitcode_pc(&self, jit_pc: usize) -> Option<u16> {
        let marker = &self.metadata.depth_trivia_marker_by_jit_pc;
        let pred = &self.metadata.depth_trivia_pred_by_jit_pc;
        if marker.is_empty() && pred.is_empty() {
            return None;
        }
        if let Ok(i) = marker.binary_search_by_key(&jit_pc, |&(off, _)| off) {
            return marker[i].1;
        }
        let search = pred.binary_search_by_key(&jit_pc, |&(off, _)| off);
        Self::predecessor_index(search).and_then(|i| pred[i].1)
    }

    /// Trivia-aware pcdep color→slot list keyed by a JitCode byte offset,
    /// resolved with exact-marker precedence and an op-start predecessor scan.
    pub fn pcdep_trivia_for_jitcode_pc(&self, jit_pc: usize) -> Option<&[(u8, u16, u16)]> {
        let marker = &self.metadata.pcdep_trivia_marker_by_jit_pc;
        let pred = &self.metadata.pcdep_trivia_pred_by_jit_pc;
        if marker.is_empty() && pred.is_empty() {
            return None;
        }
        if let Ok(i) = marker.binary_search_by_key(&jit_pc, |&(off, _)| off) {
            return Some(&marker[i].1);
        }
        let search = pred.binary_search_by_key(&jit_pc, |&(off, _)| off);
        Self::predecessor_index(search).map(|i| pred[i].1.as_slice())
    }

    /// Trivia-aware const Ref slot list keyed by a JitCode byte offset,
    /// resolved with exact-marker precedence and an op-start predecessor scan.
    pub fn const_ref_trivia_for_jitcode_pc(&self, jit_pc: usize) -> Option<&[(u16, i64)]> {
        let marker = &self.metadata.const_ref_trivia_marker_by_jit_pc;
        let pred = &self.metadata.const_ref_trivia_pred_by_jit_pc;
        if marker.is_empty() && pred.is_empty() {
            return None;
        }
        if let Ok(i) = marker.binary_search_by_key(&jit_pc, |&(off, _)| off) {
            return Some(&marker[i].1);
        }
        let search = pred.binary_search_by_key(&jit_pc, |&(off, _)| off);
        Self::predecessor_index(search).map(|i| pred[i].1.as_slice())
    }

    /// #73-core: trivia-aware result color keyed by a JitCode byte offset,
    /// resolved with the SAME two tiers as `python_pc_for_jitcode_pc`.
    /// Equals
    /// `result_color_at_pc[skip_python_trivia_forward(python_pc_for_jitcode_pc(jit_pc))]`
    /// by construction, including a trailing-trivia overshoot as `None`.
    /// `None` when the twin is empty (skeleton / fixture) — distinguish that
    /// from an in-table `None` via [`Self::result_color_trivia_populated`].
    pub fn result_color_trivia_for_jitcode_pc(&self, jit_pc: usize) -> Option<u16> {
        let marker = &self.metadata.result_color_trivia_marker_by_jit_pc;
        let pred = &self.metadata.result_color_trivia_pred_by_jit_pc;
        if marker.is_empty() && pred.is_empty() {
            return None;
        }
        if let Ok(i) = marker.binary_search_by_key(&jit_pc, |&(off, _)| off) {
            return marker[i].1;
        }
        let search = pred.binary_search_by_key(&jit_pc, |&(off, _)| off);
        Self::predecessor_index(search).and_then(|i| pred[i].1)
    }

    /// task#73 S5 phase-0: codewrite-time resume marker keyed by a JitCode byte
    /// offset, resolved with the SAME two tiers as `python_pc_for_jitcode_pc`:
    /// an EXACT marker match first (block-head precedence), else a PREDECESSOR
    /// scan of the op-start table (markers excluded).
    pub fn resume_marker_for_jitcode_pc(&self, jit_pc: usize) -> Option<usize> {
        let marker = &self.metadata.resume_marker_marker_by_jit_pc;
        let pred = &self.metadata.resume_marker_pred_by_jit_pc;
        if marker.is_empty() && pred.is_empty() {
            return None;
        }
        if let Ok(i) = marker.binary_search_by_key(&jit_pc, |&(off, _)| off) {
            return marker[i].1;
        }
        let search = pred.binary_search_by_key(&jit_pc, |&(off, _)| off);
        Self::predecessor_index(search).and_then(|i| pred[i].1)
    }

    /// task#73 S5 phase-2: codewrite-time after-residual fallthrough marker
    /// keyed by a JitCode byte offset, resolved with the SAME two tiers as
    /// `python_pc_for_jitcode_pc`: an EXACT marker match first (block-head
    /// precedence), else a PREDECESSOR scan of the op-start table (markers
    /// excluded).
    pub fn after_residual_marker_for_jitcode_pc(&self, jit_pc: usize) -> Option<usize> {
        let marker = &self.metadata.after_residual_marker_marker_by_jit_pc;
        let pred = &self.metadata.after_residual_marker_pred_by_jit_pc;
        if marker.is_empty() && pred.is_empty() {
            return None;
        }
        if let Ok(i) = marker.binary_search_by_key(&jit_pc, |&(off, _)| off) {
            return marker[i].1;
        }
        let search = pred.binary_search_by_key(&jit_pc, |&(off, _)| off);
        Self::predecessor_index(search).and_then(|i| pred[i].1)
    }

    /// Result color of the call-result operand slot at the after-residual
    /// fallthrough PC, keyed by a JitCode byte offset with the SAME exact-marker
    /// / predecessor-op-start tiers as `after_residual_marker_for_jitcode_pc`.
    /// Equals the legacy
    /// `result_color_at_pc[semantic_fallthrough_pc(python_pc_for_jitcode_pc(jit_pc))]`
    /// for a call-op coordinate; `u16::MAX` where the return stack is empty;
    /// `None` when the twin is empty (skeleton / fixture) or the fallthrough is
    /// past the table end.
    pub fn result_color_after_residual_for_jitcode_pc(&self, jit_pc: usize) -> Option<u16> {
        let marker = &self.metadata.result_color_after_residual_marker_by_jit_pc;
        let pred = &self.metadata.result_color_after_residual_pred_by_jit_pc;
        if marker.is_empty() && pred.is_empty() {
            return None;
        }
        if let Ok(i) = marker.binary_search_by_key(&jit_pc, |&(off, _)| off) {
            return marker[i].1;
        }
        let search = pred.binary_search_by_key(&jit_pc, |&(off, _)| off);
        Self::predecessor_index(search).and_then(|i| pred[i].1)
    }

    /// Post-`residual_call` catch resume marker keyed by a JitCode byte
    /// offset, resolved with the SAME exact-marker / predecessor-op-start
    /// tiers as `python_pc_for_jitcode_pc`.
    pub fn after_residual_call_resume_for_jitcode_pc(&self, jit_pc: usize) -> Option<usize> {
        let marker = &self.metadata.after_residual_call_resume_marker_by_jit_pc;
        let pred = &self.metadata.after_residual_call_resume_pred_by_jit_pc;
        if marker.is_empty() && pred.is_empty() {
            return None;
        }
        if let Ok(i) = marker.binary_search_by_key(&jit_pc, |&(off, _)| off) {
            return marker[i].1;
        }
        let search = pred.binary_search_by_key(&jit_pc, |&(off, _)| off);
        Self::predecessor_index(search).and_then(|i| pred[i].1)
    }

    /// task#50 #73-core: whether the trivia depth twin carries entries. `false`
    /// for skeleton / fixture installs where both tiers are
    /// empty. The audit uses this to distinguish an in-table `None` (overshoot,
    /// which must equal the raw reader's `None`) from an empty-twin `None` (where
    /// the raw inversion still resolves a value and the twin does not apply).
    pub fn depth_trivia_populated(&self) -> bool {
        !self.metadata.depth_trivia_marker_by_jit_pc.is_empty()
            || !self.metadata.depth_trivia_pred_by_jit_pc.is_empty()
    }

    /// Whether the trivia-aware result-color twin carries entries. `false`
    /// distinguishes a skeleton / fixture from an in-table `None` caused by a
    /// trailing-trivia overshoot.
    pub fn result_color_trivia_populated(&self) -> bool {
        !self
            .metadata
            .result_color_trivia_marker_by_jit_pc
            .is_empty()
            || !self.metadata.result_color_trivia_pred_by_jit_pc.is_empty()
    }

    /// Resolve a guard frame from its carried direct JitCode coordinate.
    /// A missing or non-decodable coordinate is unrepresentable and returns
    /// `None`; callers must decline the guard/trace before publishing resume
    /// data rather than reconstructing a coordinate from a Python pc.
    ///
    /// The encoder (box collection) and decoder (box count + setposition)
    /// both funnel through this with identical `(carried, op_live)`,
    /// so the chosen offset — and hence the live-box layout — is symmetric
    /// by construction.
    ///
    /// The carried word is preferred only when it is a `-live-`-anchored
    /// coordinate ([`JitCode::can_decode_live_vars`]).
    pub fn resolve_resume_pc_with_jitcode_pc(&self, carried: i32, op_live: u8) -> Option<usize> {
        // #73 S3.5: a depth-0 branch guard may carry its `orgpc` tagged into the
        // word's negative space; expand it to the block-head marker the baseline
        // would have carried before any offset use. No-op for offsets /
        // NO_JITCODE_PC (the flip-off case), so byte-identical when off.
        let carried = crate::jitcode_dispatch::expand_branch_carried(self, carried);
        let used_carried = carried != majit_ir::resumedata::NO_JITCODE_PC
            && carried >= 0
            && self.jitcode.can_decode_live_vars(carried as usize, op_live);
        if used_carried {
            Some(carried as usize)
        } else {
            None
        }
    }

    /// Resolve a kept-stack branch guard's resume offset from its carried
    /// direct JitCode coordinate alone (gh#368: the direct coordinate is the
    /// resume key). The full-body-walk bridge entry seam only ever reaches a
    /// non-sentinel `carried` (both callers gate on `!= NO_JITCODE_PC`), and a
    /// kept-stack branch guard's carried word is always a `-live-`-anchored
    /// startpoint, so the stored Python pc is redundant here — the
    /// `resume_jitcode_pc_for` fallback [`Self::resolve_resume_pc_with_jitcode_pc`]
    /// keeps for other seams is dead at this one. `None` when the carried word
    /// does not name a decodable startpoint (the caller keeps the pc_map entry).
    pub fn resolve_bridge_walk_entry_pc(&self, carried: i32, op_live: u8) -> Option<usize> {
        let carried = crate::jitcode_dispatch::expand_branch_carried(self, carried);
        let used_carried = carried != majit_ir::resumedata::NO_JITCODE_PC
            && carried >= 0
            && self.jitcode.can_decode_live_vars(carried as usize, op_live);
        if used_carried {
            Some(carried as usize)
        } else {
            None
        }
    }

    /// Skeleton slot inserted by [`Self::skeleton`] — neither `code`
    /// nor the per-Python-PC maps populated yet. See the discriminator
    /// table on the module doc.
    ///
    /// A skeleton is the only produced mode with empty `code`; PerCodeObject
    /// fills `code`, so the empty-`code` test names the placeholder directly.
    /// Callers prefer this name over the negated-pair form.
    pub fn is_skeleton(&self) -> bool {
        self.jitcode.code.is_empty()
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
                after_residual_call_resume_marker_by_jit_pc: Vec::new(),
                after_residual_call_resume_pred_by_jit_pc: Vec::new(),
                n_py_instrs: 0,
                block_head_py_by_jit_pc: Vec::new(),
                py_floor_by_jit_pc: Vec::new(),
                merge_entry_by_green: Vec::new(),
                pcdep_by_jit_pc: Vec::new(),
                depth_pred_by_jit_pc: Vec::new(),
                depth_trivia_marker_by_jit_pc: Vec::new(),
                depth_trivia_pred_by_jit_pc: Vec::new(),
                pcdep_trivia_marker_by_jit_pc: Vec::new(),
                pcdep_trivia_pred_by_jit_pc: Vec::new(),
                const_ref_trivia_marker_by_jit_pc: Vec::new(),
                const_ref_trivia_pred_by_jit_pc: Vec::new(),
                result_color_trivia_marker_by_jit_pc: Vec::new(),
                result_color_trivia_pred_by_jit_pc: Vec::new(),
                resume_marker_marker_by_jit_pc: Vec::new(),
                resume_marker_pred_by_jit_pc: Vec::new(),
                after_residual_marker_marker_by_jit_pc: Vec::new(),
                after_residual_marker_pred_by_jit_pc: Vec::new(),
                result_color_after_residual_marker_by_jit_pc: Vec::new(),
                result_color_after_residual_pred_by_jit_pc: Vec::new(),
                // Encoder/decoder readers in
                // `get_list_of_active_boxes`, `regalloc::external/input_indices`,
                // and `setup_bridge_sym::portal_red_regs_at` sentinel-skip both
                // values together. A real `0` here would alias every locals-
                // bank color 0 read and silently substitute `sym.frame` for
                // unrelated locals/stack slots.
                portal_frame_reg: u16::MAX,
                portal_ec_reg: u16::MAX,
                built_as_portal: false,
                stack_base: 0,
                max_stackdepth: 0,
                has_color_map: false,
                const_ref_slots_by_jit_pc: Vec::new(),
                is_drained: false,
            },
            code_ptr,
            false,
        )
    }
}
