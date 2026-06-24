//! Runtime call descriptor constructors.
//!
//! No `rpython/jit/metainterp/call_descr.py` file exists. This module is
//! the Rust runtime boundary for descriptors produced by
//! `rpython/jit/codewriter/call.py::getcalldescr` through
//! `cpu.calldescrof(...)` (`rpython/jit/backend/model.py:180`) and then
//! consumed by metainterp, blackhole, optimizer, and backend call paths.
//! Keeping the constructors here avoids a fake metainterp upstream file
//! while still making the call-descr surface explicit.

use std::sync::Arc;

use majit_backend::JitCellToken;
use majit_ir::effectinfo::EffectInfoCell;
use majit_ir::{
    CallDescr, DescrRef, EffectInfo, ExtraEffect, OopSpecIndex, PyreHelperKind, Type,
    VableExpansion,
};

/// Generic CallDescr for function call operations.
///
/// Stores per-call-site EffectInfo, matching RPython's
/// `effectinfo_from_writeanalyze` (call.py:320). The EI is wrapped in
/// `EffectInfoCell` so `compute_bitstrings` can install the compacted
/// bitstrings post-construction (see `EffectInfoCell` doc and
/// `Descr::set_effect_bitstrings` SAFETY note).
#[derive(Debug)]
struct MetaCallDescr {
    heapcache_index: u32,
    arg_types: Vec<Type>,
    result_type: Type,
    result_signed: bool,
    result_size: usize,
    effect_info: EffectInfoCell,
}

/// `compile.py:187 isinstance(descr, JitCellToken)` parity.
///
/// RPython's `op.getdescr()` for a `CALL_ASSEMBLER_*` op IS a `JitCellToken`
/// — `record_loop_or_bridge` reads `descr.number` directly and calls
/// `original.record_jump_to(descr)` without any indirection. majit cannot
/// inherit-from the trait, but it preserves the *identity* contract by
/// owning an `Arc<JitCellToken>` here. Callers (`direct_assembler_call`,
/// `compile_tmp_callback`) clone the same Arc that the warm cell /
/// `CompiledEntry::token` /` MemoryManager.alive_loops` already hold, so the
/// keepalive walker's downcast recovers the production token's strong
/// reference rather than a number-recovered side-table lookup.
#[derive(Debug)]
struct MetaCallAssemblerDescr {
    arg_types: Vec<Type>,
    result_type: Type,
    target_token: Arc<JitCellToken>,
    vable_expansion: Option<VableExpansion>,
}

impl majit_ir::Descr for MetaCallDescr {
    fn index(&self) -> u32 {
        self.heapcache_index
    }
    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
    /// `effectinfo.py:537-538 setattr(ei, 'bitstring_*', …)` — invoked
    /// by `effectinfo::compute_bitstrings` after class assignment.
    /// Delegates to `EffectInfoCell::set_bitstrings` which encodes the
    /// single-writer setup-time mutation through `UnsafeCell` rather
    /// than the earlier raw-pointer cast (Rust aliasing model).
    fn set_effect_bitstrings(
        &self,
        readonly_descrs_fields: Option<Vec<u8>>,
        write_descrs_fields: Option<Vec<u8>>,
        readonly_descrs_arrays: Option<Vec<u8>>,
        write_descrs_arrays: Option<Vec<u8>>,
        readonly_descrs_interiorfields: Option<Vec<u8>>,
        write_descrs_interiorfields: Option<Vec<u8>>,
    ) {
        self.effect_info.set_bitstrings(
            readonly_descrs_fields,
            write_descrs_fields,
            readonly_descrs_arrays,
            write_descrs_arrays,
            readonly_descrs_interiorfields,
            write_descrs_interiorfields,
        );
    }
}

impl CallDescr for MetaCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }
    fn result_type(&self) -> Type {
        self.result_type
    }
    fn result_size(&self) -> usize {
        self.result_size
    }
    fn is_result_signed(&self) -> bool {
        self.result_signed
    }
    fn get_extra_info(&self) -> &EffectInfo {
        self.effect_info.get()
    }
}

impl majit_ir::Descr for MetaCallAssemblerDescr {
    fn index(&self) -> u32 {
        u32::MAX
    }
    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
    fn as_loop_token_descr(&self) -> Option<&dyn majit_ir::descr::LoopTokenDescr> {
        Some(self)
    }
}

impl CallDescr for MetaCallAssemblerDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }
    fn result_type(&self) -> Type {
        self.result_type
    }
    fn result_size(&self) -> usize {
        8
    }
    fn call_target_token(&self) -> Option<u64> {
        Some(self.target_token.number)
    }
    fn call_virtualizable_index(&self) -> Option<usize> {
        self.target_token.virtualizable_arg_index
    }
    fn get_extra_info(&self) -> &EffectInfo {
        static INFO: EffectInfo = EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
        &INFO
    }
    fn vable_expansion(&self) -> Option<&VableExpansion> {
        self.vable_expansion.as_ref()
    }
}

impl majit_ir::descr::LoopTokenDescr for MetaCallAssemblerDescr {
    fn loop_token_number(&self) -> u64 {
        self.target_token.number
    }

    fn call_virtualizable_index(&self) -> Option<usize> {
        self.target_token.virtualizable_arg_index
    }

    fn token_handle_any(&self) -> Option<&dyn std::any::Any> {
        Some(&self.target_token)
    }
}

/// Default EffectInfo for an external residual call without per-callee
/// heap analyzer output — the `EF_CAN_RAISE` row of `call.py:300-301
/// elif self._canraise(op):` selected through
/// `effectinfo_from_writeanalyze` with the
/// `graphanalyze.py:60 analyze_external_call` default
/// (`bottom_result()` = empty set).
///
/// Shape: `extraeffect=CanRaise`, every `_*_descrs_*` raw set =
/// `Some(Vec::new())`, every `*_descrs_*` bitstring = `Some(Vec::new())`,
/// `can_collect=true` (PyPy `effectinfo.py:283` default).
/// `effectinfo.py:293-299` else-branch: when the analyzer returns
/// non-`top_set` effects, raw sets START at `[]` and grow with actual
/// effects; for an empty effects set they stay `[]`. The matching
/// invariant `effectinfo.py:149-162` `RandomEffects ⇔ raw=None`
/// keeps this distinct from `MOST_GENERAL` (`extraeffect=RandomEffects
/// ⇔ raw=None ⇔ bitstring=None`).
///
/// **Why not `MOST_GENERAL`**: `EF_RANDOM_EFFECTS` is reserved for
/// callees the `RandomEffectsAnalyzer` (`effectinfo.py:410-415`)
/// flags via `funcobj.random_effects_on_gcobjs`; collapsing the
/// plain "no-analyzer" case onto it routes every residual call
/// through `OptHeap.call_has_random_effects → clean_caches` (full
/// heap invalidation) and `check_forces_virtual_or_virtualizable()`
/// → `GUARD_NOT_FORCED`, both of which PyPy reserves for genuinely
/// random callees. The empty-frozenset shape lets
/// `compute_bitstrings` produce zero-length bitstrings so
/// `bitcheck` short-circuits to false; cached heap state survives
/// across these calls per PyPy.
pub fn default_effect_info() -> EffectInfo {
    EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None)
}

/// `EF_CANNOT_RAISE` analyzer-absent fallback — the `call.py:303
/// else:` row of `call.py:282-303 getcalldescr` selected when
/// `self._canraise(op) == False`, fed through
/// `effectinfo_from_writeanalyze` with the
/// `graphanalyze.py:60 analyze_external_call` default
/// (`bottom_result()` = empty set).
///
/// Shape: `extraeffect=CannotRaise`, every `_*_descrs_*` raw set =
/// `Some(Vec::new())`, every `*_descrs_*` bitstring = `Some(Vec::new())`,
/// `can_collect=true` (the writeanalyzer's
/// `effectinfo.py:283 can_collect=True` default).
/// `effectinfo.py:293-299` else-branch builds empty raw sets when
/// the analyzer returns non-`top_set` effects.
///
/// Distinct from [`CANNOT_RAISE_NO_HEAP_EFFECT_INFO`] only by
/// `can_collect`: that const carries `can_collect=false` for helpers
/// the producer additionally asserts cannot trigger GC; the
/// analyzer-absent default here keeps the
/// `effectinfo_from_writeanalyze(can_collect=True)` PyPy default.
/// `check_can_raise()` (`effectinfo.py:236`) reads
/// `extraeffect > EF_CANNOT_RAISE` so the canonical walker omits
/// the trailing `GUARD_NO_EXCEPTION` for this slot.
pub fn cannot_raise_effect_info() -> EffectInfo {
    EffectInfo::const_new(ExtraEffect::CannotRaise, OopSpecIndex::None)
}

/// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` analyzer-absent fallback —
/// the `call.py:288-289 if self.virtualizable_analyzer.analyze(op)`
/// row of `call.py:282-303 getcalldescr`, fed through
/// `effectinfo_from_writeanalyze` with the
/// `graphanalyze.py:60 analyze_external_call` default
/// (`bottom_result()` = empty set).
///
/// Shape: `extraeffect=ForcesVirtualOrVirtualizable`, every
/// `_*_descrs_*` raw set = `Some(Vec::new())`, every `*_descrs_*`
/// bitstring = `Some(Vec::new())`, `can_collect=true` (PyPy
/// `effectinfo.py:364-365` `if extraeffect >= EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE:
/// can_collect = True`).
///
/// **Distinct from `MOST_GENERAL`**: `EF_RANDOM_EFFECTS` is reserved
/// for the `RandomEffectsAnalyzer` (`effectinfo.py:410-415
/// random_effects_on_gcobjs`) branch. `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
/// is the dedicated virtualizable-forcing slot — both pass
/// `check_forces_virtual_or_virtualizable()` via the `>=` test at
/// `effectinfo.py:249-250`, but only `RandomEffects` trips
/// `has_random_effects()` (`effectinfo.py:252`) and routes
/// `OptHeap` through `clean_caches`. Collapsing MayForce to
/// `MOST_GENERAL` over-invalidates the heap cache PyPy keeps live
/// for analyzer-empty virtualizable-forcing callees.
pub fn forces_virtual_or_virtualizable_effect_info() -> EffectInfo {
    EffectInfo::const_new(
        ExtraEffect::ForcesVirtualOrVirtualizable,
        OopSpecIndex::None,
    )
}

/// `EF_CANNOT_RAISE` for a callee that the producer statically knows
/// touches no heap state and cannot trigger GC — typically a flat TLS
/// read/write or a buffer-flush shim.  `call.py:320-324
/// effectinfo_from_writeanalyze` would compute empty
/// `readonly_descrs_*` / `write_descrs_*` bitsets and `can_collect =
/// False` from `read_analyzer` / `write_analyzer` / `collect_analyzer`
/// for such helpers.  Using [`cannot_raise_effect_info()`] for them is
/// the analyzer-absent conservative fallback, which over-reports the
/// callee as a heap mutator and inflates GC map / liveness work; this
/// constant is the matching analyzer-output for known-flat helpers.
pub const CANNOT_RAISE_NO_HEAP_EFFECT_INFO: EffectInfo = EffectInfo {
    extraeffect: ExtraEffect::CannotRaise,
    oopspecindex: OopSpecIndex::None,
    pyre_helper: PyreHelperKind::None,
    _readonly_descrs_fields: Some(Vec::new()),
    _write_descrs_fields: Some(Vec::new()),
    _readonly_descrs_arrays: Some(Vec::new()),
    _write_descrs_arrays: Some(Vec::new()),
    _readonly_descrs_interiorfields: Some(Vec::new()),
    _write_descrs_interiorfields: Some(Vec::new()),
    readonly_descrs_fields: Some(Vec::new()),
    write_descrs_fields: Some(Vec::new()),
    readonly_descrs_arrays: Some(Vec::new()),
    write_descrs_arrays: Some(Vec::new()),
    readonly_descrs_interiorfields: Some(Vec::new()),
    write_descrs_interiorfields: Some(Vec::new()),
    can_invalidate: false,
    can_collect: false,
    single_write_descr_array: None,
    extradescrs: None,
    call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
};

/// `EF_ELIDABLE_CANNOT_RAISE` with `OS_INT_PY_DIV` oopspec — Python `//`
/// (floor division). RPython parity: jtransform.py:2046-2047
/// `_handle_int_special` classifies `int.py_div` as
/// `EF_ELIDABLE_CANNOT_RAISE`. Source-level zero/overflow wrappers
/// (`rint.py:417 ll_int_py_div_zer`, `:429 ll_int_py_div_ovf_zer`)
/// are inlined into the calling graph before the JIT sees this
/// oopspec call; their checks become runtime guards in the trace,
/// not properties of this call descriptor. The optimizer's
/// `optimize_call_int_py_div` (rewrite.py:713-766) reads the
/// `OS_INT_PY_DIV` oopspec to specialize power-of-2 divisors to
/// `int_rshift`, constant 1 to identity, constant -1 to `int_neg`, etc.
/// Callee is pure: no heap touched, no GC trigger, no raise.
pub const INT_PY_DIV_EFFECT_INFO: EffectInfo = EffectInfo {
    extraeffect: ExtraEffect::ElidableCannotRaise,
    oopspecindex: OopSpecIndex::IntPyDiv,
    pyre_helper: PyreHelperKind::None,
    _readonly_descrs_fields: Some(Vec::new()),
    _write_descrs_fields: Some(Vec::new()),
    _readonly_descrs_arrays: Some(Vec::new()),
    _write_descrs_arrays: Some(Vec::new()),
    _readonly_descrs_interiorfields: Some(Vec::new()),
    _write_descrs_interiorfields: Some(Vec::new()),
    readonly_descrs_fields: Some(Vec::new()),
    write_descrs_fields: Some(Vec::new()),
    readonly_descrs_arrays: Some(Vec::new()),
    write_descrs_arrays: Some(Vec::new()),
    readonly_descrs_interiorfields: Some(Vec::new()),
    write_descrs_interiorfields: Some(Vec::new()),
    can_invalidate: false,
    can_collect: false,
    single_write_descr_array: None,
    extradescrs: None,
    call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
};

/// Counterpart of [`INT_PY_DIV_EFFECT_INFO`] for Python `%`. RPython
/// parity: jtransform.py:2046-2047 classifies `int.py_mod` as
/// `EF_ELIDABLE_CANNOT_RAISE`; zero/overflow checks from the source
/// wrappers (`rint.py:509 ll_int_py_mod_zer`, `:520
/// ll_int_py_mod_ovf_zer`) are inlined upstream of the JIT trace.
pub const INT_PY_MOD_EFFECT_INFO: EffectInfo = EffectInfo {
    extraeffect: ExtraEffect::ElidableCannotRaise,
    oopspecindex: OopSpecIndex::IntPyMod,
    pyre_helper: PyreHelperKind::None,
    _readonly_descrs_fields: Some(Vec::new()),
    _write_descrs_fields: Some(Vec::new()),
    _readonly_descrs_arrays: Some(Vec::new()),
    _write_descrs_arrays: Some(Vec::new()),
    _readonly_descrs_interiorfields: Some(Vec::new()),
    _write_descrs_interiorfields: Some(Vec::new()),
    readonly_descrs_fields: Some(Vec::new()),
    write_descrs_fields: Some(Vec::new()),
    readonly_descrs_arrays: Some(Vec::new()),
    write_descrs_arrays: Some(Vec::new()),
    readonly_descrs_interiorfields: Some(Vec::new()),
    write_descrs_interiorfields: Some(Vec::new()),
    can_invalidate: false,
    can_collect: false,
    single_write_descr_array: None,
    extradescrs: None,
    call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
};

/// `EF_ELIDABLE_CANNOT_RAISE` (effectinfo.py:17). Selected by
/// `call.py:299 getcalldescr` when `_canraise(op) == False` for an
/// elidable callee — `pyjitpl.py:2126 do_residual_call` records
/// `CALL_PURE_*` without the trailing `GUARD_NO_EXCEPTION` because
/// `effectinfo.check_can_raise()` (`effectinfo.py:232`) is false for
/// `extraeffect == 0`.
pub const ELIDABLE_CANNOT_RAISE_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None);

/// `EF_ELIDABLE_OR_MEMORYERROR` (effectinfo.py:20). Selected by
/// `call.py:295 getcalldescr` when `_canraise(op) == "mem"` — i.e.
/// the elidable callee's only failure mode is `MemoryError`. Same
/// dispatch as `EF_ELIDABLE_CAN_RAISE` (`check_can_raise()` is true
/// for extraeffect ≥ 3) but distinguishes memory-only raises for the
/// optimizer.
pub const ELIDABLE_OR_MEMERROR_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::ElidableOrMemoryError, OopSpecIndex::None);

/// `EF_ELIDABLE_CAN_RAISE` (effectinfo.py:21). Pure calls do not need
/// the conservative flush — `effectinfo_from_writeanalyze` (effectinfo.py:
/// 169-181) clears `_write_descrs_*` for elidable extraeffects. With
/// the bitsets at zero this becomes "no writes" inside
/// `force_from_effectinfo`, matching upstream.
pub const ELIDABLE_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::ElidableCanRaise, OopSpecIndex::None);

/// `EF_LOOPINVARIANT` (effectinfo.py:18). Same write-mask treatment as
/// elidable; the trace optimizer recognises the opcode and skips cache
/// invalidation regardless of the bitsets.
pub const LOOPINVARIANT_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::LoopInvariant, OopSpecIndex::None);

/// Per-callee analyzer-result slot.  Mirrors `call.py:282-303 getcalldescr`'s
/// `extraeffect` selection without the `raise_analyzer` /
/// `readwrite_analyzer` / `collect_analyzer` / `randomeffects_analyzer`
/// graph-based machinery (the analyzers operate on RPython low-level
/// graphs, which pyre does not have).  Producers that statically know
/// the callee's classification — typically because the helper carries
/// a `#[elidable]` / `#[elidable_cannot_raise]` / `#[dont_look_inside]`
/// attribute — pick the matching slot at registration time;
/// [`effect_info_for_slot`] resolves it to the corresponding
/// [`EffectInfo`] const at descr construction.
///
/// `MayForce` (`EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`) and `ReleaseGil`
/// (`EF_RANDOM_EFFECTS` + non-zero `call_release_gil_target`) are
/// deliberately omitted — those EI values carry runtime-resolved
/// `target.concrete_ptr` / `save_err` slots that the const factory at
/// `jitcode/assembler.rs:emit_canonical_call_*_via_target` constructs
/// inline.  Adding them here would require duplicating the
/// `(1, 0)` sentinel + `resolve_call_release_gil_target` substitution,
/// which is out of scope for the slot enum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum EffectInfoSlot {
    /// `EF_CAN_RAISE` — `call.py:300-301 elif self._canraise(op):`
    /// branch of `getcalldescr`. Maps to [`default_effect_info()`]:
    /// `CanRaise + Some(empty)` raw sets + `Some(empty)` bitstrings,
    /// matching the writeanalyzer-empty external-call shape
    /// (`graphanalyze.py:60` `bottom_result()` + `effectinfo.py:293-299`
    /// else-branch). Default slot for producers without per-callee
    /// heap analysis.
    ///
    /// **Distinct from `RandomEffects`**: `EF_RANDOM_EFFECTS` is the
    /// `RandomEffectsAnalyzer` (`effectinfo.py:410-415`) outcome for
    /// callees flagged `random_effects_on_gcobjs`; collapsing the
    /// plain "no-analyzer" path onto it triggers `clean_caches` +
    /// `GUARD_NOT_FORCED` PyPy reserves for genuinely-random callees.
    /// No pyre producer constructs a true `RandomEffects` slot today
    /// (none of the registered helpers are `random_effects_on_gcobjs`).
    #[default]
    CanRaise,
    /// `EF_CANNOT_RAISE` — `call.py:303` `else` branch.
    CannotRaise,
    /// `EF_CANNOT_RAISE` + analyzer-confirmed empty heap. Maps to
    /// `CANNOT_RAISE_NO_HEAP_EFFECT_INFO` (`effectinfo.py:281-283`).
    CannotRaiseNoHeap,
    /// `EF_ELIDABLE_CAN_RAISE` — `call.py:297` `elif cr:` branch.
    ElidableCanRaise,
    /// `EF_ELIDABLE_CANNOT_RAISE` — `call.py:299` `else` branch under
    /// `elif elidable:`.
    ElidableCannotRaise,
    /// `EF_ELIDABLE_OR_MEMORYERROR` — `call.py:295` `if cr == "mem":`.
    ElidableOrMemerror,
    /// `EF_LOOPINVARIANT` — `call.py:291` `elif loopinvariant:`.
    LoopInvariant,
}

/// Resolve a [`EffectInfoSlot`] to its matching [`EffectInfo`] const.
///
/// `call.py:320 effectinfo_from_writeanalyze` constructs the final EI
/// from the `extraeffect` plus the analyzer outputs; pyre's per-slot
/// const captures the analyzer-absent fallback for that `extraeffect`.
pub fn effect_info_for_slot(slot: EffectInfoSlot) -> EffectInfo {
    match slot {
        EffectInfoSlot::CanRaise => default_effect_info(),
        EffectInfoSlot::CannotRaise => cannot_raise_effect_info(),
        EffectInfoSlot::CannotRaiseNoHeap => CANNOT_RAISE_NO_HEAP_EFFECT_INFO.clone(),
        EffectInfoSlot::ElidableCanRaise => ELIDABLE_EFFECT_INFO,
        EffectInfoSlot::ElidableCannotRaise => ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        EffectInfoSlot::ElidableOrMemerror => ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        EffectInfoSlot::LoopInvariant => LOOPINVARIANT_EFFECT_INFO,
    }
}

/// Pick the upstream-equivalent default effect for an opcode whose
/// callee has not been write-analyzed.
///
/// `pyjitpl.py:1991-1995 do_residual_or_indirect_call` selects between
/// CALL / CALL_PURE / CALL_LOOPINVARIANT / CALL_MAY_FORCE based on
/// `descr.get_extra_info().extraeffect`. Pyre baked the choice into the
/// opcode at codewriter time, so reverse the mapping here so the descr
/// the optimizer reads carries the matching effect class.
///
/// `CALL_MAY_FORCE` maps to [`forces_virtual_or_virtualizable_effect_info`]
/// (`effectinfo.py:23 EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`).
/// `CALL_RELEASE_GIL` cannot be reconstructed from the opcode alone —
/// upstream `effectinfo.py:271-273 MOST_GENERAL` pairs `EF_RANDOM_EFFECTS`
/// with a `call_release_gil_target` funcptr that this helper does not
/// see, so the analyzer-absent default is fail-loud: any production
/// path that needs a release-GIL EI must build it explicitly via
/// [`make_call_descr_with_effect`] with the resolved target.
pub fn default_effect_for_opcode(opcode: majit_ir::OpCode) -> EffectInfo {
    if opcode.is_call_pure() {
        ELIDABLE_EFFECT_INFO
    } else if opcode.is_call_loopinvariant() {
        LOOPINVARIANT_EFFECT_INFO
    } else if opcode.is_call_may_force() {
        forces_virtual_or_virtualizable_effect_info()
    } else if opcode.is_call_release_gil() {
        unreachable!(
            "default_effect_for_opcode: CALL_RELEASE_GIL (`{opcode:?}`) requires \
             call_release_gil_target funcptr; build the EffectInfo explicitly via \
             make_call_descr_with_effect (effectinfo.py:271-273 MOST_GENERAL)"
        );
    } else {
        default_effect_info()
    }
}

/// Create a CallDescr with the conservative
/// [`default_effect_info()`] (`EF_CAN_RAISE` + `Some(Vec::new())`
/// raw sets + `Some(Vec::new())` bitstrings + `can_collect=true`).
/// This is the analyzer-absent fallback mirroring
/// `call.py:300-301 elif self._canraise(op):` fed through
/// `effectinfo_from_writeanalyze` with the
/// `graphanalyze.py:60 analyze_external_call` default
/// (`bottom_result()` = empty set) — the
/// `effectinfo.py:293-299` else-branch shape, distinct from
/// `MOST_GENERAL`.
///
/// Production producers should prefer one of the more specific factories
/// so the per-callee classification reaches the trace IR:
///
/// * [`make_call_descr_from_target_slot`] when a resolved
///   [`crate::jitcode::JitCallTarget`] is available — threads the
///   macro-time [`EffectInfoSlot`] (`call.py:282-303 getcalldescr` parity).
/// * [`make_call_descr_for_opcode`] when only the call opcode family is
///   known (`pyjitpl.py:1991-1995 do_residual_or_indirect_call`'s
///   `EF_LOOPINVARIANT` / `EF_ELIDABLE_*` reverse-mapping).
/// * [`make_call_descr_with_effect`] when an explicit `EffectInfo` has
///   been hand-built (release-gil targets, oopspec specializations).
///
/// Remaining direct callers of this fallback are restricted to
/// `#[cfg(test)]` fixtures (pyjitpl/optimizeopt/backend test stubs)
/// where the conservative descr is the test's intent — matching the
/// "no analyzer ran" path the production fallbacks above subsume.
pub fn make_call_descr(arg_types: &[Type], result_type: Type) -> DescrRef {
    make_call_descr_with_effect(arg_types, result_type, default_effect_info())
}

/// Create a CallDescr whose effect info matches the call opcode family.
pub fn make_call_descr_for_opcode(
    opcode: majit_ir::OpCode,
    arg_types: &[Type],
    result_type: Type,
) -> DescrRef {
    make_call_descr_with_effect(arg_types, result_type, default_effect_for_opcode(opcode))
}

/// Create a CallDescr from a per-target [`EffectInfoSlot`] classification.
///
/// `call.py:282-303 getcalldescr` selects `extraeffect` per callsite
/// from the analyzer chain; pyre's analyzer-absent equivalent is the
/// `JitCallTarget.effect_info_slot` macro-time classification.  This
/// factory is the per-target entry point — callers that have a
/// resolved [`crate::jitcode::JitCallTarget`] thread its slot through.
pub fn make_call_descr_from_target_slot(
    arg_types: &[Type],
    result_type: Type,
    slot: EffectInfoSlot,
) -> DescrRef {
    make_call_descr_with_effect(arg_types, result_type, effect_info_for_slot(slot))
}

fn result_metadata(result_type: Type) -> (bool, usize) {
    let result_size = match result_type {
        Type::Int | Type::Ref | Type::Float => 8,
        Type::Void => 0,
    };
    (result_type == Type::Int, result_size)
}

/// call.py:320 `effectinfo_from_writeanalyze` parity. Create a
/// CallDescr with explicit per-call-site EffectInfo.
pub fn make_call_descr_with_effect(
    arg_types: &[Type],
    result_type: Type,
    effect_info: EffectInfo,
) -> DescrRef {
    // `effectinfo.py:182-184` invariant: no new `EffectInfo` may be
    // constructed after `compute_bitstrings` has run; PyPy enforces this
    // implicitly through codewriter lifecycle ordering, with `Ellipsis`
    // as a post-hoc bitcheck-time tripwire.  Pyre allows trace-time
    // mints, so the gate is at the construction site.  Trivial-raw
    // EIs (`raw=None` ⇒ random-effects, or `raw=Some(empty)` ⇒
    // concrete-empty) keep the invariant intact because
    // `compute_bitstrings` would map them to a `None`/empty bitstring
    // independent of any other EI's (eisetr, eisetw) class — adding
    // one post-setup never reshuffles existing class assignments.
    // Non-trivial raw sets (`Some(non-empty)`) would shift the
    // partition and silently invalidate every cached bitstring.
    if majit_ir::effectinfo::compute_bitstrings_has_run() && effect_info.has_non_trivial_raw_set() {
        panic!(
            "make_call_descr_with_effect: EffectInfo with non-trivial raw \
             descr set constructed after compute_bitstrings ran.  PyPy \
             effectinfo.py:182-184 forbids the same shape via the \
             Ellipsis sentinel + bitcheck panic.  Fix: ensure all call \
             descrs whose analyzer outputs concrete frozensets are \
             minted before `MetaInterpStaticData::finish_setup_descrs` \
             runs (codewriter setup phase).\n  effect_info: {effect_info:?}"
        );
    }
    let (result_signed, result_size) = result_metadata(result_type);
    // effectinfo.py:144-146: `if tgt_func: key += (object(),)  # don't
    // care about caching in this case` — release-gil targets bypass the
    // EffectInfo._cache via a fresh object() key.  The call descr still
    // lives in GcCache._cache_call; the key just carries a per-mint
    // breaker so release-gil call descrs never structurally collapse.
    let key = if effect_info.call_release_gil_target.0 != 0 {
        majit_ir::descr::LLType::func_key_with_fresh_release_gil_breaker(
            arg_types,
            result_type,
            result_signed,
            result_size,
            &effect_info,
        )
    } else {
        majit_ir::descr::LLType::func_key(
            arg_types,
            result_type,
            result_signed,
            result_size,
            &effect_info,
        )
    };
    let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
    gc.intern_call_descr_with(key, || {
        let descr: DescrRef = Arc::new(MetaCallDescr {
            heapcache_index: majit_ir::descr::next_call_descr_heapcache_index(),
            arg_types: arg_types.to_vec(),
            result_type,
            result_signed,
            result_size,
            effect_info: EffectInfoCell::new(effect_info),
        });
        descr
    })
}

/// Create a CallDescr for CALL_MAY_FORCE_* operations.
///
/// The trait-dispatch leg records a residual through this descr; the walker
/// leg records the equivalent residual through a calldescr the codewriter
/// builds with `forces_virtual_or_virtualizable_effect_info()`
/// (`CallFlavor::MayForce`). Both must carry the same
/// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` `EffectInfo` so the optimizer and
/// `do_residual_call` treat the two legs' may-force ops identically
/// (`MetaCallMayForceDescr::get_extra_info` mirrors that constructor).
pub fn make_call_may_force_descr(arg_types: &[Type], result_type: Type) -> DescrRef {
    #[derive(Debug)]
    struct MetaCallMayForceDescr {
        arg_types: Vec<Type>,
        result_type: Type,
    }

    impl majit_ir::Descr for MetaCallMayForceDescr {
        fn index(&self) -> u32 {
            u32::MAX
        }
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for MetaCallMayForceDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }
        fn result_type(&self) -> Type {
            self.result_type
        }
        fn result_size(&self) -> usize {
            0
        }
        fn get_extra_info(&self) -> &EffectInfo {
            // Byte-identical to `forces_virtual_or_virtualizable_effect_info()`
            // (the `CallFlavor::MayForce` row the codewriter stamps on the
            // walker-leg calldescr): `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
            // with analyzer-empty read/write bitsets. The two legs must
            // agree on the descr shape for the same residual.
            //
            // `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` is the only extraeffect
            // consistent with the `CALL_MAY_FORCE_*` opcode this descr
            // accompanies: `check_forces_virtual_or_virtualizable()` reads
            // `extraeffect >= EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`, so the
            // earlier `EF_CAN_RAISE` (5 < 6) failed that test while still
            // riding a may-force op. `> EF_CANNOT_RAISE` keeps
            // `check_can_raise()` true so the may-force sequence still
            // records its trailing `GUARD_NO_EXCEPTION`.
            //
            // The empty bitsets are faithful, not a shortcut: the
            // analyzer-absent fallback (`effectinfo_from_writeanalyze` with
            // `bottom_result()`) produces empty `read/write_descrs_*`, so
            // `force_from_effectinfo` finds no descr bits set and leaves
            // cached heap state live across the call. Promoting instead to
            // `EF_RANDOM_EFFECTS` would trip `has_random_effects()` and
            // route OptHeap through `clean_caches`, over-invalidating heap
            // PyPy keeps live for analyzer-empty virtualizable-forcing
            // callees.
            static INFO: EffectInfo = EffectInfo::const_new(
                ExtraEffect::ForcesVirtualOrVirtualizable,
                OopSpecIndex::None,
            );
            &INFO
        }
    }

    Arc::new(MetaCallMayForceDescr {
        arg_types: arg_types.to_vec(),
        result_type,
    })
}

/// `compile.py:187 isinstance(descr, JitCellToken)` parity factory.
///
/// Create a `CALL_ASSEMBLER_*` descr that owns the same `Arc<JitCellToken>`
/// as the production warm cell / `CompiledEntry::token` / `alive_loops`.
/// `direct_assembler_call` (`pyjitpl.py:3589-3609`) is the canonical caller —
/// it threads the cell's compiled token through, so `record_loop_or_bridge`'s
/// keepalive walker downcasts the descr and pushes that same Arc into
/// `original.keepalive_tokens`, matching `compile.py:187 record_jump_to(descr)`.
pub fn make_call_assembler_descr(
    target_token: Arc<JitCellToken>,
    arg_types: &[Type],
    result_type: Type,
) -> DescrRef {
    Arc::new(MetaCallAssemblerDescr {
        arg_types: arg_types.to_vec(),
        result_type,
        target_token,
        vable_expansion: None,
    })
}

/// Number-only factory for callers that have not yet been threaded an
/// `Arc<JitCellToken>` (jitcode dispatch in `dispatch.rs`, test fixtures).
///
/// Synthesises a fresh stand-alone `Arc<JitCellToken>` with the requested
/// `target_number` so the descr keeps the same shape as the identity-preserving
/// path. Identity is **not** preserved — the keepalive walker recovers the
/// real Arc via `jitcell_token_by_number(target_number)` for these descrs
/// (`pyjitpl.rs:record_loop_or_bridge` Arc-fallback inside the
/// CALL_ASSEMBLER branch). Sites transitioning to
/// `make_call_assembler_descr` once the Arc is available upstream remove
/// the lookup.
pub fn make_call_assembler_descr_by_number(
    target_number: u64,
    arg_types: &[Type],
    result_type: Type,
    virtualizable_arg_index: Option<usize>,
) -> DescrRef {
    let mut tok = JitCellToken::new(target_number);
    tok.virtualizable_arg_index = virtualizable_arg_index;
    make_call_assembler_descr(Arc::new(tok), arg_types, result_type)
}

/// rewrite.py:665-695 handle_call_assembler: create a CallDescr that carries
/// virtualizable expansion info. The backend reads fields from the frame
/// reference to populate the callee's full inputarg jitframe layout.
pub fn make_call_assembler_descr_with_vable(
    target_token: Arc<JitCellToken>,
    arg_types: &[Type],
    result_type: Type,
    expansion: VableExpansion,
) -> DescrRef {
    Arc::new(MetaCallAssemblerDescr {
        arg_types: arg_types.to_vec(),
        result_type,
        target_token,
        vable_expansion: Some(expansion),
    })
}

/// Number-only sibling of `make_call_assembler_descr_with_vable` for transitional
/// callers (jitcode dispatch). See `make_call_assembler_descr_by_number`.
pub fn make_call_assembler_descr_with_vable_by_number(
    target_number: u64,
    arg_types: &[Type],
    result_type: Type,
    expansion: VableExpansion,
) -> DescrRef {
    let tok = Arc::new(JitCellToken::new(target_number));
    make_call_assembler_descr_with_vable(tok, arg_types, result_type, expansion)
}

#[cfg(test)]
mod set_effect_bitstrings_tests {
    use super::*;
    use majit_ir::EffectInfo;

    /// `Descr::set_effect_bitstrings` writes through to the cached
    /// `MetaCallDescr.effect_info`, visible to subsequent
    /// `cd.get_extra_info()` reads.  Mirrors `effectinfo.py:537-538
    /// setattr(ei, 'bitstring_*', …)`.
    #[test]
    fn set_effect_bitstrings_publishes_to_get_extra_info() {
        use majit_ir::descr::SimpleFieldDescr;
        let f3: DescrRef = Arc::new(SimpleFieldDescr::new(3, 0, 8, Type::Int, false));
        let f7: DescrRef = Arc::new(SimpleFieldDescr::new(7, 0, 8, Type::Int, false));
        let mut ei = EffectInfo::default();
        ei._readonly_descrs_fields = Some(vec![f3, f7]);
        ei._write_descrs_fields = Some(vec![]);
        ei._readonly_descrs_arrays = Some(vec![]);
        ei._write_descrs_arrays = Some(vec![]);
        ei._readonly_descrs_interiorfields = Some(vec![]);
        ei._write_descrs_interiorfields = Some(vec![]);
        let descr = make_call_descr_with_effect(&[Type::Int], Type::Int, ei);
        let cd = descr.as_call_descr().unwrap();
        // Pre-set: bitstring_readonly_descrs_fields was seeded at
        // construction by make_call_descr_with_effect via Default.
        // After set_effect_bitstrings the new value wins.
        descr.set_effect_bitstrings(
            Some(vec![0x88]),
            Some(vec![0x00]),
            Some(vec![0x00]),
            Some(vec![0x00]),
            Some(vec![0x00]),
            Some(vec![0x00]),
        );
        let ei_after = cd.get_extra_info();
        assert_eq!(
            ei_after.readonly_descrs_fields.as_deref(),
            Some(&[0x88u8][..])
        );
        assert_eq!(ei_after.write_descrs_fields.as_deref(), Some(&[0x00u8][..]));
    }

    /// Default `Descr::set_effect_bitstrings` is a no-op for descrs
    /// without an `EffectInfo` (e.g. field/array/size/fail descrs).
    #[test]
    fn default_set_effect_bitstrings_is_noop_for_non_call_descrs() {
        // `SimpleFieldDescr` does not override `set_effect_bitstrings`,
        // so calling it should not panic and should not affect any
        // other descr state.
        let descr: DescrRef = Arc::new(majit_ir::descr::SimpleFieldDescr::new(
            42,
            0,
            8,
            Type::Int,
            false,
        ));
        descr.set_effect_bitstrings(
            Some(vec![0xff]),
            Some(vec![0xff]),
            Some(vec![0xff]),
            Some(vec![0xff]),
            Some(vec![0xff]),
            Some(vec![0xff]),
        );
        // Field descr's own getters still work normally.
        assert_eq!(descr.index(), 42);
    }

    /// End-to-end integration: after `compute_bitstrings` runs over a
    /// constructed all_descrs vector, calling
    /// `Descr::set_effect_bitstrings` for each call descr publishes
    /// bitstrings keyed by `descr.get_ei_index()`. Mirrors
    /// `effectinfo.py:528-538` write-back loop.
    #[test]
    fn compute_bitstrings_then_set_publishes_eiindex_keyed_bitstrings() {
        use majit_ir::descr::SimpleFieldDescr;
        // Two field descrs.
        let f1: DescrRef = Arc::new(SimpleFieldDescr::new(91_000_001, 0, 8, Type::Int, false));
        let f2: DescrRef = Arc::new(SimpleFieldDescr::new(91_000_002, 0, 8, Type::Int, false));

        // Two EIs that BOTH read `f1` (Arc identity).
        let mut ei_a = EffectInfo::default();
        ei_a._readonly_descrs_fields = Some(vec![f1.clone()]);
        ei_a._write_descrs_fields = Some(vec![]);
        ei_a._readonly_descrs_arrays = Some(vec![]);
        ei_a._write_descrs_arrays = Some(vec![]);
        ei_a._readonly_descrs_interiorfields = Some(vec![]);
        ei_a._write_descrs_interiorfields = Some(vec![]);

        let mut ei_b = EffectInfo::default();
        ei_b._readonly_descrs_fields = Some(vec![f1.clone()]);
        ei_b._write_descrs_fields = Some(vec![]);
        ei_b._readonly_descrs_arrays = Some(vec![]);
        ei_b._write_descrs_arrays = Some(vec![]);
        ei_b._readonly_descrs_interiorfields = Some(vec![]);
        ei_b._write_descrs_interiorfields = Some(vec![]);

        let cd_a = make_call_descr_with_effect(&[Type::Int], Type::Int, ei_a.clone());
        let cd_b = make_call_descr_with_effect(&[Type::Float], Type::Float, ei_b.clone());

        // Run compute_bitstrings the way `MetaInterpStaticData::finish_setup_descrs`
        // does: clone EIs, mutate clones, write back via the trait.
        let all_descrs: Vec<DescrRef> = vec![f1.clone(), f2.clone(), cd_a.clone(), cd_b.clone()];
        let mut owned_eis: Vec<EffectInfo> = vec![ei_a.clone(), ei_b.clone()];
        {
            let mut ei_refs: Vec<&mut EffectInfo> = owned_eis.iter_mut().collect();
            majit_ir::effectinfo::compute_bitstrings(&all_descrs, &mut ei_refs);
        }
        cd_a.set_effect_bitstrings(
            owned_eis[0].readonly_descrs_fields.clone(),
            owned_eis[0].write_descrs_fields.clone(),
            owned_eis[0].readonly_descrs_arrays.clone(),
            owned_eis[0].write_descrs_arrays.clone(),
            owned_eis[0].readonly_descrs_interiorfields.clone(),
            owned_eis[0].write_descrs_interiorfields.clone(),
        );
        cd_b.set_effect_bitstrings(
            owned_eis[1].readonly_descrs_fields.clone(),
            owned_eis[1].write_descrs_fields.clone(),
            owned_eis[1].readonly_descrs_arrays.clone(),
            owned_eis[1].write_descrs_arrays.clone(),
            owned_eis[1].readonly_descrs_interiorfields.clone(),
            owned_eis[1].write_descrs_interiorfields.clone(),
        );

        // f1 was assigned ei_index = 0 (first descr in the only class).
        // f2 was not in any EI so ei_index stays at u32::MAX.
        assert_eq!(f1.get_ei_index(), 0);
        assert_eq!(f2.get_ei_index(), u32::MAX);

        // Both EIs' bitstrings encode a bit at f1's ei_index = 0.
        let bs_a = cd_a
            .as_call_descr()
            .unwrap()
            .get_extra_info()
            .readonly_descrs_fields
            .clone()
            .expect("ei_a readonly bitstring");
        assert!(majit_ir::bitstring::bitcheck(&bs_a, f1.get_ei_index()));
        let bs_b = cd_b
            .as_call_descr()
            .unwrap()
            .get_extra_info()
            .readonly_descrs_fields
            .clone()
            .expect("ei_b readonly bitstring");
        assert!(majit_ir::bitstring::bitcheck(&bs_b, f1.get_ei_index()));
    }

    /// `GcCache._cache_call` returns entries minted by
    /// `make_call_descr_with_effect`.  Used by
    /// `MetaInterpStaticData::finish_setup_descrs` to walk the full
    /// EI population for `compute_bitstrings`.
    #[test]
    fn gc_cache_call_snapshot_returns_recent_entries() {
        use majit_ir::descr::SimpleFieldDescr;
        let f1: DescrRef = Arc::new(SimpleFieldDescr::new(1, 0, 8, Type::Int, false));
        let mut ei = EffectInfo::default();
        ei._readonly_descrs_fields = Some(vec![f1]);
        ei._write_descrs_fields = Some(vec![]);
        ei._readonly_descrs_arrays = Some(vec![]);
        ei._write_descrs_arrays = Some(vec![]);
        ei._readonly_descrs_interiorfields = Some(vec![]);
        ei._write_descrs_interiorfields = Some(vec![]);
        let descr = make_call_descr_with_effect(&[Type::Int, Type::Ref], Type::Float, ei);

        let cached = majit_ir::descr::gc_cache().lock().unwrap().snapshot_calls();
        // The descr we just constructed is in the cache. We also
        // tolerate the cache holding entries from earlier tests in the
        // same process; we only assert membership of OUR descr.
        let my_idx = descr.index();
        let found = cached.iter().any(|d| d.index() == my_idx);
        assert!(
            found,
            "GcCache._cache_call snapshot must include the descr we just made"
        );
    }

    /// The trait-dispatch leg's `make_call_may_force_descr` and the walker
    /// leg's `forces_virtual_or_virtualizable_effect_info()` must classify a
    /// may-force residual identically: `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
    /// (consistent with the `CALL_MAY_FORCE_*` opcode), can-raise (trailing
    /// `GUARD_NO_EXCEPTION` retained), no random effects (no `clean_caches`
    /// over-invalidation), and no oopspec.
    #[test]
    fn may_force_descr_matches_forces_virtual_effect_info() {
        let canonical = forces_virtual_or_virtualizable_effect_info();
        let descr = make_call_may_force_descr(&[Type::Ref], Type::Ref);
        let ei = descr.as_call_descr().unwrap().get_extra_info();

        assert_eq!(ei.extraeffect, canonical.extraeffect);
        assert_eq!(ei.extraeffect, ExtraEffect::ForcesVirtualOrVirtualizable);
        assert!(ei.check_forces_virtual_or_virtualizable());
        assert!(ei.check_can_raise(false));
        assert!(!ei.has_random_effects());
        assert!(!ei.has_oopspec());
        assert!(!ei.check_can_invalidate());
    }
}
