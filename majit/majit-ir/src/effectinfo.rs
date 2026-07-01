//! Side effect classification for calls.
//!
//! Translated from `rpython/jit/codewriter/effectinfo.py`.
//!
//! TODO: the upstream module path is
//! `rpython/jit/codewriter/effectinfo.py`, but in pyre this lives in
//! `majit-ir` because `LLType::for_call` (descr.rs:46) holds
//! `extraeffect`/`oopspecindex` indices and `EffectInfo` is constructed
//! from RPython call analysis. Putting it in `majit-translate` would
//! create a circular crate dependency (`majit-ir` ↔ `majit-translate`).
//! The name and contents otherwise mirror upstream line-by-line.

use crate::descr::DescrRef;
use serde::{Deserialize, Serialize};

/// effectinfo.py:9-10 `class UnsupportedFieldExc(Exception)`.
#[derive(Debug, Clone)]
pub struct UnsupportedFieldExc(pub String);

impl std::fmt::Display for UnsupportedFieldExc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnsupportedFieldExc {}

/// `EffectInfo` with setup-time interior mutability for the bitstring
/// fields.
///
/// PyPy `effectinfo.py:182-184` documents that `bitstring_*_descrs_*`
/// fields are "initialized later, in compute_bitstrings()" — the EI
/// object is constructed once with raw `_*_descrs_*` frozensets, then
/// mutated by `effectinfo.compute_bitstrings(self.all_descrs)`
/// (`pyjitpl.py:2287-2290`) to install the compacted bitstrings.
/// Python's mutable object model makes the in-place setattr trivial;
/// Rust requires explicit interior mutability for the same shape.
///
/// `EffectInfoCell` wraps the EI in `UnsafeCell` and exposes:
/// - `get(&self) -> &EffectInfo` for the immutable-borrow read path
///   used by every `cd.get_extra_info()` consumer (heap.rs / virtualize.rs
///   / rewrite.rs / pure.rs / intbounds.rs / vstring.rs etc., 40+ sites).
/// - `set_bitstrings(&self, …)` for the single setup-time writer
///   (`MetaInterpStaticData::finish_setup_descrs`).
///
/// SAFETY contract (encoded via the `set_bitstrings` documentation,
/// not enforced statically): exactly one writer at JIT-setup time,
/// no concurrent reader-borrow in flight during the write. Pyre's
/// JIT pipeline is single-threaded today; the `OnceLock`-like
/// publish-once pattern at `pyre-jit-trace::state::finish_setup_if_needed`
/// (gated on `was_done`) honours this contract structurally. Replacing
/// the earlier raw `&EffectInfo as *const → *mut` cast with `UnsafeCell`
/// keeps the same operational behaviour while making the contract
/// explicit and respecting the Rust aliasing model — `UnsafeCell::get()`
/// returns a `*mut T` whose mutation is permitted given a unique-
/// access invariant on the caller, which is exactly what we have.
#[derive(Debug)]
pub struct EffectInfoCell {
    cell: std::cell::UnsafeCell<EffectInfo>,
}

// SAFETY: pyre's JIT pipeline is single-threaded today; `EffectInfoCell`
// is shared across threads only as part of `Arc<dyn Descr>` cache
// entries that read `get()` after `finish_setup_descrs` has installed
// the final bitstrings. The contract documented on `set_bitstrings`
// requires the writer to have unique access at the moment of the call.
unsafe impl Send for EffectInfoCell {}
unsafe impl Sync for EffectInfoCell {}

impl EffectInfoCell {
    /// Wrap an `EffectInfo` for setup-time mutation.
    pub fn new(ei: EffectInfo) -> Self {
        Self {
            cell: std::cell::UnsafeCell::new(ei),
        }
    }

    /// Read-only borrow.  Used by every `CallDescr::get_extra_info()`
    /// implementor.  SAFETY: shared `&EffectInfo` is sound as long as
    /// no concurrent `set_bitstrings` is in flight; see type doc.
    pub fn get(&self) -> &EffectInfo {
        // SAFETY: the only mutation goes through `set_bitstrings`,
        // which is documented as setup-time single-writer; readers
        // observe the final value after the JIT-setup happens-before
        // edge. Pyre is single-threaded today.
        unsafe { &*self.cell.get() }
    }

    /// `effectinfo.py:537-538 setattr(ei, 'bitstring_*', ...)` —
    /// install the compacted bitstrings produced by
    /// `compute_bitstrings`.  Sole caller is
    /// `MetaInterpStaticData::finish_setup_descrs`.
    ///
    /// SAFETY: see type doc. The argument signature mirrors
    /// `Descr::set_effect_bitstrings`, mutating the six bitstring
    /// fields and leaving every other EI field untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn set_bitstrings(
        &self,
        readonly_descrs_fields: Option<Vec<u8>>,
        write_descrs_fields: Option<Vec<u8>>,
        readonly_descrs_arrays: Option<Vec<u8>>,
        write_descrs_arrays: Option<Vec<u8>>,
        readonly_descrs_interiorfields: Option<Vec<u8>>,
        write_descrs_interiorfields: Option<Vec<u8>>,
    ) {
        // SAFETY: see type doc. Single-writer setup-time mutation.
        unsafe {
            let ei = &mut *self.cell.get();
            ei.readonly_descrs_fields = readonly_descrs_fields;
            ei.write_descrs_fields = write_descrs_fields;
            ei.readonly_descrs_arrays = readonly_descrs_arrays;
            ei.write_descrs_arrays = write_descrs_arrays;
            ei.readonly_descrs_interiorfields = readonly_descrs_interiorfields;
            ei.write_descrs_interiorfields = write_descrs_interiorfields;
        }
    }
}

impl Clone for EffectInfoCell {
    fn clone(&self) -> Self {
        // Cloning observes the current EI state; this is what
        // existing test fixtures need.
        Self::new(self.get().clone())
    }
}

/// effectinfo.py:266-269: frozenset_or_none(x)
///
/// `frozenset(x)` if `x is not None`, else `None`. Pyre's lift of
/// `frozenset[Descr]` is `Vec<DescrRef>` sorted+deduped by
/// `Arc::as_ptr` (object identity, matching PyPy's
/// `id(descr)`-keyed frozenset).  Every consumer
/// (`PartialEq`, `Hash`, `EiCanonKey`, `EffectInfoKey`,
/// `compute_bitstrings`) walks the raw set expecting this
/// canonical Arc-pointer-sorted shape — without it, two
/// structurally-identical sets would compare unequal.
pub fn frozenset_or_none(x: Option<Vec<DescrRef>>) -> Option<Vec<DescrRef>> {
    x.map(canonicalize_descr_set)
}

/// Sort by `Arc::as_ptr` thin pointer (data address only — vtable
/// dropped via `*const () as usize`) and dedup adjacent duplicates.
/// Matches PyPy's `frozenset(x)` identity semantic where two clones
/// of the same `Arc` collapse to one entry, while two distinct Arcs
/// (even of the same descr type) remain separate.
pub fn canonicalize_descr_set(mut v: Vec<DescrRef>) -> Vec<DescrRef> {
    v.sort_by_key(descr_ptr_id);
    v.dedup_by(|a, b| std::sync::Arc::ptr_eq(a, b));
    v
}

/// Project a `DescrRef` to a thin-pointer identity key — `Arc::as_ptr`
/// returns a `*const dyn Descr` (wide pointer); we strip the vtable
/// component via `*const () as usize` so the key is plain `usize`,
/// suitable for `Hash`/`Ord`/`Eq`. Two clones of the same `Arc`
/// share a data pointer; two distinct `Arc`s never do.
#[inline]
pub fn descr_ptr_id(d: &DescrRef) -> usize {
    std::sync::Arc::as_ptr(d) as *const () as usize
}

/// Element-wise Arc-identity equality on a canonicalized
/// `Option<Vec<DescrRef>>`. Both sides assumed already passed
/// through `canonicalize_descr_set` — same Arc-ptr-sorted order,
/// dedup'd. Falls back to length compare + `Arc::ptr_eq` per slot.
fn descr_set_eq(a: &Option<Vec<DescrRef>>, b: &Option<Vec<DescrRef>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(va), Some(vb)) => {
            va.len() == vb.len()
                && va
                    .iter()
                    .zip(vb.iter())
                    .all(|(x, y)| std::sync::Arc::ptr_eq(x, y))
        }
        _ => false,
    }
}

/// Hash a canonicalized `Option<Vec<DescrRef>>` by its element ptr ids.
fn descr_set_hash<H: std::hash::Hasher>(s: &Option<Vec<DescrRef>>, state: &mut H) {
    use std::hash::Hash;
    match s {
        None => 0u8.hash(state),
        Some(v) => {
            1u8.hash(state);
            v.len().hash(state);
            for d in v {
                descr_ptr_id(d).hash(state);
            }
        }
    }
}

/// effectinfo.py:380-390: consider_struct(TYPE, fieldname)
///
/// In RPython this filters out `lltype.Void` fields and non-`GcStruct`
/// types, plus the `OBJECT.typeptr` special case. Pyre lacks `lltype`
/// metadata at this layer, so the predicate is intentionally permissive
/// (the codewriter has already filtered out non-GC fields earlier).
/// TODO.
pub fn consider_struct(_type_name: &str, _fieldname: &str) -> bool {
    true
}

/// effectinfo.py:392-397: consider_array(ARRAY)
///
/// Same caveat as `consider_struct`. TODO.
pub fn consider_array(_array_name: &str) -> bool {
    true
}

/// Side effect classification for calls.
///
/// effectinfo.py:13-263: `class EffectInfo`. The seven `EF_*` constants
/// and the `OS_*` family live as associated constants on this struct
/// plus the [`ExtraEffect`] / [`OopSpecIndex`] enums for type-safe match.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectInfo {
    pub extraeffect: ExtraEffect,
    pub oopspecindex: OopSpecIndex,
    /// pyre-only: identifies a pyre custom-bytecode helper `residual_call`
    /// (`binary_op` / `compare` / `load_const` / `box_int` / `store_subscr`
    /// / `load_global`) so the full-body walker can re-emit its speculative
    /// specialization.  Held OFF `oopspecindex` on purpose: these helpers have
    /// no upstream OS_* identity, so parking them on `oopspecindex` would make
    /// `has_oopspec()` true and divert the production reghint / vstring passes
    /// (both read `has_oopspec()` behaviorally) off the ordinary-call path, and
    /// would break the `_OS_CANRAISE` invariant for the CanRaise members.
    /// `None` for every ordinary call, so `has_oopspec()` and the OS_* universe
    /// stay identical to upstream.
    pub pyre_helper: PyreHelperKind,
    // ── effectinfo.py:128-145 raw descr sets ──
    //
    // PyPy stores `_readonly_descrs_fields: frozenset[Descr]` (and the
    // five sibling sets) at `EffectInfo.__init__` time
    // (`effectinfo.py:128-145 frozenset_or_none`). `compute_bitstrings`
    // (`effectinfo.py:465-547`) later walks every EI, partitions the
    // descrs into (eisetr, eisetw) equivalence classes per category by
    // `id(descr)` (frozenset element identity), assigns
    // `descr.ei_index = class_index`, and only then encodes the
    // bitstring_* fields below.
    //
    // Pyre's lift carries `Vec<DescrRef>` (Arc identity) so the
    // partition step honours PyPy's object-identity semantic — two
    // distinct descrs with the same `descr.index()` (e.g. two structs'
    // `index_in_parent = 0` fields at `call.rs:3849`) stay separate.
    // Each Vec is sorted-deduped by `Arc::as_ptr` at construction
    // (`canonicalize_descr_set`); `None` mirrors PyPy's
    // `_readonly_descrs_fields = None` wildcard (random-effects EI).
    /// effectinfo.py:128 `_readonly_descrs_fields = frozenset_or_none(readonly_descrs_fields)`.
    #[serde(skip)]
    pub _readonly_descrs_fields: Option<Vec<DescrRef>>,
    /// effectinfo.py:131 `_write_descrs_fields`.
    #[serde(skip)]
    pub _write_descrs_fields: Option<Vec<DescrRef>>,
    /// effectinfo.py:129 `_readonly_descrs_arrays`.
    #[serde(skip)]
    pub _readonly_descrs_arrays: Option<Vec<DescrRef>>,
    /// effectinfo.py:132 `_write_descrs_arrays`.
    #[serde(skip)]
    pub _write_descrs_arrays: Option<Vec<DescrRef>>,
    /// effectinfo.py:130 `_readonly_descrs_interiorfields`.
    #[serde(skip)]
    pub _readonly_descrs_interiorfields: Option<Vec<DescrRef>>,
    /// effectinfo.py:133 `_write_descrs_interiorfields`.
    #[serde(skip)]
    pub _write_descrs_interiorfields: Option<Vec<DescrRef>>,
    /// effectinfo.py:185 bitstring_readonly_descrs_fields. `None` = wildcard
    /// (effectinfo.py:488-489 sets the bitstring to `None` for `EF_RANDOM_EFFECTS`).
    pub readonly_descrs_fields: Option<Vec<u8>>,
    /// effectinfo.py:188 bitstring_write_descrs_fields.
    pub write_descrs_fields: Option<Vec<u8>>,
    /// effectinfo.py:186 bitstring_readonly_descrs_arrays.
    pub readonly_descrs_arrays: Option<Vec<u8>>,
    /// effectinfo.py:189 bitstring_write_descrs_arrays.
    pub write_descrs_arrays: Option<Vec<u8>>,
    /// effectinfo.py:187 bitstring_readonly_descrs_interiorfields.
    /// effectinfo.py:327-340: interiorfield reads also set array read bits.
    pub readonly_descrs_interiorfields: Option<Vec<u8>>,
    /// effectinfo.py:190 bitstring_write_descrs_interiorfields.
    pub write_descrs_interiorfields: Option<Vec<u8>>,
    /// effectinfo.py: can_invalidate
    pub can_invalidate: bool,
    /// effectinfo.py:194: can_collect — whether this call can trigger GC collection.
    /// RPython: set by collect_analyzer.analyze(op, self.seen_gc).
    pub can_collect: bool,
    /// effectinfo.py:201-206: single_write_descr_array
    #[serde(skip)]
    pub single_write_descr_array: Option<DescrRef>,
    /// effectinfo.py:196: extradescrs — extra descriptors carried by oopspec helpers
    /// (LIBFFI, ARRAYCOPY/ARRAYMOVE etc.).
    #[serde(skip)]
    pub extradescrs: Option<Vec<DescrRef>>,
    /// effectinfo.py:114, 197: call_release_gil_target = (target_fn_addr, save_err)
    /// `_NO_CALL_RELEASE_GIL_TARGET = (llmemory.NULL, 0)` by default.
    pub call_release_gil_target: (u64, i32),
}

/// Manual PartialEq: single_write_descr_array is excluded (like RPython's
/// cache key which also excludes it — effectinfo.py:155-164).
///
/// PyPy `effectinfo.py:152-164` keys the EI cache on the raw frozensets
/// (`(_readonly_descrs_fields, _write_descrs_fields, …)`); the bitstrings
/// don't exist yet at construction time, so they must NOT be part of
/// equality. We honour that here: identity is on `_*_descrs_*` (and
/// extraeffect / oopspec / can_invalidate / can_collect / release_gil),
/// not on the lazily-populated `bitstring_*` fields.
impl PartialEq for EffectInfo {
    fn eq(&self, other: &Self) -> bool {
        self.extraeffect == other.extraeffect
            && self.oopspecindex == other.oopspecindex
            && self.pyre_helper == other.pyre_helper
            && descr_set_eq(
                &self._readonly_descrs_fields,
                &other._readonly_descrs_fields,
            )
            && descr_set_eq(&self._write_descrs_fields, &other._write_descrs_fields)
            && descr_set_eq(
                &self._readonly_descrs_arrays,
                &other._readonly_descrs_arrays,
            )
            && descr_set_eq(&self._write_descrs_arrays, &other._write_descrs_arrays)
            && descr_set_eq(
                &self._readonly_descrs_interiorfields,
                &other._readonly_descrs_interiorfields,
            )
            && descr_set_eq(
                &self._write_descrs_interiorfields,
                &other._write_descrs_interiorfields,
            )
            && self.can_invalidate == other.can_invalidate
            && self.can_collect == other.can_collect
            && self.call_release_gil_target == other.call_release_gil_target
    }
}

impl Eq for EffectInfo {}

/// Manual Hash matching the PartialEq subset (skips
/// `single_write_descr_array` and `extradescrs` for the same reason
/// `effectinfo.py:155-164` excludes them from the cache key). Raw
/// descr sets are hashed by element `Arc::as_ptr` ids (Arc-identity
/// lift of PyPy's `id(descr)` frozenset hash).
impl std::hash::Hash for EffectInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.extraeffect.hash(state);
        self.oopspecindex.hash(state);
        self.pyre_helper.hash(state);
        descr_set_hash(&self._readonly_descrs_fields, state);
        descr_set_hash(&self._write_descrs_fields, state);
        descr_set_hash(&self._readonly_descrs_arrays, state);
        descr_set_hash(&self._write_descrs_arrays, state);
        descr_set_hash(&self._readonly_descrs_interiorfields, state);
        descr_set_hash(&self._write_descrs_interiorfields, state);
        self.can_invalidate.hash(state);
        self.can_collect.hash(state);
        self.call_release_gil_target.hash(state);
    }
}

impl Default for EffectInfo {
    fn default() -> Self {
        EffectInfo {
            extraeffect: ExtraEffect::CanRaise,
            oopspecindex: OopSpecIndex::None,
            pyre_helper: PyreHelperKind::None,
            // effectinfo.py:128-145 frozenset_or_none: empty frozenset for
            // a non-random-effects EI with no field/array touches yet.
            _readonly_descrs_fields: Some(Vec::new()),
            _write_descrs_fields: Some(Vec::new()),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            // effectinfo.py:175-181: empty frozenset for elidable, but `__new__`
            // requires a non-None value for non-RandomEffects EIs. Empty Vec
            // is the bitstring equivalent (no descrs touched).
            readonly_descrs_fields: Some(Vec::new()),
            write_descrs_fields: Some(Vec::new()),
            readonly_descrs_arrays: Some(Vec::new()),
            write_descrs_arrays: Some(Vec::new()),
            readonly_descrs_interiorfields: Some(Vec::new()),
            write_descrs_interiorfields: Some(Vec::new()),
            single_write_descr_array: None,
            extradescrs: None,
            // RPython effectinfo.py:125: can_collect=True default
            can_invalidate: false,
            can_collect: true,
            // effectinfo.py:114, 123: call_release_gil_target=_NO_CALL_RELEASE_GIL_TARGET
            call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
        }
    }
}

/// effectinfo.py:17-24: `EF_*` extraeffect constants.
///
/// TODO: the Rust enum is an adaptation over RPython's bare
/// integer constants for type safety. The `repr(u8)` discriminants are
/// the upstream values verbatim so ordering comparisons (`>`, `>=`)
/// behave identically to RPython.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ExtraEffect {
    /// effectinfo.py:17 `EF_ELIDABLE_CANNOT_RAISE = 0`
    ElidableCannotRaise = 0,
    /// effectinfo.py:18 `EF_LOOPINVARIANT = 1`
    LoopInvariant = 1,
    /// effectinfo.py:19 `EF_CANNOT_RAISE = 2`
    CannotRaise = 2,
    /// effectinfo.py:20 `EF_ELIDABLE_OR_MEMORYERROR = 3`
    ElidableOrMemoryError = 3,
    /// effectinfo.py:21 `EF_ELIDABLE_CAN_RAISE = 4`
    ElidableCanRaise = 4,
    /// effectinfo.py:22 `EF_CAN_RAISE = 5`
    CanRaise = 5,
    /// effectinfo.py:23 `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE = 6`
    ForcesVirtualOrVirtualizable = 6,
    /// effectinfo.py:24 `EF_RANDOM_EFFECTS = 7`
    RandomEffects = 7,
}

/// effectinfo.py:27-105: `OS_*` oopspec index constants.
///
/// Rust enum + `repr(u16)` for type safety; values mirror upstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u16)]
pub enum OopSpecIndex {
    None = 0,
    Arraycopy = 1,
    Str2Unicode = 2,
    ShrinkArray = 3,
    DictLookup = 4,
    ThreadlocalrefGet = 5,
    NotInTrace = 8,
    Arraymove = 9,
    IntPyDiv = 12,
    IntUdiv = 13,
    IntPyMod = 14,
    IntUmod = 15,
    StrConcat = 22,
    StrSlice = 23,
    StrEqual = 24,
    StreqSliceChecknull = 25,
    StreqSliceNonnull = 26,
    StreqSliceChar = 27,
    StreqNonnull = 28,
    StreqNonnullChar = 29,
    StreqChecknullChar = 30,
    StreqLengthok = 31,
    StrCmp = 32,
    UniConcat = 42,
    UniSlice = 43,
    UniEqual = 44,
    UnieqSliceChecknull = 45,
    UnieqSliceNonnull = 46,
    UnieqSliceChar = 47,
    UnieqNonnull = 48,
    UnieqNonnullChar = 49,
    UnieqChecknullChar = 50,
    UnieqLengthok = 51,
    UniCmp = 52,
    LibffiCall = 62,
    LlongInvert = 69,
    LlongAdd = 70,
    LlongSub = 71,
    LlongMul = 72,
    LlongLt = 73,
    LlongLe = 74,
    LlongEq = 75,
    LlongNe = 76,
    LlongGt = 77,
    LlongGe = 78,
    LlongAnd = 79,
    LlongOr = 80,
    LlongLshift = 81,
    LlongRshift = 82,
    LlongXor = 83,
    LlongFromInt = 84,
    LlongToInt = 85,
    LlongFromFloat = 86,
    LlongToFloat = 87,
    LlongUlt = 88,
    LlongUle = 89,
    LlongUgt = 90,
    LlongUge = 91,
    LlongUrshift = 92,
    LlongFromUint = 93,
    LlongUToFloat = 94,
    MathSqrt = 100,
    MathReadTimestamp = 101,
    RawMallocVarsizeChar = 110,
    RawFree = 111,
    StrCopyToRaw = 112,
    UniCopyToRaw = 113,
    JitForceVirtual = 120,
    JitForceVirtualizable = 121,
}

/// pyre-only recognition tag for custom-bytecode helper `residual_call`s.
///
/// The pyre codewriter lowers several Python bytecodes to a single runtime
/// helper `residual_call`: `binary_op` / `compare` carry the operator as a
/// compact int tag; `load_const` / `load_global` re-read an indexed code
/// slot; `box_int` wraps a raw Int; `store_subscr` does `obj[key] = value`.
/// The full-body walker re-emits each one's speculative specialization
/// (e.g. `guard_class` + `getfield_gc` + `int/float_OP` + `new_with_vtable`
/// for `binary_op`), but it cannot match the helper by fnaddr because
/// pyre-jit-trace does not depend on pyre-jit — so the calldescr's
/// [`EffectInfo`] carries this tag instead.
///
/// Deliberately separate from [`OopSpecIndex`]: these helpers have no
/// upstream OS_* identity, [`EffectInfo::has_oopspec`] must stay false for
/// them so the production reghint / vstring passes treat them as ordinary
/// calls, and the CanRaise members (`load_const` / `load_global` / `box_int`)
/// would otherwise violate the `_OS_CANRAISE` invariant (effectinfo.py:198).
/// Discriminants are pyre-internal (no upstream meaning).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum PyreHelperKind {
    #[default]
    None,
    BinaryOp,
    CompareOp,
    LoadConst,
    BoxInt,
    StoreSubscr,
    LoadGlobal,
    /// `bh_load_name_fn(frame, w_name, namei)` — the LOAD_NAME frame-receiver
    /// helper (`pyopcode.py:945-955`, w_locals probe + LOAD_GLOBAL fallback).
    /// The full-body walker recognises this tag to fold a module-scope name
    /// read (frame's `w_locals_object` is null, so `w_locals` aliases
    /// `w_globals`) through the same module-dict cell fast path as
    /// [`PyreHelperKind::LoadGlobal`], eliding the per-iteration residual.
    LoadName,
    /// `bh_call_fn_N(callable, null_or_self, args...)` — the CALL-family
    /// Python-call helper.  `null_or_self` (arg index 1) is a sentinel
    /// the helper checks before use (a non-null receiver is prepended as
    /// arg0; `PY_NULL` means "no receiver" and is never dereferenced), so
    /// a concrete-NULL there is the NORMAL plain-call shape, not the
    /// broken baked-NULL-globals shape the walker's may-force NULL-ref
    /// gate rejects.
    CallFn,
    /// `bh_truth_fn(value)` — the TO_BOOL / `POP_JUMP_IF_*` truth helper
    /// (`opcode_ops::truth_value`).  Returns the operand's truthiness as a
    /// raw Int.  The full-body walker recognises this tag to specialise a
    /// provably-int operand to a pure `guard_class INT` + `getfield intval`
    /// + `int_is_true`, eliding the `CALL_MAY_FORCE` (and its
    /// `GUARD_NOT_FORCED` / `GUARD_NO_EXCEPTION`) the generic residual emits.
    Truth,
    /// `bh_unpack_sequence_fn(count, seq)` — the UNPACK_SEQUENCE validator
    /// emitted by the codewriter UNPACK_SEQUENCE arm.  Validates the exact
    /// length and returns the normalized tuple.  The full-body walker
    /// recognises this tag to fold an arity-2 specialised int tuple to a
    /// `guard_class` + pass-through, eliding the opaque residual so the
    /// partner [`PyreHelperKind::UnpackItem`] reads fold against it.
    UnpackSequence,
    /// `bh_unpack_item_fn(index, tuple)` — the per-index UNPACK_SEQUENCE item
    /// reader.  The full-body walker recognises this tag to read `value0` /
    /// `value1` from a specialised int tuple directly (`getfield_gc_pure_i`
    /// + `wrapint`), so the unpacked items stay unboxed ints through the
    /// downstream BINARY_OP int fold.
    UnpackItem,
    /// `bh_newtuple_from_array(array)` — the BUILD_TUPLE array consumer that
    /// `lower_tuple_build_hlop_to_insn` emits after `new_array_clear` +
    /// `setarrayitem_gc`.  The full-body walker recognises this tag to
    /// virtualize an arity-2 plain-int tuple as a `spec_ii`
    /// `new_with_vtable` + `value0` / `value1` (mirroring
    /// `trace_build_tuple_value`), so the backing array build and the
    /// partner [`PyreHelperKind::UnpackItem`] reads DCE to a pure-int loop.
    NewtupleFromArray,
    /// `n_varargs_fn(frame, exc, cause)` — the RAISE-family residual the
    /// codewriter emits for `n argc>=1` (`build_n_varargs_fn_residual_call_r_r_insn`).
    /// `cause` (the trailing Ref arg) is a `PY_NULL` sentinel for `raise X`
    /// without `from` (checked, never dereferenced when null), so a concrete
    /// NULL there is the normal shape — not the broken baked-NULL shape the
    /// walker's may-force NULL-ref gate rejects.  The full-body walker
    /// recognises this tag (gated `PYRE_FBW_RAISE`) to exempt the trailing
    /// NULL `cause` in both twin NULL-ref guards so the FBW path can own the
    /// raise instead of declining to the trait.
    RaiseVarargs,
    /// `get_current_exception()` — the PUSH_EXC_INFO `prev = ec.sys_exc_value`
    /// save residual (`() → Ref`, TLS read via `cpu.get_current_exception_fn`).
    /// The full-body walker recognises this tag (gated `PYRE_FBW_RAISE`) to
    /// lower it to `GETFIELD_GC_R(ec, sys_exc_value)` so the exc-info save
    /// participates in the balanced save/restore the heap optimizer
    /// dead-store-eliminates, letting a non-escaping exception stay virtual.
    GetCurrentException,
    /// `set_current_exception(exc)` — the PUSH_EXC_INFO store and the
    /// POP_EXCEPT restore residual (`(exc: Ref) → Void`, TLS write via
    /// `cpu.set_current_exception_fn`).  The full-body walker recognises this
    /// tag (gated `PYRE_FBW_RAISE`) to lower it to `SETFIELD_GC(ec, exc,
    /// sys_exc_value)`; paired with the [`GetCurrentException`] save on the
    /// same descr-identity field, a balanced never-read save/restore is
    /// dead-store-eliminated so the virtual exception de-escapes and DCEs.
    SetCurrentException,
    /// `for_iter_next(iter)` — the FOR_ITER advance residual the codewriter
    /// emits for the continue arm (`jit_next` via `cpu.for_iter_next_fn`).
    /// It advances the real shared heap iterator concretely during the
    /// authoritative walk and returns the consumed item (`Ref`, or null at
    /// exhaustion).  The full-body walker recognises this tag to stash the
    /// consumed item + the FOR_ITER body pc so an aborting walk can DELIVER
    /// the in-flight iteration to the live frame instead of dropping it (the
    /// iterator advance is an irreversible side effect with no journal undo).
    ForIterNext,
    /// `store_deref_value(cell, value)` — the STORE_DEREF residual
    /// (`bh_store_deref_value_fn` via `cpu.store_deref_value_fn`).  It mutates
    /// the cell's contents in place and RETURNS the slot value (`Ref`), so it
    /// is a value-returning heap write the #57 Option C body-effect guard's
    /// `Void`-result write proxy cannot see; the tag lets that guard flag it
    /// (Finding #1) so an aborting FOR_ITER walk refuses delivery rather than
    /// re-running the body and doubling the cell write.
    StoreDeref,
    /// `jit_list_append(list, value)` — the LIST_APPEND residual the
    /// codewriter emits for a comprehension append (`[f(x) for x in xs]`
    /// inlines LIST_APPEND into the enclosing function).  The two Ref
    /// operands are the peeked target list and the popped value; the result
    /// is void.  The full-body walker recognises this tag to fold the append
    /// through the same orthodox `w_list_append` descent as the method-call
    /// form ([`PyreHelperKind::CallFn`] `lst.append(x)`), recording the
    /// native array store instead of the opaque `jit_list_append` residual.
    /// The recognition has no bound-method callable to pin: the list and
    /// value arrive directly as the residual's Ref operands.
    ListAppendValue,
}

impl EffectInfo {
    pub fn check_is_elidable(&self) -> bool {
        self.extraeffect == ExtraEffect::ElidableCanRaise
            || self.extraeffect == ExtraEffect::ElidableOrMemoryError
            || self.extraeffect == ExtraEffect::ElidableCannotRaise
    }

    pub fn check_can_invalidate(&self) -> bool {
        self.can_invalidate
    }

    pub fn check_can_collect(&self) -> bool {
        self.can_collect
    }

    pub fn check_forces_virtual_or_virtualizable(&self) -> bool {
        self.extraeffect >= ExtraEffect::ForcesVirtualOrVirtualizable
    }

    /// `effectinfo.py:252-253` `has_random_effects(self)`:
    ///
    /// ```python
    /// def has_random_effects(self):
    ///     return self.extraeffect >= self.EF_RANDOM_EFFECTS
    /// ```
    ///
    /// Uses `>=` (not `==`).  `EF_RANDOM_EFFECTS = 7` is the highest
    /// ExtraEffect numeric value today, so the two operators pick the
    /// same set; the structural parity matters because future upstream
    /// additions that slot a value after `EF_RANDOM_EFFECTS` would need
    /// this predicate to remain inclusive.
    pub fn has_random_effects(&self) -> bool {
        self.extraeffect >= ExtraEffect::RandomEffects
    }

    /// Whether the oopspec identifies a special-cased operation.
    pub fn has_oopspec(&self) -> bool {
        self.oopspecindex != OopSpecIndex::None
    }

    /// effectinfo.py:232-236: check_can_raise(ignore_memoryerror)
    pub fn check_can_raise(&self, ignore_memoryerror: bool) -> bool {
        if ignore_memoryerror {
            self.extraeffect > ExtraEffect::ElidableOrMemoryError
        } else {
            self.extraeffect > ExtraEffect::CannotRaise
        }
    }

    /// effectinfo.py:255-257: is_call_release_gil()
    /// `tgt_func, tgt_saveerr = self.call_release_gil_target; return bool(tgt_func)`
    pub fn is_call_release_gil(&self) -> bool {
        let (tgt_func, _tgt_saveerr) = self.call_release_gil_target;
        tgt_func != 0
    }

    /// Const-compatible constructor for static initialization.
    ///
    /// Empty bitstrings (`Some(Vec::new())`) match `effectinfo.py:175-181`
    /// for elidable EIs whose `_write_descrs_*` collapse to `frozenset()`,
    /// later compiled to a zero-length bitstring by `compute_bitstrings`.
    pub const fn const_new(extraeffect: ExtraEffect, oopspecindex: OopSpecIndex) -> Self {
        EffectInfo {
            extraeffect,
            oopspecindex,
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
            single_write_descr_array: None,
            extradescrs: None,
            can_invalidate: false,
            can_collect: true,
            call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
        }
    }

    /// Create a new EffectInfo with the given effect and oopspec.
    pub fn new(extraeffect: ExtraEffect, oopspecindex: OopSpecIndex) -> Self {
        EffectInfo {
            extraeffect,
            oopspecindex,
            ..Default::default()
        }
    }

    /// effectinfo.py:64: _OS_offset_uni = OS_UNI_CONCAT - OS_STR_CONCAT
    pub const _OS_OFFSET_UNI: u16 = OopSpecIndex::UniConcat as u16 - OopSpecIndex::StrConcat as u16;

    /// effectinfo.py:114: _NO_CALL_RELEASE_GIL_TARGET = (llmemory.NULL, 0)
    pub const _NO_CALL_RELEASE_GIL_TARGET: (u64, i32) = (0, 0);

    /// effectinfo.py:108-112: _OS_CANRAISE — oopspecindex values that can raise
    /// even when extraeffect <= EF_CANNOT_RAISE.
    pub fn _is_os_canraise(idx: OopSpecIndex) -> bool {
        matches!(
            idx,
            OopSpecIndex::None
                | OopSpecIndex::Str2Unicode
                | OopSpecIndex::LibffiCall
                | OopSpecIndex::RawMallocVarsizeChar
                | OopSpecIndex::JitForceVirtual
                | OopSpecIndex::ShrinkArray
                | OopSpecIndex::DictLookup
                | OopSpecIndex::NotInTrace
        )
    }

    /// effectinfo.py:271-273: MOST_GENERAL
    /// `EffectInfo(None, None, None, None, None, None, EF_RANDOM_EFFECTS, can_invalidate=True)`.
    ///
    /// `effectinfo.py:149-155` + `compute_bitstrings` line 488-489 keep
    /// the bitstrings as `None` for `EF_RANDOM_EFFECTS`; the optimizer's
    /// `has_random_effects()` guard (heap.py:460 / heap.rs:2602) prevents
    /// `check_*_descr_*` from being called on the wildcard.
    pub const MOST_GENERAL: EffectInfo = EffectInfo {
        extraeffect: ExtraEffect::RandomEffects,
        oopspecindex: OopSpecIndex::None,
        pyre_helper: PyreHelperKind::None,
        _readonly_descrs_fields: None,
        _write_descrs_fields: None,
        _readonly_descrs_arrays: None,
        _write_descrs_arrays: None,
        _readonly_descrs_interiorfields: None,
        _write_descrs_interiorfields: None,
        readonly_descrs_fields: None,
        write_descrs_fields: None,
        readonly_descrs_arrays: None,
        write_descrs_arrays: None,
        readonly_descrs_interiorfields: None,
        write_descrs_interiorfields: None,
        single_write_descr_array: None,
        extradescrs: None,
        can_invalidate: true,
        can_collect: true,
        call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
    };

    // ── Bitstring check methods (effectinfo.py:211-230 parity) ──
    //
    // PyPy `effectinfo.py:211-230` does NOT short-circuit on
    // `EF_RANDOM_EFFECTS`: each `check_*` is a one-liner
    // `bitstring.bitcheck(self.bitstring_*, ei_index)`.  The
    // `EF_RANDOM_EFFECTS` case is handled at construction time —
    // `effectinfo.py:149-156` keeps the `_readonly_*`/`_write_*` sets as
    // `None` and `compute_bitstrings` (`effectinfo.py:484-489`) sets the
    // bitstring fields to `None` too.  The contract is that callers must
    // gate via `has_random_effects()` first (heap.py:460) so the
    // `None`-bitstring case is never queried.
    //
    // Pyre `MOST_GENERAL` (line 380) populates every bitset with `None`,
    // matching `effectinfo.py:271-273 MOST_GENERAL`.  The optimizer
    // caller (`heap.rs:2589 call_has_random_effects`) gates the same
    // way as PyPy, so the `None`-bitstring case must never be queried;
    // the helpers below `expect()` the bitstring rather than silently
    // returning `false`, mirroring `bitstring.bitcheck(None, ...)`'s
    // RPython fail-fast (`TypeError: object of type 'NoneType' has no
    // len()`).

    /// effectinfo.py:211-213: check_readonly_descr_field(fielddescr)
    pub fn check_readonly_descr_field(&self, descr_idx: u32) -> bool {
        crate::bitstring::bitcheck(
            self.readonly_descrs_fields
                .as_deref()
                .expect("check_readonly_descr_field on EF_RANDOM_EFFECTS — caller must gate via has_random_effects()"),
            descr_idx,
        )
    }

    /// effectinfo.py:214-216: check_write_descr_field(fielddescr)
    pub fn check_write_descr_field(&self, descr_idx: u32) -> bool {
        crate::bitstring::bitcheck(
            self.write_descrs_fields
                .as_deref()
                .expect("check_write_descr_field on EF_RANDOM_EFFECTS — caller must gate via has_random_effects()"),
            descr_idx,
        )
    }

    /// Raw-set fallback for the macro / `JitDriver` path, where
    /// `compute_bitstrings` never runs: the bitstring `write_descrs_fields`
    /// stays empty and a field descr's `ei_index` stays at the `u32::MAX`
    /// sentinel, so every `check_write_descr_field` lookup misses.  Returns
    /// true iff this EI's concrete `_write_descrs_fields` names `descr` by
    /// pointer identity — the same identity `compute_bitstrings` keys on.
    /// `OptHeap::force_from_effectinfo` consults this only when
    /// `compute_bitstrings_has_run()` is false, leaving the translated
    /// (rustpython) bitstring path untouched.
    pub fn writes_field_descr_by_identity(&self, descr: &DescrRef) -> bool {
        match self._write_descrs_fields.as_deref() {
            Some(fields) => {
                let target = descr_ptr_id(descr);
                fields.iter().any(|d| descr_ptr_id(d) == target)
            }
            None => false,
        }
    }

    /// effectinfo.py:217-219: check_readonly_descr_array(arraydescr)
    pub fn check_readonly_descr_array(&self, descr_idx: u32) -> bool {
        crate::bitstring::bitcheck(
            self.readonly_descrs_arrays
                .as_deref()
                .expect("check_readonly_descr_array on EF_RANDOM_EFFECTS — caller must gate via has_random_effects()"),
            descr_idx,
        )
    }

    /// effectinfo.py:220-222: check_write_descr_array(arraydescr)
    pub fn check_write_descr_array(&self, descr_idx: u32) -> bool {
        crate::bitstring::bitcheck(
            self.write_descrs_arrays
                .as_deref()
                .expect("check_write_descr_array on EF_RANDOM_EFFECTS — caller must gate via has_random_effects()"),
            descr_idx,
        )
    }

    /// effectinfo.py:223-226: check_readonly_descr_interiorfield (NOTE: not used so far)
    pub fn check_readonly_descr_interiorfield(&self, descr_idx: u32) -> bool {
        crate::bitstring::bitcheck(
            self.readonly_descrs_interiorfields
                .as_deref()
                .expect("check_readonly_descr_interiorfield on EF_RANDOM_EFFECTS — caller must gate via has_random_effects()"),
            descr_idx,
        )
    }

    /// effectinfo.py:227-230: check_write_descr_interiorfield (NOTE: not used so far)
    pub fn check_write_descr_interiorfield(&self, descr_idx: u32) -> bool {
        crate::bitstring::bitcheck(
            self.write_descrs_interiorfields
                .as_deref()
                .expect("check_write_descr_interiorfield on EF_RANDOM_EFFECTS — caller must gate via has_random_effects()"),
            descr_idx,
        )
    }

    /// effectinfo.py:201-206: set single_write_descr_array.
    ///
    /// Builder: attaches the actual array DescrRef for ARRAYCOPY/ARRAYMOVE
    /// unrolling. RPython sets this in `EffectInfo.__new__()` when
    /// `_write_descrs_arrays` has exactly one element.
    pub fn with_single_write_descr_array(mut self, descr: DescrRef) -> Self {
        self.single_write_descr_array = Some(descr);
        self
    }

    /// effectinfo.py:201-206: auto-set single_write_descr_array.
    ///
    /// RPython sets this when `_write_descrs_arrays` has exactly one
    /// element. Pyre's bitstring counterpart counts the set bits.
    pub fn set_single_write_descr_array(&mut self, descr: DescrRef) {
        let count: u32 = match &self.write_descrs_arrays {
            Some(bs) => bs.iter().map(|b| b.count_ones()).sum(),
            None => 0,
        };
        if count == 1 {
            self.single_write_descr_array = Some(descr);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// effectinfo.py:422-461: CallInfoCollection
// ════════════════════════════════════════════════════════════════════════

/// effectinfo.py:422: `class CallInfoCollection(object)`.
///
/// Maps oopspec indices to `(calldescr, func_as_int)` pairs. Used to
/// look up the implementation of special-cased operations (arraycopy,
/// string ops, etc.).
#[derive(Debug, Clone, Default)]
pub struct CallInfoCollection {
    /// effectinfo.py:425: `_callinfo_for_oopspec` — `{oopspecindex: (calldescr, func_as_int)}`
    entries: std::collections::HashMap<OopSpecIndex, (DescrRef, u64)>,
    /// majit extension: func_as_int → function name.
    /// RPython derives names from `func.ptr._obj._name` at `see_raw_object` time.
    /// Since majit has no function pointers, we store the name at `add()` time.
    func_names: std::collections::HashMap<u64, String>,
}

impl CallInfoCollection {
    pub fn new() -> Self {
        Self::default()
    }

    /// effectinfo.py:430-431: add(oopspecindex, calldescr, func_as_int)
    pub fn add(&mut self, oopspec: OopSpecIndex, calldescr: DescrRef, func_addr: u64) {
        self.entries.insert(oopspec, (calldescr, func_addr));
    }

    /// Register the name for a function address.
    /// RPython: the name is `func.ptr._obj._name`, extracted by `see_raw_object`.
    /// In majit, we must store it explicitly since we have no pointer linkage.
    pub fn register_func_name(&mut self, func_addr: u64, name: String) {
        self.func_names.insert(func_addr, name);
    }

    /// effectinfo.py:433-434: has_oopspec(oopspecindex)
    pub fn has_oopspec(&self, oopspec: OopSpecIndex) -> bool {
        self.entries.contains_key(&oopspec)
    }

    /// effectinfo.py:439-447: callinfo_for_oopspec(oopspecindex)
    /// Returns (calldescr, func_as_int) for the oopspec, or `(None, 0)` on a
    /// miss (KeyError) — the calldescr is optional, the func defaults to 0.
    pub fn callinfo_for_oopspec(&self, oopspec: OopSpecIndex) -> (Option<&DescrRef>, u64) {
        match self.entries.get(&oopspec) {
            Some((calldescr, func)) => (Some(calldescr), *func),
            None => (None, 0),
        }
    }

    /// effectinfo.py:436-437: all_function_addresses_as_int()
    pub fn all_function_addresses_as_int(&self) -> Vec<u64> {
        self.entries.values().map(|(_, addr)| *addr).collect()
    }

    /// Look up function name by address.
    /// RPython: `see_raw_object(func.ptr)` derives name from `func.ptr._obj._name`.
    pub fn func_name(&self, addr: u64) -> Option<&str> {
        self.func_names.get(&addr).map(String::as_str)
    }
}

// ════════════════════════════════════════════════════════════════════════
// effectinfo.py:465-547 `compute_bitstrings(all_descrs)`.
/// `effectinfo.py:182-184` "no new EffectInfo after compute_bitstrings"
/// invariant — flipped to `true` on the first
/// `MetaInterpStaticData::finish_setup_descrs` call.  Late call-descr
/// minters consult this through
/// [`compute_bitstrings_has_run()`] and refuse to introduce a new
/// `EffectInfo` whose raw `_*_descrs_*` sets are non-trivial (any of
/// the six `Some(non-empty Vec<u32>)`); a non-trivial raw set after
/// the bitstring compaction would shift the (eisetr, eisetw) class
/// partition without re-running `compute_bitstrings`, leaving every
/// existing EI's bitstring stale.
///
/// PyPy's analog is the `Ellipsis` sentinel at
/// `effectinfo.py:185-190` plus the implicit lifecycle ordering
/// (codewriter mints all EIs before `compute_bitstrings`).  Pyre's
/// architecture allows trace-time mints, so the invariant is enforced
/// at the construction site instead of via post-hoc bitcheck failure.
static COMPUTE_BITSTRINGS_RAN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Read accessor used by `make_call_descr_with_effect` to gate
/// post-setup mints.  Returns `true` iff
/// `MetaInterpStaticData::finish_setup_descrs` has executed.
pub fn compute_bitstrings_has_run() -> bool {
    COMPUTE_BITSTRINGS_RAN.load(std::sync::atomic::Ordering::Acquire)
}

/// Setter — invoked by `finish_setup_descrs` after `compute_bitstrings`
/// returns.  Idempotent; subsequent calls keep the flag set.
pub fn mark_compute_bitstrings_ran() {
    COMPUTE_BITSTRINGS_RAN.store(true, std::sync::atomic::Ordering::Release);
}

impl EffectInfo {
    /// `effectinfo.py:149-162` invariant probe.  An EI carries a
    /// non-trivial raw descr set iff at least one of the six
    /// `_*_descrs_*` slots is `Some(non-empty)`.  Trivial shapes
    /// (`None` ⇒ random-effects, `Some(empty)` ⇒ analyzer-confirmed
    /// no-heap) leave every existing EI's bitstring valid because
    /// `compute_bitstrings` would map an empty raw set to an empty
    /// bitstring without disturbing other EIs' (eisetr, eisetw)
    /// classes.
    pub fn has_non_trivial_raw_set(&self) -> bool {
        let any_non_empty = |s: &Option<Vec<DescrRef>>| s.as_ref().is_some_and(|v| !v.is_empty());
        any_non_empty(&self._readonly_descrs_fields)
            || any_non_empty(&self._write_descrs_fields)
            || any_non_empty(&self._readonly_descrs_arrays)
            || any_non_empty(&self._write_descrs_arrays)
            || any_non_empty(&self._readonly_descrs_interiorfields)
            || any_non_empty(&self._write_descrs_interiorfields)
    }
}

/// `effectinfo.py:147-148 EffectInfo._cache` parity key.
///
/// PyPy keys `EffectInfo._cache` (the EI factory cache) on the tuple
/// `(readonly_descrs_fields, readonly_descrs_arrays,
///   readonly_descrs_interiorfields, write_descrs_fields,
///   write_descrs_arrays, write_descrs_interiorfields, extraeffect,
///   oopspecindex, can_invalidate, can_collect)` plus an `object()`
/// breaker for `call_release_gil_target != 0` (line 144-146).
/// `effectinfo.py:511-512 frozenset(eisetr)` then collapses the
/// per-descr (eisetr, eisetw) lists by `id(ei)`, which is identical
/// to the cache key in PyPy's runtime because the cache hit returns
/// the same EI instance.
///
/// Pyre lacks a process-global `EffectInfo._cache` today, so we
/// reconstruct the key here for `compute_bitstrings`'s internal
/// canonicalisation. Two structurally-equal `EffectInfo` values
/// produce equal `EiCanonKey`s; `compute_bitstrings` then merges
/// them into one canonical id, matching PyPy's post-frozenset class
/// identity. Release-gil targets get a unique counter (mirroring
/// PyPy's `object()` sentinel) so they remain distinct even when
/// the rest of the EI matches.
/// `effectinfo.py:144-146` release-gil cache-breaker.
///
/// PyPy: `if tgt_func: key += (object(),)` — every release-gil EI
/// gets a fresh `object()` so its `_cache` key is unique, making
/// each release-gil EI its own object identity even when
/// `(tgt_func, tgt_saveerr)` happens to match a previously-cached
/// instance. This shows up downstream in `effectinfo.py:511-512
/// frozenset(eisetr)` where `id(ei)` distinguishes release-gil EIs.
///
/// Pyre's lift: `NoTarget` for non-release-gil EIs (structural dedup
/// applies); `Unique(u64)` from a process-monotonic counter for
/// release-gil EIs (each `from_effect_info` call gets a fresh id,
/// matching PyPy's `object()` semantic — same release-gil EI cloned
/// twice still produces two distinct canonical keys, so
/// `compute_bitstrings`' canonical-id dedup never collapses
/// release-gil EIs across distinct mint sites).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ReleaseGilCacheKey {
    NoTarget,
    Unique(u64),
}

static NEXT_RELEASE_GIL_KEY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// `effectinfo.py:147-148` `EffectInfo._cache` post-frozenset key
/// canonicalization for compute_bitstrings's per-EI dedup pass.
///
/// Each raw set is projected to its sorted `Vec<usize>` of
/// `Arc::as_ptr` thin-pointer ids — the Rust lift of PyPy's
/// `frozenset[id(descr)]`.  Two structurally-identical EIs (same
/// extraeffect/oopspec/release_gil + same set of descr Arcs) hash
/// equal; two EIs whose raw sets differ in any descr Arc identity
/// hash unequal.  Conversion via `descr_set_to_ptr_set` deepens
/// `canonicalize_descr_set` (already sorted/dedup) by stripping the
/// vtable so the key is plain `Vec<usize>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EiCanonKey {
    extraeffect: ExtraEffect,
    oopspecindex: OopSpecIndex,
    readonly_descrs_fields: Option<Vec<usize>>,
    write_descrs_fields: Option<Vec<usize>>,
    readonly_descrs_arrays: Option<Vec<usize>>,
    write_descrs_arrays: Option<Vec<usize>>,
    readonly_descrs_interiorfields: Option<Vec<usize>>,
    write_descrs_interiorfields: Option<Vec<usize>>,
    can_invalidate: bool,
    can_collect: bool,
    /// `effectinfo.py:144-146` release-gil target breaker. PyPy
    /// inserts a fresh `object()` per release-gil EI; pyre's lift
    /// matches via [`ReleaseGilCacheKey::Unique`] from a process-
    /// monotonic counter.
    call_release_gil: ReleaseGilCacheKey,
}

/// Project an Arc-identity raw set to a thin-pointer ptr-id Vec.
/// Caller's `Vec<DescrRef>` is assumed already canonicalised
/// (sorted by `Arc::as_ptr`, dedup'd) — the resulting `Vec<usize>`
/// inherits that ordering.
fn descr_set_to_ptr_set(s: &Option<Vec<DescrRef>>) -> Option<Vec<usize>> {
    s.as_ref()
        .map(|v| v.iter().map(descr_ptr_id).collect::<Vec<_>>())
}

/// Public re-export of [`descr_set_to_ptr_set`] for cross-module
/// use — `LLType::func_key` builds its cache-key ptr-id Vecs by
/// projecting the EI raw sets through this helper.
pub fn descr_set_to_ptr_set_pub(s: &Option<Vec<DescrRef>>) -> Option<Vec<usize>> {
    descr_set_to_ptr_set(s)
}

impl EiCanonKey {
    fn from_effect_info(ei: &EffectInfo) -> Self {
        let call_release_gil = if ei.call_release_gil_target.0 != 0 {
            ReleaseGilCacheKey::Unique(
                NEXT_RELEASE_GIL_KEY.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            )
        } else {
            ReleaseGilCacheKey::NoTarget
        };
        Self {
            extraeffect: ei.extraeffect,
            oopspecindex: ei.oopspecindex,
            readonly_descrs_fields: descr_set_to_ptr_set(&ei._readonly_descrs_fields),
            write_descrs_fields: descr_set_to_ptr_set(&ei._write_descrs_fields),
            readonly_descrs_arrays: descr_set_to_ptr_set(&ei._readonly_descrs_arrays),
            write_descrs_arrays: descr_set_to_ptr_set(&ei._write_descrs_arrays),
            readonly_descrs_interiorfields: descr_set_to_ptr_set(
                &ei._readonly_descrs_interiorfields,
            ),
            write_descrs_interiorfields: descr_set_to_ptr_set(&ei._write_descrs_interiorfields),
            can_invalidate: ei.can_invalidate,
            can_collect: ei.can_collect,
            call_release_gil,
        }
    }
}

/// `effectinfo.py:465-547` `compute_bitstrings`.
///
/// Walks every `EffectInfo` once: for the three categories
/// (`fields`/`arrays`/`interiorfields`), partitions the descrs that
/// appear in some EI's `_readonly_descrs_*` / `_write_descrs_*` set
/// into (eisetr, eisetw) equivalence classes — descrs whose membership
/// pattern across all EIs is identical share an `ei_index`. The
/// bitstrings on each EI are then encoded with the descr-side
/// `descr.get_ei_index()` rather than the global descr index, which is
/// what makes upstream's "4000+ descrs → 373 ei classes" compaction
/// possible.
///
/// Inputs:
/// - `all_descrs`: every descriptor that participates in any EI's
///   readonly/write set (typically: every field/array/interiorfield
///   descr the codewriter has minted by this point). Descrs not in any
///   EI keep their `ei_index = u32::MAX` sentinel; PyPy
///   `effectinfo.py:496` writes the same `descr.ei_index = sys.maxint`
///   default for the no-EI branch.
/// - `all_eis`: the full set of EI handles to compact. EIs whose
///   `_readonly_descrs_fields` is `None` (random-effects /
///   `MOST_GENERAL`) follow PyPy's `effectinfo.py:485-489` rule:
///   their bitstring fields are also forced to `None`, the call site
///   is expected to gate via `has_random_effects()`.
///
/// Effect: writes to `descr.set_ei_index(...)` (interior atomic) for
/// descrs reachable through `all_descrs` AND rewrites every
/// `EffectInfo.bitstring_*` field. Idempotent — calling it twice on
/// the same input is a no-op.
///
/// PyPy `effectinfo.py:526 descr.ei_index = …` stamps the per-descr
/// `ei_index` directly on the descriptor object; pyre matches via the
/// `Descr::set_ei_index` atomic. Readers (`heap.rs::field_effect_index`
/// etc.) then resolve through `descr.get_ei_index()` alone — no
/// process-global side table.
pub fn compute_bitstrings(all_descrs: &[DescrRef], all_eis: &mut [&mut EffectInfo]) {
    use std::collections::HashMap;

    // `effectinfo.py:479-496` `for descr in all_descrs:` pre-loop —
    // every non-call descr's `ei_index` is initialised to
    // `sys.maxint` BEFORE any bitstring classification.  Pyre lifts
    // this so re-running `compute_bitstrings` over an `all_descrs`
    // that already received per-descr stamps from a prior invocation
    // does not leak stale `ei_index` values into the post-classification
    // raw-set probes (`heap.rs::field_effect_index` reads
    // `descr.get_ei_index()` directly with no side-table fallback).
    for descr in all_descrs {
        if descr.as_call_descr().is_none() {
            descr.set_ei_index(u32::MAX);
        }
    }

    // effectinfo.py:484-489: random-effects EIs zero out bitstrings.
    // The matching invariant — when `_readonly_descrs_fields` is None,
    // every sibling raw set is None too — is asserted line-for-line.
    //
    // Same loop also enforces the *frozenset-or-none* canonical form
    // (sorted-by-`Arc::as_ptr` + dedup'd by `Arc::ptr_eq`) on every
    // non-None raw set.  Caller sites usually pre-canonicalise via
    // `canonicalize_descr_set`, but routing every set through here
    // is the self-enforcing equivalent of `effectinfo.py:128-145`'s
    // `frozenset_or_none(...)` invariant — once `compute_bitstrings`
    // returns, the `_*_descrs_*` Vecs match PyPy's frozenset semantic.
    fn canonicalize(set: &mut Option<Vec<DescrRef>>) {
        if let Some(v) = set.take() {
            *set = Some(canonicalize_descr_set(v));
        }
    }
    for ei in all_eis.iter_mut() {
        if ei._readonly_descrs_fields.is_none() {
            assert!(ei._write_descrs_fields.is_none());
            assert!(ei._readonly_descrs_arrays.is_none());
            assert!(ei._write_descrs_arrays.is_none());
            assert!(ei._readonly_descrs_interiorfields.is_none());
            assert!(ei._write_descrs_interiorfields.is_none());
            ei.readonly_descrs_fields = None;
            ei.write_descrs_fields = None;
            ei.readonly_descrs_arrays = None;
            ei.write_descrs_arrays = None;
            ei.readonly_descrs_interiorfields = None;
            ei.write_descrs_interiorfields = None;
        } else {
            canonicalize(&mut ei._readonly_descrs_fields);
            canonicalize(&mut ei._write_descrs_fields);
            canonicalize(&mut ei._readonly_descrs_arrays);
            canonicalize(&mut ei._write_descrs_arrays);
            canonicalize(&mut ei._readonly_descrs_interiorfields);
            canonicalize(&mut ei._write_descrs_interiorfields);
        }
    }

    // Per-category descr-Arc enumeration.
    //
    // `effectinfo.py:493-495`:
    // ```python
    // descrs = {'fields': set(), 'arrays': set(),
    //           'interiorfields': set()}
    // for ei in all_eis:
    //     descrs['fields'].update(ei._readonly_descrs_fields)
    //     ...
    // ```
    // Python's `set.update` keys on `id(descr)`, so the partition is
    // intrinsically collision-free by object identity. Pyre's lift
    // builds the same set per category inside the loop below
    // (`category_descrs`), keyed on `Arc::as_ptr` ptr-id — direct lift
    // of Python's `id()`.
    //
    // Descrs in `all_descrs` that never enter any EI raw set keep
    // their `ei_index = u32::MAX` sentinel via the entry-loop pre-init
    // (`effectinfo.py:496` `descr.ei_index = sys.maxint`), which runs
    // unconditionally at the top of `compute_bitstrings` over every
    // non-call descr — no per-category lookup is needed here.

    // `effectinfo.py:147-148` `EffectInfo._cache` parity — PyPy keys
    // the EffectInfo factory cache on the structural tuple (raw sets,
    // extraeffect, oopspecindex, can_invalidate, can_collect,
    // call_release_gil_target) and returns the same EI instance for
    // structurally-identical requests. `effectinfo.py:511-512
    // frozenset(eisetr)` then collapses by identity (id(ei)). Pyre
    // lacks the EI._cache today (each `EffectInfoCell` is per-call-
    // descr), so structurally-identical EIs land at distinct
    // positions in `all_eis`. Canonicalise here: assign each position
    // a `canonical_id` based on first-occurrence of its structural
    // shape; identical shapes share the canonical id, matching PyPy's
    // post-frozenset (eisetr, eisetw) class identity.
    let mut canonical_id_for_position: Vec<usize> = Vec::with_capacity(all_eis.len());
    let mut first_position_for_shape: HashMap<EiCanonKey, usize> = HashMap::new();
    for ei in all_eis.iter() {
        let key = EiCanonKey::from_effect_info(ei);
        let canon = *first_position_for_shape
            .entry(key)
            .or_insert(canonical_id_for_position.len());
        canonical_id_for_position.push(canon);
    }

    // Three category iterations match `for key in descrs:` at
    // `effectinfo.py:498-540`. Each category is independent: a descr
    // is partitioned within its category, so the same `descr.ei_index`
    // bank is shared across categories — `descr.ei_index` is unique
    // per category (a fielddescr never appears in `arrays` etc.).
    for category in 0..3usize {
        // Gather every descr Arc that appears in any active EI for
        // this category (`effectinfo.py:493-494`
        // `descrs[key].update(...)`). `category_descrs` is keyed by
        // ptr-id (Arc identity) and stores `DescrRef` clones so the
        // downstream `set_ei_index` fan-out has the actual Arc.
        let mut category_descrs: HashMap<usize, DescrRef> = HashMap::new();
        for ei in all_eis.iter() {
            let (r, w) = pick_category(ei, category);
            if let Some(rs) = r {
                for d in rs {
                    category_descrs
                        .entry(descr_ptr_id(d))
                        .or_insert_with(|| d.clone());
                }
            }
            if let Some(ws) = w {
                for d in ws {
                    category_descrs
                        .entry(descr_ptr_id(d))
                        .or_insert_with(|| d.clone());
                }
            }
        }

        // For each descr in the category, compute (eisetr, eisetw) —
        // PyPy `effectinfo.py:505-512` builds these as Python lists
        // then collapses via `frozenset()` on `id(ei)`. Pyre's lift
        // pushes `canonical_id_for_position[ei_i]` (instead of the raw
        // position) and `dedup`s per descr to mirror the frozenset
        // collapse. The popularity-sort count
        // (`size_of_both_sets = len(r) + len(w)`,
        // `effectinfo.py:519-520`) thus weights each logical EI once
        // per descr, regardless of how many call descrs share its
        // structural shape.
        //
        // Membership probe is `Arc::as_ptr` ptr-id binary_search.
        // The raw sets are pre-canonicalised by ptr-id at the
        // `canonicalize_descr_set` step above so binary_search by
        // ptr-id is exact.
        let mut all_sets: Vec<(usize, DescrRef, Vec<usize>, Vec<usize>)> =
            Vec::with_capacity(category_descrs.len());
        // Fix iteration order — sort by ptr-id so the popularity-sort
        // tie-break below is deterministic across runs.
        let mut sorted_descrs: Vec<(usize, DescrRef)> = category_descrs.into_iter().collect();
        sorted_descrs.sort_by_key(|(pid, _)| *pid);
        for (pid, descr) in sorted_descrs {
            let mut eisetr: Vec<usize> = Vec::new();
            let mut eisetw: Vec<usize> = Vec::new();
            for (ei_i, ei) in all_eis.iter().enumerate() {
                let canon = canonical_id_for_position[ei_i];
                let (r, w) = pick_category(ei, category);
                if let Some(rs) = r {
                    if rs.binary_search_by_key(&pid, descr_ptr_id).is_ok() {
                        eisetr.push(canon);
                    }
                }
                if let Some(ws) = w {
                    if ws.binary_search_by_key(&pid, descr_ptr_id).is_ok() {
                        eisetw.push(canon);
                    }
                }
            }
            // `frozenset(eisetr)` semantic: dedup canonical IDs (sort
            // stays trivial for `Vec<usize>` Hash/Eq downstream).
            eisetr.sort_unstable();
            eisetr.dedup();
            eisetw.sort_unstable();
            eisetw.dedup();
            all_sets.push((pid, descr, eisetr, eisetw));
        }

        // `effectinfo.py:519-521`: heuristic — sort by len(eisetr) +
        // len(eisetw) descending so the most popular descrs claim the
        // low ei_index slots, reducing total bitstring length. Tie-
        // break on ptr-id ascending for determinism.
        all_sets.sort_by(|a, b| {
            (b.2.len() + b.3.len())
                .cmp(&(a.2.len() + a.3.len()))
                .then(a.0.cmp(&b.0))
        });

        // `effectinfo.py:523-526`: assign ei_index per (eisetr, eisetw)
        // class. `mapping.setdefault((eisetr, eisetw), len(mapping))`
        // gives each new class the next sequential index.
        let mut mapping: HashMap<(Vec<usize>, Vec<usize>), u32> = HashMap::new();
        let mut descr_to_eiindex_by_ptr: HashMap<usize, (DescrRef, u32)> = HashMap::new();
        for (pid, descr, eisetr, eisetw) in all_sets {
            let next = mapping.len() as u32;
            let ei_index = *mapping.entry((eisetr, eisetw)).or_insert(next);
            descr_to_eiindex_by_ptr.insert(pid, (descr, ei_index));
        }

        // Write `descr.ei_index = class_idx` (`effectinfo.py:526`).
        // The Arc is guaranteed to be the actual descr from the raw
        // set, so two distinct Arcs that share `descr.index()` each
        // get their own `set_ei_index` call.
        for (_pid, (descr, ei_index)) in &descr_to_eiindex_by_ptr {
            descr.set_ei_index(*ei_index);
        }

        // `effectinfo.py:496` `descr.ei_index = sys.maxint`:
        // descrs from `all_descrs` that did NOT appear in any EI raw
        // set keep their `ei_index = u32::MAX` sentinel set by the
        // entry-loop pre-init at the top of this function. Readers
        // (`heap.rs::field_effect_index`, etc.) treat MAX as
        // no-bitstring per `bitstring.py:18 if byte_number >=
        // len(bitstring)`.

        // `effectinfo.py:528-538`: encode the bitstring on each active
        // EI using `descr.get_ei_index()` per descr in the raw set.
        // Inactive (random-effects) EIs were already zeroed above.
        for ei in all_eis.iter_mut() {
            let (r_raw, w_raw) = pick_category(ei, category);
            // Skip inactive EIs.
            if r_raw.is_none() {
                continue;
            }
            let r = r_raw.unwrap();
            let w = w_raw.unwrap();
            let r_indices: Vec<u32> = r
                .iter()
                .map(|d| {
                    descr_to_eiindex_by_ptr
                        .get(&descr_ptr_id(d))
                        .map(|(_, ei_index)| *ei_index)
                        .expect("compute_bitstrings: descr in EI raw set must be in the partition")
                })
                .collect();
            let w_indices: Vec<u32> = w
                .iter()
                .map(|d| {
                    descr_to_eiindex_by_ptr
                        .get(&descr_ptr_id(d))
                        .map(|(_, ei_index)| *ei_index)
                        .expect("compute_bitstrings: descr in EI raw set must be in the partition")
                })
                .collect();
            // `effectinfo.py:533-534 assert sys.maxint not in bitstrr` —
            // every descr in an active EI's set must have been assigned
            // a finite `ei_index` above (we panic with `expect` if not).
            let r_bs = crate::bitstring::make_bitstring(&r_indices);
            let w_bs = crate::bitstring::make_bitstring(&w_indices);
            store_category_bitstrings(ei, category, Some(r_bs), Some(w_bs));
        }
    }
}

/// Helper: project `(_readonly_descrs_*, _write_descrs_*)` for a category.
fn pick_category(
    ei: &EffectInfo,
    category: usize,
) -> (Option<&Vec<DescrRef>>, Option<&Vec<DescrRef>>) {
    match category {
        0 => (
            ei._readonly_descrs_fields.as_ref(),
            ei._write_descrs_fields.as_ref(),
        ),
        1 => (
            ei._readonly_descrs_arrays.as_ref(),
            ei._write_descrs_arrays.as_ref(),
        ),
        2 => (
            ei._readonly_descrs_interiorfields.as_ref(),
            ei._write_descrs_interiorfields.as_ref(),
        ),
        _ => unreachable!("compute_bitstrings: invalid category"),
    }
}

/// Helper: install `(bitstring_readonly_*, bitstring_write_*)` for a
/// category. Mirrors `setattr(ei, 'bitstring_readonly_descrs_' + key, …)`
/// at `effectinfo.py:537-538`.
fn store_category_bitstrings(
    ei: &mut EffectInfo,
    category: usize,
    r: Option<Vec<u8>>,
    w: Option<Vec<u8>>,
) {
    match category {
        0 => {
            ei.readonly_descrs_fields = r;
            ei.write_descrs_fields = w;
        }
        1 => {
            ei.readonly_descrs_arrays = r;
            ei.write_descrs_arrays = w;
        }
        2 => {
            ei.readonly_descrs_interiorfields = r;
            ei.write_descrs_interiorfields = w;
        }
        _ => unreachable!("compute_bitstrings: invalid category"),
    }
}

#[cfg(test)]
mod compute_bitstrings_tests {
    use super::*;
    use crate::descr::{DescrRef, SimpleArrayDescr, SimpleFieldDescr};
    use crate::value::Type;
    use std::sync::Arc;

    fn mk_field(idx: u32) -> DescrRef {
        Arc::new(SimpleFieldDescr::new(idx, 0, 8, Type::Int, false)) as DescrRef
    }
    fn mk_array(idx: u32) -> DescrRef {
        Arc::new(SimpleArrayDescr::new(idx, 0, 8, 0, Type::Int)) as DescrRef
    }

    /// Two EIs share a fielddescr in their write set; the descr's
    /// `ei_index` should be the same after compute_bitstrings, and
    /// each EI's `bitstring_write_descrs_fields` should set bit
    /// `descr.get_ei_index()`.
    #[test]
    fn shared_descr_collapses_to_one_class() {
        let f0 = mk_field(0);
        let f1 = mk_field(1);
        let mut ei_a = EffectInfo {
            _write_descrs_fields: Some(vec![f0.clone(), f1.clone()]),
            _readonly_descrs_fields: Some(Vec::new()),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        let mut ei_b = EffectInfo {
            _write_descrs_fields: Some(vec![f0.clone(), f1.clone()]),
            _readonly_descrs_fields: Some(Vec::new()),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        let descrs = vec![f0.clone(), f1.clone()];
        let mut eis: Vec<&mut EffectInfo> = vec![&mut ei_a, &mut ei_b];
        compute_bitstrings(&descrs, &mut eis);

        // Both descrs appear in identical EIs (eisetr=∅, eisetw={A,B}),
        // so both go in the same class → same ei_index.
        assert_eq!(f0.get_ei_index(), f1.get_ei_index());
        assert_ne!(f0.get_ei_index(), u32::MAX);

        // Both EIs encode the same descr → bitstring with the bit at
        // `descr.get_ei_index()` set.
        let bs = ei_a.write_descrs_fields.as_ref().unwrap();
        assert!(crate::bitstring::bitcheck(bs, f0.get_ei_index()));
        assert!(crate::bitstring::bitcheck(bs, f1.get_ei_index()));
    }

    /// Two descrs with different (eisetr, eisetw) get different
    /// `ei_index`. Only one EI reads `f0`; both write `f1`.
    #[test]
    fn divergent_membership_yields_distinct_classes() {
        let f0 = mk_field(0);
        let f1 = mk_field(1);
        let mut ei_a = EffectInfo {
            _readonly_descrs_fields: Some(vec![f0.clone()]),
            _write_descrs_fields: Some(vec![f1.clone()]),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        let mut ei_b = EffectInfo {
            _readonly_descrs_fields: Some(Vec::new()),
            _write_descrs_fields: Some(vec![f1.clone()]),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        let descrs = vec![f0.clone(), f1.clone()];
        let mut eis: Vec<&mut EffectInfo> = vec![&mut ei_a, &mut ei_b];
        compute_bitstrings(&descrs, &mut eis);
        assert_ne!(f0.get_ei_index(), f1.get_ei_index());
    }

    /// `MOST_GENERAL`-style EI with `_readonly_descrs_* = None` keeps
    /// bitstrings None and asserts `effectinfo.py:485-487`.
    #[test]
    fn random_effects_zeros_bitstrings() {
        let f0 = mk_field(0);
        let mut ei_random = EffectInfo {
            _readonly_descrs_fields: None,
            _write_descrs_fields: None,
            _readonly_descrs_arrays: None,
            _write_descrs_arrays: None,
            _readonly_descrs_interiorfields: None,
            _write_descrs_interiorfields: None,
            extraeffect: ExtraEffect::RandomEffects,
            // Pre-existing bitstring should be wiped.
            readonly_descrs_fields: Some(vec![0xff]),
            write_descrs_fields: Some(vec![0xff]),
            readonly_descrs_arrays: Some(vec![0xff]),
            write_descrs_arrays: Some(vec![0xff]),
            readonly_descrs_interiorfields: Some(vec![0xff]),
            write_descrs_interiorfields: Some(vec![0xff]),
            ..Default::default()
        };
        let descrs = vec![f0.clone()];
        let mut eis: Vec<&mut EffectInfo> = vec![&mut ei_random];
        compute_bitstrings(&descrs, &mut eis);
        assert!(ei_random.readonly_descrs_fields.is_none());
        assert!(ei_random.write_descrs_fields.is_none());
        assert!(ei_random.readonly_descrs_arrays.is_none());
        assert!(ei_random.write_descrs_arrays.is_none());
        assert!(ei_random.readonly_descrs_interiorfields.is_none());
        assert!(ei_random.write_descrs_interiorfields.is_none());
        // Untouched descr keeps the sentinel.
        assert_eq!(f0.get_ei_index(), u32::MAX);
    }

    /// Field and array descrs with the same global `index()` get
    /// independent `ei_index`es because the partitioning is per-category.
    #[test]
    fn category_partitions_are_independent() {
        let f0 = mk_field(0);
        let a0 = mk_array(0);
        let mut ei = EffectInfo {
            _readonly_descrs_fields: Some(vec![f0.clone()]),
            _write_descrs_fields: Some(Vec::new()),
            _readonly_descrs_arrays: Some(vec![a0.clone()]),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        // Both descrs have global index 0 but live in different
        // partitions; both should get ei_index = 0 in their own
        // partition (the class assignment starts fresh per category).
        let descrs = vec![f0.clone(), a0.clone()];
        let mut eis: Vec<&mut EffectInfo> = vec![&mut ei];
        compute_bitstrings(&descrs, &mut eis);
        // Now that compute_bitstrings keys partitions by Arc identity,
        // the field and the array both get assigned ei_index in their
        // own categories — each should be != u32::MAX.
        assert_ne!(f0.get_ei_index(), u32::MAX);
        assert_ne!(a0.get_ei_index(), u32::MAX);
    }

    /// Two distinct field descrs that happen to share the same
    /// `descr.index()` (e.g. two structs' `index_in_parent = 0`
    /// fields per `call.rs:3849`) MUST get distinct `ei_index`
    /// values when their (eisetr, eisetw) shape diverges.  PyPy
    /// `effectinfo.py:493-526` guarantees this by `id(descr)`
    /// (object identity) keyed `frozenset`/`set` — the Arc-identity
    /// lift in pyre matches the same semantic.  Earlier versions
    /// keyed the partition on `descr.index()`, silently collapsing
    /// these two distinct descrs into one ei_index.  This test
    /// pins the parity invariant.
    #[test]
    fn distinct_arcs_share_index_get_distinct_ei_indices() {
        let f_a = mk_field(99);
        let f_b = mk_field(99);
        // Two different EIs reference different Arcs even though
        // both Arcs report `descr.index() == 99`.
        let mut ei_reads_a = EffectInfo {
            _readonly_descrs_fields: Some(vec![f_a.clone()]),
            _write_descrs_fields: Some(Vec::new()),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        let mut ei_writes_b = EffectInfo {
            _readonly_descrs_fields: Some(Vec::new()),
            _write_descrs_fields: Some(vec![f_b.clone()]),
            _readonly_descrs_arrays: Some(Vec::new()),
            _write_descrs_arrays: Some(Vec::new()),
            _readonly_descrs_interiorfields: Some(Vec::new()),
            _write_descrs_interiorfields: Some(Vec::new()),
            ..Default::default()
        };
        let descrs = vec![f_a.clone(), f_b.clone()];
        let mut eis: Vec<&mut EffectInfo> = vec![&mut ei_reads_a, &mut ei_writes_b];
        compute_bitstrings(&descrs, &mut eis);
        assert_ne!(f_a.get_ei_index(), u32::MAX);
        assert_ne!(f_b.get_ei_index(), u32::MAX);
        // f_a appears only in eisetr, f_b only in eisetw → distinct
        // (eisetr, eisetw) class → distinct ei_index.
        assert_ne!(f_a.get_ei_index(), f_b.get_ei_index());
    }
}
