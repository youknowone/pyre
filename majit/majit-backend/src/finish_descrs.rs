//! `compile.py:618-672, 1092-1099` line-by-line port of the FINISH /
//! exception descr classes shared between metainterp and backend.
//!
//! These are class-distinct `AbstractFailDescr` subclasses used by the
//! `MetaInterpStaticData.finish_setup` attachment chain and by backend
//! `find_descr_by_ptr` synthetic exit dispatch.  Kept in the backend
//! crate so backends can construct them directly without depending on
//! `majit-metainterp`; the metainterp re-exports from here for caller
//! compatibility.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use majit_ir::{Descr, FailDescr, Type};

/// `compile.py:623-624` `class _DoneWithThisFrameDescr(AbstractFailDescr):
/// final_descr = True`.
///
/// Shared base fields for the four `DoneWithThisFrame*` subclasses —
/// a stable `fail_arg_types` vector plus the `final_descr = True`
/// marker exposed through `FailDescr::is_finish()`.  Inherits the
/// `history.py:132` `_attrs_` slots `adr_jump_offset` / `rd_locs` from
/// `AbstractFailDescr`; backend codegen
/// (`assembler.py:849 patch_pending_failure_recoveries` /
/// `llsupport/assembler.py:279 guardtok.faildescr.rd_locs = positions`)
/// stamps them on every descr regardless of class.
#[derive(Debug)]
struct DoneWithThisFrameDescrBase {
    /// `history.py:122` `index = -1`.  For this descriptor family
    /// `set_descr_index` is never called (no `setup_descrs` pass); we
    /// keep the AbstractDescr default of -1.
    descr_index: AtomicI32,
    /// `handle_fail` (`compile.py:632`, 641, 650, 659) reads the result
    /// out of `deadframe[0]`.  pyre carries the same one-slot shape via
    /// `fail_arg_types`.
    fail_arg_types: Vec<Type>,
    /// `history.py:132` `AbstractFailDescr._attrs_` `adr_jump_offset`.
    /// Stamped by `assembler.py:849 patch_pending_failure_recoveries`
    /// when the recovery stub gets a final address.  `0` until stamped.
    adr_jump_offset: UnsafeCell<usize>,
    /// `history.py:132` `AbstractFailDescr._attrs_` `rd_locs`.  Written
    /// by `llsupport/assembler.py:279`.  Empty until codegen stamps it.
    rd_locs: UnsafeCell<Vec<u16>>,
}

// Safety: single-threaded JIT (RPython GIL parity).
unsafe impl Send for DoneWithThisFrameDescrBase {}
unsafe impl Sync for DoneWithThisFrameDescrBase {}

impl DoneWithThisFrameDescrBase {
    fn new(fail_arg_types: Vec<Type>) -> Self {
        Self {
            descr_index: AtomicI32::new(-1),
            fail_arg_types,
            adr_jump_offset: UnsafeCell::new(0),
            rd_locs: UnsafeCell::new(Vec::new()),
        }
    }
    fn adr_jump_offset(&self) -> usize {
        unsafe { *self.adr_jump_offset.get() }
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        unsafe { *self.adr_jump_offset.get() = offset };
    }
    fn rd_locs(&self) -> &[u16] {
        unsafe { &*self.rd_locs.get() }
    }
    fn set_rd_locs(&self, locs: Vec<u16>) {
        unsafe { *self.rd_locs.get() = locs };
    }
}

/// `compile.py:626-629` `class DoneWithThisFrameDescrVoid(_DoneWithThisFrameDescr)`.
#[derive(Debug)]
pub struct DoneWithThisFrameDescrVoid(DoneWithThisFrameDescrBase);

impl DoneWithThisFrameDescrVoid {
    pub fn new() -> Self {
        Self(DoneWithThisFrameDescrBase::new(Vec::new()))
    }
}

impl Default for DoneWithThisFrameDescrVoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Descr for DoneWithThisFrameDescrVoid {
    fn get_descr_index(&self) -> i32 {
        self.0.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.0.descr_index.store(index, Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for DoneWithThisFrameDescrVoid {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.0.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        // `compile.py:624` `final_descr = True`.
        true
    }
    fn adr_jump_offset(&self) -> usize {
        self.0.adr_jump_offset()
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        self.0.set_adr_jump_offset(offset);
    }
    fn rd_locs(&self) -> &[u16] {
        self.0.rd_locs()
    }
    fn set_rd_locs(&self, locs: Vec<u16>) {
        self.0.set_rd_locs(locs);
    }
}

/// `compile.py:631-638` `class DoneWithThisFrameDescrInt(_DoneWithThisFrameDescr)`.
#[derive(Debug)]
pub struct DoneWithThisFrameDescrInt(DoneWithThisFrameDescrBase);

impl DoneWithThisFrameDescrInt {
    pub fn new() -> Self {
        Self(DoneWithThisFrameDescrBase::new(vec![Type::Int]))
    }
}

impl Default for DoneWithThisFrameDescrInt {
    fn default() -> Self {
        Self::new()
    }
}

impl Descr for DoneWithThisFrameDescrInt {
    fn get_descr_index(&self) -> i32 {
        self.0.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.0.descr_index.store(index, Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for DoneWithThisFrameDescrInt {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.0.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        true
    }
    fn adr_jump_offset(&self) -> usize {
        self.0.adr_jump_offset()
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        self.0.set_adr_jump_offset(offset);
    }
    fn rd_locs(&self) -> &[u16] {
        self.0.rd_locs()
    }
    fn set_rd_locs(&self, locs: Vec<u16>) {
        self.0.set_rd_locs(locs);
    }
}

/// `compile.py:640-647` `class DoneWithThisFrameDescrRef(_DoneWithThisFrameDescr)`.
#[derive(Debug)]
pub struct DoneWithThisFrameDescrRef(DoneWithThisFrameDescrBase);

impl DoneWithThisFrameDescrRef {
    pub fn new() -> Self {
        Self(DoneWithThisFrameDescrBase::new(vec![Type::Ref]))
    }
}

impl Default for DoneWithThisFrameDescrRef {
    fn default() -> Self {
        Self::new()
    }
}

impl Descr for DoneWithThisFrameDescrRef {
    fn get_descr_index(&self) -> i32 {
        self.0.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.0.descr_index.store(index, Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for DoneWithThisFrameDescrRef {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.0.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        true
    }
    fn adr_jump_offset(&self) -> usize {
        self.0.adr_jump_offset()
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        self.0.set_adr_jump_offset(offset);
    }
    fn rd_locs(&self) -> &[u16] {
        self.0.rd_locs()
    }
    fn set_rd_locs(&self, locs: Vec<u16>) {
        self.0.set_rd_locs(locs);
    }
}

/// `compile.py:649-656` `class DoneWithThisFrameDescrFloat(_DoneWithThisFrameDescr)`.
#[derive(Debug)]
pub struct DoneWithThisFrameDescrFloat(DoneWithThisFrameDescrBase);

impl DoneWithThisFrameDescrFloat {
    pub fn new() -> Self {
        Self(DoneWithThisFrameDescrBase::new(vec![Type::Float]))
    }
}

impl Default for DoneWithThisFrameDescrFloat {
    fn default() -> Self {
        Self::new()
    }
}

impl Descr for DoneWithThisFrameDescrFloat {
    fn get_descr_index(&self) -> i32 {
        self.0.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.0.descr_index.store(index, Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for DoneWithThisFrameDescrFloat {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.0.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        true
    }
    fn adr_jump_offset(&self) -> usize {
        self.0.adr_jump_offset()
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        self.0.set_adr_jump_offset(offset);
    }
    fn rd_locs(&self) -> &[u16] {
        self.0.rd_locs()
    }
    fn set_rd_locs(&self, locs: Vec<u16>) {
        self.0.set_rd_locs(locs);
    }
}

/// Pyre extension for CPython-compatible multi-result FINISH exits. PyPy's
/// `_DoneWithThisFrameDescr*` family has only 0/1-result classes, but callers
/// still need `fail_arg_types()` to preserve the actual terminal layout.
#[derive(Debug)]
pub struct DoneWithThisFrameDescrMulti(DoneWithThisFrameDescrBase);

impl DoneWithThisFrameDescrMulti {
    pub fn new(fail_arg_types: Vec<Type>) -> Self {
        Self(DoneWithThisFrameDescrBase::new(fail_arg_types))
    }
}

/// Canonical cache for multi-result `DoneWithThisFrameDescr*` instances,
/// keyed by `fail_arg_types`.  `compile.py:626-656`'s 0/1-result family
/// is `module-level singleton`; pyre extends it to N-ary tuples here
/// (see `make_finish_fail_descr_typed`).  The cache restores the same
/// pointer-identity contract: every multi-result FINISH of a given
/// type list resolves to the same `Arc::ptr_eq` identity.
///
/// `Vec<(Vec<Type>, Arc<...>)>` instead of `HashMap` per project rules
/// (`AGENTS.md` forbids HashMap/BTreeMap).  Linear scan is fine here —
/// the cache size is bounded by the number of distinct FINISH result
/// type lists in the program (typically a small handful).
static DONE_MULTI_CACHE: OnceLock<Mutex<Vec<(Vec<Type>, Arc<DoneWithThisFrameDescrMulti>)>>> =
    OnceLock::new();

fn done_multi_cache() -> &'static Mutex<Vec<(Vec<Type>, Arc<DoneWithThisFrameDescrMulti>)>> {
    DONE_MULTI_CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

/// Return the canonical `Arc<DoneWithThisFrameDescrMulti>` for a given
/// `fail_arg_types` list.  Two callers with structurally equal type
/// lists share one `Arc` (`Arc::ptr_eq` round-trip), restoring the
/// singleton semantics that `compile.py:626-656` gives 0/1-result
/// finishes.
pub fn get_or_attach_done_with_this_frame_descr_multi(
    fail_arg_types: Vec<Type>,
) -> Arc<DoneWithThisFrameDescrMulti> {
    let mut cache = done_multi_cache()
        .lock()
        .expect("done_multi_cache mutex poisoned");
    if let Some((_, descr)) = cache.iter().find(|(k, _)| k == &fail_arg_types) {
        return Arc::clone(descr);
    }
    let descr = Arc::new(DoneWithThisFrameDescrMulti::new(fail_arg_types.clone()));
    cache.push((fail_arg_types, Arc::clone(&descr)));
    descr
}

impl Descr for DoneWithThisFrameDescrMulti {
    fn get_descr_index(&self) -> i32 {
        self.0.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.0.descr_index.store(index, Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for DoneWithThisFrameDescrMulti {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.0.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        true
    }
    fn adr_jump_offset(&self) -> usize {
        self.0.adr_jump_offset()
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        self.0.set_adr_jump_offset(offset);
    }
    fn rd_locs(&self) -> &[u16] {
        self.0.rd_locs()
    }
    fn set_rd_locs(&self, locs: Vec<u16>) {
        self.0.set_rd_locs(locs);
    }
}

/// `compile.py:658-662` `class ExitFrameWithExceptionDescrRef(_DoneWithThisFrameDescr)`.
#[derive(Debug)]
pub struct ExitFrameWithExceptionDescrRef(DoneWithThisFrameDescrBase);

impl ExitFrameWithExceptionDescrRef {
    pub fn new() -> Self {
        Self(DoneWithThisFrameDescrBase::new(vec![Type::Ref]))
    }
}

impl Default for ExitFrameWithExceptionDescrRef {
    fn default() -> Self {
        Self::new()
    }
}

impl Descr for ExitFrameWithExceptionDescrRef {
    fn get_descr_index(&self) -> i32 {
        self.0.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.0.descr_index.store(index, Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for ExitFrameWithExceptionDescrRef {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.0.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        // `compile.py:658` inherits `final_descr = True` from `_DoneWithThisFrameDescr`.
        true
    }
    fn is_exit_frame_with_exception(&self) -> bool {
        // `compile.py:658` subclass identity: ExitFrameWithExceptionDescrRef
        // dispatches to `jitexc.ExitFrameWithExceptionRef` via `handle_fail`.
        true
    }
    fn adr_jump_offset(&self) -> usize {
        self.0.adr_jump_offset()
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        self.0.set_adr_jump_offset(offset);
    }
    fn rd_locs(&self) -> &[u16] {
        self.0.rd_locs()
    }
    fn set_rd_locs(&self, locs: Vec<u16>) {
        self.0.set_rd_locs(locs);
    }
}

/// `compile.py:1092-1099` `class PropagateExceptionDescr(AbstractFailDescr)`.
///
/// `handle_fail` reads the exception out of the `deadframe` and raises
/// `jitexc.ExitFrameWithExceptionRef`.  Stored on
/// `JitDriverStaticData.propagate_exc_descr` and on
/// `MetaInterpStaticData.propagate_exception_descr` so
/// `compile_tmp_callback` can reference it when emitting the
/// `GUARD_NO_EXCEPTION` descriptor.
#[derive(Debug)]
pub struct PropagateExceptionDescr {
    /// `history.py:122` `index = -1` default.
    descr_index: AtomicI32,
    /// `history.py:132` `AbstractFailDescr._attrs_` `adr_jump_offset`
    /// — inherited from `AbstractFailDescr` (compile.py:1092 `class
    /// PropagateExceptionDescr(AbstractFailDescr)`).  Stamped by
    /// `assembler.py:849 patch_pending_failure_recoveries` when the
    /// `GUARD_NO_EXCEPTION` recovery stub at `compile.py:1141` is
    /// finalised.
    adr_jump_offset: UnsafeCell<usize>,
    /// `history.py:132` `_attrs_` `rd_locs` — same inheritance path.
    rd_locs: UnsafeCell<Vec<u16>>,
}

// Safety: single-threaded JIT (RPython GIL parity).
unsafe impl Send for PropagateExceptionDescr {}
unsafe impl Sync for PropagateExceptionDescr {}

impl PropagateExceptionDescr {
    pub fn new() -> Self {
        Self {
            descr_index: AtomicI32::new(-1),
            adr_jump_offset: UnsafeCell::new(0),
            rd_locs: UnsafeCell::new(Vec::new()),
        }
    }
}

impl Descr for PropagateExceptionDescr {
    fn get_descr_index(&self) -> i32 {
        self.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.descr_index.store(index, Ordering::Relaxed);
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for PropagateExceptionDescr {
    fn fail_index(&self) -> u32 {
        u32::MAX
    }
    fn fail_arg_types(&self) -> &[Type] {
        // `compile.py:1141` `ResOperation(rop.GUARD_NO_EXCEPTION, [], descr=faildescr)`
        // `operations[1].setfailargs([])` — no fail args.
        &[]
    }
    fn is_finish(&self) -> bool {
        // `compile.py:1092` `class PropagateExceptionDescr(AbstractFailDescr)` —
        // inherits `final_descr = False`.  This is a guard descr, not a finish.
        false
    }
    fn adr_jump_offset(&self) -> usize {
        // Safety: single-threaded JIT (RPython GIL parity).
        unsafe { *self.adr_jump_offset.get() }
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        // Safety: single-threaded JIT (RPython GIL parity).
        unsafe { *self.adr_jump_offset.get() = offset };
    }
    fn rd_locs(&self) -> &[u16] {
        // Safety: single-threaded JIT (RPython GIL parity).
        unsafe { &*self.rd_locs.get() }
    }
    fn set_rd_locs(&self, locs: Vec<u16>) {
        // Safety: single-threaded JIT (RPython GIL parity).
        unsafe { *self.rd_locs.get() = locs };
    }
}
