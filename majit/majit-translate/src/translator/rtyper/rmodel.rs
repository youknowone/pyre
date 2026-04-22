//! RPython `rpython/rtyper/rmodel.py` — Repr base class + setup
//! state machine + inputconst / mangle helpers + VoidRepr /
//! SimplePointerRepr leaves.
//!
//! ## Scope of this scaffold
//!
//! Upstream rmodel.py is 474 LOC. This port lands the subset that
//! `rpbc.py FunctionReprBase` / `rclass.py ClassRepr` directly consume:
//!
//! | upstream | Rust mirror |
//! |---|---|
//! | `setupstate` (`rmodel.py:10-15`) | [`Setupstate`] enum |
//! | `class Repr(object)` (`rmodel.py:17-246`) | [`Repr`] trait + [`ReprState`] state helper |
//! | `class VoidRepr(Repr)` + `impossible_repr` (`rmodel.py:353-359`) | [`VoidRepr`] + [`impossible_repr`] |
//! | `class SimplePointerRepr(Repr)` (`rmodel.py:365-375`) | [`SimplePointerRepr`] |
//! | `def inputconst(reqtype, value)` (`rmodel.py:379-395`) | [`inputconst`] |
//! | `def mangle(prefix, name)` (`rmodel.py:402-408`) | [`mangle`] |
//! | `class BrokenReprTyperError(TyperError)` (`rmodel.py:397-400`) | already ported in [`error::TyperError::BrokenRepr`][crate::translator::rtyper::error::TyperError::BrokenRepr] |
//!
//! ## Deferred to follow-up commits
//!
//! * `rtype_*` methods that take a `HighLevelOp` (e.g. `rtype_getattr`,
//!   `rtype_str`, `rtype_bool`, `rtype_simple_call`, ...) — these land
//!   after the `HighLevelOp` copy/dispatch/inputarg surface is ported
//!   (`rpython/rtyper/rtyper.py:617+`).
//! * `Repr.__getattr__` autosetup side effect (`rmodel.py:95-106`) —
//!   Rust has no `__getattr__`. Callers that previously relied on it
//!   must invoke [`Repr::setup`] explicitly before reading derived
//!   fields; this matches upstream's own `setup()` call sequencing in
//!   `rtyper.py:call_all_setups` (`:241`).
//! * `CanBeNull` / `IteratorRepr` / `VoidRepr.get_ll_*` secondary
//!   methods — land alongside `rtyper.py:bindingrepr` / `getrepr`
//!   dispatch in Commit 3.2.
//! * `pairtype(Repr, Repr)` default conversions (`rmodel.py:298-348`) —
//!   upstream's double-dispatch mechanism lands with the conversion
//!   table port.
//!
//! ## Parity checkpoints
//!
//! The full dispatch chain upstream is:
//!
//! ```text
//! rtyper.specialize()                           rtyper.py:177
//!   → specialize_more_blocks()                   rtyper.py:198
//!     → specialize_block(block)                  rtyper.py:283
//!       → for hop in highlevelops(...):           rtyper.py:307
//!           translate_hl_to_ll(hop)               → rtyper.py:translate_op_*
//!             → Repr.rtype_simple_call(hop)       ← pyre lands this in 3.2
//!                → FunctionReprBase.call(hop)     ← pyre lands this in 3.3
//!                   → emit direct_call            ← already backed by jtransform
//! ```
//!
//! This commit scaffolds the bottom of the chain (`Repr` base + leaves)
//! so follow-ups can land in order without retrofitting infrastructure.

use std::fmt::{self, Debug};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::flowspace::model::{ConstValue, Constant};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

/// RPython `setupstate` (`rmodel.py:10-15`).
///
/// ```python
/// class setupstate(object):
///     NOTINITIALIZED = 0
///     INPROGRESS = 1
///     BROKEN = 2
///     FINISHED = 3
///     DELAYED = 4
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Setupstate {
    /// Initial state; [`Repr::setup`] still has to run.
    NotInitialized = 0,
    /// Inside an active [`Repr::_setup_repr`] call. Re-entry is an
    /// `AssertionError` upstream (`rmodel.py:45-47`).
    InProgress = 1,
    /// `_setup_repr()` raised `TyperError`; subsequent `setup()` calls
    /// re-raise `BrokenReprTyperError` (`rmodel.py:42-44`).
    Broken = 2,
    /// `_setup_repr()` returned normally; fields are ready to read.
    Finished = 3,
    /// `setup()` deferred pending an outer-pass pre-registration
    /// (`rmodel.py:82-93`).
    Delayed = 4,
}

impl Setupstate {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Setupstate::NotInitialized,
            1 => Setupstate::InProgress,
            2 => Setupstate::Broken,
            3 => Setupstate::Finished,
            4 => Setupstate::Delayed,
            _ => unreachable!("invalid Setupstate u8={value}"),
        }
    }
}

/// RPython `Repr._initialized` field (`rmodel.py:26`).
///
/// Concrete Repr types embed a `ReprState` alongside their own fields.
/// Interior mutability uses [`AtomicU8`] so Repr instances stored in
/// `OnceLock`-backed singletons (`impossible_repr`, per-lltype caches)
/// can survive Rust's `Sync` bound. Upstream Python is single-threaded
/// so this only matters for the Rust adaptation; ordering is
/// `Relaxed` since `setup()` is serialized by the rtyper's own
/// sequential control flow.
#[derive(Debug)]
pub struct ReprState {
    initialized: AtomicU8,
}

impl ReprState {
    /// Construct a fresh state in [`Setupstate::NotInitialized`].
    pub fn new() -> Self {
        ReprState {
            initialized: AtomicU8::new(Setupstate::NotInitialized as u8),
        }
    }

    /// Current state (read).
    pub fn get(&self) -> Setupstate {
        Setupstate::from_u8(self.initialized.load(Ordering::Relaxed))
    }

    /// Force-set (write).
    pub fn set(&self, state: Setupstate) {
        self.initialized.store(state as u8, Ordering::Relaxed);
    }
}

impl Default for ReprState {
    fn default() -> Self {
        Self::new()
    }
}

/// RPython `class Repr(object)` (`rmodel.py:17-246`).
///
/// Trait object so a `RPythonTyper.reprs: HashMap<SomeValue, Arc<dyn
/// Repr>>` (future 3.2 commit) can store heterogeneous Repr instances.
/// Most default methods match upstream's `Repr` base methods verbatim;
/// only the required fields (`lowleveltype`, state) are abstract.
pub trait Repr: Debug {
    /// RPython `Repr.lowleveltype` (`rmodel.py:26` + each subclass).
    fn lowleveltype(&self) -> &LowLevelType;

    /// Access to the embedded [`ReprState`]. Concrete types store one
    /// and return a reference here.
    fn state(&self) -> &ReprState;

    /// RPython `Repr.__repr__` (`rmodel.py:29-30`):
    /// `return '<%s %s>' % (self.__class__.__name__, self.lowleveltype)`.
    fn repr_string(&self) -> String {
        format!(
            "<{} {}>",
            self.class_name(),
            self.lowleveltype().short_name()
        )
    }

    /// RPython `Repr.compact_repr` (`rmodel.py:32-33`):
    /// `return '%s %s' % (self.__class__.__name__.replace('Repr','R'),
    ///                    self.lowleveltype._short_name())`.
    fn compact_repr(&self) -> String {
        format!(
            "{} {}",
            self.class_name().replace("Repr", "R"),
            self.lowleveltype().short_name()
        )
    }

    /// Concrete class name (for the `__repr__` / `compact_repr`
    /// formatters and TyperError messages). Rust does not have
    /// `__class__.__name__`; each concrete Repr returns its type name
    /// here.
    fn class_name(&self) -> &'static str;

    /// RPython `Repr.setup(self)` (`rmodel.py:35-59`).
    ///
    /// ```python
    /// def setup(self):
    ///     if self._initialized == setupstate.FINISHED:
    ///         return
    ///     elif self._initialized == setupstate.BROKEN:
    ///         raise BrokenReprTyperError(
    ///             "cannot setup already failed Repr: %r" %(self,))
    ///     elif self._initialized == setupstate.INPROGRESS:
    ///         raise AssertionError(
    ///             "recursive invocation of Repr setup(): %r" %(self,))
    ///     elif self._initialized == setupstate.DELAYED:
    ///         raise AssertionError(
    ///             "Repr setup() is delayed and cannot be called yet: %r" %(self,))
    ///     assert self._initialized == setupstate.NOTINITIALIZED
    ///     self._initialized = setupstate.INPROGRESS
    ///     try:
    ///         self._setup_repr()
    ///     except TyperError:
    ///         self._initialized = setupstate.BROKEN
    ///         raise
    ///     else:
    ///         self._initialized = setupstate.FINISHED
    /// ```
    fn setup(&self) -> Result<(), TyperError> {
        let state = self.state();
        match state.get() {
            Setupstate::Finished => return Ok(()),
            Setupstate::Broken => {
                return Err(TyperError::broken_repr(format!(
                    "cannot setup already failed Repr: {}",
                    self.repr_string()
                )));
            }
            Setupstate::InProgress => {
                // upstream `raise AssertionError` — pyre surfaces the
                // same diagnostic through TyperError since Rust has no
                // AssertionError class and callers already handle
                // TyperError on the specialize path.
                panic!(
                    "recursive invocation of Repr setup(): {}",
                    self.repr_string()
                );
            }
            Setupstate::Delayed => {
                panic!(
                    "Repr setup() is delayed and cannot be called yet: {}",
                    self.repr_string()
                );
            }
            Setupstate::NotInitialized => {}
        }
        state.set(Setupstate::InProgress);
        match self._setup_repr() {
            Ok(()) => {
                state.set(Setupstate::Finished);
                Ok(())
            }
            Err(e) => {
                state.set(Setupstate::Broken);
                Err(e)
            }
        }
    }

    /// RPython `Repr._setup_repr(self)` (`rmodel.py:61-62`).
    ///
    /// Default no-op. Concrete subclasses override for recursive /
    /// two-step initialization.
    fn _setup_repr(&self) -> Result<(), TyperError> {
        Ok(())
    }

    /// RPython `Repr.setup_final(self)` (`rmodel.py:64-74`).
    ///
    /// ```python
    /// def setup_final(self):
    ///     if self._initialized == setupstate.BROKEN:
    ///         raise BrokenReprTyperError(...)
    ///     assert self._initialized == setupstate.FINISHED
    ///     self._setup_repr_final()
    /// ```
    fn setup_final(&self) -> Result<(), TyperError> {
        let state = self.state();
        match state.get() {
            Setupstate::Broken => Err(TyperError::broken_repr(format!(
                "cannot perform setup_final_touch on failed Repr: {}",
                self.repr_string()
            ))),
            Setupstate::Finished => self._setup_repr_final(),
            other => panic!(
                "setup_final() on repr with state {other:?}: {}",
                self.repr_string()
            ),
        }
    }

    /// RPython `Repr._setup_repr_final(self)` (`rmodel.py:76-77`).
    /// Default no-op.
    fn _setup_repr_final(&self) -> Result<(), TyperError> {
        Ok(())
    }

    /// RPython `Repr.is_setup_delayed(self)` (`rmodel.py:79-80`).
    fn is_setup_delayed(&self) -> bool {
        matches!(self.state().get(), Setupstate::Delayed)
    }

    /// RPython `Repr.set_setup_delayed(self, flag)` (`rmodel.py:82-88`).
    ///
    /// ```python
    /// def set_setup_delayed(self, flag):
    ///     assert self._initialized in (setupstate.NOTINITIALIZED,
    ///                                  setupstate.DELAYED)
    ///     if flag:
    ///         self._initialized = setupstate.DELAYED
    ///     else:
    ///         self._initialized = setupstate.NOTINITIALIZED
    /// ```
    fn set_setup_delayed(&self, flag: bool) {
        let state = self.state();
        let current = state.get();
        assert!(
            matches!(current, Setupstate::NotInitialized | Setupstate::Delayed),
            "set_setup_delayed requires NotInitialized/Delayed, got {current:?}"
        );
        if flag {
            state.set(Setupstate::Delayed);
        } else {
            state.set(Setupstate::NotInitialized);
        }
    }

    /// RPython `Repr.set_setup_maybe_delayed(self)` (`rmodel.py:90-93`).
    ///
    /// ```python
    /// def set_setup_maybe_delayed(self):
    ///     if self._initialized == setupstate.NOTINITIALIZED:
    ///         self._initialized = setupstate.DELAYED
    ///     return self._initialized == setupstate.DELAYED
    /// ```
    fn set_setup_maybe_delayed(&self) -> bool {
        let state = self.state();
        if matches!(state.get(), Setupstate::NotInitialized) {
            state.set(Setupstate::Delayed);
        }
        matches!(state.get(), Setupstate::Delayed)
    }

    /// RPython `Repr.convert_const(self, value)` (`rmodel.py:120-125`).
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     "Convert the given constant value to the low-level repr of 'self'."
    ///     if not self.lowleveltype._contains_value(value):
    ///         raise TyperError("convert_const(self = %r, value = %r)" % (
    ///             self, value))
    ///     return value
    /// ```
    ///
    /// Upstream returns the Python value directly (the `_contains_value`
    /// predicate doubles as a type cast); the pyre adaptation wraps the
    /// value in a `Constant` whose `concretetype` carries the repr's
    /// lowleveltype, matching what `inputconst` / `specialize_block`
    /// expect downstream.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        if !self.lowleveltype().contains_value(value) {
            return Err(TyperError::message(format!(
                "convert_const(self = {}, value = {:?})",
                self.repr_string(),
                value
            )));
        }
        Ok(Constant::with_concretetype(
            value.clone(),
            self.lowleveltype().clone(),
        ))
    }

    /// RPython `Repr.special_uninitialized_value(self)` (`rmodel.py:127-128`).
    fn special_uninitialized_value(&self) -> Option<ConstValue> {
        None
    }

    /// RPython `Repr.can_ll_be_null(self, s_value)` (`rmodel.py:150-155`).
    ///
    /// Default `true` (conservative) matching upstream.
    fn can_ll_be_null(&self) -> bool {
        true
    }

    /// RPython `Repr._freeze_(self)` (`rmodel.py:108-109`).
    /// Always true — Repr instances are immutable once created.
    fn freeze(&self) -> bool {
        true
    }
}

/// RPython `inputconst(reqtype, value)` (`rmodel.py:379-395`).
///
/// Upstream supports `reqtype` as either `Repr` or `LowLevelType`.
/// Pyre splits into two functions because Rust's type system prefers
/// explicit dispatch over Python-style duck typing. Use
/// [`inputconst_from_lltype`] when the caller already has a bare
/// `LowLevelType`.
///
/// ```python
/// def inputconst(reqtype, value):
///     if isinstance(reqtype, Repr):
///         value = reqtype.convert_const(value)
///         lltype = reqtype.lowleveltype
///     elif isinstance(reqtype, LowLevelType):
///         lltype = reqtype
///     else:
///         raise TypeError(repr(reqtype))
///     if not lltype._contains_value(value):
///         raise TyperError("inputconst(): expected a %r, got %r" %
///                          (lltype, value))
///     c = Constant(value)
///     c.concretetype = lltype
///     return c
/// ```
pub fn inputconst<R: Repr + ?Sized>(
    reqtype: &R,
    value: &ConstValue,
) -> Result<Constant, TyperError> {
    let c = reqtype.convert_const(value)?;
    // `convert_const` already populated `concretetype`; double-check
    // contains_value as upstream does post-convert.
    if !reqtype.lowleveltype().contains_value(&c.value) {
        return Err(TyperError::message(format!(
            "inputconst(): expected a {}, got {:?}",
            reqtype.lowleveltype().short_name(),
            c.value
        )));
    }
    Ok(c)
}

/// RPython `inputconst(LowLevelType, value)` overload (`rmodel.py:386-394`).
pub fn inputconst_from_lltype(
    lltype: &LowLevelType,
    value: &ConstValue,
) -> Result<Constant, TyperError> {
    if !lltype.contains_value(value) {
        return Err(TyperError::message(format!(
            "inputconst(): expected a {}, got {:?}",
            lltype.short_name(),
            value
        )));
    }
    Ok(Constant::with_concretetype(value.clone(), lltype.clone()))
}

/// RPython `mangle(prefix, name)` (`rmodel.py:402-408`).
///
/// ```python
/// def mangle(prefix, name):
///     """Make a unique identifier from the prefix and the name.  The name
///     is allowed to start with $."""
///     if name.startswith('$'):
///         return '%sinternal_%s' % (prefix, name[1:])
///     else:
///         return '%s_%s' % (prefix, name)
/// ```
pub fn mangle(prefix: &str, name: &str) -> String {
    if let Some(stripped) = name.strip_prefix('$') {
        format!("{prefix}internal_{stripped}")
    } else {
        format!("{prefix}_{name}")
    }
}

// ____________________________________________________________
// Concrete Repr leaves that every downstream port needs.

/// RPython `class VoidRepr(Repr)` (`rmodel.py:353-359`).
///
/// ```python
/// class VoidRepr(Repr):
///     lowleveltype = Void
///     def get_ll_eq_function(self): return None
///     def get_ll_hash_function(self): return ll_hash_void
///     get_ll_fasthash_function = get_ll_hash_function
///     def ll_str(self, nothing): raise AssertionError("unreachable code")
/// impossible_repr = VoidRepr()
/// ```
#[derive(Debug)]
pub struct VoidRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl VoidRepr {
    pub fn new() -> Self {
        VoidRepr {
            state: ReprState::new(),
            lltype: LowLevelType::Void,
        }
    }
}

impl Default for VoidRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for VoidRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "VoidRepr"
    }
}

/// RPython `impossible_repr = VoidRepr()` (`rmodel.py:359`) — the
/// singleton VoidRepr used for `SomeImpossibleValue` (`rmodel.py:288+`).
///
/// Rust returns an [`Arc<VoidRepr>`] clone of the cached singleton so
/// the pointer identity upstream relies on (`r is impossible_repr`) is
/// preserved through `Arc::ptr_eq`.
pub fn impossible_repr() -> std::sync::Arc<VoidRepr> {
    static REPR: OnceLock<std::sync::Arc<VoidRepr>> = OnceLock::new();
    REPR.get_or_init(|| std::sync::Arc::new(VoidRepr::new()))
        .clone()
}

/// RPython `class SimplePointerRepr(Repr)` (`rmodel.py:365-375`).
///
/// ```python
/// class SimplePointerRepr(Repr):
///     "Convenience Repr for simple ll pointer types with no operation on them."
///
///     def __init__(self, lowleveltype):
///         self.lowleveltype = lowleveltype
///
///     def convert_const(self, value):
///         if value is not None:
///             raise TyperError("%r only supports None as prebuilt constant, "
///                              "got %r" % (self, value))
///         return lltype.nullptr(self.lowleveltype.TO)
/// ```
#[derive(Debug)]
pub struct SimplePointerRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl SimplePointerRepr {
    /// `lowleveltype` must be a [`LowLevelType::Ptr`] variant upstream
    /// (`rmodel.py:365-375`); enforce via debug assertion.
    pub fn new(lowleveltype: LowLevelType) -> Self {
        debug_assert!(
            matches!(lowleveltype, LowLevelType::Ptr(_)),
            "SimplePointerRepr requires Ptr lowleveltype, got {lowleveltype:?}"
        );
        SimplePointerRepr {
            state: ReprState::new(),
            lltype: lowleveltype,
        }
    }
}

impl Repr for SimplePointerRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "SimplePointerRepr"
    }

    /// RPython override (`rmodel.py:371-375`): only accept `None`, emit
    /// `nullptr(self.lowleveltype.TO)`.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        if !matches!(value, ConstValue::None) {
            return Err(TyperError::message(format!(
                "{} only supports None as prebuilt constant, got {value:?}",
                self.repr_string()
            )));
        }
        // upstream returns `lltype.nullptr(self.lowleveltype.TO)`; pyre
        // stores the null sentinel via `ConstValue::None` with the
        // Ptr lowleveltype on the Constant, matching what downstream
        // emit_const_r pipes through.
        Ok(Constant::with_concretetype(
            ConstValue::None,
            self.lltype.clone(),
        ))
    }
}

/// RPython `rptr.py:27-118` — `class PtrRepr(Repr)`.
#[derive(Debug)]
pub struct PtrRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl PtrRepr {
    pub fn new(ptrtype: crate::translator::rtyper::lltypesystem::lltype::Ptr) -> Self {
        PtrRepr {
            state: ReprState::new(),
            lltype: LowLevelType::Ptr(Box::new(ptrtype)),
        }
    }
}

impl Repr for PtrRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "PtrRepr"
    }
}

fn ptr_from_container_lowleveltype(
    ptrtype: &crate::translator::rtyper::lltypesystem::lltype::InteriorPtr,
) -> crate::translator::rtyper::lltypesystem::lltype::Ptr {
    use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget};

    Ptr {
        TO: match &*ptrtype.PARENTTYPE {
            LowLevelType::Struct(t) => PtrTarget::Struct((**t).clone()),
            LowLevelType::Array(t) => PtrTarget::Array((**t).clone()),
            LowLevelType::FixedSizeArray(t) => PtrTarget::FixedSizeArray((**t).clone()),
            LowLevelType::Opaque(t) => PtrTarget::Opaque((**t).clone()),
            LowLevelType::ForwardReference(t) => PtrTarget::ForwardReference((**t).clone()),
            other => panic!("InteriorPtrRepr parent must be a container type, got {other:?}"),
        },
    }
}

fn interior_ptr_lowleveltype(
    ptrtype: &crate::translator::rtyper::lltypesystem::lltype::InteriorPtr,
) -> LowLevelType {
    use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget};

    let parent_ptr = ptr_from_container_lowleveltype(ptrtype);
    if ptrtype.offsets.iter().any(|offset| {
        matches!(
            offset,
            crate::translator::rtyper::lltypesystem::lltype::InteriorOffset::Index(_)
        )
    }) {
        LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(parent_ptr._interior_ptr_type_with_index(&ptrtype.TO)),
        }))
    } else {
        LowLevelType::Ptr(Box::new(parent_ptr))
    }
}

/// RPython `rptr.py:220-298` — `class InteriorPtrRepr(Repr)`.
#[derive(Debug)]
pub struct InteriorPtrRepr {
    state: ReprState,
    lltype: LowLevelType,
    _ptrtype: crate::translator::rtyper::lltypesystem::lltype::InteriorPtr,
    pub v_offsets: Vec<Option<Constant>>,
    pub parentptrtype: crate::translator::rtyper::lltypesystem::lltype::Ptr,
}

impl InteriorPtrRepr {
    pub fn new(ptrtype: crate::translator::rtyper::lltypesystem::lltype::InteriorPtr) -> Self {
        let lltype = interior_ptr_lowleveltype(&ptrtype);
        let mut v_offsets = Vec::new();
        let mut numitemoffsets = 0;
        for offset in &ptrtype.offsets {
            match offset {
                crate::translator::rtyper::lltypesystem::lltype::InteriorOffset::Index(_) => {
                    numitemoffsets += 1;
                    v_offsets.push(None);
                }
                crate::translator::rtyper::lltypesystem::lltype::InteriorOffset::Field(name) => {
                    v_offsets.push(Some(Constant::with_concretetype(
                        ConstValue::Str(name.clone()),
                        LowLevelType::Void,
                    )));
                }
            }
        }
        assert!(numitemoffsets <= 1);
        let parentptrtype = ptr_from_container_lowleveltype(&ptrtype);
        InteriorPtrRepr {
            state: ReprState::new(),
            lltype,
            _ptrtype: ptrtype,
            v_offsets,
            parentptrtype,
        }
    }
}

impl Repr for InteriorPtrRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "InteriorPtrRepr"
    }
}

/// RPython `rptr.py:195-211` — `class LLADTMethRepr(Repr)`.
#[derive(Debug)]
pub struct LLADTMethRepr {
    state: ReprState,
    lltype: LowLevelType,
    pub func: ConstValue,
    pub ll_ptrtype: crate::translator::rtyper::lltypesystem::lltype::LowLevelPointerType,
}

impl LLADTMethRepr {
    pub fn new(adtmeth: &crate::annotator::model::SomeLLADTMeth) -> Self {
        let lltype = match &adtmeth.ll_ptrtype {
            crate::translator::rtyper::lltypesystem::lltype::LowLevelPointerType::Ptr(ptr) => {
                LowLevelType::Ptr(Box::new(ptr.clone()))
            }
            crate::translator::rtyper::lltypesystem::lltype::LowLevelPointerType::InteriorPtr(
                ptr,
            ) => interior_ptr_lowleveltype(ptr),
        };
        LLADTMethRepr {
            state: ReprState::new(),
            lltype,
            func: adtmeth.func.clone(),
            ll_ptrtype: adtmeth.ll_ptrtype.clone(),
        }
    }
}

impl Repr for LLADTMethRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "LLADTMethRepr"
    }
}

// ____________________________________________________________
// `rtyper_makekey` / `rtyper_makerepr` per-SomeXxx dispatch
// (rmodel.py:276-293 + each r*.py `__extend__(SomeXxx)` block).

/// RPython `S.rtyper_makekey()` upstream tuple hashed by
/// `RPythonTyper.reprs` (rtyper.py:54+149).
///
/// Upstream returns a tuple like `(S.__class__, knowntype, unsigned)`
/// which Python uses as a dict key. Rust mirrors via a typed enum so
/// each variant carries the right discriminating data for the reprs
/// cache.
///
/// Only `Impossible` is populated in this commit — other SomeValue
/// variants land with their respective concrete Repr ports. The
/// `Pending` variant carries a debug string so unimplemented keys
/// still round-trip through `HashMap<ReprKey, Arc<dyn Repr>>` during
/// bring-up.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReprKey {
    /// RPython `SomeImpossibleValue.rtyper_makekey = (self.__class__,)`
    /// (rmodel.py:292-293).
    Impossible,
    /// Pending variant — carries a textual discriminator from
    /// `rtyper_makekey` arm that hasn't been ported yet.
    Pending(String),
}

/// RPython `SomeXxx.rtyper_makekey()` dispatcher.
///
/// Upstream attaches a `rtyper_makekey` method per SomeXxx via the
/// `__extend__` metaclass pattern (rmodel.py:292, rint.py:190,
/// rbool.py:43, ...). Pyre centralises into this match since Rust has
/// no `__extend__` equivalent.
pub fn rtyper_makekey(s_obj: &crate::annotator::model::SomeValue) -> ReprKey {
    use crate::annotator::model::SomeValue;
    match s_obj {
        // rmodel.py:292-293: SomeImpossibleValue.rtyper_makekey = (self.__class__,).
        SomeValue::Impossible => ReprKey::Impossible,
        // Remaining variants defer to their r*.rs ports. Emit a
        // deterministic `Pending` key so the reprs cache still
        // distinguishes entries by variant-shape — identical
        // shape-strings collapse to one pending entry (matching
        // upstream `(class, knowntype, ...)` identity for identical
        // tuples).
        other => ReprKey::Pending(format!("{other:?}")),
    }
}

/// RPython `SomeXxx.rtyper_makerepr(rtyper)` dispatcher.
///
/// Upstream r*.py modules each contribute one arm per SomeXxx via
/// `__extend__` (rmodel.py:289, rint.py:186, rbool.py:40, ...). Pyre
/// centralises into this match; as each concrete Repr lands in its
/// r*.rs file the arm flips from `MissingRTypeOperation` to the real
/// constructor.
pub fn rtyper_makerepr(
    s_obj: &crate::annotator::model::SomeValue,
    rtyper: &crate::translator::rtyper::rtyper::RPythonTyper,
) -> Result<std::sync::Arc<dyn Repr>, TyperError> {
    use crate::annotator::model::SomeValue;
    let _ = rtyper; // unused until concrete repr ports consume it
    match s_obj {
        // rmodel.py:289-290: SomeImpossibleValue.rtyper_makerepr
        // returns the singleton `impossible_repr`.
        SomeValue::Impossible => Ok(impossible_repr() as std::sync::Arc<dyn Repr>),
        // Every other SomeValue variant maps 1:1 to a concrete Repr
        // defined in an r*.py upstream module. Pyre reports a
        // structured MissingRTypeOperation with the expected target
        // module so follow-up cascading ports can anchor on the
        // surface without guessing.
        SomeValue::Integer(_) => Err(TyperError::missing_rtype_operation(
            "SomeInteger.rtyper_makerepr — port rpython/rtyper/rint.py IntegerRepr",
        )),
        SomeValue::Bool(_) => Err(TyperError::missing_rtype_operation(
            "SomeBool.rtyper_makerepr — port rpython/rtyper/rbool.py BoolRepr",
        )),
        SomeValue::Float(_) | SomeValue::SingleFloat(_) | SomeValue::LongFloat(_) => {
            Err(TyperError::missing_rtype_operation(
                "SomeFloat.rtyper_makerepr — port rpython/rtyper/rfloat.py FloatRepr",
            ))
        }
        SomeValue::String(_)
        | SomeValue::UnicodeString(_)
        | SomeValue::ByteArray(_)
        | SomeValue::Char(_)
        | SomeValue::UnicodeCodePoint(_) => Err(TyperError::missing_rtype_operation(
            "SomeString/ByteArray/Char.rtyper_makerepr — port rpython/rtyper/rstr.py",
        )),
        SomeValue::Instance(_) | SomeValue::Exception(_) => {
            Err(TyperError::missing_rtype_operation(
                "SomeInstance.rtyper_makerepr — port rpython/rtyper/rclass.py InstanceRepr",
            ))
        }
        SomeValue::Tuple(_) => Err(TyperError::missing_rtype_operation(
            "SomeTuple.rtyper_makerepr — port rpython/rtyper/rtuple.py TupleRepr",
        )),
        SomeValue::List(_) => Err(TyperError::missing_rtype_operation(
            "SomeList.rtyper_makerepr — port rpython/rtyper/rlist.py ListRepr",
        )),
        SomeValue::Dict(_) => Err(TyperError::missing_rtype_operation(
            "SomeDict.rtyper_makerepr — port rpython/rtyper/rdict.py DictRepr",
        )),
        SomeValue::Iterator(_) => Err(TyperError::missing_rtype_operation(
            "SomeIterator.rtyper_makerepr — port rpython/rtyper/rrange.py EnumerateIteratorRepr etc.",
        )),
        SomeValue::PBC(_) => Err(TyperError::missing_rtype_operation(
            "SomePBC.rtyper_makerepr — port rpython/rtyper/rpbc.py PBCRepr family",
        )),
        SomeValue::Builtin(_) | SomeValue::BuiltinMethod(_) => {
            Err(TyperError::missing_rtype_operation(
                "SomeBuiltin.rtyper_makerepr — port rpython/rtyper/rpbc.py FunctionRepr",
            ))
        }
        SomeValue::None_(_) => Err(TyperError::missing_rtype_operation(
            "SomeNone.rtyper_makerepr — port rpython/rtyper/rnone.py NoneRepr",
        )),
        SomeValue::Object(_) | SomeValue::Type(_) => Err(TyperError::missing_rtype_operation(
            "SomeObject/SomeType.rtyper_makerepr — port rpython/rtyper/robject.py",
        )),
        SomeValue::Property(_) => Err(TyperError::missing_rtype_operation(
            "SomeProperty.rtyper_makerepr — port rpython/rtyper/rproperty.py",
        )),
        SomeValue::WeakRef(_) => Err(TyperError::missing_rtype_operation(
            "SomeWeakRef.rtyper_makerepr — port rpython/rtyper/rweakref.py",
        )),
        SomeValue::TypeOf(_) => Err(TyperError::missing_rtype_operation(
            "SomeTypeOf.rtyper_makerepr — no direct upstream counterpart; pyre adaptation",
        )),
        SomeValue::Ptr(ptr) => {
            Ok(std::sync::Arc::new(PtrRepr::new(ptr.ll_ptrtype.clone()))
                as std::sync::Arc<dyn Repr>)
        }
        SomeValue::InteriorPtr(ptr) => Ok(std::sync::Arc::new(InteriorPtrRepr::new(
            ptr.ll_ptrtype.clone(),
        )) as std::sync::Arc<dyn Repr>),
        SomeValue::LLADTMeth(adtmeth) => {
            Ok(std::sync::Arc::new(LLADTMethRepr::new(adtmeth)) as std::sync::Arc<dyn Repr>)
        }
    }
}

// VoidRepr and SimplePointerRepr must be Display-able via Repr's
// `repr_string()` when formatted through Debug. Since `#[derive(Debug)]`
// already handles it, no additional impl is required. (TyperError's
// Display uses repr_string() through self.repr_string() calls.)
impl fmt::Display for VoidRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repr_string())
    }
}

impl fmt::Display for SimplePointerRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repr_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::model::{SomePtr, SomeValue};
    use crate::translator::rtyper::llannotation::{SomeInteriorPtr, SomeLLADTMeth};
    use crate::translator::rtyper::lltypesystem::lltype::FuncType;
    use crate::translator::rtyper::rtyper::RPythonTyper;

    fn rtyper_for_tests() -> RPythonTyper {
        let ann = RPythonAnnotator::new(None, None, None, false);
        RPythonTyper::new(&ann)
    }

    #[test]
    fn voidrepr_lowleveltype_is_void_and_reprs_match_upstream() {
        // rmodel.py:353-359: VoidRepr.lowleveltype = Void.
        let r = VoidRepr::new();
        assert_eq!(r.lowleveltype(), &LowLevelType::Void);
        // rmodel.py:30 `<%s %s>` formatter.
        assert_eq!(r.repr_string(), "<VoidRepr Void>");
        // rmodel.py:33 compact_repr — "VoidRepr" → "VoidR" replacement,
        // then short_name.
        assert_eq!(r.compact_repr(), "VoidR Void");
    }

    #[test]
    fn setup_transitions_notinitialized_to_finished() {
        // rmodel.py:35-59: NOTINITIALIZED → INPROGRESS → FINISHED.
        let r = VoidRepr::new();
        assert_eq!(r.state().get(), Setupstate::NotInitialized);
        r.setup().expect("setup should succeed on default VoidRepr");
        assert_eq!(r.state().get(), Setupstate::Finished);
        // Second call returns immediately (rmodel.py:40-41).
        r.setup().expect("setup should be idempotent once FINISHED");
        assert_eq!(r.state().get(), Setupstate::Finished);
    }

    #[test]
    fn setup_on_broken_state_raises_broken_repr_typer_error() {
        // rmodel.py:42-44.
        let r = VoidRepr::new();
        r.state().set(Setupstate::Broken);
        let err = r.setup().unwrap_err();
        assert!(err.is_broken_repr());
        assert!(err.to_string().contains("<VoidRepr Void>"));
    }

    #[test]
    #[should_panic(expected = "recursive invocation of Repr setup()")]
    fn setup_on_inprogress_panics_like_upstream_assertion() {
        // rmodel.py:45-47 uses `raise AssertionError` — pyre panics.
        let r = VoidRepr::new();
        r.state().set(Setupstate::InProgress);
        let _ = r.setup();
    }

    #[test]
    fn set_setup_delayed_toggles_state_both_ways() {
        // rmodel.py:82-88.
        let r = VoidRepr::new();
        r.set_setup_delayed(true);
        assert_eq!(r.state().get(), Setupstate::Delayed);
        assert!(r.is_setup_delayed());
        r.set_setup_delayed(false);
        assert_eq!(r.state().get(), Setupstate::NotInitialized);
        assert!(!r.is_setup_delayed());
    }

    #[test]
    fn set_setup_maybe_delayed_only_promotes_from_notinitialized() {
        // rmodel.py:90-93.
        let r = VoidRepr::new();
        assert!(r.set_setup_maybe_delayed());
        assert_eq!(r.state().get(), Setupstate::Delayed);
        // Already Delayed: returns true (the membership check) without
        // changing state.
        assert!(r.set_setup_maybe_delayed());
    }

    #[test]
    fn convert_const_on_voidrepr_accepts_any_value() {
        // rmodel.py:120-125 — `convert_const` delegates to
        // `_contains_value`, and `Void._contains_value` (lltype.py:194-197)
        // returns True for any value. So `VoidRepr` accepts None, Int,
        // Bool, etc. as valid constants of lowlevel type Void.
        let r = VoidRepr::new();
        let c = r.convert_const(&ConstValue::None).unwrap();
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Void));

        let c = r.convert_const(&ConstValue::Int(42)).unwrap();
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Void));

        let c = r.convert_const(&ConstValue::Bool(true)).unwrap();
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Void));
    }

    #[test]
    fn inputconst_wraps_value_and_records_lowleveltype() {
        // rmodel.py:379-395.
        let r = VoidRepr::new();
        let c = inputconst(&r, &ConstValue::None).unwrap();
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Void));

        let c2 = inputconst_from_lltype(&LowLevelType::Signed, &ConstValue::Int(7)).unwrap();
        assert_eq!(c2.concretetype.as_ref(), Some(&LowLevelType::Signed));
    }

    #[test]
    fn inputconst_from_lltype_rejects_mismatched_constant() {
        let err = inputconst_from_lltype(&LowLevelType::Bool, &ConstValue::Int(0)).unwrap_err();
        assert!(err.to_string().contains("inputconst"));
    }

    #[test]
    fn mangle_uses_internal_prefix_for_dollar_names() {
        // rmodel.py:402-408.
        assert_eq!(mangle("cls", "$hidden"), "clsinternal_hidden");
        assert_eq!(mangle("cls", "method"), "cls_method");
    }

    #[test]
    fn impossible_repr_is_shared_singleton_pointer() {
        // `impossible_repr = VoidRepr()` module-level singleton
        // (rmodel.py:359). Two calls return Arcs to the same instance.
        let a = impossible_repr();
        let b = impossible_repr();
        assert!(std::sync::Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn simple_pointer_repr_only_accepts_none_constant() {
        // rmodel.py:365-375.
        let ptr_ty = LowLevelType::Ptr(Box::new(
            crate::translator::rtyper::lltypesystem::lltype::Ptr {
                TO: crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Func(FuncType {
                    args: vec![],
                    result: LowLevelType::Void,
                }),
            },
        ));
        let r = SimplePointerRepr::new(ptr_ty.clone());
        assert_eq!(r.lowleveltype(), &ptr_ty);

        let c = r.convert_const(&ConstValue::None).unwrap();
        assert_eq!(c.concretetype.as_ref(), Some(&ptr_ty));

        let err = r.convert_const(&ConstValue::Int(0)).unwrap_err();
        assert!(
            err.to_string()
                .contains("only supports None as prebuilt constant"),
            "expected upstream phrase; got {err}"
        );
    }

    #[test]
    fn ptr_somevalues_make_rptr_reprs() {
        use crate::translator::rtyper::lltypesystem::lltype::{Ptr, PtrTarget};

        let ptr = Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Void,
            }),
        };
        let rtyper = rtyper_for_tests();
        let repr = rtyper_makerepr(&SomeValue::Ptr(SomePtr::new(ptr.clone())), &rtyper).unwrap();
        assert_eq!(repr.class_name(), "PtrRepr");
        assert_eq!(repr.lowleveltype(), &LowLevelType::Ptr(Box::new(ptr)));
    }

    #[test]
    fn interior_ptr_somevalues_make_rptr_reprs() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            InteriorOffset, InteriorPtr, PtrTarget, StructType,
        };

        let parent = LowLevelType::Struct(Box::new(StructType::new(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let interior = InteriorPtr {
            PARENTTYPE: Box::new(parent.clone()),
            TO: Box::new(LowLevelType::Signed),
            offsets: vec![InteriorOffset::Field("x".into())],
        };
        let rtyper = rtyper_for_tests();
        let repr = rtyper_makerepr(
            &SomeValue::InteriorPtr(SomeInteriorPtr::new(interior)),
            &rtyper,
        )
        .unwrap();
        assert_eq!(repr.class_name(), "InteriorPtrRepr");
        assert_eq!(
            repr.lowleveltype(),
            &LowLevelType::Ptr(Box::new(
                crate::translator::rtyper::lltypesystem::lltype::Ptr {
                    TO: PtrTarget::Struct(match parent {
                        LowLevelType::Struct(t) => *t,
                        _ => unreachable!(),
                    }),
                }
            ))
        );
    }

    #[test]
    fn interior_ptr_repr_records_offsets_like_rptr_init() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            InteriorOffset, InteriorPtr, PtrTarget, StructType,
        };

        let parent = LowLevelType::Struct(Box::new(StructType::new(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let repr = InteriorPtrRepr::new(InteriorPtr {
            PARENTTYPE: Box::new(parent),
            TO: Box::new(LowLevelType::Signed),
            offsets: vec![InteriorOffset::Field("x".into())],
        });
        assert_eq!(repr.v_offsets.len(), 1);
        let Some(c_offset) = &repr.v_offsets[0] else {
            panic!("field offset should be stored as a Void constant");
        };
        assert_eq!(c_offset.value, ConstValue::Str("x".into()));
        assert_eq!(c_offset.concretetype, Some(LowLevelType::Void));
        assert!(matches!(repr.parentptrtype.TO, PtrTarget::Struct(_)));
    }

    #[test]
    #[should_panic]
    fn interior_ptr_repr_rejects_multiple_item_offsets_like_rptr_init() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            ArrayType, InteriorOffset, InteriorPtr,
        };

        let _ = InteriorPtrRepr::new(InteriorPtr {
            PARENTTYPE: Box::new(LowLevelType::Array(Box::new(ArrayType::gc(
                LowLevelType::Signed,
            )))),
            TO: Box::new(LowLevelType::Signed),
            offsets: vec![InteriorOffset::Index(0), InteriorOffset::Index(1)],
        });
    }

    #[test]
    fn lladtmeth_somevalues_make_rptr_reprs() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelPointerType, Ptr, PtrTarget,
        };

        let ptr = Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Signed,
            }),
        };
        let adtmeth = SomeLLADTMeth::new(LowLevelPointerType::Ptr(ptr.clone()), ConstValue::None);
        let rtyper = rtyper_for_tests();
        let repr = rtyper_makerepr(&SomeValue::LLADTMeth(adtmeth), &rtyper).unwrap();
        assert_eq!(repr.class_name(), "LLADTMethRepr");
        assert_eq!(repr.lowleveltype(), &LowLevelType::Ptr(Box::new(ptr)));
    }

    #[test]
    fn repr_default_predicates_match_upstream_defaults() {
        // rmodel.py:108-109 `_freeze_` → True.
        // rmodel.py:127-128 `special_uninitialized_value` → None.
        // rmodel.py:150-155 `can_ll_be_null` → True.
        let r = VoidRepr::new();
        assert!(r.freeze());
        assert!(r.special_uninitialized_value().is_none());
        assert!(r.can_ll_be_null());
    }
}
