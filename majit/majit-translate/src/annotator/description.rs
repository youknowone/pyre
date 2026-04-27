//! Descriptor classes for constant callables / classes / frozen PBCs.
//!
//! RPython upstream: `rpython/annotator/description.py` (637 LOC).
//!
//! Commit 1 ported the self-contained family structures
//! (`CallFamily`, `FrozenAttrFamily`, `ClassAttrFamily`,
//! `NoStandardGraph`). Commit 2 added the `Desc` base class and the
//! `FunctionDesc` data-shell (description.py:132-406). Commit 3
//! (this commit) adds `MethodDesc`, `FrozenDesc`, and
//! `MethodOfFrozenDesc` — description.py:407-637 — completing the
//! description.py port surface. ClassDef references are carried via
//! opaque [`ClassDefKey`] handles for pointer-identity hashing on
//! the live `Rc<RefCell<super::classdesc::ClassDef>>` (the
//! `classdesc.py` port lives in [`super::classdesc`]).
//!
//! ## PRE-EXISTING-ADAPTATION: DescKey
//!
//! Upstream keys `CallFamily.descs`, `FrozenAttrFamily.descs`, and
//! `ClassAttrFamily.descs` on the `Desc` instance itself — Python
//! dict uses object identity by default for user classes. The Rust
//! port uses a [`DescKey`] newtype wrapping `usize` (the pointer
//! identity of the live `Rc<RefCell<Desc>>` variants behind
//! [`DescEntry`]). [`DescEntry::desc_key`] derives the key from
//! `Rc::as_ptr` per variant; tests use [`DescKey::from_raw`].

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::argument::{ArgErr, ArgumentsForTranslation};
use super::bookkeeper::{Bookkeeper, PositionKey};
use super::model::{
    AnnotatorError, SomeInstance, SomeObjectTrait, SomePBC, SomeValue, SomeValueTag,
    s_impossible_value, union,
};
use super::policy::Specializer;
use super::signature::ParamType;
use crate::flowspace::argument::{CallShape, Signature};
use crate::flowspace::model::{
    BlockRefExt, ConstValue, Constant, HostObject, checkgraph, host_getattr,
};
use crate::flowspace::pygraph::PyGraph;
use crate::tool::algo::unionfind::UnionFindInfo;

type GraphBuilder<'a> = Box<
    dyn FnOnce(
            &crate::translator::translator::TranslationContext,
            HostObject,
        ) -> Result<Rc<PyGraph>, AnnotatorError>
        + 'a,
>;

/// Opaque identity handle for a `Desc` instance.
///
/// Upstream uses Python object identity; the Rust port stores a
/// `usize` derived from `Rc::as_ptr(&desc)`. The live conversion
/// happens per-variant inside [`DescEntry::desc_key`] (see line ~553
/// for the per-variant `Rc::as_ptr` calls).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DescKey(pub usize);

impl DescKey {
    /// Turn a raw pointer identity into a [`DescKey`]. Used by tests
    /// and (soon) the Desc constructor in commit 2.
    pub fn from_raw(id: usize) -> Self {
        DescKey(id)
    }

    /// Compute a DescKey from the identity of an `Rc<T>` pointer — the
    /// Rust equivalent of Python dict-by-identity keying.
    pub fn from_rc<T: ?Sized>(rc: &Rc<T>) -> Self {
        DescKey(Rc::as_ptr(rc) as *const () as usize)
    }
}

fn alloc_desc_key() -> DescKey {
    static NEXT_DESC_KEY: AtomicUsize = AtomicUsize::new(1);
    DescKey(NEXT_DESC_KEY.fetch_add(1, Ordering::Relaxed))
}

/// RPython `class NoStandardGraph(Exception)` (description.py:185-186).
///
/// "The function doesn't have a single standard non-specialized
/// graph."
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoStandardGraph(pub DescKey);

impl std::fmt::Display for NoStandardGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NoStandardGraph({:?})", self.0)
    }
}

impl std::error::Error for NoStandardGraph {}

/// Upstream `calltable_add_row` stores graph values keyed by
/// `desc.rowkey()` (description.py:67). Once FunctionDesc lands,
/// `rowkey()` returns the Desc itself (identity); for now we carry
/// the identity via [`DescKey`].
pub type CallTableRow = HashMap<DescKey, Rc<PyGraph>>;

/// Identity-based equality for two [`CallTableRow`]s. Upstream's
/// dict-`__eq__` falls back to per-pair equality on `FunctionGraph`
/// instances — which is object identity because the class has no
/// `__eq__` override.
fn row_eq(a: &CallTableRow, b: &CallTableRow) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (k, v) in a {
        match b.get(k) {
            Some(bv) => {
                if !Rc::ptr_eq(v, bv) {
                    return false;
                }
            }
            None => return false,
        }
    }
    true
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum GraphCacheKey {
    None,
    Const(ConstValue),
    LowLevelType(crate::translator::rtyper::lltypesystem::lltype::LowLevelType),
    KnownType(super::model::KnownType),
    Desc { key: DescKey, name: String },
    KeyComp(crate::translator::rtyper::annlowlevel::KeyComp),
    String(String),
    Int(i64),
    Position(PositionKey),
    AccessDirect,
    SomeValueTag(SomeValueTag),
    Tuple(Vec<GraphCacheKey>),
}

impl From<&str> for GraphCacheKey {
    fn from(value: &str) -> Self {
        GraphCacheKey::String(value.to_string())
    }
}

impl From<String> for GraphCacheKey {
    fn from(value: String) -> Self {
        GraphCacheKey::String(value)
    }
}

/// RPython `class CallFamily(object)` (description.py:11-59).
///
/// "A family of Desc objects that could be called from common call
/// sites. The call families are conceptually a partition of all
/// (callable) Desc objects, where the equivalence relation is the
/// transitive closure of `d1~d2 if d1 and d2 might be called at the
/// same call site`."
#[derive(Debug)]
pub struct CallFamily {
    /// RPython `self.descs = {desc: True}` (description.py:21).
    pub descs: HashMap<DescKey, ()>,
    /// RPython `self.calltables = {}` (description.py:22).
    pub calltables: HashMap<CallShape, Vec<CallTableRow>>,
    /// RPython `self.total_calltable_size = 0` (description.py:23).
    pub total_calltable_size: usize,
    /// RPython `CallFamily.normalized = False` class-level default
    /// (description.py:17).
    pub normalized: bool,
    /// RPython `CallFamily.modified = True` class-level default
    /// (description.py:18).
    pub modified: bool,
}

impl CallFamily {
    /// RPython `CallFamily.__init__(desc)` (description.py:20-23).
    pub fn new(desc: DescKey) -> Self {
        let mut descs = HashMap::new();
        descs.insert(desc, ());
        CallFamily {
            descs,
            calltables: HashMap::new(),
            total_calltable_size: 0,
            normalized: false,
            modified: true,
        }
    }

    /// RPython `CallFamily.update(other)` (description.py:25-31).
    ///
    /// `absorb = update` (description.py:32) is the UnionFind
    /// alias — Rust exposes [`Self::absorb`] as a thin delegator.
    pub fn update(&mut self, other: &CallFamily) {
        self.modified = true;
        self.normalized = self.normalized || other.normalized;
        for (k, v) in &other.descs {
            self.descs.insert(*k, *v);
        }
        for (shape, table) in &other.calltables {
            for row in table {
                self.calltable_add_row(shape.clone(), row.clone());
            }
        }
    }

    /// RPython `CallFamily.absorb = update` (description.py:32) —
    /// UnionFind API entry point.
    pub fn absorb(&mut self, other: &CallFamily) {
        self.update(other);
    }

    /// RPython `CallFamily.calltable_lookup_row(callshape, row)`
    /// (description.py:34-43).
    ///
    /// Upstream raises `LookupError` on miss; Rust returns
    /// `Option<usize>` so the miss is lossless.
    ///
    /// Row equality uses pointer identity on
    /// `Rc<FunctionGraph>` values — upstream's dict-`==` is identity
    /// comparison on the graph objects since `FunctionGraph` is not
    /// `__eq__`-overridden.
    pub fn calltable_lookup_row(&self, callshape: &CallShape, row: &CallTableRow) -> Option<usize> {
        let table = self.calltables.get(callshape)?;
        table
            .iter()
            .position(|existing_row| row_eq(existing_row, row))
    }

    /// RPython `CallFamily.calltable_add_row(callshape, row)`
    /// (description.py:45-52).
    pub fn calltable_add_row(&mut self, callshape: CallShape, row: CallTableRow) {
        if self.calltable_lookup_row(&callshape, &row).is_none() {
            self.modified = true;
            let table = self.calltables.entry(callshape).or_default();
            table.push(row);
            self.total_calltable_size += 1;
        }
    }

    /// RPython `CallFamily.find_row(bookkeeper, descs, args, op)`
    /// (description.py:54-59).
    pub fn find_row(
        &self,
        bookkeeper: &Rc<Bookkeeper>,
        descs: &[DescEntry],
        args: &ArgumentsForTranslation,
        op_key: Option<PositionKey>,
    ) -> Result<(CallShape, usize), AnnotatorError> {
        let shape = args.rawshape();
        let _guard = bookkeeper.at_position(None);
        let row = build_calltable_row(descs, args, op_key)?;
        let index = self.calltable_lookup_row(&shape, &row).ok_or_else(|| {
            AnnotatorError::new("LookupError: calltable row not found in CallFamily")
        })?;
        Ok((shape, index))
    }
}

impl UnionFindInfo for Rc<RefCell<CallFamily>> {
    fn absorb(&mut self, other: Self) {
        self.borrow_mut().absorb(&other.borrow());
    }
}

/// RPython `class FrozenAttrFamily(object)` (description.py:71-96).
#[derive(Debug)]
pub struct FrozenAttrFamily {
    /// RPython `self.descs = {desc: True}` (description.py:79).
    pub descs: HashMap<DescKey, ()>,
    /// RPython `self.read_locations = {}` (description.py:80).
    pub read_locations: HashMap<super::bookkeeper::PositionKey, ()>,
    /// RPython `self.attrs = {}` (description.py:81).
    pub attrs: HashMap<String, SomeValue>,
}

impl FrozenAttrFamily {
    /// RPython `FrozenAttrFamily.__init__(desc)` (description.py:78-81).
    pub fn new(desc: DescKey) -> Self {
        let mut descs = HashMap::new();
        descs.insert(desc, ());
        FrozenAttrFamily {
            descs,
            read_locations: HashMap::new(),
            attrs: HashMap::new(),
        }
    }

    /// RPython `FrozenAttrFamily.update(other)` (description.py:83-86).
    pub fn update(&mut self, other: &FrozenAttrFamily) {
        for (k, v) in &other.descs {
            self.descs.insert(*k, *v);
        }
        for (k, v) in &other.read_locations {
            self.read_locations.insert(k.clone(), *v);
        }
        for (k, v) in &other.attrs {
            self.attrs.insert(k.clone(), v.clone());
        }
    }

    /// RPython `FrozenAttrFamily.absorb = update` (description.py:87).
    pub fn absorb(&mut self, other: &FrozenAttrFamily) {
        self.update(other);
    }

    /// RPython `FrozenAttrFamily.get_s_value(attrname)`
    /// (description.py:89-93).
    pub fn get_s_value(&self, attrname: &str) -> SomeValue {
        match self.attrs.get(attrname) {
            Some(v) => v.clone(),
            None => s_impossible_value(),
        }
    }

    /// RPython `FrozenAttrFamily.set_s_value(attrname, s_value)`
    /// (description.py:95-96).
    pub fn set_s_value(&mut self, attrname: impl Into<String>, s_value: SomeValue) {
        self.attrs.insert(attrname.into(), s_value);
    }
}

impl UnionFindInfo for Rc<RefCell<FrozenAttrFamily>> {
    fn absorb(&mut self, other: Self) {
        self.borrow_mut().absorb(&other.borrow());
    }
}

/// RPython `class ClassAttrFamily(object)` (description.py:99-128).
///
/// "A family of ClassDesc objects that have common 'getattr' sites
/// for a given attribute name."
#[derive(Debug)]
pub struct ClassAttrFamily {
    /// RPython `self.descs = {desc: True}` (description.py:114).
    pub descs: HashMap<DescKey, ()>,
    /// RPython `self.read_locations = {}` (description.py:115).
    pub read_locations: HashMap<super::bookkeeper::PositionKey, ()>,
    /// RPython `self.s_value = s_ImpossibleValue` (description.py:116).
    pub s_value: SomeValue,
    /// Upstream sets this dynamically in
    /// `rpython.rtyper.normalizecalls.merge_classpbc_getattr_into_classdef`
    /// (normalizecalls.py:232) — Python attaches it to the live family
    /// instance after computing the common base of `descs`. The Rust
    /// port stores it as a real field that defaults to `None` and is
    /// populated by [`crate::translator::rtyper::normalizecalls::
    /// merge_classpbc_getattr_into_classdef`]. Consumed by
    /// `ClassesPBCRepr.get_access_set` (rpbc.py:946) to look up the
    /// `ClassRepr` that hosts the shared vtable slot for `attrname`.
    pub commonbase: Option<std::rc::Rc<std::cell::RefCell<super::classdesc::ClassDef>>>,
}

impl ClassAttrFamily {
    /// RPython `ClassAttrFamily.__init__(desc)` (description.py:113-116).
    pub fn new(desc: DescKey) -> Self {
        let mut descs = HashMap::new();
        descs.insert(desc, ());
        ClassAttrFamily {
            descs,
            read_locations: HashMap::new(),
            s_value: s_impossible_value(),
            commonbase: None,
        }
    }

    /// RPython `ClassAttrFamily.update(other)` (description.py:118-121).
    ///
    /// Returns `Result<(), UnionError>` because the attribute `union`
    /// can fail (`UnionError` bubbles out of any pair-union branch).
    pub fn update(&mut self, other: &ClassAttrFamily) -> Result<(), super::model::UnionError> {
        for (k, v) in &other.descs {
            self.descs.insert(*k, *v);
        }
        for (k, v) in &other.read_locations {
            self.read_locations.insert(k.clone(), *v);
        }
        self.s_value = union(&self.s_value, &other.s_value)?;
        Ok(())
    }

    /// RPython `ClassAttrFamily.absorb = update` (description.py:122).
    pub fn absorb(&mut self, other: &ClassAttrFamily) -> Result<(), super::model::UnionError> {
        self.update(other)
    }

    /// RPython `ClassAttrFamily.get_s_value(attrname)`
    /// (description.py:124-125). `attrname` is carried explicitly by
    /// upstream for interface symmetry with `FrozenAttrFamily`, though
    /// it is ignored in favour of the single-attribute
    /// `self.s_value`.
    pub fn get_s_value(&self, _attrname: &str) -> SomeValue {
        self.s_value.clone()
    }

    /// RPython `ClassAttrFamily.set_s_value(attrname, s_value)`
    /// (description.py:127-128).
    pub fn set_s_value(&mut self, _attrname: &str, s_value: SomeValue) {
        self.s_value = s_value;
    }
}

impl UnionFindInfo for Rc<RefCell<ClassAttrFamily>> {
    fn absorb(&mut self, other: Self) {
        // `absorb = update` (description.py:122). `update` propagates a
        // `UnionError`; `UnionFindInfo` has no Result channel, so we
        // swallow here — the error surfaces at the next
        // `get_s_value` / set_s_value consumer if the value becomes
        // malformed.
        let _ = self.borrow_mut().absorb(&other.borrow());
    }
}

// ---------------------------------------------------------------------------
// Desc + FunctionDesc (description.py:132-406).
// ---------------------------------------------------------------------------

/// RPython `class Desc(object)` (description.py:132-182).
///
/// Base class for every "description" the bookkeeper interns. Rust has
/// no single inheritance, so the subclass-specific state lives on
/// [`FunctionDesc`] / [`MethodDesc`] / [`FrozenDesc`] /
/// [`MethodOfFrozenDesc`] (commits 2-3) which each wrap a `Desc` via
/// composition. Upstream-facing methods `querycallfamily` /
/// `getcallfamily` / `mergecallfamilies` / `queryattrfamily` /
/// `bind_under` / `simplify_desc_set` live here because upstream
/// dispatches them on every `Desc` instance.
#[derive(Debug)]
pub struct Desc {
    /// Upstream identity key used by the bookkeeper's UnionFind
    /// tables. Python uses the Desc object itself; Rust assigns a
    /// stable key at construction time so `&self` methods can still
    /// address the same partition.
    pub identity: DescKey,
    /// RPython `self.bookkeeper` (description.py:136).
    pub bookkeeper: Rc<Bookkeeper>,
    /// RPython `self.pyobj` (description.py:138). `None` matches
    /// upstream's `pyobj=None` default.
    pub pyobj: Option<HostObject>,
}

impl Desc {
    /// RPython `Desc.__init__(bookkeeper, pyobj=None)`
    /// (description.py:135-138).
    pub fn new(bookkeeper: Rc<Bookkeeper>, pyobj: Option<HostObject>) -> Self {
        Desc {
            identity: alloc_desc_key(),
            bookkeeper,
            pyobj,
        }
    }

    /// RPython `Desc.querycallfamily()` (description.py:146-153).
    ///
    pub fn querycallfamily(&self) -> Option<Rc<RefCell<CallFamily>>> {
        self.bookkeeper
            .pbc_maximal_call_families
            .borrow_mut()
            .get(&self.identity)
            .cloned()
    }

    /// RPython `Desc.getcallfamily()` (description.py:155-159).
    ///
    pub fn getcallfamily(&self) -> Result<Rc<RefCell<CallFamily>>, AnnotatorError> {
        let mut families = self.bookkeeper.pbc_maximal_call_families.borrow_mut();
        let rep = families.find_rep(self.identity);
        Ok(families
            .get(&rep)
            .cloned()
            .expect("UnionFind.find_rep() must materialize a CallFamily"))
    }

    /// RPython `Desc.mergecallfamilies(*others)` (description.py:161-170).
    ///
    pub fn mergecallfamilies(&self, others: &[&Desc]) -> Result<bool, AnnotatorError> {
        if others.is_empty() {
            return Ok(false);
        }
        let mut changed = false;
        let mut families = self.bookkeeper.pbc_maximal_call_families.borrow_mut();
        let mut rep = families.find_rep(self.identity);
        for desc in others {
            let (changed1, new_rep) = families.union(rep, desc.identity);
            changed |= changed1;
            rep = new_rep;
        }
        Ok(changed)
    }

    /// RPython `Desc.queryattrfamily()` (description.py:172-175).
    ///
    /// Base implementation returns `None` — upstream overrides on
    /// [`FrozenDesc`] / `ClassDesc` (commits 3+).
    pub fn queryattrfamily(&self) -> Option<()> {
        None
    }

    /// Pointer-identity key for [`CallFamily`] / `FrozenAttrFamily` /
    /// `ClassAttrFamily` storage. Upstream uses `desc` itself as the
    /// key (dict-by-identity).
    pub fn desc_key(self_rc: &Rc<RefCell<Desc>>) -> DescKey {
        DescKey::from_rc(self_rc)
    }
}

/// RPython `Desc.simplify_desc_set(descs)` — `@staticmethod` base
/// (description.py:180-182). No-op; only [`MethodDesc::simplify_desc_set`]
/// overrides it. Dispatched polymorphically from
/// [`super::model::SomePBC::simplify`].
pub(crate) fn simplify_desc_set_default(
    _descs: &mut std::collections::BTreeMap<DescKey, DescEntry>,
) {
}

/// PRE-EXISTING-ADAPTATION: single-inheritance Desc hierarchy in Rust.
///
/// Upstream `rpython/annotator/description.py` + `classdesc.py` define
/// `FunctionDesc / MethodDesc / ClassDesc / FrozenDesc /
/// MethodOfFrozenDesc / MemoDesc` as subclasses of a common `Desc`
/// base. Python `isinstance(desc, FunctionDesc)` branches at every
/// bookkeeper callsite. The Rust port uses composition — each subclass
/// wraps [`Desc`] as a `base` field — so the common storage shape for
/// `bookkeeper.descs` needs a discriminated container. [`DescEntry`]
/// is that container: one variant per upstream subclass, each
/// variant holding `Rc<RefCell<SubclassDesc>>`. Identity (upstream
/// `desc is other_desc`) routes through [`DescEntry::desc_key`] which
/// returns the inner `Rc::as_ptr` identity.
///
/// Callsite mapping (upstream → Rust):
/// * `isinstance(desc, FunctionDesc)` → `matches!(entry,
///   DescEntry::Function(_))`
/// * `desc.pycall(...)` → per-variant dispatch via `match entry { ... }`
/// * `bookkeeper.descs[pyobj] = result` →
///   `bookkeeper.descs.insert(pyobj, entry)`
#[derive(Clone, Debug)]
pub enum DescEntry {
    /// upstream `FunctionDesc` — ported file description.py:190-393.
    Function(Rc<RefCell<FunctionDesc>>),
    /// upstream `MethodDesc` — description.py:407-519.
    Method(Rc<RefCell<MethodDesc>>),
    /// upstream `FrozenDesc` — description.py:528-599.
    Frozen(Rc<RefCell<FrozenDesc>>),
    /// upstream `MethodOfFrozenDesc` — description.py:601-637.
    MethodOfFrozen(Rc<RefCell<MethodOfFrozenDesc>>),
    /// upstream `ClassDesc` — classdesc.py:488-918. The full
    /// `__init__` body is ported at [`super::classdesc::ClassDesc::new`];
    /// [`super::classdesc::ClassDesc::new_shell`] is the test-only
    /// shell that skips mixin resolution and `add_source_attribute`.
    Class(Rc<RefCell<super::classdesc::ClassDesc>>),
}

impl DescEntry {
    /// RPython `id(desc)` — pointer-identity handle. Used as the dict
    /// key by `CallFamily.descs`, `FrozenAttrFamily.descs`, and
    /// `ClassAttrFamily.descs`.
    pub fn desc_key(&self) -> DescKey {
        match self {
            DescEntry::Function(rc) => DescKey::from_rc(rc),
            DescEntry::Method(rc) => DescKey::from_rc(rc),
            DescEntry::Frozen(rc) => DescKey::from_rc(rc),
            DescEntry::MethodOfFrozen(rc) => DescKey::from_rc(rc),
            DescEntry::Class(rc) => DescKey::from_rc(rc),
        }
    }

    /// RPython `desc.pyobj` — the wrapped host object. FunctionDesc /
    /// FrozenDesc / ClassDesc all carry one. MethodDesc /
    /// MethodOfFrozenDesc return `None` (upstream `MethodDesc.pyobj is
    /// None`).
    pub fn pyobj(&self) -> Option<HostObject> {
        match self {
            DescEntry::Function(rc) => rc.borrow().base.pyobj.clone(),
            DescEntry::Method(_) => None,
            DescEntry::Frozen(rc) => rc.borrow().base.pyobj.clone(),
            DescEntry::MethodOfFrozen(_) => None,
            DescEntry::Class(rc) => Some(rc.borrow().pyobj.clone()),
        }
    }

    /// RPython `type(desc)` classifier — maps enum variants back to
    /// the [`super::model::DescKind`] used by
    /// [`super::model::SomePBC::getKind`] (model.py:560-566).
    pub fn kind(&self) -> super::model::DescKind {
        match self {
            DescEntry::Function(_) => super::model::DescKind::Function,
            DescEntry::Method(_) => super::model::DescKind::Method,
            DescEntry::Class(_) => super::model::DescKind::Class,
            DescEntry::Frozen(_) => super::model::DescKind::Frozen,
            DescEntry::MethodOfFrozen(_) => super::model::DescKind::MethodOfFrozen,
        }
    }

    /// RPython `Desc.bind_under(classdef, name)` (description.py:177-178,
    /// 350-355, 447-449).
    ///
    /// Default: return self (Desc.bind_under on Frozen/Class/MethodOfFrozen).
    /// Function: delegate to `FunctionDesc.bind_under` → `MethodDesc`.
    /// Method: warn + re-bind via `funcdesc.bind_under`.
    pub fn bind_under(
        &self,
        classdef: &Rc<RefCell<super::classdesc::ClassDef>>,
        name: &str,
    ) -> DescEntry {
        match self {
            DescEntry::Function(fd) => {
                DescEntry::Method(FunctionDesc::bind_under(fd, classdef, name))
            }
            DescEntry::Method(md) => DescEntry::Method(md.borrow().bind_under(classdef, name)),
            // upstream: `Desc.bind_under` default = `return self`.
            DescEntry::Frozen(_) | DescEntry::MethodOfFrozen(_) | DescEntry::Class(_) => {
                self.clone()
            }
        }
    }

    /// RPython `desc.s_read_attribute(name)` dispatch — mirrors the
    /// upstream polymorphism where `FrozenDesc` / `ClassDesc` define
    /// the method and the remaining subclasses leave it absent
    /// (upstream raises `AttributeError` at the Python level when
    /// reached). `pbc_getattr` (bookkeeper.py:465, 475) consumes the
    /// result directly so the contract here is:
    ///
    /// * `Frozen(fd)` → [`FrozenDesc::s_read_attribute`] (description.py:560-566)
    /// * `Class(cd)` → [`super::classdesc::ClassDesc::s_read_attribute`] (classdesc.py:775-782)
    /// * Function / Method / MethodOfFrozen → `AnnotatorError` matching
    ///   upstream's "no s_read_attribute on this Desc" AttributeError.
    pub fn s_read_attribute(&self, name: &str) -> Result<super::model::SomeValue, AnnotatorError> {
        match self {
            DescEntry::Frozen(rc) => rc.borrow().s_read_attribute(name),
            DescEntry::Class(rc) => super::classdesc::ClassDesc::s_read_attribute(rc, name),
            DescEntry::Function(_) | DescEntry::Method(_) | DescEntry::MethodOfFrozen(_) => {
                Err(AnnotatorError::new(format!(
                    "AttributeError: {:?} has no s_read_attribute (pbc_getattr of {:?})",
                    self.kind(),
                    name
                )))
            }
        }
    }

    /// RPython `desc.get_call_parameters(args_s)` polymorphism as used
    /// by `annrpython.py:96`. Upstream calls the method directly on the
    /// descriptor object; only `FunctionDesc` implements it, so reaching
    /// any other variant is the Python-level "missing attribute"
    /// condition rather than a Rust-specific type assertion.
    pub fn get_call_parameters(
        &self,
        args_s: Vec<super::model::SomeValue>,
    ) -> Result<(Rc<PyGraph>, Vec<super::model::SomeValue>), AnnotatorError> {
        match self {
            DescEntry::Function(rc) => rc.borrow().get_call_parameters(args_s),
            DescEntry::Method(_)
            | DescEntry::Frozen(_)
            | DescEntry::MethodOfFrozen(_)
            | DescEntry::Class(_) => Err(AnnotatorError::new(format!(
                "AttributeError: {:?} has no get_call_parameters",
                self.kind()
            ))),
        }
    }

    /// Shorthand predicates matching upstream `isinstance(desc, ...)`
    /// dispatch at bookkeeper callsites.
    pub fn is_function(&self) -> bool {
        matches!(self, DescEntry::Function(_))
    }
    pub fn is_method(&self) -> bool {
        matches!(self, DescEntry::Method(_))
    }
    pub fn is_frozen(&self) -> bool {
        matches!(self, DescEntry::Frozen(_))
    }
    pub fn is_method_of_frozen(&self) -> bool {
        matches!(self, DescEntry::MethodOfFrozen(_))
    }
    pub fn is_class(&self) -> bool {
        matches!(self, DescEntry::Class(_))
    }

    /// Variant-specific getter. Returns `None` when `self` is a
    /// different variant — the caller is expected to branch via
    /// `is_function()` first.
    pub fn as_function(&self) -> Option<Rc<RefCell<FunctionDesc>>> {
        match self {
            DescEntry::Function(rc) => Some(rc.clone()),
            _ => None,
        }
    }
    pub fn as_class(&self) -> Option<Rc<RefCell<super::classdesc::ClassDesc>>> {
        match self {
            DescEntry::Class(rc) => Some(rc.clone()),
            _ => None,
        }
    }
    pub fn as_frozen(&self) -> Option<Rc<RefCell<FrozenDesc>>> {
        match self {
            DescEntry::Frozen(rc) => Some(rc.clone()),
            _ => None,
        }
    }
    pub fn as_method(&self) -> Option<Rc<RefCell<MethodDesc>>> {
        match self {
            DescEntry::Method(rc) => Some(rc.clone()),
            _ => None,
        }
    }
    pub fn as_method_of_frozen(&self) -> Option<Rc<RefCell<MethodOfFrozenDesc>>> {
        match self {
            DescEntry::MethodOfFrozen(rc) => Some(rc.clone()),
            _ => None,
        }
    }

    /// Upstream `desc.get_graph(args, op)` polymorphic dispatch used by
    /// `build_calltable_row` (description.py:62-68).
    pub fn get_graph(
        &self,
        args: &ArgumentsForTranslation,
        op_key: Option<PositionKey>,
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        match self {
            DescEntry::Function(rc) => rc.borrow().get_graph(args, op_key),
            DescEntry::Method(rc) => rc.borrow().get_graph(args, op_key),
            DescEntry::MethodOfFrozen(rc) => rc.borrow().get_graph(args, op_key),
            DescEntry::Frozen(_) | DescEntry::Class(_) => Err(AnnotatorError::new(format!(
                "{:?} has no get_graph() for build_calltable_row",
                self.kind()
            ))),
        }
    }

    /// Upstream `desc.rowkey()` polymorphic dispatch used by
    /// `build_calltable_row` (description.py:66-67).
    pub fn rowkey(&self) -> Result<DescKey, AnnotatorError> {
        match self {
            DescEntry::Function(rc) => Ok(FunctionDesc::rowkey(rc)),
            DescEntry::Method(rc) => Ok(rc.borrow().rowkey()),
            DescEntry::MethodOfFrozen(rc) => Ok(rc.borrow().rowkey()),
            DescEntry::Frozen(_) | DescEntry::Class(_) => Err(AnnotatorError::new(format!(
                "{:?} has no rowkey() for build_calltable_row",
                self.kind()
            ))),
        }
    }
}

/// Identity equality — two [`DescEntry`]s are equal iff they wrap the
/// same underlying `Rc`. Upstream `desc is other_desc`.
impl PartialEq for DescEntry {
    fn eq(&self, other: &Self) -> bool {
        self.desc_key() == other.desc_key()
    }
}

impl Eq for DescEntry {}

/// RPython `class FunctionDesc(Desc)` (description.py:190-393).
///
/// "The 'FunctionDesc' wraps a Python function or method." Fields
/// match upstream line-by-line. Graph-building / specialization /
/// pycall paths are wired to `RPythonAnnotator.recursivecall`.
#[derive(Debug)]
pub struct FunctionDesc {
    /// Embedded base-class state (description.py:195 `super().__init__`).
    pub base: Desc,
    /// RPython `FunctionDesc.knowntype = types.FunctionType`
    /// (description.py:191). Carried as a tag field for parity with
    /// upstream class-level attribute.
    pub knowntype: FunctionDescKnownType,
    /// RPython `self.name` (description.py:196).
    pub name: String,
    /// RPython `self.signature` (description.py:197).
    pub signature: Signature,
    /// RPython `self.defaults` (description.py:198) — defaults to an
    /// empty tuple when upstream receives `None`. Stored as
    /// `Vec<Constant>` so `parse_arguments` can call `bookkeeper.
    /// immutablevalue` at use time (matching upstream's deferred
    /// conversion) and `getuniquegraph` can compare against
    /// `PyGraph.defaults` structurally (description.py:223-225).
    pub defaults: Vec<Constant>,
    /// RPython `self.specializer` (description.py:202).
    pub specializer: Option<Specializer>,
    /// RPython `self._cache` (description.py:203) — keyed by the
    /// specialization result.
    ///
    /// Upstream keys entries with arbitrary Python values
    /// (specializer-dependent). The Rust port mirrors that shape with
    /// [`GraphCacheKey`], which preserves the tuple / `None` /
    /// descriptor / type distinctions used by `specialize.py`.
    /// Values are [`PyGraph`] because
    /// `buildgraph` / `translator.buildflowgraph` produces
    /// `PyGraph` instances; this is what lets `getuniquegraph`
    /// access `graph.signature` / `graph.defaults` for the
    /// description.py:223-225 comparison.
    pub(crate) cache: RefCell<HashMap<GraphCacheKey, Rc<PyGraph>>>,
    /// Upstream `self.pyobj._signature_` (description.py:294, :315).
    /// Carried on FunctionDesc rather than HostObject because
    /// `_signature_` is a function-level attribute.
    ///
    /// Stored as a tuple `(params, return)` matching
    /// upstream `_signature_ = ([ParamType], ParamType)`; the inner
    /// vector of [`ParamType`] is consumed by
    /// [`super::signature::enforce_signature_args`].
    pub annsignature: Option<Rc<AnnSignature>>,
    /// Upstream `self.pyobj._annenforceargs_` (description.py:314-324).
    /// Carried directly on FunctionDesc for the same reason as
    /// [`Self::annsignature`]. `Some(sig)` means an `@enforceargs(...)`
    /// decoration was applied and `normalize_args` should invoke it.
    pub annenforceargs: Option<Rc<super::signature::Sig>>,
}

/// RPython `FunctionDesc.knowntype = types.FunctionType` class-level
/// attribute (description.py:191). Carried as a tag so subclasses can
/// override (e.g. `MemoDesc` uses the same type).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionDescKnownType {
    /// Upstream `types.FunctionType`.
    Function,
    /// Upstream `types.BuiltinFunctionType` for certain frozen PBCs.
    BuiltinFunction,
}

/// Upstream `_signature_` tuple (params_s, result_s) attached to
/// functions via `@signature(...)`. Stored as a named struct so it
/// can live behind a single `Option<Rc<_>>` field.
pub struct AnnSignature {
    /// Parameter type declarations — one [`ParamType`] per argument.
    pub params: Vec<ParamType>,
    /// Declared return type.
    pub result: ParamType,
}

impl std::fmt::Debug for AnnSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnnSignature")
            .field("params", &self.params.len())
            .field("result", &self.result)
            .finish()
    }
}

/// RPython `build_calltable_row(descs, args, op)` (description.py:62-68).
pub fn build_calltable_row(
    descs: &[DescEntry],
    args: &ArgumentsForTranslation,
    op_key: Option<PositionKey>,
) -> Result<CallTableRow, AnnotatorError> {
    let mut row = CallTableRow::new();
    for desc in descs {
        let graph = desc.get_graph(args, op_key.clone())?;
        row.insert(desc.rowkey()?, graph);
    }
    Ok(row)
}

impl FunctionDesc {
    /// RPython `FunctionDesc.__init__(bookkeeper, pyobj, name,
    /// signature, defaults, specializer=None)` (description.py:193-203).
    pub fn new(
        bookkeeper: Rc<Bookkeeper>,
        pyobj: Option<HostObject>,
        name: impl Into<String>,
        signature: Signature,
        defaults: Option<Vec<Constant>>,
        specializer: Option<Specializer>,
    ) -> Self {
        FunctionDesc {
            base: Desc::new(bookkeeper, pyobj),
            knowntype: FunctionDescKnownType::Function,
            name: name.into(),
            signature,
            // upstream: `self.defaults = defaults if defaults is not None else ()`.
            defaults: defaults.unwrap_or_default(),
            specializer,
            cache: RefCell::new(HashMap::new()),
            annsignature: None,
            annenforceargs: None,
        }
    }

    /// RPython `FunctionDesc.getgraphs()` (description.py:215-216).
    pub fn getgraphs(&self) -> Vec<Rc<PyGraph>> {
        self.cache.borrow().values().cloned().collect()
    }

    /// RPython `FunctionDesc.rowkey()` (description.py:365-366).
    /// Returns self's identity — upstream `return self`.
    pub fn rowkey(self_rc: &Rc<RefCell<FunctionDesc>>) -> DescKey {
        self_rc.borrow().base.identity
    }

    fn key_to_cache_name(key: &str) -> String {
        let mut out = String::with_capacity(key.len());
        for ch in key.chars() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                out.push(ch);
            } else {
                out.push('_');
            }
        }
        out
    }

    fn nameof_const(value: &ConstValue) -> String {
        match value {
            ConstValue::Int(i) => i.to_string(),
            ConstValue::Bool(b) => {
                if *b {
                    "True".to_string()
                } else {
                    "False".to_string()
                }
            }
            ConstValue::ByteStr(s) => String::from_utf8_lossy(s).into_owned(),
            ConstValue::UniStr(s) => s.clone(),
            ConstValue::HostObject(obj) => obj
                .qualname()
                .rsplit('.')
                .next()
                .unwrap_or(obj.qualname())
                .to_string(),
            other => format!("{other:?}"),
        }
    }

    fn nameof_desc(desc: &DescEntry) -> String {
        match desc {
            DescEntry::Function(rc) => rc.borrow().name.clone(),
            DescEntry::Method(rc) => rc.borrow().name.clone(),
            DescEntry::MethodOfFrozen(rc) => rc.borrow().funcdesc.borrow().name.clone(),
            DescEntry::Class(rc) => rc.borrow().name.clone(),
            DescEntry::Frozen(rc) => rc
                .borrow()
                .base
                .pyobj
                .as_ref()
                .map(|obj| {
                    obj.qualname()
                        .rsplit('.')
                        .next()
                        .unwrap_or(obj.qualname())
                        .to_string()
                })
                .unwrap_or_else(|| format!("{:?}", desc.desc_key())),
        }
    }

    fn nameof_cache_key(key: &GraphCacheKey) -> String {
        fn nameof_somevalue_tag(tag: SomeValueTag) -> &'static str {
            match tag {
                SomeValueTag::Impossible => "SomeImpossibleValue",
                SomeValueTag::Object => "SomeObject",
                SomeValueTag::Type => "SomeType",
                SomeValueTag::Float => "SomeFloat",
                SomeValueTag::SingleFloat => "SomeSingleFloat",
                SomeValueTag::LongFloat => "SomeLongFloat",
                SomeValueTag::Integer => "SomeInteger",
                SomeValueTag::Bool => "SomeBool",
                SomeValueTag::String => "SomeString",
                SomeValueTag::UnicodeString => "SomeUnicodeString",
                SomeValueTag::ByteArray => "SomeByteArray",
                SomeValueTag::Char => "SomeChar",
                SomeValueTag::UnicodeCodePoint => "SomeUnicodeCodePoint",
                SomeValueTag::List => "SomeList",
                SomeValueTag::Tuple => "SomeTuple",
                SomeValueTag::Dict => "SomeDict",
                SomeValueTag::Iterator => "SomeIterator",
                SomeValueTag::Instance => "SomeInstance",
                SomeValueTag::Exception => "SomeException",
                SomeValueTag::PBC => "SomePBC",
                SomeValueTag::None_ => "SomeNone",
                SomeValueTag::Property => "SomeProperty",
                SomeValueTag::Ptr => "SomePtr",
                SomeValueTag::InteriorPtr => "SomeInteriorPtr",
                SomeValueTag::LLADTMeth => "SomeLLADTMeth",
                SomeValueTag::Builtin => "SomeBuiltin",
                SomeValueTag::BuiltinMethod => "SomeBuiltinMethod",
                SomeValueTag::WeakRef => "SomeWeakRef",
                SomeValueTag::TypeOf => "SomeTypeOf",
            }
        }

        match key {
            GraphCacheKey::None => "None".to_string(),
            GraphCacheKey::Const(value) => Self::nameof_const(value),
            GraphCacheKey::LowLevelType(lltype) => lltype.short_name(),
            GraphCacheKey::KnownType(known_type) => known_type.to_string(),
            GraphCacheKey::Desc { name, .. } => name.clone(),
            GraphCacheKey::KeyComp(keycomp) => keycomp.to_string(),
            GraphCacheKey::String(value) => value.clone(),
            GraphCacheKey::Int(value) => value.to_string(),
            GraphCacheKey::Position(pk) => {
                format!(
                    "call_location_g{:x}_b{:x}_i{}",
                    pk.graph_id, pk.block_id, pk.op_index
                )
            }
            GraphCacheKey::AccessDirect => "AccessDirect".to_string(),
            GraphCacheKey::SomeValueTag(tag) => nameof_somevalue_tag(*tag).to_string(),
            GraphCacheKey::Tuple(items) => items
                .iter()
                .map(Self::nameof_cache_key)
                .collect::<Vec<_>>()
                .join("_"),
        }
    }

    fn append_cache_key(
        &self,
        key: GraphCacheKey,
        suffix: GraphCacheKey,
    ) -> Result<GraphCacheKey, AnnotatorError> {
        match (key, suffix) {
            (key, GraphCacheKey::None) => Ok(key),
            (GraphCacheKey::Tuple(mut left), GraphCacheKey::Tuple(right)) => {
                left.extend(right);
                Ok(GraphCacheKey::Tuple(left))
            }
            (GraphCacheKey::None, suffix) => Err(AnnotatorError::new(format!(
                "{}.maybe_star_args: cannot combine None cache key with {:?}",
                self.name, suffix
            ))),
            (left, right) => Err(AnnotatorError::new(format!(
                "{}.maybe_star_args: expected tuple cache keys, got {:?} and {:?}",
                self.name, left, right
            ))),
        }
    }

    /// RPython `FunctionDesc.cachedgraph(key, alt_name=None, builder=None)`
    /// (description.py:228-248).
    pub(crate) fn cachedgraph(
        &self,
        key: GraphCacheKey,
        alt_name: Option<&str>,
        builder: Option<GraphBuilder<'_>>,
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        if let Some(existing) = self.cache.borrow().get(&key) {
            return Ok(existing.clone());
        }
        let computed_alt_name = alt_name.map(str::to_string).or_else(|| {
            if matches!(&key, GraphCacheKey::None) {
                None
            } else {
                let postfix = Self::key_to_cache_name(&Self::nameof_cache_key(&key));
                Some(format!("{}__{}", self.name, postfix))
            }
        });
        let graph = self.buildgraph(computed_alt_name.as_deref(), builder)?;
        self.cache.borrow_mut().insert(key, graph.clone());
        Ok(graph)
    }

    /// RPython `FunctionDesc.parse_arguments(args, graph=None)`
    /// (description.py:251-270).
    ///
    /// Functional port. Defaults are walked through
    /// [`Bookkeeper::immutablevalue`]; `args.match_signature` does
    /// the positional / kwarg / default layout, matching upstream's
    /// `ArgErr → AnnotatorError` wrap.
    pub fn parse_arguments(
        &self,
        args: &ArgumentsForTranslation,
        graph: Option<&PyGraph>,
    ) -> Result<Vec<SomeValue>, AnnotatorError> {
        // upstream description.py:256-264 — `if defaults:` walks each
        // Python value through `self.bookkeeper.immutablevalue(x)` at
        // parse time. `graph_defaults` overrides `self.defaults` when
        // the specializer built a `PyGraph` whose `__defaults__` tuple
        // diverged from the bare FunctionDesc's.
        let mut defs_s: Vec<SomeValue> = Vec::new();
        if let Some(graph) = graph {
            let signature = graph.signature.borrow();
            let graph_defaults = graph.defaults.borrow();
            for d in graph_defaults.as_deref().unwrap_or_default() {
                // upstream: `if x is NODEFAULT: defs_s.append(None)`.
                // NODEFAULT is not modelled as a dedicated sentinel; the
                // immutablevalue path returns `s_impossible_value` for the
                // placeholder Constant used to represent a missing default.
                let s = self.base.bookkeeper.immutablevalue(&d.value)?;
                defs_s.push(s);
            }
            return args
                .match_signature(&signature, Some(&defs_s))
                .map_err(|e: ArgErr| {
                    AnnotatorError::new(format!(
                        "signature mismatch: {}() {}",
                        self.name,
                        e.getmsg()
                    ))
                });
        }
        for d in self.defaults.as_slice() {
            // upstream: `if x is NODEFAULT: defs_s.append(None)`.
            // NODEFAULT is not modelled as a dedicated sentinel; the
            // immutablevalue path returns `s_impossible_value` for the
            // placeholder Constant used to represent a missing default.
            let s = self.base.bookkeeper.immutablevalue(&d.value)?;
            defs_s.push(s);
        }
        args.match_signature(&self.signature, Some(&defs_s))
            .map_err(|e: ArgErr| {
                AnnotatorError::new(format!(
                    "signature mismatch: {}() {}",
                    self.name,
                    e.getmsg()
                ))
            })
    }

    /// RPython `FunctionDesc.normalize_args(inputs_s)`
    /// (description.py:307-326).
    ///
    /// Honours `@enforceargs` and `@signature(...)` decorators carried
    /// via [`Self::annenforceargs`] / [`Self::annsignature`]. Upstream
    /// raises on having both simultaneously; the Rust port returns
    /// `AnnotatorError` with the same message.
    pub fn normalize_args(&self, inputs_s: &mut [SomeValue]) -> Result<(), AnnotatorError> {
        let enforceargs = self.annenforceargs.clone();
        let signature = self.annsignature.clone();
        if enforceargs.is_some() && signature.is_some() {
            return Err(AnnotatorError::new(format!(
                "{}: signature and enforceargs cannot both be used",
                self.name
            )));
        }
        if let Some(sig) = enforceargs {
            // upstream: `enforceargs(self, inputs_s)` — Sig.__call__
            // (signature.py:113-147).
            sig.call(&self.name, &self.base.bookkeeper, inputs_s)
                .map_err(|e| -> AnnotatorError { e.into() })?;
            return Ok(());
        }
        if let Some(sig) = signature {
            super::signature::enforce_signature_args(
                &self.name,
                &self.base.bookkeeper,
                &sig.params,
                inputs_s,
            )
            .map_err(|e| -> AnnotatorError { e.into() })?;
        }
        Ok(())
    }

    /// RPython `FunctionDesc.buildgraph(alt_name=None, builder=None)`
    /// (description.py:205-213).
    pub fn buildgraph(
        &self,
        alt_name: Option<&str>,
        builder: Option<GraphBuilder<'_>>,
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        let pyobj = self
            .base
            .pyobj
            .clone()
            .ok_or_else(|| AnnotatorError::new("FunctionDesc.buildgraph: missing pyobj"))?;
        let annotator = self.base.bookkeeper.annotator();
        let translator = &annotator.translator;
        let pygraph = match builder {
            Some(builder) => builder(translator, pyobj.clone())?,
            None => translator.buildflowgraph(pyobj, false).map_err(|err| {
                AnnotatorError::new(format!("FunctionDesc.buildgraph({}): {err}", self.name))
            })?,
        };
        if let Some(alt_name) = alt_name {
            pygraph.graph.borrow_mut().name = alt_name.to_string();
        }
        Ok(pygraph)
    }

    /// RPython `FunctionDesc.getuniquegraph()` (description.py:218-226).
    ///
    /// Three checks, in order:
    ///   1. `len(self._cache) != 1` → `NoStandardGraph`.
    ///   2. `graph.signature != self.signature or graph.defaults !=
    ///      self.defaults` (and not `relax_sig_check`) →
    ///      `NoStandardGraph`.
    ///   3. otherwise return the cached graph.
    ///
    /// `relax_sig_check` is a function-level attribute on upstream's
    /// Python function object that lowlevel annotator callers set to
    /// allow signature mismatches. The Rust `GraphFunc` carries this
    /// as an optional bool mirroring the same "attribute absent"
    /// default-to-False contract.
    pub fn getuniquegraph(&self) -> Result<Rc<PyGraph>, AnnotatorError> {
        let cache = self.cache.borrow();
        if cache.len() != 1 {
            return Err(AnnotatorError::new(format!(
                "NoStandardGraph({})",
                self.name
            )));
        }
        let (_, graph) = cache.iter().next().unwrap();
        // upstream description.py:222-225 — structural signature /
        // defaults match. `PyGraph.signature` / `PyGraph.defaults`
        // reach these via the composed `FunctionGraph` base (see
        // `pygraph.rs`); `FunctionDesc.defaults` is `Vec<Constant>`
        // specifically so this comparison is a plain value-eq.
        let relax_sig_check = self
            .base
            .pyobj
            .as_ref()
            .and_then(|pyobj| pyobj.user_function())
            .and_then(|func| func.relax_sig_check)
            .unwrap_or(false);
        let graph_signature = graph.signature.borrow();
        let graph_defaults = graph.defaults.borrow();
        if (*graph_signature != self.signature || graph_defaults.as_ref() != Some(&self.defaults))
            && !relax_sig_check
        {
            return Err(AnnotatorError::new(format!(
                "NoStandardGraph({}): signature/defaults mismatch",
                self.name
            )));
        }
        Ok(graph.clone())
    }

    /// RPython `flatten_star_args(funcdesc, args_s)`
    /// (specialize.py:14-58).
    ///
    /// Flattens the trailing `SomeTuple` for a `*arg` signature into
    /// individual argument annotations, returning the new args_s and
    /// an optional cache-key suffix (`"star_N"` where `N` is the
    /// flattened tuple length). For signatures without a `*arg` this
    /// is the identity: `(args_s, None)`.
    ///
    /// The star-args path uses an upstream-style builder closure to
    /// rewrite the fresh graph so it no longer takes a `*arg` tuple.
    pub(crate) fn flatten_star_args(
        &self,
        args_s: &[SomeValue],
    ) -> Result<(Vec<SomeValue>, GraphCacheKey, Option<GraphBuilder<'_>>), AnnotatorError> {
        // upstream: `argnames, vararg, kwarg = funcdesc.signature`.
        let vararg = self.signature.varargname.clone();
        let kwarg = self.signature.kwargname.clone();
        // upstream: `assert not kwarg`.
        if kwarg.is_some() {
            return Err(AnnotatorError::new(format!(
                "{}: functions with ** arguments are not supported",
                self.name
            )));
        }
        if vararg.is_none() {
            // upstream: `return args_s, None, None`.
            return Ok((args_s.to_vec(), GraphCacheKey::None, None));
        }
        // upstream: `assert len(args_s) == len(argnames) + 1`.
        let argnames_len = self.signature.argnames.len();
        if args_s.len() != argnames_len + 1 {
            return Err(AnnotatorError::new(format!(
                "{}.flatten_star_args: expected {} args, got {}",
                self.name,
                argnames_len + 1,
                args_s.len()
            )));
        }
        // upstream: `s_tuple = args_s[-1]; assert isinstance(s_tuple, SomeTuple)`.
        let s_tuple = &args_s[args_s.len() - 1];
        let tuple = match s_tuple {
            SomeValue::Tuple(t) => t,
            other => {
                return Err(AnnotatorError::new(format!(
                    "{}.flatten_star_args: *arg is {:?}, expected SomeTuple",
                    self.name, other
                )));
            }
        };
        // upstream: `s_len = s_tuple.len(); assert s_len.is_constant();
        //             nb_extra_args = s_len.const`.
        let nb_extra_args = tuple.items.len();
        // upstream: `flattened_s = list(args_s[:-1]); flattened_s.extend(s_tuple.items)`.
        let mut flattened_s: Vec<SomeValue> = args_s[..args_s.len() - 1].to_vec();
        flattened_s.extend(tuple.items.iter().cloned());
        // upstream: `key = ('star', nb_extra_args)`.
        let key_suffix = GraphCacheKey::Tuple(vec![
            GraphCacheKey::String("star".to_string()),
            GraphCacheKey::Int(nb_extra_args as i64),
        ]);
        let builder_name = self.name.clone();
        let builder: Option<GraphBuilder<'_>> = Some(Box::new(
            move |translator: &crate::translator::translator::TranslationContext,
                  func: HostObject| {
                let pygraph = translator.buildflowgraph(func, false).map_err(|err| {
                    AnnotatorError::new(format!("FunctionDesc.buildgraph({}): {err}", builder_name))
                })?;
                Self::apply_star_args_builder(pygraph.as_ref(), nb_extra_args)?;
                Ok(pygraph)
            },
        ));
        Ok((flattened_s, key_suffix, builder))
    }

    fn maybe_star_args(
        &self,
        key: GraphCacheKey,
        args_s: &[SomeValue],
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        let (_flattened_s, star_key, builder) = self.flatten_star_args(args_s)?;
        let combined_key = self.append_cache_key(key, star_key)?;
        self.cachedgraph(combined_key, None, builder)
    }

    pub(crate) fn default_specialize(
        &self,
        inputcells: &mut Vec<SomeValue>,
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        // upstream `default_specialize(funcdesc, args_s)`
        // (specialize.py:60-85).
        let (flattened_cells, mut key, builder) = self.flatten_star_args(inputcells)?;
        let mut flattened_cells = flattened_cells;
        let inputcells: &mut [SomeValue] = if !matches!(key, GraphCacheKey::None) {
            flattened_cells.as_mut_slice()
        } else {
            inputcells.as_mut_slice()
        };

        let jit_look_inside = self
            .base
            .pyobj
            .as_ref()
            .and_then(|pyobj| pyobj.user_function())
            .and_then(|func| func._jit_look_inside_)
            .unwrap_or(true);
        let mut access_directly = false;
        for s_obj in inputcells.iter_mut() {
            if let SomeValue::Instance(inst) = s_obj {
                let has_flag = inst.flags.get("access_directly").copied().unwrap_or(false);
                if has_flag {
                    if jit_look_inside {
                        access_directly = true;
                        key = GraphCacheKey::Tuple(vec![GraphCacheKey::AccessDirect, key.clone()]);
                        break;
                    } else {
                        let mut new_flags = inst.flags.clone();
                        new_flags.remove("access_directly");
                        *s_obj = SomeValue::Instance(SomeInstance::new(
                            inst.classdef.clone(),
                            inst.can_be_none,
                            new_flags,
                        ));
                    }
                }
            }
        }
        let graph = self.cachedgraph(key, None, builder)?;
        if access_directly {
            graph.access_directly.set(true);
        }
        Ok(graph)
    }

    fn specialize_argvalue(
        &self,
        args_s: &[SomeValue],
        parms: &[String],
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        let mut key_parts = Vec::with_capacity(parms.len());
        for parm in parms {
            let idx: usize = parm.parse().map_err(|_| {
                AnnotatorError::new(format!(
                    "specialize:arg expects integer indices, got {parm:?}"
                ))
            })?;
            let s = args_s.get(idx).ok_or_else(|| {
                AnnotatorError::new(format!(
                    "specialize:arg index {idx} out of range for {}",
                    self.name
                ))
            })?;
            match s {
                SomeValue::PBC(pbc) if pbc.descriptions.len() == 1 => {
                    let desc = pbc.any_description().expect("single-desc PBC");
                    key_parts.push(GraphCacheKey::Desc {
                        key: desc.desc_key(),
                        name: Self::nameof_desc(desc),
                    });
                }
                _ if s.is_constant() => {
                    let value = s.const_().expect("constant annotation missing const_box");
                    key_parts.push(GraphCacheKey::Const(value.clone()));
                }
                _ => {
                    return Err(AnnotatorError::new(format!(
                        "specialize:arg({idx}): argument not constant: {s:?}"
                    )));
                }
            }
        }
        self.maybe_star_args(GraphCacheKey::Tuple(key_parts), args_s)
    }

    /// Mutation step inside the upstream `builder(translator, func)`
    /// closure from `specialize.py:29-52`.
    fn apply_star_args_builder(
        pygraph: &PyGraph,
        nb_extra_args: usize,
    ) -> Result<(), AnnotatorError> {
        use crate::flowspace::model::{Block, Hlvalue, Link, Variable};
        use crate::flowspace::operation::{HLOperation, OpKind};

        // upstream: `argnames, vararg, kwarg = graph.signature; assert vararg`.
        let old_sig = pygraph.signature.borrow().clone();
        if old_sig.varargname.is_none() {
            return Err(AnnotatorError::new(format!(
                "apply_star_args_builder({}): graph signature missing *arg",
                pygraph.graph.borrow().name
            )));
        }
        if old_sig.kwargname.is_some() {
            return Err(AnnotatorError::new(format!(
                "apply_star_args_builder({}): graph signature unexpectedly has **arg",
                pygraph.graph.borrow().name
            )));
        }
        let mut graph = pygraph.graph.borrow_mut();
        // upstream: `argscopy = [Variable(v) for v in graph.getargs()]`.
        let old_args = graph.getargs();
        let argscopy: Vec<Variable> = old_args
            .iter()
            .map(|hl| match hl {
                Hlvalue::Variable(v) => Variable::named(&v.name()),
                _ => Variable::new(),
            })
            .collect();
        // upstream: `starargs = [Variable('stararg%d'%i) for i in range(nb_extra_args)]`.
        let starargs: Vec<Variable> = (0..nb_extra_args)
            .map(|i| Variable::named(format!("stararg{}", i)))
            .collect();
        // upstream: `newstartblock = Block(argscopy[:-1] + starargs)`.
        let mut new_inputargs: Vec<Hlvalue> = argscopy[..argscopy.len() - 1]
            .iter()
            .map(|v| Hlvalue::Variable(v.clone()))
            .collect();
        new_inputargs.extend(starargs.iter().map(|v| Hlvalue::Variable(v.clone())));
        let newstartblock = Block::shared(new_inputargs);
        // upstream: `newtup = op.newtuple(*starargs); newtup.result = argscopy[-1]`.
        //
        // `HLOperation` is the flow-time handler object; the persisted
        // form is `SpaceOperation(opname, args, result)`. The annotator
        // consumes `operations: Vec<SpaceOperation>`, so emit the
        // SpaceOperation form directly.
        let star_args_hl: Vec<Hlvalue> = starargs
            .iter()
            .map(|v| Hlvalue::Variable(v.clone()))
            .collect();
        let newtup_result = Hlvalue::Variable(argscopy[argscopy.len() - 1].clone());
        let newtup_sp = crate::flowspace::model::SpaceOperation::new(
            OpKind::NewTuple.opname(),
            star_args_hl,
            newtup_result,
        );
        let _ = HLOperation::new(OpKind::NewTuple, Vec::new()); // keep import alive
        newstartblock.borrow_mut().operations.push(newtup_sp);
        // upstream: `newstartblock.closeblock(Link(argscopy, graph.startblock))`.
        let link_args: Vec<Hlvalue> = argscopy
            .iter()
            .map(|v| Hlvalue::Variable(v.clone()))
            .collect();
        let link = Rc::new(std::cell::RefCell::new(Link::new(
            link_args,
            Some(graph.startblock.clone()),
            None,
        )));
        newstartblock.closeblock(vec![link]);
        // upstream: `graph.startblock = newstartblock`.
        graph.startblock = newstartblock;
        drop(graph);

        let mut argnames = old_sig.argnames;
        argnames.extend((0..nb_extra_args).map(|i| format!(".star{}", i)));
        *pygraph.signature.borrow_mut() = Signature::new(argnames, None, None);
        if nb_extra_args > 0 {
            *pygraph.defaults.borrow_mut() = None;
        }
        checkgraph(&pygraph.graph.borrow());
        Ok(())
    }

    /// RPython `FunctionDesc.specialize(inputcells, op=None)`
    /// (description.py:272-281).
    ///
    /// The optional `op_key` parameter threads the call-site position
    /// from the flow-space operation through the annotator driver. As
    /// in upstream `description.py:272-281`, a missing explicit
    /// `op_key` first reuses `bookkeeper.position_key` before
    /// dispatching to the specializer.
    pub fn specialize(
        &self,
        inputcells: &mut Vec<SomeValue>,
        op_key: Option<PositionKey>,
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        let op_key = op_key.or_else(|| self.base.bookkeeper.current_position_key());
        self.normalize_args(inputcells)?;
        match self.specializer.as_ref().unwrap_or(&Specializer::Default) {
            Specializer::Default => self.default_specialize(inputcells),
            Specializer::Arg { parms } => self.specialize_argvalue(inputcells, parms),
            Specializer::ArgOrVar { parms } => {
                for parm in parms {
                    let idx: usize = parm.parse().map_err(|_| {
                        AnnotatorError::new(format!(
                            "specialize:arg_or_var expects integer indices, got {parm:?}"
                        ))
                    })?;
                    let s = inputcells.get(idx).ok_or_else(|| {
                        AnnotatorError::new(format!(
                            "specialize:arg_or_var index {idx} out of range for {}",
                            self.name
                        ))
                    })?;
                    if !s.is_constant() {
                        return self.maybe_star_args(GraphCacheKey::None, inputcells);
                    }
                }
                self.specialize_argvalue(inputcells, parms)
            }
            Specializer::Argtype { parms } => {
                let mut key_parts = Vec::with_capacity(parms.len());
                for parm in parms {
                    let idx: usize = parm.parse().map_err(|_| {
                        AnnotatorError::new(format!(
                            "specialize:argtype expects integer indices, got {parm:?}"
                        ))
                    })?;
                    let s = inputcells.get(idx).ok_or_else(|| {
                        AnnotatorError::new(format!(
                            "specialize:argtype index {idx} out of range for {}",
                            self.name
                        ))
                    })?;
                    key_parts.push(GraphCacheKey::KnownType(s.knowntype()));
                }
                self.maybe_star_args(GraphCacheKey::Tuple(key_parts), inputcells)
            }
            Specializer::Arglistitemtype { parms } => {
                let idx: usize = parms
                    .first()
                    .ok_or_else(|| {
                        AnnotatorError::new("specialize:arglistitemtype requires one index")
                    })?
                    .parse()
                    .map_err(|_| {
                        AnnotatorError::new(format!(
                            "specialize:arglistitemtype expects integer indices, got {parms:?}"
                        ))
                    })?;
                let key = match inputcells.get(idx) {
                    Some(SomeValue::List(list)) => {
                        GraphCacheKey::KnownType(list.listdef.s_value().knowntype())
                    }
                    _ => GraphCacheKey::None,
                };
                self.maybe_star_args(key, inputcells)
            }
            Specializer::CallLocation => {
                // upstream `specialize_call_location(funcdesc,
                // args_s, op)` (specialize.py:368-370) — key is
                // `(op,)`, one specialisation per call site.
                let position_key = op_key.ok_or_else(|| {
                    AnnotatorError::new("specialize_call_location: missing call-site position")
                })?;
                let key = GraphCacheKey::Tuple(vec![GraphCacheKey::Position(position_key)]);
                self.maybe_star_args(key, inputcells)
            }
            Specializer::Memo => Err(AnnotatorError::new(
                "FunctionDesc.specialize: memo specializer still requires \
                 rpython/annotator/specialize.py MemoTable support",
            )),
            Specializer::LowLevelDefault => {
                crate::translator::rtyper::annlowlevel::LowLevelAnnotatorPolicy::new(None)
                    .default_specialize(self, inputcells)
                    .map_err(|e| AnnotatorError::new(e.to_string()))
            }
            Specializer::MixLevelDefault => {
                let rtyper = self
                    .base
                    .bookkeeper
                    .annotator()
                    .translator
                    .rtyper()
                    .ok_or_else(|| {
                        AnnotatorError::new(
                            "FunctionDesc.specialize: MixLevel specializer requires translator.rtyper",
                        )
                    })?;
                crate::translator::rtyper::annlowlevel::MixLevelAnnotatorPolicy {
                    ll: crate::translator::rtyper::annlowlevel::LowLevelAnnotatorPolicy::new(Some(
                        &rtyper,
                    )),
                }
                .default_specialize(self, inputcells)
                .map_err(|e| AnnotatorError::new(e.to_string()))
            }
            Specializer::Ll { .. } => {
                crate::translator::rtyper::annlowlevel::LowLevelAnnotatorPolicy::new(None)
                    .specialize_ll(self, inputcells)
                    .map_err(|e| AnnotatorError::new(e.to_string()))
            }
            Specializer::LlAndArg { parms } => {
                let argindices: Vec<usize> = parms
                    .iter()
                    .map(|parm| {
                        parm.parse().map_err(|_| {
                            AnnotatorError::new(format!(
                                "specialize:ll_and_arg expects integer indices, got {parm:?}"
                            ))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                crate::translator::rtyper::annlowlevel::LowLevelAnnotatorPolicy::new(None)
                    .specialize_ll_and_arg(self, inputcells, &argindices)
                    .map_err(|e| AnnotatorError::new(e.to_string()))
            }
            Specializer::ArgLltype { parms } => {
                let i: usize = parms
                    .first()
                    .ok_or_else(|| AnnotatorError::new("specialize:arglltype requires one index"))?
                    .parse()
                    .map_err(|_| {
                        AnnotatorError::new(format!(
                            "specialize:arglltype expects integer indices, got {parms:?}"
                        ))
                    })?;
                let rtyper = self
                    .base
                    .bookkeeper
                    .annotator()
                    .translator
                    .rtyper()
                    .ok_or_else(|| {
                        AnnotatorError::new(
                            "FunctionDesc.specialize: arglltype requires translator.rtyper",
                        )
                    })?;
                crate::translator::rtyper::annlowlevel::MixLevelAnnotatorPolicy {
                    ll: crate::translator::rtyper::annlowlevel::LowLevelAnnotatorPolicy::new(Some(
                        &rtyper,
                    )),
                }
                .specialize_arglltype(self, inputcells, i)
                .map_err(|e| AnnotatorError::new(e.to_string()))
            }
            Specializer::GenConst { parms } => {
                let i: usize = parms
                    .first()
                    .ok_or_else(|| AnnotatorError::new("specialize:genconst requires one index"))?
                    .parse()
                    .map_err(|_| {
                        AnnotatorError::new(format!(
                            "specialize:genconst expects integer indices, got {parms:?}"
                        ))
                    })?;
                let rtyper = self
                    .base
                    .bookkeeper
                    .annotator()
                    .translator
                    .rtyper()
                    .ok_or_else(|| {
                        AnnotatorError::new(
                            "FunctionDesc.specialize: genconst requires translator.rtyper",
                        )
                    })?;
                crate::translator::rtyper::annlowlevel::MixLevelAnnotatorPolicy {
                    ll: crate::translator::rtyper::annlowlevel::LowLevelAnnotatorPolicy::new(Some(
                        &rtyper,
                    )),
                }
                .specialize_genconst(self, inputcells, i)
                .map_err(|e| AnnotatorError::new(e.to_string()))
            }
        }
    }

    /// RPython `FunctionDesc.pycall(whence, args, s_previous_result, op=None)`
    /// (description.py:283-305).
    ///
    /// ```python
    /// def pycall(self, whence, args, s_previous_result, op=None):
    ///     inputcells = self.parse_arguments(args)
    ///     graph = self.specialize(inputcells, op)
    ///     assert isinstance(graph, FunctionGraph)
    ///     new_args = args.unmatch_signature(self.signature, inputcells)
    ///     inputcells = self.parse_arguments(new_args, graph)
    ///     annotator = self.bookkeeper.annotator
    ///     result = annotator.recursivecall(graph, whence, inputcells)
    ///     signature = getattr(self.pyobj, '_signature_', None)
    ///     if signature:
    ///         sigresult = enforce_signature_return(self, signature[1], result)
    ///         if sigresult is not None:
    ///             annotator.addpendingblock(graph, graph.returnblock, [sigresult])
    ///             result = sigresult
    ///     result = unionof(result, s_previous_result)
    ///     return result
    /// ```
    pub fn pycall(
        &self,
        whence: Option<(
            crate::flowspace::model::GraphRef,
            crate::flowspace::model::BlockRef,
            usize,
        )>,
        args: &ArgumentsForTranslation,
        s_previous_result: &SomeValue,
        op_key: Option<PositionKey>,
    ) -> Result<SomeValue, AnnotatorError> {
        // upstream: `inputcells = self.parse_arguments(args)`.
        let mut inputcells = self.parse_arguments(args, None)?;
        // upstream: `graph = self.specialize(inputcells, op)`.
        let graph = self.specialize(&mut inputcells, op_key.clone())?;
        // upstream: `assert isinstance(graph, FunctionGraph)` — Rust
        // specialize() statically returns `Rc<PyGraph>` (PyGraph is the
        // FunctionGraph subclass upstream), so the assertion collapses.

        // upstream: `new_args = args.unmatch_signature(self.signature, inputcells)`.
        let new_args = args
            .unmatch_signature(&self.signature, &inputcells)
            .map_err(|e| {
                AnnotatorError::new(format!("{}.unmatch_signature: {}", self.name, e.getmsg()))
            })?;
        // upstream: `inputcells = self.parse_arguments(new_args, graph)`.
        let inputcells = self.parse_arguments(&new_args, Some(graph.as_ref()))?;

        // upstream: `annotator = self.bookkeeper.annotator`.
        let annotator = self.base.bookkeeper.annotator();

        // upstream: `result = annotator.recursivecall(graph, whence, inputcells)`.
        let mut result = annotator.recursivecall(&graph.graph, whence, &inputcells);

        // upstream: `signature = getattr(self.pyobj, '_signature_', None)`
        //   `if signature: sigresult = enforce_signature_return(...)`.
        if let Some(annsig) = &self.annsignature {
            let sigresult = super::signature::enforce_signature_return(
                &self.name,
                &self.base.bookkeeper,
                &annsig.result,
                &result,
            )?;
            // upstream: `if sigresult is not None:`.
            if let Some(sigresult) = sigresult {
                // upstream: `annotator.addpendingblock(graph, graph.returnblock, [sigresult])`.
                let returnblock = graph.graph.borrow().returnblock.clone();
                annotator.addpendingblock(&graph.graph, &returnblock, &[sigresult.clone()]);
                result = sigresult;
            }
        }

        // upstream: `result = unionof(result, s_previous_result)`.
        super::model::unionof([&result, s_previous_result])
            .map_err(|e| AnnotatorError::new(format!("{}.pycall unionof: {}", self.name, e)))
    }

    /// RPython `FunctionDesc.get_graph(args, op)` (description.py:328-330).
    pub fn get_graph(
        &self,
        args: &ArgumentsForTranslation,
        op_key: Option<PositionKey>,
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        let mut inputs_s = self.parse_arguments(args, None)?;
        self.specialize(&mut inputs_s, op_key)
    }

    /// RPython `FunctionDesc.get_call_parameters(args_s)`
    /// (description.py:332-348).
    ///
    /// ```python
    /// def get_call_parameters(self, args_s):
    ///     args = simple_args(args_s)
    ///     inputcells = self.parse_arguments(args)
    ///     graph = self.specialize(inputcells)
    ///     assert isinstance(graph, FunctionGraph)
    ///     # recreate the args object because inputcells may have been
    ///     # changed
    ///     new_args = args.unmatch_signature(self.signature, inputcells)
    ///     inputcells = self.parse_arguments(new_args, graph)
    ///     signature = getattr(self.pyobj, '_signature_', None)
    ///     if signature:
    ///         s_result = finish_type(signature[1], self.bookkeeper, self.pyobj)
    ///         if s_result is not None:
    ///             self.bookkeeper.annotator.addpendingblock(
    ///                 graph, graph.returnblock, [s_result])
    ///     return graph, inputcells
    /// ```
    ///
    /// Driver-entry variant of [`Self::pycall`]: prepares
    /// `(graph, inputcells)` for the caller to feed into
    /// `addpendinggraph`, pre-seeding the return block from
    /// `_signature_` when present. Does not invoke `recursivecall` — the
    /// caller (annrpython.build_types) takes ownership of completing
    /// the analysis via `build_graph_types`.
    pub fn get_call_parameters(
        &self,
        args_s: Vec<SomeValue>,
    ) -> Result<(Rc<PyGraph>, Vec<SomeValue>), AnnotatorError> {
        // upstream: `args = simple_args(args_s)`.
        let args = super::argument::simple_args(args_s);
        // upstream: `inputcells = self.parse_arguments(args)`.
        let mut inputcells = self.parse_arguments(&args, None)?;
        // upstream: `graph = self.specialize(inputcells)`.
        let graph = self.specialize(&mut inputcells, None)?;
        // upstream: `assert isinstance(graph, FunctionGraph)` — Rust
        // specialize() statically returns `Rc<PyGraph>` (PyGraph is the
        // FunctionGraph subclass upstream), so the assertion collapses.

        // upstream: `new_args = args.unmatch_signature(self.signature, inputcells)`.
        let new_args = args
            .unmatch_signature(&self.signature, &inputcells)
            .map_err(|e| {
                AnnotatorError::new(format!(
                    "{}.get_call_parameters unmatch_signature: {}",
                    self.name,
                    e.getmsg()
                ))
            })?;
        // upstream: `inputcells = self.parse_arguments(new_args, graph)`.
        let inputcells = self.parse_arguments(&new_args, Some(graph.as_ref()))?;

        // upstream: `signature = getattr(self.pyobj, '_signature_', None)`
        //   `if signature: s_result = finish_type(signature[1], ...)`.
        if let Some(annsig) = &self.annsignature {
            let s_result =
                super::signature::finish_type(&annsig.result, &self.base.bookkeeper, &self.name)
                    .map_err(|e| -> AnnotatorError { e.into() })?;
            // upstream: `if s_result is not None:`.
            if let Some(s_result) = s_result {
                // upstream: `self.bookkeeper.annotator.addpendingblock(
                //     graph, graph.returnblock, [s_result])`.
                let annotator = self.base.bookkeeper.annotator();
                let returnblock = graph.graph.borrow().returnblock.clone();
                annotator.addpendingblock(&graph.graph, &returnblock, &[s_result]);
            }
        }

        // upstream: `return graph, inputcells`.
        Ok((graph, inputcells))
    }

    /// RPython `FunctionDesc.bind_under(classdef, name)`
    /// (description.py:350-355).
    ///
    /// ```python
    /// def bind_under(self, classdef, name):
    ///     # XXX static methods
    ///     return self.bookkeeper.getmethoddesc(self,
    ///                                          classdef,   # originclassdef,
    ///                                          None,       # selfclassdef
    ///                                          name)
    /// ```
    pub fn bind_under(
        self_rc: &Rc<RefCell<FunctionDesc>>,
        originclassdef: &Rc<RefCell<super::classdesc::ClassDef>>,
        name: &str,
    ) -> Rc<RefCell<MethodDesc>> {
        let bk = self_rc.borrow().base.bookkeeper.clone();
        // upstream defaults: `selfclassdef=None`, `flags={}`.
        bk.getmethoddesc(
            self_rc,
            ClassDefKey::from_classdef(originclassdef),
            None,
            name,
            std::collections::BTreeMap::new(),
        )
    }

    /// RPython `FunctionDesc.consider_call_site(descs, args, s_result, op)`
    /// (description.py:357-363).
    ///
    pub fn consider_call_site(
        descs: &[Rc<RefCell<FunctionDesc>>],
        args: &ArgumentsForTranslation,
        _s_result: &SomeValue,
        op_key: Option<PositionKey>,
    ) -> Result<(), AnnotatorError> {
        if descs.is_empty() {
            return Ok(());
        }
        let family = descs[0].borrow().base.getcallfamily()?;
        let shape = args.rawshape();
        let desc_entries: Vec<DescEntry> = descs.iter().cloned().map(DescEntry::Function).collect();
        let row = build_calltable_row(&desc_entries, args, op_key)?;
        family.borrow_mut().calltable_add_row(shape, row);
        let borrowed: Vec<_> = descs.iter().skip(1).map(|d| d.borrow()).collect();
        let others: Vec<&Desc> = borrowed.iter().map(|d| &d.base).collect();
        descs[0].borrow().base.mergecallfamilies(&others)?;
        Ok(())
    }

    /// RPython `FunctionDesc.get_s_signatures(shape)`
    /// (description.py:368-393).
    ///
    /// ```python
    /// def get_s_signatures(self, shape):
    ///     family = self.getcallfamily()
    ///     table = family.calltables.get(shape)
    ///     if table is None:
    ///         return []
    ///     else:
    ///         graph_seen = {}
    ///         s_sigs = []
    ///         binding = self.bookkeeper.annotator.binding
    ///         def enlist(graph):
    ///             if graph in graph_seen: return
    ///             graph_seen[graph] = True
    ///             s_sig = ([binding(v) for v in graph.getargs()],
    ///                      binding(graph.getreturnvar()))
    ///             if s_sig in s_sigs: return
    ///             s_sigs.append(s_sig)
    ///         for row in table:
    ///             for graph in row.itervalues():
    ///                 enlist(graph)
    ///         return s_sigs
    /// ```
    pub fn get_s_signatures(
        &self,
        shape: &CallShape,
    ) -> Result<Vec<(Vec<SomeValue>, SomeValue)>, AnnotatorError> {
        let family = self.base.getcallfamily()?;
        // upstream: `table = family.calltables.get(shape)`.
        let rows: Vec<CallTableRow> = {
            let family_ref = family.borrow();
            match family_ref.calltables.get(shape) {
                Some(table) => table.clone(),
                None => return Ok(Vec::new()),
            }
        };

        // upstream: `binding = self.bookkeeper.annotator.binding`.
        let annotator = self.base.bookkeeper.annotator();

        // upstream: `graph_seen = {}; s_sigs = []`.
        // Graph identity dedup via `Rc::as_ptr` on Rc<RefCell<FunctionGraph>>.
        let mut graph_seen: std::collections::HashSet<
            *const std::cell::RefCell<crate::flowspace::model::FunctionGraph>,
        > = std::collections::HashSet::new();
        let mut s_sigs: Vec<(Vec<SomeValue>, SomeValue)> = Vec::new();

        // upstream: `for row in table: for graph in row.itervalues(): enlist(graph)`.
        for row in &rows {
            for pygraph in row.values() {
                let graph_ref = &pygraph.graph;
                let key = Rc::as_ptr(graph_ref);
                // upstream: `if graph in graph_seen: return`.
                if !graph_seen.insert(key) {
                    continue;
                }
                // upstream: `s_sig = ([binding(v) for v in graph.getargs()],
                //                     binding(graph.getreturnvar()))`.
                let (args_hl, returnvar) = {
                    let g = graph_ref.borrow();
                    (g.getargs(), g.getreturnvar())
                };
                let mut args_s: Vec<SomeValue> = Vec::with_capacity(args_hl.len());
                for v in &args_hl {
                    args_s.push(annotator.binding(v));
                }
                let return_s = annotator.binding(&returnvar);
                let s_sig = (args_s, return_s);
                // upstream: `if s_sig in s_sigs: return`.
                if !s_sigs.iter().any(|existing| existing == &s_sig) {
                    s_sigs.push(s_sig);
                }
            }
        }
        Ok(s_sigs)
    }
}

/// RPython `class MemoDesc(FunctionDesc)` (description.py:395-404).
///
/// Overrides `pycall` to project the specialized graph's return
/// annotation directly instead of driving a fresh `recursivecall`.
#[derive(Debug)]
pub struct MemoDesc {
    pub base: FunctionDesc,
}

impl MemoDesc {
    pub fn new(base: FunctionDesc) -> Self {
        MemoDesc { base }
    }

    /// RPython `MemoDesc.pycall(whence, args, s_previous_result, op=None)`
    /// (description.py:395-404).
    ///
    /// ```python
    /// def pycall(self, whence, args, s_previous_result, op=None):
    ///     inputcells = self.parse_arguments(args)
    ///     s_result = self.specialize(inputcells, op)
    ///     if isinstance(s_result, FunctionGraph):
    ///         s_result = s_result.getreturnvar().annotation
    ///         if s_result is None:
    ///             s_result = s_ImpossibleValue
    ///     s_result = unionof(s_result, s_previous_result)
    ///     return s_result
    /// ```
    ///
    /// In the Rust port `specialize` always returns `Rc<PyGraph>`
    /// (upstream's union `SomeValue | FunctionGraph` collapses because
    /// the memoize-on-Some hook isn't ported yet — specialize.py :memo
    /// branch). The `isinstance(s_result, FunctionGraph)` arm is the
    /// always-taken path here.
    pub fn pycall(
        &self,
        _whence: Option<(
            crate::flowspace::model::GraphRef,
            crate::flowspace::model::BlockRef,
            usize,
        )>,
        args: &ArgumentsForTranslation,
        s_previous_result: &SomeValue,
        op_key: Option<super::bookkeeper::PositionKey>,
    ) -> Result<SomeValue, AnnotatorError> {
        use super::model::s_impossible_value;
        // upstream: `inputcells = self.parse_arguments(args)`.
        let mut inputcells = self.base.parse_arguments(args, None)?;
        // upstream: `s_result = self.specialize(inputcells, op)`.
        let graph = self.base.specialize(&mut inputcells, op_key)?;
        // upstream: `if isinstance(s_result, FunctionGraph): s_result =
        //              s_result.getreturnvar().annotation`. In Rust
        // `specialize` always returns a PyGraph (FunctionGraph subclass),
        // so we always take this branch.
        let returnvar = graph.graph.borrow().getreturnvar();
        let s_result = match &returnvar {
            crate::flowspace::model::Hlvalue::Variable(v) => v
                .annotation
                .borrow()
                .as_ref()
                .map(|rc| (**rc).clone())
                .unwrap_or_else(s_impossible_value),
            _ => s_impossible_value(),
        };
        // upstream: `s_result = unionof(s_result, s_previous_result)`.
        super::model::unionof([&s_result, s_previous_result])
            .map_err(|e| AnnotatorError::new(format!("MemoDesc.pycall unionof: {}", e)))
    }
}

// ---------------------------------------------------------------------------
// MethodDesc + FrozenDesc + MethodOfFrozenDesc (description.py:407-637).
// ---------------------------------------------------------------------------

/// Opaque handle for `ClassDef` references used by
/// [`MethodDesc::originclassdef`] / `selfclassdef`. Upstream carries
/// a live `ClassDef` object; Rust uses a pointer-identity handle so
/// the same `Rc<RefCell<super::classdesc::ClassDef>>` value hashes
/// stably across borrows. The classdesc registry itself is ported
/// in [`super::classdesc`] — see [`Self::from_classdef`] for the
/// `Rc::as_ptr` conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClassDefKey(pub usize);

impl ClassDefKey {
    pub fn from_raw(id: usize) -> Self {
        ClassDefKey(id)
    }

    pub fn from_classdef(classdef: &Rc<RefCell<super::classdesc::ClassDef>>) -> Self {
        ClassDefKey(Rc::as_ptr(classdef) as usize)
    }
}

/// RPython `class MethodDesc(Desc)` (description.py:407-519).
#[derive(Debug)]
pub struct MethodDesc {
    pub base: Desc,
    /// RPython `self.funcdesc` (description.py:413).
    pub funcdesc: Rc<RefCell<FunctionDesc>>,
    /// RPython `self.originclassdef` (description.py:414).
    pub originclassdef: ClassDefKey,
    /// RPython `self.selfclassdef` (description.py:415). `None` for
    /// unbound method descriptions.
    pub selfclassdef: Option<ClassDefKey>,
    /// RPython `self.name` (description.py:416).
    pub name: String,
    /// RPython `self.flags = flags` (description.py:417) — upstream
    /// default is `{}`; Rust mirrors with an empty BTreeMap for
    /// stable iteration order.
    pub flags: std::collections::BTreeMap<String, bool>,
}

impl MethodDesc {
    /// RPython `MethodDesc.__init__(bookkeeper, funcdesc,
    /// originclassdef, selfclassdef, name, flags={})`
    /// (description.py:410-417).
    pub fn new(
        bookkeeper: Rc<Bookkeeper>,
        funcdesc: Rc<RefCell<FunctionDesc>>,
        originclassdef: ClassDefKey,
        selfclassdef: Option<ClassDefKey>,
        name: impl Into<String>,
        flags: std::collections::BTreeMap<String, bool>,
    ) -> Self {
        MethodDesc {
            base: Desc::new(bookkeeper, None),
            funcdesc,
            originclassdef,
            selfclassdef,
            name: name.into(),
            flags,
        }
    }

    /// RPython `MethodDesc.getuniquegraph()` (description.py:429-430).
    pub fn getuniquegraph(&self) -> Result<Rc<PyGraph>, AnnotatorError> {
        self.funcdesc.borrow().getuniquegraph()
    }

    /// RPython `MethodDesc.func_args(args)` (description.py:432-437).
    ///
    /// ```python
    /// def func_args(self, args):
    ///     from rpython.annotator.model import SomeInstance
    ///     if self.selfclassdef is None:
    ///         raise AnnotatorError("calling %r" % (self,))
    ///     s_instance = SomeInstance(self.selfclassdef, flags=self.flags)
    ///     return args.prepend(s_instance)
    /// ```
    ///
    /// The Rust port stores `selfclassdef` as a `ClassDefKey` (pointer
    /// identity) rather than a live `Rc<RefCell<ClassDef>>` to avoid
    /// reference cycles in the descriptor graph. The `lookup_classdef`
    /// indirection retrieves the live handle at call time — semantically
    /// equivalent to Python dereferencing `self.selfclassdef`.
    pub fn func_args(
        &self,
        args: &ArgumentsForTranslation,
    ) -> Result<ArgumentsForTranslation, AnnotatorError> {
        // upstream: `if self.selfclassdef is None: raise AnnotatorError(...)`.
        let Some(selfclassdef) = self.selfclassdef else {
            return Err(AnnotatorError::new(format!(
                "calling unbound MethodDesc {:?}",
                self.name
            )));
        };
        let classdef = self
            .base
            .bookkeeper
            .lookup_classdef(selfclassdef)
            .ok_or_else(|| {
                AnnotatorError::new(format!(
                    "MethodDesc.func_args: unknown ClassDefKey {:?}",
                    selfclassdef
                ))
            })?;
        // upstream: `s_instance = SomeInstance(self.selfclassdef, flags=self.flags)`.
        let s_instance =
            SomeValue::Instance(SomeInstance::new(Some(classdef), false, self.flags.clone()));
        // upstream: `return args.prepend(s_instance)`.
        Ok(args.prepend(s_instance))
    }

    /// RPython `MethodDesc.pycall(whence, args, s_previous_result, op)`
    /// (description.py:439-441).
    ///
    /// ```python
    /// def pycall(self, whence, args, s_previous_result, op=None):
    ///     return self.funcdesc.pycall(whence, self.func_args(args),
    ///                                 s_previous_result, op)
    /// ```
    pub fn pycall(
        &self,
        whence: Option<(
            crate::flowspace::model::GraphRef,
            crate::flowspace::model::BlockRef,
            usize,
        )>,
        args: &ArgumentsForTranslation,
        s_previous_result: &SomeValue,
        op_key: Option<PositionKey>,
    ) -> Result<SomeValue, AnnotatorError> {
        let func_args = self.func_args(args)?;
        self.funcdesc
            .borrow()
            .pycall(whence, &func_args, s_previous_result, op_key)
    }

    /// RPython `MethodDesc.get_graph(args, op)` (description.py:443-445).
    pub fn get_graph(
        &self,
        args: &ArgumentsForTranslation,
        op_key: Option<PositionKey>,
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        let func_args = self.func_args(args)?;
        self.funcdesc.borrow().get_graph(&func_args, op_key)
    }

    /// RPython `MethodDesc.bind_under(classdef, name)` (description.py:447-449).
    ///
    /// ```python
    /// def bind_under(self, classdef, name):
    ///     self.bookkeeper.warning("rebinding an already bound %r" % (self,))
    ///     return self.funcdesc.bind_under(classdef, name)
    /// ```
    pub fn bind_under(
        &self,
        classdef: &Rc<RefCell<super::classdesc::ClassDef>>,
        name: &str,
    ) -> Rc<RefCell<MethodDesc>> {
        // upstream: `self.bookkeeper.warning("rebinding an already bound %r" % (self,))`.
        self.base
            .bookkeeper
            .warning(format!("rebinding an already bound MethodDesc {:?}", self));
        FunctionDesc::bind_under(&self.funcdesc, classdef, name)
    }

    /// RPython `MethodDesc.bind_self(newselfclassdef, flags={})`
    /// (description.py:451-456).
    pub fn bind_self(
        &self,
        newselfclassdef: ClassDefKey,
        flags: std::collections::BTreeMap<String, bool>,
    ) -> Result<Rc<RefCell<MethodDesc>>, AnnotatorError> {
        Ok(self.base.bookkeeper.getmethoddesc(
            &self.funcdesc,
            self.originclassdef,
            Some(newselfclassdef),
            &self.name,
            flags,
        ))
    }

    /// RPython `MethodDesc.rowkey()` (description.py:467-471).
    ///
    /// "we are computing call families and call tables that always
    /// contain FunctionDescs, not MethodDescs. The present method
    /// returns the FunctionDesc to use as a key in that family."
    pub fn rowkey(&self) -> DescKey {
        self.funcdesc.borrow().base.identity
    }

    /// RPython `MethodDesc.consider_call_site(descs, args, s_result, op)`
    /// (description.py:458-465).
    ///
    /// ```python
    /// @staticmethod
    /// def consider_call_site(descs, args, s_result, op):
    ///     cnt, keys, star = rawshape(args)
    ///     shape = cnt + 1, keys, star  # account for the extra 'self'
    ///     row = build_calltable_row(descs, args, op)
    ///     family = descs[0].getcallfamily()
    ///     family.calltable_add_row(shape, row)
    ///     descs[0].mergecallfamilies(*descs[1:])
    /// ```
    ///
    /// Structural mirror of [`FunctionDesc::consider_call_site`] with
    /// `shape.shape_cnt += 1` for the implicit `self` argument that
    /// `MethodDesc.get_graph` prepends via `func_args`.
    pub fn consider_call_site(
        descs: &[Rc<RefCell<MethodDesc>>],
        args: &ArgumentsForTranslation,
        _s_result: &SomeValue,
        op_key: Option<PositionKey>,
    ) -> Result<(), AnnotatorError> {
        if descs.is_empty() {
            return Ok(());
        }
        let family = descs[0].borrow().base.getcallfamily()?;
        let mut shape = args.rawshape();
        shape.shape_cnt += 1;
        let desc_entries: Vec<DescEntry> = descs.iter().cloned().map(DescEntry::Method).collect();
        let row = build_calltable_row(&desc_entries, args, op_key)?;
        family.borrow_mut().calltable_add_row(shape, row);
        let borrowed: Vec<_> = descs.iter().skip(1).map(|d| d.borrow()).collect();
        let others: Vec<&Desc> = borrowed.iter().map(|d| &d.base).collect();
        descs[0].borrow().base.mergecallfamilies(&others)?;
        Ok(())
    }

    /// RPython `MethodDesc.simplify_desc_set(descs)` (description.py:473-519).
    pub fn simplify_desc_set(descs: &mut std::collections::BTreeMap<DescKey, DescEntry>) {
        let mut lst: Vec<Rc<RefCell<MethodDesc>>> = descs
            .values()
            .filter_map(|entry| entry.as_method())
            .collect();
        if lst.len() <= 1 {
            return;
        }

        // description.py:486-504 — keep only the intersection of all flags.
        let mut commonflags = lst[0].borrow().flags.clone();
        commonflags.retain(|key, value| {
            lst[1..]
                .iter()
                .all(|desc| desc.borrow().flags.get(key) == Some(value))
        });

        let mut replacements: Vec<(DescKey, DescEntry)> = Vec::new();
        for desc in &lst {
            let borrowed = desc.borrow();
            if borrowed.flags != commonflags {
                let newdesc = borrowed.base.bookkeeper.getmethoddesc(
                    &borrowed.funcdesc,
                    borrowed.originclassdef,
                    borrowed.selfclassdef,
                    &borrowed.name,
                    commonflags.clone(),
                );
                replacements.push((DescKey::from_rc(desc), DescEntry::Method(newdesc)));
            }
        }
        for (old_key, new_entry) in replacements {
            descs.remove(&old_key);
            descs.insert(new_entry.desc_key(), new_entry);
        }

        // description.py:505-519 — remove methods shadowed by a more
        // general selfclassdef on the same func/origin/name triple.
        lst = descs
            .values()
            .filter_map(|entry| entry.as_method())
            .collect();
        let mut groups: HashMap<(DescKey, ClassDefKey, String), Vec<Rc<RefCell<MethodDesc>>>> =
            HashMap::new();
        for desc in &lst {
            let borrowed = desc.borrow();
            if borrowed.selfclassdef.is_none() {
                continue;
            }
            groups
                .entry((
                    DescKey::from_rc(&borrowed.funcdesc),
                    borrowed.originclassdef,
                    borrowed.name.clone(),
                ))
                .or_default()
                .push(desc.clone());
        }
        for group in groups.values() {
            if group.len() <= 1 {
                continue;
            }
            let mut remove: Vec<DescKey> = Vec::new();
            for desc1 in group {
                let borrowed1 = desc1.borrow();
                let Some(cdef1) = borrowed1.selfclassdef else {
                    continue;
                };
                for desc2 in group {
                    let borrowed2 = desc2.borrow();
                    let Some(cdef2) = borrowed2.selfclassdef else {
                        continue;
                    };
                    if cdef1 == cdef2 {
                        continue;
                    }
                    let Some(classdef1) = borrowed1.base.bookkeeper.lookup_classdef(cdef1) else {
                        continue;
                    };
                    let Some(classdef2) = borrowed2.base.bookkeeper.lookup_classdef(cdef2) else {
                        continue;
                    };
                    if classdef1.borrow().issubclass(&classdef2) {
                        remove.push(DescKey::from_rc(desc1));
                        break;
                    }
                }
            }
            for key in remove {
                descs.remove(&key);
            }
        }
    }
}

/// RPython helper `new_or_old_class(c)` (description.py:522-526).
///
/// Upstream returns `c.__class__` when available, and `type(c)`
/// otherwise. The HostObject carrier exposes the same query via
/// [`HostObject::class_of`], with `Opaque` remaining the only
/// unresolved case because its runtime type is intentionally hidden
/// from the Rust port.
pub fn new_or_old_class(pyobj: &HostObject) -> Option<HostObject> {
    pyobj.class_of()
}

/// RPython `read_attribute` callback type for [`FrozenDesc`]
/// (description.py:532 `lambda attr: getattr(pyobj, attr)`). Returns
/// `Ok(value)` for a present attribute, `Err(HostGetAttrError::Missing)`
/// for upstream's `AttributeError`, and
/// `Err(HostGetAttrError::Unsupported)` for every other host-side
/// failure (upstream's non-AttributeError exceptions propagate instead
/// of being swallowed). Kept as a boxed `Fn` so subclasses / test
/// harnesses can wire in custom attribute sources the way upstream
/// allows the caller to pass `read_attribute=...` into
/// `FrozenDesc.__init__`.
pub type FrozenReadAttr =
    Box<dyn Fn(&str) -> Result<ConstValue, crate::flowspace::model::HostGetAttrError>>;

/// RPython `class FrozenDesc(Desc)` (description.py:528-599).
pub struct FrozenDesc {
    pub base: Desc,
    /// RPython `self.attrcache = {}` (description.py:535).
    pub attrcache: RefCell<HashMap<String, ConstValue>>,
    /// RPython `self.knowntype = new_or_old_class(pyobj)` (description.py:536).
    pub knowntype: Option<HostObject>,
    /// RPython `self._read_attribute` (description.py:533-534) — the
    /// caller-supplied attribute-source closure, or the
    /// `lambda attr: getattr(pyobj, attr)` default when none is
    /// provided. Defaults to [`FrozenDesc::default_read_attribute`]
    /// which reads through `HostObject::module_get` for modules.
    _read_attribute: FrozenReadAttr,
}

impl std::fmt::Debug for FrozenDesc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrozenDesc")
            .field("base", &self.base)
            .field("attrcache", &self.attrcache)
            .field("knowntype", &self.knowntype)
            .field("_read_attribute", &"<closure>")
            .finish()
    }
}

impl FrozenDesc {
    /// RPython `FrozenDesc.__init__(bookkeeper, pyobj,
    /// read_attribute=None)` (description.py:530-537).
    ///
    /// With `read_attribute = None`, upstream installs the default
    /// `lambda attr: getattr(pyobj, attr)`. The Rust port supplies
    /// [`Self::default_read_attribute`] — a host-level `getattr`
    /// mirror for the HostObject kinds currently modelled.
    pub fn new(bookkeeper: Rc<Bookkeeper>, pyobj: HostObject) -> Result<Self, AnnotatorError> {
        let callback = Self::default_read_attribute(pyobj.clone());
        Self::new_with_read_attribute(bookkeeper, pyobj, callback)
    }

    /// RPython `FrozenDesc.__init__(..., read_attribute=callback)` —
    /// the non-default branch (description.py:530-534). Callers that
    /// need a custom attribute source hand a closure in directly;
    /// matches upstream's factory pattern.
    pub fn new_with_read_attribute(
        bookkeeper: Rc<Bookkeeper>,
        pyobj: HostObject,
        read_attribute: FrozenReadAttr,
    ) -> Result<Self, AnnotatorError> {
        // upstream assert: `assert bool(pyobj), "__nonzero__ unsupported on frozen PBC %r"`.
        // Rust's HostObject has no __nonzero__ semantic; we accept any HostObject.
        let knowntype = new_or_old_class(&pyobj);
        Ok(FrozenDesc {
            base: Desc::new(bookkeeper, Some(pyobj)),
            attrcache: RefCell::new(HashMap::new()),
            knowntype,
            _read_attribute: read_attribute,
        })
    }

    /// Default `read_attribute` closure — Rust equivalent of upstream
    /// `lambda attr: getattr(pyobj, attr)` (description.py:534).
    ///
    /// The Rust port routes through [`crate::flowspace::model::host_getattr`].
    /// Upstream's lambda raises `AttributeError` when the name is
    /// absent and re-raises every other host exception untouched; the
    /// Rust callback mirrors that contract by returning
    /// `HostGetAttrError::Missing` / `HostGetAttrError::Unsupported`
    /// verbatim rather than collapsing both to "attribute missing".
    pub fn default_read_attribute(pyobj: HostObject) -> FrozenReadAttr {
        Box::new(move |attr: &str| host_getattr(&pyobj, attr))
    }

    /// RPython `FrozenDesc.has_attribute(attr)` (description.py:539-546).
    ///
    /// Upstream wraps `self._read_attribute(attr)` in
    /// `try: ... except AttributeError: return False`; every other
    /// exception escapes. The Rust port mirrors that: `Missing` maps
    /// to `Ok(false)`, `Unsupported` (and any other host-side failure)
    /// escapes as an `AnnotatorError`.
    pub fn has_attribute(&self, attr: &str) -> Result<bool, AnnotatorError> {
        if self.attrcache.borrow().contains_key(attr) {
            return Ok(true);
        }
        match self.read_attribute_raw(attr) {
            Ok(_) => Ok(true),
            Err(crate::flowspace::model::HostGetAttrError::Missing) => Ok(false),
            Err(crate::flowspace::model::HostGetAttrError::Unsupported) => {
                Err(AnnotatorError::new(format!(
                    "FrozenDesc.has_attribute({attr:?}): host getattr is unsupported",
                )))
            }
        }
    }

    /// RPython `FrozenDesc.warn_missing_attribute(attr)` (description.py:548-551).
    pub fn warn_missing_attribute(&self, attr: &str) -> Result<bool, AnnotatorError> {
        Ok(!self.has_attribute(attr)? && !attr.starts_with('$'))
    }

    /// Internal helper — reads an attribute through the stored
    /// `_read_attribute` closure. `Err(Missing)` means upstream's
    /// `AttributeError`; `Err(Unsupported)` means a non-AttributeError
    /// host failure the caller must surface. When the caller supplied
    /// a custom callback, this dispatches to it, matching
    /// `FrozenDesc._read_attribute(attr)` in description.py:543 /
    /// 557 / 570.
    fn read_attribute_raw(
        &self,
        attr: &str,
    ) -> Result<ConstValue, crate::flowspace::model::HostGetAttrError> {
        (self._read_attribute)(attr)
    }

    /// RPython `FrozenDesc.read_attribute(attr)` (description.py:553-558).
    ///
    /// Upstream `read_attribute` propagates `AttributeError` (caller's
    /// `s_read_attribute` then re-catches into `s_ImpossibleValue`)
    /// and re-raises every other exception. The Rust port distinguishes
    /// the two errors in the returned message so downstream callers can
    /// decide whether to fall through (Missing) or bail (Unsupported).
    pub fn read_attribute(&self, attr: &str) -> Result<ConstValue, AnnotatorError> {
        if let Some(v) = self.attrcache.borrow().get(attr) {
            return Ok(v.clone());
        }
        match self.read_attribute_raw(attr) {
            Ok(v) => {
                self.attrcache
                    .borrow_mut()
                    .insert(attr.to_string(), v.clone());
                Ok(v)
            }
            Err(crate::flowspace::model::HostGetAttrError::Missing) => Err(AnnotatorError::new(
                format!("AttributeError: frozen desc has no attribute {attr:?}"),
            )),
            Err(crate::flowspace::model::HostGetAttrError::Unsupported) => {
                Err(AnnotatorError::new(format!(
                    "FrozenDesc.read_attribute({attr:?}): host getattr is unsupported",
                )))
            }
        }
    }

    /// RPython `FrozenDesc.s_read_attribute(attr)` (description.py:560-566).
    ///
    /// Upstream catches only `AttributeError`; other exceptions
    /// escape. The Rust port mirrors that split: `Missing` collapses
    /// to `s_ImpossibleValue`, `Unsupported` (and any other
    /// host-side failure) propagates as an `AnnotatorError`.
    pub fn s_read_attribute(&self, attr: &str) -> Result<SomeValue, AnnotatorError> {
        if let Some(v) = self.attrcache.borrow().get(attr) {
            return self.base.bookkeeper.immutablevalue(v);
        }
        match self.read_attribute_raw(attr) {
            Ok(v) => {
                self.attrcache
                    .borrow_mut()
                    .insert(attr.to_string(), v.clone());
                self.base.bookkeeper.immutablevalue(&v)
            }
            Err(crate::flowspace::model::HostGetAttrError::Missing) => Ok(s_impossible_value()),
            Err(crate::flowspace::model::HostGetAttrError::Unsupported) => {
                Err(AnnotatorError::new(format!(
                    "FrozenDesc.s_read_attribute({attr:?}): host getattr is unsupported",
                )))
            }
        }
    }

    /// RPython `FrozenDesc.create_new_attribute(name, value)`
    /// (description.py:568-575).
    pub fn create_new_attribute(
        &self,
        name: impl Into<String>,
        value: ConstValue,
    ) -> Result<(), AnnotatorError> {
        let name = name.into();
        if self.read_attribute(&name).is_ok() {
            return Err(AnnotatorError::new(format!("name clash: {name:?}")));
        }
        self.attrcache.borrow_mut().insert(name, value);
        Ok(())
    }

    /// RPython `FrozenDesc.getattrfamily(attrname=None)`
    /// (description.py:577-581).
    ///
    pub fn getattrfamily(&self) -> Result<Rc<RefCell<FrozenAttrFamily>>, AnnotatorError> {
        let mut families = self.base.bookkeeper.frozenpbc_attr_families.borrow_mut();
        let rep = families.find_rep(self.base.identity);
        Ok(families
            .get(&rep)
            .cloned()
            .expect("UnionFind.find_rep() must materialize a FrozenAttrFamily"))
    }

    /// RPython `FrozenDesc.queryattrfamily(attrname=None)` (description.py:583-590).
    pub fn queryattrfamily(&self) -> Option<Rc<RefCell<FrozenAttrFamily>>> {
        self.base
            .bookkeeper
            .frozenpbc_attr_families
            .borrow_mut()
            .get(&self.base.identity)
            .cloned()
    }

    /// RPython `FrozenDesc.mergeattrfamilies(others, attrname=None)`
    /// (description.py:592-599).
    pub fn mergeattrfamilies(&self, others: &[&FrozenDesc]) -> Result<bool, AnnotatorError> {
        if others.is_empty() {
            return Ok(false);
        }
        let mut changed = false;
        let mut families = self.base.bookkeeper.frozenpbc_attr_families.borrow_mut();
        let mut rep = families.find_rep(self.base.identity);
        for desc in others {
            let (changed1, new_rep) = families.union(rep, desc.base.identity);
            changed |= changed1;
            rep = new_rep;
        }
        Ok(changed)
    }
}

/// RPython `class MethodOfFrozenDesc(Desc)` (description.py:602-637).
#[derive(Debug)]
pub struct MethodOfFrozenDesc {
    pub base: Desc,
    pub funcdesc: Rc<RefCell<FunctionDesc>>,
    pub frozendesc: Rc<RefCell<FrozenDesc>>,
}

impl MethodOfFrozenDesc {
    /// RPython `MethodOfFrozenDesc.__init__(bookkeeper, funcdesc,
    /// frozendesc)` (description.py:605-608).
    pub fn new(
        bookkeeper: Rc<Bookkeeper>,
        funcdesc: Rc<RefCell<FunctionDesc>>,
        frozendesc: Rc<RefCell<FrozenDesc>>,
    ) -> Self {
        MethodOfFrozenDesc {
            base: Desc::new(bookkeeper, None),
            funcdesc,
            frozendesc,
        }
    }

    /// RPython `MethodOfFrozenDesc.func_args(args)` (description.py:614-617):
    ///
    /// ```python
    /// def func_args(self, args):
    ///     from rpython.annotator.model import SomePBC
    ///     s_self = SomePBC([self.frozendesc])
    ///     return args.prepend(s_self)
    /// ```
    ///
    pub fn func_args(
        &self,
        args: &ArgumentsForTranslation,
    ) -> Result<ArgumentsForTranslation, AnnotatorError> {
        // upstream: `s_self = SomePBC([self.frozendesc]);
        //            return args.prepend(s_self)`.
        let s_self = SomePBC::new(vec![DescEntry::Frozen(self.frozendesc.clone())], false);
        Ok(args.prepend(SomeValue::PBC(s_self)))
    }

    /// RPython `MethodOfFrozenDesc.pycall(whence, args,
    /// s_previous_result, op)` (description.py:619-621).
    ///
    /// ```python
    /// def pycall(self, whence, args, s_previous_result, op=None):
    ///     return self.funcdesc.pycall(whence, self.func_args(args),
    ///                                 s_previous_result, op)
    /// ```
    pub fn pycall(
        &self,
        whence: Option<(
            crate::flowspace::model::GraphRef,
            crate::flowspace::model::BlockRef,
            usize,
        )>,
        args: &ArgumentsForTranslation,
        s_previous_result: &SomeValue,
        op_key: Option<PositionKey>,
    ) -> Result<SomeValue, AnnotatorError> {
        let func_args = self.func_args(args)?;
        self.funcdesc
            .borrow()
            .pycall(whence, &func_args, s_previous_result, op_key)
    }

    /// RPython `MethodOfFrozenDesc.get_graph(args, op)` (description.py:623-625).
    pub fn get_graph(
        &self,
        args: &ArgumentsForTranslation,
        op_key: Option<PositionKey>,
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        let func_args = self.func_args(args)?;
        self.funcdesc.borrow().get_graph(&func_args, op_key)
    }

    /// RPython `MethodOfFrozenDesc.rowkey()` (description.py:636-637).
    pub fn rowkey(&self) -> DescKey {
        self.funcdesc.borrow().base.identity
    }

    /// RPython `MethodOfFrozenDesc.consider_call_site(descs, args,
    /// s_result, op)` (description.py:627-634).
    ///
    /// Identical structure to [`MethodDesc::consider_call_site`] —
    /// `shape.shape_cnt += 1` for the bound frozen instance that
    /// `func_args` prepends via `SomePBC([self.frozendesc])`.
    pub fn consider_call_site(
        descs: &[Rc<RefCell<MethodOfFrozenDesc>>],
        args: &ArgumentsForTranslation,
        _s_result: &SomeValue,
        op_key: Option<PositionKey>,
    ) -> Result<(), AnnotatorError> {
        if descs.is_empty() {
            return Ok(());
        }
        let family = descs[0].borrow().base.getcallfamily()?;
        let mut shape = args.rawshape();
        shape.shape_cnt += 1;
        let desc_entries: Vec<DescEntry> = descs
            .iter()
            .cloned()
            .map(DescEntry::MethodOfFrozen)
            .collect();
        let row = build_calltable_row(&desc_entries, args, op_key)?;
        family.borrow_mut().calltable_add_row(shape, row);
        let borrowed: Vec<_> = descs.iter().skip(1).map(|d| d.borrow()).collect();
        let others: Vec<&Desc> = borrowed.iter().map(|d| &d.base).collect();
        descs[0].borrow().base.mergecallfamilies(&others)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{SomeInteger, SomeString, SomeValue};
    use crate::flowspace::argument::CallShape;
    use rustpython_compiler::{Mode, compile as rp_compile};
    use rustpython_compiler_core::bytecode::ConstantData;

    fn desc_key(id: usize) -> DescKey {
        DescKey::from_raw(id)
    }

    fn simple_shape() -> CallShape {
        CallShape {
            shape_cnt: 0,
            shape_keys: Vec::new(),
            shape_star: false,
        }
    }

    #[test]
    fn call_family_initial_state() {
        let cf = CallFamily::new(desc_key(1));
        assert!(cf.descs.contains_key(&desc_key(1)));
        assert!(cf.modified);
        assert!(!cf.normalized);
        assert_eq!(cf.total_calltable_size, 0);
    }

    #[test]
    fn call_family_update_merges_descs_and_calltables() {
        let mut cf1 = CallFamily::new(desc_key(1));
        let mut cf2 = CallFamily::new(desc_key(2));
        cf2.calltable_add_row(simple_shape(), HashMap::new());
        cf1.update(&cf2);
        assert!(cf1.descs.contains_key(&desc_key(1)));
        assert!(cf1.descs.contains_key(&desc_key(2)));
        assert_eq!(cf1.total_calltable_size, 1);
    }

    #[test]
    fn call_family_calltable_dedup() {
        let mut cf = CallFamily::new(desc_key(1));
        cf.calltable_add_row(simple_shape(), HashMap::new());
        cf.calltable_add_row(simple_shape(), HashMap::new());
        // Same row — add should dedup.
        assert_eq!(cf.total_calltable_size, 1);
        assert_eq!(
            cf.calltable_lookup_row(&simple_shape(), &HashMap::new()),
            Some(0)
        );
    }

    #[test]
    fn call_family_calltable_lookup_miss() {
        let cf = CallFamily::new(desc_key(1));
        assert!(
            cf.calltable_lookup_row(&simple_shape(), &HashMap::new())
                .is_none()
        );
    }

    #[test]
    fn call_family_absorb_alias() {
        let mut a = CallFamily::new(desc_key(1));
        let b = CallFamily::new(desc_key(2));
        a.absorb(&b);
        assert!(a.descs.contains_key(&desc_key(2)));
    }

    #[test]
    fn frozen_attr_family_tracks_attrs() {
        let mut faf = FrozenAttrFamily::new(desc_key(1));
        faf.set_s_value("x", SomeValue::Integer(SomeInteger::default()));
        assert!(matches!(faf.get_s_value("x"), SomeValue::Integer(_)));
        assert!(matches!(faf.get_s_value("y"), SomeValue::Impossible));
    }

    #[test]
    fn frozen_attr_family_update_merges() {
        let mut a = FrozenAttrFamily::new(desc_key(1));
        let mut b = FrozenAttrFamily::new(desc_key(2));
        a.set_s_value("x", SomeValue::Integer(SomeInteger::default()));
        b.set_s_value("y", SomeValue::String(SomeString::default()));
        a.update(&b);
        assert!(matches!(a.get_s_value("x"), SomeValue::Integer(_)));
        assert!(matches!(a.get_s_value("y"), SomeValue::String(_)));
        assert!(a.descs.contains_key(&desc_key(2)));
    }

    #[test]
    fn class_attr_family_tracks_single_attr() {
        let mut caf = ClassAttrFamily::new(desc_key(1));
        caf.set_s_value("x", SomeValue::Integer(SomeInteger::default()));
        // `attrname` is ignored by upstream — same s_value regardless.
        assert!(matches!(caf.get_s_value("x"), SomeValue::Integer(_)));
        assert!(matches!(caf.get_s_value("y"), SomeValue::Integer(_)));
    }

    #[test]
    fn class_attr_family_update_unions_s_value() {
        // Use compatible integer types to stay within the Phase 4
        // A4.6 pair-union subset.
        let mut a = ClassAttrFamily::new(desc_key(1));
        let mut b = ClassAttrFamily::new(desc_key(2));
        a.set_s_value("x", SomeValue::Integer(SomeInteger::new(true, false)));
        b.set_s_value("x", SomeValue::Integer(SomeInteger::new(false, false)));
        a.update(&b).unwrap();
        // Widened to signed.
        if let SomeValue::Integer(si) = &a.s_value {
            assert!(!si.nonneg);
        } else {
            panic!("expected SomeInteger s_value after union");
        }
    }

    #[test]
    fn no_standard_graph_error_carries_desc_key() {
        let err = NoStandardGraph(desc_key(42));
        assert!(format!("{err}").contains("42"));
    }

    /// `ClassAttrFamily.commonbase` defaults to `None` until populated
    /// by `merge_classpbc_getattr_into_classdef` (normalizecalls.py:232).
    /// `update`/`absorb` propagate descs / s_value but never touch
    /// `commonbase` — upstream attaches it post-hoc on the live family
    /// instance rather than during merging.
    #[test]
    fn class_attr_family_commonbase_default_none_and_unchanged_by_update() {
        let mut a = ClassAttrFamily::new(desc_key(1));
        let b = ClassAttrFamily::new(desc_key(2));
        assert!(a.commonbase.is_none(), "default commonbase must be None");
        a.update(&b).unwrap();
        assert!(a.commonbase.is_none(), "update must not touch commonbase");
    }

    // ---- Desc + FunctionDesc (commit 2) ----

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    fn ann_bk() -> (
        Rc<crate::annotator::annrpython::RPythonAnnotator>,
        Rc<Bookkeeper>,
    ) {
        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        (ann, bk)
    }

    fn int_sig(names: &[&str]) -> Signature {
        Signature::new(names.iter().map(|s| s.to_string()).collect(), None, None)
    }

    fn compiled_graph_func(
        src: &str,
        defaults: Vec<Constant>,
    ) -> crate::flowspace::model::GraphFunc {
        let code = rp_compile(src, Mode::Exec, "<test>".into(), Default::default())
            .expect("compile should succeed");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body should be a code constant");
        crate::flowspace::model::GraphFunc::from_host_code(
            crate::flowspace::bytecode::HostCode::from_code(inner),
            Constant::new(ConstValue::Dict(Default::default())),
            defaults,
        )
    }

    #[test]
    fn desc_new_carries_bookkeeper_and_pyobj() {
        let bk = bk();
        let desc = Desc::new(bk.clone(), None);
        assert!(desc.pyobj.is_none());
        assert!(Rc::ptr_eq(&desc.bookkeeper, &bk));
    }

    #[test]
    fn desc_querycallfamily_is_none_without_bookkeeper_unionfind() {
        let desc = Desc::new(bk(), None);
        assert!(desc.querycallfamily().is_none());
    }

    #[test]
    fn desc_getcallfamily_materializes_unionfind_entry() {
        let desc = Desc::new(bk(), None);
        assert!(desc.querycallfamily().is_none());
        let family = desc.getcallfamily().expect("callfamily must be created");
        assert!(family.borrow().descs.contains_key(&desc.identity));
        assert!(desc.querycallfamily().is_some());
    }

    #[test]
    fn function_desc_new_sets_all_fields() {
        let bk = bk();
        let fd = FunctionDesc::new(bk, None, "f", int_sig(&["x"]), None, None);
        assert_eq!(fd.name, "f");
        assert_eq!(fd.signature.argnames, vec!["x".to_string()]);
        assert_eq!(fd.knowntype, FunctionDescKnownType::Function);
        assert!(fd.defaults.is_empty());
        assert!(fd.cache.borrow().is_empty());
    }

    #[test]
    fn function_desc_getgraphs_returns_cached_values() {
        let fd = FunctionDesc::new(bk(), None, "f", int_sig(&[]), None, None);
        assert!(fd.getgraphs().is_empty());
    }

    #[test]
    fn function_desc_cachedgraph_derives_alt_name_from_key() {
        let (_ann, bk) = ann_bk();
        let func = compiled_graph_func("def f():\n    return 1\n", Vec::new());
        let host = HostObject::new_user_function(func.clone());
        let fd = FunctionDesc::new(
            bk,
            Some(host),
            "f",
            func.code.as_ref().unwrap().signature.clone(),
            Some(func.defaults.clone()),
            None,
        );
        let graph = fd
            .cachedgraph(
                GraphCacheKey::String("arg-or var".to_string()),
                None,
                Some(Box::new(|_translator, _func| {
                    Ok(make_pygraph(
                        Signature::new(Vec::new(), None, None),
                        Some(Vec::new()),
                    ))
                })),
            )
            .expect("builder graph");

        assert_eq!(graph.graph.borrow().name, "f__arg_or_var");
    }

    #[test]
    fn specialize_argtype_uses_known_type_name_in_cache_key() {
        let (_ann, bk) = ann_bk();
        let func = compiled_graph_func("def f(x):\n    return x\n", Vec::new());
        let host = HostObject::new_user_function(func.clone());
        let fd = FunctionDesc::new(
            bk,
            Some(host),
            "f",
            func.code.as_ref().unwrap().signature.clone(),
            Some(func.defaults.clone()),
            Some(Specializer::Argtype {
                parms: vec!["0".to_string()],
            }),
        );
        let mut inputcells = vec![SomeValue::Integer(SomeInteger::default())];
        let graph = fd
            .specialize(&mut inputcells, None)
            .expect("argtype specialization");

        assert_eq!(graph.graph.borrow().name, "f__int");
    }

    /// Build a PyGraph with a given signature/defaults for the
    /// getuniquegraph tests. Full `PyGraph::new` requires a HostCode;
    /// we construct the same shape manually so the test stays scoped
    /// to the FunctionDesc / PyGraph field comparison upstream runs.
    fn make_pygraph(sig: Signature, defaults: Option<Vec<Constant>>) -> Rc<PyGraph> {
        use crate::flowspace::model::{FunctionGraph, GraphFunc};
        let empty_globals = Constant::new(ConstValue::Dict(Default::default()));
        let func = GraphFunc::new("f", empty_globals);
        // Synthetic startblock — getuniquegraph doesn't traverse it.
        let startblock = crate::flowspace::model::Block::shared(vec![]);
        let graph = FunctionGraph::new("f", startblock);
        Rc::new(PyGraph {
            graph: Rc::new(std::cell::RefCell::new(graph)),
            func,
            signature: std::cell::RefCell::new(sig),
            defaults: std::cell::RefCell::new(defaults),
            access_directly: std::cell::Cell::new(false),
        })
    }

    #[test]
    fn function_desc_getuniquegraph_returns_single_cached_graph() {
        // upstream description.py:218-226 — a cache of exactly one
        // graph whose signature/defaults match returns that graph.
        let sig = int_sig(&["x"]);
        let fd = FunctionDesc::new(bk(), None, "f", sig.clone(), None, None);
        let graph = make_pygraph(sig, Some(Vec::new()));
        fd.cache
            .borrow_mut()
            .insert(GraphCacheKey::String("key".to_string()), graph.clone());
        let g = fd.getuniquegraph().expect("single matching graph");
        assert!(Rc::ptr_eq(&g, &graph));
    }

    #[test]
    fn function_desc_getuniquegraph_errors_on_signature_mismatch() {
        // upstream description.py:222-225 — `graph.signature !=
        // self.signature` raises NoStandardGraph. Requires both
        // `FunctionDesc.defaults: Vec<Constant>` and a `PyGraph`-typed
        // cache to be structurally comparable.
        let fd = FunctionDesc::new(bk(), None, "f", int_sig(&["x"]), None, None);
        let diverging_sig = int_sig(&["x", "y"]);
        fd.cache.borrow_mut().insert(
            GraphCacheKey::String("key".to_string()),
            make_pygraph(diverging_sig, Some(Vec::new())),
        );
        let err = fd.getuniquegraph().unwrap_err();
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("signature/defaults mismatch")
        );
    }

    #[test]
    fn function_desc_getuniquegraph_errors_on_defaults_mismatch() {
        let sig = int_sig(&["x"]);
        // FunctionDesc declares no defaults; graph claims `defaults =
        // (10,)`. Upstream description.py:223-225 flags this.
        let fd = FunctionDesc::new(bk(), None, "f", sig.clone(), None, None);
        fd.cache.borrow_mut().insert(
            GraphCacheKey::String("key".to_string()),
            make_pygraph(sig, Some(vec![Constant::new(ConstValue::Int(10))])),
        );
        let err = fd.getuniquegraph().unwrap_err();
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("signature/defaults mismatch")
        );
    }

    #[test]
    fn function_desc_getuniquegraph_errors_on_graph_defaults_none() {
        let sig = int_sig(&["x"]);
        let fd = FunctionDesc::new(bk(), None, "f", sig.clone(), None, None);
        fd.cache.borrow_mut().insert(
            GraphCacheKey::String("key".to_string()),
            make_pygraph(sig, None),
        );
        let err = fd.getuniquegraph().unwrap_err();
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("signature/defaults mismatch")
        );
    }

    #[test]
    fn function_desc_getuniquegraph_allows_mismatch_when_relax_sig_check() {
        let mut func = crate::flowspace::model::GraphFunc::new(
            "f",
            Constant::new(ConstValue::Dict(Default::default())),
        );
        func.relax_sig_check = Some(true);
        let pyobj = HostObject::new_user_function(func);
        let fd = FunctionDesc::new(bk(), Some(pyobj), "f", int_sig(&["x"]), None, None);
        let diverging_sig = int_sig(&["x", "y"]);
        let graph = make_pygraph(diverging_sig, Some(Vec::new()));
        fd.cache
            .borrow_mut()
            .insert(GraphCacheKey::String("key".to_string()), graph.clone());
        let got = fd
            .getuniquegraph()
            .expect("relax_sig_check must bypass signature/default mismatch");
        assert!(Rc::ptr_eq(&got, &graph));
    }

    #[test]
    fn function_desc_parse_arguments_respects_defaults() {
        // def f(a, b=10): ...   called with f(1)
        let bk = bk();
        let fd = FunctionDesc::new(
            bk,
            None,
            "f",
            int_sig(&["a", "b"]),
            // upstream's `defaults = (10,)` as a raw Python value; the
            // parse_arguments path walks each item through
            // `bookkeeper.immutablevalue` at call time.
            Some(vec![Constant::new(ConstValue::Int(10))]),
            None,
        );
        let args = ArgumentsForTranslation::new(
            vec![SomeValue::Integer(SomeInteger::new(true, false))],
            None,
            None,
        );
        let inputs = fd.parse_arguments(&args, None).unwrap();
        assert_eq!(inputs.len(), 2);
    }

    #[test]
    fn function_desc_parse_arguments_errors_on_signature_mismatch() {
        let bk = bk();
        let fd = FunctionDesc::new(bk, None, "f", int_sig(&["a"]), None, None);
        // Call with two args where signature expects one, no vararg.
        let args = ArgumentsForTranslation::new(
            vec![
                SomeValue::Integer(SomeInteger::new(true, false)),
                SomeValue::Integer(SomeInteger::new(true, false)),
            ],
            None,
            None,
        );
        let err = fd.parse_arguments(&args, None).unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("signature mismatch"));
    }

    #[test]
    fn function_desc_parse_arguments_with_graph_defaults_none_does_not_fallback() {
        let sig = Signature::new(vec!["a".into()], Some("rest".into()), None);
        let fd = FunctionDesc::new(
            bk(),
            None,
            "f",
            sig,
            Some(vec![Constant::new(ConstValue::Int(10))]),
            None,
        );
        let graph = make_pygraph(
            Signature::new(vec!["a".into(), ".star0".into()], None, None),
            None,
        );
        let args = ArgumentsForTranslation::new(
            vec![SomeValue::Integer(SomeInteger::new(true, false))],
            None,
            None,
        );
        let err = fd.parse_arguments(&args, Some(graph.as_ref())).unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("signature mismatch"));
    }

    #[test]
    fn function_desc_buildgraph_surfaces_flow_error_from_objspace() {
        use crate::flowspace::bytecode::HostCode;
        use crate::flowspace::objspace::CO_NEWLOCALS;

        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let mut func = crate::flowspace::model::GraphFunc::new(
            "f",
            Constant::new(ConstValue::Dict(Default::default())),
        );
        func.code = Some(Box::new(HostCode {
            id: HostCode::fresh_identity(),
            co_name: "f".to_string(),
            co_filename: "<test>".to_string(),
            co_firstlineno: 1,
            co_nlocals: 0,
            co_argcount: 0,
            co_stacksize: 0,
            co_flags: CO_NEWLOCALS,
            co_code: rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            co_varnames: Vec::new(),
            co_freevars: Vec::new(),
            co_cellvars: Vec::new(),
            consts: Vec::new(),
            names: Vec::new(),
            co_lnotab: Vec::new(),
            exceptiontable: Vec::new().into_boxed_slice(),
            signature: Signature::new(Vec::new(), None, None),
        }));
        let host = HostObject::new_user_function(func);
        let fd = FunctionDesc::new(
            ann.bookkeeper.clone(),
            Some(host),
            "f",
            int_sig(&[]),
            None,
            None,
        );
        let err = fd.buildgraph(None, None).unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("BytecodeCorruption"));
    }

    #[test]
    fn function_desc_pycall_without_pyobj_errors_cleanly() {
        // FunctionDesc with no pyobj can't reach `buildgraph` — pycall
        // surfaces the missing-pyobj error via specialize → cachedgraph
        // → buildgraph rather than panicking.
        let fd = FunctionDesc::new(bk(), None, "f", int_sig(&[]), None, None);
        let args = ArgumentsForTranslation::new(vec![], None, None);
        let err = fd
            .pycall(None, &args, &SomeValue::Impossible, None)
            .unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("missing pyobj"));
    }

    #[test]
    fn flatten_star_args_identity_for_plain_signature() {
        let fd = FunctionDesc::new(bk(), None, "f", int_sig(&["a", "b"]), None, None);
        let args = vec![
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::Integer(SomeInteger::default()),
        ];
        let (out, key, builder) = fd.flatten_star_args(&args).expect("identity path");
        assert!(matches!(key, GraphCacheKey::None));
        assert!(builder.is_none());
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn flatten_star_args_rejects_kwargs() {
        let sig = Signature::new(vec!["a".into()], None, Some("kw".into()));
        let fd = FunctionDesc::new(bk(), None, "f", sig, None, None);
        let args = vec![
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::Integer(SomeInteger::default()),
        ];
        let err = match fd.flatten_star_args(&args) {
            Ok(_) => panic!("expected flatten_star_args to fail"),
            Err(err) => err,
        };
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("** arguments are not supported")
        );
    }

    #[test]
    fn flatten_star_args_expands_tuple_for_vararg() {
        use super::super::model::SomeTuple;
        let sig = Signature::new(vec!["a".into()], Some("rest".into()), None);
        let fd = FunctionDesc::new(bk(), None, "f", sig, None, None);
        let args = vec![
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ])),
        ];
        let (out, key, builder) = fd.flatten_star_args(&args).expect("vararg path");
        assert_eq!(out.len(), 3, "a + 2 flattened star args");
        assert_eq!(
            key,
            GraphCacheKey::Tuple(vec![
                GraphCacheKey::String("star".to_string()),
                GraphCacheKey::Int(2),
            ])
        );
        assert!(builder.is_some());
    }

    #[test]
    fn default_specialize_rewrites_access_directly_in_place_when_jit_disabled() {
        let (_ann, bk) = ann_bk();
        let mut func = compiled_graph_func("def f(a):\n    return a\n", Vec::new());
        func._jit_look_inside_ = Some(false);
        let host = HostObject::new_user_function(func.clone());
        let fd = FunctionDesc::new(
            bk,
            Some(host),
            "f",
            func.code.as_ref().unwrap().signature.clone(),
            Some(func.defaults.clone()),
            None,
        );
        let mut flags = std::collections::BTreeMap::new();
        flags.insert("access_directly".into(), true);
        let mut inputcells = vec![SomeValue::Instance(SomeInstance::new(None, false, flags))];

        let graph = fd.specialize(&mut inputcells, None).unwrap();

        assert!(!graph.access_directly.get());
        match &inputcells[0] {
            SomeValue::Instance(inst) => assert!(!inst.flags.contains_key("access_directly")),
            other => panic!("expected SomeInstance, got {other:?}"),
        }
    }

    #[test]
    fn default_specialize_vararg_builder_updates_signature_defaults_and_entry_block() {
        use super::super::model::SomeTuple;

        let (_ann, bk) = ann_bk();
        let defaults = vec![Constant::new(ConstValue::Int(1))];
        let func = compiled_graph_func("def f(a=1, *rest):\n    return a\n", defaults.clone());
        let host = HostObject::new_user_function(func.clone());
        let fd = FunctionDesc::new(
            bk,
            Some(host),
            "f",
            func.code.as_ref().unwrap().signature.clone(),
            Some(defaults),
            None,
        );
        let mut inputcells = vec![
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ])),
        ];

        let graph = fd.specialize(&mut inputcells, None).unwrap();

        assert_eq!(
            inputcells.len(),
            2,
            "caller inputcells must stay unflattened"
        );
        assert_eq!(
            graph.signature.borrow().argnames,
            vec!["a".to_string(), ".star0".to_string(), ".star1".to_string()]
        );
        assert!(graph.signature.borrow().varargname.is_none());
        assert!(graph.signature.borrow().kwargname.is_none());
        assert!(graph.defaults.borrow().is_none());
        let startblock = graph.graph.borrow().startblock.clone();
        assert_eq!(startblock.borrow().inputargs.len(), 3);
        assert!(startblock.borrow().exits[0].borrow().prevblock.is_some());
    }

    #[test]
    fn specialize_arg_or_var_uses_constant_value_key() {
        let (_ann, bk) = ann_bk();
        let func = compiled_graph_func("def f(x):\n    return x\n", Vec::new());
        let host = HostObject::new_user_function(func.clone());
        let fd = FunctionDesc::new(
            bk,
            Some(host),
            "f",
            func.code.as_ref().unwrap().signature.clone(),
            Some(func.defaults.clone()),
            Some(Specializer::ArgOrVar {
                parms: vec!["0".to_string()],
            }),
        );
        let mut s_int = SomeInteger::default();
        s_int.base.const_box = Some(Constant::new(ConstValue::Int(42)));
        let mut inputcells = vec![SomeValue::Integer(s_int)];

        let graph = fd
            .specialize(&mut inputcells, None)
            .expect("arg_or_var graph");

        assert_eq!(graph.graph.borrow().name, "f__42");
    }

    #[test]
    fn specialize_arg_accepts_single_desc_pbc() {
        let (_ann, bk) = ann_bk();
        let callee_func = compiled_graph_func("def callee():\n    return 1\n", Vec::new());
        let callee_host = HostObject::new_user_function(callee_func.clone());
        let callee = Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            Some(callee_host),
            "callee",
            callee_func.code.as_ref().unwrap().signature.clone(),
            Some(callee_func.defaults.clone()),
            None,
        )));
        let func = compiled_graph_func("def f(x):\n    return x\n", Vec::new());
        let host = HostObject::new_user_function(func.clone());
        let fd = FunctionDesc::new(
            bk,
            Some(host),
            "f",
            func.code.as_ref().unwrap().signature.clone(),
            Some(func.defaults.clone()),
            Some(Specializer::Arg {
                parms: vec!["0".to_string()],
            }),
        );
        let pbc = SomePBC::new(vec![DescEntry::Function(callee)], false);
        let mut inputcells = vec![SomeValue::PBC(pbc)];

        let graph = fd.specialize(&mut inputcells, None).expect("arg graph");

        assert_eq!(graph.graph.borrow().name, "f__callee");
    }

    #[test]
    fn memo_desc_pycall_without_pyobj_errors_cleanly() {
        // MemoDesc shares FunctionDesc's specialize path; no pyobj
        // surfaces the same buildgraph error via specialize →
        // cachedgraph → buildgraph.
        let fd = FunctionDesc::new(bk(), None, "f", int_sig(&[]), None, None);
        let md = MemoDesc::new(fd);
        let args = ArgumentsForTranslation::new(vec![], None, None);
        let err = md
            .pycall(None, &args, &SomeValue::Impossible, None)
            .unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("missing pyobj"));
    }

    #[test]
    fn function_desc_normalize_args_rejects_both_enforceargs_and_signature() {
        let bk = bk();
        let mut fd = FunctionDesc::new(bk.clone(), None, "f", int_sig(&["a"]), None, None);
        fd.annenforceargs = Some(Rc::new(super::super::signature::Sig::new(Vec::new())));
        fd.annsignature = Some(Rc::new(AnnSignature {
            params: Vec::new(),
            result: ParamType::Marker(super::super::signature::TypeMarker::AnyType),
        }));
        let mut inputs = vec![SomeValue::Integer(SomeInteger::default())];
        let err = fd.normalize_args(&mut inputs).unwrap_err();
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("signature and enforceargs")
        );
    }

    #[test]
    fn memodesc_wraps_functiondesc() {
        let fd = FunctionDesc::new(bk(), None, "m", int_sig(&[]), None, None);
        let md = MemoDesc::new(fd);
        assert_eq!(md.base.name, "m");
    }

    // ---- MethodDesc + FrozenDesc + MethodOfFrozenDesc (commit 3) ----

    fn wrap_fd(bk: &Rc<Bookkeeper>, name: &str) -> Rc<RefCell<FunctionDesc>> {
        Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            name,
            int_sig(&["self"]),
            None,
            None,
        )))
    }

    #[test]
    fn method_desc_new_carries_funcdesc_and_classes() {
        let bk = bk();
        let fd = wrap_fd(&bk, "m");
        let md = MethodDesc::new(
            bk,
            fd.clone(),
            ClassDefKey::from_raw(1),
            Some(ClassDefKey::from_raw(2)),
            "m",
            std::collections::BTreeMap::new(),
        );
        assert_eq!(md.name, "m");
        assert_eq!(md.originclassdef, ClassDefKey::from_raw(1));
        assert_eq!(md.selfclassdef, Some(ClassDefKey::from_raw(2)));
        assert!(Rc::ptr_eq(&md.funcdesc, &fd));
    }

    #[test]
    fn method_desc_rowkey_returns_funcdesc_identity() {
        let bk = bk();
        let fd = wrap_fd(&bk, "m");
        let md = MethodDesc::new(
            bk,
            fd.clone(),
            ClassDefKey::from_raw(1),
            None,
            "m",
            std::collections::BTreeMap::new(),
        );
        assert_eq!(md.rowkey(), FunctionDesc::rowkey(&fd));
    }

    #[test]
    fn method_desc_func_args_errors_when_unbound() {
        let bk = bk();
        let fd = wrap_fd(&bk, "m");
        let md = MethodDesc::new(
            bk,
            fd,
            ClassDefKey::from_raw(1),
            None,
            "m",
            std::collections::BTreeMap::new(),
        );
        let args = ArgumentsForTranslation::new(vec![], None, None);
        let err = md.func_args(&args).unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("unbound MethodDesc"));
    }

    #[test]
    fn method_desc_func_args_prepends_someinstance_when_bound() {
        let bk = bk();
        let classdef = crate::annotator::classdesc::ClassDef::new_standalone("pkg.C", None);
        bk.register_classdef(classdef.clone());
        let fd = wrap_fd(&bk, "m");
        let md = MethodDesc::new(
            bk,
            fd,
            ClassDefKey::from_raw(1),
            Some(ClassDefKey::from_classdef(&classdef)),
            "m",
            std::collections::BTreeMap::new(),
        );
        let args = ArgumentsForTranslation::new(vec![], None, None);
        let out = md.func_args(&args).expect("bound method args");
        assert!(matches!(out.arguments_w[0], SomeValue::Instance(_)));
    }

    #[test]
    fn method_desc_simplify_desc_set_intersects_flags_and_drops_shadowed_subclass() {
        let bk = bk();
        let base = crate::annotator::classdesc::ClassDef::new_standalone("pkg.Base", None);
        let child = crate::annotator::classdesc::ClassDef::new_standalone("pkg.Child", Some(&base));
        bk.register_classdef(base.clone());
        bk.register_classdef(child.clone());
        let fd = wrap_fd(&bk, "m");
        let origin = ClassDefKey::from_classdef(&base);

        let mut base_flags = std::collections::BTreeMap::new();
        base_flags.insert("access_directly".into(), true);
        let mut child_flags = base_flags.clone();
        child_flags.insert("fresh_malloc".into(), true);

        let base_md = bk.getmethoddesc(
            &fd,
            origin,
            Some(ClassDefKey::from_classdef(&base)),
            "m",
            base_flags.clone(),
        );
        let child_md = bk.getmethoddesc(
            &fd,
            origin,
            Some(ClassDefKey::from_classdef(&child)),
            "m",
            child_flags,
        );

        let mut descs = std::collections::BTreeMap::new();
        descs.insert(DescKey::from_rc(&base_md), DescEntry::Method(base_md));
        descs.insert(DescKey::from_rc(&child_md), DescEntry::Method(child_md));

        MethodDesc::simplify_desc_set(&mut descs);

        assert_eq!(descs.len(), 1);
        let remaining = descs
            .values()
            .next()
            .unwrap()
            .as_method()
            .expect("remaining desc must be MethodDesc");
        let borrowed = remaining.borrow();
        assert_eq!(
            borrowed.selfclassdef,
            Some(ClassDefKey::from_classdef(&base))
        );
        assert_eq!(borrowed.flags, base_flags);
    }

    #[test]
    fn frozen_desc_new_accepts_module_host_object() {
        let bk = bk();
        let pyobj = HostObject::new_module("os");
        let fd = FrozenDesc::new(bk, pyobj).unwrap();
        assert!(fd.attrcache.borrow().is_empty());
        assert_eq!(
            fd.knowntype.as_ref().map(HostObject::qualname),
            Some("module")
        );
    }

    #[test]
    fn new_or_old_class_maps_host_kinds_to_host_types() {
        let cls = HostObject::new_class("C", vec![]);
        assert_eq!(
            new_or_old_class(&cls).as_ref().map(HostObject::qualname),
            Some("type")
        );

        let inst = HostObject::new_instance(cls.clone(), vec![]);
        assert_eq!(
            new_or_old_class(&inst).as_ref().map(HostObject::qualname),
            Some("C")
        );

        let func = HostObject::new_user_function(crate::flowspace::model::GraphFunc::new(
            "f",
            Constant::new(ConstValue::Dict(Default::default())),
        ));
        assert_eq!(
            new_or_old_class(&func).as_ref().map(HostObject::qualname),
            Some("function")
        );

        let prop = HostObject::new_property("p", None, None, None);
        assert_eq!(
            new_or_old_class(&prop).as_ref().map(HostObject::qualname),
            Some("property")
        );
    }

    #[test]
    fn frozen_desc_reads_module_attribute_through_module_get() {
        let bk = bk();
        let pyobj = HostObject::new_module("os");
        let sentinel = HostObject::new_class("Sentinel", vec![]);
        pyobj.module_set("x", sentinel);
        let fd = FrozenDesc::new(bk, pyobj).unwrap();
        let v = fd.read_attribute("x").unwrap();
        assert!(matches!(v, ConstValue::HostObject(_)));
        assert!(fd.has_attribute("x").unwrap());
    }

    #[test]
    fn frozen_desc_missing_attribute_errors() {
        let bk = bk();
        let pyobj = HostObject::new_module("os");
        let fd = FrozenDesc::new(bk, pyobj).unwrap();
        assert!(!fd.has_attribute("missing").unwrap());
        let err = fd.read_attribute("missing").unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("AttributeError"));
    }

    #[test]
    fn frozen_desc_warn_missing_skips_dollar_prefix() {
        let bk = bk();
        let pyobj = HostObject::new_module("os");
        let fd = FrozenDesc::new(bk, pyobj).unwrap();
        assert!(fd.warn_missing_attribute("x").unwrap());
        assert!(!fd.warn_missing_attribute("$memofield_x").unwrap());
    }

    #[test]
    fn frozen_desc_s_read_attribute_returns_impossible_on_miss() {
        let bk = bk();
        let pyobj = HostObject::new_module("os");
        let fd = FrozenDesc::new(bk, pyobj).unwrap();
        let s = fd.s_read_attribute("missing").unwrap();
        assert!(matches!(s, SomeValue::Impossible));
    }

    #[test]
    fn frozen_desc_default_read_checks_instance_dict_before_mro() {
        use crate::flowspace::model::ConstValue;
        let bk = bk();
        // class C: a = 1
        let class_c = HostObject::new_class("C", vec![]);
        class_c.class_set("a", ConstValue::Int(1));
        // instance = C()
        let inst = HostObject::new_instance(class_c.clone(), vec![]);
        // Per-instance shadow: inst.a = 2
        inst.instance_set("a", ConstValue::Int(2));
        // Instance-only attribute: inst.b = 3
        inst.instance_set("b", ConstValue::Int(3));

        let fd = FrozenDesc::new(bk, inst).unwrap();
        // Instance dict shadows class MRO.
        let a = fd.read_attribute("a").unwrap();
        assert!(matches!(a, ConstValue::Int(2)));
        // Instance-only attribute.
        let b = fd.read_attribute("b").unwrap();
        assert!(matches!(b, ConstValue::Int(3)));
        // Attribute only on class MRO: class_set on class, not
        // instance, still reachable.
        class_c.class_set("c", ConstValue::Int(10));
        // Bypass attrcache by creating a fresh FrozenDesc.
        let inst2 = HostObject::new_instance(class_c.clone(), vec![]);
        let fd2 = FrozenDesc::new(bk2(), inst2).unwrap();
        let c = fd2.read_attribute("c").unwrap();
        assert!(matches!(c, ConstValue::Int(10)));
    }

    fn bk2() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    #[test]
    fn frozen_desc_create_new_attribute_and_read() {
        let bk = bk();
        let pyobj = HostObject::new_module("mod");
        let fd = FrozenDesc::new(bk, pyobj).unwrap();
        fd.create_new_attribute("k", ConstValue::Int(42)).unwrap();
        assert!(matches!(
            fd.read_attribute("k").unwrap(),
            ConstValue::Int(42)
        ));
    }

    #[test]
    fn frozen_desc_create_new_attribute_rejects_name_clash() {
        let bk = bk();
        let pyobj = HostObject::new_module("mod");
        let fd = FrozenDesc::new(bk, pyobj).unwrap();
        fd.create_new_attribute("k", ConstValue::Int(1)).unwrap();
        let err = fd
            .create_new_attribute("k", ConstValue::Int(2))
            .unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("name clash"));
    }

    #[test]
    fn frozen_desc_custom_read_attribute_callback_is_invoked() {
        // upstream description.py:530-534 — when the caller passes a
        // `read_attribute` closure, it becomes the attribute source
        // (bypassing the default `getattr(pyobj, attr)`). The Rust
        // port exposes this via `new_with_read_attribute`. Verify the
        // custom callback is what `read_attribute` consults, and that
        // the attrcache memoises its result (description.py:553-558).
        let bk = bk();
        let pyobj = HostObject::new_module("mod");
        let calls = Rc::new(std::cell::Cell::new(0usize));
        let callback: FrozenReadAttr = {
            let calls = calls.clone();
            Box::new(
                move |attr: &str| -> Result<ConstValue, crate::flowspace::model::HostGetAttrError> {
                    calls.set(calls.get() + 1);
                    match attr {
                        "x" => Ok(ConstValue::Int(42)),
                        _ => Err(crate::flowspace::model::HostGetAttrError::Missing),
                    }
                },
            )
        };
        let fd = FrozenDesc::new_with_read_attribute(bk, pyobj, callback).unwrap();
        // First read hits the callback.
        assert!(matches!(
            fd.read_attribute("x").unwrap(),
            ConstValue::Int(42)
        ));
        assert_eq!(calls.get(), 1);
        // Second read is served from attrcache — callback count stays 1.
        assert!(matches!(
            fd.read_attribute("x").unwrap(),
            ConstValue::Int(42)
        ));
        assert_eq!(calls.get(), 1);
        // Missing attribute surfaces as AttributeError-equivalent and
        // does route to the callback (not cached).
        assert!(fd.read_attribute("missing").is_err());
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn frozen_desc_default_reader_walks_class_mro() {
        // upstream description.py:534 — default reader is
        // `lambda attr: getattr(pyobj, attr)`. `getattr(cls, name)` on
        // a class walks `__mro__` after checking `cls.__dict__`, so an
        // attribute defined only on a base must still resolve when the
        // FrozenDesc wraps the subclass.
        let bk = bk();
        let base = HostObject::new_class("Base", vec![]);
        base.class_set("CONST", ConstValue::Int(99));
        let sub = HostObject::new_class("Sub", vec![base]);
        let fd = FrozenDesc::new(bk, sub).unwrap();
        let v = fd.read_attribute("CONST").unwrap();
        assert!(matches!(v, ConstValue::Int(99)));
    }

    #[test]
    fn frozen_desc_default_reader_unwraps_staticmethod() {
        let bk = bk();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func =
            HostObject::new_user_function(crate::flowspace::model::GraphFunc::new("f", globals));
        let cls = HostObject::new_class("Cls", vec![]);
        cls.class_set(
            "f",
            ConstValue::HostObject(HostObject::new_staticmethod("Cls.f", func.clone())),
        );

        let fd = FrozenDesc::new(bk, cls).unwrap();
        let v = fd.read_attribute("f").unwrap();
        assert_eq!(v, ConstValue::HostObject(func));
    }

    #[test]
    fn frozen_desc_default_reader_routes_instance_through_class() {
        // upstream default reader: `getattr(instance, name)` falls
        // through to the class hierarchy when the instance has no
        // per-object dict. Rust HostObject::Instance models no per-
        // instance dict, so the class-hierarchy lookup is the
        // authoritative path.
        let bk = bk();
        let cls = HostObject::new_class("Cls", vec![]);
        cls.class_set("TAG", ConstValue::byte_str("tag"));
        let instance = HostObject::new_instance(cls, vec![]);
        let fd = FrozenDesc::new(bk, instance).unwrap();
        let v = fd.read_attribute("TAG").unwrap();
        assert!(v.string_eq("tag"));
    }

    #[test]
    fn frozen_desc_default_reader_unwraps_instance_staticmethod() {
        let bk = bk();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func =
            HostObject::new_user_function(crate::flowspace::model::GraphFunc::new("f", globals));
        let cls = HostObject::new_class("Cls", vec![]);
        cls.class_set(
            "f",
            ConstValue::HostObject(HostObject::new_staticmethod("Cls.f", func.clone())),
        );
        let instance = HostObject::new_instance(cls, vec![]);

        let fd = FrozenDesc::new(bk, instance).unwrap();
        let v = fd.read_attribute("f").unwrap();
        assert_eq!(v, ConstValue::HostObject(func));
    }

    #[test]
    fn frozen_desc_default_reader_binds_instance_function_method() {
        let bk = bk();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func =
            HostObject::new_user_function(crate::flowspace::model::GraphFunc::new("f", globals));
        let cls = HostObject::new_class("Cls", vec![]);
        cls.class_set("f", ConstValue::HostObject(func));
        let instance = HostObject::new_instance(cls.clone(), vec![]);

        let fd = FrozenDesc::new(bk, instance.clone()).unwrap();
        let v = fd.read_attribute("f").unwrap();
        let ConstValue::HostObject(bound) = v else {
            panic!("expected bound-method HostObject");
        };
        assert!(bound.is_bound_method());
        assert_eq!(bound.bound_method_self(), Some(&instance));
        assert_eq!(bound.bound_method_name(), Some("f"));
        assert_eq!(bound.bound_method_origin_class(), Some(&cls));

        let s = fd.s_read_attribute("f").unwrap();
        match s {
            SomeValue::PBC(pbc) => {
                assert_eq!(
                    pbc.get_kind().unwrap(),
                    crate::annotator::model::DescKind::Method
                )
            }
            other => panic!("expected method SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn frozen_desc_default_reader_binds_classmethod() {
        let bk = bk();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func =
            HostObject::new_user_function(crate::flowspace::model::GraphFunc::new("cm", globals));
        let cls = HostObject::new_class("Cls", vec![]);
        cls.class_set(
            "cm",
            ConstValue::HostObject(HostObject::new_classmethod("Cls.cm", func)),
        );

        let fd = FrozenDesc::new(bk, cls.clone()).unwrap();
        let v = fd.read_attribute("cm").unwrap();
        let ConstValue::HostObject(bound) = v else {
            panic!("expected bound-method HostObject");
        };
        assert!(bound.is_bound_method());
        assert_eq!(bound.bound_method_self(), Some(&cls));
        assert_eq!(bound.bound_method_name(), Some("cm"));
        assert_eq!(bound.bound_method_origin_class(), Some(&cls));

        let err = fd.s_read_attribute("cm").unwrap_err();
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("classmethods are not supported")
        );
    }

    #[test]
    fn frozen_desc_family_methods_use_unionfind() {
        let bk = bk();
        let pyobj = HostObject::new_module("mod");
        let fd = FrozenDesc::new(bk, pyobj).unwrap();
        assert!(fd.queryattrfamily().is_none());
        let family = fd.getattrfamily().expect("family");
        assert!(family.borrow().descs.contains_key(&fd.base.identity));
        assert!(!fd.mergeattrfamilies(&[]).unwrap());
    }

    #[test]
    fn method_of_frozen_desc_new_and_rowkey() {
        let bk = bk();
        let fd = wrap_fd(&bk, "m");
        let frozen = Rc::new(RefCell::new(
            FrozenDesc::new(bk.clone(), HostObject::new_module("mod")).unwrap(),
        ));
        let mfd = MethodOfFrozenDesc::new(bk, fd.clone(), frozen);
        assert_eq!(mfd.rowkey(), FunctionDesc::rowkey(&fd));
    }

    #[test]
    fn method_of_frozen_desc_func_args_prepends_frozen_pbc() {
        let bk = bk();
        let fd = wrap_fd(&bk, "m");
        let frozen = Rc::new(RefCell::new(
            FrozenDesc::new(bk.clone(), HostObject::new_module("mod")).unwrap(),
        ));
        let mfd = MethodOfFrozenDesc::new(bk, fd, frozen);
        let args = ArgumentsForTranslation::new(vec![], None, None);
        let out = mfd.func_args(&args).expect("method-of-frozen args");
        assert!(matches!(out.arguments_w[0], SomeValue::PBC(_)));
    }

    #[test]
    fn build_calltable_row_uses_method_rowkey_dispatch() {
        let (_ann, bk) = ann_bk();
        let classdef = crate::annotator::classdesc::ClassDef::new_standalone("pkg.C", None);
        bk.register_classdef(classdef.clone());
        let func = compiled_graph_func("def m(self):\n    return self\n", Vec::new());
        let host = HostObject::new_user_function(func.clone());
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            Some(host),
            "m",
            func.code.as_ref().unwrap().signature.clone(),
            Some(func.defaults.clone()),
            None,
        )));
        let md = bk.getmethoddesc(
            &fd,
            ClassDefKey::from_classdef(&classdef),
            Some(ClassDefKey::from_classdef(&classdef)),
            "m",
            std::collections::BTreeMap::new(),
        );
        let args = ArgumentsForTranslation::new(vec![], None, None);

        let row = build_calltable_row(&[DescEntry::Method(md)], &args, None).unwrap();

        let key = FunctionDesc::rowkey(&fd);
        assert_eq!(row.len(), 1);
        assert!(row.contains_key(&key));
    }

    #[test]
    fn call_family_find_row_reuses_current_bookkeeper_position() {
        let (_ann, bk) = ann_bk();
        let func = compiled_graph_func("def f(x):\n    return x\n", Vec::new());
        let host = HostObject::new_user_function(func.clone());
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            Some(host),
            "f",
            func.code.as_ref().unwrap().signature.clone(),
            Some(func.defaults.clone()),
            None,
        )));
        let args = ArgumentsForTranslation::new(
            vec![SomeValue::Integer(SomeInteger::default())],
            None,
            None,
        );
        FunctionDesc::consider_call_site(&[fd.clone()], &args, &SomeValue::Impossible, None)
            .unwrap();
        bk.enter(Some(PositionKey::new(7, 8, 9)));

        let family = fd.borrow().base.getcallfamily().unwrap();
        let (shape, index) = family
            .borrow()
            .find_row(&bk, &[DescEntry::Function(fd)], &args, None)
            .unwrap();

        assert_eq!(shape, args.rawshape());
        assert_eq!(index, 0);
        assert_eq!(bk.current_position_key(), Some(PositionKey::new(7, 8, 9)));
        bk.leave();
    }

    #[test]
    fn call_location_specializer_uses_current_bookkeeper_position() {
        let (_ann, bk) = ann_bk();
        let func = compiled_graph_func("def f(x):\n    return x\n", Vec::new());
        let host = HostObject::new_user_function(func.clone());
        let fd = FunctionDesc::new(
            bk.clone(),
            Some(host),
            "f",
            func.code.as_ref().unwrap().signature.clone(),
            Some(func.defaults.clone()),
            Some(Specializer::CallLocation),
        );

        let mut inputs1 = vec![SomeValue::Integer(SomeInteger::default())];
        bk.set_position_key(Some(PositionKey::new(1, 2, 3)));
        let graph1 = fd
            .specialize(&mut inputs1, None)
            .expect("first call-location specialization");

        let mut inputs2 = vec![SomeValue::Integer(SomeInteger::default())];
        bk.set_position_key(Some(PositionKey::new(1, 2, 4)));
        let graph2 = fd
            .specialize(&mut inputs2, None)
            .expect("second call-location specialization");

        assert!(!Rc::ptr_eq(&graph1, &graph2));
        assert_eq!(fd.cache.borrow().len(), 2);
    }
}
