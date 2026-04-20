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
//! opaque [`ClassDefKey`] handles until `classdesc.py` lands.
//!
//! ## PRE-EXISTING-ADAPTATION: DescKey
//!
//! Upstream keys `CallFamily.descs`, `FrozenAttrFamily.descs`, and
//! `ClassAttrFamily.descs` on the `Desc` instance itself — Python
//! dict uses object identity by default for user classes. The Rust
//! port uses a [`DescKey`] newtype wrapping `usize` (the pointer
//! identity of the eventual `Rc<Desc>`). Until commits 2-3 land the
//! real `Desc` hierarchy, callers pass a pointer-derived usize to
//! key entries; once `Desc` is ported, the helper
//! [`DescKey::from_desc_ptr`] will supply the identity hash from
//! `Rc::as_ptr`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::argument::{ArgErr, ArgumentsForTranslation};
use super::bookkeeper::{Bookkeeper, PositionKey};
use super::model::{
    AnnotatorError, SomeInstance, SomeObjectTrait, SomePBC, SomeValue, s_impossible_value, union,
};
use super::policy::Specializer;
use super::signature::ParamType;
use crate::flowspace::argument::{CallShape, Signature};
use crate::flowspace::model::{ConstValue, Constant, HostObject};
use crate::flowspace::objspace::build_flow;
use crate::flowspace::pygraph::PyGraph;
use crate::tool::algo::unionfind::UnionFindInfo;

/// Opaque identity handle for a `Desc` instance.
///
/// Upstream uses Python object identity; the Rust port stores a
/// `usize` derived from `Rc::as_ptr(&desc)`. Commits 2-3 add a
/// helper constructor once the `Desc` struct lands.
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

    // `find_row(bookkeeper, descs, args, op)` (description.py:54-59) is
    // deferred — it calls `build_calltable_row` which in turn invokes
    // `desc.get_graph(args, op)`. `Desc.get_graph` lands in
    // description.py commits 2/3 alongside the FunctionDesc port.
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
        }
    }

    /// RPython `ClassAttrFamily.update(other)` (description.py:118-121).
    ///
    /// Returns `Result<(), UnionError>` because `unionof` on the
    /// attribute value can fail (the pair-union subset is still
    /// Phase 5 P5.2+ dependent).
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

    /// RPython `Desc.simplify_desc_set(descs)` (description.py:180-182).
    ///
    /// Base `@staticmethod` no-op; subclasses override to collapse
    /// shadowed MethodDescs.
    pub fn simplify_desc_set(_descs: &mut Vec<Rc<RefCell<Desc>>>) {}

    /// Pointer-identity key for [`CallFamily`] / `FrozenAttrFamily` /
    /// `ClassAttrFamily` storage. Upstream uses `desc` itself as the
    /// key (dict-by-identity).
    pub fn desc_key(self_rc: &Rc<RefCell<Desc>>) -> DescKey {
        DescKey::from_rc(self_rc)
    }
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
    /// upstream `ClassDesc` — classdesc.py:488-918. Currently the c1
    /// shell (`classdesc::ClassDesc::new_shell`) until c2 lands the
    /// full `__init__` body.
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
/// match upstream line-by-line; the graph-building / specialization
/// / pycall methods are stubbed pending the annrpython.py driver
/// (see doc on [`Self::pycall`] for the dep list).
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
    /// (specializer-dependent). The Rust port stores the serialized
    /// key as a [`String`] — `cachedgraph(key, …)` at description.py:
    /// 228-249 builds an `alt_name` from the same value, giving a
    /// natural stringification. Values are [`PyGraph`] because
    /// `buildgraph` / `translator.buildflowgraph` produces
    /// `PyGraph` instances; this is what lets `getuniquegraph`
    /// access `graph.signature` / `graph.defaults` for the
    /// description.py:223-225 comparison.
    pub cache: RefCell<HashMap<String, Rc<PyGraph>>>,
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

    fn cachedgraph(
        &self,
        key: &str,
        alt_name: Option<&str>,
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        if let Some(existing) = self.cache.borrow().get(key) {
            return Ok(existing.clone());
        }
        let computed_alt_name = alt_name.or_else(|| if key.is_empty() { None } else { Some(key) });
        let graph = self.buildgraph(computed_alt_name)?;
        self.cache
            .borrow_mut()
            .insert(key.to_string(), graph.clone());
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
        graph_signature: Option<&Signature>,
        graph_defaults: Option<&[ConstValue]>,
    ) -> Result<Vec<SomeValue>, AnnotatorError> {
        let (signature, defaults_const) = match (graph_signature, graph_defaults) {
            (Some(sig), Some(d)) => (sig, Some(d)),
            (None, _) => (&self.signature, None),
            (Some(sig), None) => (sig, None),
        };
        // upstream description.py:256-264 — `if defaults:` walks each
        // Python value through `self.bookkeeper.immutablevalue(x)` at
        // parse time. `graph_defaults` overrides `self.defaults` when
        // the specializer built a `PyGraph` whose `__defaults__` tuple
        // diverged from the bare FunctionDesc's.
        let mut defs_s: Vec<SomeValue> = Vec::new();
        let defaults_source: Vec<ConstValue> = match defaults_const {
            Some(ds) => ds.to_vec(),
            None => self.defaults.iter().map(|c| c.value.clone()).collect(),
        };
        for d in &defaults_source {
            // upstream: `if x is NODEFAULT: defs_s.append(None)`.
            // We represent NODEFAULT as s_impossible_value, matching
            // the Phase 5 P5.2 representation; a richer "missing"
            // encoding lands when NODEFAULT surfaces as a real enum.
            let s = self.base.bookkeeper.immutablevalue(d)?;
            defs_s.push(s);
        }
        args.match_signature(signature, Some(&defs_s))
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
        if enforceargs.is_some() {
            // upstream: `enforceargs(self, inputs_s)` — Sig.__call__
            // is Phase 5 P5.2 blocked (signature.py:113-147). Return
            // loud error rather than silently skip.
            return Err(AnnotatorError::new(format!(
                "{}: Sig.__call__ path requires description.py commit 3+ bookkeeper \
                 integration (Phase 5 P5.2 dep — see signature.py:113-147)",
                self.name
            )));
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
    ///
    pub fn buildgraph(&self, alt_name: Option<&str>) -> Result<Rc<PyGraph>, AnnotatorError> {
        let pyobj = self
            .base
            .pyobj
            .clone()
            .ok_or_else(|| AnnotatorError::new("FunctionDesc.buildgraph: missing pyobj"))?;
        let graph_func = pyobj.user_function().cloned().ok_or_else(|| {
            AnnotatorError::new(format!(
                "FunctionDesc.buildgraph({}): pyobj is not a user function",
                self.name
            ))
        })?;
        let flow_graph = build_flow(graph_func.clone()).map_err(|err| {
            AnnotatorError::new(format!("FunctionDesc.buildgraph({}): {:?}", self.name, err))
        })?;
        let code = graph_func.code.as_ref().ok_or_else(|| {
            AnnotatorError::new(format!(
                "FunctionDesc.buildgraph({}): missing code object",
                self.name
            ))
        })?;
        let signature = code.signature.clone();
        let mut pygraph = PyGraph {
            graph: flow_graph,
            func: graph_func,
            signature,
            defaults: self.defaults.clone(),
        };
        if let Some(alt_name) = alt_name {
            pygraph.graph.name = alt_name.to_string();
        }
        Ok(Rc::new(pygraph))
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
    /// allow signature mismatches. The Rust `HostObject::UserFunction`
    /// surface doesn't carry arbitrary attributes yet (classdesc.py
    /// dep), so this port always runs the structural check; once
    /// that surface lands, a `pyobj.get_bool_attr("relax_sig_check")`
    /// gate gets added.
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
        if graph.signature != self.signature || graph.defaults != self.defaults {
            return Err(AnnotatorError::new(format!(
                "NoStandardGraph({}): signature/defaults mismatch",
                self.name
            )));
        }
        Ok(graph.clone())
    }

    /// RPython `FunctionDesc.specialize(inputcells, op=None)`
    /// (description.py:272-281).
    ///
    /// The optional `op_key` parameter threads the call-site position
    /// from the flow-space operation through the annotator driver so
    /// [`Specializer::CallLocation`] can cache graphs per call site
    /// (upstream `specialize_call_location(funcdesc, args_s, op)` keys
    /// on `(op,)` — specialize.py:368-370). When `None`, CallLocation
    /// falls back to a shared key — acceptable because the annrpython
    /// driver isn't wired yet (Phase 5 P5.2+ dep).
    pub fn specialize(
        &self,
        inputcells: &mut [SomeValue],
        op_key: Option<PositionKey>,
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        self.normalize_args(inputcells)?;
        match self.specializer.as_ref().unwrap_or(&Specializer::Default) {
            Specializer::Default => {
                // upstream `default_specialize(funcdesc, args_s)`
                // (specialize.py:60-85). Scan args_s for
                // `SomeInstance` carrying `access_directly`; when
                // `_jit_look_inside_` (default True) is set, fork the
                // cached graph into an access-directly variant. When
                // `_jit_look_inside_=False`, strip the flag from the
                // instance annotation so the regular graph is used.
                //
                // `flatten_star_args` (specialize.py:14-58) — the
                // *arg builder/key pair — is deferred; it needs
                // `translator.buildflowgraph` which isn't in this
                // port yet. Functions without varargs (the common
                // case) take the `args_s, None, None` fallback.
                let jit_look_inside = self
                    .base
                    .pyobj
                    .as_ref()
                    .and_then(|pyobj| pyobj.class_get("_jit_look_inside_"))
                    .map(|v| !matches!(v, ConstValue::Bool(false)))
                    .unwrap_or(true);
                let mut access_directly = false;
                for s_obj in inputcells.iter_mut() {
                    if let SomeValue::Instance(inst) = s_obj {
                        let has_flag = inst.flags.get("access_directly").copied().unwrap_or(false);
                        if has_flag {
                            if jit_look_inside {
                                access_directly = true;
                                break;
                            } else {
                                inst.flags.remove("access_directly");
                            }
                        }
                    }
                }
                if access_directly {
                    // upstream `key = (AccessDirect, key)` — tuple key.
                    // Rust encodes as prefix on the cache-name string.
                    self.cachedgraph(
                        "access_directly",
                        Some(&format!("{}_access_directly", self.name)),
                    )
                } else {
                    self.cachedgraph("", None)
                }
            }
            Specializer::Arg { parms } => {
                let mut parts = Vec::with_capacity(parms.len());
                for parm in parms {
                    let idx: usize = parm.parse().map_err(|_| {
                        AnnotatorError::new(format!(
                            "specialize:arg expects integer indices, got {parm:?}"
                        ))
                    })?;
                    let s = inputcells.get(idx).ok_or_else(|| {
                        AnnotatorError::new(format!(
                            "specialize:arg index {idx} out of range for {}",
                            self.name
                        ))
                    })?;
                    if !s.is_constant() {
                        return Err(AnnotatorError::new(format!(
                            "specialize:arg({idx}): argument not constant: {s:?}"
                        )));
                    }
                    parts.push(format!("arg{idx}_{s:?}"));
                }
                let key = Self::key_to_cache_name(&parts.join("__"));
                self.cachedgraph(&key, Some(&format!("{}_{}", self.name, key)))
            }
            Specializer::ArgOrVar { parms } => {
                let mut all_constant = true;
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
                        all_constant = false;
                        break;
                    }
                }
                if all_constant {
                    let key = Self::key_to_cache_name(&format!("arg_or_var_{parms:?}"));
                    self.cachedgraph(&key, Some(&format!("{}_{}", self.name, key)))
                } else {
                    self.cachedgraph("", None)
                }
            }
            Specializer::Argtype { parms } => {
                let mut parts = Vec::with_capacity(parms.len());
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
                    parts.push(format!("{:?}", s.knowntype()));
                }
                let key = Self::key_to_cache_name(&parts.join("__"));
                self.cachedgraph(&key, Some(&format!("{}_{}", self.name, key)))
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
                        format!("{:?}", list.listdef.s_value().knowntype())
                    }
                    _ => "none".to_string(),
                };
                let key = Self::key_to_cache_name(&key);
                self.cachedgraph(&key, Some(&format!("{}_{}", self.name, key)))
            }
            Specializer::CallLocation => {
                // upstream `specialize_call_location(funcdesc,
                // args_s, op)` (specialize.py:368-370) — key is
                // `(op,)`, one specialisation per call site. When the
                // caller hasn't threaded the position (annrpython
                // driver not yet wired), fall back to a shared
                // `"call_location"` bucket — over-merges but is
                // conservative (no wrong specialisation).
                let key = match op_key {
                    Some(pk) => format!(
                        "call_location_g{:x}_b{:x}_i{}",
                        pk.graph_id, pk.block_id, pk.op_index
                    ),
                    None => "call_location".to_string(),
                };
                self.cachedgraph(&key, None)
            }
            Specializer::Memo => Err(AnnotatorError::new(
                "FunctionDesc.specialize: memo specializer still requires \
                 rpython/annotator/specialize.py MemoTable support",
            )),
            Specializer::Ll { .. } | Specializer::LlAndArg { .. } => Err(AnnotatorError::new(
                "FunctionDesc.specialize: low-level specializers require annlowlevel.py",
            )),
        }
    }

    /// RPython `FunctionDesc.pycall(whence, args, s_previous_result, op=None)`
    /// (description.py:283-305).
    ///
    /// **Not ported.** Requires annrpython.py (`recursivecall`,
    /// `addpendingblock`) + specialize.py.
    pub fn pycall(
        &self,
        _args: &ArgumentsForTranslation,
        _s_previous_result: &SomeValue,
    ) -> Result<SomeValue, AnnotatorError> {
        Err(AnnotatorError::new(
            "FunctionDesc.pycall requires annrpython.py + specialize.py \
             (Phase 5 P5.2+ dep — see description.py:283-305)",
        ))
    }

    /// RPython `FunctionDesc.get_graph(args, op)` (description.py:328-330).
    pub fn get_graph(
        &self,
        args: &ArgumentsForTranslation,
        op_key: Option<PositionKey>,
    ) -> Result<Rc<PyGraph>, AnnotatorError> {
        let mut inputs_s = self.parse_arguments(args, None, None)?;
        self.specialize(&mut inputs_s, op_key)
    }

    /// RPython `FunctionDesc.bind_under(classdef, name)`
    /// (description.py:350-355).
    ///
    /// **Not ported.** Requires `bookkeeper.getmethoddesc` +
    /// `MethodDesc` which land in description.py commit 3 and
    /// classdesc.py.
    pub fn bind_under(
        &self,
        _classdef_name: &str,
        _name: &str,
    ) -> Result<Rc<RefCell<Desc>>, AnnotatorError> {
        Err(AnnotatorError::new(
            "FunctionDesc.bind_under requires bookkeeper.getmethoddesc + MethodDesc \
             (description.py commit 3+ dep)",
        ))
    }

    /// RPython `FunctionDesc.consider_call_site(descs, args, s_result, op)`
    /// (description.py:357-363).
    ///
    pub fn consider_call_site(
        descs: &[Rc<RefCell<FunctionDesc>>],
        args: &ArgumentsForTranslation,
        _s_result: &SomeValue,
    ) -> Result<(), AnnotatorError> {
        if descs.is_empty() {
            return Ok(());
        }
        let family = descs[0].borrow().base.getcallfamily()?;
        let shape = args.rawshape();
        let mut row = CallTableRow::new();
        for desc in descs {
            // op_key threading = annrpython.py dep (Phase 5 P5.2+).
            // CallLocation specialisation falls back to shared key.
            let graph = desc.borrow().get_graph(args, None)?;
            row.insert(FunctionDesc::rowkey(desc), graph);
        }
        family.borrow_mut().calltable_add_row(shape, row);
        let borrowed: Vec<_> = descs.iter().skip(1).map(|d| d.borrow()).collect();
        let others: Vec<&Desc> = borrowed.iter().map(|d| &d.base).collect();
        descs[0].borrow().base.mergecallfamilies(&others)?;
        Ok(())
    }

    /// RPython `FunctionDesc.get_s_signatures(shape)`
    /// (description.py:368-393).
    ///
    /// **Not ported.** Requires `getcallfamily` + annrpython
    /// `binding(v)` accessor.
    pub fn get_s_signatures(
        &self,
        _shape: &CallShape,
    ) -> Result<Vec<(Vec<SomeValue>, SomeValue)>, AnnotatorError> {
        Err(AnnotatorError::new(
            "FunctionDesc.get_s_signatures requires annrpython.py + \
             Bookkeeper.pbc_maximal_call_families (Phase 5 P5.2+ dep — \
             see description.py:368-393)",
        ))
    }
}

/// RPython `class MemoDesc(FunctionDesc)` (description.py:395-404).
///
/// Overrides `pycall` to return the specializer result directly
/// instead of wrapping it in a graph call. **Deferred** until
/// `FunctionDesc.specialize` + `FunctionDesc.pycall` land.
#[derive(Debug)]
pub struct MemoDesc {
    pub base: FunctionDesc,
}

impl MemoDesc {
    pub fn new(base: FunctionDesc) -> Self {
        MemoDesc { base }
    }
}

// ---------------------------------------------------------------------------
// MethodDesc + FrozenDesc + MethodOfFrozenDesc (description.py:407-637).
// ---------------------------------------------------------------------------

/// Opaque handle for `ClassDef` references used by
/// [`MethodDesc::originclassdef`] / `selfclassdef`. Upstream carries
/// a live `ClassDef` object; Rust keeps a pointer-identity handle
/// until `classdesc.py` lands the real registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    /// **Not ported.** Requires `SomeInstance(selfclassdef,
    /// flags=flags)` construction with a live ClassDef — blocked on
    /// `classdesc.py`. The Rust port currently returns AnnotatorError
    /// so callers hitting bound-method calls surface the missing
    /// dep cleanly.
    pub fn func_args(
        &self,
        args: &ArgumentsForTranslation,
    ) -> Result<ArgumentsForTranslation, AnnotatorError> {
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
        let s_instance =
            SomeValue::Instance(SomeInstance::new(Some(classdef), false, self.flags.clone()));
        Ok(args.prepend(s_instance))
    }

    /// RPython `MethodDesc.pycall(whence, args, s_previous_result, op)`
    /// (description.py:439-441).
    ///
    /// Delegates to `funcdesc.pycall(func_args(args), ...)`; both
    /// dependencies are deferred to later commits.
    pub fn pycall(
        &self,
        args: &ArgumentsForTranslation,
        s_previous_result: &SomeValue,
    ) -> Result<SomeValue, AnnotatorError> {
        let func_args = self.func_args(args)?;
        self.funcdesc.borrow().pycall(&func_args, s_previous_result)
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
    /// Upstream emits a warning then re-binds via `funcdesc.bind_under`.
    /// The Rust port just delegates — warning channel lands with the
    /// annrpython.py driver.
    pub fn bind_under(
        &self,
        classdef: &str,
        name: &str,
    ) -> Result<Rc<RefCell<Desc>>, AnnotatorError> {
        self.funcdesc.borrow().bind_under(classdef, name)
    }

    /// RPython `MethodDesc.bind_self(newselfclassdef, flags={})`
    /// (description.py:451-456).
    ///
    /// **Not ported.** Requires `bookkeeper.getmethoddesc` — Phase
    /// 5 P5.2 dep on description.py commit 4 / classdesc.py.
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
    ) -> Result<(), AnnotatorError> {
        if descs.is_empty() {
            return Ok(());
        }
        let family = descs[0].borrow().base.getcallfamily()?;
        let mut shape = args.rawshape();
        shape.shape_cnt += 1;
        let mut row = CallTableRow::new();
        for desc in descs {
            // upstream: `build_calltable_row(descs, args, op)` iterates
            // `desc.get_graph(args, None)` per-desc; MethodDesc.get_graph
            // delegates to funcdesc.get_graph after func_args prepends
            // `self`. `rowkey` returns funcdesc.identity so the family
            // is keyed by the underlying FunctionDesc (description.py:
            // 467-471).
            let graph = desc.borrow().get_graph(args, None)?;
            row.insert(DescKey::from_rc(&desc.borrow().funcdesc), graph);
        }
        family.borrow_mut().calltable_add_row(shape, row);
        let borrowed: Vec<_> = descs.iter().skip(1).map(|d| d.borrow()).collect();
        let others: Vec<&Desc> = borrowed.iter().map(|d| &d.base).collect();
        descs[0].borrow().base.mergecallfamilies(&others)?;
        Ok(())
    }

    // `MethodDesc.simplify_desc_set` (description.py:473-519) — the
    // selfclassdef / flags collapse — is deferred until classdesc.py
    // provides `ClassDef.issubclass` and `bookkeeper.getmethoddesc`.
}

/// RPython helper `new_or_old_class(c)` (description.py:522-526).
///
/// Upstream returns `c.__class__` when the instance carries one, and
/// `type(c)` otherwise. The Rust port only has enough HostObject
/// plumbing to cover the instance-with-class path via
/// [`HostObject::instance_class`]; `type(c)` for modules / opaque /
/// builtins resolves to a generic `ModuleType` / `type` that our
/// HostObject model does not yet carry, so those cases still return
/// `None`. When classdesc.py / full HostObject getattr land, the
/// fallback becomes a real `type(pyobj)` lookup.
pub fn new_or_old_class(pyobj: &HostObject) -> Option<HostObject> {
    pyobj.instance_class().cloned()
}

/// Walk the MRO of `cls` and return the first class-dict entry for
/// `attr`. Mirror of Python's `getattr(cls, attr)` when `cls` has no
/// per-object shadow: check `cls.__dict__[attr]`, then walk base
/// classes via `cls.__mro__`.
fn class_mro_get(cls: &HostObject, attr: &str) -> Option<ConstValue> {
    if let Some(v) = cls.class_get(attr) {
        return Some(v);
    }
    let mro = cls.mro()?;
    for base in mro.iter().skip(1) {
        if let Some(v) = base.class_get(attr) {
            return Some(v);
        }
    }
    None
}

/// RPython `read_attribute` callback type for [`FrozenDesc`]
/// (description.py:532 `lambda attr: getattr(pyobj, attr)`). The
/// closure returns `Some(ConstValue)` for a present attribute and
/// `None` to signal upstream's `AttributeError`. Kept as a boxed `Fn`
/// so subclasses / test harnesses can wire in custom attribute
/// sources the way upstream allows the caller to pass
/// `read_attribute=...` into `FrozenDesc.__init__`.
pub type FrozenReadAttr = Box<dyn Fn(&str) -> Option<ConstValue>>;

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
    /// [`Self::default_read_attribute`] — a module-aware reader that
    /// falls through to `None` (= AttributeError) for non-module
    /// HostObjects; once classdesc.py lands, full host-level getattr
    /// replaces the default.
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
    /// Python's `getattr(obj, attr)` does a type-dispatched lookup:
    ///
    /// * On **modules** — `obj.__dict__[attr]`.
    /// * On **classes** — `obj.__dict__[attr]`, else walk `__mro__` of
    ///   `obj` (class-of-class).
    /// * On **instances** — `obj.__dict__[attr]`, else walk the MRO of
    ///   `obj.__class__`.
    ///
    /// The Rust port covers all three by reflecting through the
    /// available `HostObject` surfaces. Instance `__dict__` isn't
    /// modelled yet, so instance-level frozen attributes fall through
    /// to the class-hierarchy lookup — matching upstream's behaviour
    /// for PBCs that expose immutable class-level attributes.
    pub fn default_read_attribute(pyobj: HostObject) -> FrozenReadAttr {
        Box::new(move |attr: &str| -> Option<ConstValue> {
            if pyobj.is_module() {
                pyobj.module_get(attr).map(ConstValue::HostObject)
            } else if pyobj.is_class() {
                // upstream `getattr(cls, attr)` — walk MRO starting at
                // self.
                class_mro_get(&pyobj, attr)
            } else if pyobj.is_instance() {
                // upstream `getattr(instance, attr)` — Rust port can
                // only resolve class-hierarchy attributes since
                // HostObject::Instance has no per-instance dict.
                pyobj
                    .instance_class()
                    .and_then(|cls| class_mro_get(cls, attr))
            } else {
                None
            }
        })
    }

    /// RPython `FrozenDesc.has_attribute(attr)` (description.py:539-546).
    pub fn has_attribute(&self, attr: &str) -> bool {
        if self.attrcache.borrow().contains_key(attr) {
            return true;
        }
        self.read_attribute_raw(attr).is_some()
    }

    /// RPython `FrozenDesc.warn_missing_attribute(attr)` (description.py:548-551).
    pub fn warn_missing_attribute(&self, attr: &str) -> bool {
        !self.has_attribute(attr) && !attr.starts_with('$')
    }

    /// Internal helper — reads an attribute through the stored
    /// `_read_attribute` closure. `None` means "attribute missing"
    /// (upstream's `AttributeError`). When the caller supplied a
    /// custom callback, this dispatches to it, matching
    /// `FrozenDesc._read_attribute(attr)` in description.py:543 /
    /// 557 / 570.
    fn read_attribute_raw(&self, attr: &str) -> Option<ConstValue> {
        (self._read_attribute)(attr)
    }

    /// RPython `FrozenDesc.read_attribute(attr)` (description.py:553-558).
    pub fn read_attribute(&self, attr: &str) -> Result<ConstValue, AnnotatorError> {
        if let Some(v) = self.attrcache.borrow().get(attr) {
            return Ok(v.clone());
        }
        match self.read_attribute_raw(attr) {
            Some(v) => {
                self.attrcache
                    .borrow_mut()
                    .insert(attr.to_string(), v.clone());
                Ok(v)
            }
            None => Err(AnnotatorError::new(format!(
                "AttributeError: frozen desc has no attribute {attr:?}"
            ))),
        }
    }

    /// RPython `FrozenDesc.s_read_attribute(attr)` (description.py:560-566).
    pub fn s_read_attribute(&self, attr: &str) -> Result<SomeValue, AnnotatorError> {
        match self.read_attribute(attr) {
            Ok(value) => self.base.bookkeeper.immutablevalue(&value),
            Err(_) => Ok(s_impossible_value()),
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
    pub fn pycall(
        &self,
        args: &ArgumentsForTranslation,
        s_previous_result: &SomeValue,
    ) -> Result<SomeValue, AnnotatorError> {
        let func_args = self.func_args(args)?;
        self.funcdesc.borrow().pycall(&func_args, s_previous_result)
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
    ) -> Result<(), AnnotatorError> {
        if descs.is_empty() {
            return Ok(());
        }
        let family = descs[0].borrow().base.getcallfamily()?;
        let mut shape = args.rawshape();
        shape.shape_cnt += 1;
        let mut row = CallTableRow::new();
        for desc in descs {
            let graph = desc.borrow().get_graph(args, None)?;
            row.insert(DescKey::from_rc(&desc.borrow().funcdesc), graph);
        }
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

    // ---- Desc + FunctionDesc (commit 2) ----

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    fn int_sig(names: &[&str]) -> Signature {
        Signature::new(names.iter().map(|s| s.to_string()).collect(), None, None)
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

    /// Build a PyGraph with a given signature/defaults for the
    /// getuniquegraph tests. Full `PyGraph::new` requires a HostCode;
    /// we construct the same shape manually so the test stays scoped
    /// to the FunctionDesc / PyGraph field comparison upstream runs.
    fn make_pygraph(sig: Signature, defaults: Vec<Constant>) -> Rc<PyGraph> {
        use crate::flowspace::model::{FunctionGraph, GraphFunc};
        let empty_globals = Constant::new(ConstValue::Dict(Default::default()));
        let func = GraphFunc::new("f", empty_globals);
        // Synthetic startblock — getuniquegraph doesn't traverse it.
        let startblock = crate::flowspace::model::Block::shared(vec![]);
        let graph = FunctionGraph::new("f", startblock);
        Rc::new(PyGraph {
            graph,
            func,
            signature: sig,
            defaults,
        })
    }

    #[test]
    fn function_desc_getuniquegraph_returns_single_cached_graph() {
        // upstream description.py:218-226 — a cache of exactly one
        // graph whose signature/defaults match returns that graph.
        let sig = int_sig(&["x"]);
        let fd = FunctionDesc::new(bk(), None, "f", sig.clone(), None, None);
        let graph = make_pygraph(sig, Vec::new());
        fd.cache.borrow_mut().insert("key".into(), graph.clone());
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
        fd.cache
            .borrow_mut()
            .insert("key".into(), make_pygraph(diverging_sig, Vec::new()));
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
            "key".into(),
            make_pygraph(sig, vec![Constant::new(ConstValue::Int(10))]),
        );
        let err = fd.getuniquegraph().unwrap_err();
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("signature/defaults mismatch")
        );
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
        let inputs = fd.parse_arguments(&args, None, None).unwrap();
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
        let err = fd.parse_arguments(&args, None, None).unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("signature mismatch"));
    }

    #[test]
    fn function_desc_buildgraph_surfaces_flow_error_from_objspace() {
        use crate::flowspace::bytecode::HostCode;
        use crate::flowspace::objspace::CO_NEWLOCALS;

        let mut func = crate::flowspace::model::GraphFunc::new(
            "f",
            Constant::new(ConstValue::Dict(Default::default())),
        );
        func.code = Some(Box::new(HostCode {
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
        let fd = FunctionDesc::new(bk(), Some(host), "f", int_sig(&[]), None, None);
        let err = fd.buildgraph(None).unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("BytecodeCorruption"));
    }

    #[test]
    fn function_desc_pycall_deferred() {
        let fd = FunctionDesc::new(bk(), None, "f", int_sig(&[]), None, None);
        let args = ArgumentsForTranslation::new(vec![], None, None);
        let err = fd.pycall(&args, &SomeValue::Impossible).unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("annrpython.py"));
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
    fn frozen_desc_new_accepts_module_host_object() {
        let bk = bk();
        let pyobj = HostObject::new_module("os");
        let fd = FrozenDesc::new(bk, pyobj).unwrap();
        assert!(fd.attrcache.borrow().is_empty());
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
        assert!(fd.has_attribute("x"));
    }

    #[test]
    fn frozen_desc_missing_attribute_errors() {
        let bk = bk();
        let pyobj = HostObject::new_module("os");
        let fd = FrozenDesc::new(bk, pyobj).unwrap();
        assert!(!fd.has_attribute("missing"));
        let err = fd.read_attribute("missing").unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("AttributeError"));
    }

    #[test]
    fn frozen_desc_warn_missing_skips_dollar_prefix() {
        let bk = bk();
        let pyobj = HostObject::new_module("os");
        let fd = FrozenDesc::new(bk, pyobj).unwrap();
        assert!(fd.warn_missing_attribute("x"));
        assert!(!fd.warn_missing_attribute("$memofield_x"));
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
            Box::new(move |attr: &str| -> Option<ConstValue> {
                calls.set(calls.get() + 1);
                match attr {
                    "x" => Some(ConstValue::Int(42)),
                    _ => None,
                }
            })
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
    fn frozen_desc_default_reader_routes_instance_through_class() {
        // upstream default reader: `getattr(instance, name)` falls
        // through to the class hierarchy when the instance has no
        // per-object dict. Rust HostObject::Instance models no per-
        // instance dict, so the class-hierarchy lookup is the
        // authoritative path.
        let bk = bk();
        let cls = HostObject::new_class("Cls", vec![]);
        cls.class_set("TAG", ConstValue::Str("tag".into()));
        let instance = HostObject::new_instance(cls, vec![]);
        let fd = FrozenDesc::new(bk, instance).unwrap();
        let v = fd.read_attribute("TAG").unwrap();
        assert!(matches!(v, ConstValue::Str(ref s) if s == "tag"));
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
}
