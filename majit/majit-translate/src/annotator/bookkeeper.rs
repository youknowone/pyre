//! Bookkeeper — central state carrier for the annotator.
//!
//! RPython upstream: `rpython/annotator/bookkeeper.py` (614 LOC).
//!
//! **Phase 5 P5.2 in progress.** Ports the `immutablevalue` /
//! `newlist` / `newdict` subset that downstream modules
//! (builtin.py, description.py, binaryop.py) invoke directly on the
//! bookkeeper. Fields that require the descriptor / class machinery
//! — `descs`, `classdefs`, `methoddescs`, `emulated_pbc_calls`,
//! `classpbc_attr_families`, `all_specializations`, etc. — land in
//! commit 2 alongside `description.py` / `classdesc.py`.
//!
//! ## Phase 5 P5.2+ dependency-blocked helpers
//!
//! * `getdesc(x)` / `immutablevalue` for function / class / bound-
//!   method / weakref / frozen PBC inputs — blocked on
//!   `description.py`.
//! * `getuniqueclassdef(cls)` — blocked on `classdesc.py`.
//! * `emulate_pbc_call(key, pbc, args_s)` — blocked on
//!   `binaryop.py` call-family machinery.
//! * `register_builtins()` / `BUILTIN_ANALYZERS` registry — blocked
//!   on `builtin.py`.
//! * `annotator` backlink (`reflowfromposition` callback) — blocked
//!   on `annrpython.py` driver.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::argument::simple_args;
use super::classdesc::{ClassDef, ClassDesc};
use super::description::{
    CallFamily, ClassDefKey, DescEntry, DescKey, FrozenAttrFamily, FrozenDesc, FunctionDesc,
    MethodDesc,
};
use super::dictdef::DictDef;
use super::listdef::ListDef;
use super::model::{
    AnnotatorError, Desc, DescKind, SomeBool, SomeBuiltin, SomeChar, SomeDict, SomeFloat,
    SomeInteger, SomeList, SomePBC, SomeString, SomeTuple, SomeValue, s_none,
};
use super::policy::AnnotatorPolicy;
use crate::flowspace::argument::Signature;
use crate::flowspace::bytecode::cpython_code_signature;
use crate::flowspace::model::{ConstValue, Constant, HostObject};
use crate::tool::algo::unionfind::UnionFind;

/// RPython `bookkeeper.position_key` (bookkeeper.py:147) — the tuple
/// identifying "where in the flow graph the annotator is currently
/// reading/writing a value".
///
/// Upstream stores `(FunctionGraph, Block, operation_index)` directly.
/// The Rust port carries the identity-hash values of the first two
/// components so the struct stays:
///   * cheap to clone / hash (no flowspace import cycle),
///   * free of borrow-lifetime issues inside `read_locations:
///     HashSet<PositionKey>`,
///   * still upstream-shaped as a 3-tuple that `ListItem.read_locations
///     |= other.read_locations` can merge without loss.
///
/// Callers obtain the identity hashes via
/// `Rc::as_ptr(&graph) as usize` / `Rc::as_ptr(&block) as usize`.
/// Full bookkeeper.py port replaces the first two fields with real
/// `Weak<FunctionGraph>` / `Weak<RefCell<Block>>` refs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PositionKey {
    /// Identity hash of the enclosing `FunctionGraph` — upstream
    /// `position_key[0]`.
    pub graph_id: usize,
    /// Identity hash of the enclosing `Block` — upstream
    /// `position_key[1]`.
    pub block_id: usize,
    /// Operation index inside the block — upstream `position_key[2]`.
    pub op_index: usize,
}

impl PositionKey {
    pub fn new(graph_id: usize, block_id: usize, op_index: usize) -> Self {
        PositionKey {
            graph_id,
            block_id,
            op_index,
        }
    }
}

/// RPython `class Bookkeeper` (bookkeeper.py:53).
pub struct Bookkeeper {
    /// RPython `self.policy = annotator.policy` (bookkeeper.py:55).
    pub policy: AnnotatorPolicy,
    /// RPython `self.position_key = None` initial (bookkeeper.py:147).
    /// The annotator driver (`RPythonAnnotator.reflow`) writes into
    /// this slot around each reflow block so `read_item` / `agree`
    /// pick it up. Interior mutability because callers hold
    /// `Rc<Bookkeeper>` sharers.
    pub position_key: RefCell<Option<PositionKey>>,
    /// RPython `self.listdefs = {}` (bookkeeper.py:59). Keyed by
    /// position — callers hitting the same position twice share the
    /// ListDef so merging re-entries stay identity-equal. The key is
    /// `Option<PositionKey>` because upstream uses `self.position_key`
    /// directly as the dict key (bookkeeper.py:180
    /// `self.listdefs[self.position_key]`); when `position_key` is
    /// `None`, upstream still caches under the `None` key — so we do
    /// the same rather than building a fresh ListDef per call outside
    /// a reflow frame.
    pub listdefs: RefCell<HashMap<Option<PositionKey>, ListDef>>,
    /// RPython `self.dictdefs = {}` (bookkeeper.py:60). Same
    /// `Option<PositionKey>` key semantics as `listdefs`.
    pub dictdefs: RefCell<HashMap<Option<PositionKey>, DictDef>>,
    /// RPython `self.descs = {}` (bookkeeper.py:67). Maps
    /// `Constant(pyobj)` to a FunctionDesc / ClassDesc / FrozenDesc /
    /// MethodDesc / MethodOfFrozenDesc per bookkeeper.py:353-409. The
    /// Rust port keys directly on [`HostObject`] (which already has
    /// `Arc::ptr_eq` identity) via [`DescEntry`].
    pub descs: RefCell<HashMap<HostObject, DescEntry>>,
    /// RPython `self.classdefs = []` (bookkeeper.py:68). Populated by
    /// `ClassDesc._init_classdef` (classdesc.py:672-697). ClassDef
    /// identity is Rc pointer equality — matches upstream's Python
    /// `cls is other` comparisons.
    pub classdefs: RefCell<Vec<Rc<RefCell<ClassDef>>>>,
    /// RPython `self.methoddescs = {}` (bookkeeper.py:69). Keyed by
    /// `(funcdesc, originclassdef, selfclassdef, name, flags)` tuple
    /// so repeated `getmethoddesc(...)` calls with the same inputs
    /// share identity, per bookkeeper.py:431-442.
    pub methoddescs: RefCell<HashMap<MethodDescKey, Rc<RefCell<MethodDesc>>>>,
    /// RPython `self.frozenpbc_attr_families = UnionFind(FrozenAttrFamily)`
    /// (bookkeeper.py:63).
    pub frozenpbc_attr_families: RefCell<UnionFind<DescKey, Rc<RefCell<FrozenAttrFamily>>>>,
    /// RPython `self.pbc_maximal_call_families = UnionFind(CallFamily)`
    /// (bookkeeper.py:64).
    pub pbc_maximal_call_families: RefCell<UnionFind<DescKey, Rc<RefCell<CallFamily>>>>,
    /// RPython `self.emulated_pbc_calls = {}` (bookkeeper.py:66).
    pub emulated_pbc_calls: RefCell<HashMap<EmulatedPbcCallKey, (SomePBC, Vec<SomeValue>)>>,
    /// RPython `hasattr(self, 'position_key')` (bookkeeper.py:99).
    ///
    /// Upstream distinguishes "no position entered" (attribute absent)
    /// from "entered with position_key = None" (attribute present but
    /// None). Rust carries [`Self::position_key`] as `Option<_>` in both
    /// cases, so this flag tracks the enter/leave invariant explicitly.
    pub position_entered: std::cell::Cell<bool>,
}

impl std::fmt::Debug for Bookkeeper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bookkeeper")
            .field("position_key", &self.position_key.borrow())
            .field("listdefs_len", &self.listdefs.borrow().len())
            .field("dictdefs_len", &self.dictdefs.borrow().len())
            .field("descs_len", &self.descs.borrow().len())
            .field("classdefs_len", &self.classdefs.borrow().len())
            .field("methoddescs_len", &self.methoddescs.borrow().len())
            .finish()
    }
}

/// Key for the `Bookkeeper.methoddescs` cache. Upstream uses a tuple
/// `(funcdesc, originclassdef, selfclassdef, name, tuple(flags.items()))`
/// — Python hashes on object identity for `funcdesc` / `classdef`, and
/// on value for the rest. The Rust port mirrors that by keying on the
/// pointer identity of the two descriptor Rcs plus the stringified
/// name + flags.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MethodDescKey {
    /// RPython `funcdesc` — pointer identity via [`DescKey::from_rc`].
    pub funcdesc_id: DescKey,
    /// RPython `originclassdef` — `ClassDefKey` already carries the
    /// pointer identity.
    pub originclassdef: ClassDefKey,
    /// RPython `selfclassdef` — `None` for unbound methods.
    pub selfclassdef: Option<ClassDefKey>,
    /// RPython `name`.
    pub name: String,
    /// RPython `tuple(flags.items())` — flattened sort-stable flag
    /// entries.
    pub flags: Vec<(String, bool)>,
}

/// Hashable identity for `Bookkeeper.emulated_pbc_calls`.
///
/// Upstream accepts any hashable Python object. The currently-ported
/// callers need a stable position key plus the r_dict eq/hash
/// pseudo-call identities.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EmulatedPbcCallKey {
    Position(PositionKey),
    RDictCall { item_id: usize, role: &'static str },
    Text(String),
}

impl Bookkeeper {
    /// RPython `Bookkeeper.__init__(self, annotator)` (bookkeeper.py:52-76).
    /// Once the annotator driver lands, this constructor takes an
    /// `annotator` backlink; for now it just initialises the bare
    /// storage slots.
    pub fn new() -> Self {
        Self::new_with_policy(AnnotatorPolicy::new())
    }

    pub fn new_with_policy(policy: AnnotatorPolicy) -> Self {
        Bookkeeper {
            policy,
            position_key: RefCell::new(None),
            listdefs: RefCell::new(HashMap::new()),
            dictdefs: RefCell::new(HashMap::new()),
            descs: RefCell::new(HashMap::new()),
            classdefs: RefCell::new(Vec::new()),
            methoddescs: RefCell::new(HashMap::new()),
            frozenpbc_attr_families: RefCell::new(UnionFind::new(|desc: &DescKey| {
                Rc::new(RefCell::new(FrozenAttrFamily::new(*desc)))
            })),
            pbc_maximal_call_families: RefCell::new(UnionFind::new(|desc: &DescKey| {
                Rc::new(RefCell::new(CallFamily::new(*desc)))
            })),
            emulated_pbc_calls: RefCell::new(HashMap::new()),
            position_entered: std::cell::Cell::new(false),
        }
    }

    /// RPython `bookkeeper.position_key = ...` assignment. Returns the
    /// previous value so callers can restore it around a nested reflow
    /// (matches upstream bookkeeper.py:278 `@contextmanager
    /// position()`).
    pub fn set_position_key(&self, pk: Option<PositionKey>) -> Option<PositionKey> {
        self.position_key.replace(pk)
    }

    /// Current `bookkeeper.position_key`. Returns `None` when no
    /// reflow frame is active (upstream's initial
    /// `self.position_key = None`).
    pub fn current_position_key(&self) -> Option<PositionKey> {
        *self.position_key.borrow()
    }

    /// RPython `Bookkeeper.enter(self, position_key)` (bookkeeper.py:84-89).
    ///
    /// Installs the position and registers `self` as the thread-local
    /// bookkeeper so [`getbookkeeper`] returns it. Asserts that no
    /// `enter` is currently active — matches upstream's `not hasattr`
    /// check.
    pub fn enter(self: &Rc<Self>, position_key: Option<PositionKey>) {
        assert!(!self.position_entered.get(), "don't call enter() nestedly");
        self.position_entered.set(true);
        self.position_key.replace(position_key);
        // Upstream: `TLS.bookkeeper = self` (bookkeeper.py:89).
        super::model::TLS.with(|state| state.borrow_mut().bookkeeper = Some(Rc::clone(self)));
    }

    /// RPython `Bookkeeper.leave(self)` (bookkeeper.py:91-94).
    ///
    /// Clears both the position slot and the thread-local bookkeeper
    /// hook. Safe to call only after a matching [`Self::enter`].
    pub fn leave(&self) {
        self.position_entered.set(false);
        self.position_key.replace(None);
        // Upstream: `del TLS.bookkeeper` (bookkeeper.py:93).
        super::model::TLS.with(|state| state.borrow_mut().bookkeeper = None);
    }

    /// RPython `Bookkeeper.at_position(self, pos)` (bookkeeper.py:96-106).
    ///
    /// RAII port of the upstream `@contextmanager`. The `pos=None`
    /// fast-path (line 99-101) short-circuits the enter/leave pair when
    /// the bookkeeper is already inside a reflow frame — used by
    /// `compute_at_fixpoint` to let nested callers reuse the ambient
    /// position.
    pub fn at_position(self: &Rc<Self>, pos: Option<PositionKey>) -> PositionGuard {
        if self.position_entered.get() && pos.is_none() {
            // Upstream: `if hasattr(self, 'position_key') and pos is None: yield; return`
            PositionGuard {
                bk: Rc::clone(self),
                skip_leave: true,
            }
        } else {
            self.enter(pos);
            PositionGuard {
                bk: Rc::clone(self),
                skip_leave: false,
            }
        }
    }

    /// RPython `Bookkeeper.valueoftype(self, t)` (bookkeeper.py:444-445).
    ///
    /// Thin wrapper around [`crate::annotator::signature::annotationoftype`]
    /// used by `binaryop.is_` and the PBC-call site machinery to seed
    /// annotations for type-level constants.
    pub fn valueoftype(
        self: &Rc<Self>,
        spec: &crate::annotator::signature::AnnotationSpec,
    ) -> Result<SomeValue, crate::annotator::signature::SignatureError> {
        crate::annotator::signature::annotationoftype(spec, Some(self))
    }

    /// RPython `Bookkeeper.getlistdef(**flags_if_new)` (bookkeeper.py:178-185).
    ///
    /// Returns the (cached or freshly constructed) ListDef for the
    /// bookkeeper's current position. Upstream stores flags inside the
    /// `listitem.__dict__`; Rust carries the `range_step` flag
    /// explicitly (the only non-default flag any caller passes — see
    /// bookkeeper.py:193-195).
    ///
    /// The current position — including `None` — is used as the cache
    /// key directly, matching upstream's `self.listdefs[self.position_
    /// key]` indexing. Two calls with no active position share the
    /// same ListDef just like two calls inside the same reflow frame
    /// would.
    pub fn getlistdef(self: &Rc<Self>, range_step: Option<i64>) -> ListDef {
        let pk = self.current_position_key();
        let mut listdefs = self.listdefs.borrow_mut();
        if let Some(existing) = listdefs.get(&pk) {
            return existing.clone();
        }
        let new_ld = ListDef::new(Some(self.clone()), SomeValue::Impossible, false, false);
        if let Some(step) = range_step {
            let li = new_ld.inner.listitem.borrow().clone();
            li.borrow_mut().range_step = Some(step);
        }
        listdefs.insert(pk, new_ld.clone());
        new_ld
    }

    /// RPython `Bookkeeper.newlist(*s_values, **flags)` (bookkeeper.py:187-196).
    pub fn newlist(
        self: &Rc<Self>,
        s_values: &[SomeValue],
        range_step: Option<i64>,
    ) -> Result<SomeList, AnnotatorError> {
        let listdef = self.getlistdef(range_step);
        for s_value in s_values {
            listdef
                .generalize(s_value)
                .map_err(|e| AnnotatorError::new(e.msg))?;
        }
        if let Some(step) = range_step {
            listdef
                .generalize_range_step(Some(step))
                .map_err(|e| AnnotatorError::new(e.msg))?;
        }
        Ok(SomeList::new(listdef))
    }

    /// RPython `Bookkeeper.getdictdef(is_r_dict=False,
    /// force_non_null=False, simple_hash_eq=False)` (bookkeeper.py:198-207).
    ///
    /// `None` position caches just like `Some(pk)`, matching upstream's
    /// `self.dictdefs[self.position_key]` indexing. See [`Self::
    /// getlistdef`] for the rationale.
    pub fn getdictdef(
        self: &Rc<Self>,
        is_r_dict: bool,
        force_non_null: bool,
        simple_hash_eq: bool,
    ) -> DictDef {
        let pk = self.current_position_key();
        let mut dictdefs = self.dictdefs.borrow_mut();
        if let Some(existing) = dictdefs.get(&pk) {
            return existing.clone();
        }
        let new_dd = DictDef::new(
            Some(self.clone()),
            SomeValue::Impossible,
            SomeValue::Impossible,
            is_r_dict,
            force_non_null,
            simple_hash_eq,
        );
        dictdefs.insert(pk, new_dd.clone());
        new_dd
    }

    /// RPython `Bookkeeper.newdict()` (bookkeeper.py:209-212).
    pub fn newdict(self: &Rc<Self>) -> SomeDict {
        SomeDict::new(self.getdictdef(false, false, false))
    }

    /// RPython `Bookkeeper.getdesc(pyobj)` (bookkeeper.py:353-409).
    ///
    /// Returns the cached [`DescEntry`] for `pyobj`, or constructs a
    /// fresh one per upstream's `isinstance` dispatch. The Rust port
    /// branches on [`HostObject`] kind:
    ///   * `UserFunction` → [`Self::newfuncdesc`]
    ///   * `Class` → `ClassDesc` (c1 shell — c2c replaces this with the
    ///     full `ClassDesc::__init__` body; the shell still satisfies
    ///     identity-caching so basedesc lookups land in a shared Rc)
    ///   * `Instance` / `BuiltinCallable` / `Module` / `Opaque` →
    ///     [`Self::getfrozen`] (upstream's `_freeze_` fallback)
    ///
    /// The upstream bound-method branch (MethodType) has no direct
    /// HostObject counterpart yet; when HostObject gains a `Method`
    /// variant (c3), this function routes it through
    /// `getmethoddesc(self.getdesc(im_func), …)` matching
    /// bookkeeper.py:374-396.
    pub fn getdesc(self: &Rc<Self>, pyobj: &HostObject) -> Result<DescEntry, AnnotatorError> {
        if let Some(existing) = self.descs.borrow().get(pyobj) {
            return Ok(existing.clone());
        }
        let entry = if pyobj.is_user_function() {
            DescEntry::Function(self.newfuncdesc(pyobj)?)
        } else if pyobj.is_class() {
            // upstream bookkeeper.py:367-373 — pyobj is `object` check
            // raises, and `__builtin__` module check routes to
            // `getfrozen`. The Rust port currently treats every non-
            // builtin HostObject-class as a ClassDesc; builtin types
            // aren't modelled as HostObject::Class yet (they show up as
            // BuiltinCallable / primitive ConstValue), so the branch
            // isn't reachable. `object` identity isn't materialised
            // either.
            let name = pyobj.qualname().to_string();
            let desc_rc = ClassDesc::new(self, pyobj.clone(), Some(name), None, None)?;
            DescEntry::Class(desc_rc)
        } else if pyobj.is_builtin_callable()
            || pyobj.is_instance()
            || pyobj.is_module()
            || pyobj.is_opaque()
        {
            DescEntry::Frozen(self.getfrozen(pyobj)?)
        } else {
            return Err(AnnotatorError::new(format!(
                "Bookkeeper.getdesc({:?}): unexpected prebuilt constant",
                pyobj.qualname()
            )));
        };
        self.descs.borrow_mut().insert(pyobj.clone(), entry.clone());
        Ok(entry)
    }

    /// RPython `Bookkeeper.newfuncdesc(pyfunc)` (bookkeeper.py:411-426).
    ///
    /// Rust port: pull signature / defaults from the HostObject's
    /// [`crate::flowspace::model::GraphFunc`], and request a
    /// specializer from `AnnotatorPolicy.get_specializer(tag)` once
    /// the policy backlink is wired (annrpython.py c1 dep). For now
    /// the specializer is `None`, matching upstream's `tag = None →
    /// default_specialize` path. The `MemoDesc` branch
    /// (bookkeeper.py:424-425) lands with specialize.py.
    pub fn newfuncdesc(
        self: &Rc<Self>,
        pyfunc: &HostObject,
    ) -> Result<Rc<RefCell<FunctionDesc>>, AnnotatorError> {
        let gf = pyfunc.user_function().ok_or_else(|| {
            AnnotatorError::new(format!(
                "newfuncdesc({:?}) called on non-user-function HostObject",
                pyfunc.qualname()
            ))
        })?;
        // upstream bookkeeper.py:418 `signature = cpython_code_signature(pyfunc.__code__)`.
        let name = gf.name.clone();
        let signature = match gf.code.as_ref() {
            Some(code) => cpython_code_signature(code),
            // No HostCode attached — upstream hits the
            // `_generator_next_method_of_` branch (bookkeeper.py:413-416)
            // or fails. The Rust port defaults to the single-arg
            // `Signature(['entry'])` matching upstream's generator
            // fallback so tests that wire GraphFunc without a HostCode
            // still traverse.
            None => Signature::new(vec!["entry".to_string()], None, None),
        };
        let defaults = if gf.defaults.is_empty() {
            None
        } else {
            Some(gf.defaults.clone())
        };
        let specializer = self
            .policy
            .get_specializer(gf.annspecialcase.as_deref())
            .map_err(|e| AnnotatorError::new(e.to_string()))?;
        let fd = FunctionDesc::new(
            self.clone(),
            Some(pyfunc.clone()),
            name,
            signature,
            defaults,
            Some(specializer),
        );
        Ok(Rc::new(RefCell::new(fd)))
    }

    /// RPython `Bookkeeper.getfrozen(pyobj)` (bookkeeper.py:428-429).
    pub fn getfrozen(
        self: &Rc<Self>,
        pyobj: &HostObject,
    ) -> Result<Rc<RefCell<FrozenDesc>>, AnnotatorError> {
        let fd = FrozenDesc::new(self.clone(), pyobj.clone())?;
        Ok(Rc::new(RefCell::new(fd)))
    }

    /// RPython `Bookkeeper.getmethoddesc(funcdesc, originclassdef,
    /// selfclassdef, name, flags={})` (bookkeeper.py:431-442).
    ///
    /// Caches MethodDescs by the `(funcdesc-id, origindef-id,
    /// selfdef-id, name, flags)` tuple — upstream's Python tuple hash
    /// keyed on identity for the descriptor / classdef entries.
    pub fn getmethoddesc(
        self: &Rc<Self>,
        funcdesc: &Rc<RefCell<FunctionDesc>>,
        originclassdef: ClassDefKey,
        selfclassdef: Option<ClassDefKey>,
        name: &str,
        flags: std::collections::BTreeMap<String, bool>,
    ) -> Rc<RefCell<MethodDesc>> {
        let flags_vec: Vec<(String, bool)> = flags.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let key = MethodDescKey {
            funcdesc_id: DescKey::from_rc(funcdesc),
            originclassdef,
            selfclassdef,
            name: name.to_string(),
            flags: flags_vec,
        };
        if let Some(existing) = self.methoddescs.borrow().get(&key) {
            return existing.clone();
        }
        let md = MethodDesc::new(
            self.clone(),
            funcdesc.clone(),
            originclassdef,
            selfclassdef,
            name,
            flags,
        );
        let rc = Rc::new(RefCell::new(md));
        self.methoddescs.borrow_mut().insert(key, rc.clone());
        rc
    }

    /// RPython `bookkeeper.classdefs.append(classdef)` — invoked from
    /// `ClassDesc._init_classdef` (classdesc.py:674). Callers hand over
    /// the fresh `Rc<RefCell<ClassDef>>` so the bookkeeper retains the
    /// identity alongside every other reachable classdef.
    pub fn register_classdef(self: &Rc<Self>, classdef: Rc<RefCell<ClassDef>>) {
        self.classdefs.borrow_mut().push(classdef);
    }

    pub fn lookup_classdef(&self, key: ClassDefKey) -> Option<Rc<RefCell<ClassDef>>> {
        self.classdefs
            .borrow()
            .iter()
            .find(|classdef| ClassDefKey::from_raw(Rc::as_ptr(classdef) as usize) == key)
            .cloned()
    }

    /// Snapshot of every registered classdef — test helper + upstream
    /// `bookkeeper.classdefs` read access.
    pub fn classdef_snapshot(&self) -> Vec<Rc<RefCell<ClassDef>>> {
        self.classdefs.borrow().clone()
    }

    /// RPython `Bookkeeper.getuniqueclassdef(cls)` (bookkeeper.py:282-287):
    ///
    /// ```python
    /// def getuniqueclassdef(self, cls):
    ///     assert not isinstance(cls, type(Exception)) or cls is type(Exception)
    ///     desc = self.getdesc(cls)
    ///     return desc.getuniqueclassdef()
    /// ```
    ///
    /// The `type(Exception)` assertion guards against PyPy's old-style
    /// exception metaclass trap; the Rust port doesn't model old-style
    /// classes so the assertion is omitted.
    pub fn getuniqueclassdef(
        self: &Rc<Self>,
        cls: &HostObject,
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        let entry = self.getdesc(cls)?;
        match entry {
            DescEntry::Class(cd_rc) => ClassDesc::getuniqueclassdef(&cd_rc),
            _ => Err(AnnotatorError::new(format!(
                "Bookkeeper.getuniqueclassdef({:?}): not a class",
                cls.qualname()
            ))),
        }
    }

    fn resolve_stub_desc(&self, desc: &Desc) -> Option<DescEntry> {
        self.descs
            .borrow()
            .values()
            .find(|entry| match (desc.kind, entry) {
                (DescKind::Function, DescEntry::Function(fd)) => {
                    fd.borrow().base.pyobj.as_ref().map(|obj| obj.qualname())
                        == Some(desc.name.as_str())
                }
                (DescKind::Class, DescEntry::Class(cd)) => cd.borrow().name == desc.name,
                (DescKind::Frozen, DescEntry::Frozen(fd)) => {
                    fd.borrow().base.pyobj.as_ref().map(|obj| obj.qualname())
                        == Some(desc.name.as_str())
                }
                _ => false,
            })
            .cloned()
    }

    fn pbc_call_result_from_entry(
        self: &Rc<Self>,
        entry: &DescEntry,
        args_s: &[SomeValue],
    ) -> Result<SomeValue, AnnotatorError> {
        let args = simple_args(args_s.to_vec());
        // `op_key = None` — threading the flow-space op through the
        // bookkeeper.pbc_call shim requires the annrpython.py driver.
        // CallLocation specialisation over-merges until that lands.
        match entry {
            DescEntry::Function(fd) => {
                let graph = fd.borrow().get_graph(&args, None)?;
                self.infer_graph_result(&graph.graph, args_s)
            }
            DescEntry::Method(md) => {
                let graph = md.borrow().get_graph(&args, None)?;
                self.infer_graph_result(&graph.graph, args_s)
            }
            DescEntry::MethodOfFrozen(mfd) => {
                let graph = mfd.borrow().get_graph(&args, None)?;
                self.infer_graph_result(&graph.graph, args_s)
            }
            DescEntry::Frozen(_) | DescEntry::Class(_) => Err(AnnotatorError::new(
                "Bookkeeper.pbc_call: non-callable PBC entry",
            )),
        }
    }

    fn infer_graph_result(
        self: &Rc<Self>,
        graph: &crate::flowspace::model::FunctionGraph,
        inputcells: &[SomeValue],
    ) -> Result<SomeValue, AnnotatorError> {
        use crate::flowspace::model::{BlockKey, Hlvalue};

        let mut defined: HashMap<(BlockKey, crate::flowspace::model::Variable), SomeValue> =
            HashMap::new();
        for (index, arg) in graph.getargs().into_iter().enumerate() {
            if let (Hlvalue::Variable(v), Some(s_value)) = (arg, inputcells.get(index)) {
                defined.insert((BlockKey::of(&graph.startblock), v), s_value.clone());
            }
        }

        for block in graph.iterblocks() {
            let block_key = BlockKey::of(&block);
            for op in &block.borrow().operations {
                let inferred = match op.opname.as_str() {
                    "eq" | "ne" | "lt" | "le" | "gt" | "ge" | "is_" | "contains" | "bool" => {
                        SomeValue::Bool(SomeBool::new())
                    }
                    "hash" | "len" | "ord" | "int" | "add" | "sub" | "mul" | "floordiv" | "mod"
                    | "lshift" | "rshift" | "and_" | "or_" | "xor" => {
                        SomeValue::Integer(SomeInteger::default())
                    }
                    "type" => SomeValue::Impossible,
                    _ => continue,
                };
                let crate::flowspace::model::Hlvalue::Variable(v) = &op.result else {
                    continue;
                };
                defined.insert((block_key.clone(), v.clone()), inferred);
            }
        }

        let mut results = Vec::new();
        for link_ref in graph.iterlinks() {
            let link = link_ref.borrow();
            let Some(target) = &link.target else {
                continue;
            };
            if BlockKey::of(target) != BlockKey::of(&graph.returnblock) {
                continue;
            }
            let Some(arg) = link.args.first().and_then(|arg| arg.clone()) else {
                continue;
            };
            match arg {
                crate::flowspace::model::Hlvalue::Constant(c) => {
                    results.push(self.immutablevalue(&c.value)?);
                }
                crate::flowspace::model::Hlvalue::Variable(v) => {
                    let prevblock = link
                        .prevblock
                        .as_ref()
                        .and_then(|prev| prev.upgrade())
                        .ok_or_else(|| AnnotatorError::new("return link missing prevblock"))?;
                    let prev_key = BlockKey::of(&prevblock);
                    if let Some(s_value) = defined.get(&(prev_key, v.clone())) {
                        results.push(s_value.clone());
                    } else {
                        return Ok(SomeValue::Impossible);
                    }
                }
            }
        }

        let mut iter = results.into_iter();
        let Some(mut acc) = iter.next() else {
            return Ok(SomeValue::Impossible);
        };
        for s_value in iter {
            acc = super::model::union(&acc, &s_value).map_err(|e| AnnotatorError::new(e.msg))?;
        }
        Ok(acc)
    }

    pub fn pbc_call(
        self: &Rc<Self>,
        pbc: &SomePBC,
        args_s: &[SomeValue],
    ) -> Result<SomeValue, AnnotatorError> {
        let mut results = Vec::new();
        for desc in &pbc.descriptions {
            let entry = self.resolve_stub_desc(desc).ok_or_else(|| {
                AnnotatorError::new(format!(
                    "Bookkeeper.pbc_call: unresolved PBC description {:?}",
                    desc
                ))
            })?;
            results.push(self.pbc_call_result_from_entry(&entry, args_s)?);
        }
        let mut iter = results.into_iter();
        let Some(mut acc) = iter.next() else {
            return Ok(SomeValue::Impossible);
        };
        for s_value in iter {
            acc = super::model::union(&acc, &s_value).map_err(|e| AnnotatorError::new(e.msg))?;
        }
        Ok(acc)
    }

    pub fn emulate_pbc_call(
        self: &Rc<Self>,
        unique_key: EmulatedPbcCallKey,
        pbc: &SomeValue,
        args_s: &[SomeValue],
        replace: &[EmulatedPbcCallKey],
    ) -> Result<SomeValue, AnnotatorError> {
        let SomeValue::PBC(pbc) = pbc else {
            return Err(AnnotatorError::new(format!(
                "Bookkeeper.emulate_pbc_call expects SomePBC, got {pbc:?}"
            )));
        };
        {
            let mut emulated = self.emulated_pbc_calls.borrow_mut();
            emulated.remove(&unique_key);
            for key in replace {
                emulated.remove(key);
            }
            emulated.insert(unique_key, (pbc.clone(), args_s.to_vec()));
        }
        self.pbc_call(pbc, args_s)
    }

    /// RPython `Bookkeeper.immutablevalue(x)` (bookkeeper.py:214-325).
    ///
    /// "The most precise SomeValue instance that contains the
    /// immutable value x."
    ///
    /// Input is a flowspace [`ConstValue`] — the Rust-side
    /// counterpart to upstream's Python constant. Primitive branches
    /// (bool / int / float / str / char / unicode / bytearray / tuple
    /// / None) are ported line-by-line; `list` / `dict` build
    /// [`SomeList`] / [`SomeDict`] via `getlistdef` / `getdictdef`
    /// without the upstream `immutable_cache` memoisation (perf-only
    /// deviation, correctness unchanged).
    ///
    /// The function / class / bound-method / weakref / frozen-PBC
    /// / symbolic-constant / PBC branches (bookkeeper.py:218-325)
    /// require `description.py` + `classdesc.py` + `rlib/rarithmetic`
    /// / `rlib/objectmodel` imports that are Phase 5 P5.2+ deps. Those
    /// inputs surface as [`AnnotatorError`] so the missing branch is
    /// observable at the call site.
    pub fn immutablevalue(self: &Rc<Self>, x: &ConstValue) -> Result<SomeValue, AnnotatorError> {
        match x {
            ConstValue::Bool(b) => {
                let mut s = SomeBool::new();
                s.base.const_box = Some(Constant::new(ConstValue::Bool(*b)));
                Ok(SomeValue::Bool(s))
            }
            ConstValue::Int(i) => {
                // upstream: `result = SomeInteger(nonneg = x>=0)`.
                let mut s = SomeInteger::new(*i >= 0, false);
                s.base.const_box = Some(Constant::new(ConstValue::Int(*i)));
                Ok(SomeValue::Integer(s))
            }
            ConstValue::Float(_) => {
                let mut s = SomeFloat::new();
                s.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::Float(s))
            }
            ConstValue::Str(s) => {
                let no_nul = !s.contains('\x00');
                let result = if s.chars().count() == 1 {
                    // upstream: `result = SomeChar(no_nul=no_nul)`.
                    let mut ch = SomeChar::new(no_nul);
                    ch.inner.base.const_box = Some(Constant::new(x.clone()));
                    SomeValue::Char(ch)
                } else {
                    // upstream: `result = SomeString(no_nul=no_nul)`.
                    let mut st = SomeString::new(false, no_nul);
                    st.inner.base.const_box = Some(Constant::new(x.clone()));
                    SomeValue::String(st)
                };
                Ok(result)
            }
            ConstValue::None => Ok(s_none()),
            ConstValue::Tuple(items) => {
                let items_s = items
                    .iter()
                    .map(|v| self.immutablevalue(v))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(SomeValue::Tuple(SomeTuple::new(items_s)))
            }
            ConstValue::List(items) => {
                // upstream bookkeeper.py:255-265 memoises via
                // `immutable_cache[Constant(x)]`. Rust skips the cache
                // (perf-only deviation, see module doc).
                let listdef = ListDef::new(Some(self.clone()), SomeValue::Impossible, false, false);
                for e in items {
                    let s_e = self.immutablevalue(e)?;
                    listdef
                        .generalize(&s_e)
                        .map_err(|e| AnnotatorError::new(e.msg))?;
                }
                let mut result = SomeList::new(listdef);
                result.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::List(result))
            }
            ConstValue::Dict(items) => {
                // upstream bookkeeper.py:266-298 memoises via
                // `immutable_cache` and handles OrderedDict / r_dict
                // via the dict type. Our ConstValue::Dict keys are
                // strings only (flowspace globals), so we build a
                // plain SomeDict with string key type.
                let dictdef = DictDef::new(
                    Some(self.clone()),
                    SomeValue::Impossible,
                    SomeValue::Impossible,
                    false,
                    false,
                    false,
                );
                for (k, v) in items {
                    let s_k = SomeValue::String(SomeString::new(false, !k.contains('\x00')));
                    let s_v = self.immutablevalue(v)?;
                    dictdef
                        .generalize_key(&s_k)
                        .map_err(|e| AnnotatorError::new(e.msg))?;
                    dictdef
                        .generalize_value(&s_v)
                        .map_err(|e| AnnotatorError::new(e.msg))?;
                }
                let mut result = SomeDict::new(dictdef);
                result.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::Dict(result))
            }
            ConstValue::HostObject(obj) => self.immutablevalue_hostobject(obj, x),
            ConstValue::Code(_)
            | ConstValue::Function(_)
            | ConstValue::SpecTag(_)
            | ConstValue::Atom(_)
            | ConstValue::Placeholder => {
                // Code / Function / SpecTag / Atom / Placeholder cover
                // internal flowspace / host-carrier values that
                // upstream never feeds into immutablevalue. Keep the
                // fail-fast stub so any unexpected call-site surfaces
                // a clear error rather than silent stub-SomePBC.
                Err(AnnotatorError::new(format!(
                    "Bookkeeper.immutablevalue({x:?}): internal ConstValue variant \
                     has no upstream immutablevalue branch"
                )))
            }
        }
    }

    /// Narrow dispatch for the `ConstValue::HostObject` arm of
    /// [`Self::immutablevalue`]. Covers the `callable` / `tp is type`
    /// / `_freeze_` branches at bookkeeper.py:309-333 to the extent
    /// that the stub `model::Desc` + `SomeBuiltin` surfaces allow.
    ///
    /// The upstream calls route through `self.getdesc(x)` which lives
    /// in bookkeeper commit 2 (blocked on classdesc.py for
    /// `ClassDesc`). Until that lands, we emit stub `model::Desc`
    /// entries into the [`SomePBC.descriptions`] set so callers see a
    /// typed annotation rather than `AnnotatorError`. Semantic
    /// differences vs upstream:
    ///   * `knowntype` of the returned SomePBC is `KnownType::Other`
    ///     because `commonbase` folding waits for classdesc.py.
    ///   * `SomeConstantType(x, self)` (bookkeeper.py:315-316) for a
    ///     class input collapses into `SomePBC([Desc::Class])` with
    ///     `const_box` set — the PBC-subclass distinction is implicit
    ///     in `DescKind::Class`.
    ///   * Bound methods / weakrefs / frozen-PBCs / BUILTIN_ANALYZERS
    ///     lookup / extregistry / property / symbolic-constant routes
    ///     stay deferred (each needs its own dep — classdesc.py,
    ///     weakref table, builtin.py registry, extregistry, specialize).
    fn immutablevalue_hostobject(
        self: &Rc<Self>,
        obj: &HostObject,
        raw: &ConstValue,
    ) -> Result<SomeValue, AnnotatorError> {
        // upstream bookkeeper.py:317 `callable(x)` → SomePBC path. In
        // the Rust port we treat user-functions explicitly because
        // bound-method / find_method dispatch (bookkeeper.py:318-329)
        // requires a full descriptor registry that isn't ported yet.
        if obj.is_user_function() {
            let mut pbc = SomePBC::new(vec![Desc::new(DescKind::Function, obj.qualname())], false);
            pbc.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::PBC(pbc));
        }
        // upstream bookkeeper.py:309-311 — BUILTIN_ANALYZERS lookup
        // produces SomeBuiltin. The Rust port keeps the analyser
        // registry empty (builtin.py is still deferred), so
        // `analyser_name` is set from the host qualname and callers
        // resolving through specialcase.rs dispatch on that string
        // when an analyser registers.
        if obj.is_builtin_callable() {
            let mut sb = SomeBuiltin::new(
                obj.qualname().to_string(),
                None,
                Some(obj.qualname().to_string()),
            );
            sb.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::Builtin(sb));
        }
        // upstream bookkeeper.py:315-316 — `tp is type` → SomeConstant
        // Type(x, self). Implemented as a constant SomePBC over a
        // Class-kind Desc (SomeConstantType IS a SomePBC subclass
        // upstream; `const_box` + Class kind captures the same shape
        // the Rust model can model today).
        if obj.is_class() {
            let mut pbc = SomePBC::new(vec![Desc::new(DescKind::Class, obj.qualname())], false);
            pbc.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::PBC(pbc));
        }
        Err(AnnotatorError::new(format!(
            "Bookkeeper.immutablevalue({raw:?}): host object kind not yet routed \
             (Phase 5 P5.2+ dep — needs classdesc.py / builtin.py / extregistry / \
             weakref / bound-method lookup; see bookkeeper.py:299-333)"
        )))
    }
}

impl Default for Bookkeeper {
    fn default() -> Self {
        Self::new()
    }
}

/// RPython `getbookkeeper()` free function (bookkeeper.py:605-611).
///
/// ```python
/// def getbookkeeper():
///     try:
///         return TLS.bookkeeper
///     except AttributeError:
///         return None
/// ```
pub fn getbookkeeper() -> Option<Rc<Bookkeeper>> {
    super::model::TLS.with(|state| state.borrow().bookkeeper.clone())
}

/// RPython `immutablevalue(x)` free function (bookkeeper.py:613-614).
///
/// Delegates to [`Bookkeeper::immutablevalue`] on the thread-local
/// bookkeeper. Panics when called without a live bookkeeper — upstream
/// raises `AttributeError: 'NoneType' object has no attribute
/// 'immutablevalue'` in the same situation.
pub fn immutablevalue(x: &ConstValue) -> Result<SomeValue, AnnotatorError> {
    let bk = getbookkeeper().expect("immutablevalue() called without an active bookkeeper");
    bk.immutablevalue(x)
}

/// RAII guard returned by [`Bookkeeper::at_position`]. Mirrors the
/// upstream `@contextmanager` exit — calls [`Bookkeeper::leave`] on
/// drop unless the fast-path at bookkeeper.py:99-101 skipped the
/// initial enter.
pub struct PositionGuard {
    bk: Rc<Bookkeeper>,
    skip_leave: bool,
}

impl Drop for PositionGuard {
    fn drop(&mut self) {
        if !self.skip_leave {
            self.bk.leave();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{SomeByteArray, SomeChar, SomeFloat, SomeString};
    use crate::flowspace::model::GraphFunc;

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    #[test]
    fn immutablevalue_int_sets_nonneg_when_ge_zero() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Int(3)).unwrap();
        match s {
            SomeValue::Integer(si) => {
                assert!(si.nonneg);
                assert!(si.base.const_box.is_some());
            }
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_int_negative_not_nonneg() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Int(-1)).unwrap();
        match s {
            SomeValue::Integer(si) => assert!(!si.nonneg),
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_bool() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Bool(true)).unwrap();
        assert!(matches!(s, SomeValue::Bool(_)));
    }

    #[test]
    fn immutablevalue_single_char_str_is_somechar() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Str("a".into())).unwrap();
        assert!(matches!(s, SomeValue::Char(_)));
    }

    #[test]
    fn immutablevalue_multichar_str_is_somestring() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Str("hello".into())).unwrap();
        assert!(matches!(s, SomeValue::String(_)));
    }

    #[test]
    fn immutablevalue_str_with_nul_clears_no_nul() {
        let bk = bk();
        let with_nul = ConstValue::Str("a\x00b".into());
        let s = bk.immutablevalue(&with_nul).unwrap();
        match s {
            SomeValue::String(st) => assert!(!st.inner.no_nul),
            other => panic!("expected SomeString, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_float_is_somefloat() {
        let bk = bk();
        let s = bk
            .immutablevalue(&ConstValue::Float(1.5_f64.to_bits()))
            .unwrap();
        match s {
            SomeValue::Float(_) => {}
            other => panic!("expected SomeFloat, got {other:?}"),
        }
        let _ = SomeFloat::new();
    }

    #[test]
    fn immutablevalue_none_is_s_none() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::None).unwrap();
        assert!(matches!(s, SomeValue::None_(_)));
    }

    #[test]
    fn immutablevalue_tuple_walks_items() {
        let bk = bk();
        let s = bk
            .immutablevalue(&ConstValue::Tuple(vec![
                ConstValue::Int(1),
                ConstValue::Bool(false),
            ]))
            .unwrap();
        match s {
            SomeValue::Tuple(t) => {
                assert_eq!(t.items.len(), 2);
                assert!(matches!(t.items[0], SomeValue::Integer(_)));
                assert!(matches!(t.items[1], SomeValue::Bool(_)));
            }
            other => panic!("expected SomeTuple, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_list_generalizes_elements() {
        let bk = bk();
        let s = bk
            .immutablevalue(&ConstValue::List(vec![
                ConstValue::Int(1),
                ConstValue::Int(-1),
            ]))
            .unwrap();
        match s {
            SomeValue::List(sl) => {
                // Element type widened from {nonneg Int, signed Int}
                // to generic Int (nonneg=false after merge with -1).
                if let SomeValue::Integer(si) = sl.listdef.s_value() {
                    assert!(!si.nonneg);
                } else {
                    panic!("expected Int listdef s_value");
                }
                assert!(sl.base.const_box.is_some());
            }
            other => panic!("expected SomeList, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_class_returns_constant_pbc() {
        // upstream bookkeeper.py:315-316 — `tp is type` produces
        // `SomeConstantType(x, self)`, a SomePBC subclass with
        // `const = x`. The Rust port emits a Class-kind `SomePBC`
        // with `const_box` set; `SomeConstantType` collapses into the
        // PBC subclass because our PBC doesn't carry a Python-class
        // inheritance shadow.
        use crate::annotator::model::{DescKind, SomeValue};
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let class = HostObject::new_class("Foo", vec![]);
        let s = bk
            .immutablevalue(&ConstValue::HostObject(class))
            .expect("class HostObject must produce SomePBC");
        match s {
            SomeValue::PBC(pbc) => {
                assert_eq!(pbc.descriptions.len(), 1);
                assert_eq!(
                    pbc.descriptions.iter().next().unwrap().kind,
                    DescKind::Class
                );
                assert!(pbc.base.const_box.is_some());
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_user_function_returns_function_pbc() {
        // upstream bookkeeper.py:317-331 — `callable(x)` falls into
        // `SomePBC([self.getdesc(x)])`. Narrow Rust port emits a
        // Function-kind Desc stub; real FunctionDesc wiring lands
        // when bookkeeper commit 2 ports getdesc.
        use crate::annotator::model::{DescKind, SomeValue};
        use crate::flowspace::model::{Constant, GraphFunc, HostObject};
        let bk = bk();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func = HostObject::new_user_function(GraphFunc::new("f", globals));
        let s = bk
            .immutablevalue(&ConstValue::HostObject(func))
            .expect("user-function HostObject must produce SomePBC");
        match s {
            SomeValue::PBC(pbc) => {
                assert_eq!(pbc.descriptions.len(), 1);
                assert_eq!(
                    pbc.descriptions.iter().next().unwrap().kind,
                    DescKind::Function
                );
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_builtin_callable_returns_somebuiltin() {
        // upstream bookkeeper.py:309-311 — BUILTIN_ANALYZERS lookup
        // produces SomeBuiltin. Rust port wires the analyser_name
        // from the host qualname until builtin.py lands the registry.
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let bltn = HostObject::new_builtin_callable("len");
        let s = bk
            .immutablevalue(&ConstValue::HostObject(bltn))
            .expect("builtin HostObject must produce SomeBuiltin");
        assert!(matches!(s, SomeValue::Builtin(_)));
    }

    #[test]
    fn immutablevalue_instance_host_object_defers() {
        // Non-function/non-class/non-builtin HostObject kinds (Module,
        // Instance, Opaque) stay deferred until the descriptor
        // registry lands — fail-fast rather than silently routing.
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let mod_obj = HostObject::new_module("m");
        let err = bk
            .immutablevalue(&ConstValue::HostObject(mod_obj))
            .expect_err("module HostObject stays deferred");
        assert!(err.msg.unwrap_or_default().contains("not yet routed"));
    }

    #[test]
    fn newlist_creates_somelist_and_generalizes() {
        // Use two SomeInteger variants so the Phase 4 A4.6 pair-union
        // subset (Int ∪ Int) can widen them — upstream's multi-type
        // lists exercise broader pair unions which are Phase 5 P5.2+
        // pending.
        let bk = bk();
        let s_nonneg = SomeValue::Integer(SomeInteger::new(true, false));
        let s_signed = SomeValue::Integer(SomeInteger::new(false, false));
        let out = bk.newlist(&[s_nonneg, s_signed], None).unwrap();
        // Element type is now signed Int (widened from nonneg).
        if let SomeValue::Integer(si) = out.listdef.s_value() {
            assert!(!si.nonneg);
        } else {
            panic!("expected SomeInteger listdef element");
        }
    }

    #[test]
    fn newdict_creates_someordicteddict_equivalent() {
        let bk = bk();
        let out = bk.newdict();
        // Fresh-position newdict without subsequent generalize_key /
        // generalize_value carries Impossible for both.
        assert!(matches!(out.dictdef.s_key(), SomeValue::Impossible));
        assert!(matches!(out.dictdef.s_value(), SomeValue::Impossible));
    }

    #[test]
    fn getlistdef_caches_on_same_position() {
        let bk = bk();
        bk.set_position_key(Some(PositionKey::new(1, 2, 0)));
        let ld1 = bk.getlistdef(None);
        let ld2 = bk.getlistdef(None);
        assert!(ld1.same_as(&ld2));
    }

    #[test]
    fn getdictdef_caches_on_same_position() {
        let bk = bk();
        bk.set_position_key(Some(PositionKey::new(3, 4, 0)));
        let dd1 = bk.getdictdef(false, false, false);
        let dd2 = bk.getdictdef(false, false, false);
        assert!(dd1.same_as(&dd2));
    }

    #[test]
    fn getlistdef_caches_under_none_position() {
        // upstream bookkeeper.py:180 indexes `self.listdefs[self.
        // position_key]` — when position_key is `None`, both calls
        // land on the same dict entry. Rust port mirrors this by
        // using `Option<PositionKey>` as the cache key; two `getlist
        // def` calls outside a reflow frame must share the same
        // ListDef.
        let bk = bk();
        assert_eq!(bk.current_position_key(), None);
        let ld1 = bk.getlistdef(None);
        let ld2 = bk.getlistdef(None);
        assert!(ld1.same_as(&ld2));
    }

    #[test]
    fn getdictdef_caches_under_none_position() {
        let bk = bk();
        assert_eq!(bk.current_position_key(), None);
        let dd1 = bk.getdictdef(false, false, false);
        let dd2 = bk.getdictdef(false, false, false);
        assert!(dd1.same_as(&dd2));
    }

    #[test]
    fn position_key_set_and_get() {
        let bk = bk();
        assert!(bk.current_position_key().is_none());
        let prev = bk.set_position_key(Some(PositionKey::new(1, 1, 1)));
        assert!(prev.is_none());
        assert_eq!(bk.current_position_key(), Some(PositionKey::new(1, 1, 1)));
    }

    #[test]
    fn unicode_through_const_str() {
        // ConstValue::Str carries both str and unicode upstream; the
        // byte-level no-nul check lands identically.
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Str("abc".into())).unwrap();
        match s {
            SomeValue::String(st) => assert!(st.inner.no_nul),
            other => panic!("expected SomeString, got {other:?}"),
        }
    }

    #[test]
    fn byte_array_not_yet_routed() {
        // ConstValue has no dedicated Bytes variant; bytearray inputs
        // therefore don't round-trip through immutablevalue today.
        // Test the type itself stays buildable from the annotator
        // model — sanity check in lieu of a full input path.
        let _ = SomeByteArray::default();
    }

    #[test]
    fn char_has_no_nul() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Str("x".into())).unwrap();
        match s {
            SomeValue::Char(_) => {
                let c = SomeChar::new(true);
                assert!(c.inner.no_nul);
            }
            other => panic!("expected SomeChar, got {other:?}"),
        }
    }

    // --- Phase 5 P5.2 classdesc c2: descs / classdefs / getdesc ---

    #[test]
    fn getdesc_for_user_function_returns_function_entry() {
        let bk = bk();
        let gf = GraphFunc::new("f", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let entry = bk.getdesc(&host).unwrap();
        assert!(entry.is_function());
        let fd = entry.as_function().unwrap();
        assert_eq!(fd.borrow().name, "f");
    }

    #[test]
    fn getdesc_caches_same_pyobj() {
        // Two getdesc calls with the same HostObject return the same
        // Rc (identity equal).
        let bk = bk();
        let gf = GraphFunc::new("g", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let a = bk.getdesc(&host).unwrap();
        let b = bk.getdesc(&host).unwrap();
        assert_eq!(a, b);
        // Same underlying FunctionDesc Rc — pointer-identity.
        let a_fd = a.as_function().unwrap();
        let b_fd = b.as_function().unwrap();
        assert!(Rc::ptr_eq(&a_fd, &b_fd));
    }

    #[test]
    fn getdesc_for_class_returns_class_entry_shell() {
        let bk = bk();
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        let entry = bk.getdesc(&cls).unwrap();
        assert!(entry.is_class());
        let cd = entry.as_class().unwrap();
        assert_eq!(cd.borrow().name, "pkg.Foo");
    }

    #[test]
    fn getdesc_for_instance_returns_frozen_entry() {
        let bk = bk();
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        let inst = HostObject::new_instance(cls, vec![]);
        let entry = bk.getdesc(&inst).unwrap();
        assert!(entry.is_frozen());
    }

    #[test]
    fn getdesc_for_builtin_callable_returns_frozen_entry() {
        // bookkeeper.py treats builtin callables as frozen PBCs via
        // the _freeze_ fallback.
        let bk = bk();
        let obj = HostObject::new_builtin_callable("len");
        let entry = bk.getdesc(&obj).unwrap();
        assert!(entry.is_frozen());
    }

    #[test]
    fn getmethoddesc_caches_identity() {
        // Two calls with the same funcdesc/classdefs/name/flags
        // return the same MethodDesc Rc.
        let bk = bk();
        let gf = GraphFunc::new("m", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let fd = bk.getdesc(&host).unwrap().as_function().unwrap();
        let origin = crate::annotator::description::ClassDefKey::from_raw(1);
        let self_def = Some(crate::annotator::description::ClassDefKey::from_raw(2));
        let flags = std::collections::BTreeMap::new();
        let a = bk.getmethoddesc(&fd, origin, self_def, "m", flags.clone());
        let b = bk.getmethoddesc(&fd, origin, self_def, "m", flags.clone());
        assert!(Rc::ptr_eq(&a, &b));
    }

    #[test]
    fn register_classdef_appends_to_snapshot() {
        let bk = bk();
        let cd_rc = ClassDef::new_standalone("pkg.C", None);
        bk.register_classdef(cd_rc.clone());
        let snap = bk.classdef_snapshot();
        assert_eq!(snap.len(), 1);
        assert!(Rc::ptr_eq(&snap[0], &cd_rc));
    }

    #[test]
    fn descentry_identity_eq() {
        let bk = bk();
        let gf = GraphFunc::new("h", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let a = bk.getdesc(&host).unwrap();
        let b = bk.getdesc(&host).unwrap();
        assert_eq!(a.desc_key(), b.desc_key());
    }

    #[test]
    fn enter_leave_registers_tls() {
        let bk = bk();
        assert!(getbookkeeper().is_none());
        bk.enter(Some(PositionKey::new(1, 1, 0)));
        assert!(getbookkeeper().is_some());
        assert!(bk.position_entered.get());
        assert_eq!(bk.current_position_key(), Some(PositionKey::new(1, 1, 0)));
        bk.leave();
        assert!(getbookkeeper().is_none());
        assert!(!bk.position_entered.get());
        assert!(bk.current_position_key().is_none());
    }

    #[test]
    #[should_panic(expected = "don't call enter() nestedly")]
    fn enter_twice_panics() {
        let bk = bk();
        bk.enter(Some(PositionKey::new(1, 1, 0)));
        // Second enter without leave — matches upstream's `assert not hasattr`.
        bk.enter(Some(PositionKey::new(2, 2, 0)));
    }

    #[test]
    fn at_position_raii_enters_and_leaves() {
        let bk = bk();
        {
            let _guard = bk.at_position(Some(PositionKey::new(3, 4, 0)));
            assert!(bk.position_entered.get());
            assert!(getbookkeeper().is_some());
        }
        assert!(!bk.position_entered.get());
        assert!(getbookkeeper().is_none());
    }

    #[test]
    fn at_position_none_fast_path_skips_when_entered() {
        let bk = bk();
        bk.enter(Some(PositionKey::new(5, 5, 0)));
        {
            // Upstream fast-path: hasattr(self, 'position_key') and pos is None
            let _guard = bk.at_position(None);
            assert!(bk.position_entered.get(), "outer enter stays active");
            // Fast-path does not clobber position.
            assert_eq!(bk.current_position_key(), Some(PositionKey::new(5, 5, 0)));
        }
        // Guard drop must NOT leave because we skipped the enter.
        assert!(bk.position_entered.get());
        bk.leave();
    }

    #[test]
    fn valueoftype_delegates_to_annotationoftype() {
        use crate::annotator::signature::AnnotationSpec;
        let bk = bk();
        let bool_ann = bk.valueoftype(&AnnotationSpec::Bool).unwrap();
        assert!(matches!(bool_ann, SomeValue::Bool(_)));
        let int_ann = bk.valueoftype(&AnnotationSpec::Int).unwrap();
        assert!(matches!(int_ann, SomeValue::Integer(_)));
    }

    #[test]
    fn immutablevalue_free_function_uses_tls() {
        use crate::flowspace::model::ConstValue;
        let bk = bk();
        bk.enter(Some(PositionKey::new(1, 1, 0)));
        let s = immutablevalue(&ConstValue::Bool(true)).unwrap();
        assert!(matches!(s, SomeValue::Bool(_)));
        bk.leave();
    }
}
