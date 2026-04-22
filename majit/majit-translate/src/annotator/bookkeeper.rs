//! Bookkeeper — central state carrier for the annotator.
//!
//! RPython upstream: `rpython/annotator/bookkeeper.py` (614 LOC).
//!
//! Structural port of `immutablevalue` / `newlist` / `newdict` and the
//! descriptor / class machinery (`descs`, `classdefs`, `methoddescs`,
//! `emulated_pbc_calls`, `frozenpbc_attr_families`, …). Fields not yet
//! landed are documented at the site that still dispatches through the
//! deferred branch.
//!
//! ## Dependency-blocked paths
//!
//! * `immutablevalue(HostObject)` — the `extregistry.is_registered(x)`
//!   branch (bookkeeper.py:312-314) returns `AnnotatorError` until the
//!   extregistry bridge lands.
//! * `BUILTIN_ANALYZERS` registry — ported in [`super::builtin`].
//!   `SomeBuiltin.analyser_name` is seeded from the host qualname;
//!   `SomeValue::call()` for `Builtin(_)` dispatches through
//!   [`super::builtin::call_builtin`]. Upstream only routes registered
//!   callables through `SomeBuiltin` (bookkeeper.py:309-311) and falls
//!   every other builtin through to the `callable(x)` branch; the Rust
//!   port preserves the "everything becomes `SomeBuiltin`" shape to
//!   keep `policy.rs` sandboxing dispatch intact, so non-registered
//!   qualnames surface at call time as
//!   `AnnotatorError::new("no analyser registered for …")`.
//! * `classpbc_attr_families` / `all_specializations` — not yet
//!   mirrored on the Rust bookkeeper; reachable only from rtyper-phase
//!   consumers.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};

use super::argument::{ArgumentsForTranslation, complex_args, simple_args};
use super::classdesc::{ClassDef, ClassDesc};
use super::description::{
    CallFamily, ClassAttrFamily, ClassDefKey, DescEntry, DescKey, FrozenAttrFamily, FrozenDesc,
    FunctionDesc, MethodDesc,
};
use super::dictdef::DictDef;
use super::listdef::ListDef;
use super::model::{
    AnnotatorError, SomeBool, SomeBuiltin, SomeChar, SomeDict, SomeFloat, SomeInteger, SomeList,
    SomePBC, SomeString, SomeTuple, SomeValue, s_none,
};
use super::policy::AnnotatorPolicy;
use crate::flowspace::argument::CallShape;
use crate::flowspace::argument::Signature;
use crate::flowspace::bytecode::cpython_code_signature;
use crate::flowspace::model::{BlockRef, ConstValue, Constant, GraphRef, HostObject};
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
/// `Rc::as_ptr(&graph) as usize` / `Rc::as_ptr(&block) as usize`. The
/// `graph_ref` / `block_ref` slots carry the actual `Weak` refs when
/// production code constructs the key via [`Self::from_refs`];
/// upstream `reflowfromposition(position_key)` unpacks `graph, block,
/// index = position_key` and needs those references back.
///
/// Test constructors use [`Self::new`] and leave the weak slots
/// dangling — downstream consumers that need the graph/block refs
/// detect this via `Weak::upgrade()` returning `None`.
///
/// Hash / Eq always use the three integer identities, so synthetic
/// test keys and production keys with equal hashes are
/// indistinguishable (mirroring upstream's Python tuple equality via
/// `id()`).
#[derive(Clone, Debug)]
pub struct PositionKey {
    /// Identity hash of the enclosing `FunctionGraph` — upstream
    /// `position_key[0]`.
    pub graph_id: usize,
    /// Identity hash of the enclosing `Block` — upstream
    /// `position_key[1]`.
    pub block_id: usize,
    /// Operation index inside the block — upstream `position_key[2]`.
    pub op_index: usize,
    /// Weak reference to the enclosing `FunctionGraph`. Populated by
    /// [`Self::from_refs`]; `None` for test-only synthetic keys.
    pub graph_ref:
        Option<std::rc::Weak<std::cell::RefCell<crate::flowspace::model::FunctionGraph>>>,
    /// Weak reference to the enclosing `Block`. Populated by
    /// [`Self::from_refs`]; `None` for test-only synthetic keys.
    pub block_ref: Option<std::rc::Weak<std::cell::RefCell<crate::flowspace::model::Block>>>,
}

impl PartialEq for PositionKey {
    fn eq(&self, other: &Self) -> bool {
        self.graph_id == other.graph_id
            && self.block_id == other.block_id
            && self.op_index == other.op_index
    }
}

impl Eq for PositionKey {}

impl std::hash::Hash for PositionKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.graph_id.hash(state);
        self.block_id.hash(state);
        self.op_index.hash(state);
    }
}

impl PositionKey {
    /// Synthetic constructor — test-only. Fills the identity triple
    /// directly and leaves the Weak refs empty.
    pub fn new(graph_id: usize, block_id: usize, op_index: usize) -> Self {
        PositionKey {
            graph_id,
            block_id,
            op_index,
            graph_ref: None,
            block_ref: None,
        }
    }

    /// Production constructor — derives the identity hashes from
    /// `Rc::as_ptr` (matches upstream's Python tuple identity via
    /// `id()`), and retains `Weak` refs so consumers like
    /// `reflowfromposition` can upgrade back to the live
    /// `FunctionGraph` / `Block`.
    pub fn from_refs(graph: &GraphRef, block: &BlockRef, op_index: usize) -> Self {
        PositionKey {
            graph_id: Rc::as_ptr(graph) as usize,
            block_id: Rc::as_ptr(block) as usize,
            op_index,
            graph_ref: Some(Rc::downgrade(graph)),
            block_ref: Some(Rc::downgrade(block)),
        }
    }

    /// Upgrade the weak graph reference, if the key carries one and
    /// the target is still alive. Mirrors upstream's `graph, _, _ =
    /// position_key` tuple unpack.
    pub fn graph(&self) -> Option<GraphRef> {
        self.graph_ref.as_ref().and_then(|w| w.upgrade())
    }

    /// Upgrade the weak block reference. Mirrors upstream's `_, block,
    /// _ = position_key` unpack.
    pub fn block(&self) -> Option<BlockRef> {
        self.block_ref.as_ref().and_then(|w| w.upgrade())
    }
}

/// RPython `class Bookkeeper` (bookkeeper.py:53).
pub struct Bookkeeper {
    /// RPython `self.annotator = annotator` (bookkeeper.py:53). A weak
    /// backlink to the owning `RPythonAnnotator`; stored as
    /// `Weak<RPythonAnnotator>` to break the Rc cycle (annotator owns
    /// the bookkeeper via `Rc`). Upgraded on demand via
    /// [`Self::annotator`].
    ///
    /// Test-only `Bookkeeper::new()` leaves this slot empty (the tests
    /// never call into annotator-dependent code paths).
    pub annotator: RefCell<Weak<crate::annotator::annrpython::RPythonAnnotator>>,
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
    /// RPython `self.classpbc_attr_families = {}` (bookkeeper.py:62) —
    /// lazy `attrname -> UnionFind(ClassAttrFamily)` map materialised by
    /// `get_classpbc_attr_families(attrname)` (bookkeeper.py:447-456).
    pub classpbc_attr_families:
        RefCell<HashMap<String, UnionFind<DescKey, Rc<RefCell<ClassAttrFamily>>>>>,
    /// RPython `self.pbc_maximal_call_families = UnionFind(CallFamily)`
    /// (bookkeeper.py:64).
    pub pbc_maximal_call_families: RefCell<UnionFind<DescKey, Rc<RefCell<CallFamily>>>>,
    /// RPython `self.emulated_pbc_calls = {}` (bookkeeper.py:66).
    pub emulated_pbc_calls: RefCell<HashMap<EmulatedPbcCallKey, (SomePBC, Vec<SomeValue>)>>,
    /// RPython `self.immutable_cache = {}` (bookkeeper.py:61), keyed by
    /// `Constant(x)` for the List / Dict / OrderedDict / r_dict branches
    /// of `immutablevalue` (bookkeeper.py:255-298). Upstream's cache
    /// memoises the returned `SomeList` / `SomeDict` so downstream
    /// `generalize_key` / `generalize` mutations on the cached
    /// `DictDef` / `ListDef` persist across calls.
    ///
    /// `transform_list_contains` (translator/transform.py:115-134)
    /// relies on this: it calls `self.annotation(Constant(dict))` after
    /// rewriting the contains lhs, then mutates the returned SomeDict's
    /// dictdef via `generalize_key`. Without the cache the second
    /// `annotation()` call builds a fresh DictDef and the mutation
    /// dissolves.
    ///
    /// **Precision deviation from upstream.** Upstream uses
    /// `Hashable(Constant(x))` which for hashable values hashes on
    /// `(type(x), x)` and falls back to `id(x)` for mutable
    /// list/dict/set values (see `rpython/tool/uid.py:27-38`). The Rust
    /// port keys this cache on the value-equal `ConstValue` enum, so two
    /// distinct prebuilt `[]` / `{}` literals with identical contents
    /// share a cache entry. Any `generalize` / `generalize_key` call on
    /// the cached `ListDef` / `DictDef` then widens both literals'
    /// annotations together — soundness is preserved (widening stays
    /// monotonic in the lattice) but precision is lost across
    /// structurally-equal distinct constants.
    ///
    /// The identity-based port needs `Hlvalue::Constant` to carry an
    /// identity-bearing wrapper (`Rc<Constant>` or an atomic id field)
    /// so the cache can key on the Rc pointer like upstream's `id(x)`
    /// fallback. That refactor is deferred until the rtyper port (Phase
    /// 6) exposes a site where the precision loss is user-visible.
    pub immutable_cache: RefCell<HashMap<ConstValue, SomeValue>>,
    /// RPython `self.pending_specializations = []` (bookkeeper.py:69).
    ///
    /// List of callbacks drained by
    /// `AnnotatorPolicy.no_more_blocks_to_annotate` before the final
    /// annotation fixpoint.
    pub pending_specializations: RefCell<Vec<Box<dyn Fn()>>>,
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
    ClassDef(ClassDefKey),
    RDictCall {
        item_id: usize,
        role: &'static str,
    },
    Text(String),
    /// RPython `('sandboxing', s_func.const)` tuple (policy.py:87). The
    /// inner identity is the `HostObject` pointer for the external
    /// callable being sandboxed.
    Sandboxing {
        callable_id: usize,
    },
}

/// RPython `Bookkeeper.pbc_call`'s `emulated` parameter
/// (bookkeeper.py:512-531). Encodes the Python-side three-state
/// polymorphism (`None` / `True` / `<position_key>`) as a Rust enum:
///
/// * `None` — real call: use `self.position_key`; pull
///   `s_previous_result` from the current op's annotation.
/// * `True` — fully emulated: `whence=None`, `op=None`,
///   `s_previous_result=s_ImpossibleValue`.
/// * `Callback(position)` — callback-style: `whence=position`, same
///   `op=None` / `s_previous_result=s_ImpossibleValue` as `True`.
#[derive(Clone, Debug)]
pub enum PbcCallEmulated {
    None,
    True,
    Callback(PositionKey),
}

impl EmulatedPbcCallKey {
    /// Build the sandbox-trampoline key used by
    /// `AnnotatorPolicy::no_more_blocks_to_annotate` (policy.py:87).
    pub fn sandboxing(func_const: &crate::flowspace::model::ConstValue) -> Self {
        use crate::flowspace::model::ConstValue;
        let callable_id = match func_const {
            ConstValue::HostObject(obj) => obj.identity_id(),
            // Non-HostObject consts reach here only if the policy walker
            // stumbles over a non-sandboxable callable; fall back to the
            // pointer of the enum discriminant so two such entries stay
            // distinct yet the key remains stable for that
            // call site.
            _ => std::ptr::addr_of!(*func_const) as usize,
        };
        EmulatedPbcCallKey::Sandboxing { callable_id }
    }
}

/// Variant wrapper over the two concrete attribute-family types that
/// `pbc_getattr` (bookkeeper.py:458-496) consumes polymorphically —
/// `ClassAttrFamily` for ClassDesc PBCs, `FrozenAttrFamily` for
/// FrozenDesc PBCs. Exposes the exact trio upstream touches:
/// `read_locations`, `get_s_value(attr)`, `set_s_value(attr, ...)`.
enum PbcAttrFamily {
    Class(Rc<RefCell<ClassAttrFamily>>),
    Frozen(Rc<RefCell<FrozenAttrFamily>>),
}

impl PbcAttrFamily {
    fn add_read_location(&self, pos: PositionKey) {
        match self {
            PbcAttrFamily::Class(rc) => {
                rc.borrow_mut().read_locations.insert(pos, ());
            }
            PbcAttrFamily::Frozen(rc) => {
                rc.borrow_mut().read_locations.insert(pos, ());
            }
        }
    }

    fn read_locations(&self) -> Vec<PositionKey> {
        match self {
            PbcAttrFamily::Class(rc) => rc.borrow().read_locations.keys().cloned().collect(),
            PbcAttrFamily::Frozen(rc) => rc.borrow().read_locations.keys().cloned().collect(),
        }
    }

    fn get_s_value(&self, attr: &str) -> SomeValue {
        match self {
            PbcAttrFamily::Class(rc) => rc.borrow().get_s_value(attr),
            PbcAttrFamily::Frozen(rc) => rc.borrow().get_s_value(attr),
        }
    }

    fn set_s_value(&self, attr: &str, v: SomeValue) {
        match self {
            PbcAttrFamily::Class(rc) => rc.borrow_mut().set_s_value(attr, v),
            PbcAttrFamily::Frozen(rc) => rc.borrow_mut().set_s_value(attr, v),
        }
    }
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
            annotator: RefCell::new(Weak::new()),
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
            classpbc_attr_families: RefCell::new(HashMap::new()),
            pbc_maximal_call_families: RefCell::new(UnionFind::new(|desc: &DescKey| {
                Rc::new(RefCell::new(CallFamily::new(*desc)))
            })),
            emulated_pbc_calls: RefCell::new(HashMap::new()),
            immutable_cache: RefCell::new(HashMap::new()),
            pending_specializations: RefCell::new(Vec::new()),
            position_entered: std::cell::Cell::new(false),
        }
    }

    /// Wire up the `self.annotator` backlink. Invoked from
    /// [`RPythonAnnotator::new`] via `Rc::new_cyclic` so the
    /// bookkeeper's weak reference points to the final `Rc<Self>`.
    pub fn set_annotator(&self, ann: Weak<crate::annotator::annrpython::RPythonAnnotator>) {
        *self.annotator.borrow_mut() = ann;
    }

    /// RPython `self.annotator` attribute access — upgrades the weak
    /// backlink to an `Rc<RPythonAnnotator>`. Panics if the backlink
    /// is absent or the annotator has been dropped; both are programmer
    /// errors mirroring upstream's assumption that `self.annotator`
    /// is always live.
    pub fn annotator(&self) -> Rc<crate::annotator::annrpython::RPythonAnnotator> {
        self.annotator
            .borrow()
            .upgrade()
            .expect("Bookkeeper.annotator backlink is absent or dropped")
    }

    /// RPython `Bookkeeper.warning(self, msg)` (bookkeeper.py:580-581).
    ///
    /// ```python
    /// def warning(self, msg):
    ///     return self.annotator.warning(msg)
    /// ```
    pub fn warning(&self, msg: impl Into<String>) {
        let msg = msg.into();
        self.annotator().warning(&msg);
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
        self.position_key.borrow().clone()
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

    /// RPython `Bookkeeper.check_no_flags_on_instances(self)`
    /// (bookkeeper.py:120-150) — post-annotation sanity check invoked
    /// by `RPythonAnnotator.validate()`.
    ///
    /// Upstream body:
    ///
    /// ```python
    /// def check_no_flags_on_instances(self):
    ///     seen = set()
    ///     def check_no_flags(s_value_or_def):
    ///         if isinstance(s_value_or_def, SomeInstance):
    ///             assert not s_value_or_def.flags, ...
    ///             check_no_flags(s_value_or_def.classdef)
    ///         elif isinstance(s_value_or_def, SomeList):
    ///             check_no_flags(s_value_or_def.listdef.listitem)
    ///         elif isinstance(s_value_or_def, SomeDict):
    ///             check_no_flags(s_value_or_def.dictdef.dictkey)
    ///             check_no_flags(s_value_or_def.dictdef.dictvalue)
    ///         elif isinstance(s_value_or_def, SomeTuple):
    ///             for s_item in s_value_or_def.items:
    ///                 check_no_flags(s_item)
    ///         elif isinstance(s_value_or_def, ClassDef):
    ///             if s_value_or_def in seen: return
    ///             seen.add(s_value_or_def)
    ///             for attr in s_value_or_def.attrs.itervalues():
    ///                 check_no_flags(attr.s_value)
    ///         elif isinstance(s_value_or_def, ListItem):
    ///             if s_value_or_def in seen: return
    ///             seen.add(s_value_or_def)
    ///             check_no_flags(s_value_or_def.s_value)
    ///
    ///     for clsdef in self.classdefs:
    ///         check_no_flags(clsdef)
    /// ```
    pub fn check_no_flags_on_instances(&self) {
        let mut seen_classdefs: HashSet<ClassDefKey> = HashSet::new();
        let mut seen_listitems: HashSet<usize> = HashSet::new();
        for clsdef in self.classdefs.borrow().iter() {
            check_no_flags_classdef(clsdef, &mut seen_classdefs, &mut seen_listitems);
        }
    }

    /// RPython `Bookkeeper.compute_at_fixpoint(self)`
    /// (bookkeeper.py:108-118) — invoked at the tail of
    /// `RPythonAnnotator.simplify()`.
    ///
    /// ```python
    /// def compute_at_fixpoint(self):
    ///     # getbookkeeper() needs to work during this function, so provide
    ///     # one with a dummy position
    ///     with self.at_position(None):
    ///         for call_op in self.annotator.call_sites():
    ///             self.consider_call_site(call_op)
    ///         for pbc, args_s in self.emulated_pbc_calls.itervalues():
    ///             args = simple_args(args_s)
    ///             pbc.consider_call_site(args, s_ImpossibleValue, None)
    ///         self.emulated_pbc_calls = {}
    /// ```
    ///
    /// Structural 1:1 port. `SomePBC::consider_call_site` currently
    /// routes through `DescKind.consider_call_site` with a bridging
    /// gap (see that method's doc); the outer scaffolding — at_position
    /// guard, call_sites drain, emulated_pbc_calls drain + clear — is
    /// parity-faithful.
    pub fn compute_at_fixpoint(self: &Rc<Self>) {
        let _guard = self.at_position(None);
        let ann = self.annotator();
        for (call_op, op_key) in ann.call_sites_with_positions() {
            // Errors surface via annotator.errors in upstream (Priority
            // #3). For now propagate-by-ignore matches the stub contract.
            let _ = self.consider_call_site(&call_op, Some(op_key));
        }
        // Snapshot values so we can mutate emulated_pbc_calls inside the
        // loop (upstream does `for pbc, args_s in
        // self.emulated_pbc_calls.itervalues()` while not mutating).
        let emulated: Vec<(SomePBC, Vec<SomeValue>)> =
            self.emulated_pbc_calls.borrow().values().cloned().collect();
        for (pbc, args_s) in emulated {
            // upstream: `args = simple_args(args_s)`;
            //            `pbc.consider_call_site(args, s_ImpossibleValue, None)`.
            let args = simple_args(args_s);
            let _ = pbc.consider_call_site(&args, &SomeValue::Impossible, None);
        }
        // upstream: `self.emulated_pbc_calls = {}`.
        self.emulated_pbc_calls.borrow_mut().clear();
    }

    /// RPython `Bookkeeper.consider_call_site(self, call_op)`
    /// (bookkeeper.py:152-166).
    ///
    /// ```python
    /// def consider_call_site(self, call_op):
    ///     from rpython.rtyper.llannotation import SomeLLADTMeth, lltype_to_annotation
    ///     annotation = self.annotator.annotation
    ///     s_callable = annotation(call_op.args[0])
    ///     args_s = [annotation(arg) for arg in call_op.args[1:]]
    ///     if isinstance(s_callable, SomeLLADTMeth):
    ///         adtmeth = s_callable
    ///         s_callable = self.immutablevalue(adtmeth.func)
    ///         args_s = [lltype_to_annotation(adtmeth.ll_ptrtype)] + args_s
    ///     if isinstance(s_callable, SomePBC):
    ///         s_result = annotation(call_op.result)
    ///         if s_result is None:
    ///             s_result = s_ImpossibleValue
    ///         args = call_op.build_args(args_s)
    ///         s_callable.consider_call_site(args, s_result, call_op)
    /// ```
    ///
    /// `SomeLLADTMeth` is an rtyper-side type the annotator never
    /// constructs; that branch is a no-op here. `call_op.build_args`
    /// is opname-sensitive (`simple_call` →
    /// `ArgumentsForTranslation(list(args_s))`, `call_args` →
    /// `fromshape`); the simple_call path uses [`simple_args`].
    pub fn consider_call_site(
        self: &Rc<Self>,
        call_op: &crate::flowspace::model::SpaceOperation,
        op_key: Option<PositionKey>,
    ) -> Result<(), AnnotatorError> {
        let ann = self.annotator();
        let Some(s_callable) = ann.annotation(&call_op.args[0]) else {
            return Ok(());
        };
        // SomeLLADTMeth path is rtyper-only; not ported.
        if let SomeValue::PBC(pbc) = &s_callable {
            let args_s: Vec<SomeValue> = call_op
                .args
                .iter()
                .skip(1)
                .filter_map(|a| ann.annotation(a))
                .collect();
            let s_result = ann
                .annotation(&call_op.result)
                .unwrap_or(SomeValue::Impossible);
            // upstream: `args = call_op.build_args(args_s)` —
            // opname-sensitive: `simple_call` → `ArgumentsForTranslation
            // (list(args_s))`, `call_args` → `fromshape(args_s[0].const,
            // args_s[1:])`.
            let args = build_args_for_op(&call_op.opname, &args_s)?;
            pbc.consider_call_site(&args, &s_result, op_key)?;
        }
        Ok(())
    }

    /// RPython `Bookkeeper.getattr_locations(self, clsdesc, attrname)`
    /// (bookkeeper.py:498-500).
    ///
    /// ```python
    /// def getattr_locations(self, clsdesc, attrname):
    ///     attrdef = clsdesc.classdef.find_attribute(attrname)
    ///     return attrdef.read_locations
    /// ```
    ///
    /// Returns a snapshot of the attribute's read locations — callers
    /// iterate + reflow outside the borrow so [`ClassDef::find_attribute`]
    /// can reacquire the RefCell if it generalizes the attribute.
    pub fn getattr_locations(
        &self,
        classdesc: &Rc<RefCell<ClassDesc>>,
        attrname: &str,
    ) -> Result<HashSet<PositionKey>, AnnotatorError> {
        // upstream: `attrdef = clsdesc.classdef.find_attribute(attrname)`
        let classdef = classdesc
            .borrow()
            .classdef
            .clone()
            .and_then(|w| w.upgrade())
            .ok_or_else(|| {
                AnnotatorError::new(
                    "Bookkeeper.getattr_locations: classdesc.classdef is not set \
                     (requires ClassDesc.getuniqueclassdef — classdesc.py:699-702)"
                        .to_string(),
                )
            })?;
        // `find_attribute` may `generalize_attr` as a side effect; run
        // it outside any borrow of `classdef`.
        let _ = ClassDef::find_attribute(&classdef, attrname)?;
        // Now re-lookup the attribute to read its read_locations.
        let attr = classdef.borrow();
        let attrdef = attr.attrs.get(attrname).ok_or_else(|| {
            AnnotatorError::new(format!(
                "Bookkeeper.getattr_locations: attribute {:?} missing on {:?}",
                attrname, attr.name
            ))
        })?;
        Ok(attrdef.read_locations.clone())
    }

    /// RPython `Bookkeeper.record_getattr(self, clsdesc, attrname)`
    /// (bookkeeper.py:502-504).
    ///
    /// ```python
    /// def record_getattr(self, clsdesc, attrname):
    ///     locations = self.getattr_locations(clsdesc, attrname)
    ///     locations.add(self.position_key)
    /// ```
    ///
    /// The Rust port cannot return the mutable set by reference without
    /// extending the RefCell borrow across the caller's chain, so it
    /// inserts directly into the attribute's `read_locations`.
    pub fn record_getattr(
        &self,
        classdesc: &Rc<RefCell<ClassDesc>>,
        attrname: &str,
    ) -> Result<(), AnnotatorError> {
        let classdef = classdesc
            .borrow()
            .classdef
            .clone()
            .and_then(|w| w.upgrade())
            .ok_or_else(|| {
                AnnotatorError::new(
                    "Bookkeeper.record_getattr: classdesc.classdef is not set".to_string(),
                )
            })?;
        // Ensure the attribute exists (upstream's
        // `clsdesc.classdef.find_attribute(attrname)` side effect).
        let _ = ClassDef::find_attribute(&classdef, attrname)?;
        // upstream: `locations.add(self.position_key)`
        if let Some(pk) = self.current_position_key() {
            let mut classdef_mut = classdef.borrow_mut();
            if let Some(attrdef) = classdef_mut.attrs.get_mut(attrname) {
                attrdef.read_locations.insert(pk);
            }
        }
        Ok(())
    }

    /// RPython `Bookkeeper.update_attr(self, clsdef, attrdef)`
    /// (bookkeeper.py:506-510).
    ///
    /// ```python
    /// def update_attr(self, clsdef, attrdef):
    ///     locations = self.getattr_locations(clsdef.classdesc, attrdef.name)
    ///     for position in locations:
    ///         self.annotator.reflowfromposition(position)
    ///     attrdef.validate(homedef=clsdef)
    /// ```
    ///
    /// Rust signature takes the attribute name (rather than an
    /// `&mut Attribute`) because the attribute lives inside
    /// `clsdef.borrow_mut().attrs`; validate is executed after a
    /// `remove`/`insert` cycle so the attribute is the exclusive mutator
    /// while the `homedef` is borrowed read-only.
    pub fn update_attr(
        &self,
        clsdef: &Rc<RefCell<ClassDef>>,
        attr_name: &str,
    ) -> Result<(), AnnotatorError> {
        let classdesc = clsdef.borrow().classdesc.clone();
        // upstream: `locations = self.getattr_locations(clsdef.classdesc, attrdef.name)`
        let locations = self.getattr_locations(&classdesc, attr_name)?;
        // upstream: `for position in locations: self.annotator.reflowfromposition(position)`
        let Some(ann) = self.annotator.borrow().upgrade() else {
            // upstream always has `self.annotator`; if the backlink is
            // not wired yet (tests constructing a Bookkeeper without an
            // annotator) skip reflow silently — same effect as upstream
            // when the pending-block queue is empty.
            return Ok(());
        };
        for position in locations {
            ann.reflowfromposition(&position);
        }
        // upstream: `attrdef.validate(homedef=clsdef)`
        let taken = clsdef.borrow_mut().attrs.remove(attr_name);
        if let Some(mut attrdef) = taken {
            let result = attrdef.validate(clsdef);
            clsdef
                .borrow_mut()
                .attrs
                .insert(attr_name.to_string(), attrdef);
            result.map_err(|e| AnnotatorError::new(e.to_string()))?;
        }
        Ok(())
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
    ///   * `BoundMethod` → `MethodDesc` / `MethodOfFrozenDesc`
    ///   * `Instance` / `BuiltinCallable` / `Module` / `Opaque` →
    ///     [`Self::getfrozen`] (upstream's `_freeze_` fallback)
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
        } else if pyobj.is_bound_method() {
            let self_obj = pyobj.bound_method_self().ok_or_else(|| {
                AnnotatorError::new("Bookkeeper.getdesc(bound method): missing __self__")
            })?;
            let func = pyobj.bound_method_func().ok_or_else(|| {
                AnnotatorError::new("Bookkeeper.getdesc(bound method): missing __func__")
            })?;
            let funcdesc = self.getdesc(func)?.as_function().ok_or_else(|| {
                AnnotatorError::new("Bookkeeper.getdesc(bound method): __func__ is not a function")
            })?;

            if self_obj.is_instance() {
                call_cleanup_method(self_obj)?;
            }
            if self_obj.is_instance() && call_freeze_method(self_obj)? {
                let frozendesc = self.getdesc(self_obj)?.as_frozen().ok_or_else(|| {
                    AnnotatorError::new(
                        "Bookkeeper.getdesc(bound method): frozen self did not produce FrozenDesc",
                    )
                })?;
                DescEntry::MethodOfFrozen(Rc::new(RefCell::new(
                    super::description::MethodOfFrozenDesc::new(self.clone(), funcdesc, frozendesc),
                )))
            } else {
                let origin_class = pyobj.bound_method_origin_class().ok_or_else(|| {
                    AnnotatorError::new(
                        "Bookkeeper.getdesc(bound method): missing origin class from descriptor lookup",
                    )
                })?;
                let self_class = if self_obj.is_class() {
                    self_obj.clone()
                } else {
                    self_obj.instance_class().cloned().ok_or_else(|| {
                        AnnotatorError::new(
                            "Bookkeeper.getdesc(bound method): regular method self has no class",
                        )
                    })?
                };
                let classdef = self.getuniqueclassdef(&self_class)?;
                if self_obj.is_instance() {
                    super::classdesc::ClassDef::see_instance(&classdef, self_obj)?;
                }
                let name = pyobj.bound_method_name().ok_or_else(|| {
                    AnnotatorError::new("Bookkeeper.getdesc(bound method): missing method name")
                })?;
                let _ = super::classdesc::ClassDef::find_attribute(&classdef, name)?;
                DescEntry::Method(self.getmethoddesc(
                    &funcdesc,
                    ClassDefKey::from_classdef(&self.getuniqueclassdef(origin_class)?),
                    Some(ClassDefKey::from_classdef(&classdef)),
                    name,
                    std::collections::BTreeMap::new(),
                ))
            }
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

    /// RPython `Bookkeeper.get_classpbc_attr_families(attrname)`
    /// (bookkeeper.py:447-456).
    ///
    /// ```python
    /// def get_classpbc_attr_families(self, attrname):
    ///     map = self.classpbc_attr_families
    ///     try:
    ///         access_sets = map[attrname]
    ///     except KeyError:
    ///         access_sets = map[attrname] = UnionFind(description.ClassAttrFamily)
    ///     return access_sets
    /// ```
    ///
    /// Rust exposes the lookup via a closure so the returned mutable
    /// handle doesn't escape the `RefCell::borrow_mut()` borrow. The
    /// attrname's UnionFind is materialised on first access with the
    /// `ClassAttrFamily`-factory, matching upstream's lazy creation.
    pub fn with_classpbc_attr_families<T>(
        &self,
        attrname: &str,
        f: impl FnOnce(&mut UnionFind<DescKey, Rc<RefCell<ClassAttrFamily>>>) -> T,
    ) -> T {
        let mut map = self.classpbc_attr_families.borrow_mut();
        let entry = map.entry(attrname.to_string()).or_insert_with(|| {
            UnionFind::new(|desc: &DescKey| Rc::new(RefCell::new(ClassAttrFamily::new(*desc))))
        });
        f(entry)
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

    /// RPython `Bookkeeper.new_exception(self, exc_classes)`
    /// (bookkeeper.py:174-176).
    ///
    /// ```python
    /// def new_exception(self, exc_classes):
    ///     clsdefs = {self.getuniqueclassdef(cls) for cls in exc_classes}
    ///     return SomeException(clsdefs)
    /// ```
    pub fn new_exception(
        self: &Rc<Self>,
        exc_classes: &[HostObject],
    ) -> Result<super::model::SomeException, super::model::AnnotatorError> {
        let mut clsdefs: Vec<Rc<RefCell<ClassDef>>> = Vec::new();
        for cls in exc_classes {
            let cd = self.getuniqueclassdef(cls)?;
            clsdefs.push(cd);
        }
        Ok(super::model::SomeException::new(clsdefs))
    }

    /// RPython `Bookkeeper.pbc_call(self, pbc, args, emulated=None)`
    /// (bookkeeper.py:512-537).
    ///
    /// ```python
    /// def pbc_call(self, pbc, args, emulated=None):
    ///     if emulated is None:
    ///         whence = self.position_key
    ///         fn, block, i = self.position_key
    ///         op = block.operations[i]
    ///         s_previous_result = self.annotator.annotation(op.result)
    ///         if s_previous_result is None:
    ///             s_previous_result = s_ImpossibleValue
    ///     else:
    ///         if emulated is True:
    ///             whence = None
    ///         else:
    ///             whence = emulated
    ///         op = None
    ///         s_previous_result = s_ImpossibleValue
    ///     results = []
    ///     for desc in pbc.descriptions:
    ///         results.append(desc.pycall(whence, args, s_previous_result, op))
    ///     s_result = unionof(*results)
    ///     return s_result
    /// ```
    ///
    /// RPython `Bookkeeper.pbc_getattr(pbc, s_attr)` (bookkeeper.py:458-496).
    ///
    /// ```python
    /// def pbc_getattr(self, pbc, s_attr):
    ///     assert s_attr.is_constant()
    ///     attr = s_attr.const
    ///     descs = list(pbc.descriptions)
    ///     first = descs[0]
    ///     if len(descs) == 1:
    ///         return first.s_read_attribute(attr)
    ///     change = first.mergeattrfamilies(descs[1:], attr)
    ///     attrfamily = first.getattrfamily(attr)
    ///     position = self.position_key
    ///     attrfamily.read_locations[position] = True
    ///     actuals = []
    ///     for desc in descs:
    ///         actuals.append(desc.s_read_attribute(attr))
    ///     s_result = unionof(*actuals)
    ///     s_oldvalue = attrfamily.get_s_value(attr)
    ///     attrfamily.set_s_value(attr, unionof(s_result, s_oldvalue))
    ///     if change:
    ///         for position in attrfamily.read_locations:
    ///             self.annotator.reflowfromposition(position)
    ///     if isinstance(s_result, SomeImpossibleValue):
    ///         for desc in descs:
    ///             try:
    ///                 attrs = desc.read_attribute('_attrs_')
    ///             except AttributeError:
    ///                 continue
    ///             if isinstance(attrs, Constant):
    ///                 attrs = attrs.value
    ///             if attr in attrs:
    ///                 raise HarmlesslyBlocked("getattr on enforced attr")
    ///     return s_result
    /// ```
    pub fn pbc_getattr(
        self: &Rc<Self>,
        pbc: &SomePBC,
        s_attr: &SomeValue,
    ) -> Result<SomeValue, super::model::AnnotatorException> {
        use super::classdesc::{ClassDesc, ClassDictEntry};
        use super::description::FrozenDesc;
        use super::model::{AnnotatorException, HarmlesslyBlocked, unionof};

        // upstream: `assert s_attr.is_constant(); attr = s_attr.const`.
        let Some(attr_const) = s_attr.const_() else {
            return Err(AnnotatorError::new(format!(
                "pbc_getattr: s_attr must be constant, got {s_attr:?}"
            ))
            .into());
        };
        let attr_name = match attr_const {
            ConstValue::Str(s) => s.clone(),
            other => {
                return Err(AnnotatorError::new(format!(
                    "pbc_getattr: attr must be a string, got {other:?}"
                ))
                .into());
            }
        };

        // upstream: `descs = list(pbc.descriptions); first = descs[0]`.
        let descs: Vec<DescEntry> = pbc.descriptions.values().cloned().collect();
        let first = descs[0].clone();

        // upstream: `if len(descs) == 1: return first.s_read_attribute(attr)`.
        if descs.len() == 1 {
            return Ok(first.s_read_attribute(&attr_name)?);
        }

        // upstream multi-desc path — mergeattrfamilies + getattrfamily +
        // attrfamily bookkeeping. Dispatch by Desc kind, since ClassDesc
        // and FrozenDesc each own their own UnionFind (bookkeeper.py:62-63).
        let (change, pbc_family) = match &first {
            DescEntry::Class(first_cd) => {
                let others_cd: Vec<Rc<RefCell<ClassDesc>>> =
                    descs[1..].iter().filter_map(|d| d.as_class()).collect();
                let change = ClassDesc::mergeattrfamilies(first_cd, &others_cd, &attr_name);
                let family = ClassDesc::getattrfamily(first_cd, &attr_name).ok_or_else(|| {
                    AnnotatorError::new(
                        "pbc_getattr: ClassDesc.getattrfamily unavailable (bookkeeper dropped)",
                    )
                })?;
                (change, PbcAttrFamily::Class(family))
            }
            DescEntry::Frozen(first_fd) => {
                let others_refs: Vec<Rc<RefCell<FrozenDesc>>> =
                    descs[1..].iter().filter_map(|d| d.as_frozen()).collect();
                let borrows: Vec<_> = others_refs.iter().map(|rc| rc.borrow()).collect();
                let others_ptrs: Vec<&FrozenDesc> = borrows.iter().map(|b| &**b).collect();
                let change = first_fd.borrow().mergeattrfamilies(&others_ptrs)?;
                let family = first_fd.borrow().getattrfamily()?;
                (change, PbcAttrFamily::Frozen(family))
            }
            _ => {
                return Err(AnnotatorError::new(format!(
                    "pbc_getattr: multi-desc PBC of kind {:?} has no attrfamily support",
                    first.kind()
                ))
                .into());
            }
        };

        // upstream: `position = self.position_key;
        //            attrfamily.read_locations[position] = True`.
        if let Some(pos) = self.position_key.borrow().clone() {
            pbc_family.add_read_location(pos);
        }

        // upstream: `actuals = [desc.s_read_attribute(attr) for desc in descs]
        //            s_result = unionof(*actuals)`.
        let mut actuals: Vec<SomeValue> = Vec::with_capacity(descs.len());
        for d in &descs {
            actuals.push(d.s_read_attribute(&attr_name)?);
        }
        let s_result = unionof(actuals.iter())
            .map_err(|e| AnnotatorError::new(format!("pbc_getattr unionof failed: {e:?}")))?;

        // upstream: `s_oldvalue = attrfamily.get_s_value(attr);
        //            attrfamily.set_s_value(attr, unionof(s_result, s_oldvalue))`.
        let s_old = pbc_family.get_s_value(&attr_name);
        let merged = unionof([&s_result, &s_old])
            .map_err(|e| AnnotatorError::new(format!("pbc_getattr merge unionof failed: {e:?}")))?;
        pbc_family.set_s_value(&attr_name, merged);

        // upstream: `if change: for position in read_locations: reflowfromposition`.
        if change {
            let positions = pbc_family.read_locations();
            if let Some(ann) = self.annotator.borrow().upgrade() {
                for pos in positions {
                    ann.reflowfromposition(&pos);
                }
            }
        }

        // upstream: HarmlesslyBlocked when s_result is Impossible and
        // attr is in some desc's enforced `_attrs_` set.
        if matches!(s_result, SomeValue::Impossible) {
            for d in &descs {
                let Some(cdesc) = d.as_class() else {
                    continue;
                };
                let attrs_entry = ClassDesc::read_attribute(&cdesc, "_attrs_");
                let Some(ClassDictEntry::Constant(c)) = attrs_entry else {
                    continue;
                };
                let contains = match &c.value {
                    // upstream `_attrs_` is typically a tuple of strings.
                    ConstValue::Tuple(vs) | ConstValue::List(vs) => vs
                        .iter()
                        .any(|v| matches!(v, ConstValue::Str(s) if *s == attr_name)),
                    _ => false,
                };
                if contains {
                    return Err(AnnotatorException::Harmless(HarmlesslyBlocked));
                }
            }
        }

        Ok(s_result)
    }

    /// Python's three-state `emulated` parameter (`None` / `True` /
    /// `<position_key>`) maps onto [`PbcCallEmulated`] in Rust.
    pub fn pbc_call(
        self: &Rc<Self>,
        pbc: &SomePBC,
        args: &super::argument::ArgumentsForTranslation,
        emulated: PbcCallEmulated,
    ) -> Result<SomeValue, AnnotatorError> {
        use super::model::s_impossible_value;
        // upstream bookkeeper.py:516-531 — 3-way branch on `emulated`.
        let (whence, op_key, s_previous_result) = match &emulated {
            PbcCallEmulated::None => {
                // upstream: `whence = self.position_key;
                //            op = block.operations[i];
                //            s_previous_result = annotation(op.result)
                //                                 or s_ImpossibleValue`.
                let pk = self.current_position_key();
                let whence: Option<(
                    crate::flowspace::model::GraphRef,
                    crate::flowspace::model::BlockRef,
                    usize,
                )> = pk.as_ref().and_then(|p| match (p.graph(), p.block()) {
                    (Some(g), Some(b)) => Some((g, b, p.op_index)),
                    _ => None,
                });
                let s_prev = if let Some(pk_ref) = pk.as_ref() {
                    if let (Some(block), Some(ann)) =
                        (pk_ref.block(), self.annotator.borrow().upgrade())
                    {
                        let i = pk_ref.op_index;
                        let block_borrow = block.borrow();
                        if i < block_borrow.operations.len() {
                            let result_var = block_borrow.operations[i].result.clone();
                            ann.annotation(&result_var)
                                .unwrap_or_else(s_impossible_value)
                        } else {
                            s_impossible_value()
                        }
                    } else {
                        s_impossible_value()
                    }
                } else {
                    s_impossible_value()
                };
                (whence, pk, s_prev)
            }
            PbcCallEmulated::True => {
                // upstream: `whence = None; op = None;
                //            s_previous_result = s_ImpossibleValue`.
                (None, None, s_impossible_value())
            }
            PbcCallEmulated::Callback(callback) => {
                // upstream: `whence = emulated; op = None;
                //            s_previous_result = s_ImpossibleValue`.
                let whence = match (callback.graph(), callback.block()) {
                    (Some(g), Some(b)) => Some((g, b, callback.op_index)),
                    _ => None,
                };
                (whence, None, s_impossible_value())
            }
        };

        // upstream: `for desc in pbc.descriptions:
        //             results.append(desc.pycall(whence, args, s_previous_result, op))`.
        let mut results: Vec<SomeValue> = Vec::with_capacity(pbc.descriptions.len());
        for entry in pbc.descriptions.values() {
            let r = match entry {
                super::description::DescEntry::Function(fd) => {
                    fd.borrow()
                        .pycall(whence.clone(), args, &s_previous_result, op_key.clone())?
                }
                super::description::DescEntry::Method(md) => {
                    md.borrow()
                        .pycall(whence.clone(), args, &s_previous_result, op_key.clone())?
                }
                super::description::DescEntry::MethodOfFrozen(mfd) => {
                    mfd.borrow()
                        .pycall(whence.clone(), args, &s_previous_result, op_key.clone())?
                }
                super::description::DescEntry::Class(cd) => super::classdesc::ClassDesc::pycall(
                    cd,
                    whence.clone(),
                    args,
                    &s_previous_result,
                    op_key.clone(),
                )?,
                super::description::DescEntry::Frozen(_) => {
                    return Err(AnnotatorError::new("pbc_call: FrozenDesc is not callable"));
                }
            };
            results.push(r);
        }
        // upstream: `s_result = unionof(*results)`.
        super::model::unionof(results.iter())
            .map_err(|e| AnnotatorError::new(format!("pbc_call unionof: {}", e)))
    }

    /// RPython `Bookkeeper.emulate_pbc_call(self, unique_key, pbc,
    /// args_s, replace=[], callback=None)` (bookkeeper.py:539-572).
    ///
    /// ```python
    /// def emulate_pbc_call(self, unique_key, pbc, args_s,
    ///                      replace=[], callback=None):
    ///     with self.at_position(None):
    ///         emulated_pbc_calls = self.emulated_pbc_calls
    ///         prev = [unique_key]
    ///         prev.extend(replace)
    ///         for other_key in prev:
    ///             if other_key in emulated_pbc_calls:
    ///                 del emulated_pbc_calls[other_key]
    ///         emulated_pbc_calls[unique_key] = pbc, args_s
    ///
    ///         args = simple_args(args_s)
    ///         if callback is None:
    ///             emulated = True
    ///         else:
    ///             emulated = callback
    ///         return self.pbc_call(pbc, args, emulated=emulated)
    /// ```
    pub fn emulate_pbc_call(
        self: &Rc<Self>,
        unique_key: EmulatedPbcCallKey,
        pbc: &SomeValue,
        args_s: &[SomeValue],
        replace: &[EmulatedPbcCallKey],
        callback: Option<PositionKey>,
    ) -> Result<SomeValue, AnnotatorError> {
        let SomeValue::PBC(pbc) = pbc else {
            return Err(AnnotatorError::new(format!(
                "Bookkeeper.emulate_pbc_call expects SomePBC, got {pbc:?}"
            )));
        };
        // upstream: `with self.at_position(None):`
        let _guard = self.at_position(None);
        {
            let mut emulated_map = self.emulated_pbc_calls.borrow_mut();
            // upstream: `prev = [unique_key]; prev.extend(replace);
            //            for other_key in prev:
            //                if other_key in emulated_pbc_calls:
            //                    del emulated_pbc_calls[other_key]`.
            emulated_map.remove(&unique_key);
            for key in replace {
                emulated_map.remove(key);
            }
            // upstream: `emulated_pbc_calls[unique_key] = pbc, args_s`.
            emulated_map.insert(unique_key, (pbc.clone(), args_s.to_vec()));
        }
        // upstream: `args = simple_args(args_s)`.
        let args = simple_args(args_s.to_vec());
        // upstream: `emulated = True if callback is None else callback`.
        let emulated = match callback {
            None => PbcCallEmulated::True,
            Some(cb) => PbcCallEmulated::Callback(cb),
        };
        // upstream: `return self.pbc_call(pbc, args, emulated=emulated)`.
        self.pbc_call(pbc, &args, emulated)
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
    /// The function / class / bound-method / weakref / frozen-PBC /
    /// property / instance branches (bookkeeper.py:218-348) route
    /// through [`Self::immutablevalue_hostobject`]. The
    /// `extregistry.is_registered(x)` branch (bookkeeper.py:312-314)
    /// still surfaces as [`AnnotatorError`].
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
                // upstream bookkeeper.py:255-265:
                //
                //     key = Constant(x)
                //     try: return self.immutable_cache[key]
                //     except KeyError:
                //         result = SomeList(ListDef(self, s_ImpossibleValue))
                //         self.immutable_cache[key] = result
                //         for e in x:
                //             result.listdef.generalize(self.immutablevalue(e))
                //         result.const_box = key
                //         return result
                if let Some(hit) = self.immutable_cache.borrow().get(x).cloned() {
                    return Ok(hit);
                }
                let listdef = ListDef::new(Some(self.clone()), SomeValue::Impossible, false, false);
                let mut result = SomeList::new(listdef.clone());
                result.base.const_box = Some(Constant::new(x.clone()));
                // Upstream stores the cache entry BEFORE populating the
                // listdef so recursive immutablevalue() hits on nested
                // lists re-use the in-flight cell. Matches
                // bookkeeper.py:261.
                self.immutable_cache
                    .borrow_mut()
                    .insert(x.clone(), SomeValue::List(result.clone()));
                for e in items {
                    let s_e = self.immutablevalue(e)?;
                    listdef
                        .generalize(&s_e)
                        .map_err(|e| AnnotatorError::new(e.msg))?;
                }
                Ok(SomeValue::List(result))
            }
            ConstValue::Dict(items) => {
                // upstream bookkeeper.py:266-298:
                //
                //     key = Constant(x)
                //     try: return self.immutable_cache[key]
                //     except KeyError:
                //         result = SomeDict(DictDef(self, ...))
                //         self.immutable_cache[key] = result
                //         for ek, ev in x.iteritems():
                //             result.dictdef.generalize_key(self.immutablevalue(ek))
                //             result.dictdef.generalize_value(self.immutablevalue(ev))
                //         result.const_box = key
                //         return result
                if let Some(hit) = self.immutable_cache.borrow().get(x).cloned() {
                    return Ok(hit);
                }
                let dictdef = DictDef::new(
                    Some(self.clone()),
                    SomeValue::Impossible,
                    SomeValue::Impossible,
                    false,
                    false,
                    false,
                );
                let mut result = SomeDict::new(dictdef.clone());
                result.base.const_box = Some(Constant::new(x.clone()));
                self.immutable_cache
                    .borrow_mut()
                    .insert(x.clone(), SomeValue::Dict(result.clone()));
                for (k, v) in items {
                    let s_k = self.immutablevalue(k)?;
                    let s_v = self.immutablevalue(v)?;
                    dictdef
                        .generalize_key(&s_k)
                        .map_err(|e| AnnotatorError::new(e.msg))?;
                    dictdef
                        .generalize_value(&s_v)
                        .map_err(|e| AnnotatorError::new(e.msg))?;
                }
                Ok(SomeValue::Dict(result))
            }
            ConstValue::HostObject(obj) => self.immutablevalue_hostobject(obj, x),
            ConstValue::Function(func) => {
                let host = HostObject::new_user_function((**func).clone());
                self.immutablevalue_hostobject(&host, &ConstValue::HostObject(host.clone()))
            }
            ConstValue::Code(_)
            | ConstValue::Graphs(_)
            | ConstValue::LLPtr(_)
            | ConstValue::SpecTag(_)
            | ConstValue::Atom(_)
            | ConstValue::Placeholder => {
                // Code / LLPtr / SpecTag / Atom / Placeholder cover
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
    /// [`Self::immutablevalue`]. Mirrors upstream
    /// `bookkeeper.py:309-333` — builtin-analyser / callable / class /
    /// frozen fallbacks.
    ///
    /// The user-function and class branches now route through
    /// [`Self::getdesc`], which produces the real
    /// [`DescEntry::Function`] / [`DescEntry::Class`] wrapping the
    /// shared `Rc<RefCell<…>>` instance from `self.descs`. The
    /// resulting [`SomePBC`] has `descriptions` populated with actual
    /// Desc objects (fix for reviewer pre-existing #1: SomePBC
    /// descriptions were just `kind+name` stubs).
    fn immutablevalue_hostobject(
        self: &Rc<Self>,
        obj: &HostObject,
        raw: &ConstValue,
    ) -> Result<SomeValue, AnnotatorError> {
        // upstream bookkeeper.py:299-306 — `tp is weakref.ReferenceType`:
        //   x1 = x()
        //   if x1 is None:
        //       result = SomeWeakRef(None)    # dead weakref
        //   else:
        //       s1 = self.immutablevalue(x1)
        //       assert isinstance(s1, SomeInstance)
        //       result = SomeWeakRef(s1.classdef)
        //
        // Upstream checks this FIRST (right after the dict cache
        // path), before property/callable/class — weakref objects
        // are themselves callable, so the order matters on upstream.
        // `HostObjectKind` variants are disjoint in the Rust port,
        // so the branches below are observationally order-
        // independent; the sequence here mirrors upstream anyway.
        if obj.is_weakref() {
            let mut wr = match obj.weakref_referent().flatten() {
                // dead weakref.
                None => super::model::SomeWeakRef::new(None),
                Some(referent) => {
                    let s1 = self.immutablevalue(&ConstValue::HostObject(referent.clone()))?;
                    let SomeValue::Instance(inst) = s1 else {
                        return Err(AnnotatorError::new(format!(
                            "Bookkeeper.immutablevalue(weakref): target {referent:?} does not \
                             annotate as SomeInstance (upstream asserts isinstance(s1, SomeInstance))"
                        )));
                    };
                    super::model::SomeWeakRef::new(inst.classdef.clone())
                }
            };
            wr.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::WeakRef(wr));
        }
        // upstream bookkeeper.py:307-308 — `elif tp is property: return SomeProperty(x)`.
        if obj.is_property() {
            return Ok(SomeValue::Property(super::model::SomeProperty::new(obj)));
        }
        // upstream bookkeeper.py:309-311 — `elif ishashable(x) and x
        // in BUILTIN_ANALYZERS: result = SomeBuiltin(...)`. The Rust
        // port routes two kinds of host objects into `SomeBuiltin`:
        //
        //   1. `HostObjectKind::BuiltinCallable` — every builtin
        //      function gets a `SomeBuiltin`, whether or not
        //      `builtin.rs` registered an analyser for it, because
        //      `policy.rs` sandboxing dispatch inspects
        //      `SomeValue::Builtin` directly. Missing analysers surface
        //      at call time via `call_builtin`'s "no analyser
        //      registered" error.
        //   2. `HostObjectKind::Class` whose qualname is registered in
        //      `BUILTIN_ANALYZERS` (e.g. `range`, `int`, `float`,
        //      `bool`, `list`, `tuple`, `enumerate`, `reversed`,
        //      `bytearray`). Upstream Python models these as
        //      `type`-valued callables, and `bookkeeper.py:309-311`'s
        //      membership test fires before `bookkeeper.py:315-316`'s
        //      `tp is type` branch. Route them to `SomeBuiltin` to
        //      match.
        let is_registered_class = obj.is_class() && super::builtin::is_registered(obj.qualname());
        if obj.is_builtin_callable() || is_registered_class {
            let mut sb = SomeBuiltin::new(
                obj.qualname().to_string(),
                None,
                Some(obj.qualname().to_string()),
            );
            sb.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::Builtin(sb));
        }
        // upstream bookkeeper.py:312-314 — `elif extregistry.is_registered(x)` —
        // deferred until the Rust port grows an extregistry bridge.
        // upstream bookkeeper.py:315-316 — `elif tp is type: result =
        // SomeConstantType(x, self)`. Implemented as a constant
        // SomePBC over the real [`ClassDesc`] returned by [`Self::getdesc`].
        if obj.is_class() {
            let entry = self.getdesc(obj)?;
            let mut pbc = SomePBC::new(vec![entry], false);
            pbc.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::PBC(pbc));
        }
        // upstream bookkeeper.py:317-331 — `elif callable(x):` splits
        // into im_self/__self__ bound-method routing (→ find_method on
        // s_self) and the plain-callable fallback (→
        // SomePBC([getdesc(x)])). The Rust port encodes these as two
        // disjoint `HostObjectKind` variants that appear in the same
        // position as upstream's single `callable(x)` branch.
        if obj.is_bound_method() {
            let self_obj = obj.bound_method_self().ok_or_else(|| {
                AnnotatorError::new("Bookkeeper.immutablevalue(bound method): missing __self__")
            })?;
            let s_self = self.immutablevalue(&ConstValue::HostObject(self_obj.clone()))?;
            if let Some(name) = obj.bound_method_name()
                && let Some(result) = s_self.find_method(name)
            {
                return Ok(result);
            }
            let entry = self.getdesc(obj)?;
            let mut pbc = SomePBC::new(vec![entry], false);
            pbc.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::PBC(pbc));
        }
        if obj.is_user_function() {
            let entry = self.getdesc(obj)?;
            let mut pbc = SomePBC::new(vec![entry], false);
            // Keep explicit const_box for parity with upstream
            // `self.const = x` (bookkeeper.py's SomePBC callers pin
            // const with the raw HostObject); SomePBC::new's single-
            // desc hack already sets this to an equivalent value but
            // we retain the write for clarity when `raw != pyobj`.
            pbc.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::PBC(pbc));
        }
        // upstream bookkeeper.py:332-338 — `elif hasattr(x, '_freeze_'):
        // assert x._freeze_() is True; result = SomePBC([getdesc(x)])`.
        // Check BEFORE the instance branch so any HostObject carrying
        // a `_freeze_` member routes to the frozen-PBC path regardless
        // of its kind (Instance / Module / Opaque all surface their
        // attribute dict the same way).
        if call_freeze_method(obj)? {
            let entry = self.getdesc(obj)?;
            let mut pbc = SomePBC::new(vec![entry], false);
            pbc.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::PBC(pbc));
        }
        // upstream bookkeeper.py:339-345 — `hasattr(x, '__class__') and
        // x.__class__.__module__ != '__builtin__'` →
        // `getuniqueclassdef(x.__class__) + see_instance + SomeInstance`.
        if let Some(class_obj) = obj.instance_class() {
            // upstream bookkeeper.py:341-342:
            //     if hasattr(x, '_cleanup_'): x._cleanup_()
            call_cleanup_method(obj)?;
            let classdef = self.getuniqueclassdef(class_obj)?;
            super::classdesc::ClassDef::see_instance(&classdef, obj)?;
            let mut inst = super::model::SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            );
            inst.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::Instance(inst));
        }
        // upstream bookkeeper.py:349 — final `raise AnnotatorError("Don't
        // know how to represent %r" % (x,))`. Module / Opaque / weakref
        // / extregistry HostObject kinds without `_freeze_` land here
        // because upstream also raises for them.
        Err(AnnotatorError::new(format!(
            "Don't know how to represent {raw:?}"
        )))
    }
}

impl Default for Bookkeeper {
    fn default() -> Self {
        Self::new()
    }
}

/// RPython `hasattr(x, '_freeze_'); assert x._freeze_() is True`
/// (bookkeeper.py:332-333).
///
/// Returns `Ok(true)` when `_freeze_` is present and returns `True`;
/// `Ok(false)` when `_freeze_` is absent (caller falls through to the
/// instance / fallback branch); `Err` when `_freeze_` is present but
/// the invocation fails or returns something other than `True`
/// (mirrors the Python `assert x._freeze_() is True` AssertionError).
///
/// Upstream RPython runs the `_freeze_` body through real Python. The
/// Rust port routes through [`HostObject::call_host`], which handles
/// `NativeCallable` (test fixtures / bootstrap builtins register
/// these) directly and returns
/// [`HostCallError::RequiresFlowEvaluator`] for a `UserFunction`
/// body. The latter surfaces as an explicit
/// "Phase 6 host-graph evaluator not yet landed" `AnnotatorError`
/// rather than a generic call failure — this keeps the boundary
/// between host-executable and flowspace-required callables
/// discoverable at the callsite.
fn call_freeze_method(obj: &HostObject) -> Result<bool, AnnotatorError> {
    let method = match lookup_zero_arg_method(obj, "_freeze_")? {
        Some(m) => m,
        None => return Ok(false),
    };
    let result = method.call_host(&[]).map_err(|err| {
        AnnotatorError::new(match err {
            crate::flowspace::model::HostCallError::RequiresFlowEvaluator(q) => format!(
                "_freeze_() on {q:?} requires the Phase 6 host-graph evaluator \
                 (user function body cannot be executed host-side yet)"
            ),
            other => format!("_freeze_() call failed: {other}"),
        })
    })?;
    match result {
        ConstValue::Bool(true) => Ok(true),
        other => Err(AnnotatorError::new(format!(
            "_freeze_() must return True, got {other:?}"
        ))),
    }
}

/// RPython `if hasattr(x, '_cleanup_'): x._cleanup_()` (bookkeeper.py:341-342).
///
/// Fires the cleanup method for side-effects and ignores the return
/// value. Absence of `_cleanup_` is silently accepted. A `UserFunction`
/// body surfaces the same `Phase 6 host-graph evaluator` marker as
/// `call_freeze_method` so the two boundary limitations are visible
/// at a single grep point.
fn call_cleanup_method(obj: &HostObject) -> Result<(), AnnotatorError> {
    let Some(method) = lookup_zero_arg_method(obj, "_cleanup_")? else {
        return Ok(());
    };
    method.call_host(&[]).map(|_| ()).map_err(|err| {
        AnnotatorError::new(match err {
            crate::flowspace::model::HostCallError::RequiresFlowEvaluator(q) => format!(
                "_cleanup_() on {q:?} requires the Phase 6 host-graph evaluator \
                 (user function body cannot be executed host-side yet)"
            ),
            other => format!("_cleanup_() call failed: {other}"),
        })
    })
}

/// Helper: resolve a zero-arg method `name` on `obj`, binding `self`
/// where applicable (Instance / Module / Class host kinds).
///
/// Returns `Some(callable)` when the name is present and resolves to
/// a callable HostObject (BoundMethod or NativeCallable); `None`
/// otherwise. Non-callable attributes that would raise `TypeError`
/// at call time are deliberately returned as `Some` so
/// [`HostObject::call_host`] can surface the failure with the
/// upstream shape.
fn lookup_zero_arg_method(
    obj: &HostObject,
    name: &str,
) -> Result<Option<HostObject>, AnnotatorError> {
    use crate::flowspace::model::host_getattr;
    match host_getattr(obj, name) {
        Ok(ConstValue::HostObject(h)) => Ok(Some(h)),
        Ok(other) => Err(AnnotatorError::new(format!(
            "{} exists on {:?} but is not callable: {other:?}",
            name,
            obj.qualname()
        ))),
        Err(crate::flowspace::model::HostGetAttrError::Missing) => Ok(None),
        Err(crate::flowspace::model::HostGetAttrError::Unsupported) => Err(AnnotatorError::new(
            format!("{} lookup on {:?} is unsupported", name, obj.qualname()),
        )),
    }
}

/// RPython `CallOp.build_args(args_s)` dispatch by opname
/// (operation.py:678-679 for `simple_call`, 699-701 for `call_args`).
///
/// `simple_call` wraps `args_s` in a flat `ArgumentsForTranslation`;
/// `call_args` reads the encoded call shape out of `args_s[0].const`
/// (`ConstValue::Tuple([Int(cnt), Tuple([Str(k0), …]), Bool(star)])`,
/// see `flowcontext::build_call_shape_constant`) and reconstructs a
/// CallShape + tail `args_s[1..]`.
// =====================================================================
// check_no_flags_on_instances walker (bookkeeper.py:124-147)
// =====================================================================

/// Entry point for the recursive sanity walk — inspects a `ClassDef`
/// and recurses through its attributes. `seen_classdefs` /
/// `seen_listitems` break cycles for attrs whose annotation cycles back.
fn check_no_flags_classdef(
    clsdef: &Rc<RefCell<ClassDef>>,
    seen_classdefs: &mut HashSet<ClassDefKey>,
    seen_listitems: &mut HashSet<usize>,
) {
    let key = ClassDefKey::from_classdef(clsdef);
    if !seen_classdefs.insert(key) {
        return;
    }
    let attrs_snapshot: Vec<SomeValue> = clsdef
        .borrow()
        .attrs
        .values()
        .map(|a| a.s_value.clone())
        .collect();
    for s_attr in attrs_snapshot {
        check_no_flags_value(&s_attr, seen_classdefs, seen_listitems);
    }
}

/// upstream `check_no_flags(SomeInstance | SomeList | SomeDict |
/// SomeTuple)` arms; recurses into container element types.
fn check_no_flags_value(
    s: &SomeValue,
    seen_classdefs: &mut HashSet<ClassDefKey>,
    seen_listitems: &mut HashSet<usize>,
) {
    match s {
        SomeValue::Instance(inst) => {
            assert!(
                inst.flags.is_empty(),
                "instance annotation with flags escaped to the heap"
            );
            if let Some(classdef) = &inst.classdef {
                check_no_flags_classdef(classdef, seen_classdefs, seen_listitems);
            }
        }
        SomeValue::List(list) => {
            let listitem = list.listdef.listitem_rc();
            check_no_flags_listitem(&listitem, seen_classdefs, seen_listitems);
        }
        SomeValue::Dict(dict) => {
            let dictkey = dict.dictdef.dictkey_rc();
            check_no_flags_listitem(&dictkey, seen_classdefs, seen_listitems);
            let dictvalue = dict.dictdef.dictvalue_rc();
            check_no_flags_listitem(&dictvalue, seen_classdefs, seen_listitems);
        }
        SomeValue::Tuple(tup) => {
            for item in &tup.items {
                check_no_flags_value(item, seen_classdefs, seen_listitems);
            }
        }
        _ => {}
    }
}

fn check_no_flags_listitem(
    li: &Rc<RefCell<super::listdef::ListItem>>,
    seen_classdefs: &mut HashSet<ClassDefKey>,
    seen_listitems: &mut HashSet<usize>,
) {
    let id = Rc::as_ptr(li) as usize;
    if !seen_listitems.insert(id) {
        return;
    }
    let s_value = li.borrow().s_value.clone();
    check_no_flags_value(&s_value, seen_classdefs, seen_listitems);
}

fn build_args_for_op(
    opname: &str,
    args_s: &[SomeValue],
) -> Result<ArgumentsForTranslation, AnnotatorError> {
    match opname {
        "simple_call" => Ok(simple_args(args_s.to_vec())),
        "call_args" => {
            if args_s.is_empty() {
                return Err(AnnotatorError::new(
                    "build_args_for_op(call_args): missing shape argument",
                ));
            }
            let shape_const = args_s[0].const_().ok_or_else(|| {
                AnnotatorError::new("build_args_for_op(call_args): args_s[0] is not a Constant")
            })?;
            let shape = call_shape_from_const(shape_const)?;
            Ok(complex_args(&shape, args_s[1..].to_vec()))
        }
        other => Err(AnnotatorError::new(format!(
            "build_args_for_op: unsupported call opname {other:?}"
        ))),
    }
}

/// Decode the `CallShape` tuple embedded in `call_args` operations.
/// Mirrors the encoding produced by
/// `flowcontext::build_call_shape_constant`:
/// `Tuple([Int(shape_cnt), Tuple([Str(key)*]), Bool(shape_star)])`.
fn call_shape_from_const(cv: &ConstValue) -> Result<CallShape, AnnotatorError> {
    let items = match cv {
        ConstValue::Tuple(items) => items,
        _ => {
            return Err(AnnotatorError::new(
                "call_shape_from_const: expected ConstValue::Tuple",
            ));
        }
    };
    if items.len() != 3 {
        return Err(AnnotatorError::new(
            "call_shape_from_const: tuple must have 3 elements",
        ));
    }
    let shape_cnt = match &items[0] {
        ConstValue::Int(n) => *n as usize,
        _ => {
            return Err(AnnotatorError::new(
                "call_shape_from_const: shape_cnt is not Int",
            ));
        }
    };
    let keys_tuple = match &items[1] {
        ConstValue::Tuple(k) => k,
        _ => {
            return Err(AnnotatorError::new(
                "call_shape_from_const: shape_keys is not Tuple",
            ));
        }
    };
    let mut shape_keys = Vec::with_capacity(keys_tuple.len());
    for k in keys_tuple {
        match k {
            ConstValue::Str(s) => shape_keys.push(s.clone()),
            _ => {
                return Err(AnnotatorError::new(
                    "call_shape_from_const: shape_keys element is not Str",
                ));
            }
        }
    }
    let shape_star = match &items[2] {
        ConstValue::Bool(b) => *b,
        _ => {
            return Err(AnnotatorError::new(
                "call_shape_from_const: shape_star is not Bool",
            ));
        }
    };
    Ok(CallShape {
        shape_cnt,
        shape_keys,
        shape_star,
    })
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
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::model::{SomeByteArray, SomeChar, SomeFloat};
    use crate::flowspace::model::GraphFunc;

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    #[test]
    fn check_no_flags_on_instances_accepts_empty_flags() {
        use crate::annotator::classdesc::{ClassDef, ClassDesc};
        let bk = bk();
        let pyobj = crate::flowspace::model::HostObject::new_class("pkg.A", vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            pyobj,
            "pkg.A".into(),
        )));
        let classdef = ClassDef::new(&bk, &desc);
        bk.classdefs.borrow_mut().push(classdef);
        // No attrs, no SomeInstance carries flags — walker must not panic.
        bk.check_no_flags_on_instances();
    }

    #[test]
    #[should_panic(expected = "instance annotation with flags escaped to the heap")]
    fn check_no_flags_on_instances_panics_on_non_empty_flags() {
        use crate::annotator::classdesc::{ClassDef, ClassDesc};
        use crate::annotator::model::SomeInstance;
        let bk = bk();
        let pyobj = crate::flowspace::model::HostObject::new_class("pkg.A", vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            pyobj,
            "pkg.A".into(),
        )));
        let classdef = ClassDef::new(&bk, &desc);
        // Plant an attribute whose value is a SomeInstance with
        // non-empty flags — upstream's assertion must fire.
        let mut flags = std::collections::BTreeMap::new();
        flags.insert("nonneg".to_string(), true);
        let bad = SomeValue::Instance(SomeInstance::new(Some(classdef.clone()), false, flags));
        let mut attrs_attr = crate::annotator::classdesc::Attribute::new("x");
        attrs_attr.s_value = bad;
        classdef
            .borrow_mut()
            .attrs
            .insert("x".to_string(), attrs_attr);
        bk.classdefs.borrow_mut().push(classdef);
        bk.check_no_flags_on_instances();
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
                    pbc.descriptions.values().next().unwrap().kind(),
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
                    pbc.descriptions.values().next().unwrap().kind(),
                    DescKind::Function
                );
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_const_function_returns_function_pbc() {
        use crate::annotator::model::{DescKind, SomeValue};
        use crate::flowspace::model::{Constant, GraphFunc};

        let bk = bk();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func = GraphFunc::new("f", globals);
        let s = bk
            .immutablevalue(&ConstValue::Function(Box::new(func)))
            .expect("ConstValue::Function must route to function SomePBC");

        match s {
            SomeValue::PBC(pbc) => {
                assert_eq!(pbc.descriptions.len(), 1);
                assert_eq!(
                    pbc.descriptions.values().next().unwrap().kind(),
                    DescKind::Function
                );
                assert!(pbc.base.const_box.is_some());
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
    fn immutablevalue_property_returns_someproperty() {
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::{Constant, GraphFunc, HostObject};

        let bk = bk();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let fget = HostObject::new_user_function(GraphFunc::new("fget", globals));
        let prop = HostObject::new_property("pkg.C.x", Some(fget.clone()), None, None);

        let s = bk
            .immutablevalue(&ConstValue::HostObject(prop))
            .expect("property HostObject must produce SomeProperty");

        match s {
            SomeValue::Property(prop) => assert_eq!(prop.fget, Some(fget)),
            other => panic!("expected SomeProperty, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_bound_method_returns_method_pbc() {
        use crate::annotator::model::{DescKind, SomeValue};
        use crate::flowspace::model::{Constant, GraphFunc, HostObject};

        let bk = bk();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func = HostObject::new_user_function(GraphFunc::new("m", globals));
        let cls = HostObject::new_class("pkg.C", vec![]);
        cls.class_set("m", ConstValue::HostObject(func.clone()));
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let meth = HostObject::new_bound_method("pkg.C.m", inst, func, "m", cls);

        let s = bk
            .immutablevalue(&ConstValue::HostObject(meth))
            .expect("bound method HostObject must produce SomePBC");

        match s {
            SomeValue::PBC(pbc) => assert_eq!(pbc.get_kind().unwrap(), DescKind::Method),
            other => panic!("expected method SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_dead_weakref_returns_none_classdef() {
        // upstream bookkeeper.py:300-302 — `x1 = x(); if x1 is None:
        // result = SomeWeakRef(None)`.
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let dead = HostObject::new_weakref("weakref.ref", None);
        let s = bk
            .immutablevalue(&ConstValue::HostObject(dead))
            .expect("dead weakref must route to SomeWeakRef");
        match s {
            SomeValue::WeakRef(wr) => assert!(wr.classdef.is_none()),
            other => panic!("expected SomeWeakRef(None), got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_live_weakref_pulls_instance_classdef() {
        // upstream bookkeeper.py:303-306 — `s1 = immutablevalue(x1);
        // assert isinstance(s1, SomeInstance); result = SomeWeakRef(s1.classdef)`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.Target", vec![]);
        let inst = HostObject::new_instance(cls, vec![]);
        let wref = HostObject::new_weakref("weakref.ref", Some(inst));
        let s = bk
            .immutablevalue(&ConstValue::HostObject(wref))
            .expect("live weakref must route to SomeWeakRef");
        match s {
            SomeValue::WeakRef(wr) => assert!(wr.classdef.is_some()),
            other => panic!("expected SomeWeakRef(Some(classdef)), got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_module_without_freeze_raises() {
        // upstream bookkeeper.py:349 — a raw Module hits the final
        // `raise AnnotatorError("Don't know how to represent %r")`
        // fallthrough because `module.__class__.__module__ ==
        // '__builtin__'` and it lacks `_freeze_`.
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let mod_obj = HostObject::new_module("m");
        let err = bk
            .immutablevalue(&ConstValue::HostObject(mod_obj))
            .expect_err("raw module must raise AnnotatorError");
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("Don't know how to represent")
        );
    }

    #[test]
    fn immutablevalue_module_with_freeze_routes_to_some_pbc() {
        // upstream bookkeeper.py:332-338 — any host with `_freeze_`
        // routes through SomePBC([getdesc(x)]). Upstream first checks
        // `hasattr(x, '_freeze_')` then asserts `x._freeze_() is True`,
        // so the Rust port registers `_freeze_` as a NativeCallable
        // that returns `Bool(true)`.
        use crate::flowspace::model::HostObject;
        use std::sync::Arc;
        let bk = bk();
        let mod_obj = HostObject::new_module("m");
        mod_obj.module_set(
            "_freeze_",
            HostObject::new_native_callable(
                "m._freeze_",
                Arc::new(|_args| Ok(ConstValue::Bool(true))),
            ),
        );
        let s = bk
            .immutablevalue(&ConstValue::HostObject(mod_obj))
            .expect("module with _freeze_ must route to SomePBC");
        assert!(matches!(s, SomeValue::PBC(_)));
    }

    #[test]
    fn immutablevalue_user_class_instance_routes_to_some_instance() {
        // upstream bookkeeper.py:339-345 — user-class instance
        // (non-`_freeze_`, non-builtin class) routes to
        // `SomeInstance(getuniqueclassdef(x.__class__))` with the
        // instance recorded via `classdef.see_instance(x)`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.Holder", vec![]);
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        inst.instance_set("x", ConstValue::Int(42));
        let s = bk
            .immutablevalue(&ConstValue::HostObject(inst.clone()))
            .expect("user-class instance must route to SomeInstance");
        match s {
            SomeValue::Instance(si) => {
                assert!(si.classdef.is_some(), "classdef must be attached");
                // `see_instance` records the instance in classdef.instances_seen.
                let classdef = si.classdef.unwrap();
                assert!(
                    classdef
                        .borrow()
                        .instances_seen
                        .contains(&inst.identity_id())
                );
            }
            other => panic!("expected SomeInstance, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_freeze_instance_routes_to_some_pbc() {
        // upstream bookkeeper.py:332-338 — `hasattr(x, '_freeze_')` +
        // `assert x._freeze_() is True` routes to
        // `SomePBC([getdesc(x)])`. The Rust port registers `_freeze_`
        // as a NativeCallable on the class dict so the descriptor
        // protocol binds `self` and `call_host` executes it.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        use std::sync::Arc;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.Frozen", vec![]);
        cls.class_set(
            "_freeze_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.Frozen._freeze_",
                Arc::new(|_args| Ok(ConstValue::Bool(true))),
            )),
        );
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let s = bk
            .immutablevalue(&ConstValue::HostObject(inst))
            .expect("_freeze_ instance must route to SomePBC");
        match s {
            SomeValue::PBC(pbc) => {
                assert_eq!(pbc.descriptions.len(), 1);
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_freeze_returning_false_raises_annotator_error() {
        // upstream bookkeeper.py:333 `assert x._freeze_() is True` —
        // a `_freeze_` returning False triggers AssertionError. The
        // Rust port surfaces this as an `AnnotatorError`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        use std::sync::Arc;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.NotFrozen", vec![]);
        cls.class_set(
            "_freeze_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.NotFrozen._freeze_",
                Arc::new(|_args| Ok(ConstValue::Bool(false))),
            )),
        );
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let err = bk
            .immutablevalue(&ConstValue::HostObject(inst))
            .expect_err("_freeze_ returning False must raise");
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("_freeze_() must return True")
        );
    }

    #[test]
    fn immutablevalue_non_callable_freeze_raises_annotator_error() {
        // upstream bookkeeper.py:332-333 performs `x._freeze_()`;
        // if `_freeze_` exists but is not callable, Python raises
        // `TypeError`. The Rust port surfaces that as an AnnotatorError.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.BadFrozen", vec![]);
        cls.class_set("_freeze_", ConstValue::Bool(true));
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let err = bk
            .immutablevalue(&ConstValue::HostObject(inst))
            .expect_err("non-callable _freeze_ must raise");
        assert!(err.msg.unwrap_or_default().contains("_freeze_ exists on"));
    }

    #[test]
    fn immutablevalue_runs_cleanup_on_instance() {
        // upstream bookkeeper.py:341-342:
        //     if hasattr(x, '_cleanup_'): x._cleanup_()
        // The Rust port invokes `_cleanup_` for side effects. We verify
        // the call happens via an AtomicBool (HostCallableFn requires
        // `Send + Sync`, so a plain `Cell` is not shareable into the
        // closure).
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.Cleaned", vec![]);
        let called = Arc::new(AtomicBool::new(false));
        let called_cl = called.clone();
        cls.class_set(
            "_cleanup_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.Cleaned._cleanup_",
                Arc::new(move |_args| {
                    called_cl.store(true, Ordering::SeqCst);
                    Ok(ConstValue::None)
                }),
            )),
        );
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let _ = bk
            .immutablevalue(&ConstValue::HostObject(inst))
            .expect("instance with _cleanup_ must not fail");
        assert!(
            called.load(Ordering::SeqCst),
            "_cleanup_ must have been invoked"
        );
    }

    #[test]
    fn getdesc_bound_method_of_frozen_instance_runs_cleanup_and_returns_method_of_frozen() {
        use crate::annotator::model::DescKind;
        use crate::flowspace::model::{Constant, GraphFunc, HostObject};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func = HostObject::new_user_function(GraphFunc::new("m", globals));
        let cls = HostObject::new_class("pkg.Frozen", vec![]);
        cls.class_set("m", ConstValue::HostObject(func.clone()));
        let cleaned = Arc::new(AtomicBool::new(false));
        let cleaned_cl = cleaned.clone();
        cls.class_set(
            "_cleanup_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.Frozen._cleanup_",
                Arc::new(move |_args| {
                    cleaned_cl.store(true, Ordering::SeqCst);
                    Ok(ConstValue::None)
                }),
            )),
        );
        cls.class_set(
            "_freeze_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.Frozen._freeze_",
                Arc::new(|_args| Ok(ConstValue::Bool(true))),
            )),
        );
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let meth = HostObject::new_bound_method("pkg.Frozen.m", inst, func, "m", cls);

        let entry = bk.getdesc(&meth).expect("frozen bound method desc");

        assert_eq!(entry.kind(), DescKind::MethodOfFrozen);
        assert!(
            cleaned.load(Ordering::SeqCst),
            "_cleanup_ must run before frozen bound-method classification"
        );
    }

    #[test]
    fn getdesc_bound_method_freeze_returning_false_raises() {
        use crate::flowspace::model::{Constant, GraphFunc, HostObject};
        use std::sync::Arc;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func = HostObject::new_user_function(GraphFunc::new("m", globals));
        let cls = HostObject::new_class("pkg.NotFrozen", vec![]);
        cls.class_set("m", ConstValue::HostObject(func.clone()));
        cls.class_set(
            "_freeze_",
            ConstValue::HostObject(HostObject::new_native_callable(
                "pkg.NotFrozen._freeze_",
                Arc::new(|_args| Ok(ConstValue::Bool(false))),
            )),
        );
        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let meth = HostObject::new_bound_method("pkg.NotFrozen.m", inst, func, "m", cls);

        let err = bk
            .getdesc(&meth)
            .expect_err("_freeze_ returning False must raise");

        assert!(
            err.msg
                .unwrap_or_default()
                .contains("_freeze_() must return True")
        );
    }

    #[test]
    fn newlist_creates_somelist_and_generalizes() {
        // Two SomeInteger variants so the `Int ∪ Int` pair-union
        // widens them (signed `∪` nonneg → signed).
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

    // --- classdesc: descs / classdefs / getdesc ---

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

    #[test]
    fn annotator_backlink_upgrades_when_alive() {
        // bookkeeper.py:52-54 — `self.annotator = annotator`. In Rust
        // the backlink is a Weak<RPythonAnnotator>; while the outer
        // `Rc<RPythonAnnotator>` is alive, `bookkeeper.annotator()`
        // must return a live Rc to the same driver.
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let got = ann.bookkeeper.annotator();
        assert!(Rc::ptr_eq(&got, &ann));
    }

    #[test]
    #[should_panic(expected = "Bookkeeper.annotator backlink is absent or dropped")]
    fn annotator_backlink_panics_when_unwired() {
        // A bare Bookkeeper::new() (test-only constructor) leaves the
        // annotator slot empty; any attempt to upgrade panics.
        let bk = Bookkeeper::new();
        let _ = bk.annotator();
    }

    #[test]
    fn compute_at_fixpoint_enters_position_and_clears_emulated() {
        // bookkeeper.py:108-118 — compute_at_fixpoint wraps the work
        // in `with self.at_position(None)` and clears
        // `emulated_pbc_calls` at the end. Both effects are
        // observable even when the inner loops find no work.
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        // Before: no position entered; emulated_pbc_calls starts empty.
        assert!(!ann.bookkeeper.position_entered.get());
        assert!(ann.bookkeeper.emulated_pbc_calls.borrow().is_empty());

        // Run — should not panic, should leave position_entered false
        // (at_position guard exits via Drop).
        ann.bookkeeper.compute_at_fixpoint();
        assert!(!ann.bookkeeper.position_entered.get());
        assert!(ann.bookkeeper.emulated_pbc_calls.borrow().is_empty());
    }
}
