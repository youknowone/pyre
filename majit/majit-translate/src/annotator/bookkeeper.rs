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
//!   branch (bookkeeper.py:312-314) is wired in
//!   [`Self::immutablevalue_hostobject`] via
//!   [`crate::translator::rtyper::extregistry::is_registered`]; per-entry
//!   value-level coverage extends as registrations land.
//! * `BUILTIN_ANALYZERS` registry — ported in [`super::builtin`].
//!   `SomeBuiltin.analyser_name` is seeded from the host qualname;
//!   `SomeValue::call()` for `Builtin(_)` dispatches through
//!   [`super::builtin::call_builtin`]. Only registered builtins route
//!   through `SomeBuiltin`, matching upstream
//!   `bookkeeper.py:309-311`.
//! * `classpbc_attr_families` / `all_specializations` — not yet
//!   mirrored on the Rust bookkeeper; reachable only from rtyper-phase
//!   consumers.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};

use super::argument::{ArgumentsForTranslation, simple_args};
use super::classdesc::{ClassDef, ClassDesc};
use super::description::{
    CallFamily, ClassAttrFamily, ClassDefKey, DescEntry, DescKey, FrozenAttrFamily, FrozenDesc,
    FunctionDesc, MethodDesc,
};
use super::dictdef::DictDef;
use super::listdef::ListDef;
use super::model::{
    AnnotatorError, SomeBool, SomeBuiltin, SomeChar, SomeDict, SomeFloat, SomeInteger, SomeList,
    SomePBC, SomeString, SomeTuple, SomeUnicodeCodePoint, SomeUnicodeString, SomeValue,
    s_impossible_value, s_none, union,
};
use super::policy::AnnotatorPolicy;
use crate::flowspace::argument::CallShape;
use crate::flowspace::argument::Signature;
use crate::flowspace::bytecode::cpython_code_signature;
use crate::flowspace::model::{BlockRef, ConstValue, Constant, GraphKey, GraphRef, HostObject};
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
    pub(crate) graph_id: usize,
    /// Identity hash of the enclosing `Block` — upstream
    /// `position_key[1]`.
    pub(crate) block_id: usize,
    /// Operation index inside the block — upstream `position_key[2]`.
    pub(crate) op_index: usize,
    /// Weak reference to the enclosing `FunctionGraph`. Populated by
    /// [`Self::from_refs`]; `None` for test-only synthetic keys.
    pub(crate) graph_ref:
        Option<std::rc::Weak<std::cell::RefCell<crate::flowspace::model::FunctionGraph>>>,
    /// Weak reference to the enclosing `Block`. Populated by
    /// [`Self::from_refs`]; `None` for test-only synthetic keys.
    pub(crate) block_ref: Option<std::rc::Weak<std::cell::RefCell<crate::flowspace::model::Block>>>,
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

/// Upstream `analyzer_for(func)` decorator (bookkeeper.py:34-38).
pub fn analyzer_for(
    reg: &mut HashMap<String, super::builtin::BuiltinAnalyzer>,
    qualname: &str,
    analyser: super::builtin::BuiltinAnalyzer,
) {
    if reg.insert(qualname.to_string(), analyser).is_some() {
        panic!("bookkeeper.rs: duplicate BUILTIN_ANALYZERS entry for {qualname}");
    }
}

impl PositionKey {
    /// Synthetic constructor — test-only. Fills the identity triple
    /// directly and leaves the Weak refs empty.
    #[cfg(test)]
    pub(crate) fn new(graph_id: usize, block_id: usize, op_index: usize) -> Self {
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
    pub(crate) fn from_refs(graph: &GraphRef, block: &BlockRef, op_index: usize) -> Self {
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
    pub(crate) fn graph(&self) -> Option<GraphRef> {
        self.graph_ref.as_ref().and_then(|w| w.upgrade())
    }

    /// Upgrade the weak block reference. Mirrors upstream's `_, block,
    /// _ = position_key` unpack.
    pub(crate) fn block(&self) -> Option<BlockRef> {
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
    pub(crate) position_key: RefCell<Option<PositionKey>>,
    /// RPython `self.listdefs = {}` (bookkeeper.py:59). Keyed by
    /// position — callers hitting the same position twice share the
    /// ListDef so merging re-entries stay identity-equal. The key is
    /// `Option<PositionKey>` because upstream uses `self.position_key`
    /// directly as the dict key (bookkeeper.py:180
    /// `self.listdefs[self.position_key]`); when `position_key` is
    /// `None`, upstream still caches under the `None` key — so we do
    /// the same rather than building a fresh ListDef per call outside
    /// a reflow frame.
    pub(crate) listdefs: RefCell<HashMap<Option<PositionKey>, ListDef>>,
    /// RPython `self.dictdefs = {}` (bookkeeper.py:60). Same
    /// `Option<PositionKey>` key semantics as `listdefs`.
    pub(crate) dictdefs: RefCell<HashMap<Option<PositionKey>, DictDef>>,
    /// RPython `self.descs = {}` (bookkeeper.py:67). Maps
    /// `Constant(pyobj)` to a FunctionDesc / ClassDesc / FrozenDesc /
    /// MethodDesc / MethodOfFrozenDesc per bookkeeper.py:353-409. The
    /// Rust port keys directly on [`HostObject`] (which already has
    /// `Arc::ptr_eq` identity) via [`DescEntry`].
    pub(crate) descs: RefCell<HashMap<HostObject, DescEntry>>,
    /// RPython `self.classdefs = []` (bookkeeper.py:68). Populated by
    /// `ClassDesc._init_classdef` (classdesc.py:672-697). ClassDef
    /// identity is Rc pointer equality — matches upstream's Python
    /// `cls is other` comparisons.
    pub classdefs: RefCell<Vec<Rc<RefCell<ClassDef>>>>,
    /// RPython `self.methoddescs = {}` (bookkeeper.py:69). Keyed by
    /// `(funcdesc, originclassdef, selfclassdef, name, flags)` tuple
    /// so repeated `getmethoddesc(...)` calls with the same inputs
    /// share identity, per bookkeeper.py:431-442.
    pub(crate) methoddescs: RefCell<HashMap<MethodDescKey, Rc<RefCell<MethodDesc>>>>,
    /// RPython `self.frozenpbc_attr_families = UnionFind(FrozenAttrFamily)`
    /// (bookkeeper.py:63).
    pub(crate) frozenpbc_attr_families: RefCell<UnionFind<DescKey, Rc<RefCell<FrozenAttrFamily>>>>,
    /// RPython `self.classpbc_attr_families = {}` (bookkeeper.py:62) —
    /// lazy `attrname -> UnionFind(ClassAttrFamily)` map materialised by
    /// `get_classpbc_attr_families(attrname)` (bookkeeper.py:447-456).
    pub(crate) classpbc_attr_families:
        RefCell<HashMap<String, UnionFind<DescKey, Rc<RefCell<ClassAttrFamily>>>>>,
    /// RPython `self.pbc_maximal_call_families = UnionFind(CallFamily)`
    /// (bookkeeper.py:64).
    pub(crate) pbc_maximal_call_families: RefCell<UnionFind<DescKey, Rc<RefCell<CallFamily>>>>,
    /// RPython `self.emulated_pbc_calls = {}` (bookkeeper.py:66).
    pub(crate) emulated_pbc_calls: RefCell<HashMap<EmulatedPbcCallKey, (SomePBC, Vec<SomeValue>)>>,
    /// RPython `bookkeeper._jit_annotation_cache = {}`
    /// (rlib/jit.py:903-914), populated lazily by
    /// `ExtEnterLeaveMarker.compute_result_annotation`.
    ///
    /// Outer key: JitDriver `HostObject` identity — RPython keys on
    /// the driver object directly (`cache = self.bookkeeper.
    /// _jit_annotation_cache[driver]`), so the Rust port keys on the
    /// `HostObject` whose `Hash` / `Eq` already match upstream's
    /// `id(driver)` semantics. Storing the `HostObject` (not just
    /// `usize`) preserves the strong reference upstream holds via the
    /// dict, avoiding identity reuse if the underlying object is
    /// dropped. Inner map: kwarg name → merged annotation across all
    /// `jit_merge_point` / `can_enter_jit` call sites for that
    /// driver. Upstream merges via `annmodel.unionof`; the Rust port
    /// uses [`super::model::union`] (model.rs:2568) with
    /// [`super::model::s_impossible_value`] (model.rs:2436) as the
    /// seed for the first hit.
    pub _jit_annotation_cache: RefCell<HashMap<HostObject, HashMap<String, SomeValue>>>,
    /// RPython `self.immutable_cache = {}` (bookkeeper.py:61), keyed by
    /// the identity of `Constant(x)` for the List / Dict / OrderedDict /
    /// r_dict branches of `immutablevalue` (bookkeeper.py:255-298).
    /// Upstream's cache memoises the returned `SomeList` / `SomeDict` so
    /// downstream `generalize_key` / `generalize` mutations on the
    /// cached `DictDef` / `ListDef` persist across calls.
    ///
    /// `transform_list_contains` (translator/transform.py:115-134)
    /// relies on this: it calls `self.annotation(Constant(dict))` after
    /// rewriting the contains lhs, then mutates the returned SomeDict's
    /// dictdef via `generalize_key`. Without the cache the second
    /// `annotation()` call builds a fresh DictDef and the mutation
    /// dissolves.
    pub immutable_cache: RefCell<HashMap<u64, SomeValue>>,
    /// RPython `self.pending_specializations = []` (bookkeeper.py:69).
    ///
    /// List of callbacks drained by
    /// `AnnotatorPolicy.no_more_blocks_to_annotate` before the final
    /// annotation fixpoint. Each callback (e.g. `MemoTable.finish`) is
    /// fallible; upstream lets the exception propagate, so the drain
    /// surfaces the first `Err` instead of swallowing or panicking on it.
    pub pending_specializations: RefCell<Vec<Box<dyn Fn() -> Result<(), AnnotatorError>>>>,
    /// RPython `self.all_specializations = {}` (bookkeeper.py:68).
    ///
    /// One entry per memo-specialised function, keyed by the funcdesc's
    /// pointer identity (upstream keys by the `funcdesc` object itself,
    /// specialize.py:285). Each value is the
    /// `UnionFind(compute_one_result)` upstream uses (specialize.py:298) —
    /// the families of argument tuples that can be called together, merged
    /// on `union`, with each result living in its
    /// [`super::specialize::MemoTable`]'s `table` — paired with a per-
    /// family host-call error latch (see [`super::specialize::MemoFamily`]).
    /// Consumed by [`super::specialize::memo`].
    pub all_specializations: RefCell<HashMap<usize, super::specialize::MemoFamily>>,
    /// RPython `hasattr(self, 'position_key')` (bookkeeper.py:99).
    ///
    /// Upstream distinguishes "no position entered" (attribute absent)
    /// from "entered with position_key = None" (attribute present but
    /// None). Rust carries [`Self::position_key`] as `Option<_>` in both
    /// cases, so this flag tracks the enter/leave invariant explicitly.
    pub position_entered: std::cell::Cell<bool>,
    /// RPython `self.needs_generic_instantiate = {}` (bookkeeper.py:72).
    ///
    /// Populated by `merge_classpbc_getattr_into_classdef`
    /// (normalizecalls.py:260-262) for every class-PBC call family that
    /// spans more than one class — `ClassesPBCRepr.call()` needs the
    /// generic instantiator. Drained by
    /// `create_instantiate_functions` (normalizecalls.py:266-273).
    ///
    /// Upstream is a `{}` dict keyed by `ClassDef` identity with every
    /// value set to `True` — i.e. used as a set with upstream's
    /// insertion-order iteration. Pyre mirrors the dict shape via a
    /// `BTreeMap<ClassDefKey, Rc<RefCell<ClassDef>>>`: keys are the
    /// `ClassDefKey` pointer-identity wrapper (matching upstream's
    /// `id(classdef)` hash) and the value retains the `Rc` needed by
    /// consumers. Iteration order is deterministic (sorted by the
    /// pointer-derived key) which differs from upstream's insertion
    /// order, but the consumer
    /// ([`crate::translator::rtyper::normalizecalls::create_instantiate_functions`])
    /// processes each classdef independently so iteration order is not
    /// observable.
    pub(crate) needs_generic_instantiate:
        RefCell<std::collections::BTreeMap<ClassDefKey, Rc<RefCell<ClassDef>>>>,
    /// TODO: no upstream equivalent.  Pyre-only struct-field
    /// metadata snapshot (`struct_name -> [(field_name, type_string)]`)
    /// used by [`Self::getuniqueclassdef_for_struct_root`] to project the
    /// registered `ClassDef.attrs`.  `None` for unit-test fixtures that
    /// build the bookkeeper directly, in which case
    /// `getuniqueclassdef_for_struct_root` leaves `attrs` empty.
    pub pyre_struct_fields: RefCell<Option<Rc<crate::front::StructFieldRegistry>>>,
    /// TODO: no upstream equivalent (RPython has no Rust enums).  Enum
    /// type-root name (dual-keyed: qualified path and bare leaf) →
    /// `discriminant value → variant name` table.  Consulted by
    /// [`Self::enum_variant_narrowing_knowntypedata`] to build the
    /// discriminant→variant narrowing `knowntypedata` the `__discriminant`
    /// getattr attaches.  `None` for unit-test fixtures that build the
    /// bookkeeper directly.
    pub pyre_enum_variant_by_discriminant:
        RefCell<Option<Rc<HashMap<String, HashMap<i64, String>>>>>,
    /// TODO: no upstream equivalent.  Interning table mapping a struct
    /// type-root name to its canonical host class `HostObject`.  Because
    /// `HostObject` equality is `Arc` pointer identity (`model.rs:233`),
    /// two `HostObject::new_class(name, ...)` calls produce distinct
    /// descs/classdefs; interning resolves a type-root string to one
    /// class OBJECT exactly once so subsequent `getuniqueclassdef`
    /// lookups key the same `ClassDef` by identity — the pyre analog of
    /// `annotationoftype(t) -> getuniqueclassdef(t)` (signature.py:103-104)
    /// where `t` is the already-resolved class object.  Used by
    /// [`Self::getuniqueclassdef_for_struct_root`].
    pub pyre_struct_root_classes: RefCell<HashMap<String, HostObject>>,
    /// TODO: no upstream equivalent.  Qualified trait path → owner
    /// root of its only concrete impl in the analyzed LLBC world
    /// (computed in `lib.rs` from `concrete_trait_methods`; multi-impl
    /// traits are absent).  The key is the trait's full `name_path()`
    /// so leaf-name collisions between distinct traits cannot pool
    /// impl owners.  `derive_subject_inputcells` consults this when a
    /// `Ref` parameter's `class_root` is a bound-trait path (generic
    /// receiver) — the unique impl's struct root then seeds the
    /// receiver's `ClassDef` through
    /// [`Self::getuniqueclassdef_for_struct_root`].
    /// RPython's annotator never needs this: it sees the concrete
    /// receiver class at every call site (`classdesc.py:749 lookup`).
    pub pyre_trait_unique_impls: RefCell<HashMap<String, String>>,
    /// Trait qualified-path (`name_path()`) → base `HostObject` for a
    /// receiver-driven method-dispatch family registered through
    /// [`Self::register_trait_family`] (issue #346).
    /// `derive_subject_inputcells` seeds a `dyn Trait` receiver whose
    /// `class_root` is a key here with the base `ClassDef`, so
    /// `getattr(receiver, method)` resolves the impl-subclass
    /// `MethodDesc` family (attrfamily merge) rather than blocking on
    /// the classdef-less shell.  Empty unless a consumer opts a trait
    /// in; pyre production registers none.
    pub pyre_trait_family_bases: RefCell<HashMap<String, HostObject>>,
    /// TODO: no upstream equivalent.  Struct names first interned by
    /// [`Self::project_pyre_field_type`]'s bare-name arm whose
    /// registry rows have not been projected yet — drained at the end
    /// of [`Self::getuniqueclassdef_for_struct_root`] so a class
    /// never exposes its untyped FORCE shells to subject flow (a
    /// later projection would be a non-monotonic attr flip).
    pub(crate) pending_struct_row_projection: RefCell<Vec<String>>,
    /// Canonical struct names whose rows [`Self::project_struct_rows`]
    /// already projected (dedups the pending drain and the bare-name
    /// arm's first-sight detection).
    pub(crate) projected_struct_rows: RefCell<std::collections::HashSet<String>>,
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
pub(crate) struct MethodDescKey {
    /// RPython `funcdesc` — pointer identity via [`DescKey::from_rc`].
    pub(crate) funcdesc_id: DescKey,
    /// RPython `originclassdef` — `ClassDefKey` already carries the
    /// pointer identity.
    pub(crate) originclassdef: ClassDefKey,
    /// RPython `selfclassdef` — `None` for unbound methods.
    pub(crate) selfclassdef: Option<ClassDefKey>,
    /// RPython `name`.
    pub(crate) name: String,
    /// RPython `tuple(flags.items())` — flattened sort-stable flag
    /// entries.
    pub(crate) flags: Vec<(String, bool)>,
}

/// Hashable identity for `Bookkeeper.emulated_pbc_calls`.
///
/// Upstream accepts any hashable Python object. The currently-ported
/// callers need a stable position key plus the r_dict eq/hash
/// pseudo-call identities.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum EmulatedPbcCallKey {
    #[allow(dead_code)]
    Position(PositionKey),
    Graph(GraphKey),
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
pub(crate) enum PbcCallEmulated {
    None,
    True,
    Callback(PositionKey),
}

impl EmulatedPbcCallKey {
    /// Build the sandbox-trampoline key used by
    /// `AnnotatorPolicy::no_more_blocks_to_annotate` (policy.py:87).
    pub(crate) fn sandboxing(func_const: &crate::flowspace::model::ConstValue) -> Self {
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
            _jit_annotation_cache: RefCell::new(HashMap::new()),
            immutable_cache: RefCell::new(HashMap::new()),
            pending_specializations: RefCell::new(Vec::new()),
            all_specializations: RefCell::new(HashMap::new()),
            position_entered: std::cell::Cell::new(false),
            needs_generic_instantiate: RefCell::new(std::collections::BTreeMap::new()),
            pyre_struct_fields: RefCell::new(None),
            pyre_enum_variant_by_discriminant: RefCell::new(None),
            pyre_struct_root_classes: RefCell::new(HashMap::new()),
            pyre_trait_unique_impls: RefCell::new(HashMap::new()),
            pyre_trait_family_bases: RefCell::new(HashMap::new()),
            pending_struct_row_projection: RefCell::new(Vec::new()),
            projected_struct_rows: RefCell::new(std::collections::HashSet::new()),
        }
    }

    /// TODO: no upstream equivalent.  Wire the pyre-only
    /// `StructFieldRegistry` so [`Self::getuniqueclassdef_for_struct_root`]
    /// can project the registered `ClassDef.attrs` from struct-field
    /// metadata.
    ///
    /// May be called before or after roots are registered:
    /// [`Self::getuniqueclassdef_for_struct_root`] re-traverses and
    /// re-projects on every call, so a root cached before the registry
    /// arrived has its attrs back-filled on the next lookup (RPython grows
    /// class attrs as annotation proceeds).  Idempotent: a second call
    /// overwrites the previous registry.
    pub fn set_pyre_struct_fields(&self, registry: Rc<crate::front::StructFieldRegistry>) {
        *self.pyre_struct_fields.borrow_mut() = Some(registry);
    }

    /// TODO: no upstream equivalent.  Wire the enum
    /// `discriminant → variant` tables consumed by
    /// [`Self::enum_variant_narrowing_knowntypedata`].  Idempotent: a
    /// second call overwrites the previous map.
    pub fn set_pyre_enum_variant_by_discriminant(
        &self,
        map: Rc<HashMap<String, HashMap<i64, String>>>,
    ) {
        *self.pyre_enum_variant_by_discriminant.borrow_mut() = Some(map);
    }

    /// TODO: no upstream equivalent.  Wire the trait →
    /// unique-concrete-impl-owner map (see
    /// [`Self::pyre_trait_unique_impls`]).  Idempotent: a second call
    /// overwrites the previous map.
    pub fn set_pyre_trait_unique_impls(&self, map: HashMap<String, String>) {
        *self.pyre_trait_unique_impls.borrow_mut() = map;
    }

    /// TODO: no upstream equivalent.  The full set of struct type-root
    /// keys in the snapshot registry (`StructFieldRegistry.fields`).
    /// The session-prologue inheritance-id pass
    /// (`PyreCallRegistry::ensure_session`) iterates these to eagerly
    /// pre-register the build-time-closed `W_*` struct hierarchy before
    /// running [`crate::translator::rtyper::normalizecalls::assign_inheritance_ids`].
    /// Returns owned `String`s (not an iterator) so the registry borrow
    /// is released before the caller re-borrows it through
    /// [`Self::getuniqueclassdef_for_struct_root`].  Empty when no
    /// registry is set (unit-test fixtures), degrading the pass to a
    /// no-op over whatever classdefs already exist.
    pub fn pyre_struct_root_names(&self) -> Vec<String> {
        self.pyre_struct_fields
            .borrow()
            .as_ref()
            .map(|reg| reg.fields.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// True when `leaf` is the trait-leaf of a registered dispatch family
    /// ([`Self::pyre_trait_family_bases`], keyed by the trait's full
    /// `name_path()`).  A method call lowered as `FunctionPath [<leaf>,
    /// method]` (`front::mir`'s `CallKind::Trait` arm spells the trait by
    /// leaf) routes through the receiver's getattr when this matches and
    /// the direct-path registry has no entry (the required-method,
    /// `>=2`-impl case).  Empty unless a consumer opts a trait in.
    pub fn is_registered_trait_family_leaf(&self, leaf: &str) -> bool {
        self.pyre_trait_family_bases
            .borrow()
            .keys()
            .any(|path| path.rsplit("::").next().unwrap_or(path) == leaf)
    }

    /// No upstream equivalent — forced by pyre's numbering order, not a
    /// casual workaround.  Upstream runs `assign_inheritance_ids` ONCE
    /// inside `RPythonTyper.specialize`, AFTER `annotator.complete()`
    /// reaches its fixpoint, when every instantiated class — variant
    /// classes included — already exists; the single numbering pass sees
    /// the complete set and needs no pre-mint.  pyre cannot follow that
    /// order: it runs incremental per-subject `specialize_more_blocks`
    /// (no single post-annotation driver), and `ClassRepr.fill_vtable_root`
    /// bakes `minid`/`maxid` as eager `Signed` constants into the
    /// cross-graph-cached vtable, so the ids must be stable in the session
    /// prologue — before any per-subject annotation discovers a
    /// lazily-minted variant class.  This pre-mint is the prologue-time
    /// analogue of upstream's "all classes exist before numbering"
    /// invariant, restricted to the enum-variant subtrees that would
    /// otherwise be minted lazily mid-session: it pre-mints the variant
    /// subclasses of every registered enum so the session-prologue
    /// [`crate::translator::rtyper::normalizecalls::assign_inheritance_ids`]
    /// pass numbers each `enum-base + variant-children` subtree as one
    /// contiguous bracket.  Called from `PyreCallRegistry::ensure_session`
    /// AFTER the struct-root loop (so the enum-base classdefs already
    /// exist, UNNUMBERED) and BEFORE the single `assign_inheritance_ids`.
    ///
    /// A variant ctor instantiation (`PyError::type_error` →
    /// `PyErrorKind::TypeError`) mints its variant class LAZILY during
    /// annotation, after the prologue numbering has already baked the
    /// enum base's `[minid,maxid]` bracket.  An on-demand
    /// `assign_inheritance_ids` re-run then cannot number the fresh
    /// child (its bracket would have to nest inside the parent's baked
    /// range — not append-safe), so `ClassesPBCRepr.redispatch_call`
    /// Skip-classifies the instantiation.  Pre-minting the whole subtree
    /// here leaves every member UNNUMBERED at the single numbering pass,
    /// so the contiguous bracket is assigned once and never shifts.
    ///
    /// Spelling MUST match the annotation path or the pre-mint is a
    /// phantom that never matches the lazily-minted class, so each variant
    /// is interned through the SAME [`Self::intern_enum_variant_host`]
    /// primitive the discriminant-narrowing resolver
    /// ([`Self::getuniqueclassdef_for_enum_variant`]) and the variant ctor
    /// arm (`flowspace_adapter`'s `SyntheticTransparentCtor`) route through:
    /// base = `intern_class_by_qualname(canonical_struct_name(root))`,
    /// variant = `intern_class_by_qualname_with_bases(
    /// "{canon_root}::{variant}", [base])`.  All three sites thus resolve
    /// one `HostObject` per variant under the `::`-qualified key.
    /// `enum_variant_by_discriminant` dual-publishes each enum under both
    /// its `::`-qualified path and its bare leaf; iterate the qualified
    /// keys only.  Gated on the same `is_enum_base` predicate the ctor arm
    /// uses (a flat class whose sole row is the synthetic `__discriminant`
    /// tag), so the pre-mint covers exactly the variant subclasses the
    /// adapter can mint — payload-bearing enums included, since their
    /// payloads live under `{enum}::{variant}` keys, not on the base.
    pub fn pre_register_enum_variant_classes(self: &Rc<Self>) {
        // Collect (qualified_root, [variant names]) for every enum base,
        // releasing both registry borrows before minting (the intern /
        // getuniqueclassdef calls re-borrow `pyre_struct_fields` and
        // `pyre_struct_root_classes`).  Sorted for a deterministic mint
        // order so the numbering bracket is reproducible.
        let mut pairs: Vec<(String, Vec<String>)> = {
            let variant_guard = self.pyre_enum_variant_by_discriminant.borrow();
            let fields_guard = self.pyre_struct_fields.borrow();
            let (Some(variants), Some(reg)) = (variant_guard.as_ref(), fields_guard.as_ref())
            else {
                return;
            };
            variants
                .iter()
                // A `::`-qualified template root (`module::Enum`) or a
                // per-instantiation root (`Result<Tuple>`, #100) — both
                // name a real enum base.  The bare leaf duplicate
                // (`Enum`) is the other key the registration publishes,
                // skipped here so the subtree numbers once.
                .filter(|(root, _)| root.contains("::") || root.contains('<'))
                .filter_map(|(root, by_discr)| {
                    let leaf = root.rsplit("::").next().unwrap_or(root);
                    reg.is_enum_base(leaf).then(|| {
                        let mut names: Vec<String> = by_discr.values().cloned().collect();
                        names.sort();
                        names.dedup();
                        (root.clone(), names)
                    })
                })
                .collect()
        };
        pairs.sort();

        for (root, variant_names) in pairs {
            // Materialize the discriminant-only base classdef before a
            // variant references it through `getmro`; idempotent with the
            // struct-root loop.  `canonical_struct_name(root)` is the same
            // spelling `intern_enum_variant_host` resolves the base under,
            // so the pre-mint and the discriminant-narrowing resolver share
            // one base lineage and the variant subtree numbers as one
            // bracket.
            let canon_root = majit_ir::descr::canonical_struct_name(&root);
            let base = self.intern_class_by_qualname(&canon_root);
            let _ = self.getuniqueclassdef(&base);
            for variant in variant_names {
                // The SAME interning primitive the discriminant-narrowing
                // resolver ([`Self::getuniqueclassdef_for_enum_variant`])
                // and the variant ctor arm (`flowspace_adapter`) use, so
                // all three sites resolve ONE variant classdef under the
                // `::`-qualified key — no `.`-vs-`::` split that would mint
                // a second, distinct sibling the single numbering pass never
                // reaches.
                let variant_host = self.intern_enum_variant_host(&root, &variant);
                let _ = self.getuniqueclassdef(&variant_host);
            }
        }
    }

    /// Push a classdef into [`Self::needs_generic_instantiate`] unless
    /// it is already present (upstream `dict[cdef] = True` idempotence).
    pub fn push_needs_generic_instantiate(&self, classdef: &Rc<RefCell<ClassDef>>) {
        self.needs_generic_instantiate
            .borrow_mut()
            .entry(ClassDefKey::from_classdef(classdef))
            .or_insert_with(|| classdef.clone());
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

    /// Non-panicking variant of [`Self::annotator`].  Returns `None`
    /// when the backlink is absent (test fixtures that wire the
    /// bookkeeper without an `RPythonAnnotator`) or when the
    /// `Rc<RPythonAnnotator>` has been dropped.  Production callers
    /// that need the annotator should prefer [`Self::annotator`];
    /// this variant is for diagnostic side-channels that should
    /// degrade gracefully when the annotator is not yet attached
    /// (e.g. lift-error recording from a registry pre-pass that may
    /// also be exercised in standalone fixture setups).
    pub fn try_annotator(&self) -> Option<Rc<crate::annotator::annrpython::RPythonAnnotator>> {
        self.annotator.borrow().upgrade()
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
    #[cfg(test)]
    pub(crate) fn set_position_key(&self, pk: Option<PositionKey>) -> Option<PositionKey> {
        self.position_key.replace(pk)
    }

    /// Current `bookkeeper.position_key`. Returns `None` when no
    /// reflow frame is active (upstream's initial
    /// `self.position_key = None`).
    pub(crate) fn current_position_key(&self) -> Option<PositionKey> {
        self.position_key.borrow().clone()
    }

    /// RPython `Bookkeeper.enter(self, position_key)` (bookkeeper.py:84-89).
    ///
    /// Installs the position and registers `self` as the thread-local
    /// bookkeeper so [`getbookkeeper`] returns it. Asserts that no
    /// `enter` is currently active — matches upstream's `not hasattr`
    /// check.
    pub(crate) fn enter(self: &Rc<Self>, position_key: Option<PositionKey>) {
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
    pub(crate) fn leave(&self) {
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
    pub(crate) fn at_position(self: &Rc<Self>, pos: Option<PositionKey>) -> PositionGuard {
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
    ///
    /// Errors from `consider_call_site` propagate verbatim: upstream
    /// `bookkeeper.py:108-118` runs both inner loops without a
    /// `try`/`except`, so a raised exception terminates
    /// `compute_at_fixpoint` and unwinds out of
    /// `RPythonAnnotator.simplify`.  Pyre's port mirrors this with
    /// early `?`-propagation; callers that previously ignored the
    /// `Result` (the `let _ = ...` shape that Issue 3.1 flagged) are
    /// migrated to propagate or to panic on uncaught errors,
    /// matching upstream's Python-exception semantics.
    pub fn compute_at_fixpoint(self: &Rc<Self>) -> Result<(), AnnotatorError> {
        let _guard = self.at_position(None);
        let ann = self.annotator();
        for (call_op, op_key) in ann.call_sites_with_positions() {
            self.consider_call_site(&call_op, Some(op_key))?;
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
            pbc.consider_call_site(&args, &SomeValue::Impossible, None)?;
        }
        // upstream: `self.emulated_pbc_calls = {}`.
        self.emulated_pbc_calls.borrow_mut().clear();
        Ok(())
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
    /// `call_op.build_args`
    /// is opname-sensitive (`simple_call` →
    /// `ArgumentsForTranslation(list(args_s))`, `call_args` →
    /// `fromshape`); the simple_call path uses [`simple_args`].
    pub(crate) fn consider_call_site(
        self: &Rc<Self>,
        call_op: &crate::flowspace::model::SpaceOperation,
        op_key: Option<PositionKey>,
    ) -> Result<(), AnnotatorError> {
        let ann = self.annotator();
        let Some(mut s_callable) = ann.annotation(&call_op.args[0]) else {
            return Ok(());
        };
        // upstream `args_s = [annotation(arg) for arg in call_op.args[1:]]`
        // (`bookkeeper.py:156`) preserves positional arity *and* the
        // `None` sentinel for not-yet-annotated args (Python list of
        // `SomeValue or None`).  Pyre's `arguments_w:
        // Vec<Option<SomeValue>>` carries None unchanged so downstream
        // `pbc_call → FunctionDesc.pycall → recursivecall` can route
        // through unbound positions exactly as upstream
        // `description.py:283-305` does (each unbound position skips
        // its `binding_join` until the fixpoint round populates it).
        let mut args_s: Vec<Option<SomeValue>> = call_op
            .args
            .iter()
            .skip(1)
            .map(|a| ann.annotation(a))
            .collect();
        if let SomeValue::LLADTMeth(adtmeth) = &s_callable {
            let func = adtmeth.func.clone();
            let ll_ptrtype = adtmeth.ll_ptrtype.clone();
            s_callable = self.immutablevalue(&func)?;
            // upstream `bookkeeper.py:158`: args_s = [lltype_to_
            // annotation(adtmeth.ll_ptrtype)] + args_s — the prepended
            // ll_ptrtype is always concrete.
            args_s.insert(
                0,
                Some(crate::translator::rtyper::llannotation::lltype_to_annotation(ll_ptrtype)),
            );
        }
        if let SomeValue::PBC(pbc) = &s_callable {
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
    pub(crate) fn getattr_locations(
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
    pub(crate) fn getdesc(
        self: &Rc<Self>,
        pyobj: &HostObject,
    ) -> Result<DescEntry, AnnotatorError> {
        if let Some(existing) = self.descs.borrow().get(pyobj) {
            return Ok(existing.clone());
        }
        let entry = if pyobj.is_user_function() {
            // upstream `newfuncdesc` already returns a MemoDesc or
            // FunctionDesc per the specializer.
            self.newfuncdesc(pyobj)?
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
            // Keep the whole `FuncDescEntry` (not `as_function()`, which
            // unwraps a memo to its base) so a bound memo method's
            // `MethodDesc.funcdesc` retains the exact MemoDesc identity.
            let funcdesc = self
                .getdesc(func)?
                .as_func_entry()
                .cloned()
                .ok_or_else(|| {
                    AnnotatorError::new(
                        "Bookkeeper.getdesc(bound method): __func__ is not a function",
                    )
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
                let (origin_class, name) = origin_of_meth(pyobj)?;
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
                let _ = super::classdesc::ClassDef::find_attribute(&classdef, &name)?;
                DescEntry::Method(self.getmethoddesc(
                    &funcdesc,
                    ClassDefKey::from_classdef(&self.getuniqueclassdef(origin_class)?),
                    Some(ClassDefKey::from_classdef(&classdef)),
                    &name,
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
    /// specializer from `AnnotatorPolicy.get_specializer(tag)`. When the
    /// specializer is `memo`, return a [`DescEntry::Memo`] wrapping the
    /// `FunctionDesc` (upstream `if specializer is memo: return MemoDesc(
    /// ...)`, bookkeeper.py:419-425); otherwise a [`DescEntry::Function`].
    pub(crate) fn newfuncdesc(
        self: &Rc<Self>,
        pyfunc: &HostObject,
    ) -> Result<DescEntry, AnnotatorError> {
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
        // Prefer the live annotator policy (respects `using_policy`
        // swap) when the backlink is present; fall back to the
        // bookkeeper-owned static `policy` slot in unit-test
        // configurations that construct `Bookkeeper::new()` without
        // attaching an annotator.
        let specializer = match self.annotator.borrow().upgrade() {
            Some(ann) => ann
                .policy
                .borrow()
                .get_specializer(gf.annspecialcase.as_deref())
                .map_err(|e| AnnotatorError::new(e.to_string()))?,
            None => self
                .policy
                .get_specializer(gf.annspecialcase.as_deref())
                .map_err(|e| AnnotatorError::new(e.to_string()))?,
        };
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            self.clone(),
            Some(pyfunc.clone()),
            name,
            signature,
            defaults,
            Some(specializer),
        )));
        // upstream: `if specializer is memo: return MemoDesc(...) else:
        // return FunctionDesc(...)`.
        if fd.borrow().is_memo() {
            Ok(DescEntry::memo(Rc::new(RefCell::new(
                super::description::MemoDesc::new(fd),
            ))))
        } else {
            Ok(DescEntry::function(fd))
        }
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
    pub(crate) fn getmethoddesc(
        self: &Rc<Self>,
        funcdesc: &super::description::FuncDescEntry,
        originclassdef: ClassDefKey,
        selfclassdef: Option<ClassDefKey>,
        name: &str,
        flags: std::collections::BTreeMap<String, bool>,
    ) -> Rc<RefCell<MethodDesc>> {
        let flags_vec: Vec<(String, bool)> = flags.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let key = MethodDescKey {
            // `FuncDescEntry::desc_key()` is the memo wrapper's identity
            // for a memo (the base FunctionDesc's for a plain function),
            // so a memo method keys to the same identity the PBC set
            // carries.
            funcdesc_id: funcdesc.desc_key(),
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

    /// Codewriter-time method-desc discovery.
    ///
    /// Walks `receiver_classdef`'s MRO for the first `SomePBC` attr
    /// under `name`, then applies `classdesc.py:344-365` filtering
    /// rules inline to collect all qualifying MethodDescs.
    ///
    /// Classdef-attrs read-only: uses `get_owner` / `attrs.get`
    /// instead of `find_attribute` (which calls `locate_attribute`
    /// → `generalize_attr_internal` mutating classdef attrs — a
    /// phase violation at codewriter time). Does prime the
    /// `Bookkeeper.methoddescs` cache via `getmethoddesc` calls.
    pub fn getmethoddesc_for_attribute(
        self: &Rc<Self>,
        receiver_classdef: &Rc<RefCell<super::classdesc::ClassDef>>,
        name: &str,
    ) -> Vec<Rc<RefCell<super::description::MethodDesc>>> {
        use super::description::DescEntry;
        use super::model::SomeValue;
        let receiver_key = super::description::ClassDefKey::from_classdef(receiver_classdef);
        let mro = super::classdesc::ClassDef::getmro(receiver_classdef);
        for cdef in mro {
            let s_value = cdef.borrow().attrs.get(name).map(|a| a.s_value.clone());
            let Some(SomeValue::PBC(pbc)) = s_value else {
                continue;
            };
            // classdesc.py:344-365 filtering rules applied inline.
            // Does not call ClassDef::lookup_filter (which calls
            // bind_self); instead calls getmethoddesc directly to
            // prime the methoddescs cache.
            let mut d: Vec<Rc<RefCell<super::description::MethodDesc>>> = Vec::new();
            let mut uplookup: Option<Rc<RefCell<super::classdesc::ClassDef>>> = None;
            let mut up_md: Option<Rc<RefCell<super::description::MethodDesc>>> = None;
            for entry in pbc.descriptions.values() {
                let DescEntry::Method(md) = entry else {
                    continue;
                };
                let (funcdesc, originclassdef, name_owned, flags, existing_self) = {
                    let md_borrow = md.borrow();
                    (
                        md_borrow.funcdesc.clone(),
                        md_borrow.originclassdef,
                        md_borrow.name.clone(),
                        md_borrow.flags.clone(),
                        md_borrow.selfclassdef,
                    )
                };
                if existing_self.is_some() {
                    // Already-bound: kept verbatim (classdesc.py:347).
                    d.push(self.getmethoddesc(
                        &funcdesc,
                        originclassdef,
                        existing_self,
                        &name_owned,
                        flags,
                    ));
                    continue;
                }
                let Some(methclassdef) = self.lookup_classdef(originclassdef) else {
                    continue;
                };
                let methclassdef_is_receiver = Rc::ptr_eq(&methclassdef, receiver_classdef);
                let meth_issub_receiver = methclassdef.borrow().issubclass(receiver_classdef);
                let receiver_issub_meth = receiver_classdef.borrow().issubclass(&methclassdef);
                if !methclassdef_is_receiver && meth_issub_receiver {
                    // Subclass-origin: bind to methclassdef
                    // (classdesc.py:349-364).
                    let methclassdef_key =
                        super::description::ClassDefKey::from_classdef(&methclassdef);
                    d.push(self.getmethoddesc(
                        &funcdesc,
                        originclassdef,
                        Some(methclassdef_key),
                        &name_owned,
                        flags,
                    ));
                } else if receiver_issub_meth {
                    // Upward: track best (classdesc.py:351-356).
                    let promote = match &uplookup {
                        None => true,
                        Some(cur) => methclassdef.borrow().issubclass(cur),
                    };
                    if promote {
                        uplookup = Some(methclassdef.clone());
                        up_md = Some(md.clone());
                    }
                }
            }
            // Upward best-match bound to receiver
            // (classdesc.py:366-367).
            if let Some(up_md_rc) = up_md {
                let (funcdesc, originclassdef, name_owned, flags) = {
                    let b = up_md_rc.borrow();
                    (
                        b.funcdesc.clone(),
                        b.originclassdef,
                        b.name.clone(),
                        b.flags.clone(),
                    )
                };
                d.push(self.getmethoddesc(
                    &funcdesc,
                    originclassdef,
                    Some(receiver_key),
                    &name_owned,
                    flags,
                ));
            }
            // classdesc.py:369-374 — once the owner's PBC is found,
            // return the filtered result even if empty. Does not
            // continue to parent attrs.
            return d;
        }
        Vec::new()
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
    pub(crate) fn with_classpbc_attr_families<T>(
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

    pub(crate) fn lookup_classdef(&self, key: ClassDefKey) -> Option<Rc<RefCell<ClassDef>>> {
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

    /// Resolve a struct type-root name to its canonical, bookkeeper-
    /// REGISTERED `ClassDef`, with instance attributes projected from
    /// the threaded `StructFieldRegistry`.
    ///
    /// pyre's `W_IntObject` /
    /// `PyFrame` / `W_DictObject` Rust structs are ports of RPython
    /// classes (`W_Root` subclasses), so their annotation is
    /// `SomeInstance(classdef)` — `InstanceRepr._setup_repr`
    /// (`rclass.py:501-509`) later lowers `classdef.attrs` into
    /// `Ptr(GcStruct(OBJECT, inst_<field>...))`.  RPython derives a
    /// `ClassDef` ONLY by class-object identity
    /// (`getuniqueclassdef(cls)` -> `getdesc(cls)` -> id() key,
    /// `bookkeeper.py:339-345`, `uid.py:24-48`); there is no
    /// name->classdef path.  Pyre mirrors this by resolving the
    /// type-root string to a host class OBJECT exactly once
    /// ([`Self::intern_class_by_qualname`]), then routing through the
    /// existing [`Self::getuniqueclassdef`] so the `ClassDef` lands in
    /// `descs`/`classdefs` keyed by identity.
    ///
    /// The struct fields are instance attributes: RPython discovers them
    /// by annotating `setattr`/`__init__`, but pyre has no such pass for
    /// host structs, so the registry projection (via
    /// [`Self::project_pyre_field_type`]) stands in.  The whole reachable
    /// struct-field graph is registered and projected — an explicit
    /// work-list bounds recursion at the reachable-struct count — so a
    /// field typed as another struct resolves to a fully populated inner
    /// `SomeInstance(classdef)`, not an attrs-empty inner stub.
    ///
    /// Order-independent w.r.t. [`Self::set_pyre_struct_fields`]: the call
    /// re-traverses and re-projects every time, so a root registered
    /// before the field registry arrived has its (and its transitive
    /// structs') attrs back-filled on a later call — RPython grows class
    /// attrs as annotation proceeds.  Identity is stable across calls
    /// (the host class is interned once), so the back-fill mutates the
    /// already-published `ClassDef` rather than minting a fresh one.
    pub fn getuniqueclassdef_for_struct_root(
        self: &Rc<Self>,
        root: &str,
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        // Pass 1 — traverse the registry's struct-field graph from `root`,
        // registering an identity-keyed `ClassDef` in `descs` for every
        // reachable struct (via `intern_class_by_qualname` ->
        // `getuniqueclassdef`).  `project_pyre_field_type`'s bare-name arm
        // resolves inner struct references through the same identity path,
        // so it sees the registered classdef.  Registration is idempotent
        // (the identity cache is the memo), so a root reached before the
        // field registry arrived still has its transitive structs
        // discovered — and its attrs re-projected in pass 2 — on a later
        // call.  Cycles are broken by testing membership against `graph`
        // itself (the
        // insertion-order result list): the reachable-struct count is small
        // (bounded by the registry, not the 200+ struct-chain depth that
        // would overflow a recursive walk), so a linear scan over `graph`
        // needs no parallel hash index and lets every node move into
        // `graph` without a clone.
        let mut worklist: Vec<String> = vec![root.to_string()];
        let mut graph: Vec<String> = Vec::new();
        while let Some(n) = worklist.pop() {
            if graph.contains(&n) {
                continue;
            }
            // Register the node's identity ClassDef in `descs` (RPython's
            // HostObject-identity cache, bookkeeper.py:361) via
            // `intern_class_by_qualname` -> `getuniqueclassdef`.  Idempotent:
            // interning resolves the type-root name to one stable
            // `HostObject`, so `getuniqueclassdef` returns the same
            // `ClassDef` on every call — the identity cache is the memo, no
            // name-keyed side table needed.
            let host = self.intern_class_by_qualname(&n);
            // Materialise the classdef lineage base-most first:
            // `getdesc` and `ClassDef::new` each recurse into their
            // base (classdesc.py:559 / :672) and both memoise, so
            // pre-seeding every base keeps the native recursion depth
            // at one frame even when the embedded-header chain is
            // registry-deep (same worklist discipline as `graph`).
            let mut lineage: Vec<HostObject> = vec![host];
            while let Some(base) = lineage
                .last()
                .and_then(|h| h.class_bases().and_then(|b| b.first().cloned()))
            {
                lineage.push(base);
            }
            for h in lineage.iter().rev() {
                self.getuniqueclassdef(h)?;
            }
            let referenced: Vec<String> = {
                let guard = self.pyre_struct_fields.borrow();
                if let Some(reg) = guard.as_ref() {
                    if let Some(fields) = reg.fields.get(&n) {
                        let mut out: Vec<String> = Vec::new();
                        for (field_name, field_ty) in fields {
                            if field_name == "__class__" {
                                continue;
                            }
                            collect_referenced_struct_names(field_ty, reg, &mut out);
                        }
                        out
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            };
            for r in referenced {
                if !graph.contains(&r) {
                    worklist.push(r);
                }
            }
            // `n` is no longer borrowed (registry lookup done); move it into
            // the ordered result list without a clone.
            graph.push(n);
        }
        // Pass 2 — project each reachable node's fields into its
        // `classdef.attrs`.  All nodes are published from pass 1, so a
        // field referencing another struct resolves to the registered
        // inner classdef.  Idempotent: the `Impossible` guard fills an attr
        // exactly once, so re-running after the field registry arrives
        // back-fills a root that was cached attrs-empty without disturbing
        // attrs already set (RPython grows class attrs as annotation
        // proceeds).
        for n in &graph {
            self.project_struct_rows(n)?;
        }
        // Drain structs first interned by `project_pyre_field_type`'s
        // bare-name arm during the loop above.  Pass-1's
        // `collect_referenced_struct_names` string parser and the
        // projector resolve type strings independently; a name only
        // the projector reaches would otherwise keep its untyped
        // FORCE shells until a LATER root intern projects it — a
        // non-monotonic mid-session attr flip for any block that
        // already read the shell (`setbinding: new value does not
        // contain old`).  Projecting here closes the window: rows
        // land before this struct_root call returns, i.e. before any
        // subject flow can read the class's attrs.
        loop {
            let next = self.pending_struct_row_projection.borrow_mut().pop();
            let Some(n) = next else { break };
            self.project_struct_rows(&n)?;
        }
        let host = self.intern_class_by_qualname(root);
        self.getuniqueclassdef(&host)
    }

    /// Resolve the subclass `ClassDef` for one variant of a Rust enum —
    /// a subclass of the enum's discriminant-only base. Composes the canonical
    /// interning primitives rather than minting a parallel enum-class
    /// registry: the base is the struct-root `ClassDef`
    /// ([`Self::getuniqueclassdef_for_struct_root`]); the variant is
    /// interned with that base as its sole `__bases__`
    /// ([`Self::intern_class_by_qualname_with_bases`]), so the resulting
    /// `ClassDef.basedef` chains through the base. That makes
    /// `commonbase(base, variant) == base`, which is exactly what
    /// `pairtype(SomeInstance, SomeInstance).improve` (binaryop.py:685)
    /// needs to narrow a `SomeInstance(base)` to `SomeInstance(variant)`.
    /// Identity is the canonical struct-root cache keyed by
    /// `canonical_struct_name`; the base `enum_root` and the
    /// `canon_root::variant` path ([`Self::intern_enum_variant_host`])
    /// each normalise to one stable `HostObject`, so repeated calls
    /// return the same `Rc` — and the variant `Rc` is the very one the
    /// prologue pre-mint numbered.
    pub fn getuniqueclassdef_for_enum_variant(
        self: &Rc<Self>,
        enum_root: &str,
        variant_name: &str,
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        // Base first — registers the discriminant-only root and publishes
        // its identity-keyed HostObject in the struct-root cache.
        self.getuniqueclassdef_for_struct_root(enum_root)?;
        // Variant subclass — interned WITH the base so ClassDesc::new
        // wires basedef. The SAME interning primitive the variant
        // CONSTRUCTOR path uses ([`Self::intern_enum_variant_host`]), so
        // both sides resolve one class object. Must precede any plain
        // intern of this path (intern_*_with_bases returns the
        // first-minted class verbatim).
        let variant_host = self.intern_enum_variant_host(enum_root, variant_name);
        // Project the variant's OWN payload rows onto the subclass —
        // the RPython sum-type layout where each subclass carries its
        // own fields (`rclass.py:82-88`), the base only the discriminant.
        // A getattr on a narrowed `SomeInstance(variant)` then resolves
        // the payload on the variant via the MRO (variant → base), with
        // the variant's own field type.  The rows are registered under
        // the canonical variant key in `front::mir`.
        let canon_root = majit_ir::descr::canonical_struct_name(enum_root);
        let variant_path = format!("{canon_root}::{variant_name}");
        self.project_struct_rows(&variant_path)?;
        self.getuniqueclassdef(&variant_host)
    }

    /// Register a trait family for receiver-driven method dispatch: a base
    /// `ClassDef` `base_root` with each `(impl_root, members)` interned as a
    /// subclass, carrying method MEMBERS so `getattr(instance, method)`
    /// resolves to a `MethodDesc` PBC family — the RPython instance-method
    /// [`MethodsPBCRepr`](crate::translator::rtyper::rpbc::MethodsPBCRepr)
    /// dispatch path.
    ///
    /// Mirrors [`Self::getuniqueclassdef_for_enum_variant`] (a base plus
    /// subclasses interned WITH that base as their sole `__bases__`, so
    /// `ClassDef.basedef`/`subdefs`/`getmro` chain), but seeds method members
    /// through [`HostObject::new_class_with_members`].  Plain pyre struct
    /// classdefs are minted member-less by
    /// [`Self::intern_class_by_qualname`] (`HostObject::new_class`) and carry
    /// only fields (`project_struct_rows`); a `dyn Trait` receiver therefore
    /// annotates to a classdef-less shell and every `getattr` on it fails.
    /// The members seeded here are what `ClassDesc::find_source_for` reads via
    /// `pyobj.class_get(name)` → `add_source_attribute`, so the trait methods
    /// become classdict attributes and `s_getattr` yields the PBC family.
    ///
    /// `base_members` (trait default-body methods) attach to the base;
    /// each impl's `members` (its required-method overrides) attach to that
    /// subclass — mirroring the trait's default-vs-required split, so the MRO
    /// getattr binds each method to its correct owner.
    ///
    /// OPT-IN: only a trait registered through this entry point gets a
    /// member-carrying host in `pyre_struct_root_classes`.  Every other
    /// (unregistered) multi-impl trait keeps its current classdef-less /
    /// fail-loud disposition, so this cannot perturb the annotation of pyre
    /// production dispatch that was never registered.
    ///
    /// Returns the base `ClassDef`.
    pub fn register_trait_family(
        self: &Rc<Self>,
        base_root: &str,
        base_members: HashMap<String, ConstValue>,
        impls: Vec<(String, HashMap<String, ConstValue>)>,
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        // Base first — its identity-keyed HostObject must be published in
        // `pyre_struct_root_classes` before the subclasses intern, so each
        // subclass's `__bases__` resolves to this exact base host (the same
        // discipline `getuniqueclassdef_for_enum_variant` relies on).
        let base_key = majit_ir::descr::canonical_struct_name(base_root);
        let base_host = crate::flowspace::model::HostObject::new_class_with_members(
            base_key.clone(),
            Vec::new(),
            base_members,
        );
        self.pyre_struct_root_classes
            .borrow_mut()
            .insert(base_key, base_host.clone());
        // Index the base by the raw `base_root` spelling too, so the
        // receiver seed (`derive_subject_inputcells`) can match a
        // `dyn Trait` receiver's `class_root` (the trait's `name_path()`)
        // directly without re-canonicalising.
        self.pyre_trait_family_bases
            .borrow_mut()
            .insert(base_root.to_string(), base_host.clone());
        let base_cd = self.getuniqueclassdef(&base_host)?;
        // Subclasses — interned WITH the base host so `ClassDesc::new` wires
        // `basedef` and `_init_classdef` pushes each into `base.subdefs`.
        for (impl_root, members) in impls {
            let impl_key = majit_ir::descr::canonical_struct_name(&impl_root);
            let impl_host = crate::flowspace::model::HostObject::new_class_with_members(
                impl_key.clone(),
                vec![base_host.clone()],
                members,
            );
            self.pyre_struct_root_classes
                .borrow_mut()
                .insert(impl_key, impl_host.clone());
            self.getuniqueclassdef(&impl_host)?;
        }
        Ok(base_cd)
    }

    /// Intern (first-mint-wins) and return the canonical variant-subclass
    /// `HostObject` for `{enum_root}::{variant_name}`, wiring the enum's
    /// discriminant-only base as its sole `__bases__` (`rclass.py:82-88`).
    ///
    /// Both the discriminant-narrowing path
    /// ([`Self::getuniqueclassdef_for_enum_variant`]) and the variant
    /// CONSTRUCTOR path (`flowspace_adapter`'s `SyntheticTransparentCtor`
    /// arm) intern through here, so a constructed `Some(x)` and a matched
    /// `Some(x)` resolve to ONE class object — the single-class-per-variant
    /// identity RPython gets from one Python class object
    /// (`bookkeeper.py:339` keys `getuniqueclassdef` by the class object,
    /// not a name string).  Splitting them into sibling classdefs would
    /// union to the base (losing the payload narrowing) and disagree on
    /// the payload attr/field owner.
    ///
    /// The cache key is `canonical_struct_name(enum_root)::variant_name`.
    /// A leaf `enum_root` (`"Color"`, the ctor side, which carries the
    /// enum leaf as its owner tail) and a canonical `enum_root`
    /// (`"module::Color"`, the narrowing side, which reads the base
    /// classdef's canonical name) normalise to the same key because
    /// `canonical_struct_name` resolves the leaf through
    /// `STRUCT_ORIGIN_REGISTRY` — the same normalisation the base itself
    /// went through.
    pub fn intern_enum_variant_host(
        self: &Rc<Self>,
        enum_root: &str,
        variant_name: &str,
    ) -> HostObject {
        let canon_root = majit_ir::descr::canonical_struct_name(enum_root);
        let base_host = self.intern_class_by_qualname(&canon_root);
        let variant_path = format!("{canon_root}::{variant_name}");
        self.intern_class_by_qualname_with_bases(&variant_path, vec![base_host])
    }

    /// Build the discriminant→variant narrowing `knowntypedata` for a
    /// Rust enum's `__discriminant` read.  Each entry is keyed by the
    /// integer tag value and narrows the `receiver` variable to the
    /// variant subclass ([`Self::getuniqueclassdef_for_enum_variant`]),
    /// so a `match disc { k => ... }` switch refines `SomeInstance(base)`
    /// to `SomeInstance(variant_k)` in arm `k` through `follow_link`'s
    /// `improve` (binaryop.py:685).  `enum_root` is the receiver class
    /// name; the variant table is dual-keyed by qualified path and bare
    /// leaf, so either spelling resolves.  Returns `None` when the class
    /// is not a registered enum root (the read then stays a plain
    /// `SomeInteger`).
    pub fn enum_variant_narrowing_knowntypedata(
        self: &Rc<Self>,
        enum_root: &str,
        receiver: &Rc<crate::flowspace::model::Variable>,
    ) -> Option<super::model::KnownTypeData> {
        let by_discr = {
            // The discriminant→variant table is keyed by the bare
            // charon-template root (one template per generic ADT), so a
            // per-instantiation receiver name (`Result<bool>`) must drop
            // its `<…>` suffix to resolve.  Bare names pass through
            // unchanged.
            let lookup_root = majit_ir::descr::strip_instantiation_suffix(enum_root);
            let guard = self.pyre_enum_variant_by_discriminant.borrow();
            let map = guard.as_ref()?;
            map.get(lookup_root)
                .or_else(|| {
                    lookup_root
                        .rsplit("::")
                        .next()
                        .and_then(|leaf| map.get(leaf))
                })
                .cloned()?
        };
        // A receiver already narrowed to a variant subclass
        // (`Option<T>::None`) reaches this read with `enum_root` carrying
        // the `::variant` tail; appending another variant
        // (`Option<T>::None::None`) deepens the per-instantiation classdef
        // every pass, so the lattice never reaches a fixpoint.  Re-narrow
        // against the base enum — strip a trailing `::<variant>` segment so
        // the discriminant maps back to the SAME variant subclasses
        // (idempotent on re-read).
        let base_enum_root: &str = {
            let mut base = enum_root;
            for (_discr, variant_name) in &by_discr {
                if let Some(stripped) = base.strip_suffix(&format!("::{variant_name}")) {
                    base = stripped;
                    break;
                }
            }
            base
        };
        let mut ktd = super::model::KnownTypeData::new();
        for (discr, variant_name) in &by_discr {
            let variant_cd = self
                .getuniqueclassdef_for_enum_variant(base_enum_root, variant_name)
                .ok()?;
            let s_variant = SomeValue::Instance(super::model::SomeInstance::new(
                Some(variant_cd),
                false,
                std::collections::BTreeMap::new(),
            ));
            super::model::add_knowntypedata(
                &mut ktd,
                super::model::ExitCaseKey::Int(*discr),
                std::slice::from_ref(receiver),
                s_variant,
            );
        }
        Some(ktd)
    }

    /// Project one struct's registry rows into its `ClassDef.attrs`
    /// — the pass-2 body of [`Self::getuniqueclassdef_for_struct_root`].
    fn project_struct_rows(self: &Rc<Self>, n: &str) -> Result<(), AnnotatorError> {
        // A monomorphised generic spelling (`Option<*mut PyObject>`,
        // `Result<Tuple>`) shares the bare template's rows; the registry
        // only carries the un-suffixed key (`Option`/`option::Option`).
        // Strip the `<…>` argument span so the lookup resolves under the
        // template key — matching `StructFieldRegistry::lookup_fields`,
        // which the bare-`reg.fields.get` here bypasses.
        let fields: Option<Vec<(String, String)>> = {
            let guard = self.pyre_struct_fields.borrow();
            guard.as_ref().and_then(|r| {
                r.fields
                    .get(majit_ir::descr::strip_generic_args(n).as_ref())
                    .cloned()
            })
        };
        let Some(fields) = fields else {
            return Ok(());
        };
        // Bare and qualified spellings of one struct intern to the
        // same canonical class — project its rows once.
        let canonical = majit_ir::descr::canonical_struct_name(n);
        if !self.projected_struct_rows.borrow_mut().insert(canonical) {
            return Ok(());
        }
        let host = self.intern_class_by_qualname(n);
        let classdef = self.getuniqueclassdef(&host)?;
        for (field_name, field_ty) in &fields {
            if field_name == "__class__" {
                continue;
            }
            let s_value = self.project_pyre_field_type(field_ty);
            let mut classdef_mut = classdef.borrow_mut();
            let attr = classdef_mut
                .attrs
                .entry(field_name.clone())
                .or_insert_with(|| super::classdesc::Attribute::new(field_name.clone()));
            // The slot may also hold the untyped FORCE_ATTRIBUTES
            // shell: `register_struct_fields` seeds Ref-typed rows
            // through `valuetype_to_someshell`, which renders every
            // `ValueType::Ref(_)` as the bare
            // `SomeInstance(classdef=None)` before this projection
            // runs (`_init_classdef` applies the force list
            // eagerly).  That shell carries strictly less
            // information than this registry projection (typed
            // lists, dicts, strings, classed instances), so
            // replace it the same way the `Impossible` placeholder
            // is filled — but ONLY that exact shell; any other
            // value was produced by real annotation flow and stays
            // (RPython grows class attrs monotonically).
            let is_untyped_force_shell = matches!(
                &attr.s_value,
                SomeValue::Instance(inst)
                    if inst.classdef.is_none()
                        && !inst.can_be_none
                        && inst.flags.is_empty()
                        && inst.base.const_box.is_none()
            );
            if matches!(attr.s_value, SomeValue::Impossible)
                || (is_untyped_force_shell && !matches!(s_value, SomeValue::Impossible))
            {
                attr.s_value = s_value;
            }
        }
        Ok(())
    }

    /// Resolve a struct type-root name to its canonical host class
    /// `HostObject`, minting it on first sight and caching by name in
    /// [`Self::pyre_struct_root_classes`].  Because `HostObject`
    /// equality is `Arc` pointer identity, this interning is what makes
    /// `getuniqueclassdef` return the SAME `ClassDef` for repeated
    /// lookups of one type-root — the pyre analog of resolving a type
    /// name to one class object before `getuniqueclassdef(cls)`.
    ///
    /// Embedded-header subclassing: a struct whose FIRST field embeds
    /// another registered type root BY VALUE (`W_IntObject { ob_header:
    /// PyObject, intval: i64 }`, `LoopBlock { base: FrameBlock }`) is
    /// the Rust spelling of upstream's class hierarchy
    /// (`W_IntObject(W_Root)`, `LoopBlock(FrameBlock)`) — base the
    /// minted class on the header root's class, the same shared-Arc
    /// shape [`Self::intern_class_by_qualname_with_bases`] uses for
    /// enum variants, so `ClassDef::commonbase` unions siblings to the
    /// header root and `cast_pointer`-style refinement connects the
    /// `InstanceRepr`s.  The base is interned under its BARE LEAF —
    /// the spelling `tyref_class_root` seeds params with — so the
    /// subclass points at the same class Arc the rest of the session
    /// uses.
    ///
    /// Each chain node's cache key is resolved through
    /// `canonical_struct_name`, so the bare and qualified spellings of
    /// one struct intern to one `HostObject` — name strings are a
    /// resolution input, never the identity itself
    /// (`getuniqueclassdef(cls)` keys by class object,
    /// bookkeeper.py:339).  Dotted qualnames pass through unchanged; a
    /// bare leaf whose origin was tombstoned by
    /// `harden_duplicate_leaf_metadata` stays unresolved and keeps an
    /// attrs-empty classdef because the field registry withdrew the
    /// bare alias.
    ///
    /// The header chain is collected iteratively with a revisit guard:
    /// registry field-type strings strip references
    /// (`tyref_to_ast_string` renders `&T` as `T`), so a reference
    /// cycle (`A.next: &B`, `B.prev: &A`) is indistinguishable from
    /// by-value embedding here — a revisited leaf ends the chain (a
    /// reference loop is linkage, not a header hierarchy), and a
    /// pathological registry chain must not exhaust the native stack
    /// (same O(reachable)-heap discipline as
    /// [`Self::getuniqueclassdef_for_struct_root`]'s worklist).
    pub fn intern_class_by_qualname(self: &Rc<Self>, name: &str) -> HostObject {
        // Collect `name`'s embedded-header chain, outermost first,
        // stopping at an already-interned class (its Arc becomes the
        // deepest base) or at a chain end (scalar / unregistered /
        // revisited first field).  The class-cache key is the canonical
        // `module::Leaf` (`canonical_struct_name`); the field-registry
        // lookup uses the raw spelling (the registry carries both
        // qualified and bare keys).
        let mut chain: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut base_host: Option<HostObject> = None;
        let mut cur = name.to_string();
        loop {
            let key = majit_ir::descr::canonical_struct_name(&cur);
            if let Some(existing) = self.pyre_struct_root_classes.borrow().get(&key) {
                if chain.is_empty() {
                    return existing.clone();
                }
                base_host = Some(existing.clone());
                break;
            }
            seen.insert(cur.rsplit("::").next().unwrap_or(&cur).to_string());
            chain.push(cur.clone());
            let next = self.pyre_struct_fields.borrow().as_ref().and_then(|reg| {
                // An enum-variant key `{enum_base}::{variant}` subclasses its
                // enum base — the discriminant-only sum-type root
                // (`rclass.py:82-88`).  Checked before the header convention
                // because a variant's first field is its payload, not a
                // header: without this the session prologue
                // (`PyreCallRegistry::ensure_session`), which pre-mints every
                // `struct_fields` key including the variant keys, would mint
                // the variant base-less, and first-mint-wins would freeze
                // that, dropping the subclass link the narrowing relies on.
                if let Some((parent, _variant)) = cur.rsplit_once("::") {
                    let parent_leaf = parent.rsplit("::").next().unwrap_or(parent);
                    if reg.is_enum_base(parent) && !seen.contains(parent_leaf) {
                        return Some(parent.to_string());
                    }
                }
                let (first_name, first_ty) = reg.fields.get(&cur)?.first()?;
                // Only header-conventional first fields mark subclassing
                // (`ob_header: PyObject` / `base: FrameBlock`).  A
                // by-value first field of another registered type is
                // otherwise plain composition (`PyError { kind:
                // PyErrorKind, … }`), not a class hierarchy.
                if first_name != "ob_header" && first_name != "base" {
                    return None;
                }
                let leaf = first_ty.rsplit("::").next().unwrap_or(first_ty);
                (reg.fields.contains_key(leaf) && !seen.contains(leaf)).then(|| leaf.to_string())
            });
            match next {
                Some(n) => cur = n,
                None => break,
            }
        }
        // Mint base-most first so each class links its base's Arc.
        for node in chain.iter().rev() {
            let key = majit_ir::descr::canonical_struct_name(node);
            let bases = base_host.iter().cloned().collect();
            let host = crate::flowspace::model::HostObject::new_class(key.clone(), bases);
            self.pyre_struct_root_classes
                .borrow_mut()
                .insert(key, host.clone());
            base_host = Some(host);
        }
        base_host.expect("chain holds at least `name` itself")
    }

    /// [`Self::intern_class_by_qualname`] with explicit `__bases__` for
    /// the first mint.  First-mint wins: a cache hit returns the
    /// existing class regardless of `bases`, so every site minting a
    /// given qualname must derive the same bases (the variant-ctor arm
    /// in `flowspace_adapter::translate_op` derives them from the
    /// ctor's own `owner_path`, which is identical at every site).
    pub fn intern_class_by_qualname_with_bases(
        self: &Rc<Self>,
        name: &str,
        bases: Vec<HostObject>,
    ) -> HostObject {
        let key = majit_ir::descr::canonical_struct_name(name);
        if let Some(existing) = self.pyre_struct_root_classes.borrow().get(&key) {
            return existing.clone();
        }
        let host = crate::flowspace::model::HostObject::new_class(key.clone(), bases);
        self.pyre_struct_root_classes
            .borrow_mut()
            .insert(key, host.clone());
        host
    }

    /// True when `name` is a type-root key in the snapshot
    /// struct-field registry ([`Self::pyre_struct_fields`]).  Enums and
    /// structs both register under their qualified path AND bare leaf
    /// (`front/mir.rs` `TypeDeclKind::Enum` / `Struct` arms), so a
    /// ctor's `owner_path` last segment answers true exactly when the
    /// owner is itself an ADT (an enum-variant ctor) rather than a
    /// module (a struct ctor).
    pub fn is_pyre_struct_root(&self, name: &str) -> bool {
        self.pyre_struct_fields
            .borrow()
            .as_ref()
            .is_some_and(|reg| reg.fields.contains_key(name))
    }

    /// True when type-root `root`'s registry rows include a field named
    /// `name`.  The cutover's class-dict method population consults this
    /// so a method member never shadows a same-named instance field — a
    /// function source for a field attribute would union-conflict in
    /// `generalize_attr`.
    pub fn pyre_struct_root_has_field(&self, root: &str, name: &str) -> bool {
        self.pyre_struct_fields
            .borrow()
            .as_ref()
            .is_some_and(|reg| {
                reg.fields
                    .get(root)
                    .is_some_and(|rows| rows.iter().any(|(field, _)| field == name))
            })
    }

    /// TODO: no upstream equivalent.  Project a Rust type
    /// string (`"Vec<i32>"`, `"Option<PyFrame>"`, `"HashMap<String,
    /// Box<W_Obj>>"`, …) into a `SomeValue` matching what RPython
    /// `s_getattr` would observe for a class attribute seeded with a
    /// value of that type.  Used by
    /// [`Self::getuniqueclassdef_for_struct_root`] to project the
    /// registered `ClassDef.attrs`.  Bare named types
    /// (`PyFrame`, `W_DictObject`) resolve to
    /// `SomeInstance(stub)` only when the registry contains a matching
    /// entry; unknown bare names fall to `Impossible`.
    pub fn project_pyre_field_type(self: &Rc<Self>, field_ty: &str) -> SomeValue {
        let t = field_ty.trim();
        // A raw-pointer field (`*const T` / `*mut T`) holds a one-word
        // pointer, not a `T`.  Projecting the pointee is right for an
        // aggregate target (`*mut Vec<T>` / `*mut Struct`): the structural
        // shell is already GcRef-kind and downstream element/field access
        // relies on it.  But for a scalar / unit pointee the pointee shell
        // (Signed / Void / …) discards the pointer-ness and diverges from
        // the legacy walker, which folds every raw-pointer field-read to
        // `Ref` → GcRef (the FieldRead op `ty` is
        // `tyref_to_value_type(raw ptr) = Ref(None)`).  Mirror that fold —
        // and the FORCE_ATTRIBUTES `tyref_to_attr_value_type` raw-ptr →
        // `Ref(None)` projection — by shelling a scalar/unit-pointee raw
        // pointer as the conservative classdef-less Ref instance
        // (`getkind = 'ref'`), the GC-safe choice for a field that may
        // hold a managed pointer.
        if let Some(pointee) = t
            .strip_prefix("*const ")
            .or_else(|| t.strip_prefix("*mut "))
        {
            let s_pointee = self.project_pyre_field_type(pointee.trim());
            if matches!(
                s_pointee,
                SomeValue::Integer(_)
                    | SomeValue::Bool(_)
                    | SomeValue::Float(_)
                    | SomeValue::SingleFloat(_)
                    | SomeValue::Char(_)
                    | SomeValue::None_(_)
            ) {
                return SomeValue::Instance(super::model::SomeInstance::new(
                    None,
                    false,
                    std::collections::BTreeMap::new(),
                ));
            }
            return s_pointee;
        }
        let stripped = t
            .trim_start_matches('&')
            .trim_start_matches("mut ")
            .trim_start_matches("*const ")
            .trim_start_matches("*mut ")
            .trim();
        match stripped {
            "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize" => {
                return super::model::s_int();
            }
            "f32" => return SomeValue::SingleFloat(super::model::SomeSingleFloat::new()),
            "f64" => return SomeValue::Float(super::model::SomeFloat::new()),
            "bool" => return super::model::s_bool(),
            // Rust `String`/`str` fields are byte string values: string
            // literals lower through `__str_const` into
            // `ConstValue::ByteStr` constants (stamped `Ptr(STR)`, the
            // rtyper's byte `StringRepr`), so the field projection must
            // meet the same `SomeString` — projecting `s_unicode0` here
            // raised `str ∪ unicode` where a literal merged into a
            // String-typed attr cell.
            "String" | "str" => return super::model::s_str0(),
            "char" => return SomeValue::Char(super::model::SomeChar::new(false)),
            "()" => return super::model::s_none(),
            _ => {}
        }
        for list_wrapper in ["Vec<", "VecDeque<"] {
            if let Some(inner) = strip_generic_one(stripped, list_wrapper) {
                let s_inner = self.project_pyre_field_type(inner);
                let listdef =
                    super::listdef::ListDef::new(Some(self.clone()), s_inner, false, false);
                return SomeValue::List(super::model::SomeList::new(listdef));
            }
        }
        if let Some(rest) = stripped.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            let inner = match rest.find(';') {
                Some(semi) => rest[..semi].trim(),
                None => rest.trim(),
            };
            let s_inner = self.project_pyre_field_type(inner);
            let listdef = super::listdef::ListDef::new(Some(self.clone()), s_inner, false, false);
            return SomeValue::List(super::model::SomeList::new(listdef));
        }
        if let Some(inner) = strip_generic_one(stripped, "Option<") {
            let s_inner = self.project_pyre_field_type(inner);
            let s_none = super::model::s_none();
            return super::model::unionof([&s_inner, &s_none]).unwrap_or(SomeValue::Impossible);
        }
        if let Some(inner) = strip_generic_one(stripped, "Result<") {
            let parts = split_generic_args(inner);
            if parts.len() == 2 {
                let s_ok = self.project_pyre_field_type(parts[0]);
                let s_err = self.project_pyre_field_type(parts[1]);
                return super::model::unionof([&s_ok, &s_err]).unwrap_or(SomeValue::Impossible);
            }
        }
        for dict_wrapper in ["HashMap<", "BTreeMap<", "IndexMap<"] {
            if let Some(inner) = strip_generic_one(stripped, dict_wrapper) {
                let parts = split_generic_args(inner);
                if parts.len() == 2 {
                    let s_key = self.project_pyre_field_type(parts[0]);
                    let s_val = self.project_pyre_field_type(parts[1]);
                    let dictdef = super::dictdef::DictDef::new(
                        Some(self.clone()),
                        s_key,
                        s_val,
                        false,
                        false,
                        false,
                    );
                    return SomeValue::Dict(super::model::SomeDict::new(dictdef));
                }
            }
        }
        for set_wrapper in ["HashSet<", "BTreeSet<", "IndexSet<"] {
            if let Some(inner) = strip_generic_one(stripped, set_wrapper) {
                let s_key = self.project_pyre_field_type(inner);
                let s_val = super::model::s_none();
                let dictdef = super::dictdef::DictDef::new(
                    Some(self.clone()),
                    s_key,
                    s_val,
                    false,
                    false,
                    false,
                );
                return SomeValue::Dict(super::model::SomeDict::new(dictdef));
            }
        }
        if let Some(inner) = stripped.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
            let parts = split_generic_args(inner);
            if !parts.is_empty() {
                let items: Vec<SomeValue> = parts
                    .iter()
                    .map(|p| self.project_pyre_field_type(p))
                    .collect();
                return SomeValue::Tuple(super::model::SomeTuple::new(items));
            }
        }
        for wrapper in [
            "Rc<",
            "Arc<",
            "Box<",
            "RefCell<",
            "Cell<",
            "Pin<",
            "NonNull<",
            "NonZero<",
            "MaybeUninit<",
            "ManuallyDrop<",
            "UnsafeCell<",
            "Wrapping<",
            "Reverse<",
        ] {
            if let Some(inner) = strip_generic_one(stripped, wrapper) {
                return self.project_pyre_field_type(inner);
            }
        }
        if let Some(inner) = strip_generic_one(stripped, "Cow<") {
            let parts = split_generic_args(inner);
            if parts.len() == 2 {
                return self.project_pyre_field_type(parts[1]);
            }
        }
        // `FixedObjectArray { len: usize, _items: [PyObjectRef; 0] }` is a
        // flexible object array.  An `arr[idx]` access lowers (front-end
        // ArrayRead -> flowspace `getitem`) onto the receiver, so the
        // receiver must model as the element list — not the wrapping
        // struct, whose `getitem` over `SomeInstance(None)` would rewrite
        // to `getattr("__getitem__")` and dead-end.  Project the `_items`
        // flexible-array tail: `[PyObjectRef; 0]` resolves through the
        // array arm above to `SomeList(item=SomeInstance(PyObjectRef),
        // resized=false)`, giving a typed element rather than a
        // classdef-less stub.
        if majit_ir::descr::canonical_struct_name(stripped) == "object_array::FixedObjectArray" {
            let items_ty = self
                .pyre_struct_fields
                .borrow()
                .as_ref()
                .and_then(|reg| reg.field_type(stripped, "_items").map(str::to_string));
            if let Some(items_ty) = items_ty {
                return self.project_pyre_field_type(&items_ty);
            }
        }
        let registered = {
            let guard = self.pyre_struct_fields.borrow();
            guard
                .as_ref()
                .map(|r| r.fields.contains_key(stripped))
                .unwrap_or(false)
        };
        if !registered {
            return SomeValue::Impossible;
        }
        // Resolve the struct-typed field to its identity-registered
        // `ClassDef` through `descs` (`intern_class_by_qualname` ->
        // `getuniqueclassdef`), the same identity path
        // `getuniqueclassdef_for_struct_root` registers it on.  Pass-1 of
        // that method has already registered every reachable struct, so this
        // returns the same `ClassDef` Rc whose `attrs` pass-2 fills in place.
        //
        // Pass-1's string parser may MISS a name this arm resolves
        // (the two walk type strings independently); queue such a
        // first-sight struct so the enclosing struct_root call's
        // pending drain projects its rows before any subject flow can
        // read the class's untyped FORCE shells.
        {
            let canonical = majit_ir::descr::canonical_struct_name(stripped);
            if !self.projected_struct_rows.borrow().contains(&canonical) {
                self.pending_struct_row_projection
                    .borrow_mut()
                    .push(stripped.to_string());
            }
        }
        let host = self.intern_class_by_qualname(stripped);
        match self.getuniqueclassdef(&host) {
            Ok(classdef) => SomeValue::Instance(super::model::SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
            Err(_) => SomeValue::Impossible,
        }
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
        // bookkeeper.py:458-466 `pbc_getattr(self, pbc, s_attr)` —
        // upstream `attr = s_attr.const` with no `isinstance(...,
        // str)` gate. Use [`as_text`] so both `ByteStr` and `UniStr`
        // pass through, mirroring the upstream constant-only check
        // (string equality and the s_read_attribute lookup work
        // across Python 2's bytes/unicode boundary).
        let Some(attr_name) = attr_const.as_text().map(str::to_owned) else {
            return Err(AnnotatorError::new(format!(
                "pbc_getattr: attr must be a string, got {attr_const:?}"
            ))
            .into());
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
                    ConstValue::Tuple(vs) | ConstValue::List(vs) => {
                        vs.iter().any(|v| v.as_text() == Some(attr_name.as_str()))
                    }
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
    pub(crate) fn pbc_call(
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
                // upstream `desc.pycall` virtual dispatch: a memo desc
                // runs `MemoDesc.pycall` (union-of-results annotation, or
                // the dispatch graph's return var); a plain function runs
                // `FunctionDesc.pycall`.
                super::description::DescEntry::Func(fe) => {
                    if let Some(md) = fe.as_memo() {
                        md.borrow().pycall(
                            whence.clone(),
                            args,
                            &s_previous_result,
                            op_key.clone(),
                        )?
                    } else {
                        fe.func().borrow().pycall(
                            whence.clone(),
                            args,
                            &s_previous_result,
                            op_key.clone(),
                        )?
                    }
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

    /// rlib/jit.py:903-914 — fold `kwds_s` into
    /// [`Self::_jit_annotation_cache`] under `driver`, unioning
    /// against any previous annotation seen for the same kwarg key.
    ///
    /// Upstream:
    ///
    /// ```python
    /// for key, s_value in kwds_s.items():
    ///     s_previous = cache.get(key, annmodel.s_ImpossibleValue)
    ///     s_value = annmodel.unionof(s_previous, s_value)
    ///     cache[key] = s_value
    /// ```
    ///
    /// `kwds_s` keys carry the upstream `'s_'` prefix (rlib/jit.py:895
    /// `expected = ['s_' + name for name in ...]`), so callers must
    /// preserve the prefix to keep the cache shape line-by-line with
    /// upstream.
    pub fn union_jit_annotation_kwds(
        self: &Rc<Self>,
        driver: &HostObject,
        kwds_s: &HashMap<String, Option<SomeValue>>,
    ) -> Result<(), AnnotatorError> {
        let mut cache_outer = self._jit_annotation_cache.borrow_mut();
        let cache = cache_outer.entry(driver.clone()).or_default();
        for (key, s_value) in kwds_s {
            // upstream `bookkeeper.py union_jit_annotation_kwds` iterates
            // `kwds_s.items()` and `union`s each `s_value` into the cache.
            // A `None` value (unbound annotation, mirroring
            // `annotator.annotation(v) is None`) contributes nothing to
            // the union — preserve the previous cache entry verbatim
            // instead of widening it through `union(prev, None)`.
            let Some(s_value) = s_value else { continue };
            let s_previous = cache.remove(key).unwrap_or_else(s_impossible_value);
            let s_unioned = union(&s_previous, s_value).map_err(|e| {
                AnnotatorError::new(format!(
                    "_jit_annotation_cache union for key {key:?}: {e:?}"
                ))
            })?;
            cache.insert(key.clone(), s_unioned);
        }
        Ok(())
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
    pub(crate) fn emulate_pbc_call(
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
    fn immutable_list_with_key(
        self: &Rc<Self>,
        key: Option<&Constant>,
        x: &ConstValue,
        items: &[ConstValue],
    ) -> Result<SomeValue, AnnotatorError> {
        if let Some(key) = key
            && let Some(hit) = self.immutable_cache.borrow().get(&key.id).cloned()
        {
            return Ok(hit);
        }
        let listdef = ListDef::new(Some(self.clone()), SomeValue::Impossible, false, false);
        let mut result = SomeList::new(listdef.clone());
        result.base.const_box = Some(key.cloned().unwrap_or_else(|| Constant::new(x.clone())));
        if let Some(key) = key {
            self.immutable_cache
                .borrow_mut()
                .insert(key.id, SomeValue::List(result.clone()));
        }
        for e in items {
            let s_e = self.immutablevalue(e)?;
            listdef
                .generalize(&s_e)
                .map_err(|e| AnnotatorError::new(e.msg))?;
        }
        Ok(SomeValue::List(result))
    }

    fn immutable_dict_with_key(
        self: &Rc<Self>,
        key: Option<&Constant>,
        x: &ConstValue,
        items: &HashMap<ConstValue, ConstValue>,
    ) -> Result<SomeValue, AnnotatorError> {
        if let Some(key) = key
            && let Some(hit) = self.immutable_cache.borrow().get(&key.id).cloned()
        {
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
        result.base.const_box = Some(key.cloned().unwrap_or_else(|| Constant::new(x.clone())));
        if let Some(key) = key {
            self.immutable_cache
                .borrow_mut()
                .insert(key.id, SomeValue::Dict(result.clone()));
        }
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

    /// RPython `Bookkeeper.immutablevalue(Constant(x))` for callers that
    /// still have the original `Constant` object and therefore its
    /// identity. List/dict branches must preserve that identity so
    /// `self.immutable_cache[key]` matches upstream.
    pub fn immutableconstant(self: &Rc<Self>, c: &Constant) -> Result<SomeValue, AnnotatorError> {
        match &c.value {
            ConstValue::List(items) => self.immutable_list_with_key(Some(c), &c.value, items),
            ConstValue::Dict(items) => self.immutable_dict_with_key(Some(c), &c.value, items),
            _ => self.immutablevalue(&c.value),
        }
    }

    /// Input is a flowspace [`ConstValue`] — the Rust-side counterpart
    /// to upstream's Python constant. Primitive branches (bool / int /
    /// float / str / char / unicode / bytearray / tuple / None) are
    /// ported line-by-line. Callers that still have the original
    /// [`Constant`] object should use [`Self::immutableconstant`] so
    /// list/dict branches can preserve `Constant(x)` identity.
    ///
    /// The function / class / bound-method / weakref / frozen-PBC /
    /// property / instance branches (bookkeeper.py:218-348) route
    /// through [`Self::immutablevalue_hostobject`]. The `_ptr`
    /// extregistry branch (bookkeeper.py:312-314 via lltype.py:_ptrEntry)
    /// routes through `translator/rtyper/extregistry.rs`.
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
            ConstValue::ByteStr(s) => {
                let no_nul = !s.contains(&0);
                let result = if s.len() == 1 {
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
            ConstValue::UniStr(s) => {
                let no_nul = !s.contains('\x00');
                let result = if s.chars().count() == 1 {
                    let mut ch = SomeUnicodeCodePoint::new(no_nul);
                    ch.inner.base.const_box = Some(Constant::new(x.clone()));
                    SomeValue::UnicodeCodePoint(ch)
                } else {
                    let mut st = SomeUnicodeString::new(false, no_nul);
                    st.inner.base.const_box = Some(Constant::new(x.clone()));
                    SomeValue::UnicodeString(st)
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
            ConstValue::List(items) => self.immutable_list_with_key(None, x, items),
            ConstValue::Dict(items) => self.immutable_dict_with_key(None, x, items),
            ConstValue::HostObject(obj) => self.immutablevalue_hostobject(obj, x),
            ConstValue::Function(func) => {
                let host = HostObject::new_user_function((**func).clone());
                self.immutablevalue_hostobject(&host, &ConstValue::HostObject(host.clone()))
            }
            ConstValue::LLPtr(_) => {
                assert!(crate::translator::rtyper::extregistry::is_registered(x));
                let entry = crate::translator::rtyper::extregistry::lookup(x).expect(
                    "Bookkeeper.immutablevalue(LLPtr): extregistry.lookup must succeed after \
                     is_registered",
                );
                let mut result = entry.compute_annotation_bk(self)?;
                // bookkeeper.py:350 — after the extregistry branch,
                // upstream falls through to the common `result.const = x`.
                // Keep that const_box on the SomePtr itself; otherwise
                // SomePtr.bool/is_immutable_constant diverge from RPython.
                match &mut result {
                    SomeValue::Ptr(ptr) => {
                        ptr.base.const_box = Some(Constant::new(x.clone()));
                    }
                    other => panic!(
                        "Bookkeeper.immutablevalue(LLPtr): _ptrEntry must return SomePtr, got {other:?}"
                    ),
                }
                Ok(result)
            }
            ConstValue::AddressOffset(_) => {
                // AddressOffset.annotation() returns SomeInteger()
                let mut s = SomeInteger::new(false, false);
                s.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::Integer(s))
            }
            ConstValue::InheritanceId { .. } => {
                // A symbolic inheritance id is a Signed integer marker.
                let mut s = SomeInteger::new(false, false);
                s.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::Integer(s))
            }
            ConstValue::LLAddress(_) => {
                // `isinstance(x, llmemory.fakeaddress)` arm of
                // `bookkeeper.py immutablevalue` — a prebuilt address
                // constant annotates as `SomeAddress()`; the const_box
                // keeps the concrete address so `is_null_address` and
                // constant folding can read it back.
                let mut s = super::model::SomeAddress::new();
                s.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::Address(s))
            }
            ConstValue::Code(_)
            | ConstValue::Graphs(_)
            | ConstValue::LowLevelType(_)
            | ConstValue::SpecTag(_)
            | ConstValue::Atom(_)
            | ConstValue::Placeholder => {
                // Code / Graphs / SpecTag / Atom / Placeholder cover
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
        // in BUILTIN_ANALYZERS: result = SomeBuiltin(...)`.
        if ishashable(obj) && super::builtin::is_registered(obj.qualname()) {
            let module_name = match crate::flowspace::model::host_getattr(obj, "__module__") {
                Ok(value) => value.into_text().unwrap_or_else(|| "unknown".to_string()),
                Err(crate::flowspace::model::HostGetAttrError::Missing) => "unknown".to_string(),
                Err(crate::flowspace::model::HostGetAttrError::Unsupported) => {
                    return Err(AnnotatorError::new(format!(
                        "Bookkeeper.immutablevalue(SomeBuiltin): getattr({:?}, '__module__') unsupported",
                        obj.qualname()
                    )));
                }
            };
            let name = match crate::flowspace::model::host_getattr(obj, "__name__") {
                Ok(value) => value.as_text().map(str::to_owned).ok_or_else(|| {
                    AnnotatorError::new(format!(
                        "Bookkeeper.immutablevalue(SomeBuiltin): {:?}.__name__ is not a string: {value:?}",
                        obj.qualname()
                    ))
                })?,
                Err(err) => {
                    return Err(AnnotatorError::new(format!(
                        "Bookkeeper.immutablevalue(SomeBuiltin): getattr({:?}, '__name__') failed: {err:?}",
                        obj.qualname()
                    )));
                }
            };
            let mut sb = SomeBuiltin::new(
                obj.qualname().to_string(),
                None,
                Some(format!("{module_name}.{name}")),
            );
            sb.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::Builtin(sb));
        }
        // upstream bookkeeper.py:312-314:
        //
        //     elif extregistry.is_registered(x):
        //         entry = extregistry.lookup(x)
        //         result = entry.compute_annotation_bk(self)
        //
        // The HostObject side of the value-level registry currently has
        // no entries (see [`extregistry::is_registered`] doc) — adding
        // one means extending the match arm there, the call below
        // automatically picks it up. Until then this branch declines
        // and falls through to the is_class / is_user_function / etc.
        // checks just like upstream's `elif extregistry.is_registered`
        // declines when `x` is not registered.
        let host_const = ConstValue::HostObject(obj.clone());
        if crate::translator::rtyper::extregistry::is_registered(&host_const) {
            let entry = crate::translator::rtyper::extregistry::lookup(&host_const).expect(
                "Bookkeeper.immutablevalue_hostobject: extregistry.lookup must succeed after \
                 is_registered",
            );
            let mut result = entry.compute_annotation_bk(self)?;
            // upstream bookkeeper.py:347 — `try: result.const = x;
            // except AttributeError: pass`. SomeValue::set_const_box
            // mirrors the swallow contract.
            result.set_const_box(Constant::new(raw.clone()));
            return Ok(result);
        }
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
        // Fallback `callable(x)` arm for `BuiltinCallable` HostObjects
        // whose qualname is not registered in `BUILTIN_ANALYZERS`
        // (e.g. the bare `std.ptr.eq` / `std.mem.align_of` /
        // `std.alloc.dealloc` host stubs published by
        // `HostEnv::bootstrap`).  Mirrors upstream `bookkeeper.py:317-331`
        // `elif callable(x): result = SomePBC([self.getdesc(x)])`:
        // PyPy doesn't distinguish "analyzer-backed builtin" from
        // "bare-stub callable" at the immutablevalue boundary — both
        // route through the same `callable(x)` arm.  Pyre splits the
        // registered-analyzer path (returns `SomeBuiltin` with an
        // analyzer ref) from the unregistered fallback (returns
        // `SomeBuiltin` with `analyzer = None`), so a call-site
        // dispatch on this `SomeBuiltin` still fails loud when the
        // analyser is genuinely missing — but `immutablevalue` itself
        // succeeds, matching upstream's "annotate first, fail at the
        // call site only" lifecycle.
        if obj.is_builtin_callable() {
            let qualname = obj.qualname().to_string();
            let mut sb = SomeBuiltin::new(qualname.clone(), None, Some(qualname));
            sb.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::Builtin(sb));
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

/// Walk `field_ty`'s type-string structure and collect every bare-named
/// type that the `StructFieldRegistry` recognises.  Used by
/// [`Bookkeeper::getuniqueclassdef_for_struct_root`]'s pass-1 discovery
/// to register a ClassDef for every reachable struct before any
/// attribute is projected.  Recursion bound is type-string nesting
/// (Vec<Vec<…>>) which is shallow, NOT the registry's field-graph
/// depth.
fn collect_referenced_struct_names(
    field_ty: &str,
    reg: &crate::front::StructFieldRegistry,
    out: &mut Vec<String>,
) {
    let stripped = field_ty
        .trim()
        .trim_start_matches('&')
        .trim_start_matches("mut ")
        .trim_start_matches("*const ")
        .trim_start_matches("*mut ")
        .trim();
    match stripped {
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize" | "f32"
        | "f64" | "bool" | "String" | "str" | "char" | "()" => return,
        _ => {}
    }
    for wrapper in [
        "Vec<",
        "VecDeque<",
        "Option<",
        "Rc<",
        "Arc<",
        "Box<",
        "RefCell<",
        "Cell<",
        "Pin<",
        "NonNull<",
        "NonZero<",
        "MaybeUninit<",
        "ManuallyDrop<",
        "UnsafeCell<",
        "Wrapping<",
        "Reverse<",
        "HashSet<",
        "BTreeSet<",
        "IndexSet<",
    ] {
        if let Some(inner) = strip_generic_one(stripped, wrapper) {
            collect_referenced_struct_names(inner, reg, out);
            return;
        }
    }
    for wrapper in ["HashMap<", "BTreeMap<", "IndexMap<", "Result<", "Cow<"] {
        if let Some(inner) = strip_generic_one(stripped, wrapper) {
            for part in split_generic_args(inner) {
                collect_referenced_struct_names(part, reg, out);
            }
            return;
        }
    }
    if let Some(rest) = stripped.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        for part in split_generic_args(rest) {
            collect_referenced_struct_names(part, reg, out);
        }
        return;
    }
    if let Some(rest) = stripped.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let inner = match rest.find(';') {
            Some(semi) => rest[..semi].trim(),
            None => rest.trim(),
        };
        collect_referenced_struct_names(inner, reg, out);
        return;
    }
    if reg.fields.contains_key(stripped) {
        out.push(stripped.to_string());
    }
}

/// Strip `Wrapper<` prefix and matching `>` suffix from a type string,
/// returning the inner generic-args slice unchanged.  Returns `None` if
/// the prefix is absent OR the suffix is not `>`.
fn strip_generic_one<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    input.strip_prefix(prefix).and_then(|s| s.strip_suffix('>'))
}

/// Split a generic-args slice on top-level `,` boundaries, respecting
/// nested `<>` / `()` / `[]` depth.
fn split_generic_args(input: &str) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    let mut depth: i32 = 0;
    let mut start: usize = 0;
    let bytes = input.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'<' | b'(' | b'[' => depth += 1,
            b'>' | b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                let segment = input[start..i].trim();
                if !segment.is_empty() {
                    out.push(segment);
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = input[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
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
/// "host-graph evaluator not yet landed" `AnnotatorError`
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
                "_freeze_() on {q:?} requires the host-graph evaluator \
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
/// body surfaces the same `host-graph evaluator` marker as
/// `call_freeze_method` so the two boundary limitations are visible
/// at a single grep point.
fn call_cleanup_method(obj: &HostObject) -> Result<(), AnnotatorError> {
    let Some(method) = lookup_zero_arg_method(obj, "_cleanup_")? else {
        return Ok(());
    };
    method.call_host(&[]).map(|_| ()).map_err(|err| {
        AnnotatorError::new(match err {
            crate::flowspace::model::HostCallError::RequiresFlowEvaluator(q) => format!(
                "_cleanup_() on {q:?} requires the host-graph evaluator \
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

pub(crate) fn build_args_for_op(
    opname: &str,
    args_s: &[Option<SomeValue>],
) -> Result<ArgumentsForTranslation, AnnotatorError> {
    match opname {
        "simple_call" => Ok(super::argument::simple_args_opt(args_s.to_vec())),
        "call_args" => {
            if args_s.is_empty() {
                return Err(AnnotatorError::new(
                    "build_args_for_op(call_args): missing shape argument",
                ));
            }
            // RPython `call_args` puts a concrete shape Constant at
            // position 0; pyre's frontend mirrors this in
            // `flowcontext::build_call_shape_constant`, so position 0
            // is always bound (matched by the unwrap below — None
            // here is a producer bug).
            let shape_arg = args_s[0].as_ref().ok_or_else(|| {
                AnnotatorError::new("build_args_for_op(call_args): args_s[0] is unbound")
            })?;
            let shape_const = shape_arg.const_().ok_or_else(|| {
                AnnotatorError::new("build_args_for_op(call_args): args_s[0] is not a Constant")
            })?;
            let shape = call_shape_from_const(shape_const)?;
            // Position 0 was the shape Constant; remaining positions
            // may carry None for unbound caller args.
            let mut data_w: Vec<Option<SomeValue>> = Vec::with_capacity(args_s.len() - 1);
            data_w.extend_from_slice(&args_s[1..]);
            Ok(ArgumentsForTranslation::fromshape(&shape, data_w))
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
        if let Some(s) = k.as_text() {
            shape_keys.push(s.to_string());
        } else {
            return Err(AnnotatorError::new(
                "call_shape_from_const: shape_keys element is not Str",
            ));
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

/// RPython `origin_of_meth(boundmeth)` (bookkeeper.py:583-593).
pub fn origin_of_meth(boundmeth: &HostObject) -> Result<(&HostObject, String), AnnotatorError> {
    let origin_class = boundmeth.bound_method_origin_class().ok_or_else(|| {
        AnnotatorError::new(format!(
            "could not match bound-method to attribute name: {boundmeth:?}"
        ))
    })?;
    let name = boundmeth.bound_method_name().ok_or_else(|| {
        AnnotatorError::new(format!(
            "could not match bound-method to attribute name: {boundmeth:?}"
        ))
    })?;
    Ok((origin_class, name.to_string()))
}

/// RPython `ishashable(x)` (bookkeeper.py:595-601).
pub fn ishashable(_x: &HostObject) -> bool {
    true
}

/// RAII guard returned by [`Bookkeeper::at_position`]. Mirrors the
/// upstream `@contextmanager` exit — calls [`Bookkeeper::leave`] on
/// drop unless the fast-path at bookkeeper.py:99-101 skipped the
/// initial enter.
pub(crate) struct PositionGuard {
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
    use rustpython_compiler::{Mode, compile as rp_compile};
    use rustpython_compiler_core::bytecode::ConstantData;

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    fn compiled_graph_func(src: &str, defaults: Vec<Constant>) -> GraphFunc {
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
        GraphFunc::from_host_code(
            crate::flowspace::bytecode::HostCode::from_code(inner),
            Constant::new(ConstValue::Dict(Default::default())),
            defaults,
        )
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
    fn getuniqueclassdef_for_struct_root_registers_by_identity_and_projects_fields() {
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "PyFrame".to_string(),
            vec![
                ("pycode".to_string(), "PyCode".to_string()),
                ("depth".to_string(), "i64".to_string()),
            ],
        );
        reg.fields.insert(
            "PyCode".to_string(),
            vec![("argcount".to_string(), "i64".to_string())],
        );
        bk.set_pyre_struct_fields(Rc::new(reg));

        let cd1 = bk
            .getuniqueclassdef_for_struct_root("PyFrame")
            .expect("PyFrame registers");
        // Identity: a second lookup returns the SAME ClassDef Rc — the
        // type-root string resolves to one interned host class object.
        let cd2 = bk
            .getuniqueclassdef_for_struct_root("PyFrame")
            .expect("PyFrame re-lookup");
        assert!(
            Rc::ptr_eq(&cd1, &cd2),
            "type-root resolves to one identity-keyed ClassDef"
        );

        // Orthodox registration: the ClassDef lands in
        // `bookkeeper.classdefs` (not a side-cache-only stub).
        let registered = bk.classdef_snapshot().iter().any(|c| Rc::ptr_eq(c, &cd1));
        assert!(
            registered,
            "ClassDef must be registered in bookkeeper.classdefs by identity"
        );

        // Instance attributes projected from the registry.
        {
            let g = cd1.borrow();
            let depth = g.attrs.get("depth").expect("depth attr projected");
            assert!(
                matches!(depth.s_value, SomeValue::Integer(_)),
                "i64 field -> SomeInteger"
            );
            let pycode = g.attrs.get("pycode").expect("pycode attr projected");
            assert!(
                matches!(pycode.s_value, SomeValue::Instance(_)),
                "struct-typed field -> SomeInstance(inner classdef)"
            );
        }

        // The transitively-referenced inner struct is ALSO registered
        // with its own fields projected (no attrs-empty inner stub).
        let inner = bk
            .getuniqueclassdef_for_struct_root("PyCode")
            .expect("PyCode registers");
        assert!(
            inner.borrow().attrs.contains_key("argcount"),
            "inner struct's own fields are projected too"
        );

        // Order-independent identity: the inner classdef reached through
        // `PyFrame.pycode` is the SAME Rc as the standalone `PyCode`
        // lookup.  With a single identity route there is no name-keyed
        // path that could mint a divergent unregistered stub.
        let pycode_via_field = {
            let g = cd1.borrow();
            match &g.attrs.get("pycode").expect("pycode attr").s_value {
                SomeValue::Instance(si) => si.classdef.clone(),
                _ => None,
            }
        };
        assert!(
            pycode_via_field
                .as_ref()
                .is_some_and(|c| Rc::ptr_eq(c, &inner)),
            "PyFrame.pycode inner classdef must be the same Rc as the standalone PyCode lookup"
        );
    }

    #[test]
    fn getuniqueclassdef_for_enum_variant_subclasses_the_base() {
        use crate::annotator::model::{SomeInstance, SomeValue};
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        // Discriminant-only base; the variant carries its own payload.
        reg.fields.insert(
            "Color".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "Color::Rgb".to_string(),
            vec![("rgb".to_string(), "i64".to_string())],
        );
        bk.set_pyre_struct_fields(Rc::new(reg));

        let base = bk
            .getuniqueclassdef_for_struct_root("Color")
            .expect("base registers");
        let variant = bk
            .getuniqueclassdef_for_enum_variant("Color", "Rgb")
            .expect("variant registers");

        // The variant is a subclass of the base — its basedef chain
        // reaches the base, so commonbase(base, variant) == base.
        assert!(
            variant.borrow().issubclass(&base),
            "variant ClassDef must be a subclass of the base"
        );
        let common = crate::annotator::classdesc::ClassDef::commonbase(&base, &variant)
            .expect("commonbase exists");
        assert!(
            Rc::ptr_eq(&common, &base),
            "commonbase(base, variant) must be the base"
        );

        // Identity: a second call returns the same Rc (canonical cache).
        let variant2 = bk
            .getuniqueclassdef_for_enum_variant("Color", "Rgb")
            .expect("variant re-lookup");
        assert!(Rc::ptr_eq(&variant, &variant2), "variant identity stable");

        // The narrowing resolver, the prologue pre-mint, and the ctor arm
        // must all intern ONE variant classdef.  A `.`-vs-`::` split would
        // mint a second, distinct classdef that the single
        // `assign_inheritance_ids` pass never reaches; assert the resolver
        // returns the classdef the pre-mint / ctor obtain through the
        // shared `intern_enum_variant_host` primitive (the `::`-qualified
        // key).
        let premint_host = bk.intern_enum_variant_host("Color", "Rgb");
        let premint = bk
            .getuniqueclassdef(&premint_host)
            .expect("pre-mint variant classdef");
        assert!(
            Rc::ptr_eq(&variant, &premint),
            "narrowing variant must be the pre-mint/ctor classdef from intern_enum_variant_host"
        );

        // The payoff: improve() narrows SomeInstance(base) to
        // SomeInstance(variant) given the variant as the refinement —
        // exactly what the discriminant-keyed knowntypedata will feed it.
        let s_base = SomeValue::Instance(SomeInstance::new(
            Some(base.clone()),
            false,
            std::collections::BTreeMap::new(),
        ));
        let s_variant = SomeValue::Instance(SomeInstance::new(
            Some(variant.clone()),
            false,
            std::collections::BTreeMap::new(),
        ));
        let narrowed = crate::annotator::binaryop::improve(&s_base, &s_variant);
        match narrowed {
            SomeValue::Instance(si) => assert!(
                si.classdef
                    .as_ref()
                    .is_some_and(|c| Rc::ptr_eq(c, &variant)),
                "improve(base, variant) narrows to the variant classdef"
            ),
            other => panic!("expected SomeInstance(variant), got {other:?}"),
        }
    }

    #[test]
    fn enum_variant_subclasses_base_even_when_struct_root_preregistered_first() {
        // The session prologue (`PyreCallRegistry::ensure_session`) pre-mints
        // EVERY `struct_fields` key through `getuniqueclassdef_for_struct_root`
        // — including the variant keys — BEFORE any narrowing's
        // `getuniqueclassdef_for_enum_variant` runs.  Since
        // `intern_class_by_qualname` is first-mint-wins, the variant must
        // acquire its enum base on that first mint, not lose it.
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "Color".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "Color::Rgb".to_string(),
            vec![("rgb".to_string(), "i64".to_string())],
        );
        bk.set_pyre_struct_fields(Rc::new(reg));

        // Prologue order: pre-register every struct root, variant keys first.
        for root in bk.pyre_struct_root_names() {
            let _ = bk.getuniqueclassdef_for_struct_root(&root);
        }
        let base = bk
            .getuniqueclassdef_for_struct_root("Color")
            .expect("base registers");
        let variant = bk
            .getuniqueclassdef_for_enum_variant("Color", "Rgb")
            .expect("variant registers");
        assert!(
            variant.borrow().issubclass(&base),
            "variant must subclass the base even when pre-registered as a struct root first"
        );
    }

    #[test]
    fn enum_variant_ctor_helper_and_narrowing_resolve_one_classdef_with_payload_attr() {
        // The variant CONSTRUCTOR path (`flowspace_adapter`, via
        // `intern_enum_variant_host`) and the discriminant-NARROWING path
        // (`getuniqueclassdef_for_enum_variant`) must resolve the SAME
        // class object — RPython's single-class-per-variant identity
        // (`rclass.py:82-88`) — and it must carry the variant's payload
        // attr.  Before the unification the ctor minted a dotted-qualname
        // sibling classdef with no payload attrs.
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        // Canonical `module::Enum` + `module::Enum::Variant` spellings —
        // already `::`-qualified, so `canonical_struct_name` passes them
        // through and the test never touches the process-global
        // `STRUCT_ORIGIN_REGISTRY`.
        reg.fields.insert(
            "shapes::Color".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "shapes::Color::Rgb".to_string(),
            vec![("rgb".to_string(), "i64".to_string())],
        );
        bk.set_pyre_struct_fields(Rc::new(reg));

        // Constructor side: the adapter interns through this helper.
        let ctor_host = bk.intern_enum_variant_host("shapes::Color", "Rgb");
        let ctor_classdef = bk
            .getuniqueclassdef(&ctor_host)
            .expect("ctor variant classdef");
        // Narrowing side: the public API the discriminant read uses.
        let narrow_classdef = bk
            .getuniqueclassdef_for_enum_variant("shapes::Color", "Rgb")
            .expect("narrowing variant classdef");

        assert!(
            Rc::ptr_eq(&ctor_classdef, &narrow_classdef),
            "constructor and narrowing must resolve ONE variant class object"
        );
        assert!(
            narrow_classdef.borrow().attrs.contains_key("rgb"),
            "the unified variant class must carry its payload attr"
        );
        // And it subclasses the discriminant-only base.
        let base = bk
            .getuniqueclassdef_for_struct_root("shapes::Color")
            .expect("base registers");
        assert!(
            narrow_classdef.borrow().issubclass(&base),
            "variant subclasses the enum base"
        );
    }

    #[test]
    fn enum_variant_ctor_bare_leaf_canonicalises_to_one_classdef_with_narrowing() {
        // The production ctor side carries the enum's *bare leaf* as its
        // owner tail (`flowspace_adapter`), while the narrowing side reads
        // the base classdef's *canonical* `module::Enum` name
        // (`unaryop.rs`).  They unify only because `front::mir` registers
        // the leaf → module origin so `canonical_struct_name(leaf)` lands
        // on the same `module::Enum` spelling.  Without that origin the
        // bare leaf passes through and the two sides mint sibling
        // classdefs.  Uses a unique leaf so replacing the process-global
        // `STRUCT_ORIGIN_REGISTRY` cannot perturb a concurrent test.
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "probemod::Newt".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "probemod::Newt::Eft".to_string(),
            vec![("eft".to_string(), "i64".to_string())],
        );
        bk.set_pyre_struct_fields(Rc::new(reg));
        // The origin `front::mir` now registers for an enum leaf.
        majit_ir::descr::register_struct_origins(
            [("Newt".to_string(), "probemod".to_string())].into(),
        );

        // Ctor side interns through `intern_enum_variant_host` with the
        // bare leaf as owner tail.
        let ctor_host = bk.intern_enum_variant_host("Newt", "Eft");
        let ctor_classdef = bk
            .getuniqueclassdef(&ctor_host)
            .expect("ctor variant classdef");
        // Narrowing side passes the base classdef's canonical name.
        let narrow_classdef = bk
            .getuniqueclassdef_for_enum_variant("probemod::Newt", "Eft")
            .expect("narrowing variant classdef");

        assert!(
            Rc::ptr_eq(&ctor_classdef, &narrow_classdef),
            "bare-leaf ctor and canonical narrowing must resolve ONE class object"
        );
        assert!(
            narrow_classdef.borrow().attrs.contains_key("eft"),
            "the unified variant class must carry its payload attr"
        );
    }

    #[test]
    fn enum_variant_narrowing_knowntypedata_keys_int_cases_to_variant_subclasses() {
        use crate::annotator::model::{ExitCaseKey, SomeInstance, SomeValue};
        use crate::flowspace::model::Variable;
        use crate::front::StructFieldRegistry;
        use std::collections::HashMap;

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "Color".to_string(),
            vec![
                ("__discriminant".to_string(), "i64".to_string()),
                ("rgb".to_string(), "i64".to_string()),
            ],
        );
        bk.set_pyre_struct_fields(Rc::new(reg));

        // discriminant 0 -> Rgb, 1 -> Named.
        let mut by_discr: HashMap<i64, String> = HashMap::new();
        by_discr.insert(0, "Rgb".to_string());
        by_discr.insert(1, "Named".to_string());
        let mut map: HashMap<String, HashMap<i64, String>> = HashMap::new();
        map.insert("Color".to_string(), by_discr);
        bk.set_pyre_enum_variant_by_discriminant(Rc::new(map));

        let base = bk
            .getuniqueclassdef_for_struct_root("Color")
            .expect("base registers");
        let receiver = Rc::new(Variable::new());
        let ktd = bk
            .enum_variant_narrowing_knowntypedata("Color", &receiver)
            .expect("enum root has a variant table");

        // One knowntypedata case per variant, keyed by the integer tag.
        assert_eq!(ktd.len(), 2, "one knowntypedata case per variant");
        for (discr, variant_name) in [(0, "Rgb"), (1, "Named")] {
            let constraints = ktd
                .get(&ExitCaseKey::Int(discr))
                .unwrap_or_else(|| panic!("missing Int({discr}) case"));
            let s_refined = constraints
                .get(&receiver)
                .expect("receiver variable is the narrowing key");
            let variant = bk
                .getuniqueclassdef_for_enum_variant("Color", variant_name)
                .expect("variant registers");
            // The refinement narrows SomeInstance(base) to the variant —
            // exactly what follow_link's improve does in arm `discr`.
            let s_base = SomeValue::Instance(SomeInstance::new(
                Some(base.clone()),
                false,
                std::collections::BTreeMap::new(),
            ));
            let narrowed = crate::annotator::binaryop::improve(&s_base, s_refined);
            match narrowed {
                SomeValue::Instance(si) => assert!(
                    si.classdef
                        .as_ref()
                        .is_some_and(|c| Rc::ptr_eq(c, &variant)),
                    "Int({discr}) narrows the receiver to {variant_name}"
                ),
                other => panic!("expected SomeInstance({variant_name}), got {other:?}"),
            }
        }

        // A class with no registered variant table yields no narrowing.
        let other_recv = Rc::new(Variable::new());
        assert!(
            bk.enum_variant_narrowing_knowntypedata("NotAnEnum", &other_recv)
                .is_none(),
            "non-enum class yields no knowntypedata"
        );
    }

    #[test]
    fn enum_variant_narrowing_is_idempotent_on_already_narrowed_receiver() {
        // A `__discriminant` re-read on a receiver already narrowed to a
        // variant subclass reaches the narrowing with `enum_root` carrying
        // the `::variant` tail (`Opt<i64>::A`).  It must re-derive the base
        // enum's variant subclasses (idempotent), NOT append another variant
        // (`Opt<i64>::A::A`) — otherwise the per-instantiation classdef
        // deepens every annotation pass and never reaches a fixpoint.  The
        // generic `<…>` instantiation is required to reproduce: the table
        // lookup strips it (`strip_instantiation_suffix`), so the narrowed
        // root resolves the same variant table while the variant-build sees
        // the full narrowed name.
        use crate::annotator::model::{ExitCaseKey, SomeValue};
        use crate::flowspace::model::Variable;
        use crate::front::StructFieldRegistry;
        use std::collections::HashMap;

        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "Opt<i64>".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        bk.set_pyre_struct_fields(Rc::new(reg));

        // Table keyed by the bare template root (`strip_instantiation_suffix`
        // reduces both `Opt<i64>` and `Opt<i64>::A` to `Opt`).
        let mut by_discr: HashMap<i64, String> = HashMap::new();
        by_discr.insert(0, "A".to_string());
        by_discr.insert(1, "B".to_string());
        let mut map: HashMap<String, HashMap<i64, String>> = HashMap::new();
        map.insert("Opt".to_string(), by_discr);
        bk.set_pyre_enum_variant_by_discriminant(Rc::new(map));

        let receiver = Rc::new(Variable::new());
        let from_base = bk
            .enum_variant_narrowing_knowntypedata("Opt<i64>", &receiver)
            .expect("base enum resolves the variant table");
        let from_variant = bk
            .enum_variant_narrowing_knowntypedata("Opt<i64>::A", &receiver)
            .expect("an already-narrowed receiver still resolves the base table");

        let classdef_of = |ktd: &crate::annotator::model::KnownTypeData, discr: i64| match ktd
            .get(&ExitCaseKey::Int(discr))
            .and_then(|c| c.get(&receiver))
            .expect("case present")
        {
            SomeValue::Instance(si) => si.classdef.clone().expect("variant carries a classdef"),
            other => panic!("expected SomeInstance, got {other:?}"),
        };
        for discr in [0i64, 1i64] {
            assert!(
                Rc::ptr_eq(
                    &classdef_of(&from_base, discr),
                    &classdef_of(&from_variant, discr)
                ),
                "Int({discr}): narrowing an already-narrowed receiver must reuse \
                 the base variant classdef, not deepen it (::A::A)"
            );
        }
    }

    #[test]
    fn getuniqueclassdef_for_struct_root_terminates_on_cyclic_graph() {
        // Localizes the route-b stack overflow: the seed fired
        // for `FixedObjectArray` (fields `len: usize`, `_items:
        // [PyObjectRef; 0]`) and the build aborted with a stack overflow.
        // Reproduce the real shape — an array struct referencing the
        // boxed-object root, which cycles back through the type object —
        // and assert transitive registration TERMINATES (returns a
        // populated classdef) rather than recursing forever.  If this test
        // overflows, the overflow is IN transitive registration; if it
        // passes, the overflow is downstream (rtyper repr setup).
        use crate::front::StructFieldRegistry;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "FixedObjectArray".to_string(),
            vec![
                ("len".to_string(), "usize".to_string()),
                ("_items".to_string(), "[PyObjectRef; 0]".to_string()),
            ],
        );
        reg.fields.insert(
            "PyObjectRef".to_string(),
            vec![("typ".to_string(), "W_TypeObject".to_string())],
        );
        reg.fields.insert(
            "W_TypeObject".to_string(),
            vec![
                // cycle: W_TypeObject -> PyObjectRef -> W_TypeObject
                ("base".to_string(), "PyObjectRef".to_string()),
                // self-reference: W_TypeObject -> W_TypeObject
                ("mro".to_string(), "Vec<W_TypeObject>".to_string()),
            ],
        );
        bk.set_pyre_struct_fields(Rc::new(reg));

        let cd = bk
            .getuniqueclassdef_for_struct_root("FixedObjectArray")
            .expect("cyclic graph registers without overflow");
        // `_items` projects to a SomeList of the boxed-object instance.
        assert!(
            cd.borrow().attrs.contains_key("_items"),
            "array field projected"
        );
        // The cyclic inner structs are registered exactly once each.
        let ty = bk
            .getuniqueclassdef_for_struct_root("W_TypeObject")
            .expect("W_TypeObject registers");
        let ty2 = bk
            .getuniqueclassdef_for_struct_root("W_TypeObject")
            .expect("W_TypeObject re-lookup");
        assert!(
            Rc::ptr_eq(&ty, &ty2),
            "cycle node interned to one identity-keyed ClassDef"
        );
    }

    #[test]
    fn getuniqueclassdef_for_struct_root_handles_deep_plus_cyclic_chain() {
        // Deep-chain regression gate for the iterative worklist.
        // The 2026-05-29 route-b measurement overflowed the native stack in
        // the RECURSIVE transitive walk of `FixedObjectArray`.  The walk now
        // drives an explicit `Vec` worklist (no native recursion over the
        // struct-field graph), so a chain far deeper than any native stack
        // can hold must register in O(reachable) heap without overflowing.
        // A back-edge from the tail to the head folds a cycle into the same
        // fixture, pinning depth and cycle handling together: if this test
        // ever overflows, the transitive registration regressed back to a
        // recursive walk.  `DEPTH` is chosen well past the frame budget a
        // recursive walk would exhaust on the default 8 MiB main stack.
        use crate::front::StructFieldRegistry;
        const DEPTH: usize = 2000;
        let bk = bk();
        let mut reg = StructFieldRegistry::default();
        for i in 0..DEPTH {
            // `Si { next: S{i+1} }` — each link references the next struct.
            reg.fields.insert(
                format!("S{i}"),
                vec![("next".to_string(), format!("S{}", i + 1))],
            );
        }
        // Terminal node: a scalar leaf plus a back-edge to the head, so the
        // reachable graph is both `DEPTH`-deep and cyclic.
        reg.fields.insert(
            format!("S{DEPTH}"),
            vec![
                ("leaf".to_string(), "i64".to_string()),
                ("back".to_string(), "S0".to_string()),
            ],
        );
        bk.set_pyre_struct_fields(Rc::new(reg));

        // Registering from the head walks the full depth + the back-edge
        // cycle without overflowing.
        let head = bk
            .getuniqueclassdef_for_struct_root("S0")
            .expect("deep + cyclic graph registers without stack overflow");
        assert!(
            matches!(
                head.borrow().attrs.get("next").map(|a| &a.s_value),
                Some(SomeValue::Instance(_))
            ),
            "head's struct-typed field projects to SomeInstance"
        );

        // A node midway down the chain is registered with its field
        // projected — the worklist reached the full depth (a recursive walk
        // would have overflowed long before here).
        let mid = format!("S{}", DEPTH / 2);
        let mid_cd = bk
            .getuniqueclassdef_for_struct_root(&mid)
            .unwrap_or_else(|e| panic!("deep node {mid} registers: {e}"));
        assert!(
            mid_cd.borrow().attrs.contains_key("next"),
            "deep node's field is projected"
        );
        let mid_cd2 = bk
            .getuniqueclassdef_for_struct_root(&mid)
            .expect("deep node re-lookup");
        assert!(
            Rc::ptr_eq(&mid_cd, &mid_cd2),
            "deep node interned to one identity-keyed ClassDef"
        );

        // The scalar leaf at the tail projects as an integer — confirming
        // the worklist reached the terminal node past the cycle.
        let tail = bk
            .getuniqueclassdef_for_struct_root(&format!("S{DEPTH}"))
            .expect("tail registers");
        assert!(
            matches!(
                tail.borrow().attrs.get("leaf").map(|a| &a.s_value),
                Some(SomeValue::Integer(_))
            ),
            "tail scalar leaf projects to SomeInteger"
        );
    }

    #[test]
    fn getuniqueclassdef_for_struct_root_reprojects_after_late_registry() {
        // Order-independence: a root resolved BEFORE the field registry
        // arrives is registered attrs-empty, then back-filled — at the SAME
        // identity — once the registry is supplied and the lookup re-runs.
        // RPython grows `classdef.attrs` as annotation proceeds; the same
        // `ClassDef` object accrues attributes, never a fresh divergent one.
        use crate::front::StructFieldRegistry;
        let bk = bk();

        // Registry-less lookup: identity-registered, but no fields to
        // project, so `attrs` stays empty.
        let early = bk
            .getuniqueclassdef_for_struct_root("PyFrame")
            .expect("PyFrame registers without a registry");
        assert!(
            bk.classdef_snapshot().iter().any(|c| Rc::ptr_eq(c, &early)),
            "registry-less root is still identity-registered"
        );
        assert!(
            !early.borrow().attrs.contains_key("depth"),
            "no registry -> no projected attrs"
        );

        // Registry arrives late.
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "PyFrame".to_string(),
            vec![
                ("pycode".to_string(), "PyCode".to_string()),
                ("depth".to_string(), "i64".to_string()),
            ],
        );
        reg.fields.insert(
            "PyCode".to_string(),
            vec![("argcount".to_string(), "i64".to_string())],
        );
        bk.set_pyre_struct_fields(Rc::new(reg));

        // Re-lookup re-projects onto the SAME ClassDef Rc.
        let late = bk
            .getuniqueclassdef_for_struct_root("PyFrame")
            .expect("PyFrame re-lookup after registry");
        assert!(
            Rc::ptr_eq(&early, &late),
            "late re-projection keeps the original identity"
        );
        {
            let g = late.borrow();
            assert!(
                matches!(
                    g.attrs.get("depth").map(|a| &a.s_value),
                    Some(SomeValue::Integer(_))
                ),
                "depth back-filled after late registry"
            );
            assert!(
                matches!(
                    g.attrs.get("pycode").map(|a| &a.s_value),
                    Some(SomeValue::Instance(_))
                ),
                "pycode back-filled as SomeInstance after late registry"
            );
        }

        // The transitive inner struct discovered on the late pass is
        // registered with its own field projected too.
        let inner = bk
            .getuniqueclassdef_for_struct_root("PyCode")
            .expect("PyCode registers on the late pass");
        assert!(
            inner.borrow().attrs.contains_key("argcount"),
            "inner struct projected on the late pass"
        );
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
        let s = bk.immutablevalue(&ConstValue::byte_str("a")).unwrap();
        assert!(matches!(s, SomeValue::Char(_)));
    }

    #[test]
    fn immutablevalue_multichar_str_is_somestring() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::byte_str("hello")).unwrap();
        assert!(matches!(s, SomeValue::String(_)));
    }

    #[test]
    fn immutablevalue_str_with_nul_clears_no_nul() {
        let bk = bk();
        let with_nul = ConstValue::byte_str("a\x00b");
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
    fn immutableconstant_list_reuses_same_constant_identity() {
        let bk = bk();
        let key = Constant::new(ConstValue::List(vec![ConstValue::Int(1)]));
        let a = bk.immutableconstant(&key).unwrap();
        let b = bk.immutableconstant(&key).unwrap();
        let (SomeValue::List(a), SomeValue::List(b)) = (a, b) else {
            panic!("expected SomeList");
        };
        assert!(a.listdef.same_as(&b.listdef));
        assert_eq!(a.base.const_box.as_ref().map(|c| c.id), Some(key.id));
        assert_eq!(b.base.const_box.as_ref().map(|c| c.id), Some(key.id));
    }

    #[test]
    fn immutableconstant_list_distinguishes_equal_but_distinct_constants() {
        let bk = bk();
        let a_key = Constant::new(ConstValue::List(vec![ConstValue::Int(1)]));
        let b_key = Constant::new(ConstValue::List(vec![ConstValue::Int(1)]));
        let a = bk.immutableconstant(&a_key).unwrap();
        let b = bk.immutableconstant(&b_key).unwrap();
        let (SomeValue::List(a), SomeValue::List(b)) = (a, b) else {
            panic!("expected SomeList");
        };
        assert!(!a.listdef.same_as(&b.listdef));
        assert_eq!(a.base.const_box.as_ref().map(|c| c.id), Some(a_key.id));
        assert_eq!(b.base.const_box.as_ref().map(|c| c.id), Some(b_key.id));
    }

    #[test]
    fn immutableconstant_dict_distinguishes_equal_but_distinct_constants() {
        let bk = bk();
        let mut a_items = HashMap::new();
        a_items.insert(ConstValue::Int(1), ConstValue::None);
        let mut b_items = HashMap::new();
        b_items.insert(ConstValue::Int(1), ConstValue::None);
        let a_key = Constant::new(ConstValue::Dict(a_items));
        let b_key = Constant::new(ConstValue::Dict(b_items));
        let a = bk.immutableconstant(&a_key).unwrap();
        let b = bk.immutableconstant(&b_key).unwrap();
        let (SomeValue::Dict(a), SomeValue::Dict(b)) = (a, b) else {
            panic!("expected SomeDict");
        };
        assert_ne!(a.dictdef, b.dictdef);
        assert_eq!(a.base.const_box.as_ref().map(|c| c.id), Some(a_key.id));
        assert_eq!(b.base.const_box.as_ref().map(|c| c.id), Some(b_key.id));
    }

    #[test]
    fn immutablevalue_llptr_returns_someptr() {
        use crate::annotator::model::SomeObjectTrait;
        use crate::flowspace::model::{Block, FunctionGraph, Hlvalue, Variable};
        use crate::translator::rtyper::lltypesystem::lltype;
        use std::cell::RefCell;
        use std::rc::Rc;

        let bk = bk();
        let start = Rc::new(RefCell::new(Block::new(vec![])));
        let ret = Variable::new();
        ret.set_concretetype(Some(lltype::LowLevelType::Void));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "f",
            start,
            Hlvalue::Variable(ret),
        )));
        let ptr = lltype::getfunctionptr(&graph, lltype::_getconcretetype).unwrap();
        let expected_type = lltype::typeOf(&ptr);

        let s = bk
            .immutablevalue(&ConstValue::LLPtr(Box::new(ptr)))
            .expect("LLPtr must route through extregistry SomePtr");

        match s {
            SomeValue::Ptr(p) => {
                assert_eq!(p.ll_ptrtype, expected_type);
                assert!(p.immutable());
                assert!(!p.can_be_none());
                assert!(p.is_constant());
                assert!(matches!(
                    p.base.const_box.as_ref().map(|c| &c.value),
                    Some(ConstValue::LLPtr(_))
                ));
            }
            other => panic!("expected SomePtr, got {other:?}"),
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
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let func = HostObject::new_user_function(compiled_graph_func(
            "def f(self):\n    return self\n",
            Vec::new(),
        ));
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
        // produces SomeBuiltin with methodname
        // `getattr(x, "__module__", "unknown") + "." + x.__name__`.
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let bltn = HostObject::new_builtin_callable("sys.getdefaultencoding");
        let s = bk
            .immutablevalue(&ConstValue::HostObject(bltn))
            .expect("builtin HostObject must produce SomeBuiltin");
        let SomeValue::Builtin(sb) = s else {
            panic!("expected SomeBuiltin");
        };
        assert_eq!(sb.methodname.as_deref(), Some("sys.getdefaultencoding"));
    }

    #[test]
    fn immutablevalue_builtin_class_uses_builtin_module_and_name_for_methodname() {
        use crate::annotator::model::SomeValue;
        let bk = bk();
        let int_cls = crate::flowspace::model::HOST_ENV
            .lookup_builtin("int")
            .expect("builtin int class");
        let s = bk
            .immutablevalue(&ConstValue::HostObject(int_cls))
            .expect("builtin class HostObject must produce SomeBuiltin");
        let SomeValue::Builtin(sb) = s else {
            panic!("expected SomeBuiltin");
        };
        assert_eq!(sb.methodname.as_deref(), Some("__builtin__.int"));
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
    fn unicode_through_const_unistr() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::uni_str("abc")).unwrap();
        match s {
            SomeValue::UnicodeString(st) => assert!(st.inner.no_nul),
            other => panic!("expected SomeUnicodeString, got {other:?}"),
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
        let s = bk.immutablevalue(&ConstValue::byte_str("x")).unwrap();
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
        assert!(matches!(entry, DescEntry::Func(_)));
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
        assert!(matches!(entry, DescEntry::Class(_)));
        let cd = entry.as_class().unwrap();
        assert_eq!(cd.borrow().name, "pkg.Foo");
    }

    #[test]
    fn getdesc_for_instance_returns_frozen_entry() {
        let bk = bk();
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        let inst = HostObject::new_instance(cls, vec![]);
        let entry = bk.getdesc(&inst).unwrap();
        assert!(matches!(entry, DescEntry::Frozen(_)));
    }

    #[test]
    fn getdesc_for_builtin_callable_returns_frozen_entry() {
        // bookkeeper.py treats builtin callables as frozen PBCs via
        // the _freeze_ fallback.
        let bk = bk();
        let obj = HostObject::new_builtin_callable("len");
        let entry = bk.getdesc(&obj).unwrap();
        assert!(matches!(entry, DescEntry::Frozen(_)));
    }

    #[test]
    fn getmethoddesc_caches_identity() {
        // Two calls with the same funcdesc/classdefs/name/flags
        // return the same MethodDesc Rc.
        let bk = bk();
        let gf = GraphFunc::new("m", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let fd = bk.getdesc(&host).unwrap().as_func_entry().cloned().unwrap();
        let origin = crate::annotator::description::ClassDefKey::from_raw(1);
        let self_def = Some(crate::annotator::description::ClassDefKey::from_raw(2));
        let flags = std::collections::BTreeMap::new();
        let a = bk.getmethoddesc(&fd, origin, self_def, "m", flags.clone());
        let b = bk.getmethoddesc(&fd, origin, self_def, "m", flags.clone());
        assert!(Rc::ptr_eq(&a, &b));
    }

    /// `getmethoddesc_for_attribute` mirrors the regular-method branch
    /// of `bookkeeper.py:383-397 getdesc`: walks the receiver classdef's
    /// MRO read-only, picks the first Method PBC, and routes through
    /// `getmethoddesc` so the upstream `methoddescs` cache is primed.
    /// Missing attribute → `None`.
    #[test]
    fn getmethoddesc_for_attribute_walks_mro_and_primes_cache() {
        use crate::annotator::classdesc::Attribute;
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::{SomePBC, SomeValue};

        let bk = bk();
        let pyobj = HostObject::new_class("PyFrame", vec![]);
        let desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, pyobj, "PyFrame".into()),
        ));
        let classdef = crate::annotator::classdesc::ClassDef::new(&bk, &desc);
        let classdef_key = crate::annotator::description::ClassDefKey::from_classdef(&classdef);

        // Mint a MethodDesc via getmethoddesc and seed the attrs.
        let gf = GraphFunc::new("push", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let fd = bk.getdesc(&host).unwrap().as_func_entry().cloned().unwrap();
        let md = bk.getmethoddesc(
            &fd,
            classdef_key,
            Some(classdef_key),
            "push",
            std::collections::BTreeMap::new(),
        );
        let pbc = SomePBC::new([DescEntry::Method(md.clone())], false);
        let mut attr = Attribute::new("push");
        attr.s_value = SomeValue::PBC(pbc);
        classdef.borrow_mut().attrs.insert("push".into(), attr);

        // Idempotent on cache hit: returns the cached MethodDesc rc.
        let results = bk.getmethoddesc_for_attribute(&classdef, "push");
        assert_eq!(results.len(), 1, "expected single Method PBC under push");
        assert!(Rc::ptr_eq(&results[0], &md));

        // Missing attribute → empty Vec.
        assert!(
            bk.getmethoddesc_for_attribute(&classdef, "absent")
                .is_empty()
        );
    }

    /// `bookkeeper.py:384` getdesc bound-method branch and
    /// `description.py:451 MethodDesc.bind_self` rebind the descriptor
    /// to the receiver classdef.  Walking from a subclass receiver
    /// must produce a MethodDesc whose `selfclassdef` is the receiver
    /// classdef key, even when the attr stores an unbound entry on a
    /// base class (`selfclassdef = None`).  `originclassdef` stays the
    /// base — that's where the method body lives.
    #[test]
    fn getmethoddesc_for_attribute_rebinds_unbound_to_receiver() {
        use crate::annotator::classdesc::Attribute;
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::{SomePBC, SomeValue};

        let bk = bk();

        // Base class carries the unbound MethodDesc in attrs.
        let base_pyobj = HostObject::new_class("Base", vec![]);
        let base_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, base_pyobj, "Base".into()),
        ));
        let base_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &base_desc);
        let base_key = crate::annotator::description::ClassDefKey::from_classdef(&base_classdef);
        bk.register_classdef(base_classdef.clone());

        // Derived class inherits from Base — MRO walks Derived → Base.
        let derived_pyobj =
            HostObject::new_class("Derived", vec![base_desc.borrow().pyobj.clone()]);
        let derived_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, derived_pyobj, "Derived".into()),
        ));
        derived_desc.borrow_mut().basedesc = Some(base_desc.clone());
        let derived_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &derived_desc);
        let derived_key =
            crate::annotator::description::ClassDefKey::from_classdef(&derived_classdef);
        bk.register_classdef(derived_classdef.clone());

        // Seed Base.attrs["m"] with an unbound MethodDesc
        // (selfclassdef = None) so the walker has to perform the bind.
        let gf = GraphFunc::new("m", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let fd = bk.getdesc(&host).unwrap().as_func_entry().cloned().unwrap();
        let unbound = bk.getmethoddesc(&fd, base_key, None, "m", std::collections::BTreeMap::new());
        assert_eq!(unbound.borrow().selfclassdef, None);
        let pbc = SomePBC::new([DescEntry::Method(unbound.clone())], false);
        let mut attr = Attribute::new("m");
        attr.s_value = SomeValue::PBC(pbc);
        base_classdef.borrow_mut().attrs.insert("m".into(), attr);

        // Walking from Derived must return a MethodDesc bound to
        // Derived (upward best-match branch — Derived.issubclass(Base))
        // with originclassdef preserved as Base.
        let results = bk.getmethoddesc_for_attribute(&derived_classdef, "m");
        assert_eq!(results.len(), 1, "expected single filtered MD");
        let bound = &results[0];
        assert_eq!(bound.borrow().originclassdef, base_key);
        assert_eq!(bound.borrow().selfclassdef, Some(derived_key));
        // The unbound carrier is untouched; the bound result is a
        // distinct MethodDesc rc (cache key differs by selfclassdef).
        assert!(!Rc::ptr_eq(bound, &unbound));
    }

    /// `classdesc.py:344-365 lookup_filter` only rebinds MDs whose
    /// `selfclassdef is None`.  Already-bound MDs (from instance
    /// attribute origin) are appended with their existing
    /// `selfclassdef` preserved.  Walking from a derived receiver
    /// must NOT clobber a MD already bound to its origin classdef.
    #[test]
    fn getmethoddesc_for_attribute_preserves_already_bound_selfclassdef() {
        use crate::annotator::classdesc::Attribute;
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::{SomePBC, SomeValue};

        let bk = bk();

        let base_pyobj = HostObject::new_class("Base", vec![]);
        let base_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, base_pyobj, "Base".into()),
        ));
        let base_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &base_desc);
        let base_key = crate::annotator::description::ClassDefKey::from_classdef(&base_classdef);

        let derived_pyobj =
            HostObject::new_class("Derived", vec![base_desc.borrow().pyobj.clone()]);
        let derived_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, derived_pyobj, "Derived".into()),
        ));
        derived_desc.borrow_mut().basedesc = Some(base_desc.clone());
        let derived_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &derived_desc);
        let derived_key =
            crate::annotator::description::ClassDefKey::from_classdef(&derived_classdef);

        // Seed Base.attrs["m"] with a MethodDesc already bound to Base
        // (selfclassdef = Some(base_key)) — emulates an instance
        // attribute carrying a pre-bound bound-method object.
        let gf = GraphFunc::new("m", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let fd = bk.getdesc(&host).unwrap().as_func_entry().cloned().unwrap();
        let already_bound = bk.getmethoddesc(
            &fd,
            base_key,
            Some(base_key),
            "m",
            std::collections::BTreeMap::new(),
        );
        assert_eq!(already_bound.borrow().selfclassdef, Some(base_key));
        let pbc = SomePBC::new([DescEntry::Method(already_bound.clone())], false);
        let mut attr = Attribute::new("m");
        attr.s_value = SomeValue::PBC(pbc);
        base_classdef.borrow_mut().attrs.insert("m".into(), attr);

        // Walking from Derived must preserve the existing selfclassdef =
        // Some(base_key) — NOT rebind to Derived.
        let results = bk.getmethoddesc_for_attribute(&derived_classdef, "m");
        assert_eq!(results.len(), 1, "expected single filtered MD");
        let resolved = &results[0];
        assert_eq!(resolved.borrow().originclassdef, base_key);
        assert_eq!(
            resolved.borrow().selfclassdef,
            Some(base_key),
            "already-bound selfclassdef must be preserved, not rebound to receiver",
        );
        assert_ne!(derived_key, base_key, "fixture sanity: keys differ");
    }

    /// `classdesc.py:344-365 lookup_filter` first branch: when an
    /// unbound MD's `originclassdef` is a strict subclass of the
    /// receiver, the method is kept and bound to `methclassdef`
    /// (NOT `receiver`).  Mirrors the comment:
    ///   "bind the method by giving it a selfclassdef.  Use the
    ///    more precise subclass that it's coming from."
    #[test]
    fn getmethoddesc_for_attribute_subclass_origin_binds_to_methclassdef() {
        use crate::annotator::classdesc::Attribute;
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::{SomePBC, SomeValue};

        let bk = bk();

        let base_pyobj = HostObject::new_class("Base", vec![]);
        let base_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, base_pyobj, "Base".into()),
        ));
        let base_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &base_desc);
        let base_key = crate::annotator::description::ClassDefKey::from_classdef(&base_classdef);
        bk.register_classdef(base_classdef.clone());

        let derived_pyobj =
            HostObject::new_class("Derived", vec![base_desc.borrow().pyobj.clone()]);
        let derived_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, derived_pyobj, "Derived".into()),
        ));
        derived_desc.borrow_mut().basedesc = Some(base_desc.clone());
        let derived_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &derived_desc);
        let derived_key =
            crate::annotator::description::ClassDefKey::from_classdef(&derived_classdef);
        bk.register_classdef(derived_classdef.clone());

        // Seed Base.attrs["m"] with an unbound MethodDesc whose
        // originclassdef is Derived (a strict subclass of the
        // receiver Base) — mimics a PBC union carrying a subclass
        // override under a base-class attribute name.
        let gf = GraphFunc::new("m", Constant::new(ConstValue::Dict(Default::default())));
        let host = HostObject::new_user_function(gf);
        let fd = bk.getdesc(&host).unwrap().as_func_entry().cloned().unwrap();
        let subclass_origin = bk.getmethoddesc(
            &fd,
            derived_key,
            None,
            "m",
            std::collections::BTreeMap::new(),
        );
        let pbc = SomePBC::new([DescEntry::Method(subclass_origin.clone())], false);
        let mut attr = Attribute::new("m");
        attr.s_value = SomeValue::PBC(pbc);
        base_classdef.borrow_mut().attrs.insert("m".into(), attr);

        // Walking from Base sees originclassdef = Derived as a strict
        // subclass — bind to Derived (methclassdef), not Base.
        let results = bk.getmethoddesc_for_attribute(&base_classdef, "m");
        assert_eq!(results.len(), 1, "expected single filtered MD");
        let bound = &results[0];
        assert_eq!(bound.borrow().originclassdef, derived_key);
        assert_eq!(
            bound.borrow().selfclassdef,
            Some(derived_key),
            "subclass-origin branch binds to methclassdef (Derived), not receiver (Base)",
        );
        assert_ne!(base_key, derived_key, "fixture sanity");
    }

    /// classdesc.py:341-367 lookup_filter collects ALL matching descs
    /// into `d`, not just the first.  A PBC with both an already-bound
    /// MD and an unbound upward-match MD must return both entries.
    #[test]
    fn getmethoddesc_for_attribute_collects_multi_desc_pbc() {
        use crate::annotator::classdesc::Attribute;
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::{SomePBC, SomeValue};

        let bk = bk();

        let base_pyobj = HostObject::new_class("Base", vec![]);
        let base_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, base_pyobj, "Base".into()),
        ));
        let base_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &base_desc);
        let base_key = crate::annotator::description::ClassDefKey::from_classdef(&base_classdef);
        bk.register_classdef(base_classdef.clone());

        let derived_pyobj =
            HostObject::new_class("Derived", vec![base_desc.borrow().pyobj.clone()]);
        let derived_desc = Rc::new(RefCell::new(
            crate::annotator::classdesc::ClassDesc::new_shell(&bk, derived_pyobj, "Derived".into()),
        ));
        derived_desc.borrow_mut().basedesc = Some(base_desc.clone());
        let derived_classdef = crate::annotator::classdesc::ClassDef::new(&bk, &derived_desc);
        let derived_key =
            crate::annotator::description::ClassDefKey::from_classdef(&derived_classdef);
        bk.register_classdef(derived_classdef.clone());

        // Two FunctionDescs so the MDs have distinct cache keys.
        let gf1 = GraphFunc::new("m", Constant::new(ConstValue::Dict(Default::default())));
        let host1 = HostObject::new_user_function(gf1);
        let fd1 = bk
            .getdesc(&host1)
            .unwrap()
            .as_func_entry()
            .cloned()
            .unwrap();
        let gf2 = GraphFunc::new("m2", Constant::new(ConstValue::Dict(Default::default())));
        let host2 = HostObject::new_user_function(gf2);
        let fd2 = bk
            .getdesc(&host2)
            .unwrap()
            .as_func_entry()
            .cloned()
            .unwrap();

        // MD1: already-bound to Base (selfclassdef = Some(base_key)).
        let already_bound = bk.getmethoddesc(
            &fd1,
            base_key,
            Some(base_key),
            "m",
            std::collections::BTreeMap::new(),
        );
        // MD2: unbound, originclassdef = Base.
        // Walking from Derived → upward match → bound to Derived.
        let unbound =
            bk.getmethoddesc(&fd2, base_key, None, "m", std::collections::BTreeMap::new());

        let pbc = SomePBC::new(
            [
                DescEntry::Method(already_bound.clone()),
                DescEntry::Method(unbound.clone()),
            ],
            false,
        );
        let mut attr = Attribute::new("m");
        attr.s_value = SomeValue::PBC(pbc);
        base_classdef.borrow_mut().attrs.insert("m".into(), attr);

        // Walking from Derived: already-bound MD preserved +
        // unbound MD bound to Derived as upward best-match = 2 entries.
        let results = bk.getmethoddesc_for_attribute(&derived_classdef, "m");
        assert_eq!(
            results.len(),
            2,
            "multi-desc PBC must collect all matching MDs, not just the first",
        );
        // First entry: already-bound, selfclassdef preserved.
        assert_eq!(results[0].borrow().selfclassdef, Some(base_key));
        // Second entry: upward best-match, bound to receiver (Derived).
        assert_eq!(results[1].borrow().selfclassdef, Some(derived_key));
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
    fn consider_call_site_routes_lladtmeth_through_immutablevalue_and_prepends_ptr() {
        use crate::annotator::model::SomeLLADTMeth;
        use crate::flowspace::model::{Hlvalue, HostObject, SpaceOperation, Variable};
        use crate::flowspace::pygraph::PyGraph;
        use crate::translator::rtyper::llannotation::lltype_to_annotation;
        use crate::translator::rtyper::lltypesystem::lltype::{FuncType, LowLevelType, Ptr};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let _guard = ann.bookkeeper.at_position(None);

        let gf = compiled_graph_func("def f(self):\n    return self\n", Vec::new());
        let func = HostObject::new_user_function(gf.clone());
        let fd = ann
            .bookkeeper
            .getdesc(&func)
            .unwrap()
            .as_function()
            .unwrap();
        fd.borrow().cache.borrow_mut().insert(
            "".into(),
            Rc::new(PyGraph::new(gf.clone(), gf.code.as_ref().unwrap())),
        );
        let ll_ptrtype = Ptr {
            TO: crate::translator::rtyper::lltypesystem::lltype::PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        };
        let mut v_callable = Variable::named("adtmeth");
        ann.setbinding(
            &mut v_callable,
            SomeValue::LLADTMeth(SomeLLADTMeth::new(
                crate::translator::rtyper::lltypesystem::lltype::LowLevelPointerType::Ptr(
                    ll_ptrtype.clone(),
                ),
                ConstValue::HostObject(func.clone()),
            )),
        );
        let call_op = SpaceOperation::new(
            "simple_call",
            vec![Hlvalue::Variable(v_callable)],
            Hlvalue::Variable(Variable::named("r")),
        );

        ann.bookkeeper
            .consider_call_site(&call_op, None)
            .expect("SomeLLADTMeth call site must flow through the wrapped function");

        let family = fd.borrow().base.getcallfamily().unwrap();
        let expected_shape =
            build_args_for_op("simple_call", &[Some(lltype_to_annotation(ll_ptrtype))])
                .unwrap()
                .rawshape();
        let family_ref = family.borrow();
        let table = family_ref
            .calltables
            .get(&expected_shape)
            .expect("SomeLLADTMeth must prepend ll_ptrtype to the call args");
        assert_eq!(table.len(), 1);
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
        ann.bookkeeper
            .compute_at_fixpoint()
            .expect("empty annotator state must not error");
        assert!(!ann.bookkeeper.position_entered.get());
        assert!(ann.bookkeeper.emulated_pbc_calls.borrow().is_empty());
    }

    #[test]
    fn struct_root_pass2_replaces_untyped_force_shell() {
        use crate::front::StructFieldRegistry;
        let bk = bk();
        // Production order (lib.rs registration): the FORCE table is
        // populated from ValueType rows BEFORE any classdef exists, so
        // `_init_classdef` seeds `varnames` with the classdef-less
        // `SomeInstance` shell (`valuetype_to_someshell(Ref)`).  Pass-2
        // of `getuniqueclassdef_for_struct_root` must replace that
        // shell with the registry-string projection (`Box<[String]>`
        // -> SomeList), not skip it as already-filled.
        crate::annotator::classdesc::register_struct_fields(
            "CodeObject",
            &[("varnames".to_string(), crate::model::ValueType::Ref(None))],
        );
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "PyFrame".to_string(),
            vec![("code".to_string(), "CodeObject".to_string())],
        );
        reg.fields.insert(
            "CodeObject".to_string(),
            vec![("varnames".to_string(), "Box<[String]>".to_string())],
        );
        bk.set_pyre_struct_fields(Rc::new(reg));
        let _root = bk
            .getuniqueclassdef_for_struct_root("PyFrame")
            .expect("PyFrame registers");
        let cd = bk
            .getuniqueclassdef_for_struct_root("CodeObject")
            .expect("CodeObject registered");
        let g = cd.borrow();
        let varnames = g.attrs.get("varnames").expect("varnames attr present");
        assert!(
            matches!(varnames.s_value, SomeValue::List(_)),
            "varnames must project to SomeList, got {:?}",
            varnames.s_value
        );
    }

    /// Fixture for the methods-on-classdef capability that `dyn Trait`
    /// receiver-driven dispatch (issue #346, aheui LinkedList) needs.
    ///
    /// A ≥2-impl trait family registered through `register_trait_family`
    /// must (a) link a base ClassDef to its impl subclasses, (b) carry the
    /// trait methods as members so `getattr(instance, method)` annotates to
    /// a `MethodDesc` PBC family instead of a classdef-less shell, and
    /// (c) let the existing `MethodsPBCRepr` port lower that family to a
    /// vtable/method dispatch.  Plain pyre struct classdefs carry fields
    /// only, so this method-member path is exercised here for the first time.
    #[test]
    fn register_trait_family_getattr_yields_method_pbc_and_rtypes_via_methods_pbc_repr() {
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::model::SomeValue;
        use crate::translator::rtyper::rpbc::MethodsPBCRepr;
        use crate::translator::rtyper::rtyper::RPythonTyper;
        use std::collections::HashMap;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();

        // A method member is a user function backed by a GraphFunc — the
        // same shape `HostObject::new_class`'s `__dict__` would carry, and
        // what `add_source_attribute` turns into a MethodDesc.
        let member = |name: &str| -> ConstValue {
            ConstValue::HostObject(HostObject::new_user_function(GraphFunc::new(
                name.to_string(),
                Constant::new(ConstValue::Dict(Default::default())),
            )))
        };

        // Base trait `T` with a default-body method `shared`; three impls
        // A/B/C each override `shared`.  This is the ≥2-impl multi-impl
        // family shape that annotates classdef-less today.
        let mut base_members = HashMap::new();
        base_members.insert("shared".to_string(), member("T::shared"));
        let mut impls = Vec::new();
        for impl_name in ["A", "B", "C"] {
            let mut m = HashMap::new();
            m.insert(
                "shared".to_string(),
                member(&format!("{impl_name}::shared")),
            );
            impls.push((impl_name.to_string(), m));
        }
        let base_cd = bk
            .register_trait_family("T", base_members, impls)
            .expect("register_trait_family must succeed");

        // (a) base ↔ subclass links.
        assert_eq!(
            base_cd.borrow().subdefs.len(),
            3,
            "base classdef must link 3 impl subclasses"
        );

        // (b) getattr(instance_of_T, "shared") resolves to a MethodDesc PBC
        // family — NOT `SomeInstance(classdef=None)`, NOT "attribute not
        // found".  `s_getattr` on the base ClassDef is exactly what
        // `SomeInstance.getattr` (unaryop.rs:4143) calls for an instance of
        // that classdef; populate the attr across the tree first (the
        // fixpoint analog).
        ClassDef::check_missing_attribute_update(&base_cd, "shared")
            .expect("attr population must succeed");
        let s_shared = ClassDef::s_getattr(&base_cd, "shared", &std::collections::BTreeMap::new())
            .expect("s_getattr(shared) must resolve");
        let pbc = match &s_shared {
            SomeValue::PBC(p) => p.clone(),
            other => panic!("expected a method PBC for `shared`, got {other:?}"),
        };
        let n_methods = pbc
            .descriptions
            .values()
            .filter(|d| d.as_method().is_some())
            .count();
        assert!(
            n_methods >= 3,
            "expected >=3 MethodDescs in the `shared` family (the subclass \
             overrides), got {n_methods}: {pbc:?}"
        );

        // (c) the `MethodsPBCRepr` port — possibly never exercised on a
        // pyre-synthesized method PBC — must accept the family and resolve
        // its owner + bound-`self` InstanceRepr (the vtable dispatch seed).
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let repr = MethodsPBCRepr::new(&rtyper, pbc)
            .expect("MethodsPBCRepr must accept the pyre-synthesized method family");
        assert_eq!(repr.methodname, "shared");
        assert!(
            Rc::ptr_eq(&repr.classdef, &base_cd),
            "MethodsPBCRepr must fold the family's owner to the base classdef"
        );
    }

    /// Diagnostic for the LinkedList wiring: the real dispatch blockers
    /// (`_get_2_values` / `head` / `dup`) are trait REQUIRED methods —
    /// declared on the trait, bodied only on the impl subclasses.  This
    /// probes a base-instance `getattr` of a method that exists ONLY on the
    /// subclasses (absent from the base member map).
    ///
    /// FINDING: it resolves to the full impl family anyway.  RPython's
    /// attrfamily merge homes a same-named attribute defined across sibling
    /// subclasses onto their common base, so `s_getattr(base, "req")` returns
    /// the {A,B,C}::req MethodDesc PBC even though the base declares no `req`.
    /// => piece-3 registration can seed the impls ONLY; it need NOT synthesize
    /// a base declaration for required (default-body-less) trait methods.
    #[test]
    fn register_trait_family_subclass_only_method_surfaces_on_base() {
        use crate::annotator::classdesc::ClassDef;
        use crate::annotator::model::SomeValue;
        use std::collections::HashMap;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let member = |name: &str| -> ConstValue {
            ConstValue::HostObject(HostObject::new_user_function(GraphFunc::new(
                name.to_string(),
                Constant::new(ConstValue::Dict(Default::default())),
            )))
        };

        // Base `T` declares NO `req`; each impl bodies `req` — the
        // required-method-with-no-default-body shape.
        let mut impls = Vec::new();
        for impl_name in ["A", "B", "C"] {
            let mut m = HashMap::new();
            m.insert("req".to_string(), member(&format!("{impl_name}::req")));
            impls.push((impl_name.to_string(), m));
        }
        let base_cd = bk
            .register_trait_family("T", HashMap::new(), impls)
            .expect("register_trait_family must succeed");

        ClassDef::check_missing_attribute_update(&base_cd, "req").expect("attr population");
        let s_req = ClassDef::s_getattr(&base_cd, "req", &std::collections::BTreeMap::new())
            .expect("s_getattr(req) must not error");
        let n_methods = match &s_req {
            SomeValue::PBC(p) => p
                .descriptions
                .values()
                .filter(|d| d.as_method().is_some())
                .count(),
            other => panic!("expected a method PBC for subclass-only `req`, got {other:?}"),
        };
        // The attrfamily merge surfaces the subclass-only method on the base:
        // one MethodDesc per impl, so a base-typed / trait-object receiver
        // dispatches across all impls without a base declaration.
        assert_eq!(
            n_methods, 3,
            "subclass-only `req` must surface all 3 impl MethodDescs on the \
             base via attrfamily merge, got {n_methods}: {s_req:?}"
        );
    }
}
