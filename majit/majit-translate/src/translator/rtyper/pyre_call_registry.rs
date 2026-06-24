//! Pyre-side `FunctionPath → (HostObject, FunctionDesc)` registry.
//!
//! ## Role
//!
//! Upstream RPython's `rpython/annotator/bookkeeper.py:353-409
//! Bookkeeper.getdesc` looks up or creates a `FunctionDesc` keyed by
//! Python object identity (`id(pyobj)`).  The bookkeeper hosts the
//! `descs: dict[id, Desc]` map so any annotator pass can resolve a
//! `Constant(<function foo>)` argument in `simple_call` to the
//! `FunctionDesc` carrying `foo`'s signature, defaults, specializer,
//! and per-graph cache.
//!
//! Pyre's surface DSL has no Python callable objects — pyre's
//! frontend produces `OpKind::Call { target: CallTarget::FunctionPath
//! { segments }, .. }` (`crate::model`) carrying only the symbolic
//! path.  The line-by-line port wraps each pyre callee in a
//! synthetic `HostObject::UserFunction { graph_func }` whose
//! `(host_object, function_desc)` pair is pre-registered in
//! `Bookkeeper.descs` keyed by the host's Arc identity.  When
//! `getdesc` later looks up the host object, it short-circuits at
//! the cache lookup at upstream `bookkeeper.py:362-364`
//! (`try: return self.descs[obj_key]; except KeyError: ...`) and
//! returns the pre-built FunctionDesc instead of falling through to
//! `newfuncdesc`.  Without the pre-register, `newfuncdesc` would
//! call `cpython_code_signature(pyfunc.__code__)`
//! (`bookkeeper.py:418`) and fail on the synthetic GraphFunc that
//! has no `code` slot — the `Signature(["entry"])` upstream branch
//! at `bookkeeper.py:413-416` is reserved for the
//! `_generator_next_method_of_` special case, not a general
//! signature fallback.
//!
//! ## Module ownership
//!
//! This module owns the surface for pyre's `FunctionPath`-keyed
//! callable identity:
//!
//! - The synthetic GraphFunc + HostObject construction (one Arc
//!   identity per FunctionPath).
//! - The FunctionDesc construction with pyre's authoritative
//!   `Signature`.
//! - The `bookkeeper.descs` pre-registration so any subsequent
//!   `getdesc(host_object)` hits the cache.
//! - The `FunctionDesc.cache` pre-fill (via `prefill_default_cache`)
//!   so the rtyper's `cachedgraph` returns the lifted leaf
//!   `PyGraph` and skips `buildflowgraph` (which requires a real
//!   Python `__code__` body).
//! - The path → entry cache so repeat lookups (and the
//!   `translate_op` consumer) share one `Hlvalue::Constant` identity
//!   across all callsites of the same path.
//!
//! ## Position vs `Bookkeeper`
//!
//! The registry lives in `translator/rtyper/` rather than folded
//! into the upstream-mirrored `Bookkeeper` because pyre lacks the
//! `id(pyobj)` keying mechanism upstream uses; the
//! `FunctionPathKey`-keyed map is a peer to `Bookkeeper.descs`,
//! not a replacement.  Each `get_or_register` call writes the
//! resulting entry into `Bookkeeper.descs` so `getdesc(host_object)`
//! and the rtyper's downstream chain
//! (`pair_simple_call → FunctionDesc.specialize → cachedgraph →
//! FunctionRepr.call`) all see the registry's entries.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::annotator::bookkeeper::Bookkeeper;
use crate::annotator::description::{DescEntry, FunctionDesc, GraphCacheKey};
use crate::flowspace::argument::Signature;
use crate::flowspace::model::{ConstValue, Constant, GraphFunc, HostObject};
use crate::flowspace::pygraph::PyGraph;

/// Stable hashable key for a function path.
///
/// Pyre's `CallTarget::FunctionPath { segments }` uses `Vec<String>`;
/// the registry keys directly off that vector so equal paths share
/// the same entry without an intermediate hash truncation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct FunctionPathKey(pub Vec<String>);

impl FunctionPathKey {
    pub fn from_segments<I, S>(segments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        FunctionPathKey(segments.into_iter().map(Into::into).collect())
    }

    pub fn segments(&self) -> &[String] {
        &self.0
    }

    /// Last segment, or empty string when the path is empty.
    /// Used as the synthetic GraphFunc / FunctionDesc display name.
    pub fn name(&self) -> &str {
        self.0.last().map(|s| s.as_str()).unwrap_or("")
    }
}

/// Per-path registry entry.  Each entry owns:
///
/// - `host_object` — synthetic `HostObject::UserFunction` whose Arc
///   identity is the canonical callable token for the path.  Slice
///   A.3c's `translate_op` wraps a clone of this in
///   `Hlvalue::Constant { value: ConstValue::HostObject(host_object),
///   ... }`.
/// - `function_desc` — `FunctionDesc` carrying pyre's authoritative
///   `Signature`.  Pre-registered in `bookkeeper.descs` keyed by
///   `host_object`'s Arc identity so `Bookkeeper::getdesc(host_object)`
///   short-circuits at the cache.
pub struct PyreFunctionEntry {
    pub host_object: HostObject,
    pub function_desc: Rc<RefCell<FunctionDesc>>,
}

impl PyreFunctionEntry {
    /// Pre-fill the entry's `FunctionDesc.cache` for the default
    /// specializer with no `*args` (the lookup key
    /// `Specializer::Default + flatten_star_args(no vararg) ->
    /// GraphCacheKey::None` produced by `default_specialize` at
    /// `description.rs:1304-1351`).
    ///
    /// `cachedgraph` (`description.rs:1037-1039`) returns the cached
    /// `Rc<PyGraph>` immediately when the key hits, so the
    /// `buildgraph` path that requires a real `HostCode` body
    /// (`description.rs:1147-1169`) is skipped — pyre's synthetic
    /// `GraphFunc` has no `HostCode` and would fail there.
    ///
    /// The producer (Slice A.4 follow-on, or a test fixture) is
    /// responsible for constructing `pygraph` from pyre's
    /// `model::FunctionGraph` for this callee — see
    /// `cutover::lift_callee_to_pygraph`.
    pub fn prefill_default_cache(&self, pygraph: Rc<PyGraph>) {
        self.function_desc
            .borrow()
            .cache
            .borrow_mut()
            .insert(GraphCacheKey::None, pygraph);
    }

    /// Record a per-entry lift failure so consumers surfacing through
    /// `cachedgraph` see the producer-side error instead of the
    /// generic `buildflowgraph: missing code object` fallback.  Called
    /// from `cutover::populate_call_registry_from_call_graphs`'s Pass
    /// 2 when `lift_callee_to_pygraph` returns Err.
    pub fn record_lift_error(&self, message: String) {
        self.function_desc.borrow().record_pyre_lift_error(message);
    }
}

/// Per-subject result of the two-phase rtyper drive (`PYRE_TWO_PHASE_RTYPE`).
///
/// Phase A (annotate-all over the portal closure) populates `value_to_var` /
/// `constant_concretetypes` via the adapter + annotate-half and records the
/// flowspace `graph_key`. Phase B (rtype-all with per-graph isolation) writes
/// `Variable.concretetype` onto the `value_to_var` Variables in place, or
/// records the `graph_key` in [`TwoPhaseTypeCache::rtype_skipped`] on failure.
/// The per-graph publish reads this back instead of re-running the real path,
/// reproducing upstream's `translator.annotate()` → `rtyper.specialize()` →
/// `codewriter.make_jitcodes()` phase ordering (`driver.py:306/345/361`).
pub struct TwoPhaseSubject {
    pub graph_key: crate::flowspace::model::GraphKey,
    pub value_to_var: crate::translator::rtyper::flowspace_adapter::LegacyToTyped,
    pub constant_concretetypes: HashMap<
        crate::flowspace::model::Variable,
        crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
    >,
}

/// Whole-program two-phase type cache, keyed by subject path canonical key
/// (`CallPath::canonical_key()`, the same string the publish receives as its
/// `diag_label`).
#[derive(Default)]
pub struct TwoPhaseTypeCache {
    pub subjects: HashMap<String, TwoPhaseSubject>,
    /// Flowspace graph keys whose Phase-B rtype failed → publish Skips them
    /// to the legacy walker (the migration-scaffold fallback).
    pub rtype_skipped: HashSet<crate::flowspace::model::GraphKey>,
    /// Set once the prepass has run so publish consults the cache instead of
    /// re-running the real path. Cleared/absent ⟹ legacy per-graph behaviour.
    pub prepass_done: bool,
}

/// Per-program registry. Constructed once per `analyze_program` /
/// `specialize_legacy_graph_with_registry_returning_value_to_var`
/// driver invocation; shares
/// its `Bookkeeper` with the `RPythonAnnotator` so pre-registered
/// entries are visible to the rtyper's `getdesc` lookup.
///
/// `entries` is the canonical `FunctionPathKey → Rc<PyreFunctionEntry>`
/// table — one row per distinct callable identity.  Mirrors RPython
/// `Bookkeeper.descs[Constant(pyobj)] = Desc` (`bookkeeper.py:353-409`),
/// where the dict is keyed by Python function-object identity and
/// holds at most one `FunctionDesc` per callable.
///
/// `aliases` carries the alternate-spelling indirection (free function
/// registered under both `a::foo` and `crate::a::foo` keys).  Each
/// alias maps to the canonical `FunctionPathKey` whose entry the
/// caller wants.  Splitting this from `entries` keeps
/// [`Self::len`] (and any future canonical-entries iterator) free of
/// double-counting: the canonical entry appears exactly once even
/// when N aliases point at it.  RPython has no equivalent indirection
/// because Python callable identity gives `Bookkeeper.getdesc` direct
/// dedup; pyre's segment-key storage stands in for that until host
/// callable identity is available.
pub struct PyreCallRegistry {
    bookkeeper: Rc<Bookkeeper>,
    entries: RefCell<HashMap<FunctionPathKey, Rc<PyreFunctionEntry>>>,
    aliases: RefCell<HashMap<FunctionPathKey, FunctionPathKey>>,
    /// Whole-program two-phase type cache (`PYRE_TWO_PHASE_RTYPE`). Empty and
    /// `prepass_done = false` unless the two-phase prepass populated it.
    two_phase: RefCell<TwoPhaseTypeCache>,
    /// Lazily-constructed `(RPythonAnnotator, RPythonTyper)` pair
    /// shared by every per-session
    /// `specialize_legacy_graph_with_registry_returning_value_to_var`
    /// call against this
    /// registry.  RPython parity: `Translator.buildannotator()` /
    /// `:buildrtyper()` (`translator.py:73-83`) construct exactly one
    /// of each per Translator, and `RPythonTyper.specialize` runs
    /// exactly once per Translator (`driver.py:345`).  The per-graph
    /// subject is added through `addpendingblock` and rtyped via
    /// `specialize_more_blocks` (rtyper.py:198-241), which only
    /// touches blocks not yet in `already_seen`.
    session: RefCell<
        Option<(
            Rc<crate::annotator::annrpython::RPythonAnnotator>,
            Rc<crate::translator::rtyper::rtyper::RPythonTyper>,
        )>,
    >,
}

impl PyreCallRegistry {
    /// Construct an empty registry sharing `bookkeeper`.
    pub fn new(bookkeeper: Rc<Bookkeeper>) -> Self {
        PyreCallRegistry {
            bookkeeper,
            entries: RefCell::new(HashMap::new()),
            aliases: RefCell::new(HashMap::new()),
            two_phase: RefCell::new(TwoPhaseTypeCache::default()),
            session: RefCell::new(None),
        }
    }

    /// Mutable access to the two-phase type cache (`PYRE_TWO_PHASE_RTYPE`).
    pub fn two_phase(&self) -> std::cell::RefMut<'_, TwoPhaseTypeCache> {
        self.two_phase.borrow_mut()
    }

    /// True once the two-phase prepass populated the cache; publish then reads
    /// cached types instead of re-running the real path.
    pub fn two_phase_prepass_done(&self) -> bool {
        self.two_phase.borrow().prepass_done
    }

    /// Thread the program-wide `StructFieldRegistry` into the shared
    /// bookkeeper so `getuniqueclassdef_for_struct_root` /
    /// `project_pyre_field_type` can project struct fields onto a classdef.  Called once from
    /// `dual_gate_registry` after `PyreCallRegistry::new` with
    /// `CallControl::struct_fields().clone()`.
    pub fn set_pyre_struct_fields(&self, registry: Rc<crate::front::StructFieldRegistry>) {
        self.bookkeeper.set_pyre_struct_fields(registry);
    }

    /// Thread the trait → unique-concrete-impl-owner map into the
    /// shared bookkeeper so `derive_subject_inputcells` can resolve a
    /// generic receiver's bound-trait `class_root` to the impl type's
    /// `ClassDef`.  Called once from `dual_gate_registry` with
    /// `CallControl::trait_unique_impls().clone()`.
    pub fn set_pyre_trait_unique_impls(&self, map: HashMap<String, String>) {
        self.bookkeeper.set_pyre_trait_unique_impls(map);
    }

    /// Thread the enum `discriminant → variant` tables into the shared
    /// bookkeeper so the `__discriminant` getattr can attach
    /// discriminant→variant narrowing `knowntypedata`.  Called once from
    /// `dual_gate_registry` with
    /// `CallControl::enum_variant_by_discriminant().clone()`.
    pub fn set_pyre_enum_variant_by_discriminant(
        &self,
        map: Rc<HashMap<String, HashMap<i64, String>>>,
    ) {
        self.bookkeeper.set_pyre_enum_variant_by_discriminant(map);
    }

    /// Seed each struct-root class `HostObject`'s dict with its
    /// registered impl-method function hosts.
    ///
    /// classdesc.py:606-618 — a class's methods live in its class
    /// `__dict__`, so `SomeInstance.getattr(method)` resolves through
    /// `check_missing_attribute_update` → `find_source_for` →
    /// `s_get_value` (Constant function arm) → `bind_callables_under`
    /// to a bound-`MethodDesc` `SomePBC`.  Pyre's struct-root classes
    /// are interned from the field registry with an empty dict, so a
    /// method getattr found no source and the block stayed blocked
    /// forever.  Walk the registered entries: every path
    /// `[.., Owner, method]` whose `Owner` is a struct-field-registry
    /// root names a method of that class — install the entry's
    /// function `HostObject` (Arc-identical to the one
    /// `bookkeeper.descs` already keys, so `getdesc` resolves to the
    /// same `FunctionDesc`/cachedgraph) under the method name on the
    /// interned class object.  Free functions register as
    /// `[module, fn]` with a lowercase module segment that is never a
    /// registry root, so the membership gate filters them out.
    ///
    /// Called once from `dual_gate_registry` after
    /// `populate_call_registry_from_call_graphs`.
    pub fn seed_struct_root_method_members(&self) {
        let guard = self.bookkeeper.pyre_struct_fields.borrow();
        let Some(reg) = guard.as_ref() else {
            return;
        };
        // Leaf owners naming more than one qualified struct are
        // ambiguous for METHOD seeding even when their bare field
        // alias survived `harden_duplicate_leaf_metadata` (identical
        // field rows keep the alias, but the duplicate structs'
        // method sets may differ).  Seeding would pool both origins'
        // methods onto the one leaf-interned class, so a getattr
        // could bind a wrong-origin method silently.  Skip those
        // owners — the dispatch stays blocked (census-visible)
        // until per-origin class interning lands.
        let mut leaf_struct_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for key in reg.fields.keys() {
            if let Some((_, leaf)) = key.rsplit_once("::") {
                *leaf_struct_counts.entry(leaf).or_default() += 1;
            }
        }
        for (key, entry) in self.entries.borrow().iter() {
            let segs = key.segments();
            let [.., owner, method] = segs else {
                continue;
            };
            if !reg.fields.contains_key(owner) {
                continue;
            }
            if leaf_struct_counts.get(owner.as_str()).copied().unwrap_or(0) > 1 {
                continue;
            }
            let class_host = self.bookkeeper.intern_class_by_qualname(owner);
            class_host.class_set(
                method.clone(),
                ConstValue::HostObject(entry.host_object.clone()),
            );
        }
    }

    /// Get-or-construct the shared `(annotator, rtyper)` pair.
    ///
    /// First call constructs both from `self.bookkeeper`; subsequent
    /// calls return the cached pair, mirroring RPython's
    /// `Translator.buildannotator()` / `:buildrtyper()` cache
    /// (`translator.py:69-83 — assert self.annotator is None;
    /// self.annotator = RPythonAnnotator(self, ...)`).  The
    /// `RPythonTyper::initialize_exceptiondata()` setup runs on
    /// construction so the first per-session caller does not need to
    /// remember it.
    pub fn ensure_session(
        &self,
    ) -> Result<
        (
            Rc<crate::annotator::annrpython::RPythonAnnotator>,
            Rc<crate::translator::rtyper::rtyper::RPythonTyper>,
        ),
        crate::translator::rtyper::error::TyperError,
    > {
        if let Some((ann, rt)) = self.session.borrow().as_ref() {
            return Ok((ann.clone(), rt.clone()));
        }
        // Mirror `TranslationContext.buildannotator()` /
        // `buildrtyper()` (`translator.py:69-83`) which both
        // construct the slot value and assign it onto the
        // context (`self.annotator = ...`, `self.rtyper = ...`).
        //
        // We cannot call `tc.buildannotator()` directly here because
        // pyre shares the registry-owned `Bookkeeper` with the
        // annotator (the registry pre-populates HostObjects /
        // FunctionDescs) and `buildannotator()` always constructs a
        // fresh bookkeeper.  TODO(buildannotator-bookkeeper-share):
        // manual construction here mirrors the upstream sequence —
        // assign `tc.annotator` / `tc.rtyper`
        // through `set_annotator` / `set_rtyper` so any downstream
        // caller observing `translator.annotator` /
        // `translator.rtyper` (e.g. `rpbc.py` / `rclass.py` policy
        // gating) sees the bound values.
        let translator = std::rc::Rc::new(crate::translator::translator::TranslationContext::new());
        let annotator = crate::annotator::annrpython::RPythonAnnotator::new_with_translator(
            Some(std::rc::Rc::clone(&translator)),
            None,
            Some(self.bookkeeper.clone()),
            false,
        );
        translator.set_annotator(std::rc::Rc::downgrade(&annotator));
        let rtyper = std::rc::Rc::new(crate::translator::rtyper::rtyper::RPythonTyper::new(
            &annotator,
        ));
        rtyper.initialize_exceptiondata()?;
        // rtyper.py:182 — `self.exceptiondata.finish(self)` is the
        // prologue of `RPythonTyper.specialize()`.  In pyre's
        // incremental per-subject flow `specialize()` is not called as
        // a single driver; per-subject `specialize_more_blocks()` runs
        // instead (cutover.rs:specialize_legacy_graph_with_registry_
        // seed).  `finish_exceptiondata` registers the standard
        // exception class reprs once via `getclassrepr` (cached on
        // `rtyper.instance_reprs`), so we hoist it into the session
        // initializer here — the call is idempotent and must precede
        // the first per-subject specialize so exception types are
        // resolvable when block specialize hits a `raise` op.
        rtyper.finish_exceptiondata()?;
        // Session-prologue inheritance-id pass (rtyper.py:182 hoist,
        // analogue of the `rtype` task's `perform_normalizations` before
        // `specialize`).  Eagerly pre-register every registered
        // struct-root `ClassDef` — each call runs the full transitive
        // struct closure plus idempotent `Impossible`-guarded attr
        // projection, the same work the per-graph adapter front-runs via
        // `derive_subject_inputcells` — then run `assign_inheritance_ids`
        // ONCE so `classdef.minid`/`maxid` get a stable bracket BEFORE
        // any vtable materialises.  `ClassRepr.fill_vtable_root` bakes
        // `minid`/`maxid` as eager `Signed` constants into the
        // cross-graph-cached vtable, so the ids must never shift after
        // the first bake; a single pre-vtable pass over the complete
        // struct-root + standard-exception prefix (exceptions registered
        // by `finish_exceptiondata` just above) guarantees that for this
        // prefix.
        //
        // SCOPE LIMIT: this numbers ONLY the struct-root +
        // standard-exception prefix.  Classdefs minted mid-session
        // (`ClassDesc::pycall` instantiation, transitive base/call-family
        // `getuniqueclassdef`, exception exit-case classes,
        // `immutablevalue_hostobject`) are NOT numbered here.  An
        // append-safe such classdef (a baseless leaf — e.g. an
        // enum-variant transparent-ctor class) is numbered ON DEMAND by
        // `ClassesPBCRepr.redispatch_call` re-running
        // `assign_inheritance_ids`: that pass is now append-stable
        // (skip-if-numbered + append-only), so re-running it never re-sorts
        // or shifts an already-baked prefix id.  A fresh subclass of an
        // already-baked class is NOT append-safe (its bracket would need to
        // nest inside the parent's baked range); it stays unnumbered and
        // the per-graph path Skip-classifies it, exactly as before.
        for root in self.bookkeeper.pyre_struct_root_names() {
            // A malformed root must not abort the whole session — the
            // per-graph path Skip-classifies it later, mirroring the
            // populate path's `is_known_unported` tolerance.
            let _ = self.bookkeeper.getuniqueclassdef_for_struct_root(&root);
        }
        // Pre-mint enum variant subclasses so each `enum-base + variants`
        // subtree is numbered as one contiguous bracket by the single pass
        // below — a variant class minted lazily after the base's bracket is
        // baked would otherwise stay unnumbered (not append-safe) and
        // Skip-classify at its `ClassesPBCRepr` instantiation.
        self.bookkeeper.pre_register_enum_variant_classes();
        crate::translator::rtyper::normalizecalls::assign_inheritance_ids(&annotator);
        translator.set_rtyper(rtyper.clone());
        *self.session.borrow_mut() = Some((annotator.clone(), rtyper.clone()));
        Ok((annotator, rtyper))
    }

    /// The bookkeeper backing this registry — exposed so
    /// `specialize_legacy_graph_with_registry_returning_value_to_var`
    /// can pass the same bookkeeper to `RPythonAnnotator::new`, ensuring
    /// pre-registered entries are visible to the rtyper's `getdesc`
    /// lookup.
    pub fn bookkeeper(&self) -> &Rc<Bookkeeper> {
        &self.bookkeeper
    }

    /// Look up an existing entry without creating one.  Returns
    /// `None` when the path has not yet been registered.
    ///
    /// First checks the canonical `entries` map; if that misses,
    /// resolves through `aliases` to the canonical key and re-reads.
    /// Mirrors RPython `Bookkeeper.getdesc(pyobj)`'s "obj_key direct
    /// lookup" plus alias indirection (`bookkeeper.py:362-364`).
    pub fn lookup(&self, key: &FunctionPathKey) -> Option<Rc<PyreFunctionEntry>> {
        if let Some(entry) = self.entries.borrow().get(key) {
            return Some(entry.clone());
        }
        let canonical = self.aliases.borrow().get(key).cloned()?;
        self.entries.borrow().get(&canonical).cloned()
    }

    /// The already-started session pair, without creating one.
    /// `None` until the first [`Self::ensure_session`] call.
    pub fn session_if_started(
        &self,
    ) -> Option<(
        Rc<crate::annotator::annrpython::RPythonAnnotator>,
        Rc<crate::translator::rtyper::rtyper::RPythonTyper>,
    )> {
        self.session.borrow().clone()
    }

    /// Find the canonical entry whose `FunctionDesc.cache` holds a
    /// `PyGraph` wrapping exactly `graph` (flowspace graph identity).
    /// Used by the failed-subject-scope repair in `cutover` to map a
    /// poisoned shared callee graph back to its registry entry.
    pub fn find_entry_with_cached_graph(
        &self,
        graph: &crate::flowspace::model::GraphRef,
    ) -> Option<(FunctionPathKey, Rc<PyreFunctionEntry>)> {
        for (key, entry) in self.entries.borrow().iter() {
            let fd = entry.function_desc.borrow();
            let cache = fd.cache.borrow();
            if cache.values().any(|pg| Rc::ptr_eq(&pg.graph, graph)) {
                return Some((key.clone(), entry.clone()));
            }
        }
        None
    }

    /// Every key resolving to `canonical`'s entry: the canonical key
    /// itself plus all registered aliases pointing at it.
    pub fn keys_for_entry(&self, canonical: &FunctionPathKey) -> Vec<FunctionPathKey> {
        let mut keys = vec![canonical.clone()];
        keys.extend(
            self.aliases
                .borrow()
                .iter()
                .filter(|(_, c)| *c == canonical)
                .map(|(a, _)| a.clone()),
        );
        keys
    }

    /// Same as [`Self::lookup`] with a narrowly-scoped cross-module
    /// leaf-match fallback for callsites whose verbatim path missed.
    ///
    /// Resolution order, in priority:
    ///
    /// 1. Verbatim literal lookup ([`Self::lookup`]) — the registered
    ///    `FunctionPathKey` hits this whenever the caller spelled the
    ///    callee under one of its registered aliases.
    /// 2. Cross-module leaf-match safety net — scan the registry for
    ///    free-function entries whose last segment equals the query
    ///    leaf.  A single match resolves; multiple matches resolve only
    ///    when they all point at the same `host_object` (alias cluster
    ///    of one source function).  PyPy has no equivalent of this
    ///    global scan — it is a PRE-EXISTING-ADAPTATION covering the
    ///    code paths where the caller spelled the alias under a
    ///    crate-level re-export threaded through `pub use`.  The
    ///    convergence check keeps the resolution unambiguous; the
    ///    `query_is_free_fn` filter keeps it from picking up
    ///    impl-method candidates that share the leaf.
    ///
    /// Restrictions match the codewriter-side leaf-match:
    ///
    /// - Only single-segment (`["bare"]`) or two-segment
    ///   (`["caller_module", "bare"]`) keys participate.  Three-plus-
    ///   segment paths are explicit qualifications and must NOT
    ///   fuzzy-match — they either resolve verbatim or surface as a
    ///   real registry miss.
    /// - Free-function entries only: the `HostObject` must satisfy
    ///   `is_user_function()`.  Method-shaped hosts (registered via
    ///   `[impl_type, method_name]` aliases) keep their identity
    ///   distinct from a same-leaf free function.
    /// - Multi-match aliases of the same source (identical
    ///   `host_object` Arc identity) converge on a single resolution.
    ///
    /// `PYRE_STRICT_TARGET_TO_PATH=1` (audit-only) disables the
    /// cross-module safety net, keeping the strict-mode envelope
    /// consistent across registry-build and codewriter call resolution.
    pub fn lookup_with_leaf_match(&self, key: &FunctionPathKey) -> Option<Rc<PyreFunctionEntry>> {
        if let Some(entry) = self.lookup(key) {
            return Some(entry);
        }
        if std::env::var_os("PYRE_STRICT_TARGET_TO_PATH").is_some() {
            return None;
        }
        let segments = key.segments();
        if segments.is_empty() || segments.len() > 2 {
            return None;
        }
        let leaf = segments.last()?;
        // Free-fn vs impl-method shape disambiguator.  A free-fn query
        // path spells `[module_or_crate, fn_name]` where every pre-leaf
        // segment is snake_case (module conv); an impl-method path
        // spells `[..., ImplType, method_name]` where the segment
        // immediately preceding the leaf is the impl target type
        // (PascalCase by Rust naming conv).  When the query is a
        // free-fn shape but the registry has an unrelated impl-method
        // candidate sharing the leaf identifier (e.g. `OpRef::is_none`
        // colliding with free-fn `pyre_object::is_none`), the impl
        // candidate should be rejected — the callsite was already typed
        // by the caller as a non-method call.  Mirrors PyPy's bookkeeper
        // never confusing `is_none(obj)` with `obj.is_none()` because
        // `Constant.value` identity differs (free fn object vs bound
        // method object); pyre's segment-key carrier reproduces that
        // distinction structurally.
        let query_is_free_fn = segments
            .iter()
            .rev()
            .skip(1)
            .all(|s| !starts_with_uppercase(s));
        let entries_borrow = self.entries.borrow();
        let matches: Vec<&Rc<PyreFunctionEntry>> = entries_borrow
            .iter()
            .filter(|(k, e)| {
                if !e.host_object.is_user_function() {
                    return false;
                }
                let cand_segs = k.segments();
                if cand_segs.last().map(|s| s != leaf).unwrap_or(true) {
                    return false;
                }
                if query_is_free_fn {
                    // Reject impl-method candidates: the segment
                    // immediately preceding the leaf is PascalCase
                    // (impl target type) while every earlier segment is
                    // snake_case (module path).
                    if cand_segs.len() >= 2
                        && let Some(impl_ty) = cand_segs.iter().rev().nth(1)
                        && starts_with_uppercase(impl_ty)
                    {
                        return false;
                    }
                } else {
                    // Impl-method-shaped query (`[Owner, leaf]`, pre-leaf
                    // PascalCase): the candidate must carry the query as
                    // a suffix.  Method resolution never crosses the
                    // owner-type boundary — upstream resolves through the
                    // receiver's `ClassDesc`, so a same-leaf method of a
                    // different type is a different desc.  Without this
                    // gate, an unregistered `[OwnerA, leaf]` query would
                    // capture the sole `[OwnerB, leaf]` registration.
                    if cand_segs.len() < segments.len()
                        || cand_segs[cand_segs.len() - segments.len()..] != *segments
                    {
                        return false;
                    }
                }
                true
            })
            .map(|(_, e)| e)
            .collect();
        if matches.len() == 1 {
            return Some(matches[0].clone());
        }
        if !matches.is_empty() {
            // Multi-alias convergence: free-function registration
            // dual-publishes each graph under several segment-key
            // shapes (canonical + crate-stripped + alias).  When all
            // matches point at the same `host_object` identity
            // (`HostObject`'s `PartialEq` is Arc-pointer equality at
            // `flowspace/model.rs:208`), the alias cluster is
            // unambiguous.
            let first_host = matches[0].host_object.clone();
            let all_same = matches.iter().all(|e| e.host_object == first_host);
            if all_same {
                return Some(matches[0].clone());
            }
        }
        None
    }

    /// Look up or insert. The first caller for a given path:
    ///
    /// 1. constructs a synthetic `GraphFunc` (`name` = `key.name()`,
    ///    empty Dict `globals` matching upstream `Constant(dict)`'s
    ///    "no globals attached" sentinel),
    /// 2. wraps it in `HostObject::new_user_function` so
    ///    `bookkeeper.getdesc` would dispatch through the
    ///    `is_user_function()` arm (`bookkeeper.rs:955-956`),
    /// 3. constructs `FunctionDesc::new(bookkeeper, Some(host),
    ///    name, signature, None, None)` with pyre's authoritative
    ///    parameter signature,
    /// 4. inserts `DescEntry::function(fd)` into `bookkeeper.descs`
    ///    keyed by `host_object`'s Arc identity so any later
    ///    `bookkeeper.getdesc(host_object)` short-circuits at the
    ///    cache lookup at upstream `bookkeeper.py:362-364`
    ///    (`try: return self.descs[obj_key]; except KeyError`).
    ///
    /// Subsequent callers receive the same `Rc<PyreFunctionEntry>`
    /// (identity-cached, matching upstream `Bookkeeper.getdesc`'s
    /// "create once, share thereafter" contract at
    /// `bookkeeper.py:362-364`).
    ///
    /// Re-registration with a *different* `Signature` panics — upstream
    /// `description.py:205 FunctionDesc.__init__` binds the signature
    /// to the underlying Python function object once at creation time,
    /// and a single `Constant(<function foo>)` cannot legally project
    /// to two distinct `FunctionDesc.signature` values.  Pyre's
    /// adapter must surface the conflict at the producer site rather
    /// than silently route the second caller through the first
    /// caller's signature.
    pub fn get_or_register(
        &self,
        key: FunctionPathKey,
        signature: Signature,
    ) -> Rc<PyreFunctionEntry> {
        if let Some(existing) = self.entries.borrow().get(&key) {
            assert_eq!(
                existing.function_desc.borrow().signature,
                signature,
                "PyreCallRegistry.get_or_register: conflicting signatures for \
                 FunctionPath {:?} — upstream description.py:205 \
                 FunctionDesc.__init__ binds the signature to the underlying \
                 Python function object once at creation time, so a single \
                 callable cannot legally project to two distinct signatures.",
                key.segments(),
            );
            return existing.clone();
        }
        let name = key.name().to_string();
        // upstream `description.py:193-203 FunctionDesc.__init__` reads
        // the signature directly; pyre's authoritative signature comes
        // from the `OpKind::Call` lowering site (function declaration's
        // parameter list).
        let graph_func = GraphFunc::new(
            name.clone(),
            // upstream `func.__globals__` defaults to an empty dict in
            // `description.py` test fixtures; pyre's synthetic
            // GraphFunc has no Python-runtime globals so we feed the
            // sentinel directly.
            Constant::new(ConstValue::Dict(HashMap::new())),
        );
        let host_object = HostObject::new_user_function(graph_func);
        let function_desc = Rc::new(RefCell::new(FunctionDesc::new(
            self.bookkeeper.clone(),
            Some(host_object.clone()),
            name,
            signature,
            None,
            None,
        )));
        // Pre-register in bookkeeper.descs so the rtyper's
        // getdesc(host_object) lookup short-circuits at the cache.
        self.bookkeeper.descs.borrow_mut().insert(
            host_object.clone(),
            DescEntry::function(function_desc.clone()),
        );
        let entry = Rc::new(PyreFunctionEntry {
            host_object,
            function_desc,
        });
        self.entries.borrow_mut().insert(key, entry.clone());
        entry
    }

    /// Atomic register + cache prefill — the contract pyre callsites
    /// actually require to reach the rtyper's `direct_call` chain
    /// without falling through to upstream `description.py:1147-1169
    /// buildgraph` (which delegates to
    /// `translator.buildflowgraph(pyobj, false)` and would fail on
    /// pyre's synthetic `GraphFunc`-without-`HostCode`).
    ///
    /// Bare `get_or_register` only inserts into `bookkeeper.descs`; a
    /// caller that forgets the matching `prefill_default_cache` would
    /// register the FunctionDesc but leave the `cachedgraph` lookup at
    /// `description.rs:1037-1039` cold, causing `pair_simple_call` to
    /// invoke `buildgraph` on the synthetic GraphFunc.  Upstream's
    /// equivalent path (`Bookkeeper.getdesc(pyfunc) -> FunctionDesc`)
    /// guarantees the FunctionDesc is paired with a real Python
    /// `__code__` body that `buildflowgraph` can consume; pyre's port
    /// has no analogue so the prefill is mandatory.  Producers should
    /// prefer this entry over the two-step pair to avoid the trap.
    ///
    /// `pygraph` is the lifted leaf body (`cutover::lift_callee_to_pygraph`).
    /// Re-registration with the same key replaces the cached pygraph
    /// and asserts on signature equality (same contract as
    /// [`Self::get_or_register`]).
    pub fn register_callee(
        &self,
        key: FunctionPathKey,
        signature: Signature,
        pygraph: Rc<PyGraph>,
    ) -> Rc<PyreFunctionEntry> {
        let entry = self.get_or_register(key, signature);
        entry.prefill_default_cache(pygraph);
        entry
    }

    /// Register `alias_key` as an alternate spelling of `canonical_key`.
    ///
    /// RPython's `Bookkeeper.descs` is keyed by `Constant(<callable>)`
    /// identity (`bookkeeper.py:353`) — the same Python function
    /// object always yields the same `FunctionDesc` regardless of
    /// which `crate::foo` / `foo` addressing alias the caller spelled.
    /// Pyre's storage is keyed by `FunctionPathKey`, so when the
    /// producer (`lib.rs::analyze_*`) registers a free function under
    /// both its canonical path AND its `crate::`-prefixed alias, the
    /// populator must point both keys at the *same* canonical entry
    /// (otherwise the same callable acquires two `HostObject` Arc
    /// identities and two `FunctionDesc`s, which breaks upstream's
    /// "create once, share thereafter" contract at
    /// `bookkeeper.py:362-364`).
    ///
    /// `canonical_key` MUST already be registered (typically via
    /// [`Self::get_or_register`]).  Panics if the key is unknown, if
    /// `alias_key` is itself a canonical entry, or if `alias_key`
    /// already aliases to a different canonical key.
    pub fn alias(&self, alias_key: FunctionPathKey, canonical_key: &FunctionPathKey) {
        assert!(
            self.entries.borrow().contains_key(canonical_key),
            "PyreCallRegistry.alias: canonical key {:?} is not registered",
            canonical_key.segments(),
        );
        assert!(
            !self.entries.borrow().contains_key(&alias_key),
            "PyreCallRegistry.alias: alias key {:?} is already a canonical entry",
            alias_key.segments(),
        );
        if let Some(existing) = self.aliases.borrow().get(&alias_key) {
            assert_eq!(
                existing,
                canonical_key,
                "PyreCallRegistry.alias: alias key {:?} already maps to a different canonical key",
                alias_key.segments(),
            );
            return;
        }
        self.aliases
            .borrow_mut()
            .insert(alias_key, canonical_key.clone());
    }

    /// Number of canonical entries — one row per distinct callable
    /// identity.  Aliases do not count.  Diagnostic only.  Mirrors
    /// `len(Bookkeeper.descs)` (`bookkeeper.py:353-409`).
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.borrow().len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries.borrow().is_empty()
    }

    // `cached_pygraphs()` retired at Step 3 (2026-05-07).  The only
    // consumer was `seed_callee_blocks`, which itself was a
    // workaround for the missing `simple_call_SomeObject` dispatch.
    // RPython's annotator discovers callee bodies through `pycall ->
    // recursivecall -> addpendingblock` (description.py:283-305,
    // annrpython.py:315-336); pyre now matches that via Step 2's
    // dispatch registration.  Callers that need a callee's PyGraph
    // walk through `registry.lookup(key).function_desc.borrow().
    // cache.borrow()` directly.

    // The `value_to_var_by_key` and `constant_concretetypes_by_key`
    // side maps were retired at Issue 2.5 (2026-05-07).  They were
    // populated by `cutover::register_callees` / `populate_call_
    // registry_from_program` but had no production reader — the only
    // consumers were per-test invariants asserting the side-table
    // contents.  RPython parity: `Variable.concretetype` and
    // `Constant.concretetype` carry the per-variable / per-constant
    // LL type after specialise (`history.py:204` `same_constant`,
    // `model.py:438`); a parallel `FunctionPathKey -> Variable` side
    // map was a pyre-only divergence with no upstream peer.  Real
    // readers must consult `Variable.concretetype` directly through
    // the `PyGraph.graph` already cached on the `PyreFunctionEntry.
    // function_desc.cache`.
}

/// Rust naming convention shape check: PascalCase / leading-uppercase
/// idents are type / struct / enum names (impl method receivers); all-
/// snake_case idents are modules / functions / primitives.  Used by
/// [`PyreCallRegistry::lookup_with_leaf_match`] to distinguish
/// `Type::method` candidates from `module::fn` candidates when the
/// query path's shape disagrees.
fn starts_with_uppercase(s: &str) -> bool {
    s.chars().next().is_some_and(|c| c.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature(arg_names: &[&str]) -> Signature {
        Signature::new(
            arg_names.iter().map(|s| s.to_string()).collect(),
            None,
            None,
        )
    }

    #[test]
    fn lookup_returns_none_for_unregistered_path() {
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        assert!(
            registry
                .lookup(&FunctionPathKey::from_segments(["foo"]))
                .is_none(),
            "unregistered path must return None"
        );
        assert!(registry.is_empty());
    }

    #[test]
    fn get_or_register_constructs_entry_with_function_desc_and_host_object() {
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        let key = FunctionPathKey::from_segments(["foo"]);
        let entry = registry.get_or_register(key.clone(), signature(&["x"]));
        assert_eq!(
            entry.function_desc.borrow().name,
            "foo",
            "FunctionDesc must carry the registered name"
        );
        assert_eq!(
            entry.function_desc.borrow().signature.argnames,
            vec!["x".to_string()],
            "FunctionDesc must carry the registered Signature"
        );
        assert!(
            entry.host_object.is_user_function(),
            "registered HostObject must be a UserFunction so getdesc routes \
             through the is_user_function() arm (bookkeeper.rs:955-956)"
        );
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn get_or_register_preregisters_function_desc_in_bookkeeper_descs() {
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk.clone());
        let entry =
            registry.get_or_register(FunctionPathKey::from_segments(["foo"]), signature(&["x"]));
        // Direct cache lookup — Bookkeeper.descs[host_object] must
        // be DescEntry::function(entry.function_desc).
        let descs = bk.descs.borrow();
        let cached = descs
            .get(&entry.host_object)
            .expect("registry must pre-insert the entry under host_object's Arc identity");
        match cached.as_function() {
            Some(fd) => {
                assert!(
                    Rc::ptr_eq(&fd, &entry.function_desc),
                    "bookkeeper.descs entry must point at the same FunctionDesc Rc the \
                     registry returned, so getdesc identity-cache parity holds"
                );
            }
            None => panic!("expected a function desc, got {cached:?}"),
        }
    }

    #[test]
    fn bookkeeper_getdesc_short_circuits_on_pre_registered_host_object() {
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk.clone());
        let entry =
            registry.get_or_register(FunctionPathKey::from_segments(["foo"]), signature(&["x"]));
        // The headline parity claim: bookkeeper.getdesc(host_object)
        // returns the pre-registered FunctionDesc rather than
        // dispatching through newfuncdesc.  Without the pre-register,
        // newfuncdesc would build a default-Signature(["entry"]) FD
        // because the synthetic GraphFunc has no HostCode.
        let resolved = bk
            .getdesc(&entry.host_object)
            .expect("pre-registered host_object must resolve via cache short-circuit");
        let fd = resolved.as_function().expect("resolves to FunctionDesc");
        assert!(
            Rc::ptr_eq(&fd, &entry.function_desc),
            "getdesc must return the same FunctionDesc the registry registered"
        );
        assert_eq!(
            fd.borrow().signature.argnames,
            vec!["x".to_string()],
            "the cached FunctionDesc must carry pyre's authoritative argnames, \
             not newfuncdesc's default fallback"
        );
    }

    #[test]
    #[should_panic(expected = "conflicting signatures")]
    fn get_or_register_panics_on_signature_conflict() {
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        let key = FunctionPathKey::from_segments(["foo"]);
        registry.get_or_register(key.clone(), signature(&["x"]));
        // Second registration with a different signature must surface
        // the conflict — upstream description.py:205 binds the
        // signature to the underlying Python function once at
        // FunctionDesc creation time; pyre's adapter cannot silently
        // route the second caller through the first caller's signature.
        registry.get_or_register(key, signature(&["y"]));
    }

    #[test]
    fn get_or_register_returns_cached_entry_on_repeat_lookup() {
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        let key = FunctionPathKey::from_segments(["foo"]);
        let first = registry.get_or_register(key.clone(), signature(&["x"]));
        let second = registry.get_or_register(key.clone(), signature(&["x"]));
        assert!(
            Rc::ptr_eq(&first, &second),
            "second get_or_register must return the same Rc as the first \
             (Bookkeeper.getdesc identity-cache parity, bookkeeper.py:353-409)"
        );
        assert_eq!(
            registry.len(),
            1,
            "duplicate registration must not grow the map"
        );
    }

    #[test]
    fn lookup_after_register_returns_same_entry() {
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        let key = FunctionPathKey::from_segments(["foo"]);
        let registered = registry.get_or_register(key.clone(), signature(&["x"]));
        let looked_up = registry
            .lookup(&key)
            .expect("lookup after register must hit");
        assert!(
            Rc::ptr_eq(&registered, &looked_up),
            "lookup must return the same Rc that get_or_register registered"
        );
    }

    #[test]
    fn distinct_paths_register_distinct_entries() {
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk.clone());
        let foo =
            registry.get_or_register(FunctionPathKey::from_segments(["foo"]), signature(&["x"]));
        let bar =
            registry.get_or_register(FunctionPathKey::from_segments(["bar"]), signature(&["y"]));
        assert!(
            !Rc::ptr_eq(&foo, &bar),
            "different paths must produce distinct entries"
        );
        assert_ne!(
            foo.host_object, bar.host_object,
            "different paths must produce distinct HostObjects (different Arc identities)"
        );
        assert_eq!(
            bk.descs.borrow().len(),
            2,
            "bookkeeper.descs must have 2 entries"
        );
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn multi_segment_path_distinct_from_single_segment() {
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        let bare =
            registry.get_or_register(FunctionPathKey::from_segments(["foo"]), signature(&["x"]));
        let qualified = registry.get_or_register(
            FunctionPathKey::from_segments(["mymod", "foo"]),
            signature(&["x"]),
        );
        assert!(
            !Rc::ptr_eq(&bare, &qualified),
            "FunctionPath {{ segments: [\"foo\"] }} and \
             FunctionPath {{ segments: [\"mymod\", \"foo\"] }} are \
             distinct callees and must register separately"
        );
    }

    // Note: end-to-end coverage of `register_callee` lives in
    // `cutover.rs::anchor_call_function_path_registered_emits_simple_call`,
    // which builds a real lifted leaf PyGraph via
    // `lift_callee_to_pygraph` and verifies the cache pre-fill lets
    // the rtyper resolve the call to `Signed`.

    #[test]
    fn lookup_with_leaf_match_prefers_free_fn_over_impl_method_collision() {
        // Free-fn `pyre_object::is_none` colliding with impl-method
        // `OpRef::is_none` (unrelated `Option::is_none`-style helper on
        // a primitive enum).  A bare-call query `["pyre_object",
        // "is_none"]` (free-fn shape — every pre-leaf segment
        // snake_case) must resolve to the free fn, not the impl method
        // — `Bookkeeper.getdesc(Constant(is_none))` would key on the
        // free-fn Python callable identity, never colliding with the
        // bound-method object.
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        let free_fn = registry.get_or_register(
            FunctionPathKey::from_segments(["pyobject", "is_none"]),
            signature(&["obj"]),
        );
        let _impl_method = registry.get_or_register(
            FunctionPathKey::from_segments(["resoperation", "OpRef", "is_none"]),
            signature(&["self"]),
        );
        let resolved = registry
            .lookup_with_leaf_match(&FunctionPathKey::from_segments(["pyre_object", "is_none"]))
            .expect("free-fn-shape query must resolve via leaf-match");
        assert!(
            Rc::ptr_eq(&resolved, &free_fn),
            "leaf-match must pick the free-fn registration when the query is free-fn shape, \
             not the colliding impl-method one"
        );
    }

    #[test]
    fn lookup_with_leaf_match_still_resolves_impl_method_when_no_free_fn() {
        // No free-fn `is_none` registered, only the impl method.  The
        // free-fn-shape query should fall through and return None
        // rather than incorrectly latching onto the impl method —
        // upstream would surface the same "no such free fn" gap as a
        // bookkeeper lookup miss.
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        let _impl_method = registry.get_or_register(
            FunctionPathKey::from_segments(["resoperation", "OpRef", "is_none"]),
            signature(&["self"]),
        );
        let resolved = registry
            .lookup_with_leaf_match(&FunctionPathKey::from_segments(["pyre_object", "is_none"]));
        assert!(
            resolved.is_none(),
            "free-fn-shape query must NOT silently latch onto an impl-method \
             candidate sharing the leaf identifier"
        );
    }

    fn make_pygraph(name: &str, sig: Signature) -> Rc<PyGraph> {
        use crate::flowspace::model::{ConstValue, FunctionGraph};
        let func = GraphFunc::new(name, Constant::new(ConstValue::Dict(Default::default())));
        let startblock = crate::flowspace::model::Block::shared(vec![]);
        Rc::new(PyGraph {
            graph: Rc::new(RefCell::new(FunctionGraph::new(name, startblock))),
            func,
            signature: RefCell::new(sig),
            defaults: RefCell::new(Some(Vec::new())),
            access_directly: std::cell::Cell::new(false),
        })
    }

    #[test]
    fn find_entry_with_cached_graph_resolves_by_graph_identity() {
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        let key = FunctionPathKey::from_segments(["m", "f"]);
        let pygraph = make_pygraph("f", signature(&["x"]));
        registry.register_callee(key.clone(), signature(&["x"]), pygraph.clone());
        let other = make_pygraph("g", signature(&["x"]));
        registry.register_callee(
            FunctionPathKey::from_segments(["m", "g"]),
            signature(&["x"]),
            other,
        );
        let (found_key, found_entry) = registry
            .find_entry_with_cached_graph(&pygraph.graph)
            .expect("entry holding the cached graph must resolve");
        assert_eq!(found_key, key);
        assert!(
            found_entry
                .function_desc
                .borrow()
                .cache
                .borrow()
                .values()
                .any(|pg| Rc::ptr_eq(&pg.graph, &pygraph.graph))
        );
        let unknown = make_pygraph("h", signature(&["x"]));
        assert!(
            registry
                .find_entry_with_cached_graph(&unknown.graph)
                .is_none()
        );
    }

    #[test]
    fn keys_for_entry_includes_canonical_and_aliases() {
        let bk = Rc::new(Bookkeeper::new());
        let registry = PyreCallRegistry::new(bk);
        let canonical = FunctionPathKey::from_segments(["m", "f"]);
        registry.get_or_register(canonical.clone(), signature(&["x"]));
        let alias = FunctionPathKey::from_segments(["crate", "m", "f"]);
        registry.alias(alias.clone(), &canonical);
        let keys = registry.keys_for_entry(&canonical);
        assert!(keys.contains(&canonical));
        assert!(keys.contains(&alias));
        assert_eq!(keys.len(), 2);
    }
}
