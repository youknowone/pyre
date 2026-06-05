//! Call control — inline vs residual decision for function calls.
//!
//! RPython equivalent: `rpython/jit/codewriter/call.py` class `CallControl`.
//!
//! Decides which functions should be inlined into JitCode ("regular") and
//! which should remain as opaque calls ("residual").  Also handles builtin
//! (oopspec) and recursive (portal) call classification.

use std::collections::{HashMap, HashSet};

use majit_ir::descr::{DescrRef, EffectInfo, ExtraEffect, OopSpecIndex};
use majit_ir::value::Type;
use serde::{Deserialize, Serialize};

use crate::front::semantic::SemanticFunction;
use crate::jitcode::{BhCallDescr, CallResultErasedKey};
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation};
use crate::parse::CallPath;
use crate::policy::JitPolicy;

// ── Graph-based analyzers (RPython effectinfo.py + canraise.py) ────
//
// RPython uses BoolGraphAnalyzer subclasses that traverse call graphs
// transitively. Each analyzer checks for specific operations:
//   - RaiseAnalyzer: Abort terminators (canraise.py)
//   - VirtualizableAnalyzer: jit_force_virtualizable/jit_force_virtual ops
//   - QuasiImmutAnalyzer: jit_force_quasi_immutable ops
//   - RandomEffectsAnalyzer: unanalyzable external calls

/// RPython: canraise.py — result of raise analysis.
///
/// `_canraise()` returns True, False, or "mem" (only MemoryError).
/// call.py:337-355.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanRaise {
    /// Function cannot raise any exception.
    No,
    /// Function can only raise MemoryError.
    MemoryErrorOnly,
    /// Function can raise arbitrary exceptions.
    Yes,
}

/// Operation-level raise classification for `_canraise()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RaiseClass {
    No,
    #[allow(dead_code)]
    MemoryErrorOnly,
    Yes,
}

fn raise_class_can_raise(value: RaiseClass, ignore_memoryerror: bool) -> bool {
    match value {
        RaiseClass::No => false,
        RaiseClass::MemoryErrorOnly => !ignore_memoryerror,
        RaiseClass::Yes => true,
    }
}

/// RPython: DependencyTracker equivalent — caches transitive analysis results.
///
/// Each analyzer in RPython has its own `seen` set (via `analyze_direct_call`).
/// We cache the final result per CallPath so repeated queries are O(1).
#[derive(Default)]
pub struct AnalysisCache {
    can_raise: HashMap<CallPath, CanRaise>,
    forces_virtualizable: HashMap<CallPath, bool>,
    random_effects: HashMap<CallPath, bool>,
    can_invalidate: HashMap<CallPath, bool>,
    /// RPython: collect_analyzer (collectanalyze.py) — can this call trigger GC?
    can_collect: HashMap<CallPath, bool>,
}

/// RPython: readwrite_analyzer.analyze(op) return value.
///
/// Represents the set of read/write effects collected from graph traversal.
/// RPython uses a set of tuples like `("struct", T, fieldname)` and
/// `compute_bitstrings(all_descrs)` (`effectinfo.py:465`) materializes
/// the EffectInfo bitstrings at the end via
/// `bitstring.make_bitstring([descr.ei_index for descr in set])`.  Pyre
/// telescopes that pipeline by collecting the per-descr `ei_index`
/// values directly (DescrIndexRegistry already holds them); each `Vec<u32>`
/// is the running equivalent of one of PyPy's `_readonly_*`/`_write_*`
/// frozensets, deduped + sorted at conversion time.  Storing as `Vec<u32>`
/// rather than `u64` removes the 64-descr ceiling so bitstrings scale
/// with the global descr count, matching PyPy's arbitrary-length
/// `bitstring.py:3-13 make_bitstring` output.
pub struct WriteAnalysis {
    pub read_fields: Vec<u32>,
    pub write_fields: Vec<u32>,
    pub read_arrays: Vec<u32>,
    pub write_arrays: Vec<u32>,
    pub read_interiorfields: Vec<u32>,
    pub write_interiorfields: Vec<u32>,
    /// `effectinfo.py:294,301-305` `readonly_descrs_fields = []`
    /// populated via `add_struct → cpu.fielddescrof(T, fieldname)` from
    /// `("readstruct", T, fieldname)` tuples.
    pub field_read_descrs: Vec<majit_ir::descr::DescrRef>,
    /// `effectinfo.py:297,301-305` `write_descrs_fields = []` from
    /// `("struct", T, fieldname)` tuples.
    pub field_write_descrs: Vec<majit_ir::descr::DescrRef>,
    /// `effectinfo.py:296,313-325` `readonly_descrs_interiorfields = []`
    /// populated via `add_interiorfield → cpu.interiorfielddescrof(T,
    /// fieldname)` from `("readinteriorfield", T, fieldname)` tuples.
    pub interior_read_descrs: Vec<majit_ir::descr::DescrRef>,
    /// `effectinfo.py:299,313-325` `write_descrs_interiorfields = []`
    /// from `("interiorfield", T, fieldname)` tuples.
    pub interior_write_descrs: Vec<majit_ir::descr::DescrRef>,
    /// `effectinfo.py:295,307-311` `readonly_descrs_arrays = []` populated
    /// via `add_array → cpu.arraydescrof(ARRAY)` from `("readarray", T)`
    /// tuples (plus `("readinteriorfield", T, _)` tuples synthesised into
    /// `("readarray", T)` at `effectinfo.py:327-340`).
    pub array_read_descrs: Vec<majit_ir::descr::DescrRef>,
    /// `effectinfo.py:298,307-311` `write_descrs_arrays = []` mirror of
    /// the read side, populated from `("array", T)` tuples (plus
    /// `("interiorfield", T, _)` synthesised into `("array", T)`).
    pub array_write_descrs: Vec<majit_ir::descr::DescrRef>,
    /// RPython: `effects is top_set` — unanalyzable (random effects).
    pub is_top: bool,
}

/// Call descriptor — `AbstractDescr`-equivalent metadata for a call op.
///
/// RPython equivalent: the `CallDescr` returned by
/// `CallControl.getcalldescr()` (call.py:236-241), wrapping
/// `EffectInfo` and the cpu-level descr identity.  Upstream stores
/// the funcptr separately as `op.args[0]`; pyre carries the funcptr
/// identity on each `OpKind` variant's dedicated `funcptr` field
/// (model.rs:247) so this struct holds only the calldescr-side data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallDescriptor {
    /// RPython `CallDescr.arg_classes`: one char per non-void FUNC argument.
    pub arg_classes: String,
    /// RPython `CallDescr.result_type`.
    pub result_type: char,
    /// RPython `descr.py:664` `result_signed`.
    pub result_signed: bool,
    /// RPython `descr.py:662` `result_size`.
    pub result_size: usize,
    /// RPython `descr.py:665` `RESULT_ERASED`.
    pub result_erased: CallResultErasedKey,
    pub extra_info: EffectInfo,
}

impl CallDescriptor {
    pub fn known(extra_info: EffectInfo) -> Self {
        Self::from_signature(&[], Type::Void, extra_info)
    }

    pub fn override_effect(extra_info: EffectInfo) -> Self {
        Self::from_signature(&[], Type::Void, extra_info)
    }

    pub fn from_signature(arg_types: &[Type], result_type: Type, extra_info: EffectInfo) -> Self {
        let arg_classes = arg_types.iter().map(|tp| type_to_argclass(*tp)).collect();
        let (result_type, result_signed, result_size, result_erased) =
            result_layout_key(result_type);
        Self {
            arg_classes,
            result_type,
            result_signed,
            result_size,
            result_erased,
            extra_info,
        }
    }

    pub fn with_signature(mut self, arg_types: &[Type], result_type: Type) -> Self {
        let (result_type, result_signed, result_size, result_erased) =
            result_layout_key(result_type);
        self.arg_classes = arg_types.iter().map(|tp| type_to_argclass(*tp)).collect();
        self.result_type = result_type;
        self.result_signed = result_signed;
        self.result_size = result_size;
        self.result_erased = result_erased;
        self
    }

    pub fn get_extra_info(&self) -> EffectInfo {
        self.extra_info.clone()
    }

    pub fn arg_types(&self) -> Vec<Type> {
        self.arg_classes
            .chars()
            .filter_map(argclass_to_ir_type)
            .collect()
    }

    pub fn result_ir_type(&self) -> Type {
        result_char_to_ir_type(self.result_type)
    }

    pub fn to_bh_calldescr(&self) -> BhCallDescr {
        BhCallDescr {
            arg_classes: self.arg_classes.clone(),
            result_type: self.result_type,
            result_signed: self.result_signed,
            result_size: self.result_size,
            result_erased: self.result_erased,
            extra_info: self.extra_info.clone(),
        }
    }

    pub fn to_descr_ref(&self) -> majit_ir::descr::DescrRef {
        majit_ir::descr::make_call_descr_full(
            0,
            self.arg_types(),
            self.result_ir_type(),
            self.result_signed,
            self.result_size,
            self.extra_info.clone(),
        )
    }
}

fn type_to_argclass(tp: Type) -> char {
    match tp {
        Type::Int => 'i',
        Type::Ref => 'r',
        Type::Float => 'f',
        Type::Void => 'v',
    }
}

fn argclass_to_ir_type(c: char) -> Option<Type> {
    match c {
        'i' | 'S' => Some(Type::Int),
        'r' => Some(Type::Ref),
        'f' | 'L' => Some(Type::Float),
        'v' => None,
        _ => None,
    }
}

fn result_char_to_ir_type(c: char) -> Type {
    match c {
        'i' | 'S' => Type::Int,
        'r' => Type::Ref,
        'f' | 'L' => Type::Float,
        'v' => Type::Void,
        _ => Type::Void,
    }
}

fn result_layout_key(result_type: Type) -> (char, bool, usize, CallResultErasedKey) {
    let result_char = type_to_argclass(result_type);
    let result_signed = result_type == Type::Int;
    let result_size = match result_type {
        Type::Int | Type::Ref | Type::Float => 8,
        Type::Void => 0,
    };
    (
        result_char,
        result_signed,
        result_size,
        CallResultErasedKey::from_ir_layout(result_type, result_signed, result_size),
    )
}

/// Call classification — RPython `guess_call_kind()` return values.
///
/// RPython: the string literals `'regular'`, `'residual'`, `'builtin'`,
/// `'recursive'` returned by `CallControl.guess_call_kind()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallKind {
    /// Inline this call — callee graph is available and is a candidate.
    /// RPython: `'regular'` → produces `inline_call_*` jitcode instruction.
    Regular,
    /// Leave as a residual call in the trace.
    /// RPython: `'residual'` → produces `residual_call_*` jitcode instruction.
    Residual,
    /// Built-in operation with oopspec semantics (list ops, string ops, etc.)
    /// RPython: `'builtin'` → special handling per oopspec name.
    Builtin,
    /// Recursive call back to the portal (JIT entry point).
    /// RPython: `'recursive'` → produces `recursive_call_*` jitcode instruction.
    Recursive,
}

/// virtualizable.py:306-307 `VirtualizableInfo.is_vtypeptr(TYPE)` —
/// identity check for the VTYPEPTR (struct-pointer type) the
/// virtualizable describes.
///
/// TODO: pyre has no `lltype` so VTYPEPTR identity is
/// expressed via a `usize` token (typically the SizeDescr identity from
/// `majit_ir::descr::descr_identity`).  Hosts attach their rich
/// `VirtualizableInfo` (defined in `majit-metainterp::virtualizable`) by
/// implementing this trait so codewriter, which sits below metainterp in
/// the crate graph, can still consult `jd.virtualizable_info` per
/// `call.py:375-385 CallControl.get_vinfo`.
pub trait VirtualizableInfoHandle: std::fmt::Debug + Send + Sync {
    /// virtualizable.py:306-307 `is_vtypeptr(TYPE) → TYPE == self.VTYPEPTR`.
    fn is_vtypeptr(&self, vtypeptr_id: usize) -> bool;
}

/// greenfield.py `GreenFieldInfo.green_fields` membership test.
///
/// TODO: same crate-boundary reasoning as
/// `VirtualizableInfoHandle`.  Hosts implement this on their rich
/// `GreenFieldInfo` so `CallControl.could_be_green_field`
/// (call.py:387-393) can walk `jd.greenfield_info` without depending on
/// metainterp.
pub trait GreenFieldInfoHandle: std::fmt::Debug + Send + Sync {
    /// `(GTYPE, fieldname) in self.green_fields`.
    fn contains_green_field(&self, gtype: &str, fieldname: &str) -> bool;
}

/// `virtualref.py VirtualRefInfo` opaque carrier handle.
///
/// TODO: same crate-boundary reasoning as
/// `VirtualizableInfoHandle`.  `VirtualRefInfo` is defined in
/// `majit-metainterp::virtualref` (the JIT runtime side); codewriter
/// sits below metainterp in the crate graph and cannot import it.
/// Hosts implement this trait on `VirtualRefInfo` so
/// `CodeWriter.setup_vrefinfo` (`codewriter.py:91-94`) can store the
/// instance on `CallControl.virtualref_info`
/// (`call.py:22 virtualref_info = None`) for later forwarding to
/// `metainterp_sd.virtualref_info = codewriter.callcontrol.virtualref_info`
/// (`pyjitpl.py:2267`).  The three accessors expose the three `u32`
/// descriptor indices the rebuilt `VirtualRefInfo` consumes on the
/// metainterp side — `descr_virtual_token` / `descr_forced` index the
/// `JitVirtualRef` field descriptors, `descr_size` indexes the struct
/// size descriptor.
pub trait VirtualRefInfoHandle: std::fmt::Debug + Send + Sync {
    /// `virtualref.py:48 jit_virtual_ref_vtable` ↔ pyre
    /// `VirtualRefInfo.descr_virtual_token` — field descr index for
    /// `JitVirtualRef.virtual_token`.
    fn descr_virtual_token(&self) -> u32;
    /// `virtualref.py:49 jit_virtual_ref_vtable` ↔ pyre
    /// `VirtualRefInfo.descr_forced` — field descr index for
    /// `JitVirtualRef.forced`.
    fn descr_forced(&self) -> u32;
    /// `virtualref.py:48-49` size token ↔ pyre
    /// `VirtualRefInfo.descr_size` — size descr index for the
    /// `JitVirtualRef` struct itself.
    fn descr_size(&self) -> u32;
}

/// `warmspot.py:262 VirtualRefInfo(self)` ↔ majit codewriter-time
/// stand-in.  The trait values are the
/// `majit_metainterp::virtualref::descr` constants; this handle
/// duplicates them so [`CodeWriter::setup_vrefinfo`] can run before
/// `make_jitcodes` without majit-translate taking a `majit-metainterp`
/// dependency.  The constants are mirrored in
/// `majit_metainterp::virtualref::descr::{VIRTUAL_TOKEN, FORCED,
/// VREF_SIZE}`; the inverse-direction parity test in
/// `majit_metainterp::virtualref::tests::default_handle_constants_match`
/// asserts the two stay aligned.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultVirtualRefInfoHandle;

impl DefaultVirtualRefInfoHandle {
    /// Mirrors `majit_metainterp::virtualref::descr::VIRTUAL_TOKEN`
    /// (= `VREF_FIELD_VIRTUAL_TOKEN`, offset=8, Ref).
    pub const DESCR_VIRTUAL_TOKEN: u32 = 0x1000_0081;
    /// Mirrors `majit_metainterp::virtualref::descr::FORCED`
    /// (= `VREF_FIELD_FORCED`, offset=16, Ref).
    pub const DESCR_FORCED: u32 = 0x1000_0101;
    /// Mirrors `majit_metainterp::virtualref::descr::VREF_SIZE`.
    pub const DESCR_SIZE: u32 = 0x7F10;
}

impl VirtualRefInfoHandle for DefaultVirtualRefInfoHandle {
    fn descr_virtual_token(&self) -> u32 {
        Self::DESCR_VIRTUAL_TOKEN
    }
    fn descr_forced(&self) -> u32 {
        Self::DESCR_FORCED
    }
    fn descr_size(&self) -> u32 {
        Self::DESCR_SIZE
    }
}

/// Codewriter-internal `GreenFieldInfoHandle` built directly from a
/// jitdriver's `greens` list during `make_virtualizable_infos`.
///
/// `contains_green_field` is a pure structural query (`(gtype,
/// fieldname) in self.green_fields`), so no runtime identity is
/// required — unlike `is_vtypeptr` which has no codewrite-time
/// equivalent.  Hosts that want their richer
/// `GreenFieldInfoHandle` impl (e.g. `majit_metainterp::greenfield::
/// GreenFieldInfo` with descriptor indices) override this placeholder
/// via [`CallControl::set_jitdriver_greenfield_info`].
#[derive(Debug, Clone)]
pub struct StaticGreenFieldInfoHandle {
    /// greenfield.py:14 `self.red_index = jd.jitdriver.reds.index(objname)`
    /// — index of the unique green-field owning red.
    pub red_index: usize,
    /// greenfield.py:18 `self.green_fields = jd.jitdriver.ll_greenfields.values()`
    /// — `(GTYPE, fieldname)` pairs.
    pub green_fields: Vec<(String, String)>,
}

impl GreenFieldInfoHandle for StaticGreenFieldInfoHandle {
    fn contains_green_field(&self, gtype: &str, fieldname: &str) -> bool {
        self.green_fields
            .iter()
            .any(|(g, f)| g == gtype && f == fieldname)
    }
}

/// RPython: `JitDriverStaticData` — per-jitdriver metadata.
///
/// RPython `metainterp/jitdriver.py`: stores green/red variable names,
/// virtualizable info, portal graph reference, etc.
#[derive(Debug, Clone)]
pub struct JitDriverStaticData {
    /// RPython: `jitdriver_sd.index`
    pub index: usize,
    /// RPython: `jitdriver.greens` — loop-invariant variable names.
    pub greens: Vec<String>,
    /// RPython: `jitdriver.reds` — loop-variant variable names.
    pub reds: Vec<String>,
    /// RPython: `jitdriver.virtualizables` — names of red variables
    /// declared as virtualizable.  Drives warmspot.py:527-545
    /// `make_virtualizable_infos` selection.
    pub virtualizables: Vec<String>,
    /// Type names (GTYPEs) for each red variable, parallel to `reds`.
    ///
    /// TODO: upstream looks up GTYPE via
    /// `jd._JIT_ENTER_FUNCTYPE.ARGS[index]` at warmspot time; pyre
    /// propagates the matching struct names from `setup_jitdriver` so
    /// `make_virtualizable_infos` can build `(GTYPE, fieldname)`
    /// `green_fields` per greenfield.py:14 / warmspot.py:540-543.
    /// May be empty when the host has not yet supplied red types
    /// (legacy callers); in that case green-field construction
    /// substitutes the variable name as a fallback.
    pub red_types: Vec<String>,
    /// Portal graph path.
    pub portal_graph: CallPath,
    /// RPython: `jd.mainjitcode` (call.py:147) — `Arc<JitCode>` shell for
    /// the portal. Set by `grab_initial_jitcodes()`. Matches the
    /// metainterp-side `JitDriverStaticData.mainjitcode` shape so the
    /// codewriter→metainterp boundary is plain Arc handoff (no index
    /// translation step).
    pub mainjitcode: Option<std::sync::Arc<crate::jitcode::JitCode>>,
    /// warmspot.py:533 `jd.index_of_virtualizable = jitdriver.reds.index(vname)`.
    ///
    /// `-1` for drivers without a virtualizable, otherwise the slot
    /// in `reds` that holds the virtualizable.
    pub index_of_virtualizable: i32,
    /// warmspot.py:545 `jd.virtualizable_info = vinfos[VTYPEPTR]`.
    ///
    /// `None` for drivers that do not declare a virtualizable.  Set
    /// from the host runtime once the metainterp-side
    /// `VirtualizableInfo` is built — codewriter only sees the trait
    /// surface required by `CallControl::get_vinfo`.
    pub virtualizable_info: Option<std::sync::Arc<dyn VirtualizableInfoHandle>>,
    /// warmspot.py:519-525 `jd.greenfield_info = GreenFieldInfo(self.cpu, jd)`.
    ///
    /// Same plumbing as `virtualizable_info` — hosts attach their rich
    /// `GreenFieldInfo` via the trait so `CallControl.could_be_green_field`
    /// can walk it.
    pub greenfield_info: Option<std::sync::Arc<dyn GreenFieldInfoHandle>>,
}

/// Call control — decides inline vs residual for each call target.
///
/// RPython: `call.py::CallControl`.
///
/// In RPython, `CallControl` discovers all candidate graphs by traversing
/// from the portal graph, then for each `direct_call` operation it classifies
/// the call as regular/residual/builtin/recursive.
///
/// In majit-translate, we don't have RPython's function pointer linkage.
/// Instead, callee graphs are collected from parsed Rust source files
/// (free functions via `collect_function_graphs` and trait impl methods
/// via `extract_trait_impls`).

/// Process-local numeric graph identity — the faithful Rust analog of
/// RPython keying its call graph on graph OBJECT identity
/// (`rpython/jit/codewriter/call.py:30 self.jitcodes = {} # map {graph:
/// jitcode}`). `crate::model::FunctionGraph` has no Python-style object
/// identity, and it is NOT the flowspace graph that `GraphKey` /
/// `lltype::Func.graph` wraps, so this is a fresh CallControl-owned token
/// rather than a reuse of `GraphKey::as_usize`. Deliberately NOT
/// `Serialize`: identity is process-local; the serde-stable graph
/// reference stays `CallPath` (`CallTarget::Method.resolved_path`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GraphId(u32);

pub struct CallControl {
    /// Free function graphs: CallPath → FunctionGraph.
    /// RPython: `funcptr._obj.graph` linkage.
    function_graphs: HashMap<CallPath, FunctionGraph>,

    /// Leaf-name → free-function `CallPath`s with `owner_root.is_none()`.
    ///
    /// Pyre-only acceleration for `target_to_path`'s cross-module
    /// leaf-match fallback (line ~3086).  RPython resolves callees by
    /// `funcptr` identity, so this fallback has no upstream analogue;
    /// it exists because Rust's name resolution surfaces produce bare
    /// or 2-segment callsites that need to find the unique
    /// `function_graphs` registration whose path ends in the same
    /// leaf.  Without an index, the resolver iterates the whole
    /// `function_graphs` HashMap on every direct call (O(N_graphs)
    /// per call op), which dominates `jtransform_transform` for
    /// large opcode arms.  Maintained incrementally by the three
    /// registration helpers below — keep in lockstep with
    /// `function_graphs` so the lookup stays consistent.
    free_fn_leaf_index: HashMap<String, Vec<CallPath>>,

    /// Method-name leaf → impl-method `CallPath`s (graphs with
    /// `owner_root.is_some()`).  Same rationale as
    /// `free_fn_leaf_index`: pyre-only acceleration for
    /// `target_to_path`'s `CallTarget::Method` receiver-leaf
    /// fallback (line ~3273), which previously walked the whole
    /// `function_graphs` HashMap looking for paths ending in
    /// `[receiver_leaf, method_name]`.
    impl_method_leaf_index: HashMap<String, Vec<CallPath>>,

    /// CallPath → [`GraphId`] mint map — established at
    /// `register_function_graph` / `register_trait_method`. Many CallPaths
    /// (the alias spellings of one source graph) may map to the SAME
    /// GraphId; see [`Self::graph_id_by_identity`].
    graph_ids: HashMap<CallPath, GraphId>,
    /// Graph-object identity `(owner_root, name)` → [`GraphId`] — the
    /// surrogate for RPython `call.py:29 {graph: jitcode}` (keyed on the
    /// graph *object*). A free function registers clones of one source
    /// graph under several alias spellings, all sharing
    /// `(None, func.name)`; interning routes through this map so every
    /// alias converges on a single GraphId and the GraphId-keyed effect
    /// maps hold one entry per graph rather than one per alias. Impl
    /// methods key on `(Some(impl_type), method_name)`, so two impls of
    /// one trait (e.g. `PyFrame::push_value` vs `MIFrame::push_value`)
    /// stay on distinct GraphIds — the bare name alone collides, the
    /// owner_root disambiguates.
    graph_id_by_identity: HashMap<(Option<String>, String), GraphId>,
    /// Monotonic [`GraphId`] counter.
    next_graph_id: u32,

    /// Trait bindings: `(trait_root, method_name)` → `Vec<impl_type>`.
    ///
    /// Keyed by the *declaring trait* (impl's `trait_name` from
    /// `parse.rs:237`), so two traits exposing the same method name do
    /// not collide (RPython `call.py:94-114` indirect branch reads
    /// `op.args[-1].value` = exact candidate graph list, not a
    /// method-name global).  Inherent impls do not populate this map;
    /// they use `function_graphs` directly via `[impl_type, method_name]`.
    trait_method_impls: HashMap<(String, String), Vec<String>>,

    /// O(1) index over `trait_method_impls` keyed by method name alone:
    /// `method_name → [impl_type, …]` across every declaring trait.
    /// Maintained incrementally in `register_trait_method` so
    /// `impls_for_method_name` is a single lookup instead of a linear scan
    /// over every `(trait_root, method_name)` entry (the effect analysis
    /// runs that scan tens of thousands of times via `target_to_path`'s
    /// trait-resolution fallback). Each entry mirrors the exact push order
    /// into `trait_method_impls`, so the resolved multiset is identical.
    /// pyre-only resolution aid — RPython keys candidate lookup on the
    /// call op's exact candidate-graph list, not on a method-name global.
    ///
    /// Convergence path: retired with `impls_for_method_name` and the
    /// name-resolution layer once indirect-call ops carry their candidate
    /// graph list directly (`call.py:94 graphs_from`, `op.args[-1].value`),
    /// removing the need to recover candidates from a method-name global.
    method_to_impl_types: HashMap<String, Vec<String>>,

    /// Candidate targets — graphs we will inline.
    /// RPython: `CallControl.candidate_graphs`.
    candidate_graphs: HashSet<CallPath>,

    /// RPython: `JitDriverStaticData` — metadata for each jitdriver.
    /// `jitdrivers_sd[i]` holds the green/red arg layout for driver i.
    jitdrivers_sd: Vec<JitDriverStaticData>,

    /// Builtin targets (oopspec operations).
    /// RPython: detected via `funcobj.graph.func.oopspec`.
    builtin_targets: HashSet<GraphId>,

    /// RPython: `CallControl.jitcodes` — map {graph_key: JitCode}.
    /// Pyre stores `Arc<JitCode>` shells so callers (e.g.
    /// `IndirectCallTargets`, `JitDriverStaticData.mainjitcode`,
    /// `enum_pending_graphs`) can hold stable handles before the assembler
    /// commits the body via `OnceLock` interior mutability.
    jitcodes: indexmap::IndexMap<CallPath, std::sync::Arc<crate::jitcode::JitCode>>,

    /// RPython call.py:174-187 resolves `getfunctionptr(graph)` to the
    /// graph's real helper address before constructing `JitCode(name,
    /// fnaddr, calldescr)`. majit's source-only codewriter cannot derive
    /// that address from a parsed `CallPath`, so hosts may pre-bind the
    /// concrete trace-call surface here. Unbound paths still fall back to
    /// the stable symbolic address shim.
    function_fnaddrs: HashMap<CallPath, i64>,

    /// RPython `rtyper._builtin_func_for_spec_cache` (`support.py:805-807`).
    ///
    /// Memoises the `(c_func, LIST_OR_DICT)` pair upstream computes
    /// from `(oopspec_name, ll_args, ll_res, extrakey)`.  Pyre stores
    /// the full [`crate::jit_codewriter::support::BuiltinFuncSpec`]
    /// (the c_func analog + LIST_OR_DICT) keyed on the same tuple
    /// shape via [`crate::jit_codewriter::support::BuiltinFuncSpecCacheKey`].
    /// Wrapped in `RefCell` so `builtin_func_for_spec` can take a
    /// shared `&CallControl` reference matching upstream's `rtyper`
    /// parameter shape while still recording cache hits.
    builtin_func_for_spec_cache: std::cell::RefCell<
        HashMap<
            crate::jit_codewriter::support::BuiltinFuncSpecCacheKey,
            crate::jit_codewriter::support::BuiltinFuncSpec,
        >,
    >,

    /// `support.py:782-794 need_result_type` side-channel.
    ///
    /// RPython attaches the flag directly on the wrapper function
    /// (e.g. `LLtypeHelpers._ll_1_dict_keys.need_result_type = True`).
    /// Pyre cannot read attributes off a function pointer, so the
    /// flag is co-registered alongside the canonical name through
    /// [`Self::register_need_result_type`].  `setup_extra_builtin`
    /// reads from this map; missing canonical names default to
    /// [`crate::jit_codewriter::support::NeedResultType::No`],
    /// matching upstream's `getattr(..., 'need_result_type', False)`
    /// missing-attribute fallback.  Wrapped in `RefCell` so
    /// registration can use `&CallControl` consistently with the
    /// fnaddr / cache registries.
    need_result_type_registry:
        std::cell::RefCell<HashMap<String, crate::jit_codewriter::support::NeedResultType>>,

    /// `support.py:691-692 wrapper = wrapper(*extra)` factory registry.
    ///
    /// RPython's `_do_builtin_call` flow for `extra is not None`
    /// (`jtransform.py:480-484` for dict / array build helpers like
    /// `_ll_2_build_dict` / `_ll_2_build_list`) calls the wrapper
    /// function with the `extra` tuple to obtain a SPECIALIZED wrapper
    /// instance — `extra` carries the concrete lltype the build helper
    /// is being specialised for (e.g. `Ptr(STR)` for the str-keyed
    /// dict builder).  Pyre cannot synthesise specialized helpers at
    /// runtime without RPython's annotator, so hosts pre-build every
    /// `(canonical_name, extrakey)` specialisation and register the
    /// resulting fnaddr here.  `setup_extra_builtin` consults this
    /// map when `extra.is_some()`, falling back to `lookup_function_fnaddr`
    /// only when no factory specialization is registered — matching
    /// upstream's `wrapper = wrapper(*extra)` factory-call semantics
    /// while keeping the call site host-driven.  Empty registry today;
    /// no `INLINE_CALLS_TO` entry uses `extra`, but the surface lets
    /// dict-build / array-build helpers land without a structural
    /// adapter at the call site.
    builtin_factory_registry: std::cell::RefCell<HashMap<(String, String), i64>>,

    /// RPython `all_jitcodes` materialized incrementally by
    /// `CodeWriter.make_jitcodes()`. Entries are appended only after a
    /// jitcode has been fully assembled.
    finished_jitcodes: Vec<std::sync::Arc<crate::jitcode::JitCode>>,

    /// RPython: `CallControl.unfinished_graphs` — graphs pending assembly.
    unfinished_graphs: Vec<CallPath>,

    /// `call.py:22 virtualref_info = None` — class-level default,
    /// populated by `CodeWriter.setup_vrefinfo`
    /// (`codewriter.py:91-94`) before
    /// `MetaInterpStaticData.finish_setup` reads it at
    /// `pyjitpl.py:2267 self.virtualref_info =
    /// codewriter.callcontrol.virtualref_info`.  Stored behind the
    /// opaque [`VirtualRefInfoHandle`] trait so metainterp can rebuild
    /// its concrete `VirtualRefInfo` without codewriter taking a
    /// metainterp dependency.
    pub virtualref_info: Option<std::sync::Arc<dyn VirtualRefInfoHandle>>,

    /// RPython: `CallControl.callinfocollection` (call.py:31).
    /// Stores oopspec function info for builtin call handling.
    pub callinfocollection: majit_ir::CallInfoCollection,

    /// `cpu.fielddescrof(T, fieldname).get_ei_index()` /
    /// `cpu.arraydescrof(ARRAY).get_ei_index()` —
    /// process-shared sequential, collision-free `ei_index` allocation
    /// (`effectinfo.py:465 compute_bitstrings`).  Lives on `CallControl`
    /// (not `AnalysisCache`) so the bytecode emit path
    /// (`assembler.rs::arraydescrof`) and the writeanalyze walker
    /// (`call.rs:3614`/`:3633`) consult a single source of truth — two
    /// independent registries would assign different indices to the
    /// same `(item_ty, array_type_id)` pair and alias distinct ARRAY
    /// identities onto each other at `force_from_effectinfo`
    /// (`heap.py:540-560`, `heap.rs:839 array_effect_index`).
    pub descr_indices: DescrIndexRegistry,

    /// Pyre extension: targets whose source carries
    /// `#[majit_macros::elidable_cannot_raise]` — a user assertion that
    /// the callee never raises.  Honoured by `getcalldescr`'s elidable
    /// branch before consulting `_canraise`, because pyre's exception
    /// analyser defaults to `analyze_external_call → True` for any
    /// callee outside `function_graphs` (Vec::len, pyframe_get_pycode,
    /// etc.) and cannot recover `EF_ELIDABLE_CANNOT_RAISE` on its own
    /// the way RPython's analyser does upstream.  Without honouring the
    /// assertion, every `#[elidable_cannot_raise]` callsite is downgraded
    /// to `ElidableCanRaise` and pays an unnecessary GUARD_NO_EXCEPTION.
    cannot_raise_assertion_targets: HashSet<GraphId>,

    /// Pyre extension: targets whose source carries
    /// `#[majit_macros::elidable_or_memerror]` — a user assertion that
    /// the callee raises only MemoryError.  Mirrors
    /// `cannot_raise_assertion_targets` for the `EF_ELIDABLE_OR_MEMORYERROR`
    /// branch of RPython's `call.py:292-298` 3-way split.
    memerror_only_assertion_targets: HashSet<GraphId>,

    /// RPython: known struct types for `get_type_flag(ARRAY.OF)` → FLAG_STRUCT.
    /// If an array's element type is in this set, the array descriptor gets
    /// `ArrayFlag::Struct` (like RPython's `isinstance(TYPE, lltype.Struct)`).
    known_struct_names: HashSet<String>,

    /// RPython: struct field type info — maps struct_name → [(field_name, type_string)].
    /// Used by `resolve_array_identity` to determine the ARRAY element type
    /// when the base of an array access comes from a FieldRead.
    /// Equivalent to `op.args[0].concretetype.TO` in RPython's rtyped graph.
    struct_fields: crate::front::StructFieldRegistry,

    /// RPython: `symbolic.get_array_token(ARRAY, tsc)[0]` — array base size.
    /// Offset from the array object pointer to the first element.
    /// RPython GcArray layout: `[length (WORD)] [items...]`, so
    /// `basesize = carray.items.offset = sizeof(Signed) = WORD`.
    /// Default: WORD (8 on 64-bit) matching RPython's standard GcArray.
    pub array_header_size: usize,

    /// RPython: `symbolic.get_field_token(STRUCT, fieldname, tsc)` / `symbolic.get_size()`.
    /// Pre-computed struct layouts from actual runtime (std::mem::offset_of! etc.).
    /// When registered, provides exact (offset, size) for struct fields,
    /// bypassing the type-string heuristic. The runtime/proc-macro populates
    /// this via `set_struct_layout()`.
    pub struct_layouts: HashMap<String, StructLayout>,
    /// RPython: collectanalyze.py:15 — _gctransformer_hint_cannot_collect_.
    /// Functions known not to trigger GC collection. The collect_analyzer
    /// returns False immediately for these.
    pub cannot_collect_targets: HashSet<GraphId>,
    /// RPython: collectanalyze.py:21-25 — funcobj.random_effects_on_gcobjs.
    /// External functions whose calls may have random effects on GC objects.
    /// analyze_can_collect returns True immediately for these.
    pub external_gc_effects: HashSet<GraphId>,

    /// RPython: rlib/jit.py:250 `@oopspec(spec)` — `getattr(func, 'oopspec', None)`.
    /// Maps call target → oopspec string (e.g. "jit.isconstant(value)").
    /// codewriter/jtransform reads this to route calls through OopSpecIndex.
    pub oopspec_targets: HashMap<GraphId, String>,
    /// `support.py:705 argnames = ll_func.__code__.co_varnames[:nb_args]` —
    /// per-target positional parameter names, used by `parse_oopspec`
    /// to resolve identifier slots in the spec's `(...)` pattern to
    /// `Index(n)` placeholders.  Pyre cannot introspect a function
    /// pointer for argnames; populated by the walker through
    /// `mark_oopspec_argnames` whenever `#[oopspec(...)]` is paired
    /// with a function signature (`front::syn_metadata::collect_jit_hints`
    /// emits a companion `"oopspec_argnames:..."` hint that
    /// `lib.rs:600` consumes alongside `"oopspec:..."`).
    ///
    /// Keyed by [`GraphId`] (graph identity), resolved from the call
    /// target via [`Self::graph_id_of`]; the writer interns the path.
    pub oopspec_argnames: HashMap<GraphId, Vec<String>>,

    /// RPython: `_immutable_fields_` per class. Maps struct_name →
    /// `(field_name, rank)` pairs declared immutable / quasi-immutable.
    /// Consulted by the heuristic fallback in `all_interiorfielddescrs`
    /// when a struct has no registered StructLayout (Path 1 already carries
    /// `rank` on `StructFieldLayout`).  Rank encoding follows
    /// `rpython/rtyper/rclass.py:644-678 _parse_field_list`.
    pub immutable_fields_by_struct: HashMap<String, Vec<(String, crate::model::ImmutableRank)>>,
    /// `descr.py:364 is_pure = ARRAY_INSIDE._immutable_field(None)` parity.
    /// Pre-computed at `set_struct_fields` time by walking
    /// `immutable_fields_by_struct` for fields with `ImmutableRank::is_array()
    /// && is_immutable()` (i.e. the `field[*]` syntax) and recording the
    /// field's type string.  `arraydescrof_concrete` consults this set
    /// when minting an `ArrayDescr` so the `is_pure` flag propagates from
    /// the field-level annotation to the array-level descr — matching
    /// `lltype.Array(_immutable=True)` semantics where the array TYPE
    /// itself carries the immutability.  Pyre annotates per-field; the
    /// summary collapses field-level marks to type-level lookup keys.
    pub immutable_array_types: HashSet<String>,
    /// Per-source-file module path collected from
    /// `ParsedInterpreter.module_path` (`parse.rs:parse_source_with_module`).
    /// Indexed by file order at `analyze_pipeline_from_parsed` invocation
    /// time.  Each entry is the crate-stripped module path
    /// (`build.rs::module_path_from_source_file`).
    ///
    /// PyPy bookkeeper resolves names lexically per source-file scope
    /// (`bookkeeper.getdesc(value)` + `annrpython.Bookkeeper.position`).
    /// Pyre's analyzer currently routes the canonicalisation through
    /// the process-global `STRUCT_ORIGIN_REGISTRY` + `canonical_struct_name`
    /// (`majit-ir/src/descr.rs:148-225`); this carrier records the
    /// per-file module path so a future per-graph lexical resolver
    /// (orthodox PyPy `getdesc` parity) can consume it.
    pub parsed_module_paths: Vec<String>,
    /// Per-source-file `use` import map collected from
    /// `ParsedInterpreter.use_imports` (`parse.rs::collect_use_imports`).
    /// Keyed on `(source_module_path, alias)` → `fully_qualified_path`.
    /// Mirrors PyPy bookkeeper's lexical/import-scope name resolution
    /// (`annrpython.Bookkeeper.getdesc` + `frame.f_globals` lookups);
    /// pyre's `qualify_type_name` does not yet consult this table,
    /// awaiting the per-FunctionGraph use_imports carrier.
    /// Populated here as the data
    /// carrier so the future resolver lands without re-plumbing.
    pub use_imports: HashMap<(String, String), String>,
    /// Metadata-only registration carrier — `(segments,
    /// Signature, return_lltype)` for every `unsafe fn` and unsafe
    /// impl-method discovered in the parsed source set.  These callees
    /// cannot lower their bodies (`build_flow.rs:215` rejects
    /// `sig.unsafety.is_some()` because raw-pointer ops are not
    /// modelled), but `OpKind::Call::FunctionPath` sites still need
    /// the path registered in `PyreCallRegistry`.  Populated by
    /// `lib.rs::analyze_pipeline_from_parsed` via
    /// `flowspace::rust_source::register::extract_unsafe_fn_stubs`;
    /// consumed by
    /// `translator::rtyper::cutover::register_unsafe_fn_stubs` from
    /// `dual_gate_registry` after the function-graph populate pass.
    pub unsafe_fn_stubs: Vec<(
        Vec<String>,
        crate::flowspace::argument::Signature,
        crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
    )>,
}

/// Heuristic struct layout — NOT equivalent to RPython's `symbolic.get_field_token()`.
///
/// RPython delegates to `ll2ctypes.get_ctypes_type(STRUCT)` or `llmemory.offsetof()`
/// for actual C-level layout. This struct holds heuristic approximations computed
/// from Rust type strings via `from_type_strings()`. Offsets and sizes may diverge
/// from actual `#[repr(C)]` layout. The runtime SHOULD override via
/// `set_struct_layout()` with values from `std::mem::offset_of!()` /
/// `rpython/jit/backend/llsupport/symbolic.py` parity: `CallControl`'s
/// struct layouts resolve the layout-dependent `llmemory` symbolic
/// offsets (`FieldOffset` → `get_field_token`, struct `ItemOffset` →
/// `get_size`) when they reach constant emission.
impl crate::translator::rtyper::lltypesystem::llmemory::OffsetLayout for CallControl {
    fn field_offset(&self, struct_name: &str, fldname: &str) -> Option<i64> {
        let layout = self.struct_layouts.get(struct_name)?;
        layout
            .fields
            .iter()
            .find(|f| f.name == fldname)
            .map(|f| f.offset as i64)
    }

    fn struct_size(&self, struct_name: &str) -> Option<i64> {
        self.struct_layouts.get(struct_name).map(|l| l.size as i64)
    }
}

/// `std::mem::size_of::<T>()` for production use.
#[derive(Debug, Clone)]
pub struct StructLayout {
    /// RPython: `symbolic.get_size(STRUCT, tsc)` — total struct size.
    pub size: usize,
    /// Per-field layout: (field_name, offset, size, type).
    /// RPython: `symbolic.get_field_token(STRUCT, name, tsc) → (offset, size)`.
    pub fields: Vec<StructFieldLayout>,
}

/// Single field within a `StructLayout`.
#[derive(Debug, Clone)]
pub struct StructFieldLayout {
    pub name: String,
    /// RPython: `cfield.offset`
    pub offset: usize,
    /// RPython: `cfield.size`
    pub size: usize,
    /// RPython: `get_type_flag(getattr(STRUCT, fieldname))`
    pub flag: majit_ir::descr::ArrayFlag,
    /// IR type classification.
    pub field_type: majit_ir::value::Type,
    /// RPython: `STRUCT._immutable_field(fieldname)` —
    /// `rpython/rtyper/rclass.py:33-37` returns `False` for mutable
    /// fields and the matching `ImmutableRanking` (truthy) for fields
    /// listed in `_immutable_fields_`.  `None` here = mutable; `Some(rank)`
    /// = declared with that rank (`?`, `[*]`, `?[*]`, or plain).  Drives
    /// `FieldDescr.is_pure` + `is_quasi_immutable` and (future)
    /// `ArrayDescr.is_pure` for `[*]` arrays.
    pub rank: Option<crate::model::ImmutableRank>,
}

impl StructFieldLayout {
    /// RPython `STRUCT._immutable_field(fieldname)` truthiness — true iff
    /// the field appears in `_immutable_fields_` (any rank).
    pub fn is_immutable(&self) -> bool {
        self.rank.is_some()
    }

    /// True iff the rank is `IR_QUASIIMMUTABLE` / `IR_QUASIIMMUTABLE_ARRAY`.
    pub fn is_quasi_immutable(&self) -> bool {
        self.rank.map(|r| r.is_quasi_immutable()).unwrap_or(false)
    }
}

impl StructLayout {
    /// Build a StructLayout from type-string heuristic.
    /// Used at pipeline init to populate struct_layouts from struct_fields.
    /// The runtime can later override with actual layout via set_struct_layout().
    ///
    /// `immutable_field_ranks`: map from field name → `ImmutableRank` for
    /// every entry in the owning class's `_immutable_fields_` declaration.
    /// RPython `STRUCT._immutable_field(fieldname)` returns the matching
    /// `ImmutableRanking` for these; fields not in the map are mutable.
    pub fn from_type_strings(
        fields: &[(String, String)],
        known_structs: &std::collections::HashSet<String>,
        known_struct_sizes: &std::collections::HashMap<String, usize>,
        immutable_field_ranks: &std::collections::HashMap<String, crate::model::ImmutableRank>,
    ) -> Self {
        // RPython: symbolic.get_array_token() computes itemsize for ANY struct,
        // even those with nested structs. UnsupportedFieldExc only affects
        // all_interiorfielddescrs (field enumeration), NOT the struct size.
        // So we always compute the full size, but mark has_nested_struct to
        // clear interior field descriptors.
        let has_nested_struct = fields
            .iter()
            .any(|(_, type_str)| known_structs.contains(type_str.as_str()));
        let mut offset: usize = 0;
        let mut layout_fields = Vec::new();
        for (name, type_str) in fields {
            // heaptracker.py:62-67: skip Void, padding, and typeptr fields.
            // typeptr is handled separately (not enumerated by all_fielddescrs).
            if name == "typeptr" || name.starts_with("c__pad") {
                // heaptracker.py:64-67
                // typeptr is still counted for offset calculation below.
                let sz = if known_structs.contains(type_str.as_str()) {
                    known_struct_sizes
                        .get(type_str.as_str())
                        .copied()
                        .unwrap_or(std::mem::size_of::<usize>())
                } else {
                    get_type_flag(type_str).2
                };
                if sz > 0 {
                    let align = sz.min(std::mem::size_of::<usize>());
                    offset = (offset + align - 1) & !(align - 1);
                    offset += sz;
                }
                continue;
            }
            let (flag, field_type, field_size) = if known_structs.contains(type_str.as_str()) {
                // RPython: symbolic.get_field_token(STRUCT, fieldname) returns the
                // actual embedded struct size, not just a pointer size.
                let nested_size = known_struct_sizes
                    .get(type_str.as_str())
                    .copied()
                    .unwrap_or(std::mem::size_of::<usize>());
                (
                    majit_ir::descr::ArrayFlag::Struct,
                    majit_ir::value::Type::Ref,
                    nested_size,
                )
            } else {
                get_type_flag(type_str)
            };
            if field_type == majit_ir::value::Type::Void || field_size == 0 {
                continue;
            }
            // RPython: alignment is typically min(field_size, WORD).
            let align = field_size.min(std::mem::size_of::<usize>());
            offset = (offset + align - 1) & !(align - 1);
            let rank = immutable_field_ranks.get(name).copied();
            layout_fields.push(StructFieldLayout {
                name: name.clone(),
                offset,
                size: field_size,
                flag,
                field_type,
                rank,
            });
            offset += field_size;
        }
        // RPython: heaptracker.py:89-90 — if nested struct exists,
        // all_interiorfielddescrs raises UnsupportedFieldExc, so
        // interior field descriptors are not enumerable. Clear fields
        // but keep the correct size.
        if has_nested_struct {
            layout_fields.clear();
        }
        let max_align = fields
            .iter()
            .map(|(_, ty)| {
                if known_structs.contains(ty.as_str()) {
                    // Use actual nested struct size for alignment.
                    known_struct_sizes
                        .get(ty.as_str())
                        .copied()
                        .unwrap_or(std::mem::size_of::<usize>())
                        .min(std::mem::size_of::<usize>())
                } else {
                    get_type_flag(ty).2
                }
            })
            .filter(|s| *s > 0)
            .max()
            .unwrap_or(8);
        let size = if offset > 0 {
            (offset + max_align - 1) & !(max_align - 1)
        } else {
            0
        };
        StructLayout {
            size,
            fields: layout_fields,
        }
    }
}

/// Sequential descriptor index assignment — majit equivalent of
/// `cpu.fielddescrof(T, fieldname).get_ei_index()` /
/// `cpu.arraydescrof(ARRAY).get_ei_index()`.
///
/// RPython: each descriptor gets an index unique within its namespace
/// (fields, arrays, interiorfields) via `effectinfo.py:465-538
/// compute_bitstrings()` — the outer `for key in descrs:` loop resets
/// `mapping = {}` per namespace, so indices can collide across
/// namespaces.  Indices are monotonic `u32` (no upper bound from the
/// bitstring representation; `make_bitstring` (`bitstring.py:3-13`)
/// sizes the byte vector to `(max_index + 7) / 8`).  Pyre mirrors this
/// with three independent counters (`next_field_index`,
/// `next_array_index`, `next_interiorfield_index`).
///
/// Array descriptors are keyed by `(item_ty, array_type_id, len_offset)` per
/// RPython's `cpu.arraydescrof(ARRAY)`, which distinguishes by ARRAY
/// lltype identity, including `ARRAY._hints['nolength']`
/// (`GcArray(Signed)` vs `GcArray(Ptr(STRUCT_X))`, `effectinfo.py:307-311`).
/// Interior-field descriptors are keyed by
/// `(array_type_id, field_name)` per
/// `cpu.interiorfielddescrof(ARRAY, fieldname)` — a separate namespace
/// from struct field indices.
#[derive(Default)]
pub struct DescrIndexRegistry {
    /// Interior-mutable so that both the writeanalyze walker
    /// (`call.rs:3614`/`:3633`) and the bytecode emit path
    /// (`assembler.rs::arraydescrof`) can publish ei_index through
    /// `&CallControl` without threading a `&mut` borrow through
    /// `getcalldescr(&self, …)` and `assemble_with_callcontrol`.
    inner: std::cell::RefCell<DescrIndexRegistryInner>,
}

#[derive(Default)]
struct DescrIndexRegistryInner {
    /// (owner_root, field_name) → unbounded `ei_index` per
    /// `effectinfo.py:465 compute_bitstrings`. The value scales with the
    /// global descr count; `bitstring.make_bitstring` (`bitstring.py:3-13`)
    /// produces a bytestring whose length matches the largest index.
    field_indices: HashMap<(Option<String>, String), u32>,
    /// (item_ty_discriminant, array_type_id, len_offset) → unbounded `ei_index`.
    /// RPython: cpu.arraydescrof(ARRAY).get_ei_index()
    array_indices: HashMap<(u8, Option<String>, Option<usize>), u32>,
    /// (array_type_id, field_name) → unbounded `ei_index`.
    /// RPython: cpu.interiorfielddescrof(ARRAY, fieldname).get_ei_index()
    /// Separate from field_indices — RPython keys on (ARRAY, fieldname)
    /// not (STRUCT, fieldname).
    interiorfield_indices: HashMap<(Option<String>, String), u32>,
    next_field_index: u32,
    next_array_index: u32,
    next_interiorfield_index: u32,
}

impl DescrIndexRegistry {
    /// RPython: `cpu.fielddescrof(T, fieldname).get_ei_index()`.
    ///
    /// Returns the unbounded per-descr `ei_index` matching PyPy's
    /// `bitstring.py:3-13 make_bitstring(lst)` — the bitstring length
    /// scales with the maximum index, not capped at any width
    /// (`effectinfo.py:465 compute_bitstrings`).
    pub fn field_index(&self, owner_root: &Option<String>, field_name: &str) -> u32 {
        let mut inner = self.inner.borrow_mut();
        let key = (owner_root.clone(), field_name.to_string());
        if let Some(&idx) = inner.field_indices.get(&key) {
            return idx;
        }
        let idx = inner.next_field_index;
        inner.next_field_index += 1;
        inner.field_indices.insert(key, idx);
        idx
    }

    /// RPython: `cpu.arraydescrof(ARRAY).get_ei_index()`
    pub fn array_index(
        &self,
        item_ty_discriminant: u8,
        array_type_id: &Option<String>,
        len_offset: Option<usize>,
    ) -> u32 {
        let mut inner = self.inner.borrow_mut();
        let key = (item_ty_discriminant, array_type_id.clone(), len_offset);
        if let Some(&idx) = inner.array_indices.get(&key) {
            return idx;
        }
        let idx = inner.next_array_index;
        inner.next_array_index += 1;
        inner.array_indices.insert(key, idx);
        idx
    }

    /// RPython: `cpu.interiorfielddescrof(ARRAY, fieldname).get_ei_index()`
    pub fn interiorfield_index(&self, array_type_id: &Option<String>, field_name: &str) -> u32 {
        let mut inner = self.inner.borrow_mut();
        let key = (array_type_id.clone(), field_name.to_string());
        if let Some(&idx) = inner.interiorfield_indices.get(&key) {
            return idx;
        }
        let idx = inner.next_interiorfield_index;
        inner.next_interiorfield_index += 1;
        inner.interiorfield_indices.insert(key, idx);
        idx
    }
}

impl CallControl {
    /// RPython: `CallControl.__init__`.
    pub fn new() -> Self {
        Self {
            function_graphs: HashMap::new(),
            free_fn_leaf_index: HashMap::new(),
            impl_method_leaf_index: HashMap::new(),
            graph_ids: HashMap::new(),
            graph_id_by_identity: HashMap::new(),
            next_graph_id: 0,
            trait_method_impls: HashMap::new(),
            method_to_impl_types: HashMap::new(),
            candidate_graphs: HashSet::new(),
            jitdrivers_sd: Vec::new(),
            builtin_targets: HashSet::new(),
            jitcodes: indexmap::IndexMap::new(),
            function_fnaddrs: HashMap::new(),
            builtin_func_for_spec_cache: std::cell::RefCell::new(HashMap::new()),
            need_result_type_registry: std::cell::RefCell::new(HashMap::new()),
            builtin_factory_registry: std::cell::RefCell::new(HashMap::new()),
            finished_jitcodes: Vec::new(),
            unfinished_graphs: Vec::new(),
            virtualref_info: None,
            callinfocollection: majit_ir::CallInfoCollection::new(),
            descr_indices: DescrIndexRegistry::default(),
            cannot_raise_assertion_targets: HashSet::new(),
            memerror_only_assertion_targets: HashSet::new(),
            known_struct_names: HashSet::new(),
            struct_fields: crate::front::StructFieldRegistry::default(),
            // RPython: symbolic.get_array_token(GcArray(T))[0] = carray.items.offset
            // = sizeof(Signed) = WORD. Standard GcArray has a length field before items.
            //
            array_header_size: std::mem::size_of::<usize>(),
            struct_layouts: HashMap::new(),
            cannot_collect_targets: HashSet::new(),
            external_gc_effects: HashSet::new(),
            oopspec_targets: HashMap::new(),
            oopspec_argnames: HashMap::new(),
            immutable_fields_by_struct: HashMap::new(),
            immutable_array_types: HashSet::new(),
            parsed_module_paths: Vec::new(),
            use_imports: HashMap::new(),
            unsafe_fn_stubs: Vec::new(),
        }
    }

    /// Recompute `immutable_array_types` from
    /// `immutable_fields_by_struct` + `struct_fields`.  Walks every
    /// `(struct_name, field_name, rank)` triple, and when `rank` is an
    /// `ImmutableArray` (or `QuasiImmutableArray` for the future quasi-
    /// array path), records the field's type string into the set.
    /// Called after both `immutable_fields_by_struct` and `struct_fields`
    /// have been populated (`lib.rs::analyze_pipeline_from_parsed`).
    pub fn recompute_immutable_array_types(&mut self) {
        self.immutable_array_types.clear();
        for (struct_name, fields) in self.immutable_fields_by_struct.iter() {
            for (field_name, rank) in fields {
                if rank.is_array() && rank.is_immutable() {
                    if let Some(field_ty) = self.struct_fields.field_type(struct_name, field_name) {
                        self.immutable_array_types.insert(field_ty.to_string());
                    }
                }
            }
        }
    }

    /// RPython `rpython/rtyper/rclass.py:644-678` —
    /// `STRUCT._immutable_field(fieldname)` returns the `ImmutableRanking`
    /// when the field is listed in `_immutable_fields_`, or `None` for
    /// plain mutable fields.  Called by `jtransform.rewrite_op_getfield`
    /// (`rpython/jit/codewriter/jtransform.py:866-906`) to decide between
    /// mutable read, pure read, and the quasi-immut guard/record pair.
    pub fn field_immutability(
        &self,
        owner_root: Option<&str>,
        field_name: &str,
    ) -> Option<crate::model::ImmutableRank> {
        let owner = owner_root?;
        self.immutable_fields_by_struct
            .get(owner)
            .and_then(|fields| {
                fields
                    .iter()
                    .find(|(n, _)| n == field_name)
                    .map(|(_, rank)| *rank)
            })
    }

    /// RPython: register struct type names for get_type_flag(ARRAY.OF).
    pub fn set_known_struct_names(&mut self, names: HashSet<String>) {
        self.known_struct_names = names;
    }

    /// RPython: register struct field types for op.args[0].concretetype resolution.
    pub fn set_struct_fields(&mut self, registry: crate::front::StructFieldRegistry) {
        self.struct_fields = registry;
    }

    /// Program-wide struct field shapes accumulated at pipeline init.
    /// Threaded into the dual-gate bookkeeper so
    /// `getuniqueclassdef_for_struct_root` / `project_pyre_field_type` can
    /// project a struct's fields onto its classdef.
    pub fn struct_fields(&self) -> &crate::front::StructFieldRegistry {
        &self.struct_fields
    }

    /// RPython: isinstance(TYPE, lltype.Struct) check.
    pub fn is_known_struct(&self, name: &str) -> bool {
        self.known_struct_names.contains(name)
    }

    /// RPython: register actual struct layout from `symbolic.get_field_token()`.
    /// The runtime calls this with layouts from `std::mem::offset_of!()` etc.
    pub fn set_struct_layout(&mut self, struct_name: String, layout: StructLayout) {
        self.struct_layouts.insert(struct_name, layout);
    }

    /// RPython: resolve a struct field's type string.
    /// For `owner::field_name`, returns the full type of the field.
    pub fn field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        self.struct_fields.field_type(owner, field_name)
    }

    /// RPython: ordered `STRUCT._names` + field types for descriptor layout
    /// reconstruction. The order is required to reproduce
    /// `symbolic.get_field_token()`.
    pub fn struct_field_entries(&self, owner: &str) -> Option<&[(String, String)]> {
        self.struct_fields.fields.get(owner).map(Vec::as_slice)
    }

    /// `cpu.arraydescrof(ARRAY)` for callers that do not already hold the
    /// codewriter-side `array_index` — resolves it via
    /// [`DescrIndexRegistry::array_index`] keyed on
    /// `(value_type_discriminant(item_ty), array_type_id, len_offset)`, the same key
    /// the `writeanalyze` walker in this file uses, then
    /// hands the resulting `ei_index` to [`arraydescrof`].
    ///
    /// Bytecode emit (`assembler.rs::arraydescrof`) and the per-callee
    /// `writeanalyze` walker must agree on the same `(item_ty,
    /// array_type_id, len_offset)` → `ei_index` mapping; routing both through
    /// `descr_indices.array_index` mirrors `effectinfo.py:307-311`'s
    /// shared `cpu.arraydescrof(ARRAY).get_ei_index()` namespace and
    /// keeps `force_from_effectinfo` (`heap.py:540-560`) from aliasing
    /// distinct ARRAY identities onto the same bitstring slot.
    pub fn arraydescrof_for_type(
        &self,
        item_ty: &crate::model::ValueType,
        array_type_id: &Option<String>,
        ir_type: majit_ir::value::Type,
        len_offset: Option<usize>,
    ) -> majit_ir::descr::DescrRef {
        let idx = self.descr_indices.array_index(
            value_type_discriminant(item_ty),
            array_type_id,
            len_offset,
        );
        self.arraydescrof(idx, array_type_id, ir_type, len_offset)
    }

    /// RPython: `cpu.arraydescrof(ARRAY)` — descr.py:348-378.
    ///
    /// `array_type_id`: full ARRAY type string (e.g. `"Vec<Point>"`), matching
    /// RPython's ARRAY lltype identity. The element type is extracted via
    /// `extract_element_type_from_str()` for struct checks and flag resolution.
    ///
    /// `len_offset`: descr.py:359-362 — `None` for the `nolength=True`
    /// shape (`ARRAY_INSIDE._hints['nolength']`), `Some(off)` for
    /// length-prefixed layouts where `off` is the byte offset of the
    /// length word inside the allocation.
    pub fn arraydescrof(
        &self,
        idx: u32,
        array_type_id: &Option<String>,
        ir_type: majit_ir::value::Type,
        len_offset: Option<usize>,
    ) -> majit_ir::descr::DescrRef {
        self.arraydescrof_concrete(idx, array_type_id, ir_type, len_offset, Some(idx))
            as majit_ir::descr::DescrRef
    }

    /// Trait-typed sibling of [`Self::arraydescrof`] returning the cached
    /// `Arc<dyn ArrayDescr>` rather than the trait-erased `DescrRef`.
    /// Used by [`Self::interiorfielddescrof`] which needs the array-descr
    /// trait surface for `SimpleInteriorFieldDescr::new`, mirroring
    /// `descr.py:430 arraydescr = get_array_descr(gc_ll_descr, ARRAY)`
    /// reuse inside `get_interiorfield_descr` (`descr.py:404-437`).
    ///
    /// `ei_publish`: `Some(array_idx)` stamps `descr.set_ei_index(array_idx)`
    /// per the codewriter array-namespace pre-seed (`effectinfo.py:307-311`);
    /// `None` leaves `ei_index = u32::MAX` for callers that embed this
    /// array into a larger descr (e.g. `InteriorFieldDescr`) where the
    /// outer descr already owns its own ei-index slot and stamping the
    /// nested array with the outer's idx would corrupt
    /// `force_from_effectinfo`'s array-bitstring lookup.
    fn arraydescrof_concrete(
        &self,
        idx: u32,
        array_type_id: &Option<String>,
        ir_type: majit_ir::value::Type,
        len_offset: Option<usize>,
        ei_publish: Option<u32>,
    ) -> std::sync::Arc<dyn majit_ir::descr::ArrayDescr> {
        // RPython: ARRAY_INSIDE.OF — extract element type from full ARRAY type.
        let elem_name = array_type_id
            .as_deref()
            .and_then(|s| extract_element_type_from_str(s).or_else(|| Some(s.to_string())))
            .as_deref()
            .map(String::from);
        let elem_ref = elem_name.as_deref();
        let is_struct = elem_ref.is_some_and(|n| self.is_known_struct(n));
        // descr.py:363 — flag = get_type_flag(ARRAY_INSIDE.OF).
        // descr.py:354 — itemsize from symbolic.get_array_token().
        // descr.py:365 — ArrayDescr(basesize, itemsize, ..., flag).
        // Even for struct(struct), itemsize is correct from symbolic.
        let (flag, item_size, item_type) = if is_struct {
            (
                majit_ir::descr::ArrayFlag::Struct,
                elem_ref.map(|n| compute_struct_size(self, n)).unwrap_or(8),
                majit_ir::value::Type::Ref,
            )
        } else if let Some(elem) = elem_ref {
            let (f, t, s) = get_type_flag(elem);
            (f, s, t)
        } else {
            (
                majit_ir::descr::ArrayFlag::from_item_type(ir_type, false),
                8,
                ir_type,
            )
        };
        // descr.py:366-370 — `concrete_type='f'` when the element OF is
        // Float/SingleFloat (pyre has a single Float); otherwise `'\x00'`.
        let concrete_type = if item_type == majit_ir::value::Type::Float {
            'f'
        } else {
            '\x00'
        };
        // descr.py:359-362 + symbolic.get_array_token — basesize follows
        // the lltype's nolength flag:
        //   `nolength=True`  → no length header → items at offset 0
        //   `nolength=False` → length at lendescr.offset → items past header
        // pyre's CallControl uses a single-word array header
        // (`array_header_size = WORD`), so the length-prefixed shape places
        // items immediately after the length word at `len_offset + WORD`.
        let base_size = match len_offset {
            None => 0,
            Some(off) => off + self.array_header_size,
        };
        // `descr.py:348-378 get_array_descr(gccache, ARRAY_OR_STRUCT)`:
        // PyPy keys `cache[ARRAY_OR_STRUCT]` on the ARRAY lltype's
        // object identity.  Pyre's analogue is the codewriter
        // `array_type_id` Rust type spelling — distinct ARRAYs disagree
        // on this string.  Without one (legacy callers that emit array
        // ops without the identity carrier plumbed) PyPy has NO
        // "merge several ARRAYs into one slot" behavior; the
        // parity-correct response is to skip cache publish and mint
        // fresh per call so shape-coincident-but-logically-distinct
        // ARRAYs do not alias.
        let ad_arc: std::sync::Arc<dyn majit_ir::descr::ArrayDescr> = match array_type_id.as_deref()
        {
            Some(atid) => {
                let path_hash_u64 = majit_ir::descr::path_hash(atid);
                let nolength = len_offset.is_none();
                let length_offset = len_offset.unwrap_or(0);
                // `descr.py:348-378 get_array_descr` cache-or-mint:
                // `LLType::Array(path_hash(atid))` cache hit returns the
                // runtime `__majit_register_descrs`-or-prior-analyzer-
                // minted `Arc<SimpleArrayDescr>`; a miss mints a fresh
                // `Arc<SimpleArrayDescr>` and caches it.  Both sides
                // converge on one Arc per ARRAY identity.
                //
                // No `set_type_id` stamp here.  PyPy `gc.py:544-549
                // init_array_descr` stamps `descr.tid` from
                // `layoutbuilder.get_type_id(A)` — a dense sequential
                // GC type id allocated by the GC layoutbuilder.  Pyre
                // does not yet port the layoutbuilder analog;
                // analyzer-side `SimpleArrayDescr.type_id`
                // stays at 0 (the `get_array_descr` cache-miss-mint
                // default at `descr.rs:515`).  Runtime-registered
                // `SimpleArrayDescr` carries a real GC tid stamped at
                // module init (`LIST_TYPE_ID`, `DICT_TYPE_ID`, …) and
                // wins the cache slot when both paths race.  The
                // structural identity used for `_cache_array` lookups
                // is `SimpleArrayDescr.cache_key` (= `path_hash(atid)`,
                // stamped at descr.rs:526-528 inside `get_array_descr`),
                // kept fully separate from `type_id` per the trait doc
                // at descr.rs:2120-2131.
                // `descr.py:364 is_pure = ARRAY_INSIDE._immutable_field(None)`
                // parity: consult the array-type-keyed
                // `immutable_array_types` set populated from `field[*]`
                // annotations.  Field-level immutability collapses onto the
                // array-type identity here so the shared per-ARRAY descr's
                // `is_pure` propagates without per-call owner threading.
                let is_pure = array_type_id
                    .as_deref()
                    .is_some_and(|aid| self.immutable_array_types.contains(aid));
                let cached: majit_ir::descr::DescrRef =
                    majit_ir::descr::gc_cache().lock().unwrap().get_array_descr(
                        majit_ir::descr::LLType::Array(path_hash_u64),
                        base_size,
                        item_size,
                        flag,
                        ir_type,
                        nolength,
                        length_offset,
                        is_pure,
                        concrete_type, // descr.py:366-370 Float-only marker
                    );
                let ad_arc: std::sync::Arc<dyn majit_ir::descr::ArrayDescr> =
                    majit_ir::descr::descr_arc_as_array_descr(cached)
                        .expect("gc_cache._cache_array slot held a non-ArrayDescr Arc");
                // descr.py:372-375 — struct arrays get interior field
                // descriptors.  `set_all_interiorfielddescrs` is
                // `OnceLock` (first-call wins) so re-populating on
                // cache hit is safe.
                if is_struct {
                    if let Some(struct_name) = elem_ref {
                        let array_key = majit_ir::descr::LLType::Array(path_hash_u64);
                        let (descrs, _) =
                            all_interiorfielddescrs(self, struct_name, array_key, ad_arc.clone());
                        if !descrs.is_empty() {
                            ad_arc.set_all_interiorfielddescrs(descrs);
                        }
                    }
                }
                ad_arc
            }
            None => {
                // No identity carrier — local mint, no cache publish.
                // `elem_ref` is `None` here so `is_struct == false`;
                // interior field descrs are not required.  Length-
                // prefixed arrays still need a lendescr; mint it locally
                // (not via `gc_cache.get_field_arraylen_descr` which
                // would publish into `_cache_arraylen` keyed on a
                // synthetic slot that other no-identity arrays would
                // alias on).
                let lendescr: Option<majit_ir::descr::DescrRef> = len_offset.map(|off| {
                    use majit_ir::descr::SimpleFieldDescr;
                    // `descr.py:264 get_field_arraylen_descr` shape:
                    // `FieldDescr("len", ofs, WORD, FLAG_SIGNED)`.
                    let word_size = std::mem::size_of::<usize>();
                    std::sync::Arc::new(SimpleFieldDescr::new_with_name(
                        u32::MAX,
                        off,
                        word_size,
                        majit_ir::value::Type::Int,
                        false,
                        majit_ir::descr::ArrayFlag::Signed,
                        "len".to_string(),
                    )) as majit_ir::descr::DescrRef
                });
                let mut ad = majit_ir::descr::SimpleArrayDescr::with_flag(
                    u32::MAX,
                    base_size,
                    item_size,
                    0,
                    ir_type,
                    flag,
                );
                ad.lendescr = lendescr;
                ad.is_pure = false;
                ad.concrete_type = concrete_type;
                let arc: std::sync::Arc<majit_ir::descr::SimpleArrayDescr> =
                    std::sync::Arc::new(ad);
                majit_ir::descr_registry::register_array(arc.clone() as majit_ir::descr::DescrRef);
                arc as std::sync::Arc<dyn majit_ir::descr::ArrayDescr>
            }
        };
        // Per-trace codewriter id stamp — analyzer's
        // `descr_indices.array_index` identifies this descr in BhDescr
        // round-trips on `pyre-jit-trace::state` decoders.
        ad_arc.set_index(idx);
        // `effectinfo.py:465 compute_bitstrings` ei_index pre-seed
        // (analyzer publishes the codewriter array_index for
        // `force_from_effectinfo` lookup before `compute_bitstrings`
        // overwrites with the (eisetr, eisetw) class index).  `None`
        // skips — interiorfielddescrof embeds this array as the
        // container and owns its own ei-index slot.
        if let Some(arr_idx) = ei_publish {
            ad_arc.set_ei_index(arr_idx);
        }
        ad_arc
    }

    /// RPython: `cpu.fielddescrof(STRUCT, fieldname)` — descr.py:215-247.
    ///
    /// Mints a `SimpleFieldDescr` from the analyzer-time struct layout
    /// knowledge cached in `self.struct_fields`. The offset is the sum
    /// of preceding field sizes (registration order), the field size +
    /// element type come from `get_type_flag(field_type_str)` (same
    /// mechanism `arraydescrof` uses for primitive item sizing).
    ///
    /// `descr.py:218-239 get_field_descr` cache-or-mint: PyPy
    /// `cpu.fielddescrof` 는 `(STRUCT, fieldname)` identity 로 cache 하여
    /// 분석기와 런타임이 동일한 `FieldDescr` Arc 에 도달한다.  Pyre 는
    /// `path_hash(STRUCT)` 를 surrogate identity 로 쓰는데, 런타임은
    /// `__majit_type_id` = `path_hash(concat!(module_path!(), "::",
    /// stringify!(Struct)))` (`jit_struct.rs:92`) 로 def-path 를 해시
    /// 한다.  분석기 `field.owner_root` 는 `qualify_type_name(type_root,
    /// ctx.module_prefix)` 로 use-site 모듈 qualifier 를 받으므로
    /// `canonical_struct_name` (Followup 2, `5fcab5ddc8`,
    /// `STRUCT_ORIGIN_REGISTRY` 컨설팅) 를 거쳐 정의-모듈 qualifier 로
    /// 표준화해야 publish 슬롯과 일치한다.  `fielddescrof_concrete` 의
    /// `path_hash` boundary 에서 canonicalise (이 모듈 fix), 그리고
    /// `interiorfielddescrof`/`all_interiorfielddescrs` 와 동일 패턴
    /// (`call.rs:1492` + `:5067`).
    ///
    /// `ei_index` consequence (`effectinfo.py:465-538 compute_bitstrings`):
    /// `EI_INDEX_TABLE` side-table 폐기 (Round 4) 이후
    /// `descr.get_ei_index()` 만 readback (`heap.rs:866`).  canonicalise
    /// 가 분석기와 런타임을 같은 `register_keyed_field` Arc 로 모으므로
    /// `set_ei_index` 가 단일 슬롯에 도달, cross-module 호출 사이트도
    /// effectinfo bitstring 을 정상 소비한다.
    ///
    /// `None` when the struct is not registered in `self.struct_fields`
    /// (unanalyzable callee — caller silently skips the raw-set push).
    pub fn fielddescrof(
        &self,
        idx: u32,
        owner_root: &str,
        field_name: &str,
    ) -> Option<majit_ir::descr::DescrRef> {
        self.fielddescrof_concrete(idx, owner_root, field_name)
    }

    /// Trait-object sibling of [`Self::fielddescrof`] returning the
    /// resolved field-descr `Arc<dyn FieldDescr>` so analyzer and
    /// runtime share the SAME Arc — `set_ei_index` stamps land on
    /// the runtime's `PyreFieldDescr` instead of a parallel
    /// analyzer-mint `SimpleFieldDescr`.  Resolution order matches
    /// PyPy `descr.py:218-239 get_field_descr`:
    ///
    ///   1. `gc_cache.get_size_descr(struct_key)` → cache hit on
    ///      runtime-published `PyreSizeDescr` (publish key = same
    ///      `path_hash(strip_crate(module_path!())::Name)` analyzer
    ///      builds for `owner_root` via `qualify_type_name` +
    ///      `ParsedInterpreter.module_path`).
    ///   2. Walk `size_descr.all_fielddescrs()` matching the bare
    ///      `field_name` against each entry's `fd.field_name()` —
    ///      PyreFieldDescr names follow `"STRUCT.field"` per
    ///      descr.py:227 so the bare match uses suffix `.field_name`
    ///      OR exact `field_name` (the latter covers SimpleFieldDescr
    ///      mints that store the bare name).
    ///   3. Found → return that trait-obj Arc with `set_index(idx)`
    ///      applied (no-op on PyreFieldDescr — fd.index() is the
    ///      deterministic `stable_field_index` carried through
    ///      BhDescr structural fields, not via the atomic).
    ///   4. Miss → fall through to
    ///      `gc_cache.get_field_descr(struct_key, ...)` mint —
    ///      analyzer-only path; runtime convergence skipped for
    ///      this `(STRUCT, fieldname)` pair (logged absence of a
    ///      runtime `build_object_descr_group` publish).
    fn fielddescrof_concrete(
        &self,
        idx: u32,
        owner_root: &str,
        field_name: &str,
    ) -> Option<majit_ir::descr::DescrRef> {
        use majit_ir::descr::{LLType, path_hash};
        let fields = self.struct_fields.fields.get(owner_root)?;
        let mut offset: usize = 0;
        // `heaptracker.py:97-113 get_fielddescr_index_in(STRUCT, fieldname)`
        // — positional index in `STRUCT._names`, skipping `Void` and
        // `typeptr`.  Pyre's `struct_fields.fields[owner_root]` is the
        // flat field list (no nested-struct inlining at analyzer time),
        // so the enumeration position with `typeptr` skipped matches
        // PyPy's `index_in_parent`.  Threaded into
        // `gc_cache.get_field_descr` so the optimizer's
        // `field_index_in_parent`-keyed slot maps
        // (`optimizeopt/heap.rs FieldDescr::index_in_parent` consumer)
        // get the slot-per-field discrimination PyPy's heaptracker
        // assigns; the previous `0`-for-every-field stamp collided
        // every field of one struct onto slot 0.
        let mut field_pos: usize = 0;
        for (fname, fty) in fields {
            let (flag, ir_type, field_size) = get_type_flag(fty);
            if fname == "typeptr" {
                // heaptracker.py:102-103: `if name == 'typeptr': continue`
                continue;
            }
            if fname == field_name {
                // `descr.py:218-239 get_field_descr(gccache, STRUCT,
                // fieldname)` cache-or-mint: a `(STRUCT, fieldname)`
                // cache hit returns the runtime
                // `__majit_register_descrs`-minted Arc; a miss mints a
                // fresh `Arc<SimpleFieldDescr>` and caches.  Analyzer
                // and runtime sides converge on the same `Arc<
                // SimpleFieldDescr>` instance — PyPy's
                // `cpu.fielddescrof(STRUCT, fieldname)` per-tuple
                // object identity.
                //
                // After the cache-or-mint resolves, stamp the
                // analyzer's per-trace `idx` (from
                // `descr_indices.field_index`) onto the descr via
                // `set_index` so trace serialization round-trips on
                // the analyzer's id (`pyre-jit-trace::state` line
                // 5879/5933 matches by `fd.index() == field_idx`).
                // The atomic write is benign on cache hit — analyzer
                // is the sole writer of this slot (the macro path
                // discards the return).
                //
                // `descr.py:229 is_immutable = STRUCT._immutable_field(
                // fieldname)` parity: consult
                // `self.immutable_fields_by_struct` populated from
                // `#[jit_immutable_fields("name", "name?", "name[*]", ...)]`
                // attributes (`collect_immutable_field_attrs`,
                // `front/ast.rs:437`).  `ImmutableRank::Immutable` and
                // `ImmutableRank::ImmutableArray` map to plain
                // `is_immutable=true`; `QuasiImmutable*` ranks map to
                // `is_quasi_immutable=true` (the `record_quasiimmut_field`
                // path `jtransform.py:895-903`).  Missing entry retains
                // the mutable default.
                // `descr.py:108-118 cache[STRUCT]` 단일 identity 와
                // 정렬: 분석기측 `owner_root` 가 use-site 모듈 qualifier
                // (`qualify_type_name(type_root, ctx.module_prefix)`,
                // `front/ast.rs:255-261`) 인 동안 런타임 publish 는
                // `path_hash(strip_crate(module_path!())::Name)` (def-path).
                // `canonical_struct_name` 가 `STRUCT_ORIGIN_REGISTRY`
                // (Followup 2, `5fcab5ddc8`) 를 통해 bare-name 입력에
                // 정의-모듈 qualifier 를 붙여 `path_hash` 가 publish 슬롯
                // 과 일치하게 한다.  Cross-module use-site 도 같은
                // `register_keyed_field` Arc 에 도달하므로 `set_ei_index`
                // 가 런타임 reader 와 단일 슬롯에서 만난다.
                // `interiorfielddescrof`/`all_interiorfielddescrs` 와 동일
                // 패턴 (`call.rs:1492` + `:5067`).
                let canonical_owner = majit_ir::descr::canonical_struct_name(owner_root);
                let struct_key = LLType::Struct(path_hash(&canonical_owner));
                // `descr.py:234-238 get_field_descr` always calls
                // `get_size_descr(gccache, STRUCT, vtable)` to bind
                // `fielddescr.parent_descr` before returning. Pyre's
                // `get_field_descr` only reads `_cache_size` (no mint).
                // Mirror upstream by minting/hitting the parent here
                // from the analyzer's struct layout knowledge:
                // `compute_struct_size` matches `symbolic.get_size(STRUCT)`;
                // analyzer has no vtable / immutability surface so we
                // pass 0 / false (a runtime `build_object_descr_group`
                // publish under the same `struct_key` carries the real
                // vtable on its PyreSizeDescr — cache-hit returns
                // *that* Arc here unchanged).
                let struct_size = compute_struct_size(self, owner_root);
                use majit_ir::descr::Descr;
                let size_descr_arc = {
                    let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
                    gc.get_size_descr(struct_key.clone(), struct_size, 0, false)
                };
                // Field-walk pass (PyPy `cpu.fielddescrof` per-tuple
                // identity convergence): when the runtime published
                // a SizeDescr under this `struct_key` (via
                // `build_object_descr_group` →
                // `register_keyed_size`), its `PyreFieldDescr`s live
                // in `size_descr.all_fielddescrs()` already.  Return
                // that Arc directly so analyzer's `set_ei_index`
                // lands on the SAME slot the runtime reads.  Name
                // match: PyreFieldDescr stores `"STRUCT.field"`
                // (descr.py:227 format) so the analyzer's bare
                // `field_name` must match as suffix; SimpleFieldDescr
                // mints store either form so exact match also wins.
                if let Some(sd) = size_descr_arc.as_size_descr() {
                    let needle = format!(".{}", field_name);
                    for fd in sd.all_fielddescrs() {
                        let stored = fd.field_name();
                        if stored == field_name || stored.ends_with(&needle) {
                            fd.set_index(idx);
                            return Some(fd.clone() as majit_ir::descr::DescrRef);
                        }
                    }
                }
                // No runtime publish for this `(STRUCT, fieldname)`
                // tuple — fall back to analyzer-only mint.  The
                // `SimpleFieldDescr.parent_descr` Weak still binds to
                // the cached SizeDescr (which may be a PyreSizeDescr
                // if the runtime published the parent but not this
                // field, or a SimpleSizeDescr from line above).
                // `descr.py:229 STRUCT._immutable_field(fieldname)` parity.
                let rank = self.field_immutability(Some(owner_root), field_name);
                let is_immutable = rank.map(|r| r.is_immutable()).unwrap_or(false);
                let is_quasi_immutable = rank.map(|r| r.is_quasi_immutable()).unwrap_or(false);
                let descr = majit_ir::descr::gc_cache().lock().unwrap().get_field_descr(
                    struct_key,
                    field_name,
                    offset,
                    field_size,
                    ir_type,
                    is_immutable,
                    is_quasi_immutable,
                    flag,
                    field_pos,
                );
                descr.set_index(idx);
                return Some(descr as majit_ir::descr::DescrRef);
            }
            offset = offset.saturating_add(field_size);
            field_pos += 1;
        }
        None
    }

    /// RPython: `cpu.interiorfielddescrof(ARRAY, fieldname)` —
    /// descr.py:404-433. Mints an interior-field descr referring to a
    /// named field inside the struct element of `array_type_id`.
    ///
    /// Like [`Self::fielddescrof`] this produces a fresh analyzer-time
    /// Arc that does not share identity with the runtime descr
    /// The struct element is resolved by extracting
    /// `ARRAY.OF` from the full container type string (`Vec<Point>` →
    /// `"Point"`), then looking up the named field's offset/size from
    /// `self.struct_fields`. The containing array's
    /// `SimpleArrayDescr` is minted inline at `Ref` element type
    /// (PyPy's `consider_array(ARRAY)` filter at `effectinfo.py:392`
    /// only emits interiorfield effects for struct arrays where
    /// `ARRAY.OF` is a GcStruct).
    ///
    /// `None` when `array_type_id` is unresolved, the element type is
    /// not a registered struct, or the field name is absent. Caller
    /// silently skips the raw-set push.
    pub fn interiorfielddescrof(
        &self,
        idx: u32,
        array_type_id: &Option<String>,
        field_name: &str,
    ) -> Option<majit_ir::descr::DescrRef> {
        use majit_ir::descr::ArrayFlag;
        let array_str = array_type_id.as_deref()?;
        // ARRAY.OF.fieldname — extract the element type from the
        // container type, then look up field info in `self.struct_fields`.
        let elem_name =
            extract_element_type_from_str(array_str).or_else(|| Some(array_str.to_string()))?;
        // Validate the element is a known struct (`consider_array(ARRAY)`
        // filter at `effectinfo.py:392-397`).
        if !self.is_known_struct(&elem_name) {
            return None;
        }
        // PyPy `descr.py:435 fielddescr = get_field_descr(gc_ll_descr,
        // REALARRAY.OF, name)` — the inner FieldDescr.index is the
        // stable per-parent slot from `heaptracker.get_fielddescr_index_in()`
        // (descr.py:228), NOT the analyzer's interiorfield-namespace
        // idx.  Pyre's `fielddescrof_concrete` stamps the caller's
        // per-trace idx onto the shared `SimpleFieldDescr` cached at
        // `_cache_field[struct_key][bare_name]`; calling that path
        // from `interiorfielddescrof` would clobber the field-namespace
        // idx already stamped by a sibling `fielddescrof` call on the
        // same descr, breaking FieldDescr.index stability.  Resolve
        // the inner FieldDescr directly through `gc_cache.get_field_descr`
        // here so the analyzer's interiorfield idx lives ONLY on the
        // outer `SimpleInteriorFieldDescr.index`, mirroring PyPy's
        // FieldDescr / InteriorFieldDescr index namespace split.
        let fields = self.struct_fields.fields.get(&elem_name)?;
        let mut offset: usize = 0;
        let mut field_pos: usize = 0;
        let mut found: Option<std::sync::Arc<dyn majit_ir::descr::FieldDescr>> = None;
        for (fname, fty) in fields {
            let (flag, ir_type, field_size) = get_type_flag(fty);
            if fname == "typeptr" {
                continue;
            }
            if fname == field_name {
                // Use-import resolver: hash the canonical
                // `defining_module::Bare` form so analyzer hits the
                // same `_cache_size` slot the runtime's qualified
                // def-path dual-publish wrote to (PyPy
                // `cache[STRUCT]` lltype-object identity).  When the
                // resolver has no entry (legacy `parse_source` entry
                // without module_path), `canonical_struct_name`
                // returns the bare name verbatim and we hit the
                // simple-name slot — same Arc via dual-publish.
                let elem_canonical = majit_ir::descr::canonical_struct_name(&elem_name);
                let struct_key =
                    majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(&elem_canonical));
                // Seed parent (Round 6 parity, descr.py:238) so the
                // returned SizeDescr Arc carries vtable/all_fielddescrs
                // populated by either the runtime publish or the
                // analyzer-only mint.
                let struct_size = compute_struct_size(self, &elem_name);
                let size_descr_arc = {
                    let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
                    gc.get_size_descr(struct_key.clone(), struct_size, 0, false)
                };
                // Field-walk pass (same convergence pattern as
                // `fielddescrof_concrete` B-3): when runtime published
                // the element struct's SizeDescr via
                // `build_object_descr_group`, its PyreFieldDescrs live
                // in `all_fielddescrs` already.  Return that Arc so
                // `compute_bitstrings`' downstream `set_ei_index` on
                // the interior field's INNER FieldDescr lands on the
                // SAME slot the runtime reads.  Name match: bare or
                // `.{field_name}` suffix per descr.py:227.
                if let Some(sd) = size_descr_arc.as_size_descr() {
                    let needle = format!(".{}", field_name);
                    for fd in sd.all_fielddescrs() {
                        let stored = fd.field_name();
                        if stored == field_name || stored.ends_with(&needle) {
                            found = Some(fd.clone());
                            break;
                        }
                    }
                }
                if found.is_none() {
                    // No runtime publish for this `(STRUCT, fieldname)` —
                    // analyzer-only mint.
                    let rank = self.field_immutability(Some(&elem_name), field_name);
                    let is_immutable = rank.map(|r| r.is_immutable()).unwrap_or(false);
                    let is_quasi_immutable = rank.map(|r| r.is_quasi_immutable()).unwrap_or(false);
                    let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
                    let mint = gc.get_field_descr(
                        struct_key,
                        field_name,
                        offset,
                        field_size,
                        ir_type,
                        is_immutable,
                        is_quasi_immutable,
                        flag,
                        field_pos,
                    );
                    found = Some(mint as std::sync::Arc<dyn majit_ir::descr::FieldDescr>);
                }
                break;
            }
            offset = offset.saturating_add(field_size);
            field_pos += 1;
        }
        let field_descr = found?;
        // `descr.py:430 arraydescr = get_array_descr(gc_ll_descr, ARRAY)`:
        // PyPy `get_interiorfield_descr` reuses the per-ARRAY cached
        // array descr.  Pyre routes through `gc_cache.get_array_descr`
        // cache-or-mint so analyzer's `arraydescrof` and
        // `interiorfielddescrof` share Arc identity for the same
        // `LLType::Array(path_hash(atid))` cache key.  `try_downcast_arc`
        // recovers `Arc<SimpleArrayDescr>` from the cache's
        // `Arc<dyn Descr>` (the `Descr::as_any` override on
        // `SimpleArrayDescr` makes the cast sound), then cast
        // `as Arc<dyn ArrayDescr>` matches the trait-object field type
        // on `SimpleInteriorFieldDescr.array_descr`.
        let item_size = compute_struct_size(self, &elem_name);
        let base_size = self.array_header_size;
        let array_key = majit_ir::descr::LLType::Array(majit_ir::descr::path_hash(array_str));
        let cached: majit_ir::descr::DescrRef = {
            let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
            gc.get_array_descr(
                array_key.clone(),
                base_size,
                item_size,
                ArrayFlag::Struct,
                majit_ir::value::Type::Ref,
                false, // !nolength — length word at offset 0
                0,     // length_offset
                false, // is_pure
                '\x00',
            )
        };
        // `descr.py:348-378 get_array_descr` cache hit returns the
        // existing `Arc<SimpleArrayDescr>` in the slot, upcast to the
        // `ArrayDescr` trait object, keeping identity across runtime /
        // analyzer paths per PyPy `cpu.arraydescrof(ARRAY)`.
        let array_descr: std::sync::Arc<dyn majit_ir::descr::ArrayDescr> =
            majit_ir::descr::descr_arc_as_array_descr(cached)
                .expect("gc_cache._cache_array slot held a non-ArrayDescr Arc");
        // `descr.py:423-438 get_interiorfield_descr` cache-or-mint:
        // key is `(ARRAY, name, arrayfieldname=None)` — for the
        // GcArray-of-Structs case (which is the only case pyre's
        // analyzer mints) `arrayfieldname` is None per descr.py:431-432.
        // Pyre encodes `None` as the empty string in the tuple key.
        // Both arms of `make_simple_descr_group`'s array-of-struct
        // population (Task D) and this analyzer mint must hit the same
        // cache slot for `cpu.interiorfielddescrof` per-tuple identity.
        // `field_descr` is already `Arc<dyn FieldDescr>` (post B-4):
        // either PyreFieldDescr from the runtime publish walk OR
        // SimpleFieldDescr from the analyzer-only mint.  Matches
        // `SimpleInteriorFieldDescr::new` (descr.rs:3521) field type.
        let descr = majit_ir::descr::gc_cache()
            .lock()
            .unwrap()
            .get_interiorfield_descr(
                array_key,
                field_name.to_string(),
                String::new(),
                array_descr,
                field_descr,
            );
        descr.set_index(idx);
        Some(descr as majit_ir::descr::DescrRef)
    }

    /// Insert into `function_graphs` and (if free function) the
    /// `free_fn_leaf_index`.  The index is the only mutable side
    /// channel that mirrors `function_graphs`, so all writes go
    /// through this helper to keep them consistent.
    fn insert_function_graph_indexed(&mut self, path: CallPath, graph: FunctionGraph) {
        if let Some(leaf) = path.segments.last() {
            let bucket = if graph.owner_root.is_none() {
                self.free_fn_leaf_index.entry(leaf.clone()).or_default()
            } else {
                self.impl_method_leaf_index.entry(leaf.clone()).or_default()
            };
            if !bucket.contains(&path) {
                bucket.push(path.clone());
            }
        }
        self.function_graphs.insert(path, graph);
    }

    /// Idempotent mint: return the existing [`GraphId`] for `path` or
    /// allocate the next one. The single GraphId mint site, the surrogate
    /// for RPython `call.py:29 {graph: jitcode}` (keyed on the graph
    /// *object*).
    ///
    /// The port has no live graph objects, so identity is recovered from
    /// the registered graph's `(owner_root, name)` pair: a free function
    /// registers one source graph under several alias spellings (`lib.rs`
    /// `free_function_alias_paths`), and every alias clone shares
    /// `(None, func.name)` (`func.name` is unique per source — a duplicate
    /// would collide on the bare alias path and panic at
    /// `register_function_graph_alias`). Routing through
    /// [`Self::graph_id_by_identity`] converges all aliases of one source
    /// graph onto a single GraphId, so the GraphId-keyed effect maps
    /// (`oopspec_targets`, `cannot_collect_targets`, the assertion sets…)
    /// hold one entry per graph — matching graph-object identity rather
    /// than the path. Impl methods carry `owner_root = Some(impl_type)`
    /// and a bare method `name`, so `PyFrame::push_value` and
    /// `MIFrame::push_value` key on distinct pairs and stay separate. The
    /// caller must register the graph (via `register_function_graph` /
    /// `register_trait_method`, which insert before interning) before its
    /// first intern; a path with no registered graph (fnaddr-only) keeps
    /// per-path identity, which is correct as it has no graph to converge.
    fn intern_graph_id(&mut self, path: &CallPath) -> GraphId {
        if let Some(&id) = self.graph_ids.get(path) {
            return id;
        }
        let identity = self
            .function_graphs
            .get(path)
            .map(|g| (g.owner_root.clone(), g.name.clone()));
        let id = match identity
            .as_ref()
            .and_then(|key| self.graph_id_by_identity.get(key).copied())
        {
            Some(existing) => existing,
            None => {
                let fresh = GraphId(self.next_graph_id);
                self.next_graph_id += 1;
                if let Some(key) = identity {
                    self.graph_id_by_identity.insert(key, fresh);
                }
                fresh
            }
        };
        self.graph_ids.insert(path.clone(), id);
        id
    }

    /// Resolve a call target to its [`GraphId`], if registered. The
    /// in-process identity primitive paired with [`Self::target_to_path`]
    /// (which stays the CallPath-producing resolver for serde/display).
    fn graph_id_of(&self, target: &CallTarget) -> Option<GraphId> {
        self.target_to_path(target)
            .and_then(|p| self.graph_ids.get(&p).copied())
    }

    /// Register a free function graph.
    /// RPython: graphs are discovered via funcptr linkage.
    pub fn register_function_graph(&mut self, path: CallPath, graph: FunctionGraph) {
        // Insert before interning so `intern_graph_id` can read the graph's
        // canonical name and converge alias spellings onto one GraphId.
        self.insert_function_graph_indexed(path.clone(), graph);
        self.intern_graph_id(&path);
    }

    /// Register a free function graph together with its hints.
    /// `hints` mirror RPython `func._jit_*_` / `_elidable_function_`
    /// attributes; they are consulted by
    /// [`crate::policy::JitPolicy::look_inside_graph`].
    pub fn register_function_graph_with_hints(
        &mut self,
        path: CallPath,
        graph: FunctionGraph,
        hints: Vec<String>,
    ) {
        self.register_function_graph(path, graph.with_hints(hints));
    }

    /// Stamp hints onto an already-registered graph. Used by call sites
    /// that registered the graph through a different path (e.g.
    /// `register_trait_method`, whose dedup guard may have skipped a fresh
    /// insert) and need the graph's `_jit_*_` / `_elidable_function_` hints
    /// populated so `look_inside_graph` reads them off `graph.hints`.
    pub fn register_function_hints_for(&mut self, path: CallPath, hints: Vec<String>) {
        if !hints.is_empty()
            && let Some(graph) = self.function_graphs.get_mut(&path)
        {
            graph.hints = hints;
        }
    }

    /// Bind a real helper trace-call address to a canonical CallPath.
    ///
    /// RPython obtains this from `getfunctionptr(graph)`; majit callers
    /// that have access to the compiled helper surface can preload the
    /// equivalent integer address here so `get_jitcode()` and
    /// `fnaddr_for_target()` no longer fall back to symbolic hashes.
    pub fn register_function_fnaddr(&mut self, path: CallPath, fnaddr: i64) {
        self.function_fnaddrs.insert(path, fnaddr);
    }

    /// Consume a `#[jit_module]::__majit_helper_trace_fnaddrs()` entry.
    ///
    /// The macro-generated registry uses `module_path!()` and therefore
    /// prefixes paths with the crate name (e.g. `"mycrate::helpers::foo"`),
    /// while codewriter canonical paths are stored both as
    /// `"helpers::foo"` and `"crate::helpers::foo"`. Bind both aliases so
    /// either spelling resolves to the real helper address.
    ///
    /// Impl methods are *not* registered through this entry point —
    /// their canonical CallPath (`[impl_type_joined, method]`) carries
    /// `impl_type_joined` as a single `::`-preserving segment
    /// (parse.rs:702, lib.rs:406-433), which the simple `split("::")`
    /// strip here cannot recover.  Use
    /// `register_macro_impl_helper_trace_fnaddr` instead, fed from the
    /// macro's sibling registry `__majit_helper_impl_trace_fnaddrs()`.
    pub fn register_macro_helper_trace_fnaddr(&mut self, full_path: &str, fnaddr: i64) {
        if fnaddr == 0 {
            return;
        }
        let segments: Vec<&str> = full_path
            .split("::")
            .filter(|segment| !segment.is_empty())
            .collect();
        if segments.is_empty() {
            return;
        }
        let canonical = if segments.len() > 1 {
            &segments[1..]
        } else {
            &segments[..]
        };
        if canonical.is_empty() {
            return;
        }
        self.register_function_fnaddr(CallPath::from_segments(canonical.iter().copied()), fnaddr);
        let mut crate_alias = Vec::with_capacity(canonical.len() + 1);
        crate_alias.push("crate");
        crate_alias.extend(canonical.iter().copied());
        self.register_function_fnaddr(CallPath::from_segments(crate_alias), fnaddr);
    }

    /// Structured binding for an impl-method helper. `impl_type_joined`
    /// is the `::`-joined type path exactly as written at the `impl`
    /// header (e.g. `"a::Foo"` for `impl a::Foo { fn bar() }`), matching
    /// the parser's `self_ty_root` canonicalization (parse.rs:702 +
    /// front/ast.rs:106 `qualify_type_name`).  Registers
    /// `[impl_type_joined, method]` as a 2-segment CallPath where
    /// `impl_type_joined` is stored verbatim as a single segment — same
    /// shape `register_trait_method` / inherent method graphs use at
    /// lib.rs:406-433, so `get_jitcode()` resolves through to this real
    /// helper address instead of the symbolic hash fallback.  RPython
    /// `call.py:174-187 getfunctionptr(graph)` parity for `<Type>::method`
    /// and `<Type as Trait>::method`.
    pub fn register_macro_impl_helper_trace_fnaddr(
        &mut self,
        module_path_with_crate: &str,
        impl_type_as_written: &str,
        method: &str,
        fnaddr: i64,
    ) {
        if fnaddr == 0 || impl_type_as_written.is_empty() || method.is_empty() {
            return;
        }
        // front/ast.rs:106 `qualify_type_name`: bare types take the
        // current module prefix; already-qualified types keep their
        // exact written form.  Module prefix is everything after the
        // first `::`-separated segment (the crate name) of
        // `module_path_with_crate`, matching the parser's `prefix`
        // argument which starts empty at crate root and accumulates
        // submodule idents (parse.rs:314-318).
        let module_prefix = module_path_with_crate
            .split_once("::")
            .map(|(_crate, rest)| rest)
            .unwrap_or("");
        let impl_type_joined = if impl_type_as_written.contains("::") || module_prefix.is_empty() {
            impl_type_as_written.to_string()
        } else {
            format!("{module_prefix}::{impl_type_as_written}")
        };
        self.register_function_fnaddr(CallPath::for_impl_method(&impl_type_joined, method), fnaddr);
    }

    /// Register a trait impl method graph.
    ///
    /// Also registers the graph in function_graphs under a synthetic
    /// CallPath so that BFS in find_all_graphs can discover it.
    /// RPython: method graphs are reachable through funcptr._obj.graph
    /// linkage — we emulate this by dual registration.
    ///
    /// `trait_root` identifies the declaring trait for polymorphic
    /// resolution (inherent impls pass `None`).  Populating
    /// `trait_method_impls` under `(trait_root, method_name)` keeps two
    /// traits with the same method name distinct per `call.py:94-114`.
    pub fn register_trait_method(
        &mut self,
        method_name: &str,
        trait_root: Option<&str>,
        impl_type: &str,
        graph: FunctionGraph,
    ) {
        if let Some(trait_root) = trait_root {
            self.trait_method_impls
                .entry((trait_root.to_string(), method_name.to_string()))
                .or_default()
                .push(impl_type.to_string());
            self.method_to_impl_types
                .entry(method_name.to_string())
                .or_default()
                .push(impl_type.to_string());
        }
        // call.py:175-187 getfunctionptr(graph) — graph identity is
        // the key. Each impl gets a distinct CallPath via
        // `for_impl_method` so PyFrame's `push_value` and MIFrame's
        // `push_value` stay separate.
        let qualified_path = CallPath::for_impl_method(impl_type, method_name);
        // Impl-method graphs carry `owner_root = Some(impl_type)`, so they
        // are excluded from `free_fn_leaf_index` by the helper's filter.
        // Still route through it so a future `owner_root.is_none()` trait
        // path stays indexed automatically.
        if !self.function_graphs.contains_key(&qualified_path) {
            // Insert before interning so `intern_graph_id` reads the graph's
            // `(owner_root, name)` identity. Each impl method registers
            // exactly once under a distinct qualified path, and its
            // `owner_root = Some(impl_type)` keeps it on a GraphId separate
            // from other impls' same-named methods.
            self.insert_function_graph_indexed(qualified_path.clone(), graph);
            self.intern_graph_id(&qualified_path);
        }
    }

    /// Mark a target as the portal entry point.
    ///
    /// RPython: `setup_jitdriver(jitdriver_sd)` + `grab_initial_jitcodes()`.
    /// The portal set is derived on demand from `jitdrivers_sd` (RPython
    /// `jitdriver_sd_from_portal_graph`), so a portal seed is a jitdriver
    /// with no green/red layout. Used by tests that need a portal without a
    /// full driver registration; production seeds via `setup_jitdriver`.
    pub fn mark_portal(&mut self, path: CallPath) {
        self.setup_jitdriver(path, Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    /// `codewriter.py:91-94 CodeWriter.setup_vrefinfo(self, vrefinfo)`.
    ///
    /// ```python
    /// def setup_vrefinfo(self, vrefinfo):
    ///     # must be called at most once
    ///     assert self.callcontrol.virtualref_info is None
    ///     self.callcontrol.virtualref_info = vrefinfo
    /// ```
    ///
    /// In pyre the body is split between
    /// `pyre-jit::CodeWriter::setup_vrefinfo` (the warm-entry wrapper)
    /// and this method on the underlying `CallControl`.  Mirrors
    /// `setup_jitdriver` immediately above, which uses the same
    /// codewriter-wrapper-to-callcontrol routing for
    /// `codewriter.py:96-99`.
    pub fn setup_vrefinfo(&mut self, vrefinfo: std::sync::Arc<dyn VirtualRefInfoHandle>) {
        // codewriter.py:93 `assert self.callcontrol.virtualref_info is None`.
        assert!(
            self.virtualref_info.is_none(),
            "setup_vrefinfo: must be called at most once (codewriter.py:92)"
        );
        // codewriter.py:94 `self.callcontrol.virtualref_info = vrefinfo`.
        self.virtualref_info = Some(vrefinfo);
    }

    /// Register a JitDriver with its green/red/virtualizable layout.
    ///
    /// RPython: `CodeWriter.setup_jitdriver(jitdriver_sd)` (codewriter.py:96-99)
    /// + `jitdriver.virtualizables` (rlib/jit.py:601-603).
    /// Each jitdriver gets a sequential index.
    ///
    /// `red_types` mirrors `_JIT_ENTER_FUNCTYPE.ARGS` for the red slot
    /// portion (warmspot.py:540-543).  Pass an empty vector if the
    /// host hasn't propagated the runtime types yet — the
    /// green-field constructor in `make_virtualizable_infos` falls
    /// back to the variable name in that case.
    pub fn setup_jitdriver(
        &mut self,
        portal_graph: CallPath,
        greens: Vec<String>,
        reds: Vec<String>,
        virtualizables: Vec<String>,
        red_types: Vec<String>,
    ) {
        let index = self.jitdrivers_sd.len();
        debug_assert!(
            red_types.is_empty() || red_types.len() == reds.len(),
            "setup_jitdriver: red_types length must match reds when supplied",
        );
        self.jitdrivers_sd.push(JitDriverStaticData {
            index,
            greens,
            reds,
            virtualizables,
            red_types,
            portal_graph,
            mainjitcode: None,
            index_of_virtualizable: -1,
            virtualizable_info: None,
            greenfield_info: None,
        });
    }

    /// warmspot.py:528-545 `jd.virtualizable_info = vinfos[VTYPEPTR]`.
    ///
    /// Attach the host-built [`VirtualizableInfoHandle`] to the
    /// pre-registered driver at `index`.  Mirrors the upstream
    /// post-construction assignment that warmspot performs once the
    /// per-driver `VirtualizableInfo` map has been built.  Pyre's host
    /// runtime calls this between [`Self::setup_jitdriver`] and
    /// [`Self::find_all_graphs`] so that
    /// [`Self::get_vinfo`] returns the matching handle.
    pub fn set_jitdriver_virtualizable_info(
        &mut self,
        index: usize,
        info: std::sync::Arc<dyn VirtualizableInfoHandle>,
    ) {
        self.jitdrivers_sd[index].virtualizable_info = Some(info);
    }

    /// warmspot.py:519-525 `jd.greenfield_info = GreenFieldInfo(cpu, jd)`.
    ///
    /// Same staging pattern as
    /// [`Self::set_jitdriver_virtualizable_info`].  Hosts compute the
    /// green-field metadata once during driver setup and attach the
    /// handle here so [`Self::could_be_green_field`] can walk it.
    pub fn set_jitdriver_greenfield_info(
        &mut self,
        index: usize,
        info: std::sync::Arc<dyn GreenFieldInfoHandle>,
    ) {
        self.jitdrivers_sd[index].greenfield_info = Some(info);
    }

    /// warmspot.py:515-545 `WarmRunnerDesc.make_virtualizable_infos`.
    ///
    /// ```python
    /// def make_virtualizable_infos(self):
    ///     vinfos = {}
    ///     for jd in self.jitdrivers_sd:
    ///         jd.greenfield_info = None
    ///         for name in jd.jitdriver.greens:
    ///             if '.' in name:
    ///                 jd.greenfield_info = GreenFieldInfo(self.cpu, jd)
    ///                 break
    ///         if not jd.jitdriver.virtualizables:
    ///             jd.virtualizable_info = None
    ///             jd.index_of_virtualizable = -1
    ///             continue
    ///         else:
    ///             assert jd.greenfield_info is None, "XXX not supported yet"
    ///         jitdriver = jd.jitdriver
    ///         assert len(jitdriver.virtualizables) == 1    # for now
    ///         [vname] = jitdriver.virtualizables
    ///         jd.index_of_virtualizable = jitdriver.reds.index(vname)
    ///         index = jd.num_green_args + jd.index_of_virtualizable
    ///         VTYPEPTR = jd._JIT_ENTER_FUNCTYPE.ARGS[index]
    ///         if VTYPEPTR not in vinfos:
    ///             vinfos[VTYPEPTR] = VirtualizableInfo(self, VTYPEPTR)
    ///         jd.virtualizable_info = vinfos[VTYPEPTR]
    /// ```
    ///
    /// TODO: upstream owns this method on
    /// `WarmRunnerDesc` (warmspot.py:451) so it can mutate the single
    /// shared `jitdrivers_sd` list (the same Python list object is
    /// referenced by both `WarmRunnerDesc.jitdrivers_sd` and
    /// `MetaInterpStaticData.jitdrivers_sd`).  Pyre splits that list
    /// into two: codewriter `CallControl::jitdrivers_sd` (build.rs
    /// time) and metainterp `MetaInterpStaticData::jitdrivers_sd`
    /// (runtime), so the warmspot logic is invoked once per side at
    /// the matching lifecycle phase.  This call covers the codewriter
    /// side; the metainterp side is wired through
    /// `MetaInterp::set_virtualizable_info` at `JitDriver::new`
    /// (jitdriver.rs:285).
    ///
    /// `greenfield_info` is constructed in-place as a
    /// [`StaticGreenFieldInfoHandle`] (the codewriter-internal default;
    /// hosts can override via
    /// [`Self::set_jitdriver_greenfield_info`] with a richer impl such
    /// as `majit_metainterp::greenfield::GreenFieldInfo`).
    ///
    /// `vinfo_factory` mirrors the upstream `VirtualizableInfo(self,
    /// VTYPEPTR)` constructor (warmspot.py:543).  Pyre's codewriter
    /// crate sits below metainterp and therefore cannot reach the
    /// rich runtime constructor; the factory closure delegates to the
    /// host (e.g. pyre `build.rs` or runtime warm-up), which can
    /// either return a real
    /// `Arc<dyn VirtualizableInfoHandle>` or `None`.  When the
    /// factory returns `None`, the slot stays empty until the host
    /// later overrides it with [`Self::set_jitdriver_virtualizable_info`]
    /// at runtime — matching pyre's
    /// `MetaInterp::set_virtualizable_info` (jitdriver.rs:285) wiring.
    /// The factory receives `(jd_idx, vtypeptr_token)` where
    /// `vtypeptr_token` is the `red_types[index_of_virtualizable]`
    /// string the codewriter resolved.
    pub fn make_virtualizable_infos<VF>(&mut self, mut vinfo_factory: VF)
    where
        VF: FnMut(usize, &str) -> Option<std::sync::Arc<dyn VirtualizableInfoHandle>>,
    {
        // warmspot.py:516 `vinfos = {}` — per-VTYPEPTR cache so multiple
        // jitdrivers sharing the same virtualizable type reuse one handle.
        let mut vinfos: std::collections::HashMap<
            String,
            std::sync::Arc<dyn VirtualizableInfoHandle>,
        > = std::collections::HashMap::new();
        self.make_virtualizable_infos_inner(&mut vinfo_factory, &mut vinfos);
    }

    fn make_virtualizable_infos_inner<VF>(
        &mut self,
        vinfo_factory: &mut VF,
        vinfos: &mut std::collections::HashMap<String, std::sync::Arc<dyn VirtualizableInfoHandle>>,
    ) where
        VF: FnMut(usize, &str) -> Option<std::sync::Arc<dyn VirtualizableInfoHandle>>,
    {
        for jd_idx in 0..self.jitdrivers_sd.len() {
            // warmspot.py:519 `jd.greenfield_info = None`
            self.jitdrivers_sd[jd_idx].greenfield_info = None;
            // warmspot.py:520-524 — scan greens for '.' and split each
            // dotted name into `(objname, fieldname)`.  Upstream
            // collects the unique `objname` set, then resolves each
            // `(objname, fieldname)` to `(GTYPE, fieldname)` via
            // `jd.jitdriver.ll_greenfields` for the
            // `green_fields` list and via `_JIT_ENTER_FUNCTYPE.ARGS`
            // for the index→GTYPE mapping (greenfield.py:14-19,
            // warmspot.py:540-543).
            let mut seen: Vec<String> = Vec::new();
            let mut parsed_pairs: Vec<(String, String)> = Vec::new();
            for name in &self.jitdrivers_sd[jd_idx].greens {
                if let Some((objname, fieldname)) = name.split_once('.') {
                    if !seen.iter().any(|s| s == objname) {
                        seen.push(objname.to_string());
                    }
                    parsed_pairs.push((objname.to_string(), fieldname.to_string()));
                }
            }
            // warmspot.py:520-524 (cont.): if any dotted green was seen,
            // construct GreenFieldInfo(cpu, jd) — pyre's codewriter has
            // no `cpu` so we build the structural placeholder
            // `StaticGreenFieldInfoHandle` here; hosts override with the
            // descriptor-aware metainterp variant via
            // `set_jitdriver_greenfield_info`.
            if !seen.is_empty() {
                // greenfield.py:11-13 `assert len(seen) == 1`.
                assert_eq!(
                    seen.len(),
                    1,
                    "greenfield.py:11 — only one instance with green fields supported, found {seen:?}",
                );
                let objname = &seen[0];
                // greenfield.py:14 `red_index = jd.jitdriver.reds.index(objname)`.
                let red_index = self.jitdrivers_sd[jd_idx]
                    .reds
                    .iter()
                    .position(|r| r == objname)
                    .unwrap_or_else(|| {
                        panic!(
                            "greenfield.py:14 — green-field owner {objname:?} not in reds {:?}",
                            self.jitdrivers_sd[jd_idx].reds
                        )
                    });
                // greenfield.py:18 `self.green_fields = jd.jitdriver.ll_greenfields.values()`
                // — values are `(GTYPE, fieldname)` pairs.  Resolve `GTYPE`
                // by looking up the red slot's type from `red_types`
                // (parallel to `reds`); legacy callers without
                // `red_types` fall back to the variable name so the
                // structural shape is preserved.
                let gtype = self.jitdrivers_sd[jd_idx]
                    .red_types
                    .get(red_index)
                    .cloned()
                    .unwrap_or_else(|| objname.to_string());
                let green_fields: Vec<(String, String)> = parsed_pairs
                    .into_iter()
                    .map(|(_objname, fieldname)| (gtype.clone(), fieldname))
                    .collect();
                self.jitdrivers_sd[jd_idx].greenfield_info =
                    Some(std::sync::Arc::new(StaticGreenFieldInfoHandle {
                        red_index,
                        green_fields,
                    }));
            }
            // warmspot.py:527-530: no virtualizable → keep None and continue.
            if self.jitdrivers_sd[jd_idx].virtualizables.is_empty() {
                self.jitdrivers_sd[jd_idx].virtualizable_info = None;
                self.jitdrivers_sd[jd_idx].index_of_virtualizable = -1;
                continue;
            }
            // warmspot.py:531-532: greenfield + virtualizable not supported.
            assert!(
                self.jitdrivers_sd[jd_idx].greenfield_info.is_none(),
                "warmspot.py:532 — greenfield + virtualizable on the same driver: XXX not supported yet",
            );
            // warmspot.py:534-538 `[vname] = jitdriver.virtualizables`
            //                    `jd.index_of_virtualizable = jitdriver.reds.index(vname)`
            assert_eq!(
                self.jitdrivers_sd[jd_idx].virtualizables.len(),
                1,
                "warmspot.py:535 — only one virtualizable per jitdriver supported",
            );
            let vname = self.jitdrivers_sd[jd_idx].virtualizables[0].clone();
            let idx = self.jitdrivers_sd[jd_idx]
                .reds
                .iter()
                .position(|r| r == &vname)
                .unwrap_or_else(|| {
                    panic!(
                        "warmspot.py:538 — virtualizable {vname:?} not in reds {:?}",
                        self.jitdrivers_sd[jd_idx].reds
                    )
                });
            self.jitdrivers_sd[jd_idx].index_of_virtualizable = idx as i32;
            // warmspot.py:540-545:
            //   index = jd.num_green_args + jd.index_of_virtualizable
            //   VTYPEPTR = jd._JIT_ENTER_FUNCTYPE.ARGS[index]
            //   if VTYPEPTR not in vinfos:
            //       vinfos[VTYPEPTR] = VirtualizableInfo(self, VTYPEPTR)
            //   jd.virtualizable_info = vinfos[VTYPEPTR]
            //
            // Pyre resolves VTYPEPTR via `red_types[index_of_virtualizable]`
            // (the `_JIT_ENTER_FUNCTYPE.ARGS` analog supplied at
            // `setup_jitdriver` time) and delegates the constructor
            // call to `vinfo_factory`.
            let vtypeptr_token = self.jitdrivers_sd[jd_idx]
                .red_types
                .get(idx)
                .cloned()
                .unwrap_or_default();
            let info = if let Some(cached) = vinfos.get(&vtypeptr_token) {
                Some(cached.clone())
            } else if let Some(fresh) = vinfo_factory(jd_idx, &vtypeptr_token) {
                vinfos.insert(vtypeptr_token, fresh.clone());
                Some(fresh)
            } else {
                None
            };
            self.jitdrivers_sd[jd_idx].virtualizable_info = info;
        }
    }

    /// call.py:357-361 `jitdriver_sd_from_portal_graph(graph)`.
    pub fn jitdriver_sd_from_portal_graph(&self, path: &CallPath) -> Option<&JitDriverStaticData> {
        self.jitdrivers_sd
            .iter()
            .find(|sd| &sd.portal_graph == path)
    }

    /// call.py:363-367 `jitdriver_sd_from_portal_runner_ptr(funcptr)`.
    ///
    /// Pyre has no separate `portal_runner_ptr` (the runner is the
    /// portal graph itself), so we reuse the path lookup.  Future
    /// phases that need the distinction can split the field.
    pub fn jitdriver_sd_from_portal_runner_ptr(
        &self,
        path: &CallPath,
    ) -> Option<&JitDriverStaticData> {
        self.jitdriver_sd_from_portal_graph(path)
    }

    /// call.py:369-373 `jitdriver_sd_from_jitdriver(jitdriver)`.
    ///
    /// Pyre identifies a jit driver by its index slot in
    /// `jitdrivers_sd`; we expose the slot lookup under the upstream
    /// name so call sites mirror RPython.
    pub fn jitdriver_sd_from_jitdriver(&self, index: usize) -> Option<&JitDriverStaticData> {
        self.jitdrivers_sd.get(index)
    }

    /// call.py:375-385 `get_vinfo(VTYPEPTR)`.
    ///
    /// ```python
    /// def get_vinfo(self, VTYPEPTR):
    ///     seen = set()
    ///     for jd in self.jitdrivers_sd:
    ///         if jd.virtualizable_info is not None:
    ///             if jd.virtualizable_info.is_vtypeptr(VTYPEPTR):
    ///                 seen.add(jd.virtualizable_info)
    ///     if seen:
    ///         assert len(seen) == 1
    ///         return seen.pop()
    ///     else:
    ///         return None
    /// ```
    ///
    /// TODO: `VTYPEPTR` is an RPython lltype pointer;
    /// pyre represents VTYPEPTR identity as a `usize` token supplied by
    /// the host (typically `descr_identity(&size_descr)` from
    /// `majit_ir::descr`).  Hosts install per-driver
    /// [`VirtualizableInfoHandle`] via `JitDriverStaticData.virtualizable_info`.
    pub fn get_vinfo(
        &self,
        vtypeptr_id: usize,
    ) -> Option<std::sync::Arc<dyn VirtualizableInfoHandle>> {
        let mut seen: Vec<std::sync::Arc<dyn VirtualizableInfoHandle>> = Vec::new();
        for jd in &self.jitdrivers_sd {
            if let Some(vinfo) = &jd.virtualizable_info {
                if vinfo.is_vtypeptr(vtypeptr_id) {
                    // Dedupe by Arc identity so the upstream
                    // `assert len(seen) == 1` translates to "at most one
                    // distinct VirtualizableInfo per VTYPEPTR".
                    let seen_already = seen
                        .iter()
                        .any(|existing| std::sync::Arc::ptr_eq(existing, vinfo));
                    if !seen_already {
                        seen.push(std::sync::Arc::clone(vinfo));
                    }
                }
            }
        }
        if seen.is_empty() {
            None
        } else {
            assert_eq!(
                seen.len(),
                1,
                "get_vinfo: multiple distinct VirtualizableInfo for VTYPEPTR"
            );
            Some(seen.into_iter().next().unwrap())
        }
    }

    /// call.py:387-393 `could_be_green_field(GTYPE, fieldname)`.
    ///
    /// ```python
    /// def could_be_green_field(self, GTYPE, fieldname):
    ///     GTYPE_fieldname = (GTYPE, fieldname)
    ///     for jd in self.jitdrivers_sd:
    ///         if jd.greenfield_info is not None:
    ///             if GTYPE_fieldname in jd.greenfield_info.green_fields:
    ///                 return True
    ///     return False
    /// ```
    ///
    /// TODO: `GTYPE` is an RPython lltype; pyre
    /// represents it by name (`&str`).  The host attaches a
    /// [`GreenFieldInfoHandle`] whose `contains_green_field` implements
    /// the `(GTYPE, fieldname) in green_fields` membership test.
    pub fn could_be_green_field(&self, gtype: &str, fieldname: &str) -> bool {
        for jd in &self.jitdrivers_sd {
            if let Some(gfinfo) = &jd.greenfield_info {
                if gfinfo.contains_green_field(gtype, fieldname) {
                    return true;
                }
            }
        }
        false
    }

    /// Mark a target as a builtin (oopspec) operation.
    pub fn mark_builtin(&mut self, path: CallPath) {
        let id = self.intern_graph_id(&path);
        self.builtin_targets.insert(id);
    }

    /// Discover candidate graphs by BFS from portal targets.
    ///
    /// RPython: `CallControl.find_all_graphs(policy)` (call.py:49-92).
    ///
    /// Walks from portal graphs transitively: for each Call op,
    /// if the callee has a graph, add it to the candidate set.
    /// Portal must be seeded via `mark_portal()` before calling.
    /// call.py:49 `find_all_graphs(self, policy)`.
    ///
    /// Discovers all candidate graphs reachable from the portal entry
    /// points. RPython uses `policy.look_inside_graph` to decide whether
    /// to follow each callee; we synthesize a `SemanticFunction` from the
    /// callee graph's own `hints` and pass it through.
    pub fn find_all_graphs(&mut self, policy: &mut dyn JitPolicy) {
        assert!(
            !self.jitdrivers_sd.is_empty(),
            "find_all_graphs requires at least one portal target; \
             use find_all_graphs_for_tests() if no portal is available"
        );
        self.find_all_graphs_bfs(policy);
    }

    /// Test-only: include all registered function graphs as candidates.
    /// Production code must use `find_all_graphs()` with portal seeded.
    #[cfg(test)]
    pub fn find_all_graphs_for_tests(&mut self) {
        if self.jitdrivers_sd.is_empty() {
            let all_paths: Vec<CallPath> = self.function_graphs.keys().cloned().collect();
            for path in all_paths {
                self.candidate_graphs.insert(path);
            }
            return;
        }
        let mut policy = crate::policy::DefaultJitPolicy::new();
        self.find_all_graphs_bfs(&mut policy);
    }

    fn find_all_graphs_bfs(&mut self, policy: &mut dyn JitPolicy) {
        // RPython call.py:49-92: BFS from portal targets.
        // For each graph, scan all Call ops. If guess_call_kind would
        // return 'regular' (i.e. graphs_from returns a graph AND it's
        // a candidate), add the callee graph to candidates and continue.
        //
        // During BFS we use target_to_path + function_graphs directly
        // (not graphs_from, which checks candidate_graphs — the set
        // we're building).
        let mut todo: Vec<CallPath> = self
            .jitdrivers_sd
            .iter()
            .map(|jd| jd.portal_graph.clone())
            .collect();
        for path in &todo {
            self.candidate_graphs.insert(path.clone());
        }
        // call.py:59-64 — seed the BFS with builtin oopspec helpers so
        // `int_abs` / `int_floordiv` / `int_mod` / `ll_math.ll_math_sqrt`
        // are reachable even when the portal does not call them
        // directly.
        //
        // ```python
        // if hasattr(self, 'rtyper'):
        //     for oopspec_name, ll_args, ll_res in support.inline_calls_to:
        //         c_func, _ = support.builtin_func_for_spec(self.rtyper,
        //                                                   oopspec_name,
        //                                                   ll_args, ll_res)
        //         todo.append(c_func.value._obj.graph)
        // ```
        //
        // TODO: pyre's BFS seed is a partial port
        // of `call.py:59-64`, not a strict line-by-line mirror.
        // Upstream seeds all four `inline_calls_to` entries (`int_abs`,
        // `int_floordiv`, `int_mod`, `ll_math.ll_math_sqrt`) into
        // `todo`, materialising the helper graph each time via
        // `c_func.value._obj.graph` — the rtyper-bound translation
        // synthesises a Rust-source-equivalent graph from the Python
        // helper body, which `MixLevelHelperAnnotator` always
        // produces.  Pyre cannot perform that synthesis: a host-bound
        // `extern "C"` function pointer carries no body the walker can
        // read, and pyre has no `MixLevelHelperAnnotator` to fabricate
        // one.  Fabricating a one-op shim graph that calls the host
        // helper would be a deviation, not parity — the shim has
        // no `oopspec` / `_jit_*_` hints, no canraise / effect
        // analysis grounded in the helper body, and would override
        // any real source-level helper graph registered later under
        // the same canonical path.  The seed loop therefore:
        //   (a) skips `int_abs` outright — pyre has no production
        //       `_ll_1_int_abs` fnaddr binding and no helper body graph.
        //       RPython `inline_calls_to` seeds the helper graph so the
        //       JIT can look inside it (`support.py:443-449` /
        //       `call.py:59-64`); a fnaddr-only binding would instead
        //       make `int_abs` an opaque extern call.  The binding waits
        //       for the rtyper-equivalent to synthesise the body graph
        //       (the same gating that blocks `_ll_2_int_*` from being
        //       seeded).
        //   (b) skips `ll_math.ll_math_sqrt` — pyre has no
        //       `ll_math_sqrt` analogue that raises
        //       `ValueError("math domain error")` on negative input
        //       per `ll_math.py:317-322`, so making the fnaddr
        //       reachable would be a semantic regression (bare
        //       `f64::sqrt()` returns NaN where upstream raises).
        //   (c) registers the function pointer for the integer
        //       residual-call entries (`_ll_2_int_floordiv` /
        //       `_ll_2_int_mod` at `pyre/jit_fnaddr.rs`) so the
        //       jtransform-emitted residual calls resolve to a real
        //       C ABI address.  These do not push the impl path onto
        //       `todo` either — `function_graphs.contains_key` fails
        //       because no graph is registered for an `extern "C"`
        //       helper.
        // All three behaviours mirror upstream `@dont_look_inside`
        // for the SAME helper — the trace cannot inline through it
        // — but the pyre case is structurally broader: even helpers
        // upstream WOULD inline through stay opaque here.  Convergence
        // path: port the integer helpers as Rust-source bodies the
        // walker can lower into a graph, register the graph via
        // `register_function_graph(canonical_name)`, then the BFS
        // seed arm below will push the impl path naturally.  This
        // requires walker reach into majit-metainterp
        // and a Rust analogue of `MixLevelHelperAnnotator.constfunc`.
        // `ll_math_sqrt` additionally needs a pyre-side raise
        // protocol (PyError emission from a residual C call) before
        // the fnaddr can be safely bound.
        for (oopspec_name, ll_args, ll_res) in crate::support::INLINE_CALLS_TO {
            // `call.py:60-64`:
            //   c_func, _ = support.builtin_func_for_spec(self.rtyper,
            //                                             oopspec_name,
            //                                             ll_args, ll_res)
            //   todo.append(c_func.value._obj.graph)
            //
            // `extra` / `extrakey` are both None — the inline_calls_to
            // entries are simple helpers without the build-helper /
            // dict-iter side tables.  Upstream `c_func.value._obj.graph`
            // is the wrapper's helper graph (e.g. the graph for
            // `_ll_2_int_mod`), so the seed lookup must key on the
            // canonical impl name produced by
            // `setup_extra_builtin`, NOT on the oopspec name.
            //
            // Pre-check via `lookup_function_fnaddr` keeps the strict
            // panic inside `setup_extra_builtin` (mirroring
            // `support.py:687-690` raise-on-miss) from firing for
            // entries pyre's host has not bound — pyre's helpers
            // (`ll_math.ll_math_sqrt`) are not all registered as
            // concrete C ABI intrinsics, so the seed loop honestly
            // skips entries with no host binding rather than
            // crashing.  Entries that ARE bound flow through
            // `builtin_func_for_spec`, populating the rtyper-
            // equivalent cache, and contribute their canonical-impl
            // graph to the BFS seed when (and only when) a
            // Rust-source graph has been registered under that
            // canonical name.
            let canonical_path = CallPath::from_segments([format!(
                "_ll_{}_{}",
                ll_args.len(),
                oopspec_name.replace('.', "_"),
            )]);
            if self.lookup_function_fnaddr(&canonical_path).is_none() {
                continue;
            }
            let spec = crate::support::builtin_func_for_spec(
                Some(self),
                oopspec_name,
                ll_args,
                *ll_res,
                None,
                None,
            );
            let impl_path = CallPath::from_segments([spec.impl_name.as_str()]);
            if self.function_graphs.contains_key(&impl_path)
                && !self.candidate_graphs.contains(&impl_path)
            {
                self.candidate_graphs.insert(impl_path.clone());
                todo.push(impl_path);
            }
        }

        while let Some(path) = todo.pop() {
            let graph = match self.function_graphs.get(&path) {
                Some(g) => g.clone(),
                None => continue,
            };
            // RPython call.py:77-90: scan all Call ops in the graph.
            // For each call, check guess_call_kind (with BFS-aware
            // is_candidate that treats "has graph" as candidate).
            for block in &graph.blocks {
                for op in &block.operations {
                    let target = match &op.kind {
                        OpKind::Call { target, .. } => target,
                        _ => continue,
                    };
                    // `call.py:97` direct_call → `funcobj.graph` — co-fetch
                    // path + graph through the single Box-identity helper.
                    // A missing graph (unregistered target) and a missing
                    // path collapse into the same "skip" decision: the
                    // earlier path-then-Some(graph) two-step never built a
                    // SemanticFunction either when the graph wasn't in
                    // `function_graphs`.
                    let (callee_path, graph_ref) = match self.target_to_path_and_graph(target) {
                        Some(pair) => pair,
                        None => continue,
                    };
                    // RPython call.py:80: kind = self.guess_call_kind(op, is_candidate)
                    // Skip recursive (portal) and builtin calls — these are NOT
                    // followed during BFS. Only "regular" calls are followed.
                    if self
                        .jitdrivers_sd
                        .iter()
                        .any(|jd| jd.portal_graph == callee_path)
                    {
                        continue; // recursive — don't follow
                    }
                    if self
                        .graph_ids
                        .get(&callee_path)
                        .is_some_and(|id| self.builtin_targets.contains(id))
                    {
                        continue; // builtin — don't follow
                    }
                    if self.candidate_graphs.contains(&callee_path) {
                        continue; // already discovered
                    }
                    // RPython call.py:84,87: callee must satisfy
                    // policy.look_inside_graph(graph). Synthesize a
                    // SemanticFunction from the stored graph + hints so
                    // the policy's `_jit_*_` / `_elidable_function_`
                    // checks fire identically to upstream.
                    let hints = graph_ref.hints.clone();
                    let graph = graph_ref.clone();
                    let func = SemanticFunction {
                        name: callee_path.last_segment().unwrap_or_default().to_string(),
                        graph,
                        return_type: None,
                        self_ty_root: None,
                        hints,
                        module_path: String::new(),
                        access_directly: false,
                        trait_root: None,
                    };
                    if policy.look_inside_graph(&func) {
                        self.candidate_graphs.insert(callee_path.clone());
                        todo.push(callee_path);
                    }
                }
            }
        }
    }

    /// RPython: `CallControl.is_candidate(graph)`.
    /// Used only after `find_all_graphs()`.
    pub fn is_candidate(&self, path: &CallPath) -> bool {
        self.candidate_graphs.contains(path)
    }

    /// RPython: `CallControl.get_jitcode(graph, called_from)`.
    ///
    /// Retrieve or create the `Arc<JitCode>` shell for the given graph.
    /// The shell carries `name` plus the graph's bound helper address when
    /// available, otherwise the stable symbolic fallback; the body is
    /// filled later by `CodeWriter::transform_graph_to_jitcode` via
    /// `JitCode::set_body`. Upstream `jitcode.index` is not assigned here;
    /// pyre follows the same rule and sets it only when the finished
    /// jitcode is appended to `all_jitcodes[]`.
    ///
    /// RPython call.py:155-172: creates JitCode(graph.name, fnaddr, calldescr)
    /// and adds graph to unfinished_graphs for later assembly.
    pub fn get_jitcode(&mut self, path: &CallPath) -> std::sync::Arc<crate::jitcode::JitCode> {
        // RPython call.py:157-158: try: return self.jitcodes[graph]
        if let Some(arc) = self.jitcodes.get(path) {
            return arc.clone();
        }
        // RPython call.py:159-165: except KeyError:
        //   must never produce JitCode for close_stack.
        assert!(
            !self.graph_has_hint(path, "close_stack"),
            "{:?} has _gctransformer_hint_close_stack_",
            path
        );
        // Shell name mirrors RPython `graph.name`. We use the path's last
        // segment to stay readable in dumps; the assembler no longer
        // touches the name (it lives on the shell from allocation).
        let name = path
            .last_segment()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{path:?}"));
        let mut shell = crate::jitcode::JitCode::new(name);
        shell.fnaddr = self
            .function_fnaddrs
            .get(path)
            .copied()
            .unwrap_or_else(|| symbolic_fnaddr_for_path(path));
        let arc = std::sync::Arc::new(shell);
        self.jitcodes.insert(path.clone(), arc.clone());
        self.unfinished_graphs.push(path.clone());
        arc
    }

    /// Read-only handle lookup. Returns `None` for paths that have not
    /// been allocated by `get_jitcode` yet.
    pub fn jitcode_handle(
        &self,
        path: &CallPath,
    ) -> Option<std::sync::Arc<crate::jitcode::JitCode>> {
        self.jitcodes.get(path).cloned()
    }

    /// RPython `codewriter.py:81 all_jitcodes.append(jitcode)` — the
    /// sole append site in upstream's `make_jitcodes` loop. `jitcode.index`
    /// is already set by `transform_graph_to_jitcode` (upstream line 68);
    /// this method is the final positional append.
    pub fn finish_jitcode(&mut self, jitcode: std::sync::Arc<crate::jitcode::JitCode>) {
        debug_assert_eq!(
            jitcode.try_index(),
            Some(self.finished_jitcodes.len()),
            "finish_jitcode: jitcode {:?} arrives with index {:?} but \
             would land at slot {}. Upstream `codewriter.py:68` assigns \
             `jitcode.index = index` inside `transform_graph_to_jitcode`, \
             which must match `len(all_jitcodes)` at the call site.",
            jitcode.name,
            jitcode.try_index(),
            self.finished_jitcodes.len(),
        );
        self.finished_jitcodes.push(jitcode);
    }

    /// Read the number of jitcodes already appended to `all_jitcodes[]`.
    /// Drain loop callers use this to compute the `index` passed into
    /// `transform_graph_to_jitcode` (upstream `codewriter.py:80
    /// self.transform_graph_to_jitcode(graph, jitcode, verbose,
    /// len(all_jitcodes))`).
    pub fn finished_jitcodes_len(&self) -> usize {
        self.finished_jitcodes.len()
    }

    /// RPython `call.py:182-187 get_jitcode_calldescr` source-of-truth for
    /// `FUNC.RESULT`. Pyre derives the calldescr's result kind char from
    /// `graph.return_type` (stamped at registration off the parsed Rust
    /// signature, mirroring `funcptr._obj.TO.RESULT`). The mapping mirrors
    /// `front/ast.rs::type_string_to_value_type`. Returns `None` when the
    /// graph carries no return type — callers (`transform_graph_to_jitcode`)
    /// fall back to a CFG scan in that case (e.g. unit-test graphs without a
    /// parsed signature).
    pub fn declared_return_kind(&self, path: &CallPath) -> Option<char> {
        let s = self.function_graphs.get(path)?.return_type.as_ref()?.trim();
        Some(return_type_string_to_kind(s))
    }
}

/// Map a Rust return-type string to the BhCallDescr kind char used by
/// blackhole / metainterp. `None`/`""`/`"()"` → `'v'`. The integer/float
/// recognizer is the same set as `front/ast.rs::type_string_to_value_type`.
fn return_type_string_to_kind(s: &str) -> char {
    match s {
        "" | "()" => 'v',
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
        | "bool" | "char" | "Self::Truth" => 'i',
        "f32" | "f64" => 'f',
        _ => 'r',
    }
}

/// RPython parity for `call.py:220-221` `FUNC.ARGS` — collect the non-void
/// argument types of a graph.  Parameters live on `startblock.inputargs`
/// (RPython `flowspace/model.py` Block), populated by the front-end at
/// `front/ast.rs:706-769` parameter registration.  The `OpKind::Input`
/// ops co-emitted with each parameter carry the declared type which we
/// recover by chasing each inputarg Variable back to its defining op.
/// Unknown/ambiguous slots default to `Ref`, matching
/// `resolve_non_void_arg_types`' fallback.
///
/// TODO: when `inputargs` is empty we fall back to
/// scanning leading `OpKind::Input` ops in the startblock.  Unit tests
/// under `jit_codewriter::jtransform::tests` build graphs directly via
/// `FunctionGraph::new` + `push_op` without populating `inputargs`; the
/// fallback keeps their "all-Input-ops-are-params" convention working
/// until they are migrated.
fn graph_non_void_arg_types(graph: &FunctionGraph) -> Vec<Type> {
    let start = graph.block(graph.startblock);
    let map_ty = |ty: &crate::model::ValueType| match ty {
        // RPython `getkind(BOOL_TYPE)` returns `'int'`
        // (`lloperation.py:108`); BoolRepr's lowleveltype is `Bool`
        // and `FUNC.ARGS` (`call.py:220-221`) records it under the
        // same `'i'` register kind as `Signed`.  Bool aliases to Int
        // so the wildcard does not silently re-classify it as Ref.
        crate::model::ValueType::Int | crate::model::ValueType::Bool => Some(Type::Int),
        crate::model::ValueType::Ref(_) => Some(Type::Ref),
        crate::model::ValueType::Float => Some(Type::Float),
        crate::model::ValueType::Void => None,
        // Unknown / State — default to Ref.
        _ => Some(Type::Ref),
    };
    if !start.inputargs.is_empty() {
        // Walk `inputargs` (`Vec<Variable>`) directly — orthodox per
        // `flowspace/model.py:Block.inputargs`.  Treat unresolved
        // slots conservatively as `Type::Ref` (the same default the
        // wildcard arm uses below).
        return start
            .inputargs
            .iter()
            .filter_map(|arg| {
                let ty = start.operations.iter().find_map(|op| match &op.kind {
                    crate::model::OpKind::Input { ty, .. } if op.result.as_ref() == Some(arg) => {
                        Some(ty)
                    }
                    _ => None,
                });
                ty.map(map_ty).unwrap_or(Some(Type::Ref))
            })
            .collect();
    }
    start
        .operations
        .iter()
        .take_while(|op| matches!(op.kind, crate::model::OpKind::Input { .. }))
        .filter_map(|op| match &op.kind {
            crate::model::OpKind::Input { ty, .. } => map_ty(ty),
            _ => None,
        })
        .collect()
}

/// RPython parity for `call.py:222` `FUNC.RESULT`. Maps the declared
/// return type string (from `graph.return_type`) to `Type`; `None` or
/// unknown string → `Type::Void` (i.e. declared-void function). Matches
/// `type_string_to_value_type` in `front/ast.rs`.
fn return_type_string_to_value_type(s: Option<&String>) -> Type {
    match s.map(String::as_str) {
        None | Some("") | Some("()") => Type::Void,
        Some("i8") | Some("i16") | Some("i32") | Some("i64") | Some("isize") | Some("u8")
        | Some("u16") | Some("u32") | Some("u64") | Some("usize") | Some("bool") | Some("char")
        | Some("Self::Truth") => Type::Int,
        Some("f32") | Some("f64") => Type::Float,
        _ => Type::Ref,
    }
}

impl CallControl {
    /// Return the completed `all_jitcodes[]` list in append order. Every
    /// entry must have both a body and a dense final `.index`.
    pub fn collect_jitcodes_in_alloc_order(&self) -> Vec<std::sync::Arc<crate::jitcode::JitCode>> {
        for (i, jitcode) in self.finished_jitcodes.iter().enumerate() {
            assert!(
                jitcode.try_body().is_some(),
                "collect_jitcodes_in_alloc_order: jitcode {:?} at slot {i} has no body",
                jitcode.name
            );
            assert_eq!(
                jitcode.index(),
                i,
                "collect_jitcodes_in_alloc_order: jitcode {:?} has index {} at slot {i}",
                jitcode.name,
                jitcode.index()
            );
        }
        self.finished_jitcodes.clone()
    }

    /// RPython: `CallControl.grab_initial_jitcodes()` (call.py:145-148).
    ///
    /// ```python
    /// def grab_initial_jitcodes(self):
    ///     for jd in self.jitdrivers_sd:
    ///         jd.mainjitcode = self.get_jitcode(jd.portal_graph)
    ///         jd.mainjitcode.jitdriver_sd = jd
    /// ```
    ///
    /// Allocates `Arc<JitCode>` shells for portal graphs and stores them
    /// directly on each jitdriver. The `jitdriver_sd` back-reference is
    /// committed later by `CodeWriter::drain_pending_graphs` once the
    /// portal's body is assembled.
    pub fn grab_initial_jitcodes(&mut self) {
        // Collect portal paths first to avoid borrow conflict.
        let portals: Vec<(usize, CallPath)> = self
            .jitdrivers_sd
            .iter()
            .enumerate()
            .map(|(i, jd)| (i, jd.portal_graph.clone()))
            .collect();
        for (jd_index, portal) in portals {
            // RPython: jd.mainjitcode = self.get_jitcode(jd.portal_graph)
            let arc = self.get_jitcode(&portal);
            self.jitdrivers_sd[jd_index].mainjitcode = Some(arc);
        }
    }

    /// RPython: `CallControl.enum_pending_graphs()` (call.py:150-153).
    ///
    /// ```python
    /// def enum_pending_graphs(self):
    ///     while self.unfinished_graphs:
    ///         graph = self.unfinished_graphs.pop()  # LIFO
    ///         yield graph, self.jitcodes[graph]
    /// ```
    ///
    /// RPython uses a generator that pops one graph at a time (LIFO).
    /// During processing, new graphs may be added to `unfinished_graphs`
    /// via `get_jitcode()`, and the generator picks them up on the next
    /// iteration. We emulate this with `enum_pending_graphs()`.
    pub fn enum_pending_graphs(
        &mut self,
    ) -> Option<(CallPath, std::sync::Arc<crate::jitcode::JitCode>)> {
        let path = self.unfinished_graphs.pop()?; // LIFO, matching RPython
        let arc = self.jitcodes[&path].clone();
        Some((path, arc))
    }

    /// Classify a call.
    ///
    /// RPython `call.py:116-139 CallControl.guess_call_kind(op, is_candidate)`
    /// — line-by-line port.  The `op.opname == 'direct_call'` branch
    /// (call.py:117-136) maps to `OpKind::Call`; the implicit
    /// `indirect_call` branch (RPython falls through to the final
    /// `graphs_from(op) is None` test at line 137) maps to
    /// `OpKind::IndirectCall`.  Closestack / oopspec / recursive
    /// checks only apply to the direct branch because the corresponding
    /// flags are attached to a single `funcobj`; for indirect calls the
    /// same restrictions are enforced family-wide in `getcalldescr`
    /// (`call.py:259-280`).
    pub fn guess_call_kind(&self, op: &SpaceOperation) -> CallKind {
        if let OpKind::Call { target, .. } = &op.kind {
            // RPython `call.py:117-136` direct_call branch.
            let path = self.target_to_path(target);
            if let Some(ref p) = path {
                // call.py:119-120 jitdriver_sd_from_portal_runner_ptr(funcptr)
                if self.jitdrivers_sd.iter().any(|jd| &jd.portal_graph == p) {
                    return CallKind::Recursive;
                }
                let id = self.graph_ids.get(p);
                // call.py:129-134 _gctransformer_hint_close_stack_ → 'residual'
                if self.graph_has_hint(p, "close_stack") {
                    return CallKind::Residual;
                }
                // call.py:135-136 oopspec → 'builtin'
                if id.is_some_and(|id| self.builtin_targets.contains(id)) {
                    return CallKind::Builtin;
                }
            }
        }
        // RPython `call.py:137-139` — both direct_call (fall-through)
        // and indirect_call reach this final classification.
        if self.graphs_from(op).is_none() {
            CallKind::Residual
        } else {
            CallKind::Regular
        }
    }

    /// Collect every candidate callee graph reachable through this op.
    ///
    /// RPython `call.py:94-114 CallControl.graphs_from(op, is_candidate)`
    /// — line-by-line port.  The `op.opname == 'direct_call'` branch
    /// (call.py:97-101) returns `[graph]` for a direct call whose target
    /// is a candidate; the `op.opname == 'indirect_call'` branch
    /// (call.py:103-112) filters the family attached to the op by
    /// `is_candidate` and returns the non-empty subset.  Both branches
    /// collapse to `None` when no candidate is reachable — the residual
    /// call path.
    ///
    /// The `is_candidate` argument from RPython `call.py:94-96` is
    /// omitted here because majit's `find_all_graphs_for_tests` /
    /// `find_all_graphs` populates `self.candidate_graphs` in bulk
    /// before any caller invokes `graphs_from`; the RPython
    /// incremental-discovery shape (where `find_all_graphs` passes its
    /// own local `is_candidate`) is not needed.
    pub fn graphs_from(&self, op: &SpaceOperation) -> Option<Vec<CallPath>> {
        match &op.kind {
            OpKind::Call { target, .. } => {
                // call.py:97-101 direct_call branch.
                let path = self.target_to_path(target)?;
                if self.candidate_graphs.contains(&path) {
                    Some(vec![path])
                } else {
                    None
                }
            }
            OpKind::IndirectCall { graphs, .. } => {
                // call.py:103-112 indirect_call branch.
                // `graphs is None` (call.py:105) → residual.
                let graphs = graphs.as_ref()?;
                let result: Vec<CallPath> = graphs
                    .iter()
                    .filter(|p| self.candidate_graphs.contains(p))
                    .cloned()
                    .collect();
                if result.is_empty() {
                    None
                } else {
                    Some(result)
                }
            }
            _ => None,
        }
    }

    /// Look up the single callee `FunctionGraph` for a direct call.
    ///
    /// Convenience accessor for call sites (majit `inline.rs`) that
    /// still work with one `&FunctionGraph` at a time.  RPython
    /// `call.py:97-101` returns the graph value inside the list, but
    /// majit callers want a borrow against `function_graphs` so we keep
    /// this lookup separate from the op-based `graphs_from`.
    pub fn direct_graph_for(&self, target: &CallTarget) -> Option<&FunctionGraph> {
        let path = self.target_to_path(target)?;
        if !self.candidate_graphs.contains(&path) {
            return None;
        }
        match target {
            CallTarget::Method {
                name,
                receiver_root,
                resolved_path,
            } => self.function_graphs.get(&path).or_else(|| {
                self.resolve_method(name, receiver_root.as_deref(), resolved_path.as_ref())
            }),
            _ => self.function_graphs.get(&path),
        }
    }

    /// Look up the registered graph alongside its `CallPath` in a single
    /// step — `call.py:97` `funcobj.graph` direct read.  The returned
    /// `&FunctionGraph` is the same identity registered under the path,
    /// without the `candidate_graphs` filter `direct_graph_for` imposes
    /// (callers that want the candidate-only view continue to use
    /// `direct_graph_for`).  The `CallPath` byproduct stays available
    /// for the `function_fnaddrs` side-table still keyed by path string.
    pub(crate) fn target_to_path_and_graph(
        &self,
        target: &CallTarget,
    ) -> Option<(CallPath, &FunctionGraph)> {
        let path = self.target_to_path(target)?;
        let graph = self.function_graphs.get(&path)?;
        Some((path, graph))
    }

    /// Convert a CallTarget to a CallPath for lookup.
    ///
    /// FunctionPath → direct path.
    /// Method → qualified CallPath([impl_type, method_name]).
    ///
    /// RPython: graph identity is by object pointer, not name.
    /// We emulate this with qualified paths that include the impl type,
    /// so different impls of the same method get distinct paths.
    pub(crate) fn target_to_path(&self, target: &CallTarget) -> Option<CallPath> {
        match target {
            CallTarget::FunctionPath { segments } => {
                let path = CallPath::from_segments(segments.iter().map(String::as_str));
                if self.function_graphs.contains_key(&path) {
                    return Some(path);
                }
                // Cross-module reference fallback.  `canonical_call_target`
                // (`front/ast.rs::canonical_call_target`) now expands
                // single-ident callsites through the caller's
                // `use_imports` *first*, so `use foo::bar; bar();`
                // resolves verbatim to `["foo", "bar"]` against the
                // registry.  The remaining case the leaf-match needs to
                // cover is the bare callsite the caller's
                // `use_imports` could not resolve directly — typically
                // a same-file declaration whose registration spelling
                // diverges from the `module_prefix` qualification (e.g.
                // `lib.rs::register_function_graph_alias` chains).
                //
                // PyPy parity: `flowcontext.py:845-866 LOAD_GLOBAL`
                // applies to bare-name references; qualified
                // `module.func` reaches the bookkeeper through its
                // explicit attribute lookup, not the `f_globals`
                // fallback.  Mirror that by restricting leaf-match to
                // the cross-module bare-callsite shape — `segments`
                // produced by `canonical_call_target` is `["bare"]` or
                // `["caller_module", "bare"]` (length ≤ 2).  Three-
                // plus-segment paths (`["std", "ptr", "copy"]`,
                // `["pyre_interpreter", "module", "name"]`) come from
                // explicitly qualified callsites and must NOT fuzzy-
                // match — they either resolve verbatim or fall through
                // to `Residual` (external host call).  Without this
                // guard, `std::ptr::copy(s,d,n)` would leaf-match
                // unrelated 2-arg `copy` impl methods and corrupt
                // `getcalldescr`'s arity check.
                //
                // `PYRE_STRICT_TARGET_TO_PATH=1` (audit-only) disables
                // the leaf-match outright so a CI sweep can quantify
                // how often the cross-module fallback fires.  Production
                // runs leave the env var unset.
                if segments.len() > 2 {
                    return Some(path);
                }
                if std::env::var_os("PYRE_STRICT_TARGET_TO_PATH").is_some() {
                    return Some(path);
                }
                if let Some(leaf) = segments.last() {
                    // Restrict leaf-match to FREE-FUNCTION graphs
                    // (`FunctionGraph.owner_root.is_none()`).  Impl-method
                    // graphs registered via `lib.rs:747
                    // for_impl_method(impl_type, name)` share the same
                    // `function_graphs` HashMap but their leaf
                    // (`copy`/`new`/etc.) collides with arbitrary
                    // free-function names — without this filter, a
                    // bare callsite would match every same-leaf impl
                    // method.  RPython parity: `bookkeeper.getdesc(
                    // callable)` resolves free functions through
                    // `FunctionDesc` (host object id) and methods
                    // through the receiver's `ClassDesc`, never
                    // crossing the two namespaces.
                    //
                    // **Cross-module disambiguation gate** — the
                    // candidate must carry the caller-supplied
                    // `segments` as a SUFFIX of its own segments.
                    // Bare callsites (`segments = [name]`) match every
                    // free-fn whose leaf is `name`; module-qualified
                    // callsites (`segments = [caller_module, name]`)
                    // only match candidates whose tail is
                    // `[caller_module, name]`.  This stops a bare
                    // callsite in `runtime_ops.rs` (resolving to
                    // `["runtime_ops", "X"]`) from collapsing onto a
                    // cross-module `["call", "X"]` registration when
                    // both modules define a same-leaf free fn with
                    // different signatures.  Mirrors PyPy
                    // `bookkeeper.getdesc(callable)`'s per-function-
                    // object identity (`annrpython.py:103-150 build_types`)
                    // which never crosses the module boundary on a
                    // syntactically-qualified callsite.
                    //
                    // `free_fn_leaf_index` pre-filters by `(leaf,
                    // owner_root.is_none())` so this only walks paths
                    // that already cleared two of the three gates;
                    // the suffix check is the only per-candidate work.
                    let target_segs: &[String] = &segments[..];
                    let candidate_carries_target_as_suffix = |k: &CallPath| -> bool {
                        let cs = &k.segments;
                        cs.len() >= target_segs.len()
                            && cs[cs.len() - target_segs.len()..] == *target_segs
                    };
                    let empty: Vec<CallPath> = Vec::new();
                    let leaf_bucket = self.free_fn_leaf_index.get(leaf).unwrap_or(&empty);
                    let matches: Vec<&CallPath> = leaf_bucket
                        .iter()
                        .filter(|k| candidate_carries_target_as_suffix(k))
                        .collect();
                    if matches.len() == 1 {
                        return Some(matches[0].clone());
                    }
                    // Multi-match: pyre's free-function registration
                    // dual-publishes each graph under `[module, name]`,
                    // `["crate", module, name]`, and `[crate_alias,
                    // module, name]` aliases (`lib.rs:465-502
                    // register_function_graph_alias` chain), so a bare
                    // callsite (`use crate::X; X();`) producing
                    // `[caller_module, X]` will leaf-match every alias
                    // simultaneously even though all aliases point at
                    // copies of the same source graph.  Disambiguate by
                    // FunctionGraph.name (the qualified source name set
                    // by `lib.rs:1342 sf.name = format!("{prefix}::{name}")`,
                    // identical across alias clones).  PyPy parity:
                    // `bookkeeper.getdesc(callable)` keys on function-
                    // object identity, so multi-alias publications of
                    // the same desc converge on a single resolution.
                    if !matches.is_empty() {
                        let first_name = self
                            .function_graphs
                            .get(matches[0])
                            .map(|g| g.name.as_str());
                        if let Some(name) = first_name {
                            let all_same = matches.iter().all(|p| {
                                self.function_graphs
                                    .get(*p)
                                    .map(|g| g.name == name)
                                    .unwrap_or(false)
                            });
                            if all_same {
                                // Every match is an alias clone of one source
                                // graph (`g.name` identical).  Return a
                                // deterministic canonical alias — the
                                // lexicographically smallest segments — rather
                                // than the bucket's arbitrary first entry, so
                                // the resolved `CallPath` is stable across runs
                                // and consistently lines up with the
                                // path-keyed registries (`function_fnaddrs`,
                                // `return_types`, `builtin_targets`,
                                // `portal_targets`).
                                let canonical = matches
                                    .iter()
                                    .copied()
                                    .min_by(|a, b| a.segments.cmp(&b.segments))
                                    .unwrap();
                                return Some(canonical.clone());
                            }
                        }
                    }
                }
                Some(path)
            }
            CallTarget::Method {
                name,
                receiver_root,
                resolved_path,
            } => {
                // call.py:181 getfunctionptr(graph) — resolved_path
                // is the graph identity key stamped by the producer.
                if let Some(path) = resolved_path {
                    if self.function_graphs.contains_key(path) {
                        return Some(path.clone());
                    }
                }
                // `call.py:97` direct_call → `funcobj.graph` — inherent
                // method receivers carry a canonical `module::Type` spelling
                // (`parse::extract_inherent_impl_methods` registration and
                // `front::ast::qualify_type_name_with_imports` callsite both
                // route bare names through `STRUCT_ORIGIN_REGISTRY`, and
                // `joined_use_path` strips the syntactic `crate::` prefix
                // off `use_imports` entries), so the single qualified
                // lookup hits the same `CallPath` registered above.
                if let Some(receiver) = receiver_root.as_deref() {
                    let qualified = CallPath::for_impl_method(receiver, name.as_str());
                    if self.function_graphs.contains_key(&qualified) {
                        return Some(qualified);
                    }
                    // Suffix-match fallback for in-impl `self.method()` calls.
                    //
                    // When the parser walks `impl PyFrame { fn pop(&mut self) {
                    // self.stack_base() } }`, the inner `self.stack_base()` is
                    // recorded as `Method { receiver_root: Some("PyFrame") }`
                    // — the syntactic spelling, not the canonical
                    // `pyframe::PyFrame`.  `for_impl_method("PyFrame",
                    // "stack_base")` produces the 2-segment
                    // `["PyFrame", "stack_base"]`, but `function_graphs`
                    // registers the impl method under the 3-segment
                    // module-qualified key `["pyframe", "PyFrame",
                    // "stack_base"]`.  The literal lookup above misses, and
                    // without this fallback every in-impl `self.method()`
                    // call falls through to residual_call — inflating IR
                    // emission whenever the BFS would have inlined the
                    // method body otherwise.
                    //
                    // Look up `function_graphs` keys whose last 2 segments
                    // match `[receiver, name]` via `impl_method_leaf_index`.
                    // Accept the match only if it is unique: an ambiguous
                    // suffix (e.g. two crates both exposing a `PyFrame::pop`)
                    // falls through to the trait resolution path, which mirrors
                    // Rust's name-resolution ambiguity error rather than
                    // silently picking one.
                    //
                    // `PYRE_STRICT_TARGET_TO_PATH=1` disables the
                    // receiver-leaf fallback alongside the FunctionPath
                    // branch above so audit sweeps observe both fallback
                    // surfaces consistently.
                    if std::env::var_os("PYRE_STRICT_TARGET_TO_PATH").is_none() {
                        let need_tail = name.as_str().to_string();
                        let receiver_leaf =
                            receiver.rsplit("::").next().unwrap_or(receiver).to_string();
                        // `impl_method_leaf_index` pre-filters by
                        // method-name leaf so only same-leaf
                        // candidates are scanned for the
                        // `[..., receiver_leaf, need_tail]` tail.
                        let empty: Vec<CallPath> = Vec::new();
                        let leaf_bucket = self
                            .impl_method_leaf_index
                            .get(&need_tail)
                            .unwrap_or(&empty);
                        let mut matched: Option<&CallPath> = None;
                        let mut multi = false;
                        for key in leaf_bucket.iter() {
                            let segs = &key.segments;
                            if segs.len() >= 2
                                && segs
                                    .get(segs.len() - 2)
                                    .map(|s| s == &receiver_leaf)
                                    .unwrap_or(false)
                            {
                                if matched.is_some() {
                                    multi = true;
                                    break;
                                }
                                matched = Some(key);
                            }
                        }
                        if !multi {
                            if let Some(path) = matched {
                                return Some(path.clone());
                            }
                        }
                    }
                }
                // Fall back to trait method resolution for polymorphic calls.
                let impl_type = self.resolve_method_impl_type(name, receiver_root.as_deref())?;
                Some(CallPath::for_impl_method(impl_type, name.as_str()))
            }
            // RPython: an `indirect_call` is a *family* of graphs — there is
            // no single CallPath to resolve to.  Post-rtyper, indirect calls
            // live as `OpKind::IndirectCall` and `graphs_from(op)` returns
            // the candidate `Vec<CallPath>` directly; `target_to_path` only
            // names direct_call-equivalent sites.  `call.py:94-114` indirect
            // branch.
            CallTarget::SyntheticTransparentCtor { .. } | CallTarget::Indirect { .. } => None,
            CallTarget::UnsupportedExpr => None,
        }
    }

    /// RPython `call.py:181-183` uses `getfunctionptr(graph)` to obtain the
    /// integer funcptr identity for a call site. majit prefers a host-bound
    /// trace-call address when one has been registered for the resolved
    /// `CallPath`; otherwise it falls back to the stable symbolic address
    /// shim for source-only analysis.
    pub fn fnaddr_for_target(&self, target: &CallTarget) -> i64 {
        if let Some(path) = self.target_to_path(target) {
            return self
                .function_fnaddrs
                .get(&path)
                .copied()
                .unwrap_or_else(|| symbolic_fnaddr_for_path(&path));
        }
        symbolic_fnaddr_for_target(target)
    }

    /// Strict lookup variant of [`fnaddr_for_target`].
    ///
    /// Returns `Some(fnaddr)` only when the host has bound a real
    /// trace-call address through `register_function_fnaddr` (or one
    /// of its macro-fed entry points); `None` when the resolved
    /// `CallPath` has no registered entry, instead of synthesising a
    /// symbolic placeholder.
    ///
    /// Used by [`crate::jit_codewriter::support::builtin_func_for_spec`]
    /// to mirror RPython's `support.py:767-808` `(c_func, LIST_OR_DICT)`
    /// shape — upstream materialises the helper through
    /// `MixLevelHelperAnnotator.constfunc(impl, ...)`, pyre consults
    /// the persistent fnaddr cache populated from
    /// `pyre-interpreter::jit_trace_fnaddrs()` and surfaces a
    /// well-typed `None` when the helper has not been registered
    /// (callers can then either skip or fall back to the symbolic
    /// placeholder explicitly).
    pub fn lookup_function_fnaddr(&self, path: &CallPath) -> Option<i64> {
        self.function_fnaddrs.get(path).copied()
    }

    /// `support.py:771-774` cache read.
    ///
    /// ```python
    /// try:
    ///     return rtyper._builtin_func_for_spec_cache[key]
    /// except (KeyError, AttributeError):
    ///     pass
    /// ```
    ///
    /// Returns `Some(spec)` on a hit, `None` on a miss.  Pyre's cache
    /// is unconditionally initialised at `CallControl::new`, so the
    /// `AttributeError` branch upstream (cache field absent on the
    /// rtyper) collapses to a `None` return.
    pub fn lookup_builtin_func_for_spec_cache(
        &self,
        key: &crate::jit_codewriter::support::BuiltinFuncSpecCacheKey,
    ) -> Option<crate::jit_codewriter::support::BuiltinFuncSpec> {
        self.builtin_func_for_spec_cache.borrow().get(key).cloned()
    }

    /// `support.py:805-807` cache write.
    ///
    /// ```python
    /// if not hasattr(rtyper, '_builtin_func_for_spec_cache'):
    ///     rtyper._builtin_func_for_spec_cache = {}
    /// rtyper._builtin_func_for_spec_cache[key] = (c_func, LIST_OR_DICT)
    /// ```
    ///
    /// Pyre takes a `&self` reference because the cache lives behind a
    /// `RefCell` — matching upstream's read-only `rtyper` parameter
    /// shape lets `builtin_func_for_spec` retain a shared borrow
    /// throughout the call.
    pub fn cache_builtin_func_for_spec(
        &self,
        key: crate::jit_codewriter::support::BuiltinFuncSpecCacheKey,
        spec: crate::jit_codewriter::support::BuiltinFuncSpec,
    ) {
        self.builtin_func_for_spec_cache
            .borrow_mut()
            .insert(key, spec);
    }

    /// `support.py:466-468` `_ll_1_dict_keys.need_result_type = True`
    /// (and friends): host-side registration of the `need_result_type`
    /// attribute against a canonical helper name.  Pyre cannot reach
    /// into a function pointer; this registry is the structural
    /// equivalent.  Call this alongside `register_function_fnaddr` for
    /// any helper that upstream marks `need_result_type = True` /
    /// `'exact'`.
    pub fn register_need_result_type(
        &self,
        canonical_name: &str,
        ty: crate::jit_codewriter::support::NeedResultType,
    ) {
        self.need_result_type_registry
            .borrow_mut()
            .insert(canonical_name.to_string(), ty);
    }

    /// `support.py:782` `getattr(impl, 'need_result_type', False)`.
    ///
    /// Returns `Some(ty)` when the host registered the flag for the
    /// canonical name; `None` when it didn't (callers default to
    /// `NeedResultType::No`, mirroring the missing-attribute
    /// behaviour of upstream's `getattr(..., default=False)`).
    pub fn lookup_need_result_type(
        &self,
        canonical_name: &str,
    ) -> Option<crate::jit_codewriter::support::NeedResultType> {
        self.need_result_type_registry
            .borrow()
            .get(canonical_name)
            .copied()
    }

    /// `support.py:691-692 wrapper = wrapper(*extra)` factory registration.
    ///
    /// Hosts that expose a dict / array build helper register one
    /// fnaddr per `(canonical_name, extrakey)` pair before the
    /// codewriter pipeline starts.  `canonical_name` is the
    /// `build_ll_<n>_<oopspec>` form `setup_extra_builtin` renders;
    /// `extrakey` is the same string `builtin_func_for_spec`'s
    /// caller passes for cache discrimination.  Mirrors upstream's
    /// `LLtypeHelpers.build_ll_<n>_<oopspec>(*extra)` factory call
    /// semantics, with the host doing the specialisation ahead of
    /// time instead of at codewriter time.
    pub fn register_builtin_factory(&self, canonical_name: &str, extrakey: &str, fnaddr: i64) {
        self.builtin_factory_registry
            .borrow_mut()
            .insert((canonical_name.to_string(), extrakey.to_string()), fnaddr);
    }

    /// `support.py:691-692` factory lookup.
    ///
    /// `setup_extra_builtin` consults this when `extra.is_some()`;
    /// `None` falls back to the plain canonical-name fnaddr lookup
    /// (matching the "register the specialized fnaddr under the
    /// build-prefixed canonical name" pre-factory-registry workaround).
    pub fn lookup_builtin_factory(&self, canonical_name: &str, extrakey: &str) -> Option<i64> {
        self.builtin_factory_registry
            .borrow()
            .get(&(canonical_name.to_string(), extrakey.to_string()))
            .copied()
    }

    /// Resolve a method call to a concrete impl graph.
    ///
    /// Every successful resolution goes through
    /// [`Self::function_graphs`] via [`CallPath::for_impl_method`] —
    /// the same `getfunctionptr(graph)` identity surface upstream uses
    /// at `call.py:175-187`.
    pub fn resolve_method(
        &self,
        name: &str,
        receiver_root: Option<&str>,
        resolved_path: Option<&CallPath>,
    ) -> Option<&FunctionGraph> {
        if let Some(path) = resolved_path {
            if let Some(g) = self.function_graphs.get(path) {
                return Some(g);
            }
        }
        let impls = self.impls_for_method_name(name);
        if impls.is_empty() {
            return None;
        }

        // Receiver-string exact-match.
        if let Some(receiver) = receiver_root {
            let path = CallPath::for_impl_method(receiver, name);
            if let Some(g) = self.function_graphs.get(&path) {
                return Some(g);
            }
        }

        // TODO(parity): retire when annotator wiring publishes
        // classdef hints before BFS runs.
        // TODO: receiver-agnostic "unique concrete impl" fallback.
        // Upstream `call.py:175-187
        // getfunctionptr(graph)` keys on graph identity, never on
        // method name, so this branch has no RPython counterpart.
        // Pyre keeps it as a BFS-coverage adaptation: the codewriter
        // producer (`codewriter.rs::stamp_classdef_hints_on_graph`)
        // runs per-graph during transform, which is AFTER
        // `find_all_graphs_bfs` (`call.rs:2398`) has already chosen
        // candidates.  When BFS walks a Call site whose receiver is a
        // generic trait variable (`<H: OpcodeStepExecutor>`) with no
        // annotator-derived classdef hint and no receiver-name match,
        // collapsing to the unique concrete impl is the only way to
        // include the impl's body in `candidate_graphs`.  Retired once
        // the annotator publishes classdef hints
        // before BFS runs; the
        // `find_all_graphs_closure_reaches_handler_graphs_from_dispatch_portal`
        // and `all_jitcodes_registry_contains_inherent_impl_methods`
        // oracles pin the BFS coverage this branch enables.
        let concrete_impls: Vec<&String> = impls
            .iter()
            .copied()
            .filter(|t| !t.starts_with("<default methods of"))
            .collect();
        if concrete_impls.len() == 1 {
            let impl_type = concrete_impls[0];
            let path = CallPath::for_impl_method(impl_type, name);
            return self.function_graphs.get(&path);
        }

        // Single registered "default methods" shim — uniqueness here
        // does not depend on the receiver because the trait default
        // body is the same for every impl that does not override it
        // (`classdesc.py:749 lookup` MRO walk).  Match only when the
        // sole registration is a default-method shim AND no
        // overriding concrete impl exists.
        if concrete_impls.is_empty() && impls.len() == 1 {
            let impl_type = impls[0];
            let path = CallPath::for_impl_method(impl_type, name);
            return self.function_graphs.get(&path);
        }

        None
    }

    /// Like `resolve_method`, but returns the impl type name.
    fn resolve_method_impl_type<'b>(
        &'b self,
        name: &str,
        receiver_root: Option<&str>,
    ) -> Option<&'b str> {
        let impls = self.impls_for_method_name(name);
        if impls.is_empty() {
            return None;
        }

        // Receiver-string exact-match.  Mirrors [`Self::resolve_method`].
        if let Some(receiver) = receiver_root {
            if let Some(impl_name) = impls.iter().copied().find(|t| t.as_str() == receiver) {
                return Some(impl_name.as_str());
            }
        }

        // TODO: receiver-agnostic "unique concrete impl" fallback — see
        // the matching branch in [`Self::resolve_method`] for the
        // rationale.  Retired
        // alongside it once the annotator publishes classdef hints
        // before BFS.
        let concrete_impls: Vec<&String> = impls
            .iter()
            .copied()
            .filter(|t| !t.starts_with("<default methods of"))
            .collect();
        if concrete_impls.len() == 1 {
            return Some(concrete_impls[0].as_str());
        }

        // Lone default-method shim — uniqueness here is a property of
        // the trait, not the receiver (every impl shares the default
        // body per `classdesc.py:749 lookup` MRO walk).
        if concrete_impls.is_empty() && impls.len() == 1 {
            return Some(impls[0].as_str());
        }
        None
    }

    /// Collect every registered impl type name for `method_name`, across
    /// all declaring traits.  Used by `resolve_method` /
    /// `resolve_method_impl_type` for concrete-receiver method calls
    /// (RPython's `funcobj.graph` resolution).  Indirect-call family
    /// lookup uses the exact `(trait_root, method_name)` key via
    /// `all_impls_for_indirect` instead.
    fn impls_for_method_name<'b>(&'b self, method_name: &str) -> Vec<&'b String> {
        self.method_to_impl_types
            .get(method_name)
            .into_iter()
            .flatten()
            .collect()
    }

    /// Collect every registered impl `CallPath` for a
    /// `(trait_root, method_name)` family, regardless of whether each
    /// one is a regular candidate.  Used by family-wide validation
    /// where the goal is to reject mixed `_elidable_function_` etc.
    /// even among residual members (`call.py:259-280`).
    pub fn all_impls_for_indirect(&self, trait_root: &str, method_name: &str) -> Vec<CallPath> {
        self.trait_method_impls
            .get(&(trait_root.to_string(), method_name.to_string()))
            .into_iter()
            .flatten()
            .map(|impl_type| CallPath::for_impl_method(impl_type.as_str(), method_name))
            .collect()
    }

    /// RPython `call.py:259-280` — family-wide validation for indirect_call.
    ///
    /// Rejects a family if any member is marked `_elidable_function_` /
    /// `_jit_loop_invariant_` / `_call_aroundstate_target_`: indirect
    /// dispatch cannot preserve the semantics those flags require, so
    /// upstream raises an Exception at getcalldescr time.  Returns a
    /// formatted error message on the first mismatch.
    pub fn check_indirect_call_family(&self, candidates: &[CallPath]) -> Result<(), String> {
        for graph in candidates {
            let err = if self.graph_has_hint(graph, "elidable") {
                Some("@jit.elidable")
            } else if self.graph_has_hint(graph, "loopinvariant") {
                Some("@jit.loop_invariant")
            } else if self.graph_has_hint(graph, "aroundstate") {
                Some("_call_aroundstate_target_")
            } else {
                None
            };
            if let Some(flag) = err {
                return Err(format!(
                    "indirect_call family includes {graph:?} marked {flag}; \
                     every candidate in an indirect family must share the \
                     same jit attribute"
                ));
            }
        }
        Ok(())
    }

    /// Access the function graphs map (for inline pass).
    pub fn function_graphs(&self) -> &HashMap<CallPath, FunctionGraph> {
        &self.function_graphs
    }

    /// Returns true when a concrete helper address was registered for this
    /// path. Such a path is a real callable surface and must not be treated
    /// as a transparent Rust enum constructor by jtransform.
    pub fn has_function_fnaddr(&self, path: &CallPath) -> bool {
        self.function_fnaddrs.contains_key(path)
    }

    /// Access the `{CallPath → Arc<JitCode>}` map.
    ///
    /// RPython: `call.py:87 self.jitcodes`. Pyre exposes the same map as a
    /// read-only view so `CodeWriter::make_jitcodes` can pair it with
    /// `collect_jitcodes_in_alloc_order` into a single `AllJitCodes`
    /// return value.
    pub fn jitcodes(
        &self,
    ) -> &indexmap::IndexMap<CallPath, std::sync::Arc<crate::jitcode::JitCode>> {
        &self.jitcodes
    }

    /// Access jitdriver static data.
    pub fn jitdrivers_sd(&self) -> &[JitDriverStaticData] {
        &self.jitdrivers_sd
    }

    // ── Per-graph JIT hint carrier ───────────────────────────────────
    //
    // `graph.hints` is the single home for the source `#[jit_*]`
    // attributes (`_elidable_function_`, `_jit_loop_invariant_`,
    // `_gctransformer_hint_close_stack_`, …).  It is seeded at
    // registration (`register_function_graph_with_hints` /
    // `register_function_hints_for`) from the parsed attribute set and
    // read back here, matching `getattr(funcobj, "_elidable_function_",
    // False)` off the funcobj.

    /// Test whether the registered graph for `path` carries the hint `tok`.
    fn graph_has_hint(&self, path: &CallPath, tok: &str) -> bool {
        self.function_graphs
            .get(path)
            .is_some_and(|g| g.hints.iter().any(|h| h == tok))
    }

    /// Ensure the hint `tok` is present on the registered graph for `path`
    /// (idempotent; no-op when no graph is registered under `path`).
    fn stamp_graph_hint(&mut self, path: &CallPath, tok: &str) {
        if let Some(g) = self.function_graphs.get_mut(path)
            && !g.hints.iter().any(|h| h == tok)
        {
            g.hints.push(tok.to_string());
        }
    }

    // ── Elidable / loop-invariant registration ──────────────────────

    /// RPython: `getattr(func, "_elidable_function_", False)` (call.py:239).
    /// Mark a target as elidable (pure function).
    pub fn mark_elidable(&mut self, path: CallPath) {
        self.stamp_graph_hint(&path, "elidable");
    }

    /// RPython: `getattr(func, "_jit_loop_invariant_", False)` (call.py:240).
    /// Mark a target as loop-invariant.
    pub fn mark_loopinvariant(&mut self, path: CallPath) {
        self.stamp_graph_hint(&path, "loopinvariant");
    }

    /// RPython: call.py:239 — check if target has `_elidable_function_`.
    pub fn is_elidable(&self, target: &CallTarget) -> bool {
        self.target_to_path(target)
            .is_some_and(|p| self.graph_has_hint(&p, "elidable"))
    }

    /// Pyre extension: register a target as carrying the
    /// `#[elidable_cannot_raise]` user assertion.
    pub fn mark_cannot_raise_assertion(&mut self, path: CallPath) {
        let id = self.intern_graph_id(&path);
        assert!(
            !self.memerror_only_assertion_targets.contains(&id),
            "conflicting elidable exception assertions for {path:?}: \
             already marked memerror-only, cannot also mark cannot-raise"
        );
        self.cannot_raise_assertion_targets.insert(id);
    }

    /// Pyre extension: check if `target` carries the
    /// `#[elidable_cannot_raise]` assertion.
    ///
    /// Additionally requires the target's fnaddr to be registered via
    /// `register_function_fnaddr`.  Without a real fnaddr,
    /// `fnaddr_for_target` returns a synthetic 64-bit hash via
    /// [`symbolic_fnaddr_for_path`]; that hash lands in the sub-jitcode
    /// `constants_i` slot for the funcbox.  When the walker observes
    /// `EF_ELIDABLE_CANNOT_RAISE` on the descr it routes through
    /// `try_fold_pure_call_via_executor`
    /// (pyre-jit-trace/src/jitcode_dispatch.rs:3099) which dereferences
    /// the constant_i as a C function pointer — SIGSEGV on the hash.
    /// Pyre's symbolic placeholder is a deviation vs RPython (where
    /// every callee has a real `MixLevelHelperAnnotator.constfunc(impl)`
    /// address); the gate restores the upstream-equivalent invariant
    /// that `EF_ELIDABLE_CANNOT_RAISE` callees are always executable.
    pub fn has_cannot_raise_assertion(&self, target: &CallTarget) -> bool {
        self.target_to_path(target).is_some_and(|p| {
            self.graph_ids
                .get(&p)
                .is_some_and(|id| self.cannot_raise_assertion_targets.contains(id))
                && self
                    .function_fnaddrs
                    .get(&p)
                    .is_some_and(|&fnaddr| fnaddr != 0)
        })
    }

    /// Pyre extension: register a target as carrying the
    /// `#[elidable_or_memerror]` user assertion.
    pub fn mark_memerror_only_assertion(&mut self, path: CallPath) {
        let id = self.intern_graph_id(&path);
        assert!(
            !self.cannot_raise_assertion_targets.contains(&id),
            "conflicting elidable exception assertions for {path:?}: \
             already marked cannot-raise, cannot also mark memerror-only"
        );
        self.memerror_only_assertion_targets.insert(id);
    }

    /// Pyre extension: check if `target` carries the
    /// `#[elidable_or_memerror]` assertion.
    ///
    /// Same fnaddr-registration gate as
    /// [`Self::has_cannot_raise_assertion`]: `EF_ELIDABLE_OR_MEMORYERROR`
    /// walker arms also route through the executor fold path when the
    /// caller has no MemoryError stamping, so a symbolic placeholder
    /// would crash there too.
    pub fn has_memerror_only_assertion(&self, target: &CallTarget) -> bool {
        self.target_to_path(target).is_some_and(|p| {
            self.graph_ids
                .get(&p)
                .is_some_and(|id| self.memerror_only_assertion_targets.contains(id))
                && self
                    .function_fnaddrs
                    .get(&p)
                    .is_some_and(|&fnaddr| fnaddr != 0)
        })
    }

    /// RPython: call.py:240 — check if target has `_jit_loop_invariant_`.
    pub fn is_loopinvariant(&self, target: &CallTarget) -> bool {
        self.target_to_path(target)
            .is_some_and(|p| self.graph_has_hint(&p, "loopinvariant"))
    }

    /// RPython: call.py:129-134 — `_gctransformer_hint_close_stack_`.
    /// Mark a target as close_stack (must never produce JitCode).
    pub fn mark_close_stack(&mut self, path: CallPath) {
        self.stamp_graph_hint(&path, "close_stack");
    }

    /// RPython: collectanalyze.py:21 — `funcobj.random_effects_on_gcobjs`.
    /// Mark an external target as having random GC effects.
    pub fn mark_external_gc_effects(&mut self, path: CallPath) {
        let id = self.intern_graph_id(&path);
        self.external_gc_effects.insert(id);
    }

    /// RPython: collectanalyze.py:15 — `_gctransformer_hint_cannot_collect_`.
    /// Mark a target as known not to trigger GC collection.
    pub fn mark_cannot_collect(&mut self, path: CallPath) {
        let id = self.intern_graph_id(&path);
        self.cannot_collect_targets.insert(id);
    }

    /// RPython: rlib/jit.py:250 `@oopspec(spec)` — store `func.oopspec = spec`.
    /// Mark a target as having an oopspec string for jtransform lowering.
    pub fn mark_oopspec(&mut self, path: CallPath, spec: String) {
        let id = self.intern_graph_id(&path);
        self.oopspec_targets.insert(id, spec);
        // call.py:135-136 `if hasattr(targetgraph.func, 'oopspec'): return
        // 'builtin'` — oopspec presence is the builtin signal. A builtin
        // call is classified `Builtin` by `guess_call_kind` and is NOT
        // followed during the candidate-graph BFS (only seeded explicitly
        // via `INLINE_CALLS_TO`). Tie the two here so every production
        // oopspec registration drives that classification, not just the
        // explicit `mark_builtin` path.
        self.builtin_targets.insert(id);
    }

    /// RPython: `getattr(func, 'oopspec', None)` — look up oopspec for a target.
    pub fn get_oopspec(&self, target: &CallTarget) -> Option<&str> {
        self.graph_id_of(target)
            .and_then(|id| self.oopspec_targets.get(&id).map(|s| s.as_str()))
    }

    /// `support.py:705 argnames = ll_func.__code__.co_varnames[:nb_args]` —
    /// register the positional parameter names of an oopspec target so
    /// `parse_oopspec` can resolve identifier slots in the spec's
    /// `(...)` pattern to `Index(n)` placeholders.  The list must
    /// match the function's actual parameter declaration order.
    ///
    /// Populated by the walker (`lib.rs:600`) whenever
    /// `front::syn_metadata::collect_jit_hints` emits the
    /// `"oopspec_argnames:..."` companion hint — i.e. when a function
    /// carries `#[oopspec(...)]` AND its signature is available at
    /// hint-collection time.  Programmatic `mark_oopspec` callers
    /// (`lib.rs:707-741` jit.* bindings) leave this unset because
    /// their bare-name specs have no `(...)` pattern to resolve.
    pub fn mark_oopspec_argnames(&mut self, path: CallPath, argnames: Vec<String>) {
        let id = self.intern_graph_id(&path);
        self.oopspec_argnames.insert(id, argnames);
    }

    /// Per-target argname lookup paired with `get_oopspec`.  Returns
    /// `None` when the target has no registered argname list
    /// (the dominant case today).
    pub fn get_oopspec_argnames(&self, target: &CallTarget) -> Option<&[String]> {
        self.graph_id_of(target)
            .and_then(|id| self.oopspec_argnames.get(&id).map(|v| v.as_slice()))
    }

    // ── Graph-based analyzers (call.py:282-303) ─────────────────────
    //
    // The five `analyze_*` methods below walk
    // `crate::model::FunctionGraph` (the flat codewriter graph), inlining
    // the generic `GraphAnalyzer.analyze_direct_call` traversal
    // (`graphanalyze.py:139-177`) into each per-analysis body with a
    // bottom-on-cycle `seen` guard. The orthodox versions are
    // ported over the flowspace graph model: `RaiseAnalyzer`
    // (`backendopt/canraise.rs`), `CollectAnalyzer`
    // (`backendopt/collectanalyze.rs`), and the shared `GraphAnalyzer`
    // framework (`backendopt/graphanalyze.rs`, with SCC-merge cycle
    // handling via `DependencyTracker`/`UnionFind`). These duplicates
    // exist only because `CallControl` operates on the flat graph, not on
    // flowspace graphs. Do NOT extract a
    // shared skeleton here — that would add a third analysis framework
    // duplicating `graphanalyze.rs` over the wrong graph model.

    /// RPython: RaiseAnalyzer.analyze() — transitive can-raise analysis.
    ///
    /// canraise.py:8-24: RaiseAnalyzer(BoolGraphAnalyzer)
    /// - `analyze_simple_operation`: checks `LL_OPERATIONS[op.opname].canraise`
    /// - `analyze_external_call`: `getattr(fnobj, 'canraise', True)`
    /// - `analyze_exceptblock_in_graph`: checks except blocks
    ///
    /// Shared implementation for the two upstream RaiseAnalyzer instances:
    /// normal mode and `do_ignore_memory_error()` mode.
    fn analyze_can_raise_impl(
        &self,
        path: &CallPath,
        seen: &mut HashSet<CallPath>,
        ignore_memoryerror: bool,
    ) -> bool {
        if !seen.insert(path.clone()) {
            return false; // cycle → bottom_result
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            // RPython: analyze_external_call → getattr(fnobj, 'canraise', True)
            None => return true,
        };
        for block in &graph.blocks {
            // RPython: analyze_simple_operation(op) per operation.
            // canraise.py:14-17: LL_OPERATIONS[op.opname].canraise
            for op in &block.operations {
                let op_result = match &op.kind {
                    OpKind::Call { target, .. } => {
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            None => return true, // unresolvable → conservative
                        };
                        self.analyze_can_raise_impl(&callee_path, seen, ignore_memoryerror)
                    }
                    OpKind::IndirectCall { graphs, .. } => match graphs.as_deref() {
                        None => true, // graphanalyze.py:117-121 → top_result()
                        Some(graphs) => {
                            for callee_path in graphs {
                                if self.analyze_can_raise_impl(
                                    callee_path,
                                    seen,
                                    ignore_memoryerror,
                                ) {
                                    return true;
                                }
                            }
                            false
                        }
                    },
                    other => raise_class_can_raise(op_can_raise(other), ignore_memoryerror),
                };
                if op_result {
                    return true;
                }
            }
        }
        // RPython `backendopt/canraise.py:27-41 analyze_exceptblock_in_graph`
        // only applies the re-raise suppression in the ignore-MemoryError
        // analyzer. The normal analyzer always treats exceptblock exits as
        // raising.
        if graph
            .blocks
            .iter()
            .flat_map(|block| block.exits.iter())
            .any(|link| link.target == graph.exceptblock)
        {
            if ignore_memoryerror && exceptblock_is_reraise_of_caught_exception(graph) {
                return false;
            }
            return true;
        }
        false
    }

    /// RPython: VirtualizableAnalyzer.analyze() (effectinfo.py:401-404).
    ///
    /// analyze_simple_operation: op.opname in ('jit_force_virtualizable',
    ///                                         'jit_force_virtual')
    fn analyze_forces_virtualizable(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false;
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            // RPython: external call → analyze_external_call → bottom_result (False).
            // VirtualizableAnalyzer does not override analyze_external_call.
            None => return false,
        };
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    // RPython: jit_force_virtualizable / jit_force_virtual
                    OpKind::VableForce { .. } => return true,
                    OpKind::Call { target, .. } => {
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            None => continue, // external call → False
                        };
                        if self.analyze_forces_virtualizable(&callee_path, seen) {
                            return true;
                        }
                    }
                    OpKind::IndirectCall { graphs, .. } => match graphs.as_deref() {
                        None => return true, // BoolGraphAnalyzer.top_result()
                        Some(graphs) => {
                            for callee_path in graphs {
                                if self.analyze_forces_virtualizable(callee_path, seen) {
                                    return true;
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        false
    }

    /// RPython: RandomEffectsAnalyzer.analyze() (effectinfo.py:410-418).
    ///
    /// ```python
    /// class RandomEffectsAnalyzer(BoolGraphAnalyzer):
    ///     def analyze_external_call(self, funcobj, seen=None):
    ///         if funcobj.random_effects_on_gcobjs:
    ///             return True
    ///         return super().analyze_external_call(funcobj, seen)
    ///     def analyze_simple_operation(self, op, graphinfo):
    ///         return False
    /// ```
    ///
    /// Key: `analyze_simple_operation` always returns False. External calls
    /// only return True if `random_effects_on_gcobjs` is set. The default
    /// `analyze_external_call` returns `bottom_result()` = False
    /// (graphanalyze.py:60-69). "No graph" ≠ random effects in RPython.
    ///
    /// In majit: functions without graphs are external calls — returns
    /// True if in `external_gc_effects`, False otherwise.
    fn analyze_random_effects(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false; // cycle → bottom_result
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            None => {
                // RPython: analyze_external_call → bottom_result (False)
                // unless funcobj.random_effects_on_gcobjs → True.
                return self
                    .graph_ids
                    .get(path)
                    .is_some_and(|id| self.external_gc_effects.contains(id));
            }
        };
        // RPython: analyze_simple_operation always returns False.
        // Only recursive calls into graphs can propagate random effects.
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::Call { target, .. } => {
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            // Unresolvable target = external call → False
                            None => continue,
                        };
                        if self.analyze_random_effects(&callee_path, seen) {
                            return true;
                        }
                    }
                    OpKind::IndirectCall { graphs, .. } => match graphs.as_deref() {
                        None => return true, // BoolGraphAnalyzer.top_result()
                        Some(graphs) => {
                            for callee_path in graphs {
                                if self.analyze_random_effects(callee_path, seen) {
                                    return true;
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        false
    }

    /// RPython: QuasiImmutAnalyzer.analyze() (effectinfo.py).
    ///
    /// analyze_simple_operation: op.opname == 'jit_force_quasi_immutable'.
    ///
    /// In majit: we don't have quasi-immutable ops in the model yet,
    /// so this always returns false. The transitive call check is still
    /// performed for future-proofing.
    fn analyze_can_invalidate(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false;
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            None => return false, // no graph → cannot invalidate (not conservative here)
        };
        for block in &graph.blocks {
            for op in &block.operations {
                // RPython: jit_force_quasi_immutable → true
                // majit: no such op yet, but check calls transitively
                match &op.kind {
                    OpKind::Call { target, .. } => {
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            None => continue,
                        };
                        if self.analyze_can_invalidate(&callee_path, seen) {
                            return true;
                        }
                    }
                    OpKind::IndirectCall { graphs, .. } => match graphs.as_deref() {
                        None => return true, // BoolGraphAnalyzer.top_result()
                        Some(graphs) => {
                            for callee_path in graphs {
                                if self.analyze_can_invalidate(callee_path, seen) {
                                    return true;
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        false
    }

    /// RPython: CollectAnalyzer (collectanalyze.py).
    ///
    /// RPython: CollectAnalyzer.analyze_direct_call(graph, seen)
    /// (collectanalyze.py + graphanalyze.py:139).
    ///
    /// Traverses graph ops with:
    /// - analyze_simple_operation (collectanalyze.py:27-33): checks malloc/
    ///   malloc_varsize with GC flavor, LL_OPERATIONS[op].canmallocgc.
    ///   In majit the codewriter graph has no LL_OPERATIONS; allocations are
    ///   only reachable transitively through calls.
    /// - analyze_direct_call: recurse into callee graphs.
    /// - analyze_external_call (graphanalyze.py:60): bottom_result() (False).
    /// - _gctransformer_hint_cannot_collect_ (collectanalyze.py:15-16):
    ///   functions in `cannot_collect_targets` are known not to collect.
    fn analyze_can_collect(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false;
        }
        let id = self.graph_ids.get(path);
        // collectanalyze.py:15: _gctransformer_hint_cannot_collect_ → False
        if id.is_some_and(|id| self.cannot_collect_targets.contains(id)) {
            return false;
        }
        // collectanalyze.py:15: _gctransformer_hint_close_stack_ → True.
        // close_stack functions always can collect.
        if self.graph_has_hint(path, "close_stack") {
            return true;
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            None => {
                // collectanalyze.py:21-25: analyze_external_call —
                // if funcobj.random_effects_on_gcobjs → True,
                // else → bottom_result() (False).
                return id.is_some_and(|id| self.external_gc_effects.contains(id));
            }
        };
        for block in &graph.blocks {
            for op in &block.operations {
                // collectanalyze.py:27-33: analyze_simple_operation
                // RPython checks: malloc/malloc_varsize with flavor='gc' → True
                //                 LL_OPERATIONS[op.opname].canmallocgc → True
                // majit codewriter graphs have no LL_OPERATIONS; the only
                // operations that can trigger GC are transitive through calls.
                // (All other OpKind variants are pure/field/array ops.)
                match &op.kind {
                    OpKind::Call { target, .. } => {
                        // graphanalyze.py:139-164: analyze_direct_call — recurse
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            // graphanalyze.py:60: external call → bottom_result (False)
                            None => continue,
                        };
                        if self.analyze_can_collect(&callee_path, seen) {
                            return true;
                        }
                    }
                    OpKind::IndirectCall { graphs, .. } => match graphs.as_deref() {
                        None => return true, // BoolGraphAnalyzer.top_result()
                        Some(graphs) => {
                            for callee_path in graphs {
                                if self.analyze_can_collect(callee_path, seen) {
                                    return true;
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
        }
        false
    }

    // ── Cached analyzer wrappers ────────────────────────────────────

    /// Cached version of _canraise for a CallTarget.
    ///
    /// RPython call.py:337-355 — `_canraise()` returns the tri-state
    /// `{False, "mem", True}` collapsed here to [`CanRaise`].
    fn cached_can_raise_path(&self, path: &CallPath, cache: &mut AnalysisCache) -> CanRaise {
        if let Some(&result) = cache.can_raise.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = if !self.analyze_can_raise_impl(path, &mut seen, false) {
            CanRaise::No
        } else {
            let mut seen_ignore_memoryerror = HashSet::new();
            if self.analyze_can_raise_impl(path, &mut seen_ignore_memoryerror, true) {
                CanRaise::Yes
            } else {
                CanRaise::MemoryErrorOnly
            }
        };
        cache.can_raise.insert(path.clone(), result);
        result
    }

    fn cached_can_raise(&self, target: &CallTarget, cache: &mut AnalysisCache) -> CanRaise {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return CanRaise::Yes,
        };
        self.cached_can_raise_path(&path, cache)
    }

    fn cached_can_raise_family(
        &self,
        graphs: Option<&[CallPath]>,
        cache: &mut AnalysisCache,
    ) -> CanRaise {
        let graphs = match graphs {
            Some(graphs) => graphs,
            None => return CanRaise::Yes,
        };
        let mut result = CanRaise::No;
        for path in graphs {
            match self.cached_can_raise_path(path, cache) {
                CanRaise::Yes => return CanRaise::Yes,
                CanRaise::MemoryErrorOnly => result = CanRaise::MemoryErrorOnly,
                CanRaise::No => {}
            }
        }
        result
    }

    /// Cached version of analyze_forces_virtualizable for a CallTarget.
    /// RPython: VirtualizableAnalyzer external calls → bottom_result (False).
    fn cached_forces_virtualizable_path(&self, path: &CallPath, cache: &mut AnalysisCache) -> bool {
        if let Some(&result) = cache.forces_virtualizable.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_forces_virtualizable(&path, &mut seen);
        cache.forces_virtualizable.insert(path.clone(), result);
        result
    }

    fn cached_forces_virtualizable(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return false, // external → False (RPython bottom_result)
        };
        self.cached_forces_virtualizable_path(&path, cache)
    }

    fn cached_forces_virtualizable_family(
        &self,
        graphs: Option<&[CallPath]>,
        cache: &mut AnalysisCache,
    ) -> bool {
        let graphs = match graphs {
            Some(graphs) => graphs,
            None => return true,
        };
        graphs
            .iter()
            .any(|path| self.cached_forces_virtualizable_path(path, cache))
    }

    /// Cached version of analyze_random_effects for a CallTarget.
    /// RPython: RandomEffectsAnalyzer defaults to False for external calls.
    fn cached_random_effects_path(&self, path: &CallPath, cache: &mut AnalysisCache) -> bool {
        if let Some(&result) = cache.random_effects.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_random_effects(&path, &mut seen);
        cache.random_effects.insert(path.clone(), result);
        result
    }

    fn cached_random_effects(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return false, // external call → False (RPython default)
        };
        self.cached_random_effects_path(&path, cache)
    }

    fn cached_random_effects_family(
        &self,
        graphs: Option<&[CallPath]>,
        cache: &mut AnalysisCache,
    ) -> bool {
        let graphs = match graphs {
            Some(graphs) => graphs,
            None => return true,
        };
        graphs
            .iter()
            .any(|path| self.cached_random_effects_path(path, cache))
    }

    /// Cached version of analyze_can_invalidate for a CallTarget.
    fn cached_can_invalidate_path(&self, path: &CallPath, cache: &mut AnalysisCache) -> bool {
        if let Some(&result) = cache.can_invalidate.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_can_invalidate(&path, &mut seen);
        cache.can_invalidate.insert(path.clone(), result);
        result
    }

    fn cached_can_invalidate(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return false,
        };
        self.cached_can_invalidate_path(&path, cache)
    }

    fn cached_can_invalidate_family(
        &self,
        graphs: Option<&[CallPath]>,
        cache: &mut AnalysisCache,
    ) -> bool {
        let graphs = match graphs {
            Some(graphs) => graphs,
            None => return true,
        };
        graphs
            .iter()
            .any(|path| self.cached_can_invalidate_path(path, cache))
    }

    /// Cached version of analyze_can_collect for a CallTarget.
    /// RPython: collect_analyzer.analyze(op, self.seen_gc) (collectanalyze.py).
    /// graphanalyze.py:60: analyze_external_call → bottom_result() (False).
    fn cached_can_collect_path(&self, path: &CallPath, cache: &mut AnalysisCache) -> bool {
        if let Some(&result) = cache.can_collect.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_can_collect(&path, &mut seen);
        cache.can_collect.insert(path.clone(), result);
        result
    }

    fn cached_can_collect(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            // graphanalyze.py:60: analyze_external_call → bottom_result() (False)
            None => return false,
        };
        self.cached_can_collect_path(&path, cache)
    }

    fn cached_can_collect_family(
        &self,
        graphs: Option<&[CallPath]>,
        cache: &mut AnalysisCache,
    ) -> bool {
        let graphs = match graphs {
            Some(graphs) => graphs,
            None => return true,
        };
        graphs
            .iter()
            .any(|path| self.cached_can_collect_path(path, cache))
    }

    // ── _canraise + getcalldescr (call.py:210-355) ──────────────────

    /// RPython: CallControl._canraise(op) (call.py:337-355).
    ///
    /// ```python
    /// def _canraise(self, op):
    ///     if op.opname == 'pseudo_call_cannot_raise':
    ///         return False
    ///     try:
    ///         if self.raise_analyzer.can_raise(op):
    ///             if self.raise_analyzer_ignore_memoryerror.can_raise(op):
    ///                 return True
    ///             else:
    ///                 return "mem"
    ///         else:
    ///             return False
    ///     except DelayedPointer:
    ///         return True
    /// ```
    pub fn _canraise(&self, target: &CallTarget, cache: &mut AnalysisCache) -> CanRaise {
        if let CallTarget::Indirect {
            trait_root,
            method_name,
        } = target
        {
            let graphs = self.all_impls_for_indirect(trait_root, method_name);
            return self.cached_can_raise_family(Some(&graphs), cache);
        }
        self.cached_can_raise(target, cache)
    }

    /// RPython `call.py:210-335 CallControl.getcalldescr(op, ...)` —
    /// line-by-line port.  One function that dispatches on `op.kind`:
    ///
    /// - `OpKind::Call` → direct_call branch (call.py:240-257): extract
    ///   elidable / loopinvariant flags from the `funcobj`, validate the
    ///   caller's NON_VOID_ARGS / RESULT against the callee graph.
    /// - `OpKind::IndirectCall` → indirect_call branch (call.py:259-280):
    ///   family-wide validation (reject mixed `_elidable_function_` etc.),
    ///   family-witness signature check, family-wide analyzer caches.
    ///
    /// Both branches converge at call.py:281-335: random_effects,
    /// can_invalidate, extraeffect resolution, effectinfo assembly,
    /// post-condition asserts, and the final
    /// `cpu.calldescrof(FUNC, NON_VOID_ARGS, RESULT, effectinfo)` wrap.
    pub fn getcalldescr(
        &self,
        op: &SpaceOperation,
        arg_types: Vec<Type>,
        result_type: Type,
        oopspecindex: OopSpecIndex,
        extraeffect: Option<ExtraEffect>,
        cache: &mut AnalysisCache,
        extradescrs: Option<Vec<DescrRef>>,
    ) -> CallDescriptor {
        // Extract the direct-call target (if any) and indirect-call family
        // (if any).  Exactly one is Some after the initial dispatch;
        // downstream branches key off this.
        enum CallShape<'a> {
            Direct(&'a CallTarget),
            Indirect(Option<&'a [CallPath]>),
        }
        let shape = match &op.kind {
            OpKind::Call { target, .. } => CallShape::Direct(target),
            OpKind::IndirectCall { graphs, .. } => CallShape::Indirect(graphs.as_deref()),
            other => panic!("getcalldescr called on non-call op: {other:?}"),
        };

        // RPython call.py:240-257 direct_call branch: read `_elidable_function_`
        // / `_jit_loop_invariant_` off the funcobj.  Indirect calls have no
        // single funcobj so the flags are always false — they are enforced
        // family-wide below.
        //
        // call.py:252-257 also reads
        // `_call_aroundstate_target_` here and packages it into the
        // EffectInfo's `call_release_gil_target`.  That direct-call
        // propagation is intentionally omitted from this translated-graph
        // calldescr path — pyre's release-GIL surface is driven by the
        // separate `#[jit_release_gil]` / jit_interp macro pipeline (the
        // metainterp assembler resolves the target there), and no translated
        // graph call carries an aroundstate target, so
        // `effectinfo_from_writeanalyze` hardcodes
        // `_NO_CALL_RELEASE_GIL_TARGET`.  The indirect family still rejects
        // aroundstate members below (call.py:271-272).
        let (elidable, loopinvariant) = match shape {
            CallShape::Direct(target) => (self.is_elidable(target), self.is_loopinvariant(target)),
            CallShape::Indirect(_) => (false, false),
        };

        // RPython call.py:259-280 indirect_call branch: family-wide
        // validation. Reject families mixing elidable/loopinvariant/
        // call_aroundstate with ordinary members.
        if let CallShape::Indirect(Some(graphs)) = shape {
            if let Err(err) = self.check_indirect_call_family(graphs) {
                panic!("getcalldescr: {err}");
            }
        }

        // RPython call.py:220-234 signature validation:
        //   NON_VOID_ARGS = [x.concretetype for x in op.args[1:]
        //                                    if x.concretetype is not Void]
        //   RESULT = op.result.concretetype
        //   FUNC = op.args[0].concretetype.TO
        //   if NON_VOID_ARGS != [T for T in FUNC.ARGS if T is not Void]: raise
        //   if RESULT != FUNC.RESULT: raise
        match shape {
            CallShape::Direct(target) => {
                // RPython call.py:223-228: NON_VOID_ARGS != FUNC.ARGS-without-void
                // → raise Exception. Parameter list is recovered from startblock
                // `OpKind::Input` ops (`front/ast.rs:706-748` convention) when
                // `block.inputargs` is unpopulated; `graph_non_void_arg_types`
                // encapsulates the convention so direct-call validation matches
                // upstream's hard-fail semantics.
                if let Some((_, graph)) = self.target_to_path_and_graph(target) {
                    {
                        let expected_arg_types = graph_non_void_arg_types(graph);
                        // RPython call.py:223-228 compares the full
                        // `concretetype` list. Pyre's caller-side
                        // `arg_types` comes from `resolve_non_void_arg_types`
                        // which falls back to `Type::Ref` whenever
                        // `FunctionGraph::concretetype_of(&v)` returns `Unknown`.
                        // Trait-method test fixtures
                        // (`transform_all_handlers_to_jitcode`) hit that
                        // path because they construct `CallControl` without
                        // populating each Variable's `concretetype`, so
                        // the kind tail of every arg appears as `Ref`
                        // even when the callee declares `Int`.  Hard-fail
                        // only on arity mismatch — the kind tail surfaces
                        // as a soft signal until full `Variable.concretetype`
                        // propagation lands (`call.py:230` parity).
                        if arg_types.len() != expected_arg_types.len() {
                            panic!(
                                "in operation calling {target}: calling a \
                                 function with non-void arg kinds \
                                 {expected_arg_types:?}, but passing actual \
                                 arg kinds {arg_types:?}",
                            );
                        }
                        // RPython call.py:230-234 `if RESULT != FUNC.RESULT:
                        // raise` only validates when the callee's signature
                        // is known. call.py:222/231 reads `FUNC.RESULT`
                        // directly off the callee's funcptr type, so every
                        // graph carries its result type intrinsically. Pyre's
                        // source is `graph.return_type`, stamped at
                        // registration from the parsed Rust signature for
                        // free functions, trait/default methods, and inherent
                        // methods (lib.rs).
                        //
                        // The check stays conditional for the same reason the
                        // arg-kind check above is arity-only: the *caller*-side
                        // `result_type` is not yet fully resolved from each
                        // Variable's concretetype, so it can surface as the
                        // `Ref` fallback even when the callee declares `Void`
                        // (a void method has `return_type == None`, which maps
                        // to `Void`). Validating unconditionally then fails
                        // spuriously on call sites whose `result_type` defaulted
                        // to `Ref` (e.g. the real handler bodies exercised by
                        // `transform_all_handlers_to_jitcode`). Making this arm
                        // always-on (full parity with call.py:231) requires the
                        // same complete `Variable.concretetype` propagation the
                        // arg-kind comment depends on; until then the direct arm
                        // skips an un-typed callee while the indirect arm
                        // validates unconditionally (a family always resolves a
                        // typed witness).
                        let declared = graph.return_type.as_ref();
                        if let Some(declared) = declared {
                            // RPython has no `Result<T, E>` type — its
                            // rtyper extracts the exception via
                            // `OperationError` propagation and presents
                            // `op.result.concretetype` as the success
                            // type alone. Pyre models RPython's
                            // `bool` + `raise oefmt(...)` shape as
                            // `Result<bool, PyError>` and threads
                            // exceptions through `?`. When validating
                            // the call's `result_type` (already
                            // unwrapped through `?` at the AST lowerer)
                            // against the declared signature, project
                            // the declared `Result<T, E>` to `T` so the
                            // comparison happens in the rtyper-derived
                            // shape upstream uses (call.py:222 `FUNC.RESULT`).
                            let effective_declared =
                                crate::front::syn_metadata::transparent_result_ok_type(declared)
                                    .map(|s| s.to_string())
                                    .unwrap_or_else(|| declared.clone());
                            let expected_result =
                                return_type_string_to_value_type(Some(&effective_declared));
                            // RPython call.py:220 hard-fails when
                            // `RESULT != FUNC.RESULT`.
                            if result_type != expected_result {
                                panic!(
                                    "in operation calling {target}: calling a \
                                     function with return type \
                                     {expected_result:?}, but the actual \
                                     return type is {result_type:?}",
                                );
                            }
                        }
                    }
                }
            }
            CallShape::Indirect(graphs) => {
                // Indirect family invariant: all candidates share one
                // signature, so validate against the first resolvable
                // witness.  Mismatch is a programming error — panic like
                // RPython's `raise Exception`.
                if let Some((witness_path, witness_graph)) = graphs
                    .into_iter()
                    .flatten()
                    .find_map(|path| self.function_graphs.get(path).map(|g| (path, g)))
                {
                    let expected_arg_types = graph_non_void_arg_types(witness_graph);
                    if arg_types != expected_arg_types {
                        panic!(
                            "indirect_call in family including {witness_path:?}: \
                             calling a function with non-void arg kinds \
                             {expected_arg_types:?}, but passing actual arg \
                             kinds {arg_types:?}",
                        );
                    }
                    // Project `Result<T, E>` → `T` to match rtyper's
                    // `op.result.concretetype` shape — see the Direct arm
                    // above for the full rationale. Source: witness_graph's
                    // return_type (RPython `funcptr.TO.RESULT`).
                    let declared = witness_graph.return_type.as_ref();
                    let effective_declared = declared.map(|declared| {
                        crate::front::syn_metadata::transparent_result_ok_type(declared)
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| declared.clone())
                    });
                    let expected_result =
                        return_type_string_to_value_type(effective_declared.as_ref());
                    if result_type != expected_result {
                        panic!(
                            "indirect_call in family including {witness_path:?}: \
                             calling a function with return type \
                             {expected_result:?}, but the actual return type \
                             is {result_type:?}",
                        );
                    }
                }
            }
        }

        // RPython call.py:282-286: random_effects + can_invalidate.
        let random_effects = match shape {
            CallShape::Direct(target) => self.cached_random_effects(target, cache),
            CallShape::Indirect(graphs) => self.cached_random_effects_family(graphs, cache),
        };
        let mut extraeffect = extraeffect;
        if random_effects {
            extraeffect = Some(ExtraEffect::RandomEffects);
        }
        let can_invalidate = random_effects
            || match shape {
                CallShape::Direct(target) => self.cached_can_invalidate(target, cache),
                CallShape::Indirect(graphs) => self.cached_can_invalidate_family(graphs, cache),
            };

        // RPython call.py:286-303: determine extraeffect when not caller-set.
        if extraeffect.is_none() {
            let forces_vable = match shape {
                CallShape::Direct(target) => self.cached_forces_virtualizable(target, cache),
                CallShape::Indirect(graphs) => {
                    self.cached_forces_virtualizable_family(graphs, cache)
                }
            };
            extraeffect = Some(if forces_vable {
                ExtraEffect::ForcesVirtualOrVirtualizable
            } else if loopinvariant {
                // call.py:290 — direct branch only.
                ExtraEffect::LoopInvariant
            } else if elidable {
                // call.py:292-298 — direct branch only.
                //
                // Pyre extension: the user-facing
                // `#[majit_macros::elidable_cannot_raise]` /
                // `#[majit_macros::elidable_or_memerror]` macros assert
                // an `EF_ELIDABLE_*` shape the on-graph `_canraise`
                // analyser cannot recover on its own — pyre's
                // `analyze_external_call` defaults to `True` (call.rs:3631)
                // so any callee that reaches Vec::len / pyframe_get_pycode
                // / etc. propagates back as CanRaise::Yes.  Honour the
                // assertion before consulting `_canraise` so the
                // `EF_ELIDABLE_CANNOT_RAISE` walker arm (no trailing
                // GUARD_NO_EXCEPTION) actually fires on annotated
                // callees.
                let assertion = match shape {
                    CallShape::Direct(target) => {
                        if self.has_cannot_raise_assertion(target) {
                            Some(ExtraEffect::ElidableCannotRaise)
                        } else if self.has_memerror_only_assertion(target) {
                            Some(ExtraEffect::ElidableOrMemoryError)
                        } else {
                            None
                        }
                    }
                    CallShape::Indirect(_) => unreachable!("indirect cannot be elidable"),
                };
                if let Some(ee) = assertion {
                    ee
                } else {
                    let canraise = match shape {
                        CallShape::Direct(target) => self._canraise(target, cache),
                        CallShape::Indirect(_) => unreachable!("indirect cannot be elidable"),
                    };
                    match canraise {
                        CanRaise::No => ExtraEffect::ElidableCannotRaise,
                        CanRaise::MemoryErrorOnly => ExtraEffect::ElidableOrMemoryError,
                        CanRaise::Yes => ExtraEffect::ElidableCanRaise,
                    }
                }
            } else {
                let canraise = match shape {
                    CallShape::Direct(target) => self._canraise(target, cache),
                    CallShape::Indirect(graphs) => self.cached_can_raise_family(graphs, cache),
                };
                match canraise {
                    CanRaise::Yes | CanRaise::MemoryErrorOnly => ExtraEffect::CanRaise,
                    CanRaise::No => ExtraEffect::CannotRaise,
                }
            });
        }
        let extraeffect = extraeffect.unwrap_or(ExtraEffect::CanRaise);

        // RPython call.py:249-251: loopinvariant functions must have no args.
        if loopinvariant && !arg_types.is_empty() {
            let target = match shape {
                CallShape::Direct(t) => t,
                _ => unreachable!(),
            };
            panic!(
                "getcalldescr: arguments not supported for loop-invariant \
                 function {target}"
            );
        }

        // RPython call.py:305-318 post-conditions on elidable / loopinvariant.
        if loopinvariant && extraeffect != ExtraEffect::LoopInvariant {
            let target = match shape {
                CallShape::Direct(t) => t,
                _ => unreachable!(),
            };
            panic!(
                "getcalldescr: {target} is marked loop-invariant but got \
                 extraeffect={extraeffect:?}"
            );
        }
        if elidable {
            let target = match shape {
                CallShape::Direct(t) => t,
                _ => unreachable!(),
            };
            if !matches!(
                extraeffect,
                ExtraEffect::ElidableCannotRaise
                    | ExtraEffect::ElidableOrMemoryError
                    | ExtraEffect::ElidableCanRaise
            ) {
                panic!(
                    "getcalldescr: {target} is marked elidable but got \
                     extraeffect={extraeffect:?}"
                );
            }
            // call.py:315-318: elidable function must have a result
            if result_type == Type::Void {
                panic!("getcalldescr: {target} is elidable but has no result");
            }
        }

        // RPython call.py:320-324 effectinfo assembly.
        let effects = match shape {
            CallShape::Direct(target) => {
                analyze_readwrite(target, &self.function_graphs, self, &self.descr_indices)
            }
            CallShape::Indirect(graphs) => analyze_readwrite_indirect_family(
                graphs,
                &self.function_graphs,
                self,
                &self.descr_indices,
            ),
        };
        let can_collect = match shape {
            CallShape::Direct(target) => self.cached_can_collect(target, cache),
            CallShape::Indirect(graphs) => self.cached_can_collect_family(graphs, cache),
        };
        let effectinfo = effectinfo_from_writeanalyze(
            effects,
            extraeffect,
            oopspecindex,
            can_invalidate,
            can_collect,
            extradescrs,
        );

        // RPython call.py:326-332 post-conditions on elidable / loopinvariant.
        if elidable || loopinvariant {
            assert!(
                effectinfo.extraeffect < ExtraEffect::ForcesVirtualOrVirtualizable,
                "getcalldescr: elidable/loopinvariant call has effect {:?} \
                 >= ForcesVirtualOrVirtualizable",
                effectinfo.extraeffect
            );
        }

        // RPython call.py:334-335:
        //   return self.cpu.calldescrof(FUNC, tuple(NON_VOID_ARGS), RESULT,
        //                               effectinfo)
        // Pyre's CallDescriptor stores the same structural cache key that
        // RPython's `cpu.calldescrof()` would use; the matching funcptr is
        // plumbed separately by callers.
        CallDescriptor::from_signature(&arg_types, result_type, effectinfo)
    }

    /// RPython: calldescr_canraise(calldescr) (call.py:357-359).
    pub fn calldescr_canraise(&self, calldescr: &CallDescriptor) -> bool {
        calldescr.extra_info.check_can_raise(false)
    }
}

fn stable_symbolic_fnaddr<T: std::hash::Hash>(value: &T) -> i64 {
    use std::hash::Hasher;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish() as i64
}

pub(crate) fn symbolic_fnaddr_for_path(path: &CallPath) -> i64 {
    stable_symbolic_fnaddr(path)
}

pub(crate) fn symbolic_fnaddr_for_target(target: &CallTarget) -> i64 {
    stable_symbolic_fnaddr(target)
}

impl Default for CallControl {
    fn default() -> Self {
        Self::new()
    }
}

// ── readwrite_analyzer / collect_analyzer (effectinfo.py:276-378) ──
//
// RPython: self.readwrite_analyzer.analyze(op, self.seen_rw) → effects
// RPython: self.collect_analyzer.analyze(op, self.seen_gc) → can_collect
// Then: effectinfo_from_writeanalyze(effects, cpu, ..., can_collect)

/// RPython: readwrite_analyzer.analyze(op, self.seen_rw).
///
/// Traverses the call graph to collect read/write effects as a WriteAnalysis.
/// This is the Rust equivalent of RPython's ReadWriteAnalyzer producing a
/// set of ("struct"/"array"/"interiorfield", T, fieldname) tuples.
fn analyze_readwrite(
    target: &CallTarget,
    function_graphs: &HashMap<CallPath, FunctionGraph>,
    cc: &CallControl,
    descr_indices: &DescrIndexRegistry,
) -> WriteAnalysis {
    let mut analysis = WriteAnalysis {
        read_fields: Vec::new(),
        write_fields: Vec::new(),
        read_arrays: Vec::new(),
        write_arrays: Vec::new(),
        read_interiorfields: Vec::new(),
        write_interiorfields: Vec::new(),
        field_read_descrs: Vec::new(),
        field_write_descrs: Vec::new(),
        interior_read_descrs: Vec::new(),
        interior_write_descrs: Vec::new(),
        array_read_descrs: Vec::new(),
        array_write_descrs: Vec::new(),
        is_top: false,
    };
    if let Some(path) = cc.target_to_path(target) {
        let mut seen = HashSet::new();
        collect_readwrite_effects(
            &path,
            function_graphs,
            cc,
            descr_indices,
            &mut seen,
            &mut analysis.read_fields,
            &mut analysis.write_fields,
            &mut analysis.read_arrays,
            &mut analysis.write_arrays,
            &mut analysis.read_interiorfields,
            &mut analysis.write_interiorfields,
            &mut analysis.field_read_descrs,
            &mut analysis.field_write_descrs,
            &mut analysis.interior_read_descrs,
            &mut analysis.interior_write_descrs,
            &mut analysis.array_read_descrs,
            &mut analysis.array_write_descrs,
            &mut analysis.is_top,
        );
        // RPython: top_set only occurs from gc_add_memory_pressure (writeanalyze.py:72).
        // External calls return empty_set (bottom_result), not top_set.
        // We currently don't have gc_add_memory_pressure, so is_top stays false.
    }
    analysis
}

/// RPython `readwrite_analyzer.analyze(op, seen)` for `indirect_call`.
///
/// Unknown families (`graphs=None`) are `top_set`; known families are the
/// union of every member graph's effects.
fn analyze_readwrite_indirect_family(
    graphs: Option<&[CallPath]>,
    function_graphs: &HashMap<CallPath, FunctionGraph>,
    cc: &CallControl,
    descr_indices: &DescrIndexRegistry,
) -> WriteAnalysis {
    let mut analysis = WriteAnalysis {
        read_fields: Vec::new(),
        write_fields: Vec::new(),
        read_arrays: Vec::new(),
        write_arrays: Vec::new(),
        read_interiorfields: Vec::new(),
        write_interiorfields: Vec::new(),
        field_read_descrs: Vec::new(),
        field_write_descrs: Vec::new(),
        interior_read_descrs: Vec::new(),
        interior_write_descrs: Vec::new(),
        array_read_descrs: Vec::new(),
        array_write_descrs: Vec::new(),
        is_top: false,
    };
    let graphs = match graphs {
        Some(graphs) => graphs,
        None => {
            analysis.is_top = true;
            return analysis;
        }
    };
    let mut seen = HashSet::new();
    for path in graphs {
        collect_readwrite_effects(
            path,
            function_graphs,
            cc,
            descr_indices,
            &mut seen,
            &mut analysis.read_fields,
            &mut analysis.write_fields,
            &mut analysis.read_arrays,
            &mut analysis.write_arrays,
            &mut analysis.read_interiorfields,
            &mut analysis.write_interiorfields,
            &mut analysis.field_read_descrs,
            &mut analysis.field_write_descrs,
            &mut analysis.interior_read_descrs,
            &mut analysis.interior_write_descrs,
            &mut analysis.array_read_descrs,
            &mut analysis.array_write_descrs,
            &mut analysis.is_top,
        );
        if analysis.is_top {
            break;
        }
    }
    analysis
}

//
// In RPython, the ReadWriteAnalyzer produces a set of tuples like:
//   ("struct", T, fieldname), ("readstruct", T, fieldname),
//   ("array", T), ("readarray", T), etc.
// These are converted to field/array descriptor bitsets.
//
// In majit, we scan the callee graph's ops directly for
// FieldRead/FieldWrite/ArrayRead/ArrayWrite and collect their
// descriptor indices into EffectInfo's bitset fields.

/// RPython: effectinfo_from_writeanalyze() (effectinfo.py:276-378).
///
/// Scans the callee's graph for field/array read/write operations
/// and populates the corresponding bitset fields in EffectInfo.
/// RPython: effectinfo_from_writeanalyze(effects, cpu, extraeffect, oopspecindex,
///     can_invalidate, call_release_gil_target, extradescr, can_collect)
/// effectinfo.py:276-378.
///
/// Takes pre-analyzed `effects` (from readwrite_analyzer) and `can_collect`
/// (from collect_analyzer) and constructs an EffectInfo.
fn effectinfo_from_writeanalyze(
    effects: WriteAnalysis,
    extraeffect: ExtraEffect,
    oopspecindex: OopSpecIndex,
    can_invalidate: bool,
    can_collect: bool,
    extradescrs: Option<Vec<DescrRef>>,
) -> EffectInfo {
    // effectinfo.py:285: if effects is top_set or extraeffect == EF_RANDOM_EFFECTS:
    if effects.is_top || extraeffect == ExtraEffect::RandomEffects {
        // effectinfo.py:286-292: every readonly/write descr is `None` (wildcard).
        return EffectInfo {
            extraeffect: ExtraEffect::RandomEffects,
            oopspecindex,
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
            extradescrs: extradescrs.clone(),
            can_invalidate,
            can_collect: true, // effectinfo.py:364-365: forces → can_collect = True
            call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
        };
    }

    // effectinfo.py:345-360: readonly = reads that have NO corresponding write.
    // PyPy semantics: `tupw not in effects` — set difference at the per-tuple
    // level.  Pyre operates on the descr-index lift of the same sets.
    let readonly_descrs_fields = subtract_index_set(&effects.read_fields, &effects.write_fields);
    let readonly_descrs_arrays = subtract_index_set(&effects.read_arrays, &effects.write_arrays);
    let readonly_descrs_interiorfields =
        subtract_index_set(&effects.read_interiorfields, &effects.write_interiorfields);

    let mut write_descrs_fields = effects.write_fields;
    let mut write_descrs_arrays = effects.write_arrays;
    let mut write_descrs_interiorfields = effects.write_interiorfields;
    let field_read_descrs_raw = effects.field_read_descrs;
    let mut field_write_descrs = effects.field_write_descrs;
    let interior_read_descrs_raw = effects.interior_read_descrs;
    let mut interior_write_descrs = effects.interior_write_descrs;
    let array_read_descrs_raw = effects.array_read_descrs;
    let mut array_write_descrs = effects.array_write_descrs;
    // Sort + dedupe the write sets so the raw `_*_descrs_*` slot we hand
    // to `compute_bitstrings` matches PyPy's `frozenset[Descr]` semantics
    // (canonical, no duplicates). `subtract_index_set` already does this
    // for the readonly sets; the write paths feed straight from the
    // analyzer.
    write_descrs_fields.sort_unstable();
    write_descrs_fields.dedup();
    write_descrs_arrays.sort_unstable();
    write_descrs_arrays.dedup();
    write_descrs_interiorfields.sort_unstable();
    write_descrs_interiorfields.dedup();

    // effectinfo.py:169-181: for elidable/loopinvariant, ignore writes.
    if matches!(
        extraeffect,
        ExtraEffect::ElidableCannotRaise
            | ExtraEffect::ElidableOrMemoryError
            | ExtraEffect::ElidableCanRaise
            | ExtraEffect::LoopInvariant
    ) {
        write_descrs_fields.clear();
        write_descrs_arrays.clear();
        write_descrs_interiorfields.clear();
        field_write_descrs.clear();
        interior_write_descrs.clear();
        array_write_descrs.clear();
    }

    // Snapshot the Arc-list before consumption — `single_write_descr_array`
    // takes ownership for its `.into_iter().next()` extract, but the EI's
    // `_write_descrs_arrays: Vec<DescrRef>` raw set below also needs it.
    let array_write_descrs_snapshot: Vec<majit_ir::descr::DescrRef> = array_write_descrs.clone();

    // effectinfo.py:201-206: single_write_descr_array
    let single_write_descr_array = if array_write_descrs.len() == 1 {
        Some(array_write_descrs.into_iter().next().unwrap())
    } else {
        None
    };

    // effectinfo.py:364-365: if extraeffect >= EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE:
    //     can_collect = True
    let can_collect = if extraeffect >= ExtraEffect::ForcesVirtualOrVirtualizable {
        true
    } else {
        can_collect
    };

    // `effectinfo.py:128-145 frozenset_or_none` parity: raw descr
    // sets carry the `cpu.fielddescrof()`/`cpu.arraydescrof()`/
    // `cpu.interiorfielddescrof()` results that the analyzer found.
    //
    // Pyre's coverage today: all 6 raw sets populated.
    //   - `_readonly_descrs_fields`, `_write_descrs_fields`: via
    //     `cc.fielddescrof(idx, owner, name)` from `field.owner_root` +
    //     `field.name` at the FieldRead / FieldWrite sites
    //     (PyPy `effectinfo.py:301-305 add_struct →
    //     cpu.fielddescrof(T, fieldname)`).
    //   - `_readonly_descrs_arrays`, `_write_descrs_arrays`: via
    //     `cc.arraydescrof()` from ArrayRead / ArrayWrite ops plus
    //     interior-field synthesised array effects
    //     (`effectinfo.py:327-340` + `:355-360`).
    //     `_write_descrs_arrays` directly drives heap optimizer array
    //     cache invalidation (`heap.py:537-571 force_from_effectinfo`).
    //   - `_readonly_descrs_interiorfields`,
    //     `_write_descrs_interiorfields`: via `cc.interiorfielddescrof(
    //     idx, array_type_id, name)` from InteriorFieldRead /
    //     InteriorFieldWrite ops (PyPy `effectinfo.py:313-325
    //     add_interiorfield → cpu.interiorfielddescrof(T, fieldname)`).
    //
    // All `cc.*descrof()` helpers silently skip when struct layout is
    // not registered with `cc.struct_fields`, mirroring PyPy's
    // `consider_struct=False` / `consider_array=False` /
    // `UnsupportedFieldExc` filters at `effectinfo.py:380-397` +
    // `:316-324`.
    //
    // Identity convergence (Followup 2 + Reviewer fix-1):
    // `__majit_type_id` 는 `path_hash(concat!(module_path!(), "::",
    // stringify!(Struct)))` (`jit_struct.rs:92`) 로 def-path 해시.
    // 분석기 `field.owner_root` 는 `qualify_type_name(type_root,
    // ctx.module_prefix)` 로 use-site qualifier 를 받는다.  `canonical_
    // struct_name` (`STRUCT_ORIGIN_REGISTRY`, `5fcab5ddc8`) 가
    // bare-name 입력을 정의-모듈 qualifier 로 표준화하므로
    // `cc.fielddescrof()` / `cc.interiorfielddescrof()` / `all_interior
    // fielddescrs` 가 `gc_cache.get_field_descr(LLType::Struct(
    // path_hash(canonical)), field_name, ...)` 로 분기 통합 —
    // 분석기와 런타임이 같은 `register_keyed_field` Arc 에 도달
    // (`descr.py:218-239 get_field_descr` per-tuple identity).
    //
    // `compute_bitstrings` 도 `descr.get_ei_index()` 단일 Arc 슬롯으로
    // 수렴하므로 heap-invalidation 이 cross-module 호출 사이트에서도
    // populated bitstring 을 정상 소비한다.
    // PyPy `effectinfo.py:345-360` `readonly` rule:
    //   elif tup[0] == "readstruct":
    //       tupw = ("struct",) + tup[1:]
    //       if tupw not in effects:
    //           add_struct(readonly_descrs_fields, tup)
    // i.e. a descr that is both read and written goes only to
    // `write_descrs_*`, never to `readonly_descrs_*`. PyPy keys
    // membership on the `("struct", T, fieldname)` tuple identity —
    // distinct descrs with the same `descr.index()` (legacy u32 id)
    // would collapse incorrectly here. Pyre's Arc identity (via
    // `Arc::as_ptr`) is the closest analogue: each `cc.fielddescrof(...)`
    // call returns one Arc per (T, fieldname); two analyzer-time Arcs
    // sharing a `descr.index()` due to side-table collisions remain
    // distinct under pointer equality, matching PyPy's tuple-identity
    // membership test.
    let field_write_ptr_set: std::collections::HashSet<*const ()> = field_write_descrs
        .iter()
        .map(|d| std::sync::Arc::as_ptr(d).cast::<()>())
        .collect();
    let mut read_descrs_fields_arcs: Vec<majit_ir::descr::DescrRef> = field_read_descrs_raw
        .into_iter()
        .filter(|d| !field_write_ptr_set.contains(&std::sync::Arc::as_ptr(d).cast::<()>()))
        .collect();
    read_descrs_fields_arcs.sort_by_key(majit_ir::effectinfo::descr_ptr_id);
    read_descrs_fields_arcs.dedup_by(|a, b| std::sync::Arc::ptr_eq(a, b));
    let mut write_descrs_fields_arcs: Vec<majit_ir::descr::DescrRef> = field_write_descrs;
    write_descrs_fields_arcs.sort_by_key(majit_ir::effectinfo::descr_ptr_id);
    write_descrs_fields_arcs.dedup_by(|a, b| std::sync::Arc::ptr_eq(a, b));
    // Same `read \ write` subtract for interiorfield + array (PyPy
    // `effectinfo.py:351-360`), again by Arc identity.
    let interior_write_ptr_set: std::collections::HashSet<*const ()> = interior_write_descrs
        .iter()
        .map(|d| std::sync::Arc::as_ptr(d).cast::<()>())
        .collect();
    let mut read_descrs_interior_arcs: Vec<majit_ir::descr::DescrRef> = interior_read_descrs_raw
        .into_iter()
        .filter(|d| !interior_write_ptr_set.contains(&std::sync::Arc::as_ptr(d).cast::<()>()))
        .collect();
    read_descrs_interior_arcs.sort_by_key(majit_ir::effectinfo::descr_ptr_id);
    read_descrs_interior_arcs.dedup_by(|a, b| std::sync::Arc::ptr_eq(a, b));
    let mut write_descrs_interior_arcs: Vec<majit_ir::descr::DescrRef> = interior_write_descrs;
    write_descrs_interior_arcs.sort_by_key(majit_ir::effectinfo::descr_ptr_id);
    write_descrs_interior_arcs.dedup_by(|a, b| std::sync::Arc::ptr_eq(a, b));
    let array_write_ptr_set: std::collections::HashSet<*const ()> = array_write_descrs_snapshot
        .iter()
        .map(|d| std::sync::Arc::as_ptr(d).cast::<()>())
        .collect();
    let mut read_descrs_arrays_arcs: Vec<majit_ir::descr::DescrRef> = array_read_descrs_raw
        .into_iter()
        .filter(|d| !array_write_ptr_set.contains(&std::sync::Arc::as_ptr(d).cast::<()>()))
        .collect();
    read_descrs_arrays_arcs.sort_by_key(majit_ir::effectinfo::descr_ptr_id);
    read_descrs_arrays_arcs.dedup_by(|a, b| std::sync::Arc::ptr_eq(a, b));
    let mut write_descrs_arrays_arcs: Vec<majit_ir::descr::DescrRef> = array_write_descrs_snapshot;
    write_descrs_arrays_arcs.sort_by_key(majit_ir::effectinfo::descr_ptr_id);
    write_descrs_arrays_arcs.dedup_by(|a, b| std::sync::Arc::ptr_eq(a, b));
    EffectInfo {
        extraeffect,
        oopspecindex,
        _readonly_descrs_fields: Some(read_descrs_fields_arcs),
        _write_descrs_fields: Some(write_descrs_fields_arcs),
        _readonly_descrs_arrays: Some(read_descrs_arrays_arcs),
        _write_descrs_arrays: Some(write_descrs_arrays_arcs),
        _readonly_descrs_interiorfields: Some(read_descrs_interior_arcs),
        _write_descrs_interiorfields: Some(write_descrs_interior_arcs),
        readonly_descrs_fields: Some(majit_ir::bitstring::make_bitstring(&readonly_descrs_fields)),
        write_descrs_fields: Some(majit_ir::bitstring::make_bitstring(&write_descrs_fields)),
        readonly_descrs_arrays: Some(majit_ir::bitstring::make_bitstring(&readonly_descrs_arrays)),
        write_descrs_arrays: Some(majit_ir::bitstring::make_bitstring(&write_descrs_arrays)),
        readonly_descrs_interiorfields: Some(majit_ir::bitstring::make_bitstring(
            &readonly_descrs_interiorfields,
        )),
        write_descrs_interiorfields: Some(majit_ir::bitstring::make_bitstring(
            &write_descrs_interiorfields,
        )),
        single_write_descr_array,
        extradescrs,
        can_invalidate,
        can_collect,
        call_release_gil_target: EffectInfo::_NO_CALL_RELEASE_GIL_TARGET,
    }
}

/// `effectinfo.py:345-360` set difference (`tupw not in effects`).
///
/// Returns the deduped sorted list of indices in `read` that have no
/// matching entry in `write`.  PyPy's reference implementation works on
/// frozensets of descr objects; pyre operates on the descr-index lift
/// already produced by `DescrIndexRegistry`.
fn subtract_index_set(read: &[u32], write: &[u32]) -> Vec<u32> {
    let write_set: std::collections::HashSet<u32> = write.iter().copied().collect();
    let mut diff: Vec<u32> = read
        .iter()
        .copied()
        .filter(|idx| !write_set.contains(idx))
        .collect();
    diff.sort_unstable();
    diff.dedup();
    diff
}

/// RPython: `op.args[0].concretetype` — resolve full ARRAY identity.
///
/// Returns the full ARRAY type string (e.g. `"Vec<Point>"`, `"Vec<i64>"`),
/// matching RPython's ARRAY lltype which is the cache key for
/// `cpu.arraydescrof(ARRAY)` (descr.py:348-351).
///
/// Resolution order:
/// 1. Parser-set `array_type_id` (full container type from variable decl)
/// 2. Producer chain trace-back for `op.args[0].concretetype`:
///    - FieldRead: field type from struct_fields (full type string)
///    - ArrayRead: propagate the array's own array_type_id
///    - Call: return type from the callee graph's return_type
/// 3. Phi/link source chain (limited depth)
/// 4. None (conservative: falls back to item_ty-only keying)
fn resolve_array_identity(
    base: &crate::flowspace::model::Variable,
    op_array_type_id: &Option<String>,
    value_producers: &HashMap<crate::flowspace::model::Variable, &crate::model::OpKind>,
    phi_sources: &HashMap<crate::flowspace::model::Variable, Option<LinkArg>>,
    cc: &CallControl,
) -> Option<String> {
    fn producer_array_identity(
        value: &crate::flowspace::model::Variable,
        value_producers: &HashMap<crate::flowspace::model::Variable, &crate::model::OpKind>,
        cc: &CallControl,
    ) -> Option<String> {
        let producer = value_producers.get(value)?;
        match producer {
            // FieldRead: self.array → full ARRAY type from struct registry.
            // RPython: op.args[0].concretetype is the ARRAY lltype directly.
            OpKind::FieldRead { field, .. } => field
                .owner_root
                .as_deref()
                .and_then(|owner| cc.field_type(owner, &field.name))
                .map(ToOwned::to_owned),
            // ArrayRead with known array_type_id: propagate.
            OpKind::ArrayRead { array_type_id, .. } if array_type_id.is_some() => {
                array_type_id.clone()
            }
            // Call result: RPython resolves via result.concretetype → full type.
            OpKind::Call { target, .. } => cc
                .target_to_path(target)
                .and_then(|callee_path| cc.function_graphs().get(&callee_path))
                .and_then(|g| g.return_type.clone()),
            OpKind::Input { .. } => None,
            _ => None,
        }
    }

    fn const_array_identity(value: &crate::flowspace::model::ConstValue) -> Option<String> {
        match value {
            crate::flowspace::model::ConstValue::List(_) => Some("list".to_string()),
            crate::flowspace::model::ConstValue::Tuple(_) => Some("tuple".to_string()),
            crate::flowspace::model::ConstValue::ByteStr(_) => Some("str".to_string()),
            crate::flowspace::model::ConstValue::UniStr(_) => Some("unicode".to_string()),
            crate::flowspace::model::ConstValue::HostObject(obj) => {
                Some(obj.instance_class().unwrap_or(obj).qualname().to_string())
            }
            _ => None,
        }
    }

    // 1. Parser-set element type (from FnArg or typed let binding).
    if op_array_type_id.is_some() {
        return op_array_type_id.clone();
    }
    // 2. Trace back to producer — RPython: op.args[0].concretetype.
    if let Some(identity) = producer_array_identity(base, value_producers, cc) {
        return Some(identity);
    }
    // 3. Phi/link: RPython concretetype propagates through block boundaries.
    // Follow inputarg → source link-arg chain (limited depth to avoid cycles).
    let mut source = LinkArg::Value(base.clone());
    for _ in 0..4 {
        match &source {
            LinkArg::Value(var) => {
                if let Some(identity) = producer_array_identity(var, value_producers, cc) {
                    return Some(identity);
                }
                // `None` entries mark inputargs merged from multiple
                // predecessors — stop chasing and fall back to the
                // `item_ty`-only path so the descr stays conservative.
                let Some(Some(next)) = phi_sources.get(var) else {
                    break;
                };
                source = next.clone();
            }
            LinkArg::Const(value) => return const_array_identity(&value.value),
        }
    }
    None
}

/// RPython: `ARRAY.OF` — extract element type from full ARRAY type string.
///
/// Handles all Rust array/container notations:
/// - `Vec<Point>` → `"Point"` (angle brackets)
/// - `[i64]` → `"i64"` (slice)
/// - `[Point; 10]` → `"Point"` (fixed-size array)
/// - `&[Point]` / `&mut [Point]` / `*const [Point]` / `*mut [Point]` —
///   the pointer-like prefix is stripped first so the slice body is
///   matched normally. Mirrors `front::ast::extract_element_type_from_str`
///   so source-level analysis and effect bookkeeping agree on the
///   item type (`descr.py:241-254 get_type_flag` reads the same
///   `ARRAY.OF` regardless of how the lltype is referenced).
fn extract_element_type_from_str(type_str: &str) -> Option<String> {
    let mut s = type_str.trim();
    loop {
        let stripped = s
            .strip_prefix("*const ")
            .or_else(|| s.strip_prefix("*mut "))
            .or_else(|| s.strip_prefix("&mut "))
            .or_else(|| s.strip_prefix("&"));
        match stripped {
            Some(rest) => s = rest.trim_start(),
            None => break,
        }
    }
    // Square brackets: [T] or [T; N]
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        let elem = if let Some(semi) = inner.find(';') {
            inner[..semi].trim()
        } else {
            inner.trim()
        };
        if !elem.is_empty() {
            return Some(elem.to_string());
        }
    }
    // Angle brackets: Vec<T>, Box<T>, etc.  Checked after the slice
    // form so `[Rc<T>]` yields `Rc<T>`, not `T` — matches the front-end
    // counterpart in `front::ast::extract_element_type_from_str`.
    if let (Some(start), Some(end)) = (s.find('<'), s.rfind('>')) {
        if start < end {
            return Some(s[start + 1..end].trim().to_string());
        }
    }
    None
}

/// Transitive read/write effect collection.
///
/// RPython: ReadWriteAnalyzer.analyze() — traverses callee graphs.
/// Produces a set of tuples: ("struct"/"readstruct"/"array"/"readarray", ...).
///
/// We collect raw reads and writes separately into bitsets. The caller
/// (`effectinfo_from_writeanalyze`) then applies the RPython rule:
/// "readonly = reads & ~writes" (effectinfo.py:345-360).
fn collect_readwrite_effects(
    path: &CallPath,
    function_graphs: &HashMap<CallPath, FunctionGraph>,
    cc: &CallControl,
    descr_indices: &DescrIndexRegistry,
    seen: &mut HashSet<CallPath>,
    read_fields: &mut Vec<u32>,
    write_fields: &mut Vec<u32>,
    read_arrays: &mut Vec<u32>,
    write_arrays: &mut Vec<u32>,
    // effectinfo.py:313-325: interiorfield descriptor sets.
    read_interiorfields: &mut Vec<u32>,
    write_interiorfields: &mut Vec<u32>,
    // effectinfo.py:294,301-305: `readonly_descrs_fields = []` populated
    // via `add_struct → cpu.fielddescrof(T, fieldname)` from
    // `("readstruct", T, fieldname)` tuples.
    field_read_descrs: &mut Vec<majit_ir::descr::DescrRef>,
    // effectinfo.py:297,301-305: `write_descrs_fields = []` from
    // `("struct", T, fieldname)` tuples.
    field_write_descrs: &mut Vec<majit_ir::descr::DescrRef>,
    // effectinfo.py:296,313-325: `readonly_descrs_interiorfields = []`
    // populated via `add_interiorfield → cpu.interiorfielddescrof(T,
    // fieldname)` from `("readinteriorfield", T, fieldname)` tuples.
    interior_read_descrs: &mut Vec<majit_ir::descr::DescrRef>,
    // effectinfo.py:299,313-325: `write_descrs_interiorfields = []`
    // from `("interiorfield", T, fieldname)` tuples.
    interior_write_descrs: &mut Vec<majit_ir::descr::DescrRef>,
    // effectinfo.py:295,307-311: `readonly_descrs_arrays = []` populated
    // via `add_array → cpu.arraydescrof(ARRAY)` from `("readarray", T)`
    // tuples (and `("readinteriorfield", T, _)` synthesised at
    // effectinfo.py:327-340).
    array_read_descrs: &mut Vec<majit_ir::descr::DescrRef>,
    // effectinfo.py:201-206,298,355-356: `write_descrs_arrays = []` —
    // also drives single_write_descr_array.
    array_write_descrs: &mut Vec<majit_ir::descr::DescrRef>,
    is_top: &mut bool,
) {
    if *is_top {
        return;
    }
    if !seen.insert(path.clone()) {
        return;
    }
    let graph = match function_graphs.get(path) {
        Some(g) => g,
        None => {
            // RPython: analyze_external_call() returns bottom_result() (empty_set),
            // NOT top_set. External calls have no KNOWN read/write effects.
            // The extraeffect (CanRaise etc.) is determined separately.
            return;
        }
    };

    // RPython: the rtyped graph gives op.args[0].concretetype directly.
    // In majit, build a Variable-keyed producer map so the array
    // identity follows from each defining op's result Variable
    // (orthodox per `flowspace/model.py:Variable` identity).
    let value_producers: HashMap<crate::flowspace::model::Variable, &crate::model::OpKind> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| op.result.as_ref().map(|v| (v.clone(), &op.kind)))
        .collect();

    // RPython: phi/link args carry concretetype through block boundaries.
    // Build inputarg → source value mapping from every exit link (upstream
    // `flowspace/model.py:244 renamevariables` walks `for link in
    // self.exits: link.args`), so resolve_array_identity can trace through
    // control-flow merges of any exit fan-out.
    // Build the phi-sources map by walking each exit link's args
    // positionally against the `Block.inputargs` list (`Vec<Variable>`,
    // orthodox per `flowspace/model.py:244 renamevariables`).
    // Conservative phi-source map: an inputarg with exactly one
    // incoming edge gets `Some(src)`; an inputarg merged from two or
    // more predecessors is demoted to `None` so `resolve_array_identity`
    // stops chasing provenance and falls back to the existing
    // `array_type_id` / item-type-only path.  Without this demotion the
    // last-writer-wins insert would make the array descr selection
    // depend on HashMap iteration order, which can stamp the wrong
    // effect bits on cross-block merges.
    let mut phi_sources: HashMap<crate::flowspace::model::Variable, Option<LinkArg>> =
        HashMap::new();
    for block in &graph.blocks {
        for link in &block.exits {
            if let Some(target_block) = graph.blocks.get(link.target.0) {
                for (target_arg, src) in target_block.inputargs.iter().zip(link.args.iter()) {
                    phi_sources
                        .entry(target_arg.clone())
                        .and_modify(|entry| *entry = None)
                        .or_insert_with(|| Some(src.clone()));
                }
            }
        }
    }

    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                // RPython: ("readstruct", T, fieldname)
                OpKind::FieldRead { field, .. } => {
                    // RPython: cpu.fielddescrof(T, fieldname).get_ei_index()
                    let idx = descr_indices.field_index(&field.owner_root, &field.name);
                    read_fields.push(idx);
                    // RPython: effectinfo.py:301-305 `add_struct →
                    // cpu.fielddescrof(T, fieldname)`. Dedup by index
                    // (frozenset semantics). Silently skipped when the
                    // struct layout is not registered with `cc.struct_fields`
                    // (analyzer-unknown owner — matches PyPy's
                    // `consider_struct=False` filter at effectinfo.py:380).
                    if let Some(owner) = field.owner_root.as_deref() {
                        if !field_read_descrs.iter().any(|d| d.index() == idx) {
                            if let Some(descr) = cc.fielddescrof(idx, owner, &field.name) {
                                field_read_descrs.push(descr);
                            }
                        }
                    }
                }
                // RPython: ("struct", T, fieldname)
                OpKind::FieldWrite { field, .. } => {
                    let idx = descr_indices.field_index(&field.owner_root, &field.name);
                    write_fields.push(idx);
                    // RPython: effectinfo.py:301-305 — same as FieldRead's
                    // implicit `add_struct` walk, just into `write_descrs_fields`.
                    if let Some(owner) = field.owner_root.as_deref() {
                        if !field_write_descrs.iter().any(|d| d.index() == idx) {
                            if let Some(descr) = cc.fielddescrof(idx, owner, &field.name) {
                                field_write_descrs.push(descr);
                            }
                        }
                    }
                }
                // RPython: ("readarray", T)
                OpKind::ArrayRead {
                    base,
                    item_ty,
                    array_type_id,
                    nolength,
                    ..
                } => {
                    // RPython: op.args[0].concretetype → cpu.arraydescrof(ARRAY).
                    // `resolve_array_identity` documents `None` as the
                    // "fall back to item_ty-only keying" path; the
                    // `or_else` covers that.
                    let resolved_id = resolve_array_identity(
                        base,
                        array_type_id,
                        &value_producers,
                        &phi_sources,
                        cc,
                    )
                    .or_else(|| array_type_id.clone());
                    let len_offset = if *nolength { None } else { Some(0) };
                    let idx = descr_indices.array_index(
                        value_type_discriminant(item_ty),
                        &resolved_id,
                        len_offset,
                    );
                    read_arrays.push(idx);
                    // RPython: effectinfo.py:307-311 + :355-356 — `add_array`
                    // walks `("readarray", T)` tuples through
                    // `cpu.arraydescrof(ARRAY)` and appends to
                    // `readonly_descrs_arrays`. Dedup by descriptor index
                    // (frozenset semantics, matching `ArrayWrite` handler).
                    if !array_read_descrs.iter().any(|d| d.index() == idx) {
                        let ir_type = match item_ty {
                            crate::model::ValueType::Int
                            | crate::model::ValueType::Unsigned
                            | crate::model::ValueType::Bool
                            | crate::model::ValueType::State => majit_ir::value::Type::Int,
                            crate::model::ValueType::Ref(_) | crate::model::ValueType::Unknown => {
                                majit_ir::value::Type::Ref
                            }
                            crate::model::ValueType::Float => majit_ir::value::Type::Float,
                            crate::model::ValueType::Void => majit_ir::value::Type::Void,
                        };
                        array_read_descrs.push(cc.arraydescrof(
                            idx,
                            &resolved_id,
                            ir_type,
                            len_offset,
                        ));
                    }
                }
                // RPython: ("array", T)
                OpKind::ArrayWrite {
                    base,
                    item_ty,
                    array_type_id,
                    nolength,
                    ..
                } => {
                    // See the matching `ArrayRead` arm above.
                    let resolved_id = resolve_array_identity(
                        base,
                        array_type_id,
                        &value_producers,
                        &phi_sources,
                        cc,
                    )
                    .or_else(|| array_type_id.clone());
                    let len_offset = if *nolength { None } else { Some(0) };
                    let idx = descr_indices.array_index(
                        value_type_discriminant(item_ty),
                        &resolved_id,
                        len_offset,
                    );
                    write_arrays.push(idx);
                    // RPython: effectinfo.py:307-311 — cpu.arraydescrof(ARRAY).
                    // Dedup by descriptor index (frozenset semantics).
                    if !array_write_descrs.iter().any(|d| d.index() == idx) {
                        let ir_type = match item_ty {
                            crate::model::ValueType::Int
                            | crate::model::ValueType::Unsigned
                            | crate::model::ValueType::Bool
                            | crate::model::ValueType::State => majit_ir::value::Type::Int,
                            crate::model::ValueType::Ref(_) | crate::model::ValueType::Unknown => {
                                majit_ir::value::Type::Ref
                            }
                            crate::model::ValueType::Float => majit_ir::value::Type::Float,
                            crate::model::ValueType::Void => majit_ir::value::Type::Void,
                        };
                        // descr.py:359-362 + ARRAY_INSIDE._hints.get(
                        // 'nolength', False): the producer-side bit
                        // carried on `OpKind::ArrayWrite` flows through
                        // here so EffectInfo descrs match the same
                        // `lendescr` shape `arraydescrof()` minted at the
                        // emit-bytecode site (assembler.rs).
                        array_write_descrs.push(cc.arraydescrof(
                            idx,
                            &resolved_id,
                            ir_type,
                            len_offset,
                        ));
                    }
                }
                // RPython: ("readinteriorfield", T, fieldname)
                // effectinfo.py:351-354: records interiorfield descriptor.
                // effectinfo.py:327-340: ALSO implicitly records array read.
                OpKind::InteriorFieldRead {
                    base,
                    field,
                    array_type_id,
                    ..
                } => {
                    // See the matching `ArrayRead` arm above.
                    let resolved_id = resolve_array_identity(
                        base,
                        array_type_id,
                        &value_producers,
                        &phi_sources,
                        cc,
                    )
                    .or_else(|| array_type_id.clone());
                    // Interior field bit — keyed on (ARRAY, fieldname),
                    // matching cpu.interiorfielddescrof(ARRAY, fieldname).
                    let ifield_idx = descr_indices.interiorfield_index(&resolved_id, &field.name);
                    read_interiorfields.push(ifield_idx);
                    // RPython: effectinfo.py:313-325 `add_interiorfield →
                    // cpu.interiorfielddescrof(ARRAY, fieldname)`. Dedup
                    // by descriptor index. Silently skipped when the
                    // array's element struct is unknown to
                    // `cc.struct_fields` or the field is absent
                    // (PyPy `effectinfo.py:316-324 consider_array` /
                    // `Void` / `UnsupportedFieldExc` filters).
                    if !interior_read_descrs.iter().any(|d| d.index() == ifield_idx) {
                        if let Some(descr) =
                            cc.interiorfielddescrof(ifield_idx, &resolved_id, &field.name)
                        {
                            interior_read_descrs.push(descr);
                        }
                    }
                    // effectinfo.py:327-340: synthesizes `("readarray", T)`
                    // for every `("readinteriorfield", T, _)` so the
                    // implicit array read is recorded; effectinfo.py:355-360
                    // then walks `add_array → cpu.arraydescrof(ARRAY)` →
                    // `readonly_descrs_arrays.append(descr)`. Interior fields
                    // only exist in struct arrays → element type is Ref.
                    // `len_offset` honours `ARRAY_INSIDE._hints.get('nolength',
                    // False)` (`descr.py:359`) so headerless array-of-structs
                    // shapes hash to a different bitstring slot than
                    // length-prefixed shapes of the same item type.
                    let len_offset = if crate::front::syn_metadata::nolength_from_array_type_id(
                        resolved_id.as_deref(),
                    ) {
                        None
                    } else {
                        Some(0)
                    };
                    let arr_idx = descr_indices.array_index(
                        value_type_discriminant(&crate::model::ValueType::Ref(None)),
                        &resolved_id,
                        len_offset,
                    );
                    read_arrays.push(arr_idx);
                    // RPython: effectinfo.py:355-360 — cpu.arraydescrof(ARRAY)
                    // appended to readonly_descrs_arrays via the synthesized
                    // ("readarray", T) tuple. Dedup by descriptor index
                    // (frozenset semantics).
                    if !array_read_descrs.iter().any(|d| d.index() == arr_idx) {
                        array_read_descrs.push(cc.arraydescrof(
                            arr_idx,
                            &resolved_id,
                            majit_ir::value::Type::Ref,
                            len_offset,
                        ));
                    }
                }
                // RPython: ("interiorfield", T, fieldname)
                // effectinfo.py:349-350: records interiorfield descriptor.
                // effectinfo.py:327-340: ALSO implicitly records array write.
                OpKind::InteriorFieldWrite {
                    base,
                    field,
                    array_type_id,
                    ..
                } => {
                    // See the matching `ArrayRead` arm above.
                    let resolved_id = resolve_array_identity(
                        base,
                        array_type_id,
                        &value_producers,
                        &phi_sources,
                        cc,
                    )
                    .or_else(|| array_type_id.clone());
                    // Interior field bit — keyed on (ARRAY, fieldname),
                    // matching cpu.interiorfielddescrof(ARRAY, fieldname).
                    let ifield_idx = descr_indices.interiorfield_index(&resolved_id, &field.name);
                    write_interiorfields.push(ifield_idx);
                    // RPython: effectinfo.py:313-325 — same as
                    // InteriorFieldRead's `add_interiorfield` walk,
                    // routed into `write_descrs_interiorfields`.
                    if !interior_write_descrs
                        .iter()
                        .any(|d| d.index() == ifield_idx)
                    {
                        if let Some(descr) =
                            cc.interiorfielddescrof(ifield_idx, &resolved_id, &field.name)
                        {
                            interior_write_descrs.push(descr);
                        }
                    }
                    // effectinfo.py:327-340: synthesizes `("array", T)`
                    // for every `("interiorfield", T, _)` so the implicit
                    // array write is recorded; effectinfo.py:355-356
                    // then walks `add_array → cpu.arraydescrof(ARRAY)` →
                    // `write_descrs_arrays.append(descr)`. Interior fields
                    // only exist in struct arrays → element type is Ref;
                    // `len_offset` reflects `ARRAY_INSIDE._hints['nolength']`
                    // (`descr.py:359`) so headerless array-of-structs shapes
                    // do not alias length-prefixed ones at the EffectInfo
                    // bitset.
                    let len_offset = if crate::front::syn_metadata::nolength_from_array_type_id(
                        resolved_id.as_deref(),
                    ) {
                        None
                    } else {
                        Some(0)
                    };
                    let arr_idx = descr_indices.array_index(
                        value_type_discriminant(&crate::model::ValueType::Ref(None)),
                        &resolved_id,
                        len_offset,
                    );
                    write_arrays.push(arr_idx);
                    // RPython: effectinfo.py:355-356 — cpu.arraydescrof(ARRAY)
                    // appended to write_descrs_arrays via the synthesized
                    // ("array", T) tuple. Dedup by descriptor index
                    // (frozenset semantics, matching ArrayWrite handler).
                    if !array_write_descrs.iter().any(|d| d.index() == arr_idx) {
                        array_write_descrs.push(cc.arraydescrof(
                            arr_idx,
                            &resolved_id,
                            majit_ir::value::Type::Ref,
                            len_offset,
                        ));
                    }
                }
                // Recursive: follow calls.
                OpKind::Call { target, .. } => {
                    if let Some(callee_path) = cc.target_to_path(target) {
                        collect_readwrite_effects(
                            &callee_path,
                            function_graphs,
                            cc,
                            descr_indices,
                            seen,
                            read_fields,
                            write_fields,
                            read_arrays,
                            write_arrays,
                            read_interiorfields,
                            write_interiorfields,
                            field_read_descrs,
                            field_write_descrs,
                            interior_read_descrs,
                            interior_write_descrs,
                            array_read_descrs,
                            array_write_descrs,
                            is_top,
                        );
                    } else {
                        // RPython: analyze_external_call() → bottom_result() (empty_set).
                        // External calls have no known read/write effects.
                        // (NOT top_set — that only comes from gc_add_memory_pressure.)
                    }
                }
                OpKind::IndirectCall { graphs, .. } => match graphs.as_deref() {
                    None => {
                        *is_top = true;
                        return;
                    }
                    Some(graphs) => {
                        for callee_path in graphs {
                            collect_readwrite_effects(
                                callee_path,
                                function_graphs,
                                cc,
                                descr_indices,
                                seen,
                                read_fields,
                                write_fields,
                                read_arrays,
                                write_arrays,
                                read_interiorfields,
                                write_interiorfields,
                                field_read_descrs,
                                field_write_descrs,
                                interior_read_descrs,
                                interior_write_descrs,
                                array_read_descrs,
                                array_write_descrs,
                                is_top,
                            );
                            if *is_top {
                                return;
                            }
                        }
                    }
                },
                _ => {}
            }
        }
    }
}

/// RPython: `heaptracker.all_interiorfielddescrs(gccache, ARRAY)`.
///
/// For an array-of-structs, iterate `STRUCT._names` and create
/// `InteriorFieldDescr(arraydescr, fielddescr)` for each field.
/// Mirrors heaptracker.py:74-92 with `get_field_descr=get_interiorfield_descr`.
///
/// Layout source priority (RPython: `symbolic.get_field_token()`):
/// 1. `cc.struct_layouts[struct_name]` — actual layout from runtime
/// 2. Type-string heuristic fallback from `get_type_flag()`
///
/// Returns `(fielddescrs, item_size)`.
fn all_interiorfielddescrs(
    cc: &CallControl,
    struct_name: &str,
    array_key: majit_ir::descr::LLType,
    array_descr: std::sync::Arc<dyn majit_ir::descr::ArrayDescr>,
) -> (Vec<majit_ir::descr::DescrRef>, usize) {
    use majit_ir::descr::{LLType, path_hash};
    // `descr.py:423-438 get_interiorfield_descr` reuses
    // `gc_cache._cache_field[REALARRAY.OF][name]` for the inner
    // FieldDescr and `gc_cache._cache_interiorfield[(ARRAY, name,
    // arrayfieldname=None)]` for the InteriorFieldDescr wrapper.
    // Route both lookups through `gc_cache.get_field_descr` /
    // `gc_cache.get_interiorfield_descr` so the analyzer's
    // `cc.interiorfielddescrof(ARRAY, fieldname)` (call.rs analyzer
    // arm above) and the struct-array `all_interiorfielddescrs`
    // population path share a single `Arc<SimpleInteriorFieldDescr>`
    // per `(ARRAY, fieldname)` tuple — PyPy `cpu.interiorfielddescrof`
    // per-tuple object identity.
    //
    // Use-import resolver: canonicalise `struct_name` to
    // `defining_module::Bare` so the analyzer hits the same
    // `_cache_size` slot the runtime's qualified def-path dual-publish
    // wrote to (PyPy `cache[STRUCT]` lltype-object identity).  When
    // the resolver has no entry, `canonical_struct_name` returns the
    // bare name verbatim and we hit the simple-name slot.
    let struct_canonical = majit_ir::descr::canonical_struct_name(struct_name);
    let struct_key = LLType::Struct(path_hash(&struct_canonical));
    // `descr.py:238 fielddescr.parent_descr = get_size_descr(gccache,
    // STRUCT, vtable)` — PyPy's `get_field_descr` calls
    // `get_size_descr` on cache miss so the freshly-minted FieldDescr
    // gets a non-None `parent_descr`.  Pyre's `gc_cache.get_field_descr`
    // only READS `_cache_size[struct_key]` (descr.rs:418); ensure the
    // slot is populated first by routing through `get_size_descr` here,
    // so the inner-FieldDescr loop below sees the parent.  Cache hit
    // is a no-op; cache miss mints the SizeDescr per `descr.py:108-118`.
    // Path 1 carries `layout.size` directly; Path 2 derives from
    // accumulated offsets after the heuristic walk.
    // Path 1: actual layout from runtime (RPython: symbolic.get_field_token)
    if let Some(layout) = cc.struct_layouts.get(struct_name) {
        let size_descr_arc = {
            let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
            // descr.py:108-118: vtable=0 here — pyre struct-array
            // interior fields are not GcStruct-of-Object so no vtable.
            // immutable_flag=false: defensive default; field-level
            // immutability lives on FieldDescr.is_immutable.
            gc.get_size_descr(struct_key.clone(), layout.size, 0, false)
        };
        let mut entries: Vec<(
            String,
            usize,
            usize,
            majit_ir::value::Type,
            bool,
            majit_ir::descr::ArrayFlag,
            bool,
        )> = Vec::new();
        for fl in &layout.fields {
            if fl.field_type == majit_ir::value::Type::Void {
                continue;
            }
            if fl.flag == majit_ir::descr::ArrayFlag::Struct {
                return (Vec::new(), 0);
            }
            entries.push((
                fl.name.clone(),
                fl.offset,
                fl.size,
                fl.field_type,
                fl.is_immutable(),
                fl.flag,
                fl.is_quasi_immutable(),
            ));
        }
        let mut result = Vec::new();
        for (
            index_in_parent,
            (name, offset, field_size, field_type, is_immutable, flag, is_quasi_immutable),
        ) in entries.iter().enumerate()
        {
            // `descr.py:435 fielddescr = get_field_descr(gc_ll_descr,
            // REALARRAY.OF, name)` — PyPy's `get_field_descr` returns the
            // SAME FieldDescr object that `cpu.fielddescrof(STRUCT, name)`
            // (the direct path) returns, because `_cache_field[STRUCT][name]`
            // is a single slot.  Pyre's runtime publishes its
            // `PyreFieldDescr` inside `PyreSizeDescr.all_fielddescrs`
            // (build_object_descr_group), NOT into `_cache_field`; the
            // analyzer's direct interiorfielddescrof path
            // (call.rs:1489) walks `sd.all_fielddescrs()` to find that
            // PyreFieldDescr.  Mirror that walk here so struct-array
            // population also reuses the runtime PyreFieldDescr Arc —
            // otherwise the two paths mint divergent FieldDescr Arcs and
            // `compute_bitstrings`' `set_ei_index` lands on a different
            // descr than `force_from_effectinfo` reads.  Name match:
            // bare or `.{name}` suffix per descr.py:227 naming convention.
            let mut walked: Option<std::sync::Arc<dyn majit_ir::descr::FieldDescr>> = None;
            if let Some(sd) = size_descr_arc.as_size_descr() {
                let needle = format!(".{}", name);
                for fd in sd.all_fielddescrs() {
                    let stored = fd.field_name();
                    if stored == *name || stored.ends_with(&needle) {
                        walked = Some(fd.clone());
                        break;
                    }
                }
            }
            let fd: std::sync::Arc<dyn majit_ir::descr::FieldDescr> = match walked {
                Some(fd) => fd,
                None => {
                    let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
                    gc.get_field_descr(
                        struct_key.clone(),
                        name,
                        *offset,
                        *field_size,
                        *field_type,
                        *is_immutable,
                        *is_quasi_immutable,
                        *flag,
                        index_in_parent,
                    )
                }
            };
            let ifd = {
                let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
                gc.get_interiorfield_descr(
                    array_key.clone(),
                    name.clone(),
                    String::new(),
                    array_descr.clone(),
                    fd,
                )
            };
            ifd.set_index(index_in_parent as u32);
            result.push(ifd as majit_ir::descr::DescrRef);
        }
        return (result, layout.size);
    }

    // Path 2: type-string heuristic fallback
    let fields = match cc.struct_fields.fields.get(struct_name) {
        Some(f) => f,
        None => return (Vec::new(), 0),
    };
    for (_, field_type_str) in fields.iter() {
        if cc.is_known_struct(field_type_str) {
            return (Vec::new(), 0);
        }
    }
    // RPython: STRUCT._immutable_field(fieldname) — class-level
    // `_immutable_fields_` declaration. Honored by all_fielddescrs.
    let immutable_ranks: std::collections::HashMap<&str, crate::model::ImmutableRank> = cc
        .immutable_fields_by_struct
        .get(struct_name)
        .map(|v| v.iter().map(|(n, r)| (n.as_str(), *r)).collect())
        .unwrap_or_default();
    let mut offset: usize = 0;
    let mut entries: Vec<(
        String,
        usize,
        usize,
        majit_ir::value::Type,
        bool,
        bool,
        majit_ir::descr::ArrayFlag,
    )> = Vec::new();
    for (_i, (field_name, field_type_str)) in fields.iter().enumerate() {
        let (flag, field_type, field_size) = get_type_flag(field_type_str);
        if field_type == majit_ir::value::Type::Void {
            continue;
        }
        // heaptracker.py:87-88 all_interiorfielddescrs:
        //   if name == 'typeptr':
        //       continue # dealt otherwise
        if field_name == "typeptr" {
            continue;
        }
        let align = field_size.min(std::mem::size_of::<usize>());
        if align > 0 {
            offset = (offset + align - 1) & !(align - 1);
        }
        let rank = immutable_ranks.get(field_name.as_str()).copied();
        entries.push((
            field_name.clone(),
            offset,
            field_size,
            field_type,
            rank.is_some(),
            rank.map(|r| r.is_quasi_immutable()).unwrap_or(false),
            flag,
        ));
        offset += field_size;
    }
    let max_align = fields
        .iter()
        .map(|(_, ty)| get_type_flag(ty).2)
        .filter(|s| *s > 0)
        .max()
        .unwrap_or(8);
    let item_size = if offset > 0 {
        (offset + max_align - 1) & !(max_align - 1)
    } else {
        0
    };
    // Path 2 mirror of the Path 1 `get_size_descr` seed — populates
    // `_cache_size[struct_key]` before the per-field loop so each
    // `gc_cache.get_field_descr` cache-miss-mint resolves
    // `parent_descr` to the size descr instead of `None` (`descr.py:238`).
    let size_descr_arc = {
        let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
        gc.get_size_descr(struct_key.clone(), item_size, 0, false)
    };
    let mut result = Vec::new();
    for (
        index_in_parent,
        (name, fld_offset, field_size, field_type, is_immutable, is_quasi_immutable, flag),
    ) in entries.iter().enumerate()
    {
        // `descr.py:435` field-walk convergence — same rationale as the
        // Path 1 arm above.  Runtime `build_object_descr_group` publishes
        // `PyreFieldDescr` Arcs inside `PyreSizeDescr.all_fielddescrs`;
        // walk that list first so struct-array population reuses the
        // runtime Arc, ensuring `cpu.interiorfielddescrof` per-tuple
        // identity matches PyPy's single FieldDescr-object-per-(STRUCT,
        // name) invariant.
        let mut walked: Option<std::sync::Arc<dyn majit_ir::descr::FieldDescr>> = None;
        if let Some(sd) = size_descr_arc.as_size_descr() {
            let needle = format!(".{}", name);
            for fd in sd.all_fielddescrs() {
                let stored = fd.field_name();
                if stored == *name || stored.ends_with(&needle) {
                    walked = Some(fd.clone());
                    break;
                }
            }
        }
        let fd: std::sync::Arc<dyn majit_ir::descr::FieldDescr> = match walked {
            Some(fd) => fd,
            None => {
                let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
                gc.get_field_descr(
                    struct_key.clone(),
                    name,
                    *fld_offset,
                    *field_size,
                    *field_type,
                    *is_immutable,
                    *is_quasi_immutable,
                    *flag,
                    index_in_parent,
                )
            }
        };
        let ifd = {
            let mut gc = majit_ir::descr::gc_cache().lock().unwrap();
            gc.get_interiorfield_descr(
                array_key.clone(),
                name.clone(),
                String::new(),
                array_descr.clone(),
                fd,
            )
        };
        ifd.set_index(index_in_parent as u32);
        result.push(ifd as majit_ir::descr::DescrRef);
    }
    (result, item_size)
}

/// RPython: `symbolic.get_array_token(ARRAY, tsc)[1]` — struct item_size.
///
/// Layout source priority:
/// 1. `cc.struct_layouts[struct_name].size` — actual layout
/// 2. Type-string heuristic fallback
fn compute_struct_size(cc: &CallControl, struct_name: &str) -> usize {
    // Path 1: actual layout from runtime (RPython: symbolic.get_size(STRUCT))
    if let Some(layout) = cc.struct_layouts.get(struct_name) {
        return layout.size;
    }
    // Path 2: heuristic fallback — RPython: symbolic always computes the full
    // struct size, even with nested structs. Nested struct sizes are looked up
    // recursively from struct_layouts.
    let fields = match cc.struct_fields.fields.get(struct_name) {
        Some(f) => f,
        None => return 0,
    };
    let mut offset: usize = 0;
    for (_, field_type_str) in fields.iter() {
        let field_size = if cc.is_known_struct(field_type_str) {
            // RPython: symbolic.get_field_token() uses actual nested struct size.
            cc.struct_layouts
                .get(field_type_str.as_str())
                .map(|l| l.size)
                .unwrap_or(std::mem::size_of::<usize>())
        } else {
            let (_, field_type, s) = get_type_flag(field_type_str);
            if field_type == majit_ir::value::Type::Void || s == 0 {
                continue;
            }
            s
        };
        let align = field_size.min(std::mem::size_of::<usize>());
        offset = (offset + align - 1) & !(align - 1);
        offset += field_size;
    }
    let max_align = fields
        .iter()
        .map(|(_, ty)| {
            if cc.is_known_struct(ty) {
                cc.struct_layouts
                    .get(ty.as_str())
                    .map(|l| l.size)
                    .unwrap_or(std::mem::size_of::<usize>())
                    .min(std::mem::size_of::<usize>())
            } else {
                get_type_flag(ty).2
            }
        })
        .filter(|s| *s > 0)
        .max()
        .unwrap_or(8);
    if offset > 0 {
        (offset + max_align - 1) & !(max_align - 1)
    } else {
        0
    }
}

/// RPython: `get_type_flag(TYPE)` (descr.py:241-254).
///
/// Returns (ArrayFlag, IR type, size in bytes).
/// The ArrayFlag encodes both category AND signedness, matching RPython:
/// - Ptr(gc) → FLAG_POINTER; Ptr(non-gc) → FLAG_UNSIGNED
/// - Struct → FLAG_STRUCT; Float → FLAG_FLOAT
/// - Bool/unsigned → FLAG_UNSIGNED; signed int → FLAG_SIGNED
fn get_type_flag(type_str: &str) -> (majit_ir::descr::ArrayFlag, majit_ir::value::Type, usize) {
    use majit_ir::descr::ArrayFlag;
    match type_str {
        // RPython: isinstance(TYPE, lltype.Ptr) and TYPE.TO._gckind == 'gc' → FLAG_POINTER
        s if s.starts_with('&')
            || s.starts_with("Box<")
            || s.starts_with("Arc<")
            || s.starts_with("Rc<")
            || s.starts_with("Vec<")
            || s.starts_with("Option<")
            || s == "String" =>
        {
            (ArrayFlag::Pointer, majit_ir::value::Type::Ref, 8)
        }
        // RPython: TYPE is lltype.Float → FLAG_FLOAT
        "f64" => (ArrayFlag::Float, majit_ir::value::Type::Float, 8),
        "f32" => (ArrayFlag::Float, majit_ir::value::Type::Float, 4),
        // RPython: rffi.cast(TYPE, -1) == -1 → FLAG_SIGNED
        "i64" | "isize" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 8),
        "i32" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 4),
        "i16" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 2),
        "i8" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 1),
        // RPython: Bool → FLAG_UNSIGNED; unsigned number → FLAG_UNSIGNED
        "u64" | "usize" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 8),
        "u32" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 4),
        "u16" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 2),
        "u8" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 1),
        "bool" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 1),
        // RPython: Void fields are skipped
        "()" => (ArrayFlag::Void, majit_ir::value::Type::Void, 0),
        // Unknown type — treat as GC pointer (conservative)
        _ => (ArrayFlag::Pointer, majit_ir::value::Type::Ref, 8),
    }
}

/// RPython: `RaiseAnalyzer.analyze_simple_operation(op)` (canraise.py:14-17).
///
/// ```python
/// canraise = LL_OPERATIONS[op.opname].canraise
/// return bool(canraise) and canraise != (self.ignore_exact_class,)
/// ```
///
/// Returns true if the operation itself (not counting transitive calls)
/// can raise an exception. When `ignore_memoryerror` is true, operations
/// that can only raise MemoryError are treated as non-raising.
fn op_can_raise(op: &OpKind) -> RaiseClass {
    // RPython canraise.py:14-18:
    //   canraise = LL_OPERATIONS[op.opname].canraise
    //   return bool(canraise) and canraise != (self.ignore_exact_class,)
    //
    // Model the tri-state directly:
    //   ()                  -> No
    //   (MemoryError,)      -> MemoryErrorOnly
    //   anything else truthy -> Yes
    match op {
        // ── Known non-raising ops (canraise = ()) ─────────────────
        // RPython LL: getfield_gc, setfield_gc → cannot raise
        OpKind::FieldRead { .. } | OpKind::FieldWrite { .. } => RaiseClass::No,
        // RPython LL: getarrayitem_gc, setarrayitem_gc → cannot raise
        OpKind::ArrayRead { .. } | OpKind::ArrayWrite { .. } => RaiseClass::No,
        // RPython LL: getinteriorfield_gc, setinteriorfield_gc → cannot raise
        OpKind::InteriorFieldRead { .. } | OpKind::InteriorFieldWrite { .. } => RaiseClass::No,
        // RPython LL: int_add, int_sub, int_lt, int_and, etc → cannot raise
        // (non-ovf, non-div arithmetic)
        OpKind::BinOp { op, .. }
            if !op.contains("div")
                && !op.contains("mod")
                && !op.contains("rem")
                && !op.contains("ovf") =>
        {
            RaiseClass::No
        }
        // RPython LL: int_neg, bool_not → cannot raise
        OpKind::UnaryOp { op, .. } if !op.contains("ovf") => RaiseClass::No,
        // RPython LL: same_as, cast_*, hint → cannot raise
        OpKind::Input { .. }
        | OpKind::ConstInt(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstFloat(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstRefAddr(_) => RaiseClass::No,
        // JIT-specific ops that cannot raise
        OpKind::GuardTrue { .. }
        | OpKind::GuardFalse { .. }
        | OpKind::GuardValue { .. }
        | OpKind::JitDebug { .. }
        | OpKind::AssertGreen { .. }
        | OpKind::CurrentTraceLength
        | OpKind::IsConstant { .. }
        | OpKind::IsVirtual { .. }
        | OpKind::RecordKnownResult { .. }
        // jtransform.py:901-903 — `record_quasiimmut_field` is pure bookkeeping
        // that the metainterp converts into a guard; cannot raise.
        | OpKind::RecordQuasiImmutField { .. }
        | OpKind::Live
        // jtransform.py:1707 `jit_merge_point` / :1718 `loop_header` — pure
        // markers consumed by the metainterp; cannot raise.
        | OpKind::JitMergePoint { .. }
        | OpKind::LoopHeader { .. } => RaiseClass::No,
        // Virtualizable field/array access (from boxes, no heap) → cannot raise
        OpKind::VableFieldRead { .. }
        | OpKind::VableFieldWrite { .. }
        | OpKind::VableArrayRead { .. }
        | OpKind::VableArrayWrite { .. } => RaiseClass::No,
        // Post-jtransform call ops: raise is determined by their descriptor,
        // not by op_can_raise. These are not "simple operations" in RPython
        // terms — they're handled by analyze() → analyze_direct_call.
        OpKind::CallResidual { .. }
        | OpKind::CallElidable { .. }
        | OpKind::CallMayForce { .. }
        | OpKind::InlineCall { .. }
        | OpKind::RecursiveCall { .. }
        | OpKind::ConditionalCall { .. }
        | OpKind::ConditionalCallValue { .. } => RaiseClass::No,

        // ── Known raising ops ─────────────────────────────────────
        // RPython LL: jit_force_virtualizable has `canrun=True`, not
        // `canraise`; effect classification handles its special meaning.
        OpKind::VableForce { .. } => RaiseClass::No,
        // RPython LL: int_floordiv, int_mod → canraise = (ZeroDivisionError,)
        OpKind::BinOp { .. } => RaiseClass::Yes, // div/mod/rem/ovf (others matched above)
        // RPython LL: int_neg_ovf → canraise = (OverflowError,)
        OpKind::UnaryOp { .. } => RaiseClass::Yes, // ovf (others matched above)

        // ── Calls handled by analyze() dispatch, not here ─────────
        // RPython: Call ops dispatch to analyze_direct_call/analyze_external_call.
        // op_can_raise is only for "simple operations" (non-call).
        // But if we see a Call here (shouldn't happen in normal flow),
        // be conservative.
        OpKind::Call { .. } => RaiseClass::Yes,

        // ── vtable entry extraction: pure memory load, no raise ──
        // RPython: op.args[0] in indirect_call is a plain Variable,
        // the address extraction itself has no raising analogue.
        OpKind::VtableMethodPtr { .. } => RaiseClass::No,
        // ── indirect_call canraise comes from the family's calldescr ──
        // (analyze() dispatch, not here). Not yet emitted; arm reserved
        // for Phase B.
        OpKind::IndirectCall { .. } => RaiseClass::Yes,

        // ── Abort placeholders: canraise.py:18 → True (conservative) ─
        // RPython: log.WARNING("Unknown operation: %s" % op.opname)
        //          return True
        OpKind::Abort { .. } => RaiseClass::Yes,
        // RPython `newtuple` is a `PureOperation` (`operation.py:542`);
        // pure tuple construction cannot raise.
        OpKind::NewTuple { .. } => RaiseClass::No,
        // `LoadStatic` reads a `static` declaration's address — a
        // compile-time constant.  `LOAD_GLOBAL` analog
        // (`flowspace/flowcontext.py:1098`); cannot raise.
        OpKind::LoadStatic { .. } => RaiseClass::No,
    }
}

fn exceptblock_is_reraise_of_caught_exception(graph: &FunctionGraph) -> bool {
    // Read the exceptblock's `evalue` slot from the unfiltered
    // `inputargs`.  RPython `flowspace/model.py:Variable` is the
    // operand identity, so the UnionFind families key on Variable
    // directly.
    let exceptblock_args = &graph.block(graph.exceptblock).inputargs;
    let Some(except_value) = exceptblock_args.get(1).cloned() else {
        return false;
    };

    let mut families = crate::tool::algo::unionfind::UnionFind::<
        crate::flowspace::model::Variable,
        (),
    >::new(|_| ());
    for block in &graph.blocks {
        for link in &block.exits {
            // Zip link args against the target block's raw
            // `inputargs` (Variable identities, positional per
            // `flowspace/model.py:244 renamevariables`).
            let target_inputargs = &graph.block(link.target).inputargs;
            for (arg, target_arg) in link.args.iter().zip(target_inputargs.iter()) {
                if let Some(value) = arg.as_variable() {
                    families.union(value.clone(), target_arg.clone());
                }
            }
        }
    }
    let except_rep = families.find_rep(except_value);
    graph
        .blocks
        .iter()
        .flat_map(|block| block.exits.iter())
        .filter_map(|link| {
            link.last_exc_value
                .as_ref()
                .and_then(|arg| arg.as_variable())
        })
        .any(|value| families.find_rep(value.clone()) == except_rep)
}

/// Map ValueType to a small integer for array descriptor indexing.
fn value_type_discriminant(ty: &crate::model::ValueType) -> u8 {
    use crate::model::ValueType;
    match ty {
        // ValueType::Bool maps to the same array-descriptor bucket as
        // Int — RPython's `cpu.arraydescrof(ARRAY)` records `BOOL_TYPE`
        // and `INT_TYPE` under the same `'int'` kind for descriptor
        // indexing (`lltypesystem/lloperation.py:108 getkind`).
        ValueType::Int | ValueType::Unsigned | ValueType::Bool => 0,
        ValueType::Ref(_) => 1,
        ValueType::Float => 2,
        ValueType::Void => 3,
        ValueType::State => 4,
        ValueType::Unknown => 5,
    }
}

// ── Builtin call effect tables ──────────────────────────────────
//
// RPython equivalent: effect classification in `call.py::getcalldescr()`
// combined with the builtin function tables.
// These tables map known function targets to their effect info,
// used by `jtransform::classify_call()` as a fallback when the
// call is not in the explicit `call_effects` config.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallTargetPattern {
    FunctionPath(&'static [&'static str]),
}

impl CallTargetPattern {
    fn matches(self, target: &CallTarget) -> bool {
        match (self, target) {
            (CallTargetPattern::FunctionPath(path), CallTarget::FunctionPath { segments }) => {
                segments.iter().map(String::as_str).eq(path.iter().copied())
            }
            _ => false,
        }
    }
}

struct CallDescriptorEntry {
    targets: &'static [CallTargetPattern],
    extraeffect: ExtraEffect,
    oopspecindex: OopSpecIndex,
}

impl CallDescriptorEntry {
    fn get_extra_info(&self) -> EffectInfo {
        EffectInfo::new(self.extraeffect, self.oopspecindex)
    }
}

// ── Builtin call descriptor table ──
//
// RPython effectinfo.py + call.py parity: pre-classified call targets.
// The codewriter matches function names to determine effect category
// and oopspec index without graph-level analysis.

const INT_ARITH_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_int_add"]),
    CallTargetPattern::FunctionPath(&["w_int_sub"]),
    CallTargetPattern::FunctionPath(&["w_int_mul"]),
    CallTargetPattern::FunctionPath(&["int_add"]),
    CallTargetPattern::FunctionPath(&["int_sub"]),
    CallTargetPattern::FunctionPath(&["int_mul"]),
    CallTargetPattern::FunctionPath(&["int_bitand"]),
    CallTargetPattern::FunctionPath(&["int_bitor"]),
    CallTargetPattern::FunctionPath(&["int_bitxor"]),
    // Qualified paths (annotator uses these for type inference).
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_add"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_sub"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_mul"]),
];

const INT_CMP_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["int_lt"]),
    CallTargetPattern::FunctionPath(&["int_le"]),
    CallTargetPattern::FunctionPath(&["int_gt"]),
    CallTargetPattern::FunctionPath(&["int_ge"]),
    CallTargetPattern::FunctionPath(&["int_eq"]),
    CallTargetPattern::FunctionPath(&["int_ne"]),
];

const FLOAT_ARITH_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_float_add"]),
    CallTargetPattern::FunctionPath(&["w_float_sub"]),
    CallTargetPattern::FunctionPath(&["float_add"]),
    CallTargetPattern::FunctionPath(&["float_sub"]),
    CallTargetPattern::FunctionPath(&["float_mul"]),
    CallTargetPattern::FunctionPath(&["float_truediv"]),
];

const FLOAT_CMP_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["float_lt"]),
    CallTargetPattern::FunctionPath(&["float_le"]),
    CallTargetPattern::FunctionPath(&["float_gt"]),
    CallTargetPattern::FunctionPath(&["float_ge"]),
    CallTargetPattern::FunctionPath(&["float_eq"]),
    CallTargetPattern::FunctionPath(&["float_ne"]),
];

// effectinfo.py: EF_ELIDABLE_CAN_RAISE — may raise (e.g. ZeroDivisionError)
// int_floordiv and int_mod have distinct oopspec indices (IntPyDiv vs IntPyMod)
// because intbounds.rs optimizes them differently.
const INT_FLOORDIV_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["int_floordiv"])];

const INT_MOD_TARGETS: &[CallTargetPattern] = &[CallTargetPattern::FunctionPath(&["int_mod"])];

// RPython `jtransform.py:587-588` — `_do_builtin_call` re-routes
// `cast_uint_to_float` / `cast_float_to_uint` to support helpers
// (`support.py:274 _ll_1_cast_*`).  Cannot raise (NaN/inf are
// caller-filtered); elidable because the conversion is pure given
// the same input bit pattern.  No upstream `OopSpecIndex` — plain
// support helpers, not `OS_*` oopspec calls.
const CAST_UINT_TO_FLOAT_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["cast_uint_to_float"])];

const CAST_FLOAT_TO_UINT_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["cast_float_to_uint"])];

const FLOAT_DIV_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["float_floordiv"]),
    CallTargetPattern::FunctionPath(&["float_mod"]),
];

const INT_SHIFT_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["int_lshift"]),
    CallTargetPattern::FunctionPath(&["int_rshift"]),
];

const INT_POW_TARGETS: &[CallTargetPattern] = &[CallTargetPattern::FunctionPath(&["int_pow"])];

// effectinfo.py: OS_STR_CONCAT etc. — string operations with oopspec
const STR_CONCAT_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["str_concat"]),
    CallTargetPattern::FunctionPath(&["jit_str_concat"]),
];

const STR_CMP_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["jit_str_compare"])];

// effectinfo.py: list operations (may raise IndexError)
const LIST_GETITEM_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["jit_list_getitem"]),
    CallTargetPattern::FunctionPath(&["w_list_getitem"]),
];

const LIST_SETITEM_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["jit_list_setitem"])];

const LIST_APPEND_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["jit_list_append"])];

// effectinfo.py: tuple access (elidable, cannot raise for valid index)
const TUPLE_GETITEM_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["jit_tuple_getitem"]),
    CallTargetPattern::FunctionPath(&["w_tuple_getitem"]),
];

// effectinfo.py: constructor-like (cannot raise, elidable)
const INT_NEW_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_int_new"]),
    CallTargetPattern::FunctionPath(&["jit_w_int_new"]),
];

const FLOAT_NEW_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_float_new"]),
    CallTargetPattern::FunctionPath(&["jit_w_float_new"]),
];

const BOOL_FROM_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["w_bool_from"])];

const CALL_DESCRIPTOR_TABLE: &[CallDescriptorEntry] = &[
    // ── Pure arithmetic (elidable, cannot raise) ──
    CallDescriptorEntry {
        targets: INT_ARITH_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: INT_CMP_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_ARITH_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_CMP_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    // ── Elidable but may raise (ZeroDivisionError, OverflowError) ──
    CallDescriptorEntry {
        targets: INT_FLOORDIV_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::IntPyDiv,
    },
    CallDescriptorEntry {
        targets: INT_MOD_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::IntPyMod,
    },
    // RPython `jtransform.py:587-588` `_do_builtin_call` casts —
    // unsigned-domain conversion helpers.  Cannot raise; elidable.
    CallDescriptorEntry {
        targets: CAST_UINT_TO_FLOAT_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: CAST_FLOAT_TO_UINT_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_DIV_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: INT_SHIFT_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: INT_POW_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    // ── String operations with oopspec ──
    CallDescriptorEntry {
        targets: STR_CONCAT_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::StrConcat,
    },
    CallDescriptorEntry {
        targets: STR_CMP_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::StrCmp,
    },
    // ── List operations (may raise, side effects) ──
    CallDescriptorEntry {
        targets: LIST_GETITEM_TARGETS,
        extraeffect: ExtraEffect::CanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: LIST_SETITEM_TARGETS,
        extraeffect: ExtraEffect::CanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: LIST_APPEND_TARGETS,
        extraeffect: ExtraEffect::CanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    // ── Tuple access (elidable for valid indices) ──
    CallDescriptorEntry {
        targets: TUPLE_GETITEM_TARGETS,
        extraeffect: ExtraEffect::ElidableCanRaise,
        oopspecindex: OopSpecIndex::None,
    },
    // ── Allocating constructors (cannot raise, but NOT elidable) ──
    // w_int_new/w_float_new allocate fresh objects — CSE would merge
    // distinct allocations, breaking Python identity (is).
    CallDescriptorEntry {
        targets: INT_NEW_TARGETS,
        extraeffect: ExtraEffect::CannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_NEW_TARGETS,
        extraeffect: ExtraEffect::CannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
    // w_bool_from returns singletons (True/False) — safe to CSE.
    CallDescriptorEntry {
        targets: BOOL_FROM_TARGETS,
        extraeffect: ExtraEffect::ElidableCannotRaise,
        oopspecindex: OopSpecIndex::None,
    },
];

fn matches_any(target: &CallTarget, patterns: &[CallTargetPattern]) -> bool {
    patterns
        .iter()
        .copied()
        .any(|pattern| pattern.matches(target))
}

/// Check if a call target is a known int arithmetic function.
/// Used by annotate pass for type inference.
pub fn is_int_arithmetic_target(target: &CallTarget) -> bool {
    matches_any(target, INT_ARITH_TARGETS)
}

/// Look up a call target in the builtin effect table.
///
/// RPython: part of `CallControl.getcalldescr()` — returns effect info
/// for known functions like `w_int_add` (elidable), `w_float_sub` (elidable).
pub fn describe_call(target: &CallTarget) -> Option<CallDescriptor> {
    CALL_DESCRIPTOR_TABLE
        .iter()
        .find(|entry| matches_any(target, entry.targets))
        .map(|entry| CallDescriptor::known(entry.get_extra_info()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ExitSwitch, FunctionGraph, Link, LinkArg, ValueType, exception_exitcase};

    /// Synthetic `OpKind::Call` wrapper — mirrors RPython test_jtransform
    /// helpers that pass a pre-built `SpaceOperation('direct_call', ...)`
    /// into `guess_call_kind` / `graphs_from`.
    fn direct_call_op(target: CallTarget) -> SpaceOperation {
        SpaceOperation {
            result: None,
            kind: OpKind::Call {
                target,
                args: vec![],
                result_ty: ValueType::Void,
            },
        }
    }

    /// Synthetic `OpKind::IndirectCall` wrapper — mirrors RPython test
    /// construction of `SpaceOperation('indirect_call', [..., c_graphs])`.
    fn indirect_call_op(graphs: Option<Vec<CallPath>>) -> SpaceOperation {
        SpaceOperation {
            result: None,
            kind: OpKind::IndirectCall {
                funcptr: crate::flowspace::model::Variable::new(),
                args: vec![],
                graphs,
                result_ty: ValueType::Void,
            },
        }
    }

    #[test]
    fn guess_call_kind_function_path() {
        let mut cc = CallControl::new();
        let graph = FunctionGraph::new("opcode_load_fast");
        let path = CallPath::from_segments(["opcode_load_fast"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["opcode_load_fast"]);
        assert_eq!(
            cc.guess_call_kind(&direct_call_op(target)),
            CallKind::Regular
        );

        let unknown = CallTarget::function_path(["unknown_function"]);
        assert_eq!(
            cc.guess_call_kind(&direct_call_op(unknown)),
            CallKind::Residual
        );
    }

    /// GraphId converges on graph-object identity (`call.py:29`), not the
    /// CallPath: two alias spellings of one source graph (same `graph.name`)
    /// share a single GraphId, so a mark applied through one alias is
    /// observed through the other; a distinct source graph stays separate.
    #[test]
    fn graph_id_converges_aliases_of_one_source_graph() {
        let mut cc = CallControl::new();
        // Two alias spellings of one source graph "canonical::source".
        let alias_a = CallPath::from_segments(["alias_a"]);
        let alias_b = CallPath::from_segments(["alias_b"]);
        cc.register_function_graph(alias_a.clone(), FunctionGraph::new("canonical::source"));
        cc.register_function_graph(alias_b, FunctionGraph::new("canonical::source"));
        // A third path is a genuinely different source graph.
        let other = CallPath::from_segments(["other_alias"]);
        cc.register_function_graph(other, FunctionGraph::new("other::source"));

        // Mark the oopspec through alias_a only.
        cc.mark_oopspec(alias_a, "list.append(l, v)".to_string());

        // alias_b observes it — the two aliases share one GraphId.
        assert_eq!(
            cc.get_oopspec(&CallTarget::function_path(["alias_b"])),
            Some("list.append(l, v)"),
            "aliases of one source graph must share a GraphId",
        );
        // The distinct source graph does not.
        assert_eq!(
            cc.get_oopspec(&CallTarget::function_path(["other_alias"])),
            None,
            "a distinct source graph must keep a separate GraphId",
        );
    }

    /// Two impls of one trait carry the SAME bare `graph.name` (production
    /// stamps the bare method name on impl-method graphs — `PyFrame` and
    /// `MIFrame` both name their graph `push_value`). The GraphId identity
    /// key is `(owner_root, name)`, so `owner_root` keeps them on distinct
    /// GraphIds: a mark on one impl must NOT leak to the other. Guards
    /// against keying identity on the bare name alone.
    #[test]
    fn graph_id_separates_same_named_methods_of_distinct_impls() {
        let mut cc = CallControl::new();
        // Both graphs share bare name "push_value"; only owner_root differs,
        // mirroring what `front/ast.rs` stamps for impl methods.
        cc.register_trait_method(
            "push_value",
            Some("Stepper"),
            "PyFrame",
            FunctionGraph::new("push_value").with_owner_root("PyFrame"),
        );
        cc.register_trait_method(
            "push_value",
            Some("Stepper"),
            "MIFrame",
            FunctionGraph::new("push_value").with_owner_root("MIFrame"),
        );

        // Mark the oopspec on PyFrame's method only (probing GraphId
        // identity — the spec string's semantics are immaterial here).
        cc.mark_oopspec(
            CallPath::from_segments(["PyFrame", "push_value"]),
            "stepper.push(f, v)".to_string(),
        );

        assert_eq!(
            cc.get_oopspec(&CallTarget::function_path(["PyFrame", "push_value"])),
            Some("stepper.push(f, v)"),
        );
        assert_eq!(
            cc.get_oopspec(&CallTarget::function_path(["MIFrame", "push_value"])),
            None,
            "same-named methods of distinct impls must keep separate GraphIds",
        );
    }

    #[test]
    fn get_jitcode_shell_falls_back_to_symbolic_fnaddr() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["opcode_load_fast"]);
        cc.register_function_graph(path.clone(), FunctionGraph::new("opcode_load_fast"));

        let jitcode = cc.get_jitcode(&path);

        assert_eq!(jitcode.fnaddr, symbolic_fnaddr_for_path(&path));
    }

    #[test]
    fn get_jitcode_shell_uses_registered_fnaddr() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["helpers", "opaque_call"]);
        cc.register_function_graph(path.clone(), FunctionGraph::new("opaque_call"));
        cc.register_function_fnaddr(path.clone(), 0xfeed_beef);

        let jitcode = cc.get_jitcode(&path);

        assert_eq!(jitcode.fnaddr, 0xfeed_beef);
    }

    #[test]
    fn register_macro_helper_trace_fnaddr_binds_canonical_and_crate_aliases() {
        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr("testcrate::helpers::opaque_call", 0x1234);

        assert_eq!(
            cc.fnaddr_for_target(&CallTarget::function_path(["helpers", "opaque_call"])),
            0x1234
        );
        assert_eq!(
            cc.fnaddr_for_target(&CallTarget::function_path([
                "crate",
                "helpers",
                "opaque_call"
            ])),
            0x1234
        );
    }

    #[test]
    fn register_macro_impl_helper_qualifies_bare_type_with_module_prefix() {
        // `impl Adder { fn add() }` at `mod impl_module` — macro emits
        // `impl_type_as_written = "Adder"` (bare), and
        // `module_path_with_crate = "testcrate::impl_module"`.  The
        // codewriter must prepend the module prefix so the canonical
        // CallPath matches the parser's `qualify_type_name("Adder",
        // "impl_module") = "impl_module::Adder"` result
        // (front/ast.rs:106).
        let mut cc = CallControl::new();
        cc.register_macro_impl_helper_trace_fnaddr(
            "testcrate::impl_module",
            "Adder",
            "add",
            0xfeed_beef,
        );

        assert_eq!(
            cc.fnaddr_for_target(&CallTarget::function_path(["impl_module", "Adder", "add"])),
            0xfeed_beef
        );
    }

    #[test]
    fn register_macro_impl_helper_keeps_qualified_type_unchanged() {
        // `impl a::Foo { fn bar() }` — already-qualified type must not
        // get the module prefix prepended (front/ast.rs:107 returns
        // bare-as-is when it contains `::`).  The canonical path
        // matches `CallPath::for_impl_method("a::Foo", "bar")`.
        let mut cc = CallControl::new();
        cc.register_macro_impl_helper_trace_fnaddr(
            "testcrate::other_module",
            "a::Foo",
            "bar",
            0x1234,
        );

        assert_eq!(
            cc.fnaddr_for_target(&CallTarget::function_path(["a", "Foo", "bar"])),
            0x1234
        );
        // Must NOT also be bound under the module-prefixed form.
        assert_ne!(
            cc.fnaddr_for_target(&CallTarget::function_path([
                "other_module",
                "a",
                "Foo",
                "bar"
            ])),
            0x1234,
        );
    }

    #[test]
    fn register_macro_impl_helper_at_crate_root_has_no_prefix() {
        // `#[jit_module]` at crate root: `module_path!()` is just the
        // crate name, so after stripping the crate there's no module
        // prefix; bare impl_type stays bare — matching the parser's
        // `prefix = ""` at crate root (parse.rs:314-318).
        let mut cc = CallControl::new();
        cc.register_macro_impl_helper_trace_fnaddr("testcrate", "Adder", "add", 0xabcd);

        assert_eq!(
            cc.fnaddr_for_target(&CallTarget::function_path(["Adder", "add"])),
            0xabcd
        );
    }

    #[test]
    fn register_macro_helper_free_fn_path_is_unchanged_by_impl_alias_split() {
        // Regression: the macro helper entry point no longer tries to
        // heuristically collapse `module::sub::fn_name` into a 2-segment
        // form, since that is indistinguishable from the qualified
        // impl-type case.  Free-fn paths bind exactly the canonical
        // strip-crate and `crate::...` aliases — nothing else.
        let mut cc = CallControl::new();
        cc.register_macro_helper_trace_fnaddr("testcrate::helpers::sub::bar", 0x4242);

        assert_eq!(
            cc.fnaddr_for_target(&CallTarget::function_path(["helpers", "sub", "bar"])),
            0x4242
        );
        assert_ne!(
            cc.fnaddr_for_target(&CallTarget::function_path(["sub", "bar"])),
            0x4242,
        );
    }

    #[test]
    fn guess_call_kind_portal() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["portal_runner"]);
        cc.mark_portal(path);

        let target = CallTarget::function_path(["portal_runner"]);
        assert_eq!(
            cc.guess_call_kind(&direct_call_op(target)),
            CallKind::Recursive
        );
    }

    #[test]
    fn guess_call_kind_builtin() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["w_int_add"]);
        cc.mark_builtin(path);

        let target = CallTarget::function_path(["w_int_add"]);
        assert_eq!(
            cc.guess_call_kind(&direct_call_op(target)),
            CallKind::Builtin
        );
    }

    #[test]
    fn resolve_method_unique_impl() {
        let mut cc = CallControl::new();
        let graph = FunctionGraph::new("PyFrame::load_local_value");
        cc.register_trait_method(
            "load_local_value",
            Some("LocalOpcodeHandler"),
            "PyFrame",
            graph,
        );

        // TODO: receiver-agnostic unique-impl fallback in
        // `resolve_method` enables BFS to monomorphise
        // generic-receiver call sites at trait dispatch.  Retired
        // when the annotator publishes classdef hints before BFS.
        assert!(
            cc.resolve_method("load_local_value", Some("handler"), None)
                .is_some()
        );
        assert!(
            cc.resolve_method("load_local_value", Some("H"), None)
                .is_some()
        );
        assert!(cc.resolve_method("load_local_value", None, None).is_some());
    }

    #[test]
    fn resolve_method_multiple_impls() {
        let mut cc = CallControl::new();
        cc.register_trait_method(
            "push_value",
            Some("LocalOpcodeHandler"),
            "PyFrame",
            FunctionGraph::new("PyFrame::push_value"),
        );
        cc.register_trait_method(
            "push_value",
            Some("LocalOpcodeHandler"),
            "MIFrame",
            FunctionGraph::new("MIFrame::push_value"),
        );

        // Concrete receiver — resolves to specific impl
        assert!(
            cc.resolve_method("push_value", Some("PyFrame"), None)
                .is_some()
        );

        // Generic receiver — can't resolve uniquely
        assert!(
            cc.resolve_method("push_value", Some("handler"), None)
                .is_none()
        );
        assert!(cc.resolve_method("push_value", Some("H"), None).is_none());
    }

    #[test]
    fn resolve_method_resolved_path_picks_registered_impl() {
        let mut cc = CallControl::new();
        cc.register_trait_method(
            "push_value",
            Some("LocalOpcodeHandler"),
            "PyFrame",
            FunctionGraph::new("PyFrame::push_value"),
        );
        cc.register_trait_method(
            "push_value",
            Some("LocalOpcodeHandler"),
            "MIFrame",
            FunctionGraph::new("MIFrame::push_value"),
        );

        assert!(cc.resolve_method("push_value", Some("H"), None).is_none());

        let pyframe_path = CallPath::for_impl_method("PyFrame", "push_value");
        let pyframe = cc.resolve_method("push_value", Some("H"), Some(&pyframe_path));
        assert!(pyframe.is_some(), "resolved_path should resolve PyFrame");
        assert_eq!(pyframe.unwrap().name, "PyFrame::push_value");

        let miframe_path = CallPath::for_impl_method("MIFrame", "push_value");
        let miframe = cc.resolve_method("push_value", Some("H"), Some(&miframe_path));
        assert!(miframe.is_some(), "resolved_path should resolve MIFrame");
        assert_eq!(miframe.unwrap().name, "MIFrame::push_value");
    }

    #[test]
    fn resolve_method_falls_back_when_resolved_path_miss() {
        let mut cc = CallControl::new();
        cc.register_trait_method(
            "load_local_value",
            Some("LocalOpcodeHandler"),
            "PyFrame",
            FunctionGraph::new("PyFrame::load_local_value"),
        );

        let unknown_path = CallPath::for_impl_method("Unknown", "load_local_value");
        assert!(
            cc.resolve_method("load_local_value", Some("handler"), Some(&unknown_path))
                .is_some()
        );
    }

    // ── getcalldescr tests ───────────────────────────���──────────────
    /// Helper: create a FunctionGraph with just a return.
    fn simple_graph(name: &str) -> FunctionGraph {
        let mut g = FunctionGraph::new(name);
        g.set_return(g.startblock, None);
        g
    }

    fn register_int_result_graph(cc: &mut CallControl, path: CallPath, graph: FunctionGraph) {
        cc.register_function_graph(path, graph.with_return_type("i64"));
    }

    /// Helper: create a FunctionGraph whose entry block routes to the
    /// canonical exceptblock, matching upstream's Link(..., exceptblock)
    /// shape for unconditional raise sites.
    fn raising_graph(name: &str) -> FunctionGraph {
        let mut g = FunctionGraph::new(name);
        g.set_raise(g.startblock, "error");
        g
    }

    /// Synthetic graph that only re-raises a previously-caught exception.
    ///
    /// This mirrors the special case in
    /// `canraise.py:27-41 analyze_exceptblock_in_graph`: the graph itself
    /// should not be treated as the origin of the exception.
    fn reraise_only_graph(name: &str) -> FunctionGraph {
        let mut g = FunctionGraph::new(name);
        let entry = g.startblock;
        let continuation = g.create_block();
        let continuation_arg_var = g.alloc_value_var();
        g.push_inputarg_var(continuation, continuation_arg_var.clone());
        let last_exception_var = g.alloc_value_var();
        let last_exc_value_var = g.alloc_value_var();
        let normal_link =
            Link::from_variables(&g, vec![continuation_arg_var.clone()], continuation, None);
        let exc_link = Link::from_variables(
            &g,
            vec![last_exception_var.clone(), last_exc_value_var.clone()],
            g.exceptblock,
            Some(exception_exitcase()),
        )
        .extravars(
            Some(LinkArg::Value(last_exception_var)),
            Some(LinkArg::Value(last_exc_value_var)),
        );
        g.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![normal_link, exc_link],
        );
        g.set_return(continuation, Some(continuation_arg_var));
        g
    }

    #[test]
    fn test_getcalldescr_cannot_raise() {
        // A simple function with no Abort → CannotRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["pure_add"]);
        register_int_result_graph(&mut cc, path.clone(), simple_graph("pure_add"));
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["pure_add"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
        assert_eq!(descriptor.extra_info.extraeffect, ExtraEffect::CannotRaise);
        assert!(!descriptor.extra_info.can_invalidate);
    }

    #[test]
    fn test_getcalldescr_can_raise() {
        // A function with Abort terminator → CanRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["failing_func"]);
        cc.register_function_graph(path.clone(), raising_graph("failing_func"));
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["failing_func"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
        assert_eq!(descriptor.extra_info.extraeffect, ExtraEffect::CanRaise);
    }

    #[test]
    fn test_getcalldescr_elidable() {
        // An elidable function that cannot raise → ElidableCannotRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["pure_lookup"]);
        register_int_result_graph(&mut cc, path.clone(), simple_graph("pure_lookup"));
        cc.mark_elidable(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["pure_lookup"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
        assert_eq!(
            descriptor.extra_info.extraeffect,
            ExtraEffect::ElidableCannotRaise
        );
    }

    #[test]
    fn test_getcalldescr_elidable_can_raise() {
        // An elidable function that CAN raise → ElidableCanRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["elidable_raiser"]);
        register_int_result_graph(&mut cc, path.clone(), raising_graph("elidable_raiser"));
        cc.mark_elidable(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["elidable_raiser"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
        assert_eq!(
            descriptor.extra_info.extraeffect,
            ExtraEffect::ElidableCanRaise
        );
    }

    #[test]
    fn test_getcalldescr_loopinvariant() {
        // A loop-invariant function → LoopInvariant.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["get_config"]);
        register_int_result_graph(&mut cc, path.clone(), simple_graph("get_config"));
        cc.mark_loopinvariant(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["get_config"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
        assert_eq!(
            descriptor.extra_info.extraeffect,
            ExtraEffect::LoopInvariant
        );
    }

    #[test]
    fn test_getcalldescr_forces_virtualizable() {
        // A function with VableForce → ForcesVirtualOrVirtualizable.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("forcer");
        let frame_var = graph.alloc_value_var();
        graph.push_inputarg_var(graph.startblock, frame_var.clone());
        graph.push_op_var(
            graph.startblock,
            OpKind::VableForce { base: frame_var },
            false,
        );
        graph.set_return(graph.startblock, None);
        let path = CallPath::from_segments(["forcer"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["forcer"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            vec![Type::Ref],
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
        assert_eq!(
            descriptor.extra_info.extraeffect,
            ExtraEffect::ForcesVirtualOrVirtualizable
        );
    }

    #[test]
    fn test_getcalldescr_extraeffect_override() {
        // When extraeffect is provided, it overrides the analyzers.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["func"]);
        register_int_result_graph(&mut cc, path, simple_graph("func"));
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["func"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            Some(ExtraEffect::ElidableCannotRaise),
            &mut cache,
            None,
        );
        assert_eq!(
            descriptor.extra_info.extraeffect,
            ExtraEffect::ElidableCannotRaise
        );
    }

    #[test]
    fn test_getcalldescr_transitive_can_raise() {
        // A function that calls another function that raises → CanRaise.
        let mut cc = CallControl::new();

        // callee: raises
        let callee_path = CallPath::from_segments(["callee"]);
        cc.register_function_graph(callee_path, raising_graph("callee"));

        // caller: calls callee (no Abort itself)
        let mut caller = FunctionGraph::new("caller");
        caller.push_op_var(
            caller.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: Vec::new(),
                result_ty: ValueType::Void,
            },
            false,
        );
        caller.set_return(caller.startblock, None);
        let caller_path = CallPath::from_segments(["caller"]);
        cc.register_function_graph(caller_path, caller);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["caller"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
        assert_eq!(descriptor.extra_info.extraeffect, ExtraEffect::CanRaise);
    }

    #[test]
    fn test_getcalldescr_unknown_target_can_raise() {
        // Unknown target (no graph) treated as external call.
        // RPython: RandomEffectsAnalyzer returns False for external calls
        // (only True if random_effects_on_gcobjs). RaiseAnalyzer returns
        // True (top_result) for unknown graphs → CanRaise.
        let cc = CallControl::new();
        let target = CallTarget::function_path(["unknown_extern"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
        assert_eq!(descriptor.extra_info.extraeffect, ExtraEffect::CanRaise);
        // RandomEffects is false, QuasiImmut is false → can_invalidate is false.
        assert!(!descriptor.extra_info.can_invalidate);
    }

    #[test]
    fn test_getcalldescr_readwrite_effects() {
        // A function with FieldRead/FieldWrite → bitsets populated.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("accessor");
        let base_var = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: base_var.clone(),
                field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldWrite {
                base: base_var.clone(),
                field: crate::model::FieldDescriptor::new("y", Some("Point".into())),
                value: base_var.clone(), // dummy
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        let path = CallPath::from_segments(["accessor"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["accessor"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
        // Should have non-empty bitsets for field reads and writes.
        assert!(
            descriptor
                .extra_info
                .readonly_descrs_fields
                .as_ref()
                .is_some_and(|bs| bs.iter().any(|&b| b != 0)),
        );
        assert!(
            descriptor
                .extra_info
                .write_descrs_fields
                .as_ref()
                .is_some_and(|bs| bs.iter().any(|&b| b != 0)),
        );
    }

    #[test]
    fn resolve_array_identity_follows_phi_chain_to_constant_link_arg() {
        use crate::flowspace::model::Variable;
        let cc = CallControl::new();
        let base = Variable::new();
        let forwarded = Variable::new();
        let value_producers: HashMap<Variable, &OpKind> = HashMap::new();
        let mut phi_sources: HashMap<Variable, Option<LinkArg>> = HashMap::new();
        phi_sources.insert(base.clone(), Some(LinkArg::Value(forwarded.clone())));
        phi_sources.insert(
            forwarded,
            Some(LinkArg::from(crate::flowspace::model::ConstValue::List(
                vec![],
            ))),
        );

        assert_eq!(
            resolve_array_identity(
                &base,
                &Option::<String>::None,
                &value_producers,
                &phi_sources,
                &cc,
            ),
            Some("list".to_string())
        );
    }

    #[test]
    fn test_getcalldescr_elidable_ignores_writes() {
        // Elidable function: write_descrs should be 0 even if graph has writes.
        // RPython effectinfo.py:181-186: ignore writes for elidable.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("pure_writer");
        let base_var = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldWrite {
                base: base_var.clone(),
                field: crate::model::FieldDescriptor::new("cache", Some("Obj".into())),
                value: base_var,
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        let path = CallPath::from_segments(["pure_writer"]);
        register_int_result_graph(&mut cc, path.clone(), graph);
        cc.mark_elidable(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["pure_writer"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
        assert_eq!(
            descriptor.extra_info.extraeffect,
            ExtraEffect::ElidableCannotRaise
        );
        // Writes should be an empty bitstring for elidable functions:
        // effectinfo.py:169-181 clears the write frozensets, and
        // compute_bitstrings serializes an empty set as an empty Vec.
        let writes = descriptor
            .extra_info
            .write_descrs_fields
            .as_ref()
            .expect("elidable getcalldescr populates write_descrs_fields");
        assert!(writes.is_empty());
    }

    #[test]
    fn test_canraise_cached() {
        // Verify caching: second call should reuse result.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["raiser"]);
        cc.register_function_graph(path, raising_graph("raiser"));

        let target = CallTarget::function_path(["raiser"]);
        let mut cache = AnalysisCache::default();

        let r1 = cc._canraise(&target, &mut cache);
        assert_eq!(r1, CanRaise::Yes);
        assert!(
            cache
                .can_raise
                .contains_key(&CallPath::from_segments(["raiser"]))
        );

        let r2 = cc._canraise(&target, &mut cache);
        assert_eq!(r2, CanRaise::Yes);
    }

    /// Characterization probe:
    /// feed ONE flat graph to both effect-analysis paths and assert they
    /// agree — the flat `CallControl._canraise` vs the orthodox
    /// `backendopt::canraise::RaiseAnalyzer` over the adapter-produced
    /// flowspace graph. On a non-raising self-contained graph both report
    /// "cannot raise"; the raise-bearing synthetic helpers are SSA-malformed
    /// so the adapter rejects them, pinning the SSA-definedness invariant.
    /// Cross-graph `direct_call` resolution (callee registered in the shared
    /// `TranslationContext.graphs` vs `top_result` when absent) is proved
    /// separately by `analyze_direct_call_resolves_registered_callee_else_top_result`
    /// in `backendopt::canraise`.
    #[test]
    fn characterize_flat_canraise_vs_flowspace_raiseanalyzer() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::backendopt::canraise::RaiseAnalyzer;
        use crate::translator::backendopt::graphanalyze::GraphAnalyzer;
        use crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace;
        use crate::translator::rtyper::pyre_call_registry::PyreCallRegistry;
        use crate::translator::translator::TranslationContext;
        use std::rc::Rc;

        // Divergence map:
        //   non-raising  : flat=No             | adapter OK, RaiseAnalyzer=false (AGREE)
        //   set_raise    : flat=Yes            | adapter REJECTS (undefined slot, EXC link)
        //   reraise_only : flat=MemoryErrorOnly| adapter REJECTS (undefined slot, NORMAL link)
        // The rejection is NOT an adapter exception-edge defect and NOT a
        // missing exception-slot seed: both synthetic helpers are simply
        // malformed — each carries an SSA-undefined value on a Link out of
        // the producer-less entry block. raising_graph (set_raise = plain
        // goto, no .extravars) routes a fresh etype/evalue on its EXCEPTION
        // link to the exceptblock; reraise_only routes an undefined
        // continuation_arg on its NORMAL fall-through link (two distinct
        // edges, same root cause). RPython's own checkgraph
        // (model.py:668-688) would reject both: every Link.args value must
        // be defined in the predecessor block (only last_exception /
        // last_exc_value may be defined only_in_link). Real front-end
        // graphs ARE well-formed and DO convert — production runs
        // function_graph_to_flowspace on every graph and check.py is green.
        // So these `.is_err()` assertions pin that the adapter correctly
        // enforces SSA-definedness, not an exception-edge limitation; the
        // well_formed_raise_* test below proves the flowspace RaiseAnalyzer
        // matches the flat _canraise on a well-formed raise graph.
        let registry = || PyreCallRegistry::new(Rc::new(Bookkeeper::new()));

        // -- non-raising: converts, and both paths agree it cannot raise --
        {
            let mut cc = CallControl::new();
            cc.register_function_graph(CallPath::from_segments(["nr"]), simple_graph("nr"));
            let flat = cc._canraise(
                &CallTarget::function_path(["nr"]),
                &mut AnalysisCache::default(),
            );
            assert_eq!(flat, CanRaise::No);

            let reg = registry();
            let out = function_graph_to_flowspace(&simple_graph("nr"), &reg)
                .expect("non-raising flat graph converts to flowspace");
            let translator = TranslationContext::new();
            translator.graphs.borrow_mut().push(out.graph.clone());
            let mut ra = RaiseAnalyzer::new(&translator);
            assert!(
                !ra.analyze_direct_call(&out.graph, None),
                "flowspace RaiseAnalyzer agrees the non-raising graph cannot raise"
            );
        }

        // -- raise-bearing SYNTHETIC helpers are malformed (an SSA-undefined
        //    Link arg out of the empty entry block), so the adapter rejects
        //    them; this pins the SSA-definedness invariant, not an
        //    exception-edge defect (real graphs convert — see doc above) --
        let raise_cases: [(&str, fn(&str) -> FunctionGraph, CanRaise); 2] = [
            ("rs", raising_graph, CanRaise::Yes),
            ("rr", reraise_only_graph, CanRaise::MemoryErrorOnly),
        ];
        for (label, build, expected_flat) in raise_cases {
            let mut cc = CallControl::new();
            cc.register_function_graph(CallPath::from_segments([label]), build(label));
            let flat = cc._canraise(
                &CallTarget::function_path([label]),
                &mut AnalysisCache::default(),
            );
            assert_eq!(flat, expected_flat);

            let reg = registry();
            assert!(
                function_graph_to_flowspace(&build(label), &reg).is_err(),
                "adapter rejects the malformed synthetic graph (SSA-undefined Link arg)"
            );
        }
    }

    #[test]
    fn test_canraise_reraise_only_graph_matches_upstream_tri_state() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["reraise_only"]);
        cc.register_function_graph(path.clone(), reraise_only_graph("reraise_only"));

        let target = CallTarget::function_path(["reraise_only"]);
        let mut cache = AnalysisCache::default();
        let result = cc._canraise(&target, &mut cache);
        assert_eq!(result, CanRaise::MemoryErrorOnly);
    }

    #[test]
    fn test_canraise_ignore_memoryerror_suppresses_reraise_only_exceptblock() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["reraise_only"]);
        cc.register_function_graph(path.clone(), reraise_only_graph("reraise_only"));

        let mut seen = HashSet::new();
        assert!(!cc.analyze_can_raise_impl(&path, &mut seen, true));
    }

    /// Graph-model unification (test-only): prove the orthodox
    /// flowspace `RaiseAnalyzer` (backendopt/canraise.rs) reproduces the
    /// SAME tri-state verdict as the flat `CallControl::_canraise` on
    /// well-formed lltype-vocabulary graphs. This locks the equivalence
    /// contract that is the precondition for deleting
    /// `analyze_can_raise_impl` once the flowspace analyzers are wired
    /// into `getcalldescr`.
    ///
    /// Tri-state coverage, parametrized by startblock op + CFG shape:
    /// - `int_add`, no exception edge → `No` (the op cannot raise and the
    ///   graph never reaches the exceptblock).
    /// - `int_add` (`canraise=[]`), `LastException` edge re-raising the
    ///   caught exception into the exceptblock → `MemoryErrorOnly` (the
    ///   default analyzer reaches the exceptblock → raises; the
    ///   `ignore_memory_error` analyzer suppresses it via
    ///   `exceptblock_is_reraise_of_caught_exception`). Exercises the
    ///   reraise-of-caught suppression in BOTH paths.
    /// - `int_add_ovf` (`canraise=[OverflowError]`) → `Yes` in both modes
    ///   (the op raises a non-MemoryError, short-circuiting the
    ///   exceptblock).
    ///
    /// `int_add`/`int_add_ovf` are lltype-vocabulary opnames present in
    /// `ll_operations()`; a non-lltype opname (e.g. `add`) would be
    /// UNKNOWN to the flowspace table and conservatively classified
    /// raising — an op-vocabulary difference, not an exceptblock
    /// divergence (the flowspace analyzer targets post-rtype graphs).
    #[test]
    fn well_formed_raise_flowspace_raiseanalyzer_matches_flat_canraise() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::translator::backendopt::canraise::RaiseAnalyzer;
        use crate::translator::backendopt::graphanalyze::GraphAnalyzer;
        use crate::translator::rtyper::flowspace_adapter::function_graph_to_flowspace;
        use crate::translator::rtyper::pyre_call_registry::PyreCallRegistry;
        use crate::translator::translator::TranslationContext;
        use std::rc::Rc;

        // Entry computes `op_result = OP(lhs, rhs)`. When `raises` is
        // set the block exits on `LastException`: the normal edge carries
        // `op_result` to the returnblock and the exception edge carries
        // the caught `(exc_type, exc_value)` into the exceptblock
        // inputargs via `.extravars` — a reraise-of-caught shape. When
        // `raises` is clear the block has a single unconditional exit to
        // the returnblock (no exceptblock reach), so the only raise
        // source is the op itself.
        fn build(name: &str, opname: &str, raises: bool) -> FunctionGraph {
            let mut g = FunctionGraph::new(name);
            let lhs = g.alloc_value_var();
            let rhs = g.alloc_value_var();
            let op_result = g.alloc_value_var();
            let ret_param = g.alloc_value_var();
            let exc_type = g.alloc_value_var();
            let exc_value = g.alloc_value_var();
            let op = crate::model::SpaceOperation {
                result: Some(op_result.clone()),
                kind: crate::model::OpKind::BinOp {
                    op: opname.to_string(),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    result_ty: crate::model::ValueType::Int,
                },
            };
            let (exitswitch, exits) = if raises {
                (
                    Some(ExitSwitch::LastException),
                    vec![
                        Link::new_mixed(
                            vec![LinkArg::Value(op_result.clone())],
                            g.returnblock,
                            None,
                        ),
                        Link::new_mixed(
                            vec![
                                LinkArg::Value(exc_type.clone()),
                                LinkArg::Value(exc_value.clone()),
                            ],
                            g.exceptblock,
                            Some(exception_exitcase()),
                        )
                        .extravars(
                            Some(LinkArg::Value(exc_type.clone())),
                            Some(LinkArg::Value(exc_value.clone())),
                        ),
                    ],
                )
            } else {
                (
                    None,
                    vec![Link::new_mixed(
                        vec![LinkArg::Value(op_result.clone())],
                        g.returnblock,
                        None,
                    )],
                )
            };
            let startblock = crate::model::Block {
                id: g.startblock,
                inputargs: vec![lhs.clone(), rhs.clone()],
                operations: vec![op],
                exitswitch,
                exits,
                dead: false,
                framestate: None,
            };
            let returnblock = crate::model::Block {
                id: g.returnblock,
                inputargs: vec![ret_param.clone()],
                operations: vec![],
                exitswitch: None,
                exits: vec![],
                dead: false,
                framestate: None,
            };
            let mut blocks = vec![startblock, returnblock];
            if raises {
                blocks.push(crate::model::Block {
                    id: g.exceptblock,
                    inputargs: vec![exc_type.clone(), exc_value.clone()],
                    operations: vec![],
                    exitswitch: None,
                    exits: vec![],
                    dead: false,
                    framestate: None,
                });
            }
            g.blocks = blocks;
            g
        }

        // (label, opname, reaches-exceptblock, expected flat tri-state)
        let cases: [(&str, &str, bool, CanRaise); 3] = [
            ("no", "int_add", false, CanRaise::No),
            ("mem", "int_add", true, CanRaise::MemoryErrorOnly),
            ("yes", "int_add_ovf", true, CanRaise::Yes),
        ];
        for (label, opname, raises, expected) in cases {
            // -- flat path: CallControl tri-state --
            let mut cc = CallControl::new();
            cc.register_function_graph(
                CallPath::from_segments(["wf_raise"]),
                build("wf_raise", opname, raises),
            );
            let flat = cc._canraise(
                &CallTarget::function_path(["wf_raise"]),
                &mut AnalysisCache::default(),
            );
            assert_eq!(
                flat, expected,
                "flat _canraise verdict for {label}/{opname}"
            );

            // -- flowspace path: adapter converts; RaiseAnalyzer reproduces
            //    the flat tri-state from the (default, ignore-MemoryError)
            //    boolean pair (default ↔ ignore=false; do_ignore_memory_error
            //    ↔ ignore=true) --
            let reg = PyreCallRegistry::new(Rc::new(Bookkeeper::new()));
            let out = function_graph_to_flowspace(&build("wf_raise", opname, raises), &reg)
                .expect("well-formed graph converts to flowspace");
            let translator = TranslationContext::new();
            translator.graphs.borrow_mut().push(out.graph.clone());

            let mut ra = RaiseAnalyzer::new(&translator);
            let fs_default = ra.analyze_direct_call(&out.graph, None);

            let mut ra_ignore = RaiseAnalyzer::new(&translator);
            ra_ignore.do_ignore_memory_error();
            let fs_ignore = ra_ignore.analyze_direct_call(&out.graph, None);

            let fs_tristate = match (fs_default, fs_ignore) {
                (false, _) => CanRaise::No,
                (true, true) => CanRaise::Yes,
                (true, false) => CanRaise::MemoryErrorOnly,
            };
            assert_eq!(
                fs_tristate, flat,
                "{label}/{opname}: flowspace RaiseAnalyzer (default={fs_default}, \
                 ignore_mem={fs_ignore}) must reproduce the flat _canraise tri-state {flat:?}"
            );
        }
    }

    #[test]
    fn test_readonly_excludes_written_fields() {
        // RPython effectinfo.py:345-348: readstruct only goes to readonly
        // if there's no corresponding write ("struct") for that field.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("rw_same_field");
        let base_var = graph.alloc_value_var();
        let field = crate::model::FieldDescriptor::new("x", Some("Point".into()));
        // Both read AND write the same field "x"
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldRead {
                base: base_var.clone(),
                field: field.clone(),
                ty: ValueType::Int,
                pure: false,
            },
            true,
        );
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldWrite {
                base: base_var.clone(),
                field: field.clone(),
                value: base_var,
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_return(graph.startblock, None);
        let path = CallPath::from_segments(["rw_same_field"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["rw_same_field"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
        // Write is set, but readonly should NOT have the same bit set.
        // RPython: readonly = reads & ~writes
        assert!(
            descriptor
                .extra_info
                .write_descrs_fields
                .as_ref()
                .is_some_and(|bs| bs.iter().any(|&b| b != 0)),
        );
        let overlap = match (
            descriptor.extra_info.readonly_descrs_fields.as_ref(),
            descriptor.extra_info.write_descrs_fields.as_ref(),
        ) {
            (Some(ro), Some(wr)) => ro.iter().zip(wr.iter()).any(|(a, b)| (a & b) != 0),
            _ => false,
        };
        assert!(
            !overlap,
            "readonly and write should not overlap for same field"
        );
    }

    #[test]
    fn test_op_can_raise_division() {
        // Division ops can raise (ZeroDivisionError).
        // RPython: LL_OPERATIONS[int_floordiv].canraise = (ZeroDivisionError,)
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("divider");
        let a_var = graph.alloc_value_var();
        let b_var = graph.alloc_value_var();
        graph.push_op_var(
            graph.startblock,
            OpKind::BinOp {
                op: "int_floordiv".to_string(),
                lhs: a_var,
                rhs: b_var,
                result_ty: ValueType::Int,
            },
            true,
        );
        graph.set_return(graph.startblock, None);
        let path = CallPath::from_segments(["divider"]);
        cc.register_function_graph(path, graph);

        let target = CallTarget::function_path(["divider"]);
        let mut cache = AnalysisCache::default();
        let result = cc._canraise(&target, &mut cache);
        assert_eq!(result, CanRaise::Yes);
    }

    #[test]
    fn struct_layout_depth3_nested_fixed_point() {
        // A contains B, B contains C.  Fixed-point iteration must
        // produce correct sizes regardless of HashMap iteration order.
        // struct C { x: i64 }            → size 8
        // struct B { c: C, y: i64 }      → size 16
        // struct A { b: B, z: i64 }      → size 24
        let mut known_structs: HashSet<String> = HashSet::new();
        known_structs.insert("C".into());
        known_structs.insert("B".into());
        known_structs.insert("A".into());

        let fields_c: Vec<(String, String)> = vec![("x".into(), "i64".into())];
        let fields_b: Vec<(String, String)> =
            vec![("c".into(), "C".into()), ("y".into(), "i64".into())];
        let fields_a: Vec<(String, String)> =
            vec![("b".into(), "B".into()), ("z".into(), "i64".into())];

        // Fixed-point iteration (same algorithm as lib.rs).
        let mut known_sizes: HashMap<String, usize> = HashMap::new();
        let all_fields: Vec<(&str, &Vec<(String, String)>)> =
            vec![("A", &fields_a), ("B", &fields_b), ("C", &fields_c)];
        loop {
            let mut changed = false;
            for (name, fields) in &all_fields {
                let layout = StructLayout::from_type_strings(
                    fields,
                    &known_structs,
                    &known_sizes,
                    &HashMap::new(),
                );
                if known_sizes.get(*name) != Some(&layout.size) {
                    known_sizes.insert(name.to_string(), layout.size);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        assert_eq!(known_sizes["C"], 8, "C: single i64");
        assert_eq!(known_sizes["B"], 16, "B: C(8) + i64(8)");
        assert_eq!(known_sizes["A"], 24, "A: B(16) + i64(8)");
    }

    #[derive(Debug)]
    struct StubVInfo {
        vtypeptr_id: usize,
    }
    impl VirtualizableInfoHandle for StubVInfo {
        fn is_vtypeptr(&self, vtypeptr_id: usize) -> bool {
            self.vtypeptr_id == vtypeptr_id
        }
    }

    #[derive(Debug)]
    struct StubGFInfo {
        green_fields: HashSet<(String, String)>,
    }
    impl GreenFieldInfoHandle for StubGFInfo {
        fn contains_green_field(&self, gtype: &str, fieldname: &str) -> bool {
            self.green_fields
                .contains(&(gtype.to_string(), fieldname.to_string()))
        }
    }

    fn cc_with_one_driver() -> CallControl {
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal_runner"]),
            vec!["pc".into()],
            vec!["frame".into()],
            vec![],
            vec![],
        );
        cc
    }

    #[test]
    fn get_vinfo_returns_none_when_no_driver_has_virtualizable_info() {
        let cc = cc_with_one_driver();
        assert!(cc.get_vinfo(0xfeed).is_none());
    }

    #[test]
    fn get_vinfo_returns_matching_handle_from_driver() {
        let mut cc = cc_with_one_driver();
        let vinfo: std::sync::Arc<dyn VirtualizableInfoHandle> = std::sync::Arc::new(StubVInfo {
            vtypeptr_id: 0xabcd,
        });
        cc.jitdrivers_sd[0].virtualizable_info = Some(std::sync::Arc::clone(&vinfo));
        let got = cc.get_vinfo(0xabcd).expect("must match");
        assert!(std::sync::Arc::ptr_eq(&got, &vinfo));
        // Non-matching id → None.
        assert!(cc.get_vinfo(0x1234).is_none());
    }

    #[test]
    #[should_panic(expected = "multiple distinct VirtualizableInfo")]
    fn get_vinfo_panics_when_multiple_distinct_infos_match_same_vtypeptr() {
        let mut cc = cc_with_one_driver();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal_runner_b"]),
            vec![],
            vec![],
            vec![],
            vec![],
        );
        cc.jitdrivers_sd[0].virtualizable_info = Some(std::sync::Arc::new(StubVInfo {
            vtypeptr_id: 0xabcd,
        }));
        cc.jitdrivers_sd[1].virtualizable_info = Some(std::sync::Arc::new(StubVInfo {
            vtypeptr_id: 0xabcd,
        }));
        let _ = cc.get_vinfo(0xabcd);
    }

    #[test]
    fn could_be_green_field_returns_false_when_no_driver_has_greenfield_info() {
        let cc = cc_with_one_driver();
        assert!(!cc.could_be_green_field("Frame", "code"));
    }

    #[test]
    fn could_be_green_field_returns_true_for_registered_pair() {
        let mut cc = cc_with_one_driver();
        let mut greens = HashSet::new();
        greens.insert(("Frame".to_string(), "code".to_string()));
        cc.jitdrivers_sd[0].greenfield_info = Some(std::sync::Arc::new(StubGFInfo {
            green_fields: greens,
        }));
        assert!(cc.could_be_green_field("Frame", "code"));
        assert!(!cc.could_be_green_field("Frame", "pc"));
        assert!(!cc.could_be_green_field("OtherFrame", "code"));
    }

    #[test]
    fn set_jitdriver_virtualizable_info_is_visible_to_get_vinfo() {
        // warmspot.py:528-545 assignment hook reachability test —
        // exercises the production wiring path (not direct field write).
        let mut cc = cc_with_one_driver();
        let info: std::sync::Arc<dyn VirtualizableInfoHandle> =
            std::sync::Arc::new(StubVInfo { vtypeptr_id: 0xab });
        cc.set_jitdriver_virtualizable_info(0, std::sync::Arc::clone(&info));
        let got = cc.get_vinfo(0xab).expect("must match after set");
        assert!(std::sync::Arc::ptr_eq(&got, &info));
    }

    #[test]
    fn set_jitdriver_greenfield_info_is_visible_to_could_be_green_field() {
        let mut cc = cc_with_one_driver();
        let mut greens = HashSet::new();
        greens.insert(("Frame".to_string(), "pc".to_string()));
        let info: std::sync::Arc<dyn GreenFieldInfoHandle> = std::sync::Arc::new(StubGFInfo {
            green_fields: greens,
        });
        cc.set_jitdriver_greenfield_info(0, info);
        assert!(cc.could_be_green_field("Frame", "pc"));
        assert!(!cc.could_be_green_field("Frame", "code"));
    }

    #[test]
    fn make_virtualizable_infos_assigns_index_and_handle_per_warmspot_py_534() {
        // warmspot.py:534-545 — single jitdriver with virtualizables=['frame'],
        // reds=['frame', 'ec'].  index_of_virtualizable must land on slot 0
        // (matching reds.index('frame')) and virtualizable_info must
        // become a populated handle whose VTYPEPTR matches the
        // owner_root token shared across all jitdrivers.
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["execute_opcode_step"]),
            vec!["pc".into()],
            vec!["frame".into(), "ec".into()],
            vec!["frame".into()],
            vec!["PyFrame".into(), "ExecutionContext".into()],
        );
        cc.make_virtualizable_infos(|_, _| None);
        // warmspot.py:534-538 — `index_of_virtualizable = reds.index('frame')`
        assert_eq!(cc.jitdrivers_sd[0].index_of_virtualizable, 0);
        // warmspot.py:540-545 — codewriter side leaves vinfo None;
        // runtime metainterp populates via set_jitdriver_virtualizable_info.
        assert!(cc.jitdrivers_sd[0].virtualizable_info.is_none());
        // warmspot.py:531-532 — virtualizables present + no dotted greens
        // → greenfield_info stays None.
        assert!(cc.jitdrivers_sd[0].greenfield_info.is_none());
    }

    #[test]
    #[should_panic(expected = "greenfield + virtualizable on the same driver")]
    fn make_virtualizable_infos_panics_on_dotted_green_with_virtualizable() {
        // warmspot.py:531-532 `assert jd.greenfield_info is None,
        // "XXX not supported yet"` — pyre keeps the assertion.
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal"]),
            vec!["frame.code".into()],
            vec!["frame".into()],
            vec!["frame".into()],
            vec!["PyFrame".into()],
        );
        cc.make_virtualizable_infos(|_, _| None);
    }

    #[test]
    fn make_virtualizable_infos_clears_when_no_virtualizable() {
        // warmspot.py:527-530 — `if not jd.jitdriver.virtualizables: ... continue`.
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal"]),
            vec!["pc".into()],
            vec!["frame".into()],
            vec![],
            vec![],
        );
        cc.jitdrivers_sd[0].virtualizable_info = Some(std::sync::Arc::new(StubVInfo {
            vtypeptr_id: 0xfeed,
        }));
        cc.jitdrivers_sd[0].index_of_virtualizable = 7;
        cc.make_virtualizable_infos(|_, _| None);
        assert!(cc.jitdrivers_sd[0].virtualizable_info.is_none());
        assert_eq!(cc.jitdrivers_sd[0].index_of_virtualizable, -1);
    }

    #[test]
    fn make_virtualizable_infos_resolves_gtype_from_red_types() {
        // greenfield.py:14,18 — green_fields holds (GTYPE, fieldname) where
        // GTYPE is the type of the red slot identified by objname.
        // Pyre threads this through `red_types` parallel to `reds`.
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal_with_greenfield"]),
            vec!["frame.code".into(), "pc".into()],
            vec!["frame".into()],
            vec![],
            vec!["PyFrame".into()],
        );
        cc.make_virtualizable_infos(|_, _| None);
        let gfinfo = cc.jitdrivers_sd[0]
            .greenfield_info
            .as_ref()
            .expect("greenfield_info populated for dotted green");
        // contains_green_field expects (GTYPE, fieldname) — resolved
        // from `red_types` not the raw `objname`.
        assert!(gfinfo.contains_green_field("PyFrame", "code"));
        assert!(!gfinfo.contains_green_field("frame", "code"));
    }

    #[test]
    fn make_virtualizable_infos_invokes_factory_and_caches_per_vtypeptr() {
        // warmspot.py:540-545 — `vinfos[VTYPEPTR]` cache: two jitdrivers
        // sharing the same VTYPEPTR token must reuse the same handle
        // (same Arc identity), and the factory must be called once
        // per unique VTYPEPTR.
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal_a"]),
            vec!["pc".into()],
            vec!["frame".into()],
            vec!["frame".into()],
            vec!["PyFrame".into()],
        );
        cc.setup_jitdriver(
            CallPath::from_segments(["portal_b"]),
            vec!["pc".into()],
            vec!["frame".into()],
            vec!["frame".into()],
            vec!["PyFrame".into()],
        );
        let mut factory_calls: Vec<String> = Vec::new();
        cc.make_virtualizable_infos(|_jd_idx, vtypeptr_token| {
            factory_calls.push(vtypeptr_token.to_string());
            Some(std::sync::Arc::new(StubVInfo {
                vtypeptr_id: 0xfeed,
            }))
        });
        assert_eq!(
            factory_calls,
            vec!["PyFrame".to_string()],
            "warmspot.py:540-545 vinfos cache must dedupe by VTYPEPTR token",
        );
        let h0 = cc.jitdrivers_sd[0]
            .virtualizable_info
            .clone()
            .expect("vinfo populated");
        let h1 = cc.jitdrivers_sd[1]
            .virtualizable_info
            .clone()
            .expect("vinfo populated");
        assert!(std::sync::Arc::ptr_eq(&h0, &h1));
    }

    #[test]
    fn make_virtualizable_infos_factory_none_keeps_slot_empty() {
        // warmspot.py:540-545 with factory→None: the codewriter slot
        // stays empty so the runtime metainterp setter populates it
        // later (jitdriver.rs:285).
        let mut cc = CallControl::new();
        cc.setup_jitdriver(
            CallPath::from_segments(["portal"]),
            vec!["pc".into()],
            vec!["frame".into()],
            vec!["frame".into()],
            vec!["PyFrame".into()],
        );
        cc.make_virtualizable_infos(|_, _| None);
        assert!(cc.jitdrivers_sd[0].virtualizable_info.is_none());
        assert_eq!(cc.jitdrivers_sd[0].index_of_virtualizable, 0);
    }

    // ── RPython indirect_call family tests ──────────────────────────

    /// `guess_call_kind` for `OpKind::IndirectCall`:
    ///   ≥1 candidate impl is a regular candidate → `Regular`
    ///   graphs `None` (unknown family)          → `Residual`
    /// RPython `call.py:116-139`.  Mirrors the
    /// `op.opname == 'indirect_call'` fall-through to the final
    /// `graphs_from(op) is None` test.
    #[test]
    fn guess_call_kind_indirect() {
        let mut cc = CallControl::new();
        cc.register_trait_method("run", Some("Handler"), "A", FunctionGraph::new("A::run"));
        cc.register_trait_method("run", Some("Handler"), "B", FunctionGraph::new("B::run"));
        cc.find_all_graphs_for_tests();

        let handler_family = cc.all_impls_for_indirect("Handler", "run");
        assert_eq!(handler_family.len(), 2);
        assert_eq!(
            cc.guess_call_kind(&indirect_call_op(Some(handler_family))),
            CallKind::Regular
        );

        // `graphs = None` → unknown family → residual path per
        // `rpython/translator/backendopt/graphanalyze.py:117`.
        assert_eq!(
            cc.guess_call_kind(&indirect_call_op(None)),
            CallKind::Residual
        );
    }

    /// `graphs_from(op)` for an `OpKind::IndirectCall` must filter by
    /// the family attached to the op, not mix impls across traits that
    /// share a method name.  RPython `call.py:103-112` indirect branch.
    #[test]
    fn graphs_from_op_filters_by_indirect_family() {
        let mut cc = CallControl::new();
        cc.register_trait_method(
            "bar",
            Some("Foo"),
            "FooImpl",
            FunctionGraph::new("FooImpl::bar"),
        );
        cc.register_trait_method(
            "bar",
            Some("Baz"),
            "BazImpl",
            FunctionGraph::new("BazImpl::bar"),
        );
        cc.find_all_graphs_for_tests();

        let foo_family = cc.all_impls_for_indirect("Foo", "bar");
        let baz_family = cc.all_impls_for_indirect("Baz", "bar");

        let foo_candidates = cc
            .graphs_from(&indirect_call_op(Some(foo_family)))
            .expect("Foo::bar family is non-empty");
        let baz_candidates = cc
            .graphs_from(&indirect_call_op(Some(baz_family)))
            .expect("Baz::bar family is non-empty");

        assert_eq!(foo_candidates.len(), 1);
        assert_eq!(
            foo_candidates[0].segments[0], "FooImpl",
            "Foo::bar must not surface BazImpl: {foo_candidates:?}"
        );
        assert_eq!(baz_candidates.len(), 1);
        assert_eq!(
            baz_candidates[0].segments[0], "BazImpl",
            "Baz::bar must not surface FooImpl: {baz_candidates:?}"
        );
    }

    /// `getcalldescr` with mixed `@jit.elidable` vs non-elidable impls
    /// panics to match RPython `call.py:259-280`.
    #[test]
    #[should_panic(expected = "indirect_call family")]
    fn getcalldescr_rejects_mixed_elidable_family() {
        use majit_ir::value::Type;
        let mut cc = CallControl::new();
        cc.register_trait_method(
            "bar",
            Some("Foo"),
            "PureImpl",
            FunctionGraph::new("PureImpl::bar"),
        );
        cc.register_trait_method(
            "bar",
            Some("Foo"),
            "ImpureImpl",
            FunctionGraph::new("ImpureImpl::bar"),
        );
        cc.mark_elidable(CallPath::from_segments(["PureImpl", "bar"]));
        cc.find_all_graphs_for_tests();

        let family = cc.all_impls_for_indirect("Foo", "bar");
        let mut cache = AnalysisCache::default();
        let _ = cc.getcalldescr(
            &indirect_call_op(Some(family)),
            vec![Type::Ref],
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
            None,
        );
    }

    /// Inherent impl (no `impl Trait for Type`) continues to resolve
    /// via `function_graphs` and classify as `Regular`, without
    /// populating `trait_method_impls`.
    #[test]
    fn inherent_method_still_direct_regression() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["Foo", "bar"]);
        cc.register_function_graph(path.clone(), FunctionGraph::new("Foo::bar"));
        cc.find_all_graphs_for_tests();

        // Inherent impl: front-end emits `CallTarget::Method` with a
        // concrete receiver_root.  No `trait_method_impls` entry.
        let target = CallTarget::method("bar", Some("Foo".to_string()));
        assert_eq!(
            cc.guess_call_kind(&direct_call_op(target)),
            CallKind::Regular
        );
        assert!(
            cc.all_impls_for_indirect("Foo", "bar").is_empty(),
            "inherent impls must not appear in any indirect-call family"
        );
    }

    /// A trait-default method (registered under the synthetic
    /// `"<default methods of Trait>"` impl-type produced by
    /// `parse.rs:327-332`) must show up in the same indirect-call
    /// family as concrete overrides.  This keeps
    /// `lower_indirect_calls`'s `all_impls_for_indirect(...)` family
    /// correct when a `dyn Trait` receiver can route to either the
    /// default body or an override at runtime — parity with RPython
    /// `rpbc.py:199-217` `c_graphs = row_of_graphs.values()`, which
    /// lists every graph reachable through the trait's vtable slot.
    #[test]
    fn dyn_trait_default_method_uses_same_indirect_family() {
        let mut cc = CallControl::new();
        cc.register_trait_method("m", Some("Foo"), "A", FunctionGraph::new("A::m"));
        cc.register_trait_method(
            "m",
            Some("Foo"),
            "<default methods of Foo>",
            FunctionGraph::new("<default methods of Foo>::m"),
        );

        let family = cc.all_impls_for_indirect("Foo", "m");
        let segs: Vec<String> = family.iter().map(|p| p.segments.join("::")).collect();
        assert!(
            segs.iter().any(|s| s.starts_with("A")),
            "family must include overriding impl A, got {segs:?}"
        );
        assert!(
            segs.iter().any(|s| s.contains("<default methods of Foo>")),
            "family must include trait default-method entry, got {segs:?}"
        );
        assert_eq!(
            family.len(),
            2,
            "family must have exactly two members (default + override), got {segs:?}"
        );
    }

    /// Two trait objects that differ only in their generic arguments
    /// (`Handler<i64>` vs `Handler<String>`) must be treated as separate
    /// indirect-call families.  Conflating them would produce a mixed
    /// family whose candidates accept incompatible argument types —
    /// `rpython/jit/codewriter/call.py:259-280`'s family validation
    /// would reject the mixed descriptor, and the dispatch would
    /// dead-end at runtime.  The family key passed to
    /// `all_impls_for_indirect` must preserve the full `trait_root`
    /// (including generic args), not a bare `Handler` root.
    #[test]
    fn dyn_trait_generic_family_key_preserved() {
        let mut cc = CallControl::new();
        cc.register_trait_method(
            "run",
            Some("Handler<i64>"),
            "A",
            FunctionGraph::new("A::run"),
        );
        cc.register_trait_method(
            "run",
            Some("Handler<String>"),
            "B",
            FunctionGraph::new("B::run"),
        );

        let i64_family = cc.all_impls_for_indirect("Handler<i64>", "run");
        let string_family = cc.all_impls_for_indirect("Handler<String>", "run");
        let bare_family = cc.all_impls_for_indirect("Handler", "run");

        let segs =
            |v: &[CallPath]| -> Vec<String> { v.iter().map(|p| p.segments.join("::")).collect() };

        let i64_segs = segs(&i64_family);
        let string_segs = segs(&string_family);
        assert_eq!(
            i64_family.len(),
            1,
            "Handler<i64> family must contain only A, got {i64_segs:?}"
        );
        assert!(
            i64_segs[0].starts_with("A"),
            "Handler<i64> must resolve to impl A, got {i64_segs:?}"
        );
        assert_eq!(
            string_family.len(),
            1,
            "Handler<String> family must contain only B, got {string_segs:?}"
        );
        assert!(
            string_segs[0].starts_with("B"),
            "Handler<String> must resolve to impl B, got {string_segs:?}"
        );
        assert!(
            bare_family.is_empty(),
            "bare `Handler` (generic args stripped) must NOT match a \
             generic-instantiated family — conflation would cross \
             incompatible argument types; got {:?}",
            segs(&bare_family),
        );
    }

    #[test]
    fn test_getcalldescr_extradescrs_propagated() {
        use std::sync::Arc;

        #[derive(Debug)]
        struct StubDescr(u32);
        impl majit_ir::Descr for StubDescr {
            fn index(&self) -> u32 {
                self.0
            }
        }

        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["pure_add"]);
        register_int_result_graph(&mut cc, path.clone(), simple_graph("pure_add"));
        cc.find_all_graphs_for_tests();

        let extra0: DescrRef = Arc::new(StubDescr(80));
        let extra1: DescrRef = Arc::new(StubDescr(81));
        let extras = Some(vec![extra0.clone(), extra1.clone()]);

        let target = CallTarget::function_path(["pure_add"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::DictLookup,
            None,
            &mut cache,
            extras,
        );
        let got = descriptor.extra_info.extradescrs.as_ref().unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].index(), 80);
        assert_eq!(got[1].index(), 81);
    }

    #[test]
    fn test_getcalldescr_extradescrs_survives_random_effects() {
        use std::sync::Arc;

        #[derive(Debug)]
        struct StubDescr(u32);
        impl majit_ir::Descr for StubDescr {
            fn index(&self) -> u32 {
                self.0
            }
        }

        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["chaotic"]);
        cc.register_function_graph(path.clone(), raising_graph("chaotic"));
        cc.find_all_graphs_for_tests();

        let extra0: DescrRef = Arc::new(StubDescr(90));
        let extras = Some(vec![extra0.clone()]);

        let target = CallTarget::function_path(["chaotic"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &direct_call_op(target.clone()),
            Vec::new(),
            Type::Int,
            OopSpecIndex::DictLookup,
            Some(ExtraEffect::RandomEffects),
            &mut cache,
            extras,
        );
        let got = descriptor.extra_info.extradescrs.as_ref().unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].index(), 90);
    }
}
