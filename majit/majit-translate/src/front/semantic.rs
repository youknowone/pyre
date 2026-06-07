//! Front-end shared types — the data shapes `front::mir` produces, and
//! that the rest of the pipeline (`analyze_pipeline_from_module_paths`,
//! `jit_codewriter::*`, `parse::*`) consumes.
//!
//! These types do not depend on any graph builder, so they live in
//! their own module rather than inside `front::mir`.
//!
//! Nothing in this module performs lowering.  Graphs are built by
//! `front::mir`, the Charon ULLBC driver.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::model::{FunctionGraph, ImmutableRank, UnknownKind};

/// Options carried through the semantic-program build.  A distinct unit
/// type so the build entry point can accept an explicit options
/// parameter while preserving the upstream `build_flow_graph` call
/// shape.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AstGraphOptions;

/// Signal that lowering was halted due to an unsupported construct.
///
/// RPython `rpython/flowspace/flowcontext.py:258,417` raises `FlowingError`
/// when the abstract interpreter hits a bytecode it cannot model; that
/// error propagates all the way out of `build_flow_graph`, aborting the
/// current graph rather than silently continuing with a synthetic value.
///
/// Pyre's `Option<Variable>` return conflates "expression legitimately
/// produced no value" (e.g. `return` / `break`) with "lowering halted"
/// — making the latter an explicit `Err` variant restores the RPython
/// invariant that unsupported constructs stop the walk at once.  The
/// `Unknown` op is still emitted at the failure site so downstream
/// passes see evidence of the drop; the `Err` just guarantees no
/// synthesised SSA value follows it.
#[derive(Debug, Clone)]
pub enum FlowingError {
    Unsupported { kind: UnknownKind },
}

/// Alias for `FlowingError`, used by callers that spell the abort
/// type as `LoweringAbort`.
pub type LoweringAbort = FlowingError;

#[derive(Debug, Clone)]
pub struct SemanticFunction {
    pub name: String,
    pub graph: FunctionGraph,
    /// RPython: `op.result.concretetype` — full return type string.
    /// Used for array identity resolution on Call result values.
    pub return_type: Option<String>,
    /// Owner type for impl methods (e.g. "MyStruct" for `impl MyStruct { fn foo() }`).
    /// Used to construct the full CallPath for return_type registration.
    pub self_ty_root: Option<String>,
    /// Module path of the defining file, as supplied to
    /// `parse_source_with_module` (e.g. `"pyframe"` for
    /// `pyre-interpreter/src/pyframe.rs`).  Empty when the caller did not
    /// supply a module path — top-level items remain at simple-name
    /// registration.
    ///
    /// Used by `lib.rs` registration so a free function's call sites that
    /// were qualified by `canonical_call_target:7494-7502` (single-segment
    /// bare call inside a non-empty module) can resolve through the
    /// `[module_path, name]` path, in addition to the bare-name and
    /// `crate::` alias paths.  Without the extra path the
    /// `#[majit_macros::elidable*]` / oopspec / loop-invariant hints
    /// registered against the bare name are silently dropped at every
    /// in-module call site.
    pub module_path: String,
    /// RPython: function-level hints set by GC transformer / decorators.
    /// "close_stack" → _gctransformer_hint_close_stack_
    /// "cannot_collect" → _gctransformer_hint_cannot_collect_
    /// "gc_effects" → random_effects_on_gcobjs
    /// "elidable" → _elidable_function_
    /// "loopinvariant" → _jit_loop_invariant_
    pub hints: Vec<String>,
    /// RPython `graph.access_directly` (flowspace attribute set by the
    /// annotator's `default_specialize` rewrite — see
    /// `description.rs:1333-1335` + `pygraph.rs:53-56`). Carried into
    /// `SemanticFunction` so `policy::look_inside_graph` can port the
    /// `policy.py:71-83` virtualizable safety gate without reaching back
    /// into the PyGraph layer.
    pub access_directly: bool,
    /// Trait name when this function is an `impl Trait for Type {…}`
    /// method, the trait's name when this is a trait default-body
    /// method, otherwise `None` (free function or inherent impl).
    ///
    /// Lets the registration loop in `lib.rs:905-1019` walk
    /// `program.functions` directly and distinguish trait-impl methods
    /// (which need `register_trait_method`) from inherent methods
    /// (which need `register_function_graph`).
    pub trait_root: Option<String>,
}

/// RPython: struct field type info for `heaptracker.all_interiorfielddescrs`.
/// Maps struct_name → vec of (field_name, field_element_type).
/// `field_element_type` is the array element type when the field is an
/// array container (e.g. `Vec<Point>` → `"Point"`), or the full type
/// string for non-array fields.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StructFieldRegistry {
    /// struct_name → [(field_name, full_field_type_string)]
    pub fields: HashMap<String, Vec<(String, String)>>,
}

impl StructFieldRegistry {
    /// Look up a field's type.  For array-typed fields like
    /// `Vec<Point>`, this returns the full type string `"Vec<Point>"`.
    /// Callers use `array_element_type_from_str` to extract `"Point"`.
    ///
    /// Resolution order: (1) exact registered key, then (2) canonical
    /// lexical resolution through `STRUCT_ORIGIN_REGISTRY` (PyPy
    /// `bookkeeper.getdesc(value)` analog) on the receiver leaf —
    /// registration dual-publishes bare + canonical so both spellings
    /// land at the same field list.  (3) crate-prefix-tolerant
    /// suffix-match shim absorbs `pyre_object::rangeobject::W_X` vs
    /// `rangeobject::W_X` divergence — orthogonal to lexical scope
    /// resolution, kept for test entries (`parse::parse_source`)
    /// that bypass `analyze_pipeline_from_module_paths`'s
    /// `register_struct_origins`.
    pub fn field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        self.lookup_fields(owner)?
            .iter()
            .find(|(name, _)| name == field_name)
            .map(|(_, ty)| ty.as_str())
    }

    fn lookup_fields(&self, owner: &str) -> Option<&[(String, String)]> {
        if let Some(fields) = self.fields.get(owner) {
            return Some(fields.as_slice());
        }
        // Canonical lexical resolution: bare receiver leaves resolve
        // through `STRUCT_ORIGIN_REGISTRY` (PyPy `bookkeeper.getdesc`
        // analog).  Registration dual-publishes bare + canonical, so a
        // miss on the exact owner falls through canonical-leaf lookup
        // before the suffix-match shim below.
        let receiver_leaf = owner.rsplit("::").next().unwrap_or(owner);
        let canonical = majit_ir::descr::canonical_struct_name(receiver_leaf);
        if canonical != receiver_leaf
            && let Some(fields) = self.fields.get(&canonical)
        {
            return Some(fields.as_slice());
        }
        let key = self.unique_suffix_owner_key(owner)?;
        self.fields.get(key).map(Vec::as_slice)
    }

    fn unique_suffix_owner_key<'a>(&'a self, owner: &str) -> Option<&'a str> {
        let mut found: Option<&str> = None;
        for key in self.fields.keys() {
            let matches =
                is_path_suffix(owner, key.as_str()) || is_path_suffix(key.as_str(), owner);
            if !matches {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(key.as_str());
        }
        found
    }
}

fn is_path_suffix(longer: &str, shorter: &str) -> bool {
    if longer.len() <= shorter.len() || !longer.ends_with(shorter) {
        return false;
    }
    let prefix_len = longer.len() - shorter.len();
    longer[..prefix_len].ends_with("::")
}

#[derive(Debug, Clone, Default)]
pub struct SemanticProgram {
    pub functions: Vec<SemanticFunction>,
    /// RPython: known struct types for `get_type_flag(ARRAY.OF)` → FLAG_STRUCT.
    pub known_struct_names: std::collections::HashSet<String>,
    /// Known trait names used to canonicalize local `dyn Trait` family keys.
    pub known_trait_names: std::collections::HashSet<String>,
    /// RPython: struct field types for resolving `op.args[0].concretetype`
    /// on FieldRead-produced array bases.
    pub struct_fields: StructFieldRegistry,
    /// RPython: `_immutable_fields_ = [...]` declared on a class body.
    /// Maps struct name → `(field_name, rank)` pairs whose value never
    /// mutates after construction (or is quasi-immutable).  Both bare and
    /// qualified struct keys are inserted (mirroring `struct_fields`) so
    /// the same lookup logic works across module-prefix variants.  Rank
    /// encoding follows `rpython/rtyper/rclass.py:644-678 _parse_field_list`.
    pub immutable_fields: HashMap<String, Vec<(String, ImmutableRank)>>,
    /// Enum discriminant → variant name, keyed by enum type (both the
    /// qualified path and the bare leaf, mirroring `struct_fields`).
    /// The opcode-dispatch MIR extractor reads
    /// `enum_variant_by_discriminant["Instruction"]` to turn a switch
    /// case value (`ExitCase::Const(Int(K))`, the variant discriminant —
    /// which is *not* the variant index) back into the variant name.
    pub enum_variant_by_discriminant: HashMap<String, HashMap<i64, String>>,
    /// `bare_struct_name → defining crate-relative module path`,
    /// harvested from the LLBC `iter_type_decls()` name paths.
    /// Feeds `majit_ir::descr::STRUCT_ORIGIN_REGISTRY` so
    /// `canonical_struct_name` resolves a bare leaf to the qualified
    /// `module::Bare` key the runtime's
    /// `build_object_descr_group_with_def_path` dual-publishes (the
    /// crate prefix is stripped to match that def-path convention).
    pub struct_origins: HashMap<String, String>,
    /// `crate-relative qualified struct name → declaration-ordered
    /// `(field, ValueType)` register classes`, harvested from the LLBC
    /// `iter_type_decls()` struct field types.  Feeds
    /// `annotator::classdesc::register_struct_fields` →
    /// `FORCE_ATTRIBUTES_INTO_CLASSES` so `ClassDef::_init_classdef`
    /// pre-fills `ClassDef.attrs` before the annotator's
    /// `attrs_populated` narrowing gate.  Key drops the crate prefix to
    /// match the qualname `_init_classdef` reads; primitive fields carry
    /// `Int`/`Unsigned`/`Bool`/`Float`, every other shape `Ref(None)`.
    pub struct_field_attrs: HashMap<String, Vec<(String, crate::model::ValueType)>>,
    /// `(path-segments, Signature, return-lltype)` for every local
    /// `unsafe fn` / unsafe impl-method whose return type resolves to
    /// unit or bool, harvested from the LLBC by
    /// `front::mir::collect_unsafe_fn_stubs_from_llbc`.  Feeds
    /// `CallControl.unsafe_fn_stubs` →
    /// `cutover::register_unsafe_fn_stubs` so the dual gate registers a
    /// stub PyGraph for each, covering the "not registered in
    /// PyreCallRegistry" Skip cluster dominated by `pyre_object::is_*`.
    /// These callees' bodies access raw pointers the flowspace adapter
    /// does not model, so only a typed signature stub is registered.
    pub unsafe_fn_stubs: Vec<(
        Vec<String>,
        crate::flowspace::argument::Signature,
        crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
    )>,
}

/// Graph lookup table built from a `SemanticProgram` so the
/// registration loops in `lib.rs` and the opcode-dispatch extractor in
/// `front::mir_dispatch` can fetch the MIR-built graph for a given
/// (impl_type or trait_root, method) pair by name.
///
/// Callers spell `self_ty_root` two ways: a qualified owner
/// ("pyframe::PyFrame"), or — for a top-level `impl Drop for PyFrame`
/// reached through `for_type` — the bare leaf "PyFrame".  The MIR
/// driver always stores the module-qualified spelling.  To bridge the
/// asymmetry without forcing callers to re-qualify, the lookup indexes
/// every impl method TWICE: once by qualified owner, once by the bare
/// leaf (rsplit on "::").  Bare-leaf collisions across distinct types
/// (e.g. `Drop::drop` on both `PyFrame` and `Other`) are tracked as
/// ambiguous and return None — the caller must then qualify.
pub struct MirGraphLookup<'a> {
    /// Impl methods (inherent + trait-impl): keyed by (self_ty_root, name).
    /// `Ok(&graph)` is a unique hit; `Err(())` marks the slot ambiguous
    /// (two or more graphs share the (owner-spelling, name) tuple).
    impl_methods: HashMap<(&'a str, &'a str), Result<&'a FunctionGraph, ()>>,
    /// Trait-default bodies: keyed by (trait_root, name) with self_ty_root None.
    /// `Ok(&graph)` is a unique hit; `Err(())` marks the slot ambiguous
    /// (two distinct traits share a bare leaf + default-method name), so
    /// the caller falls back rather than registering an arbitrary body.
    trait_defaults: HashMap<(&'a str, &'a str), Result<&'a FunctionGraph, ()>>,
    /// Free functions (no impl owner, no trait root): keyed by bare name.
    /// `Ok(&graph)` is a unique hit; `Err(())` marks the slot ambiguous
    /// (two or more free functions share a bare name across modules).
    /// Lets the opcode-dispatch extractor resolve `execute_opcode_step`
    /// and each `execute_<op>` handler graph from the MIR program by
    /// name.
    free_functions: HashMap<&'a str, Result<&'a FunctionGraph, ()>>,
}

impl<'a> MirGraphLookup<'a> {
    /// Build the lookup by walking `program.functions` once.  The
    /// borrows are tied to `program`'s lifetime, so the caller must
    /// keep `program` alive for the duration of the lookup's use.
    pub fn from_program(program: &'a SemanticProgram) -> Self {
        let mut impl_methods: HashMap<(&'a str, &'a str), Result<&'a FunctionGraph, ()>> =
            HashMap::new();
        let mut trait_defaults: HashMap<(&'a str, &'a str), Result<&'a FunctionGraph, ()>> =
            HashMap::new();
        let mut free_functions: HashMap<&'a str, Result<&'a FunctionGraph, ()>> = HashMap::new();
        for f in &program.functions {
            if let Some(owner) = f.self_ty_root.as_deref() {
                Self::insert_or_mark_ambiguous(&mut impl_methods, owner, f.name.as_str(), &f.graph);
                // Also index by the bare leaf for callers that pass an
                // unqualified owner (e.g. top-level `impl Drop for
                // PyFrame` reached through `for_type`).  Bare leaf is the
                // last "::"-separated segment; identical to qualified
                // when self_ty_root has no module prefix.
                let leaf = owner.rsplit("::").next().unwrap_or(owner);
                if leaf != owner {
                    Self::insert_or_mark_ambiguous(
                        &mut impl_methods,
                        leaf,
                        f.name.as_str(),
                        &f.graph,
                    );
                }
            } else if let Some(tr) = f.trait_root.as_deref() {
                // Mark bare-leaf trait-name collisions ambiguous, mirroring
                // the impl_methods / free_functions tables, so two distinct
                // traits with a same-named default method do not last-win.
                Self::insert_or_mark_ambiguous(&mut trait_defaults, tr, f.name.as_str(), &f.graph);
            } else {
                // Free function: index by bare name so the
                // opcode-dispatch extractor can resolve
                // `execute_opcode_step` and each `execute_<op>` handler.
                Self::insert_free_or_mark_ambiguous(&mut free_functions, f.name.as_str(), &f.graph);
            }
        }
        Self {
            impl_methods,
            trait_defaults,
            free_functions,
        }
    }

    fn insert_or_mark_ambiguous(
        map: &mut HashMap<(&'a str, &'a str), Result<&'a FunctionGraph, ()>>,
        owner: &'a str,
        name: &'a str,
        graph: &'a FunctionGraph,
    ) {
        use std::collections::hash_map::Entry;
        match map.entry((owner, name)) {
            Entry::Vacant(v) => {
                v.insert(Ok(graph));
            }
            Entry::Occupied(mut o) => {
                let existing = *o.get();
                if let Ok(g0) = existing {
                    // Same FunctionGraph reference is fine (same entry
                    // visited via dual-key insert); only mark ambiguous
                    // when the pointer differs.
                    if !std::ptr::eq(g0, graph) {
                        let _ = o.insert(Err(()));
                    }
                }
                // already Err(()): stays ambiguous.
            }
        }
    }

    fn insert_free_or_mark_ambiguous(
        map: &mut HashMap<&'a str, Result<&'a FunctionGraph, ()>>,
        name: &'a str,
        graph: &'a FunctionGraph,
    ) {
        use std::collections::hash_map::Entry;
        match map.entry(name) {
            Entry::Vacant(v) => {
                v.insert(Ok(graph));
            }
            Entry::Occupied(mut o) => {
                if let Ok(g0) = *o.get() {
                    if !std::ptr::eq(g0, graph) {
                        let _ = o.insert(Err(()));
                    }
                }
                // already Err(()): stays ambiguous.
            }
        }
    }

    /// Returns the MIR graph for a free function (no impl owner, no
    /// trait root) by bare name.  Returns None when the name does not
    /// resolve to a unique graph (no entry, or two modules share the
    /// bare name).
    pub fn lookup_free(&self, name: &str) -> Option<&'a FunctionGraph> {
        match self.free_functions.get(name).copied()? {
            Ok(g) => Some(g),
            Err(()) => None,
        }
    }

    /// Returns the MIR graph for an inherent or trait-impl method.
    /// Returns None when the (owner, name) tuple does not resolve to
    /// a unique graph (either no entry or ambiguous bare-leaf).
    pub fn lookup_impl_method(&self, impl_type: &str, name: &str) -> Option<&'a FunctionGraph> {
        match self.impl_methods.get(&(impl_type, name)).copied()? {
            Ok(g) => Some(g),
            Err(()) => None,
        }
    }

    /// Returns the MIR graph for a trait-default body.  Returns None
    /// when the (trait_root, name) tuple does not resolve to a unique
    /// graph (no entry or ambiguous bare-leaf trait name).
    pub fn lookup_trait_default(&self, trait_root: &str, name: &str) -> Option<&'a FunctionGraph> {
        match self.trait_defaults.get(&(trait_root, name)).copied()? {
            Ok(g) => Some(g),
            Err(()) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FunctionGraph;

    fn free_fn(name: &str) -> SemanticFunction {
        SemanticFunction {
            name: name.into(),
            graph: FunctionGraph::new(name),
            return_type: None,
            self_ty_root: None,
            module_path: String::new(),
            hints: Vec::new(),
            access_directly: false,
            trait_root: None,
        }
    }

    fn impl_method(owner: &str, name: &str) -> SemanticFunction {
        SemanticFunction {
            self_ty_root: Some(owner.into()),
            ..free_fn(name)
        }
    }

    fn program(functions: Vec<SemanticFunction>) -> SemanticProgram {
        SemanticProgram {
            functions,
            ..Default::default()
        }
    }

    #[test]
    fn lookup_free_resolves_unique_free_function() {
        let prog = program(vec![
            free_fn("execute_opcode_step"),
            free_fn("execute_pop_top"),
            impl_method("PyFrame", "push"),
        ]);
        let lookup = MirGraphLookup::from_program(&prog);
        assert!(lookup.lookup_free("execute_opcode_step").is_some());
        assert!(lookup.lookup_free("execute_pop_top").is_some());
        // An impl method is not a free function.
        assert!(lookup.lookup_free("push").is_none());
        // An unknown name resolves to nothing.
        assert!(lookup.lookup_free("execute_nope").is_none());
    }

    #[test]
    fn lookup_free_returns_none_on_ambiguous_bare_name() {
        // Two free functions sharing a bare name (e.g. the same helper
        // name in two modules) must not bind either graph.
        let prog = program(vec![free_fn("helper"), free_fn("helper")]);
        let lookup = MirGraphLookup::from_program(&prog);
        assert!(lookup.lookup_free("helper").is_none());
    }
}
