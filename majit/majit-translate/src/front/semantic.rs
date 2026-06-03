//! Front-end shared types — the data shapes both `front::ast` and
//! `front::mir` produce, and that the rest of the pipeline
//! (`analyze_pipeline_from_parsed`, `jit_codewriter::*`, `parse::*`)
//! consumes.
//!
//! Carving these out of `front::ast` is the first step toward
//! retiring the AST graph builder (issue #97 Step 6.E/6.F): the
//! types do not depend on the syn-tree walker, so they belong
//! outside the builder that is about to disappear.
//!
//! Nothing in this module performs lowering.  Builders live in
//! `front::ast` (legacy syn-AST graph builder) and `front::mir`
//! (Charon ULLBC driver, the default path under
//! `mir-frontend`).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::model::{FunctionGraph, ImmutableRank, UnknownKind};

/// Options carried through the AST graph builder.  Retained as a
/// distinct unit type so the legacy `build_semantic_program_with_options`
/// API can keep accepting an explicit options parameter while
/// preserving the upstream `build_flow_graph` call shape.
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

/// Legacy alias: callers that pre-date the `FlowingError` / `Lowered`
/// split in this file still reference `LoweringAbort`.  Kept as a
/// type alias so renames can ride a separate commit.
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
    /// Added 2026-05-25 as the missing piece for Step 6.E:
    /// `parse::extract_trait_impls` currently carries the trait_name
    /// in a parallel `TraitImplInfo` record so the registration loop
    /// in `lib.rs:905-1019` can distinguish trait-impl methods (which
    /// need `register_trait_method`) from inherent methods (which
    /// need `register_function_graph`).  Surfacing it here lets that
    /// loop walk `program.functions` directly and the AST-side
    /// extractors retire.
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
    /// that bypass `analyze_pipeline_from_parsed`'s
    /// `register_struct_origins`.
    pub fn field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        self.lookup_fields(owner)?
            .iter()
            .find(|(name, _)| name == field_name)
            .map(|(_, ty)| ty.as_str())
    }

    /// Per-scope `field_type` lookup: route `owner` through the call
    /// site's `use_imports` + `module_prefix` first (PyPy
    /// `frame.f_globals` analog) so a bare receiver leaf lands at the
    /// canonical key before the program-wide bookkeeper fallback.
    pub fn field_type_in_scope(
        &self,
        owner: &str,
        field_name: &str,
        prefix: &str,
        use_imports: &HashMap<String, String>,
    ) -> Option<&str> {
        let canonical_owner = qualify_type_name_with_imports(owner, prefix, use_imports);
        self.field_type(&canonical_owner, field_name)
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
}

/// Step 6.E Slice 3.C — graph lookup table built from a
/// `SemanticProgram` so the AST-side `extract_trait_impls` /
/// `extract_inherent_impl_methods` collectors can skip
/// `build_function_graph_with_self_ty_pub` when the MIR builder already
/// produced a graph for the same (impl_type or trait_root, method)
/// pair.  Cuts the largest single AST-graph-builder consumer in the
/// MIR-covered surface and is a prerequisite for retiring the
/// AST graph builder bulk under issue #97 Step 6.F.
///
/// AST callers spell `self_ty_root` two ways depending on extraction
/// path: inherent extract qualifies through `module_path` ("pyframe::
/// PyFrame"), but trait-impl extract uses prefix="" so a top-level
/// `impl Drop for PyFrame` yields the bare leaf "PyFrame".  MIR
/// always stores the module-qualified spelling.  To bridge the
/// asymmetry without forcing AST callers to re-qualify, the lookup
/// indexes every impl method TWICE: once by qualified owner, once by
/// the bare leaf (rsplit on "::").  Bare-leaf collisions across
/// distinct types (e.g. `Drop::drop` on both `PyFrame` and
/// `Other`) are tracked as ambiguous and return None — the caller
/// must then qualify.
pub struct MirGraphLookup<'a> {
    /// Impl methods (inherent + trait-impl): keyed by (self_ty_root, name).
    /// `Ok(&graph)` is a unique hit; `Err(())` marks the slot ambiguous
    /// (two or more graphs share the (owner-spelling, name) tuple).
    impl_methods: HashMap<(&'a str, &'a str), Result<&'a FunctionGraph, ()>>,
    /// Trait-default bodies: keyed by (trait_root, name) with self_ty_root None.
    trait_defaults: HashMap<(&'a str, &'a str), &'a FunctionGraph>,
}

impl<'a> MirGraphLookup<'a> {
    /// Build the lookup by walking `program.functions` once.  The
    /// borrows are tied to `program`'s lifetime, so the caller must
    /// keep `program` alive for the duration of the lookup's use.
    pub fn from_program(program: &'a SemanticProgram) -> Self {
        let mut impl_methods: HashMap<(&'a str, &'a str), Result<&'a FunctionGraph, ()>> =
            HashMap::new();
        let mut trait_defaults = HashMap::new();
        for f in &program.functions {
            if let Some(owner) = f.self_ty_root.as_deref() {
                Self::insert_or_mark_ambiguous(&mut impl_methods, owner, f.name.as_str(), &f.graph);
                // Also index by the bare leaf for AST callers (e.g.
                // top-level `impl Drop for PyFrame`) whose
                // self_ty_root is unqualified.  Bare leaf is the
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
                trait_defaults.insert((tr, f.name.as_str()), &f.graph);
            }
        }
        Self {
            impl_methods,
            trait_defaults,
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

    /// Returns the MIR graph for an inherent or trait-impl method.
    /// Returns None when the (owner, name) tuple does not resolve to
    /// a unique graph (either no entry or ambiguous bare-leaf).
    pub fn lookup_impl_method(&self, impl_type: &str, name: &str) -> Option<&'a FunctionGraph> {
        match self.impl_methods.get(&(impl_type, name)).copied()? {
            Ok(g) => Some(g),
            Err(()) => None,
        }
    }

    /// Returns the MIR graph for a trait-default body.
    pub fn lookup_trait_default(&self, trait_root: &str, name: &str) -> Option<&'a FunctionGraph> {
        self.trait_defaults.get(&(trait_root, name)).copied()
    }
}

/// Qualify a bare type name with module prefix or, when the resolver
/// knows the canonical defining module, with the canonical prefix.
///
/// Resolves a per-source `use <path> as alias` table first, then the
/// program-wide `STRUCT_ORIGIN_REGISTRY` canonical-defining-module
/// table, then falls back to `prefix::bare`.
///
/// `bookkeeper.py:353-409 getdesc` resolves a bare identifier first in
/// the frame's `f_globals` (the file's own imports), then in the
/// program-wide scope summary; pyre's `STRUCT_ORIGIN_REGISTRY` plays
/// the role of the program-wide scope, while `use_imports` carries
/// the per-source `f_globals` slice.
///
/// `use_imports` is expected to be `GraphBuildContext.use_imports` —
/// each entry maps a local alias (`use other_mod::Foo as Q` →
/// `Q → other_mod::Foo`, plain `use other_mod::Foo` →
/// `Foo → other_mod::Foo`) to the fully-qualified path.  Pass
/// `&HashMap::new()` when the call site has no per-source scope
/// (parse-time registration, test fixtures, `lower_expr_into_graph`);
/// resolution then reduces to `STRUCT_ORIGIN_REGISTRY` + `prefix::bare`.
pub fn qualify_type_name_with_imports(
    bare: &str,
    prefix: &str,
    use_imports: &HashMap<String, String>,
) -> String {
    if bare.contains("::") {
        return bare.to_string();
    }
    if let Some(full) = use_imports.get(bare) {
        return full.clone();
    }
    if prefix.is_empty() {
        return bare.to_string();
    }
    let canonical = majit_ir::descr::canonical_struct_name(bare);
    if canonical != bare {
        return canonical;
    }
    format!("{}::{}", prefix, bare)
}
