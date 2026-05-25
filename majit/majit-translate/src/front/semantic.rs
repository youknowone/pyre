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
    /// RPython: op.result.concretetype — whole-program function return types.
    /// Maps exact callee path (e.g. "a::helper", "Type::method") → return type.
    /// Stored here so that downstream consumers (parse.rs method graph building)
    /// can use them for array type identity resolution.
    pub fn_return_types: HashMap<String, String>,
    /// RPython: `_immutable_fields_ = [...]` declared on a class body.
    /// Maps struct name → `(field_name, rank)` pairs whose value never
    /// mutates after construction (or is quasi-immutable).  Both bare and
    /// qualified struct keys are inserted (mirroring `struct_fields`) so
    /// the same lookup logic works across module-prefix variants.  Rank
    /// encoding follows `rpython/rtyper/rclass.py:644-678 _parse_field_list`.
    pub immutable_fields: HashMap<String, Vec<(String, ImmutableRank)>>,
    /// Whole-program `pub const` / `pub static` declarations gathered
    /// across every parsed file.  Keyed by `(module_path, name)` so
    /// the same bare name (`INT_TYPE`) can disambiguate between
    /// different defining modules.  Mirrors PyPy's
    /// `bookkeeper.getdesc(TYPE)` whole-program registry — pyre
    /// carries the data per-`(module, name)` because Rust has no
    /// `lltype` object identity to key off.
    ///
    /// Populated by [`collect_program_metadata_pub`] / the per-file
    /// build entries from each [`ParsedInterpreter::module_statics`].
    pub module_statics: HashMap<(String, String), crate::parse::ModuleStaticDecl>,
}

/// Pre-walk metadata produced by `collect_program_metadata_pub` —
/// the four registries that `build_function_graph` /
/// `build_function_graph_with_self_ty_pub` need before per-function
/// graph build can resolve typed call shapes.
pub struct ProgramMetadata {
    pub known_struct_names: std::collections::HashSet<String>,
    pub known_trait_names: std::collections::HashSet<String>,
    pub struct_fields: StructFieldRegistry,
    pub fn_return_types: HashMap<String, String>,
    /// Bare struct name → defining module path (use-import resolver
    /// support).  Populated when `collect_struct_names` walks per-file
    /// `ParsedInterpreter.module_path` non-empty: each top-level
    /// `Struct` registers as `struct_origins["Struct"] = module_path`.
    /// PyPy parity: `annotator.bookkeeper.getdesc(TYPE)` resolves the
    /// canonical defining-module path for every lltype reference;
    /// pyre carries names as strings so this map carries that
    /// resolution.  Empty when every parsed file was supplied via the
    /// bare `parse_source` entry — caller falls back to the
    /// dual-publish runtime convergence.
    pub struct_origins: HashMap<String, String>,
    /// Merged use-import table across all parsed files: each entry
    /// `(file_module_path, alias) → fully_qualified_path` mirrors the
    /// per-file `ParsedInterpreter.use_imports` populated by
    /// `parse::collect_use_imports`.  Keyed by `(module, alias)` rather
    /// than `alias` alone because the same alias `Foo` can resolve to
    /// different paths in different files (`use other_a::Foo` in one
    /// vs `use other_b::Foo` in another).
    pub use_imports: HashMap<(String, String), String>,
    /// Merged module-static table across all parsed files: each entry
    /// `(file_module_path, static_name) → ModuleStaticDecl` mirrors the
    /// per-file `ParsedInterpreter.module_statics` populated by
    /// `parse::collect_module_statics`.  Keyed by `(module, name)` —
    /// the same bare static name (e.g. `LOCAL`) can appear in multiple
    /// files; the per-file key disambiguates.
    pub module_statics: HashMap<(String, String), crate::parse::ModuleStaticDecl>,
}

/// Concatenate a file's `module_path` with an inline-`mod` chain
/// (the `nested` half of a `parsed.module_statics` key) into the
/// program-wide module-static lookup key used by `lookup_module_
/// static_literal`.  Either component may be empty.
pub fn qualify_module_path(module_path: &str, nested: &str) -> String {
    match (module_path.is_empty(), nested.is_empty()) {
        (true, true) => String::new(),
        (false, true) => module_path.to_string(),
        (true, false) => nested.to_string(),
        (false, false) => format!("{}::{}", module_path, nested),
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
