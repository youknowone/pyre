//! Front-end shared types — the data shapes `front::mir` produces, and
//! that the rest of the pipeline (`analyze_pipeline_from_module_paths`,
//! `codewriter::*`, `parse::*`) consumes.
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
/// RPython `rpython/flowspace/flowcontext.py` raises `FlowingError`
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
    /// Charon's concrete trait-impl identity when this function comes from an
    /// `Impl { Trait: id }` name segment. RPython `CallControl.graphs_from`
    /// obtains the graph from the function object; the Rust port uses this id
    /// to keep same-named impl methods distinct in `CallPath`.
    pub trait_impl_id: Option<u64>,
    /// Module path of the defining file, crate-stripped (e.g.
    /// `"pyframe"` for `pyre-interpreter/src/pyframe.rs`), populated by
    /// `front::mir` from the module portion of Charon's `name_path()`.
    /// Empty when the producer did not supply a module path — top-level
    /// items remain at simple-name registration.
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
    /// Trait name when this function is an `impl Trait for Type {…}`
    /// method, the trait's name when this is a trait default-body
    /// method, otherwise `None` (free function or inherent impl).
    ///
    /// Lets the registration loop in `lib.rs` walk
    /// `program.functions` directly and distinguish trait-impl methods
    /// (which need `register_trait_method`) from inherent methods
    /// (which need `register_function_graph`).
    pub trait_root: Option<String>,
    /// Fully-qualified `name_path()` of the trait when this function
    /// is an `impl Trait for Type {…}` method, otherwise `None`.
    /// Distinguishes traits whose leaf names collide — the unique-impl
    /// map (`trait_unique_impls`) keys on this, not `trait_root`.
    /// Trait default bodies leave it `None`: Charon names them with
    /// only the trait leaf segment, and they never feed the
    /// unique-impl map.
    pub trait_qualified: Option<String>,
    /// `true` when the function returns `*mut PyObject` (a `PyObjectRef`),
    /// detected structurally by `front::mir::output_type_is_objectptr`.
    /// The MIR driver leaves every `return_type` `None`, so a
    /// `dont_look_inside` callee returning an object pointer would
    /// residualize as a `None`→`Void` stub (a miscompile — the caller
    /// needs the pointer).  `merge_hints_from_llbcs` reads this flag and,
    /// for a `dont_look_inside` callee, stamps the object-pointer
    /// `return_type` marker so the residual prefill projects a `Ref`
    /// result.
    pub returns_objectptr: bool,
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
    /// suffix-match shim absorbs `pyre_object::functional::W_X` vs
    /// `functional::W_X` divergence — orthogonal to lexical scope
    /// resolution, kept for test entries (`parse::parse_source`)
    /// that bypass `analyze_pipeline_from_module_paths`'s
    /// `register_struct_origins`.
    pub fn field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        self.lookup_fields(owner)?
            .iter()
            .find(|(name, _)| name == field_name)
            .map(|(_, ty)| ty.as_str())
    }

    /// True when `owner` is registered as an enum base class — its sole
    /// row is the synthetic `__discriminant` tag (`rclass.py:499-518`: the
    /// sum-type base carries only the discriminant, each variant subclass
    /// carries its own payload fields under `{enum}::{variant}` keys).  A
    /// struct or non-enum owner has its own field rows and returns false.
    ///
    /// The discriminator is the row shape, not a stored type-kind flag,
    /// because the registry erases `TypeDeclKind` once rows are projected.
    /// That is sound: `__discriminant` is a reserved key emitted only by
    /// the enum-base row builder in `front/mir.rs`; a `TypeDeclKind::Struct`
    /// projects its real Rust field names, none of which is `__discriminant`
    /// (`from_type_strings` skips synthetic `__pad`/typeptr rows and never
    /// mints that name).  So a real single-field struct cannot be
    /// misclassified as an enum base.
    pub fn is_enum_base(&self, owner: &str) -> bool {
        self.lookup_fields(owner)
            .is_some_and(|rows| matches!(rows, [(name, _)] if name == "__discriminant"))
    }

    /// Whether `owner` itself, or one of its enum variants, declares
    /// `field_name`.
    ///
    /// Rust keeps fields and methods in separate namespaces, while the
    /// RPython class model used by the annotator has one attribute namespace.
    /// A method on an enum base therefore also collides with a same-named
    /// payload field on a variant: seeding that method on the base would make
    /// the variant constructor inherit a function as the field's default.
    pub fn owner_or_variant_has_field(&self, owner: &str, field_name: &str) -> bool {
        if self.field_type(owner, field_name).is_some() {
            return true;
        }
        if !self.is_enum_base(owner) {
            return false;
        }
        let canonical_owner = majit_ir::descr::canonical_struct_name(owner);
        self.fields.iter().any(|(key, rows)| {
            let Some((parent, _variant)) = key.rsplit_once("::") else {
                return false;
            };
            majit_ir::descr::canonical_struct_name(parent) == canonical_owner
                && rows.iter().any(|(field, _)| field == field_name)
        })
    }

    fn lookup_fields(&self, owner: &str) -> Option<&[(String, String)]> {
        // A per-instantiation enum spelling (`Result<Tuple>`,
        // `Result<Tuple>::Ok`) shares the bare template's rows: the
        // reference-payload split exists only to separate annotator attr
        // unification across instantiations, not to give each one its own
        // row layout.  Drop the `<…>` argument span (keeping any trailing
        // `::variant`) so the lookup resolves under the bare key the
        // template registered.
        let stripped = majit_ir::descr::strip_generic_args(owner);
        let owner = stripped.as_ref();
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

/// Exact memory layout of a type, resolved from Charon's per-target
/// `variant_layouts.field_offsets` in the LLBC (not the `#[repr(C)]`-
/// approximating heuristic).  Keyed in [`SemanticProgram::exact_layouts`]
/// by owner root: a struct leaf / qualified name, or — for an enum
/// variant — `{leaf}::{variant}`.  `field_offsets` carries the field's
/// byte offset within the type; the heuristic provider is used only for
/// roots without an entry here.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExactLayout {
    /// Total byte size when Charon resolved it.
    pub size: Option<u64>,
    /// Byte alignment when Charon resolved it.
    pub align: Option<u64>,
    /// `field_name → byte offset within the type`.
    pub field_offsets: HashMap<String, u64>,
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
    /// encoding follows `rpython/rtyper/rclass.py _parse_field_list`.
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
    /// Exact per-type memory layout (byte offsets, size, align) harvested
    /// from Charon's LLBC `type_decl.layout`, keyed by the type's
    /// [`StructId`](majit_ir::descr::StructId) object identity (one entry
    /// per type definition, enum variants included).  Feeds the exact
    /// `LayoutProvider`, replacing the heuristic for types that have an
    /// entry; the heuristic remains the fallback.  Keying on the identity
    /// token rather than a name string makes two distinct definitions
    /// that share a leaf name structurally distinct, removing the
    /// last-writer-wins collision the bare-leaf string key had.
    pub exact_layouts: HashMap<majit_ir::descr::StructId, ExactLayout>,
    /// Resolves a struct / enum-variant name in any spelling (full-crate,
    /// crate-stripped, bare leaf, variant analogs) back to its canonical
    /// [`StructId`](majit_ir::descr::StructId), or `None` for a bare leaf
    /// two distinct modules share.  Registered into
    /// `majit_ir::descr::STRUCT_ID_BY_NAME` so the layout consumers that
    /// only hold a string (`llmemory::FieldOffset`'s `st._name`, a nested
    /// field's rendered type) can reach the identity-keyed layout maps.
    pub struct_ids: HashMap<String, Option<majit_ir::descr::StructId>>,
    /// `(path-segments, Signature, return-lltype)` for every local
    /// `unsafe fn` / unsafe impl-method whose return type resolves to
    /// unit or bool, harvested from the LLBC by
    /// `front::mir::collect_unsafe_fn_stubs_from_llbc`.  Feeds
    /// `CallControl.unsafe_fn_stubs` →
    /// `cutover::register_unsafe_fn_stubs` so the dual gate registers a
    /// stub PyGraph for each, covering the "not registered in
    /// CallRegistry" Skip cluster dominated by `pyre_object::is_*`.
    /// These callees' bodies access raw pointers the flowspace adapter
    /// does not model, so only a typed signature stub is registered.
    pub unsafe_fn_stubs: Vec<(
        Vec<String>,
        crate::flowspace::argument::Signature,
        Option<String>,
    )>,
    /// `(path-segments, Signature, result ValueType)` for every method on
    /// a foreign **opaque** ADT owner (`malachite_bigint::bigint::BigInt`,
    /// …) whose result is faithfully modelable, harvested from the LLBC by
    /// `front::mir::collect_foreign_opaque_method_externals`.  Feeds
    /// `CallControl.foreign_opaque_method_externals` →
    /// `cutover::register_foreign_opaque_method_externals` so the
    /// `CallTarget::FunctionPath` form (which `impl_method_owner` falls
    /// back to for an opaque owner) resolves instead of panicking
    /// `SomeInstance.getattr` on the classdef-less receiver — the
    /// `register_external` / `@jit.dont_look_inside` analog.
    pub foreign_opaque_method_externals: Vec<(
        Vec<String>,
        crate::flowspace::argument::Signature,
        crate::model::ValueType,
    )>,
}

/// Graph lookup table built from a `SemanticProgram` so registration and
/// codewriter graph discovery can fetch the MIR-built graph for a given
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
    /// Lets ordinary free-function registration and graph discovery resolve
    /// a unique MIR-built graph by its unqualified name.
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
                if let Ok(g0) = *o.get()
                    && !std::ptr::eq(g0, graph)
                {
                    let _ = o.insert(Err(()));
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
        self.free_functions.get(name).copied()?.ok()
    }

    /// Returns the MIR graph for an inherent or trait-impl method.
    /// Returns None when the (owner, name) tuple does not resolve to
    /// a unique graph (either no entry or ambiguous bare-leaf).
    pub fn lookup_impl_method(&self, impl_type: &str, name: &str) -> Option<&'a FunctionGraph> {
        self.impl_methods.get(&(impl_type, name)).copied()?.ok()
    }

    /// Returns the MIR graph for a trait-default body.  Returns None
    /// when the (trait_root, name) tuple does not resolve to a unique
    /// graph (no entry or ambiguous bare-leaf trait name).
    pub fn lookup_trait_default(&self, trait_root: &str, name: &str) -> Option<&'a FunctionGraph> {
        self.trait_defaults.get(&(trait_root, name)).copied()?.ok()
    }
}

/// RPython `specialize.py default_specialize` — the LLBC path's stand-in.
///
/// Upstream writes `graph.access_directly = True` for a graph whose ARGUMENT
/// annotation arrived carrying the flag that the `hint` ExtRegistryEntry in
/// `rlib/jit.py` mints on `SomeInstance.flags`. That flag travels
/// callee-ward through the annotator, so the function that SPELLS the hint is
/// not the flagged one: its own arguments arrive unflagged. What gets flagged
/// is every callee the hinted value is passed to.
///
/// The LLBC front end has no annotator to carry a flag on an annotation, so
/// the propagation is done here, over the op stream: seed on the
/// `OpKind::Hint` ops, follow the hinted variables into the arguments of each
/// `OpKind::Call`, and iterate until the flagged-parameter sets stop growing.
///
/// Two deliberate under-approximations, both because the consumer
/// (`policy::look_inside_graph`) ABORTS translation on a flag it cannot
/// honour, so a missing flag costs a safety net while an extra one costs the
/// build:
///
/// * **Reached both ways stays unflagged.** `default_specialize` SPECIALIZES:
///   the flagged callee is a second graph keyed `(AccessDirect, <base key>)`
///   and unflagged callers keep the original. One graph per function cannot
///   split, so a parameter is flagged only when EVERY resolved call site
///   passes a hinted value there.
///
///   Porting that second graph would not make the flag mean more here, so do
///   not read this clause as a deferred precision fix. Upstream's flag has
///   teeth in `rvirtualizable.replace_force_virtualizable_with_call`, which
///   DROPS an injected `jit_force_virtualizable` carrying `access_directly`
///   instead of rewriting it to the interpreter copy's `direct_call`
///   (`hook_access_field`'s own test of the flag is commented out upstream, so
///   that pass is where it is read). Pyre injects nothing: the interpreter is
///   native Rust the translator does not build, so there is one copy, and an
///   injected op could only land in the jitcode — the copy
///   `rewrite_op_jit_force_virtualizable` deletes from. Nor can the
///   hand-placed `executioncontext::jit_force_virtualizable` marker be made
///   flag-conditioned in the direction that would matter: the residual copy is
///   compiled Rust, so nothing is removable from it at translation time. The
///   sole reader of `FunctionGraph::access_directly` in this tree is
///   `policy::look_inside_graph`'s abort, so flagging more graphs can only
///   widen that abort.
/// * **Alias closure follows links and representation casts only.** A
///   variable the hint produced, a parameter that arrives flagged, whatever
///   a `Link` binds one of those to in a successor block, and whatever a
///   `__cast_pointer` / `__cast_instance_intrinsic` marker derives from one
///   counts as hinted; nothing else does. Links are followed because
///   `annotator.py` propagates the annotation along them; the two markers
///   because `jtransform` rewrites them to `same_as`, so each is the one
///   variable an RPython flow graph would have had. A hinted value that
///   reaches a callee through anything else — a wrapper struct, an
///   accessor, a `Vec` — is not followed. `Rvalue::Use` needs no rule: the
///   front end reuses the operand's `Variable` rather than emitting a copy.
///
/// `default_specialize` also reads `_jit_look_inside_` off the callee before
/// it specializes: when that is False it deletes `access_directly` from the
/// argument annotation rather than keying a second graph, so a
/// `dont_look_inside` function is never flagged by any caller. The harvested
/// marker set is passed in and a callee it names is skipped, which is that
/// deletion — without it the pass would hand `policy::look_inside_graph` a
/// flag on exactly the graphs it aborts on.
///
/// `rlib/jit.py`'s `hint` entry only mints the flag for a `SomeInstance`
/// whose classdesc declares `_virtualizable_`, and deletes it on every other
/// value. `virtualizable_roots` is that declaration, registered out of band
/// by the consumer (`virtualizable_decl`), and `class_roots` is how far the
/// lowered graph records the analogue of `s_x.classdef`. A hint on a value
/// whose root is unrecorded or undeclared seeds nothing — the erasing branch.
///
/// A callee that does not resolve to a lowered function is skipped entirely
/// — it can neither be flagged nor veto a flag, since there is no graph for
/// the gate to read. A `FunctionPath` resolves by crate-stripped path; a
/// `Method` resolves by its stamped path when it has one and otherwise by
/// the `(receiver root, method leaf)` pair, and an ambiguous pair is treated
/// as unresolved.
///
/// The pass is inert in the production translation — it flags no graph at all
/// — but the reason is inside it, not upstream of it.
///
/// Re-measured by reproducing the prepass's own input: the canonical LLBC set
/// (`majit-rlib`, `pyre-object`, `pyre-interpreter`, `pyre-jit`; no sidecars
/// on a native build) and the module-path list derived from the same 296
/// source files. That lowers 33,976 functions carrying **18 `OpKind::Hint`
/// ops**, and every virtualizable hint site is present and correctly
/// classified:
///
/// ```text
/// pyframe::createframe_obj                     -> [AccessDirectly, FreshVirtualizable]
/// pyframe::<Impl>::finish_for_call_with_globals_obj -> [AccessDirectly, FreshVirtualizable]
/// eval::<Impl>::dispatch                       -> [AccessDirectly]
/// executioncontext::app_profile_call           -> [NoAccessDirectly]
/// executioncontext::<Impl>::_trace::<Impl>::call -> [NoAccessDirectly]
/// ```
///
/// This pass runs over that finished list (`front::mir` calls it once, after
/// the whole function set exists), so it sees all of them. With the roots
/// registry seeded as production seeds it — `roots=1 ["PyFrame"]` — it still
/// flags **zero** graphs. The hints arrive and the pass erases them.
///
/// An earlier census recorded here read zero `OpKind::Hint` ops and neither
/// frame constructor in the set. Both halves are wrong: the constructors are
/// in the set and carry their hints. Its conclusion survived its evidence,
/// which is why the reason mattered — an absent hint would put the fix
/// upstream, and an erased one puts it in the two under-approximations below.
/// They are live suspects today, not contingencies — and the producer of each
/// hinted value, which is what `class_roots` has to recognise, says which:
///
/// ```text
/// createframe_obj            AccessDirectly     <- SyntheticTransparentCtor{name:"PyFrame", is_struct}
/// finish_for_call_..._obj    AccessDirectly     <- SyntheticTransparentCtor{name:"PyFrame", is_struct}
/// eval::dispatch             AccessDirectly     <- Input{name:"frame", class_root:Some("PyFrame")}
/// app_profile_call           NoAccessDirectly   <- Input{name:"frame", class_root:Some("PyFrame")}
/// createframe_obj            FreshVirtualizable <- (no op's result: a block inputarg)
/// finish_for_call_..._obj    FreshVirtualizable <- (no op's result: a block inputarg)
/// ```
///
/// So the first under-approximation is NOT the eraser: every one of those four
/// resolves to the bare leaf `PyFrame`, which is the spelling the declared set
/// holds.
///
/// Nor does it erase `fresh_virtualizable`, though an earlier reading of that
/// table said so. Both sites spell
/// `hint_fresh_virtualizable(hint_access_directly(frame))`, and the outer
/// hint's operand is not any op's `result` — but that is a property of the
/// census, not of the graph. Every call here is a block terminator, so the
/// inner hint's result crosses a link and the outer operand arrives as the
/// successor's `inputarg`; a census matching operands against op results
/// cannot see one. The hint conversion does emit a result (`front::mir`), and
/// `class_roots`'s fixpoint carries a root across BOTH hops — the `Hint`
/// result and the `Link` into `inputargs`. The value is rooted and does seed.
/// It is also redundant while it does: `hint_seed_sets` treats
/// `FreshVirtualizable` and `AccessDirectly` identically and files operand and
/// result alike, so the two hints on one frame seed the same set.
///
/// (The chained spelling is still a deviation, for reasons that have nothing
/// to do with this pass: `rlib/jit.py` asserts `"lone fresh_virtualizable
/// hint"` on the outer call, upstream spelling both kwargs on one `hint()`
/// (`pyframe.py PyFrame.__init__`); and `jtransform.rs rewrite_op_hint` files
/// `vable_flags` against the operand while clearing the map per block, so a
/// chained hint files it one block away from the stores it covers.)
///
/// That leaves the second as the eraser, and as the only one. The flag is set
/// only through `reaching`: a `(callee, pos)` pair needs `flagged > 0 &&
/// unflagged == 0`, so one unhinted caller passing a frame at that position is
/// enough to erase it, and `PyFrame` has many. Seeding the roots registry
/// changes nothing — `roots=1 ["PyFrame"]` and an empty registry both flag
/// zero graphs — which is what rules the root declaration out as the cause.
///
/// * `class_roots` answers with the ctor's own name for a struct literal,
///   because that is the spelling everything else uses — `adt_node_class_root`
///   ends on `name.rsplit("::")`, and the declared set holds the bare leaf.
///   `front::mir` fills the same op's `result_ty` with `owner_path::leaf`, so
///   reading that instead would answer with a spelling no comparison matches.
///   A root reachable only through `result_ty` is still unrecorded, and a hint
///   on an unrecorded or undeclared root seeds nothing — the erasing branch.
///   This is the first thing to test against the five sites above.
///
/// * One graph per function, where upstream keys a second on
///   `(AccessDirect, key)`. A callee reached by both a hinted and an unhinted
///   caller cannot be flagged, since the one graph has to answer for both;
///   `reaching` counts the two and only flags a parameter no unflagged caller
///   supplies. That leans the safe way — `policy::look_inside_graph` aborts on
///   a flag it cannot honour — but it means a flag's absence is not evidence
///   the flag was not wanted.
pub(crate) fn propagate_access_directly(
    functions: &mut [SemanticFunction],
    dont_look_inside: &std::collections::HashSet<String>,
    virtualizable_roots: &std::collections::HashSet<String>,
) {
    use std::collections::{HashMap, HashSet};

    // `None` marks a path more than one lowered function answers to.
    // `cachedgraph(key)` upstream is keyed on the `FunctionDesc`, i.e. the
    // function object itself; a crate-stripped path is a weaker key, so a
    // collision has to be refused rather than silently resolved to whichever
    // function was collected last.
    let mut index: HashMap<String, Option<usize>> = HashMap::new();
    for (i, func) in functions.iter().enumerate() {
        index
            .entry(path_of(func))
            .and_modify(|slot| *slot = None)
            .or_insert(Some(i));
    }

    // A method call target names the receiver's type root and the method,
    // not a path, so it cannot be looked up in `index`. `call.rs
    // target_to_path` reaches the same graph through
    // `CallPath::for_impl_method(impl_type, name)`; here the same pair is
    // the key. `None` marks a pair more than one lowered function answers
    // to — an ambiguous callee is skipped exactly like an unresolvable one,
    // since flagging the wrong graph would arm the gate against a function
    // the hint never reached.
    let mut method_index: HashMap<(String, String), Option<usize>> = HashMap::new();
    for (i, func) in functions.iter().enumerate() {
        let Some(root) = &func.self_ty_root else {
            continue;
        };
        method_index
            .entry((root.clone(), leaf_of(&func.name)))
            .and_modify(|slot| *slot = None)
            .or_insert(Some(i));
    }

    // Everything below is derived once, because none of it depends on which
    // parameters are flagged: the call edges, the aliasing pairs a seed
    // travels along, and the seeds the hints themselves mint. Only the
    // flagged-parameter seeds change between rounds.
    let mut edges: Vec<Vec<(usize, usize, u64)>> = Vec::with_capacity(functions.len());
    let mut aliases: Vec<Vec<(u64, u64)>> = Vec::with_capacity(functions.len());
    let mut hint_seeds: Vec<HashSet<u64>> = Vec::with_capacity(functions.len());
    let mut killed: Vec<HashSet<u64>> = Vec::with_capacity(functions.len());
    for func in functions.iter() {
        aliases.push(alias_pairs(func));
        let (seeds, dead) = hint_seed_sets(func, virtualizable_roots);
        hint_seeds.push(seeds);
        killed.push(dead);
        let mut func_edges = Vec::new();
        for block in &func.graph.blocks {
            for op in &block.operations {
                let crate::model::OpKind::Call { target, args, .. } = &op.kind else {
                    continue;
                };
                let Some(callee) = resolve_callee(target, &index, &method_index) else {
                    continue;
                };
                // `default_specialize` reads `_jit_look_inside_` off the
                // callee and, when it is False, DELETES `access_directly`
                // from the argument annotation instead of specializing — so a
                // `dont_look_inside` graph is never flagged, no matter which
                // caller reaches it. Dropping the edge here is that deletion:
                // the argument neither flags the callee nor travels further
                // through it.
                if dont_look_inside.contains(&path_of(&functions[callee])) {
                    continue;
                }
                for (pos, arg) in args.iter().enumerate() {
                    func_edges.push((callee, pos, arg.id()));
                }
            }
        }
        edges.push(func_edges);
    }
    // Close the hint seeds and the killed set over the aliases once: neither
    // depends on which parameters are flagged, so neither changes between
    // rounds.
    //
    // The killed set is closed IN PLACE because the fixpoint below subtracts
    // it a second time, and both subtractions have to name the same set.  A
    // killed value that reaches a merge arrives there under a freshly minted
    // `inputarg`, and that same id is an alias target of whatever the other
    // arm passed — so the raw set, which holds only the hint's own result,
    // does not name it.  `binaryop.py pairtype(SomeInstance, SomeInstance)
    // .union` deletes any flag key the other side lacks, so a merge one arm
    // re-bound away carries no flag; closing the killed set is how that
    // intersection is reproduced here.
    for (i, seeds) in hint_seeds.iter_mut().enumerate() {
        close_ids_over_aliases(&aliases[i], seeds);
        close_ids_over_aliases(&aliases[i], &mut killed[i]);
        seeds.retain(|id| !killed[i].contains(id));
    }

    // Per function, the parameter positions that arrive carrying the flag.
    let mut flagged_params: Vec<Vec<usize>> = vec![Vec::new(); functions.len()];

    // `default_specialize` runs per call as the annotator reaches it, so a
    // flag one call introduces is visible to the next. Iterating to a
    // fixpoint is how a single pass over a static op stream reproduces that;
    // the bound is the number of functions, since each round either adds a
    // flagged parameter or stops.
    for _round in 0..=functions.len() {
        // (flagged, unflagged) call-site counts per callee parameter.
        let mut reaching: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        for (i, func) in functions.iter().enumerate() {
            let mut seeds = hint_seeds[i].clone();
            if !flagged_params[i].is_empty() {
                let inputargs = &func.graph.block(func.graph.startblock).inputargs;
                for &pos in &flagged_params[i] {
                    if let Some(arg) = inputargs.get(pos) {
                        seeds.insert(arg.id());
                    }
                }
                close_ids_over_aliases(&aliases[i], &mut seeds);
                seeds.retain(|id| !killed[i].contains(id));
            }
            for &(callee, pos, arg) in &edges[i] {
                let slot = reaching.entry((callee, pos)).or_insert((0, 0));
                if seeds.contains(&arg) {
                    slot.0 += 1;
                } else {
                    slot.1 += 1;
                }
            }
        }

        let mut next: Vec<Vec<usize>> = vec![Vec::new(); functions.len()];
        for ((callee, pos), (flagged, unflagged)) in reaching {
            if flagged > 0 && unflagged == 0 {
                next[callee].push(pos);
            }
        }
        for slots in &mut next {
            slots.sort_unstable();
        }
        if next == flagged_params {
            break;
        }
        flagged_params = next;
    }

    for (func, slots) in functions.iter_mut().zip(&flagged_params) {
        func.graph.access_directly = !slots.is_empty();
    }
}

fn path_of(f: &SemanticFunction) -> String {
    if f.module_path.is_empty() {
        f.name.clone()
    } else {
        format!("{}::{}", f.module_path, f.name)
    }
}

/// The seeds the hint ops themselves mint, and the values the
/// `access_directly=False` spelling strips.
///
/// `rlib/jit.py`'s `hint` entry sets the flag only when the value is a
/// `SomeInstance` whose classdesc declares `_virtualizable_`, and deletes it
/// on every other value. `virtualizable_roots` is that declaration set,
/// supplied by the consumer the same way the codewriter's `vable_fields` is,
/// and `class_roots` is how far the lowered graph records the analogue of
/// `s_x.classdef`. A value whose root the graph does not record is not seeded,
/// which is the erasing branch.
fn hint_seed_sets(
    func: &SemanticFunction,
    virtualizable_roots: &std::collections::HashSet<String>,
) -> (
    std::collections::HashSet<u64>,
    std::collections::HashSet<u64>,
) {
    let roots = class_roots(func);
    let mut seeds = std::collections::HashSet::new();
    let mut killed = std::collections::HashSet::new();
    for block in &func.graph.blocks {
        for op in &block.operations {
            let crate::model::OpKind::Hint { value, kind } = &op.kind else {
                continue;
            };
            match kind {
                // `rlib/jit.py` mints `fresh_virtualizable` only alongside
                // `access_directly`, so both spellings seed.
                crate::hints::HintKind::AccessDirectly
                | crate::hints::HintKind::FreshVirtualizable => {
                    let virtualizable = roots
                        .get(&value.id())
                        .is_some_and(|root| virtualizable_roots.contains(root));
                    if !virtualizable {
                        continue;
                    }
                    seeds.insert(value.id());
                    if let Some(result) = &op.result {
                        seeds.insert(result.id());
                    }
                }
                // `hint(x, access_directly=False)` deletes the flags rather
                // than setting them, so its RESULT is unflagged even when its
                // operand was. Upstream gets that from re-binding the name to
                // the stripped annotation; here the result has to be excluded
                // by hand.
                crate::hints::HintKind::NoAccessDirectly => {
                    if let Some(result) = &op.result {
                        killed.insert(result.id());
                    }
                }
                _ => {}
            }
        }
    }
    (seeds, killed)
}

/// The class root each value carries, as far as the lowered graph records it.
///
/// Upstream reads `s_x.classdef.classdesc` straight off the annotation. The
/// LLBC front end records the same identity in three places — a parameter's
/// `class_root` or `Ref` leaf, a call's `result_ty` leaf, and the `<Root>`
/// segment of a representation-cast marker — and carries it across `Hint` ops
/// and `Link`s, which is why this walks to a fixpoint.
fn class_roots(func: &SemanticFunction) -> std::collections::HashMap<u64, String> {
    use crate::model::{CallTarget, OpKind, ValueType};

    fn leaf_root(ty: &ValueType) -> Option<&str> {
        match ty {
            ValueType::Ref(Some(root)) => Some(root.as_str()),
            _ => None,
        }
    }

    let mut roots: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    for block in &func.graph.blocks {
        for op in &block.operations {
            let Some(result) = &op.result else {
                continue;
            };
            let root = match &op.kind {
                OpKind::Input { ty, class_root, .. } => class_root
                    .as_deref()
                    .or_else(|| leaf_root(ty))
                    .map(str::to_string),
                OpKind::Call {
                    target, result_ty, ..
                } => match target {
                    // A struct literal names the type it builds; the front
                    // end spells one as a transparent ctor.  This is read
                    // BEFORE `result_ty` because the two spell the class
                    // differently: the ctor carries the bare leaf, while
                    // `front::mir` fills the same op's `result_ty` with
                    // `owner_path::leaf`.  Every other producer of a root —
                    // `adt_node_class_root`, which is where `OpKind::Input`'s
                    // `class_root` comes from — collapses to the bare leaf, and
                    // so does the declared virtualizable set, so reading
                    // `result_ty` first would answer with a spelling nothing
                    // else uses and no comparison could match.
                    CallTarget::SyntheticTransparentCtor {
                        name, is_struct, ..
                    } if *is_struct => Some(name.clone()),
                    // `__cast_pointer/<Root>` and
                    // `__cast_instance_intrinsic/<Root>` carry the target
                    // class in the path.
                    CallTarget::FunctionPath { segments } if is_representation_cast(segments) => {
                        segments.get(1).cloned()
                    }
                    _ => leaf_root(result_ty).map(str::to_string),
                },
                _ => None,
            };
            if let Some(root) = root {
                roots.insert(result.id(), root);
            }
        }
    }
    // `Hint` and `Link` are identity for the class; propagate across both.
    loop {
        let mut grew = false;
        for block in &func.graph.blocks {
            for op in &block.operations {
                let OpKind::Hint { value, .. } = &op.kind else {
                    continue;
                };
                let (Some(result), Some(root)) =
                    (op.result.as_ref(), roots.get(&value.id()).cloned())
                else {
                    continue;
                };
                if roots.insert(result.id(), root).is_none() {
                    grew = true;
                }
            }
            for link in &block.exits {
                let target = func.graph.block(link.target);
                for (pos, arg) in link.args.iter().enumerate() {
                    let (Some(arg), Some(inputarg)) =
                        (arg.as_variable(), target.inputargs.get(pos))
                    else {
                        continue;
                    };
                    let Some(root) = roots.get(&arg.id()).cloned() else {
                        continue;
                    };
                    if roots.insert(inputarg.id(), root).is_none() {
                        grew = true;
                    }
                }
            }
        }
        if !grew {
            break;
        }
    }
    roots
}

/// The leaf segment of a lowered function's name.
///
/// `front::mir` spells an impl method's name with its `<Impl>::` qualifier,
/// while a `CallTarget::Method` carries the bare method name beside the
/// receiver root, so the two meet at the leaf.
fn leaf_of(name: &str) -> String {
    match name.rsplit_once("::") {
        Some((_, leaf)) => leaf.to_string(),
        None => name.to_string(),
    }
}

/// The lowered function a call target names, if it is one.
///
/// A call spells its target with the declaration's full `name_path`, while a
/// `SemanticFunction` is keyed on the crate-stripped path, so the two are
/// compared after stripping. A method target carries no path at all and is
/// resolved through the receiver-root pair instead.
fn resolve_callee(
    target: &crate::model::CallTarget,
    index: &std::collections::HashMap<String, Option<usize>>,
    method_index: &std::collections::HashMap<(String, String), Option<usize>>,
) -> Option<usize> {
    match target {
        crate::model::CallTarget::FunctionPath { segments } => {
            let joined = segments.join("::");
            *index.get(crate::front::mir::strip_crate_prefix(&joined).as_str())?
        }
        crate::model::CallTarget::Method {
            name,
            receiver_root,
            resolved_path,
        } => {
            // `stamp_classdef_hints_on_graph` stamps the resolved path when
            // the receiver's classdef is known; it is the same key
            // `call.rs target_to_path` hands `function_graphs`, so prefer it.
            if let Some(path) = resolved_path
                && let Some(Some(i)) = index
                    .get(crate::front::mir::strip_crate_prefix(&path.to_string()).as_str())
                    .copied()
            {
                return Some(i);
            }
            let root = receiver_root.as_ref()?;
            *method_index.get(&(root.clone(), leaf_of(name)))?
        }
        _ => None,
    }
}

/// Whether a call is one of `front::mir`'s representation-cast markers.
///
/// `jtransform.rs rewrite_op_cast_pointer` rewrites both to `same_as`
/// (`jtransform.py Transformer.rewrite_op_cast_pointer`), so the result is
/// the operand under a different static type — the one variable an RPython
/// flow graph would have had.
fn is_representation_cast(segments: &[String]) -> bool {
    matches!(
        segments.first().map(String::as_str),
        Some("__cast_pointer") | Some(crate::runtime_names::shims::CAST_INSTANCE)
    )
}

/// Every `(from, to)` pair a value aliases along: a `Link` into the
/// `inputarg` it binds, and a representation cast into its result.
///
/// The link half is what upstream gets for free. `annotator.py` propagates an
/// annotation along a `Link` into the target block's `inputargs`, so a value
/// that crosses a block boundary keeps its `SomeInstance.flags`. Pyre's front
/// end mints a fresh `Variable` for each `inputarg`, exactly as
/// `flowspace/model.py` does, so without this the flag dies at the first
/// branch — and every hint site in the interpreter has one between the hint
/// and the calls that consume the frame.
fn alias_pairs(func: &SemanticFunction) -> Vec<(u64, u64)> {
    let mut pairs = Vec::new();
    for block in &func.graph.blocks {
        for op in &block.operations {
            let crate::model::OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            let crate::model::CallTarget::FunctionPath { segments } = target else {
                continue;
            };
            if !is_representation_cast(segments) {
                continue;
            }
            if let (Some(src), Some(result)) = (args.first(), op.result.as_ref()) {
                pairs.push((src.id(), result.id()));
            }
        }
        for link in &block.exits {
            let target = func.graph.block(link.target);
            for (pos, arg) in link.args.iter().enumerate() {
                if let (Some(arg), Some(inputarg)) = (arg.as_variable(), target.inputargs.get(pos))
                {
                    pairs.push((arg.id(), inputarg.id()));
                }
            }
        }
    }
    pairs
}

/// Extend `ids` along `pairs` to a fixpoint.
fn close_ids_over_aliases(pairs: &[(u64, u64)], ids: &mut std::collections::HashSet<u64>) {
    loop {
        let mut grew = false;
        for (from, to) in pairs {
            if ids.contains(from) {
                grew |= ids.insert(*to);
            }
        }
        if !grew {
            break;
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
            trait_impl_id: None,
            module_path: String::new(),
            hints: Vec::new(),
            trait_root: None,
            trait_qualified: None,
            returns_objectptr: false,
        }
    }

    /// A caller whose startblock hints its first parameter and then passes
    /// arguments to each named callee.
    fn caller_hinting_arg0(
        name: &str,
        hint: Option<crate::hints::HintKind>,
        calls: &[(&str, usize)],
    ) -> SemanticFunction {
        use crate::flowspace::model::Variable;
        use crate::model::{CallTarget, OpKind, SpaceOperation, ValueType};

        let mut f = free_fn(name);
        let param = Variable::named("frame");
        declare_frame_param(&mut f, &param);
        let start = f.graph.startblock;
        let hinted = Variable::named("hinted");
        if let Some(kind) = hint {
            f.graph.block_mut(start).operations.push(SpaceOperation {
                result: Some(hinted.clone()),
                kind: OpKind::Hint {
                    value: param.clone(),
                    kind,
                },
            });
        }
        // With no hint the same slot passes the bare parameter, so the two
        // shapes differ only in whether the value was hinted.
        let passed = if hint.is_some() { hinted } else { param };
        for (callee, arity) in calls {
            let args = (0..*arity)
                .map(|i| {
                    if i == 0 {
                        passed.clone()
                    } else {
                        Variable::named("other")
                    }
                })
                .collect();
            f.graph.block_mut(start).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![(*callee).to_string()],
                    },
                    args,
                    result_ty: ValueType::Void,
                },
            });
        }
        f
    }

    /// An inherent-impl method, spelled the way `front::mir` spells one:
    /// the `<Impl>::` qualifier in the name and the receiver type in
    /// `self_ty_root`.
    fn impl_method_qualified(root: &str, name: &str) -> SemanticFunction {
        let mut f = free_fn(&format!("<Impl>::{name}"));
        f.self_ty_root = Some(root.to_string());
        f
    }

    /// A caller that hints its first parameter, optionally routes the hinted
    /// value through a representation cast, and calls one method on it.
    fn caller_hinting_into_method(
        name: &str,
        through_cast: bool,
        receiver_root: &str,
        method: &str,
    ) -> SemanticFunction {
        use crate::flowspace::model::Variable;
        use crate::model::{CallTarget, OpKind, SpaceOperation, ValueType};

        let mut f = free_fn(name);
        let param = Variable::named("frame");
        declare_frame_param(&mut f, &param);
        let start = f.graph.startblock;
        let hinted = Variable::named("hinted");
        f.graph.block_mut(start).operations.push(SpaceOperation {
            result: Some(hinted.clone()),
            kind: OpKind::Hint {
                value: param.clone(),
                kind: crate::hints::HintKind::AccessDirectly,
            },
        });
        let passed = if through_cast {
            let cast = Variable::named("cast");
            f.graph.block_mut(start).operations.push(SpaceOperation {
                result: Some(cast.clone()),
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__cast_pointer".to_string(), receiver_root.to_string()],
                    },
                    args: vec![hinted],
                    result_ty: ValueType::Ref(Some(receiver_root.to_string())),
                },
            });
            cast
        } else {
            hinted
        };
        f.graph.block_mut(start).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::Call {
                target: CallTarget::Method {
                    name: method.to_string(),
                    receiver_root: Some(receiver_root.to_string()),
                    resolved_path: None,
                },
                args: vec![passed],
                result_ty: ValueType::Void,
            },
        });
        f
    }

    /// Declare `param` a `PyFrame`-rooted input on `f`'s startblock, the way
    /// `front::mir` records a typed pointer parameter. Without it the minter's
    /// class test has nothing to read and the hint seeds nothing.
    fn declare_frame_param(f: &mut SemanticFunction, param: &crate::flowspace::model::Variable) {
        use crate::model::{OpKind, SpaceOperation, ValueType};
        let start = f.graph.startblock;
        f.graph.block_mut(start).inputargs = vec![param.clone()];
        f.graph.block_mut(start).operations.push(SpaceOperation {
            result: Some(param.clone()),
            kind: OpKind::Input {
                name: "frame".to_string(),
                ty: ValueType::Ref(Some("PyFrame".to_string())),
                class_root: None,
            },
        });
    }

    /// The `_virtualizable_` declaration the tests hint against. Every test
    /// helper types its hinted value `Ref(Some("PyFrame"))`, so this is the
    /// set that makes the minter's class test pass.
    fn vable_roots() -> std::collections::HashSet<String> {
        ["PyFrame".to_string()].into()
    }

    fn flags(functions: &[SemanticFunction]) -> Vec<(String, bool)> {
        functions
            .iter()
            .map(|f| (f.name.clone(), f.graph.access_directly))
            .collect()
    }

    /// `specialize.py default_specialize` flags the CALLEE a flagged argument
    /// reaches. The function that spells the hint has unflagged arguments of
    /// its own, so it is not flagged — getting this backwards names exactly
    /// the one graph upstream leaves alone.
    #[test]
    fn the_callee_is_flagged_and_the_hint_site_is_not() {
        let mut fns = vec![
            caller_hinting_arg0(
                "dispatch",
                Some(crate::hints::HintKind::AccessDirectly),
                &[("handle_bytecode", 1)],
            ),
            free_fn("handle_bytecode"),
        ];
        propagate_access_directly(&mut fns, &Default::default(), &vable_roots());
        assert_eq!(
            flags(&fns),
            vec![
                ("dispatch".to_string(), false),
                ("handle_bytecode".to_string(), true),
            ]
        );
    }

    /// The flag travels further than one call: a flagged parameter seeds the
    /// callee's own calls, which is the fixpoint `default_specialize` gets for
    /// free by running per call as the annotator reaches it.
    #[test]
    fn a_flagged_parameter_seeds_the_next_callee() {
        let mut fns = vec![
            caller_hinting_arg0(
                "dispatch",
                Some(crate::hints::HintKind::AccessDirectly),
                &[("handle_bytecode", 1)],
            ),
            caller_hinting_arg0("handle_bytecode", None, &[("execute", 1)]),
            free_fn("execute"),
        ];
        propagate_access_directly(&mut fns, &Default::default(), &vable_roots());
        assert_eq!(
            flags(&fns)
                .into_iter()
                .filter(|(_, on)| *on)
                .map(|(n, _)| n)
                .collect::<Vec<_>>(),
            vec!["handle_bytecode".to_string(), "execute".to_string()]
        );
    }

    /// A method call carries no path, so resolving it by the receiver root is
    /// the only way the flag reaches the callee at all: every consumer of a
    /// hinted `PyFrame` in the interpreter is spelled `frame.method(..)`.
    #[test]
    fn a_method_callee_resolves_through_the_receiver_root() {
        let mut fns = vec![
            caller_hinting_into_method(
                "createframe_obj",
                false,
                "PyFrame",
                "initialize_frame_scopes",
            ),
            impl_method_qualified("PyFrame", "initialize_frame_scopes"),
        ];
        propagate_access_directly(&mut fns, &Default::default(), &vable_roots());
        assert!(!fns[0].graph.access_directly);
        assert!(
            fns[1].graph.access_directly,
            "a method call must reach its callee through the receiver root"
        );
    }

    /// Two lowered functions answering to one `(receiver root, leaf)` pair
    /// cannot be told apart, and flagging the wrong one would arm the gate
    /// against a graph the hint never reached. Treat the pair as unresolved.
    #[test]
    fn an_ambiguous_receiver_pair_flags_neither() {
        let mut fns = vec![
            caller_hinting_into_method("caller", false, "PyFrame", "run"),
            impl_method_qualified("PyFrame", "run"),
            impl_method_qualified("PyFrame", "run"),
        ];
        propagate_access_directly(&mut fns, &Default::default(), &vable_roots());
        assert!(!fns[1].graph.access_directly);
        assert!(!fns[2].graph.access_directly);
    }

    /// The front end mints a fresh `Variable` for every block `inputarg`, so
    /// a hinted value that crosses one branch is a different variable on the
    /// far side. `annotator.py` carries the annotation along the `Link`; this
    /// pass has to carry the seed the same way, or every interpreter hint
    /// dies before reaching the calls that consume the frame.
    #[test]
    fn a_link_carries_the_flag_into_the_successor_block() {
        use crate::flowspace::model::Variable;
        use crate::model::{CallTarget, OpKind, SpaceOperation, ValueType};

        let mut caller = free_fn("dispatch");
        let param = Variable::named("frame");
        declare_frame_param(&mut caller, &param);
        let start = caller.graph.startblock;
        let hinted = Variable::named("hinted");
        caller
            .graph
            .block_mut(start)
            .operations
            .push(SpaceOperation {
                result: Some(hinted.clone()),
                kind: OpKind::Hint {
                    value: param,
                    kind: crate::hints::HintKind::AccessDirectly,
                },
            });
        // The successor's inputarg is a DIFFERENT Variable, as the front end
        // mints it.
        let carried = Variable::named("carried");
        let body = caller.graph.create_block();
        caller.graph.block_mut(body).inputargs = vec![carried.clone()];
        caller
            .graph
            .block_mut(body)
            .operations
            .push(SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::Method {
                        name: "handle_bytecode".to_string(),
                        receiver_root: Some("PyFrame".to_string()),
                        resolved_path: None,
                    },
                    args: vec![carried],
                    result_ty: ValueType::Void,
                },
            });
        caller.graph.set_goto(start, body, vec![hinted]);

        let mut fns = vec![caller, impl_method_qualified("PyFrame", "handle_bytecode")];
        propagate_access_directly(&mut fns, &Default::default(), &vable_roots());
        assert!(
            fns[1].graph.access_directly,
            "a Link must carry the flag into the successor's inputarg"
        );
    }

    /// The front end retypes a hinted pointer through a `__cast_pointer`
    /// marker before it reaches the callee. `jtransform` rewrites that marker
    /// to `same_as`, so the flag has to survive it — otherwise every hint in
    /// the interpreter dies one op after it is spelled.
    #[test]
    fn a_representation_cast_carries_the_flag_to_the_callee() {
        let mut fns = vec![
            caller_hinting_into_method("app_profile_call", true, "PyFrame", "getclass"),
            impl_method_qualified("PyFrame", "getclass"),
        ];
        propagate_access_directly(&mut fns, &Default::default(), &vable_roots());
        assert!(
            fns[1].graph.access_directly,
            "a representation cast must not lose the flag"
        );
    }

    /// `rlib/jit.py`'s `hint` entry mints the flag only for a `SomeInstance`
    /// whose classdesc declares `_virtualizable_`, and deletes it on every
    /// other value. A hint spelled on something the consumer never declared
    /// must therefore seed nothing — otherwise a stray hint hands
    /// `policy::look_inside_graph` a flag it aborts on.
    #[test]
    fn a_hint_on_an_undeclared_root_seeds_nothing() {
        let mut fns = vec![
            caller_hinting_arg0(
                "dispatch",
                Some(crate::hints::HintKind::AccessDirectly),
                &[("handle_bytecode", 1)],
            ),
            free_fn("handle_bytecode"),
        ];
        // The hinted value is rooted `PyFrame`, but nothing declares it.
        propagate_access_directly(&mut fns, &Default::default(), &Default::default());
        assert!(
            !fns[1].graph.access_directly,
            "a hint on an undeclared class must not seed"
        );
    }

    /// `default_specialize` deletes `access_directly` from the argument
    /// annotation when the callee's `_jit_look_inside_` is False, so a
    /// `dont_look_inside` graph is never flagged. Those are exactly the graphs
    /// `policy::look_inside_graph` refuses, so flagging one turns a working
    /// build into the safety-net panic.
    #[test]
    fn a_dont_look_inside_callee_is_never_flagged() {
        let mut fns = vec![
            caller_hinting_arg0(
                "dispatch",
                Some(crate::hints::HintKind::AccessDirectly),
                &[("residual", 1)],
            ),
            free_fn("residual"),
        ];
        let opaque: std::collections::HashSet<String> = ["residual".to_string()].into();
        propagate_access_directly(&mut fns, &opaque, &vable_roots());
        assert!(
            !fns[1].graph.access_directly,
            "a dont_look_inside callee must never carry the flag"
        );
    }

    /// A crate-stripped path is a weaker key than upstream's `FunctionDesc`
    /// identity, so a collision has to be refused. Resolving it to whichever
    /// function was collected last would flag a graph the hint never reached.
    #[test]
    fn an_ambiguous_path_flags_neither() {
        let mut fns = vec![
            caller_hinting_arg0(
                "dispatch",
                Some(crate::hints::HintKind::AccessDirectly),
                &[("handle_bytecode", 1)],
            ),
            free_fn("handle_bytecode"),
            free_fn("handle_bytecode"),
        ];
        propagate_access_directly(&mut fns, &Default::default(), &vable_roots());
        assert!(!fns[1].graph.access_directly);
        assert!(!fns[2].graph.access_directly);
    }

    /// Upstream SPECIALIZES, so an unflagged caller keeps the original graph.
    /// One graph per function cannot split, and the consumer aborts the build,
    /// so a callee reached both ways stays unflagged.
    #[test]
    fn a_callee_reached_both_ways_stays_unflagged() {
        let mut fns = vec![
            caller_hinting_arg0(
                "dispatch",
                Some(crate::hints::HintKind::AccessDirectly),
                &[("handle_bytecode", 1)],
            ),
            caller_hinting_arg0("other_caller", None, &[("handle_bytecode", 1)]),
            free_fn("handle_bytecode"),
        ];
        propagate_access_directly(&mut fns, &Default::default(), &vable_roots());
        assert!(
            !fns[2].graph.access_directly,
            "reached with and without the flag must stay unflagged"
        );
    }

    /// `executioncontext.py app_profile_call` re-binds the frame through
    /// `hint(frame, access_directly=False)` before handing it to arbitrary
    /// Python — "from here on, frame is just a normal w_object". The callees
    /// below that point must not be flagged, or the gate aborts a build
    /// upstream completes.
    #[test]
    fn the_false_spelling_kills_the_seed_for_everything_below_it() {
        use crate::flowspace::model::Variable;
        use crate::model::{BlockId, CallTarget, OpKind, SpaceOperation, ValueType};

        // A callee that receives the flag, which then re-binds it away and
        // passes the result on.
        let mut mid = caller_hinting_arg0("app_profile_call", None, &[]);
        let param = mid.graph.block(BlockId(0)).inputargs[0].clone();
        let normal = Variable::named("normal_w_object");
        mid.graph
            .block_mut(BlockId(0))
            .operations
            .push(SpaceOperation {
                result: Some(normal.clone()),
                kind: OpKind::Hint {
                    value: param,
                    kind: crate::hints::HintKind::NoAccessDirectly,
                },
            });
        mid.graph
            .block_mut(BlockId(0))
            .operations
            .push(SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["call_function".to_string()],
                    },
                    args: vec![normal],
                    result_ty: ValueType::Void,
                },
            });

        let mut fns = vec![
            caller_hinting_arg0(
                "dispatch",
                Some(crate::hints::HintKind::AccessDirectly),
                &[("app_profile_call", 1)],
            ),
            mid,
            free_fn("call_function"),
        ];
        propagate_access_directly(&mut fns, &Default::default(), &vable_roots());
        assert!(
            fns[1].graph.access_directly,
            "app_profile_call itself is reached with the flag"
        );
        assert!(
            !fns[2].graph.access_directly,
            "the flag must not survive the access_directly=False re-bind"
        );
    }

    /// `front::mir` fills a transparent ctor's `result_ty` with
    /// `owner_path::leaf`, while both the declared virtualizable set and
    /// `OpKind::Input`'s `class_root` (via `adt_node_class_root`, which ends
    /// on `name.rsplit("::")`) carry the bare leaf.  A hint on a value a
    /// struct literal produced — which is what BOTH frame constructors hint —
    /// therefore only seeds if the ctor's own name is what the class test
    /// reads.  Every other test here types its value through
    /// `declare_frame_param`, which spells the bare leaf, so this is the only
    /// one that can see the difference.
    #[test]
    fn a_hint_on_a_struct_literal_seeds_through_the_ctor_name() {
        use crate::flowspace::model::Variable;
        use crate::model::{CallTarget, OpKind, SpaceOperation, ValueType};

        let mut ctor = free_fn("createframe_obj");
        let start = ctor.graph.startblock;
        let frame = Variable::named("frame");
        ctor.graph.block_mut(start).operations.push(SpaceOperation {
            result: Some(frame.clone()),
            kind: OpKind::Call {
                target: CallTarget::synthetic_transparent_struct_ctor(
                    vec!["pyre_interpreter".to_string(), "pyframe".to_string()],
                    "PyFrame",
                ),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some("pyre_interpreter::pyframe::PyFrame".to_string())),
            },
        });
        let hinted = Variable::named("hinted");
        ctor.graph.block_mut(start).operations.push(SpaceOperation {
            result: Some(hinted.clone()),
            kind: OpKind::Hint {
                value: frame,
                kind: crate::hints::HintKind::FreshVirtualizable,
            },
        });
        ctor.graph.block_mut(start).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["init_cells".to_string()],
                },
                args: vec![hinted],
                result_ty: ValueType::Void,
            },
        });

        let mut fns = vec![ctor, free_fn("init_cells")];
        propagate_access_directly(&mut fns, &Default::default(), &vable_roots());
        assert!(
            fns[1].graph.access_directly,
            "the ctor's own name is the spelling the declared set holds"
        );
    }

    /// A merge whose arms disagree is unflagged, because
    /// `binaryop.py pairtype(SomeInstance, SomeInstance).union` keeps only the
    /// flags BOTH sides carry — it deletes any key the other annotation lacks.
    /// So a block reached both by a value the `False` spelling killed and by
    /// one that still carries the flag must not be flagged.
    ///
    /// The sibling above cannot see this: there the killed value is consumed
    /// in its own block, where the hint's own result id is what the seed set
    /// is filtered against.  Here the merge gives the killed value a fresh
    /// `inputarg` id, and that id is also an alias target of the flagged
    /// parameter — so it only stays out of the seed set if the killed set is
    /// closed over the aliases the same way the seeds are.
    #[test]
    fn a_merge_of_a_killed_and_a_flagged_value_is_not_flagged() {
        use crate::flowspace::model::Variable;
        use crate::model::{
            BlockId, CallTarget, ExitCase, Link, OpKind, SpaceOperation, ValueType,
        };

        let mut mid = caller_hinting_arg0("app_profile_call", None, &[]);
        let start = mid.graph.startblock;
        let param = mid.graph.block(BlockId(0)).inputargs[0].clone();
        let normal = Variable::named("normal_w_object");
        mid.graph.block_mut(start).operations.push(SpaceOperation {
            result: Some(normal.clone()),
            kind: OpKind::Hint {
                value: param.clone(),
                kind: crate::hints::HintKind::NoAccessDirectly,
            },
        });

        // The merge block's inputarg is a DIFFERENT Variable, as the front end
        // mints it, and both arms feed it.
        let merged = Variable::named("merged");
        let body = mid.graph.create_block();
        mid.graph.block_mut(body).inputargs = vec![merged.clone()];
        mid.graph.block_mut(body).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["call_function".to_string()],
                },
                args: vec![merged],
                result_ty: ValueType::Void,
            },
        });
        let still_flagged =
            Link::from_variables(&mid.graph, vec![param], body, Some(ExitCase::Bool(true)));
        let re_bound =
            Link::from_variables(&mid.graph, vec![normal], body, Some(ExitCase::Bool(false)));
        mid.graph.block_mut(start).exits = vec![still_flagged, re_bound];

        let mut fns = vec![
            caller_hinting_arg0(
                "dispatch",
                Some(crate::hints::HintKind::AccessDirectly),
                &[("app_profile_call", 1)],
            ),
            mid,
            free_fn("call_function"),
        ];
        propagate_access_directly(&mut fns, &Default::default(), &vable_roots());
        assert!(
            fns[1].graph.access_directly,
            "app_profile_call itself is still reached with the flag"
        );
        assert!(
            !fns[2].graph.access_directly,
            "a merge one arm re-bound away is not flagged, as `union` intersects"
        );
    }

    /// Only the argument position the hinted value occupies is flagged, so a
    /// callee that never receives it is left alone.
    #[test]
    fn an_unrelated_callee_is_not_flagged() {
        let mut fns = vec![
            caller_hinting_arg0(
                "dispatch",
                Some(crate::hints::HintKind::AccessDirectly),
                &[("handle_bytecode", 1), ("log", 2)],
            ),
            free_fn("handle_bytecode"),
            free_fn("log"),
        ];
        // `log` takes the hinted value at position 0 too in this fixture, so
        // flip it: give it only the unrelated operand.  Found by target
        // rather than by index — the startblock also carries the parameter
        // declaration and the hint.
        let start = fns[0].graph.startblock;
        let log_call = fns[0]
            .graph
            .block_mut(start)
            .operations
            .iter_mut()
            .find_map(|op| match &mut op.kind {
                crate::model::OpKind::Call {
                    target: crate::model::CallTarget::FunctionPath { segments },
                    args,
                    ..
                } if segments.last().is_some_and(|leaf| leaf == "log") => Some(args),
                _ => None,
            })
            .expect("the log call");
        log_call[0] = crate::flowspace::model::Variable::named("unrelated");
        propagate_access_directly(&mut fns, &Default::default(), &vable_roots());
        assert!(fns[1].graph.access_directly, "handle_bytecode is flagged");
        assert!(!fns[2].graph.access_directly, "log is not");
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

    /// finding 2a: the enum variant-ctor discrimination probes the
    /// qualified enum-base spelling before the bare leaf, so a ctor of an
    /// enum whose leaf collides across modules — where the bare alias was
    /// withdrawn and `lookup_fields`'s suffix shim is ambiguous — still
    /// classifies as a variant ctor instead of misrouting to the
    /// struct-ctor branch.
    #[test]
    fn enum_variant_ctor_discriminates_via_qualified_spelling_after_leaf_collision() {
        // Two distinct enums share the leaf `E`.  `harden_duplicate_leaf_
        // metadata` withdrew the bare `E` alias on the collision; both
        // qualified spellings — the full `name_path`s — survive.
        let mut reg = StructFieldRegistry::default();
        let disc = vec![("__discriminant".to_string(), "i64".to_string())];
        reg.fields.insert("crate::m1::E".to_string(), disc.clone());
        reg.fields.insert("crate::m2::E".to_string(), disc);

        // The bare `E` misses: no exact key, and the suffix shim is
        // ambiguous across the two qualified spellings.
        assert!(
            !reg.is_enum_base("E"),
            "ambiguous bare leaf does not resolve to an enum base"
        );
        // Each qualified spelling resolves exactly.
        assert!(reg.is_enum_base("crate::m1::E"));
        assert!(reg.is_enum_base("crate::m2::E"));

        // The discrimination the variant ctor performs for `m1::E::V`:
        // probe the qualified `owner_path.join("::")` first, then the bare
        // tail.  The qualified probe routes correctly where the bare-only
        // probe (prior behaviour) would misroute to the struct-ctor branch.
        let owner_path = ["crate".to_string(), "m1".to_string(), "E".to_string()];
        let owner_tail = owner_path.last();
        let owner_qual = owner_path.join("::");
        assert!(
            reg.is_enum_base(&owner_qual) || owner_tail.is_some_and(|t| reg.is_enum_base(t)),
            "qualified probe routes the ctor to the variant path"
        );
        assert!(
            !owner_tail.is_some_and(|t| reg.is_enum_base(t)),
            "bare-tail-only probe misses under collision"
        );
    }

    #[test]
    fn enum_base_method_collides_with_variant_payload_field() {
        let mut reg = StructFieldRegistry::default();
        let base = vec![("__discriminant".to_string(), "i64".to_string())];
        let variant = vec![("w_obj".to_string(), "PyObjectRef".to_string())];
        reg.fields
            .insert("buffer::Buffer".to_string(), base.clone());
        reg.fields.insert("Buffer".to_string(), base);
        reg.fields
            .insert("buffer::Buffer::Array".to_string(), variant.clone());
        reg.fields.insert("Buffer::Array".to_string(), variant);

        assert!(reg.owner_or_variant_has_field("buffer::Buffer", "w_obj"));
        assert!(reg.owner_or_variant_has_field("Buffer", "w_obj"));
        assert!(!reg.owner_or_variant_has_field("buffer::Buffer", "readonly"));
    }
}
