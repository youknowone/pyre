//! MIR-driven flowspace driver.
//!
//! This module consumes Charon's ULLBC (a basic-block CFG derived from
//! rustc MIR) and produces the same [`FunctionGraph`] shape the rest of
//! the codewriter pipeline consumes.
//!
//! It is structurally simpler than a recursive-walk driver because the
//! input is already in CFG form: a driver that reconstructs a CFG from a
//! recursive AST walk needs to reconstruct join points, lazily install
//! per-block locals, thread Variables between blocks, and track per-scope
//! bindings. None of that is needed here, because every join point is
//! already an explicit MIR basic block with explicit predecessor edges.
//!
//! ## Reference
//!
//! `rpython/flowspace/flowcontext.py:399-465`
//! ([`FlowContext.build_flow`], [`FlowContext.record_block`],
//! [`FlowContext.mergeblock`]).
//!
//! The RPython reference iterates Python *bytecode positions* and uses
//! `mergeblock` to discover join points lazily. MIR's CFG already has
//! explicit predecessor edges and explicit block boundaries, so the
//! mergeblock dance collapses to a no-op: every join point is already
//! a single MIR basic block with N predecessors.
//!
//! ## Scope — production coverage
//!
//! The driver lowers the entire 4-function corpus end-to-end (see
//! `tests/test_mir_frontend.rs`) and achieves ≥ 99.9% coverage on the
//! real `pyre-interpreter.ullbc` (5434 / 5435 functions) and
//! `pyre-object.ullbc` (1717 / 1718) snapshots, gated by the stress
//! test in `tests/test_mir_stress.rs`. Surfaces handled:
//!
//! ### Statements
//!   - `Assign(Local, Rvalue)` — primary lowering site.
//!   - `Assign(Projection(.., Field|Deref|Index), ...)` — emits a
//!     side-effectful `FieldWrite` / `ArrayWrite` / `__deref_write`.
//!   - `StorageLive` / `StorageDead` / `PlaceMention` — skipped.
//!   - `Assert` — stripped (overflow asserts collapse into success
//!     edge; see the `TermKind::Assert` lowering note below).
//!
//! ### Rvalues
//!   - `Use(operand)` — same-Variable alias.
//!   - `BinaryOp` — `OpKind::BinOp` with a canonical snake_case label
//!     (`add`, `eq`, `and`, …) so the assembler reaches the wired
//!     `int_*` / `ptr_*` keys without inventing PascalCase shapes.
//!   - `UnaryOp` — `OpKind::UnaryOp` with a canonical label
//!     (`neg`, `invert`, `cast_int_to_float`, …) per `binop_label` /
//!     `unary_op_label`.
//!   - `Ref` / `RawPtr` — same-Variable alias (JIT does not model
//!     lifetimes).
//!   - `Cast` — same-Variable alias.
//!   - `Discriminant(place)` — synthetic `FieldRead("__discriminant")`.
//!   - `Aggregate` — synthetic `Call(SyntheticTransparentCtor)`.
//!   - `ShallowInitBox` — synthetic `Call(SyntheticTransparentCtor)`.
//!   - `Repeat` / `Len` / `NullaryOp` — synthetic `Call(__array_repeat
//!     / __len / __nullary_*)`.
//!
//! ### Terminators
//!   - `Return` → `returnblock`.
//!   - `UnwindResume` / `Abort` → `exceptblock`.
//!   - `Goto { target }` — direct edge.
//!   - `Switch { discr, targets }` — `ExitSwitch::Value` + per-arm
//!     `Link` with `ExitCase::Bool` / `ExitCase::Const`.
//!   - `Call` — Direct / Trait → `Call(FunctionPath)`; Dynamic →
//!     synthetic `Call(__dyn_call)` threading the fat-pointer
//!     receiver. (A faithful `IndirectCall` lowering needs vtable
//!     metadata Charon does not yet surface.)
//!   - `Drop` — pass-through `Goto` (JIT does not model destructor
//!     semantics).
//!   - `Assert` — strip and forward to the success target.
//!
//! ### Constants
//!   - `Scalar(Signed|Unsigned|Isize|Usize)` → `ConstInt`.
//!   - `Bool` → `ConstBool`. `Float` → `ConstFloat`.
//!   - `Str` / `Char` / `ByteStr` → synthetic `Call(__str_const)`.
//!   - `FnDef` → synthetic 0-arg `Call(FunctionPath)`.
//!   - `Opaque(reason)` / `VTableRef` / `TraitConst` — synthetic
//!     opaque-string Call. Deferred to a later widening pass when
//!     Charon surfaces the underlying impl/method.
//!
//! Anything not in the above set returns [`LowerError::Unsupported`]
//! with the precise shape that prompted the failure — the driver grows
//! by widening this surface, not by failing silently.

use majit_charon_reader::{
    Llbc,
    ullbc::{
        BasicBlock, CallClass, CallFunc, CallKind, CallPayload, FunDecl, FunId, NameSeg, Operand,
        Place, PlaceKind, ProjectionElem, RegularCall, Rvalue, StmtKind, SwitchTargets, TermKind,
        TyRef, TypeDecl, TypeDeclKind, Unstructured,
    },
};

use crate::flowspace::model::{ConstValue, Variable};
use crate::model::{
    BlockId, CallTarget, ExitCase, ExitSwitch, FieldDescriptor, FrameState, FunctionGraph, Link,
    LinkArg, OpKind, SpaceOperation, ValueType,
};

/// Top-level entry — load `function_name` out of `llbc`, lower it,
/// return the constructed [`FunctionGraph`].
///
/// The lookup is the same `ends_with("::<name>")` rule the reader's
/// `local_fn` uses. Replace with a fully-qualified-path lookup once
/// the call graph plumbing makes it useful.
pub fn lower_function(llbc: &Llbc, function_name: &str) -> Result<FunctionGraph, LowerError> {
    let fd = llbc
        .local_fn(function_name)
        .ok_or_else(|| LowerError::FunctionNotFound(function_name.to_string()))?;
    lower_fun_decl(llbc, fd)
}

/// Merge functions and metadata from a slice of LLBCs into one
/// `SemanticProgram`.  When `pyre-jit-trace` parses pyre-object +
/// pyre-interpreter together, each crate's `.ullbc` is supplied so
/// cross-crate calls in the merged SemanticProgram resolve.  Per-LLBC
/// duplicates (a function defined in both, e.g. via dependency closure)
/// keep the first occurrence.
pub fn build_semantic_program_from_llbcs(
    llbcs: &[Llbc],
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    build_semantic_program_from_llbcs_with_static_addrs(llbcs, crate::HostStaticAddrs::default())
}

pub fn build_semantic_program_from_llbcs_with_static_addrs(
    llbcs: &[Llbc],
    static_addrs: crate::HostStaticAddrs<'_>,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    build_semantic_program_from_llbcs_with_static_addrs_filtered(llbcs, static_addrs, None)
}

pub fn build_semantic_program_from_llbcs_with_static_addrs_and_module_paths(
    llbcs: &[Llbc],
    static_addrs: crate::HostStaticAddrs<'_>,
    module_paths: &[&str],
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    let module_filter = normalize_module_filter(module_paths);
    build_semantic_program_from_llbcs_with_static_addrs_filtered(
        llbcs,
        static_addrs,
        module_filter.as_ref(),
    )
}

fn build_semantic_program_from_llbcs_with_static_addrs_filtered(
    llbcs: &[Llbc],
    static_addrs: crate::HostStaticAddrs<'_>,
    module_filter: Option<&std::collections::HashSet<String>>,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    let mut merged: Option<crate::front::semantic::SemanticProgram> = None;
    // Dedup key combines `self_ty_root` (the impl owner, when known),
    // `module_path`, and `name`.  Without `self_ty_root`, two distinct
    // impl methods would collide on a shared `{module_path}::{name}`:
    // both `impl FrameDebugData { fn new(...) }` and `impl PyFrame {
    // fn new(...) }` land under `module_path = "pyframe::<Impl>"`
    // (the Impl NameSeg renders as `<Impl>`), so without the owner in
    // the key the second would be silently dropped.  Falls back to the
    // bare `{module_path}::{name}` shape (or just `name`) for entries
    // that have no `self_ty_root`.
    let mut seen_function_keys = std::collections::HashSet::new();
    let mut seen_struct_names = std::collections::HashSet::new();
    let mut seen_trait_names = std::collections::HashSet::new();
    let dedup_key = |f: &crate::front::semantic::SemanticFunction| -> String {
        let path = if f.module_path.is_empty() {
            f.name.clone()
        } else {
            format!("{}::{}", f.module_path, f.name)
        };
        match f.self_ty_root.as_deref() {
            Some(owner) => format!("{path}@{owner}"),
            None => path,
        }
    };
    for llbc in llbcs {
        let prog = build_semantic_program_from_llbc_with_static_addrs_filtered(
            llbc,
            static_addrs,
            module_filter,
        )?;
        match &mut merged {
            None => {
                for f in &prog.functions {
                    seen_function_keys.insert(dedup_key(f));
                }
                for n in &prog.known_struct_names {
                    seen_struct_names.insert(n.clone());
                }
                for n in &prog.known_trait_names {
                    seen_trait_names.insert(n.clone());
                }
                merged = Some(prog);
            }
            Some(acc) => {
                for f in prog.functions {
                    if seen_function_keys.insert(dedup_key(&f)) {
                        acc.functions.push(f);
                    }
                }
                for n in prog.known_struct_names {
                    if seen_struct_names.insert(n.clone()) {
                        acc.known_struct_names.insert(n);
                    }
                }
                for n in prog.known_trait_names {
                    if seen_trait_names.insert(n.clone()) {
                        acc.known_trait_names.insert(n);
                    }
                }
                for (key, fields) in prog.struct_fields.fields {
                    acc.struct_fields.fields.entry(key).or_insert(fields);
                }
                for (enum_key, by_discr) in prog.enum_variant_by_discriminant {
                    acc.enum_variant_by_discriminant
                        .entry(enum_key)
                        .or_insert(by_discr);
                }
                for (leaf, module) in prog.struct_origins {
                    acc.struct_origins.entry(leaf).or_insert(module);
                }
                for (key, rows) in prog.struct_field_attrs {
                    acc.struct_field_attrs.entry(key).or_insert(rows);
                }
                for (key, layout) in prog.exact_layouts {
                    acc.exact_layouts.entry(key).or_insert(layout);
                }
                // Merge the name → StructId resolver, collapsing a key to
                // `None` when two crates disagree on the identity (a
                // cross-crate bare-leaf clash).
                for (key, id) in prog.struct_ids {
                    acc.struct_ids
                        .entry(key)
                        .and_modify(|slot| {
                            if *slot != id {
                                *slot = None;
                            }
                        })
                        .or_insert(id);
                }
            }
        }
    }
    // The per-file builder hardened each program individually, but the
    // `or_insert` merges above can re-introduce a bare-leaf alias that
    // is unique within one crate yet collides across crates (e.g. the
    // pyre-interpreter and pyre-jit `FrameBlock`s).  Re-derive the
    // verdict from the merged qualified keys.  (The exact-layout channel
    // is keyed by `StructId` and so is collision-free by construction —
    // only the still-string-keyed metadata channels need hardening.)
    if let Some(acc) = &mut merged {
        harden_duplicate_leaf_metadata(
            &mut acc.struct_fields,
            &mut acc.struct_origins,
            &mut acc.enum_variant_by_discriminant,
        );
    }
    Ok(
        merged.unwrap_or_else(|| crate::front::semantic::SemanticProgram {
            functions: Vec::new(),
            known_struct_names: std::collections::HashSet::new(),
            known_trait_names: std::collections::HashSet::new(),
            struct_fields: crate::front::semantic::StructFieldRegistry::default(),
            immutable_fields: std::collections::HashMap::new(),
            enum_variant_by_discriminant: std::collections::HashMap::new(),
            struct_origins: std::collections::HashMap::new(),
            struct_field_attrs: std::collections::HashMap::new(),
            exact_layouts: std::collections::HashMap::new(),
            struct_ids: std::collections::HashMap::new(),
            unsafe_fn_stubs: Vec::new(),
            foreign_opaque_method_externals: Vec::new(),
        }),
    )
}

fn normalize_module_filter(module_paths: &[&str]) -> Option<std::collections::HashSet<String>> {
    let modules: std::collections::HashSet<String> = module_paths
        .iter()
        .copied()
        .filter(|module_path| !module_path.is_empty())
        .map(str::to_string)
        .collect();
    (!modules.is_empty()).then_some(modules)
}

/// Build a [`SemanticProgram`] by lowering every local function
/// declaration in `llbc`.  This is the production pipeline's
/// program-build entry point (`lib.rs:134`).
///
/// **Whole-program metadata** (`known_struct_names`,
/// `known_trait_names`, `struct_fields`) is populated from
/// `type_decls` / `trait_decls`; struct field-type strings are resolved
/// by [`tyref_to_ast_string`] from Charon's type IR.  `immutable_fields`
/// stays empty until the `#[majit_macros::immutable]` attribute is
/// surfaced by Charon.
///
/// Functions Charon could not extract (opaque body / `null` entry) or
/// global-initializer bodies are skipped silently — they are not JIT
/// call targets.  A function whose MIR shape the driver cannot yet lower
/// produces a [`LowerError`] that is captured per-function: a recognised,
/// tracked gap (an uninitialised-local read that survives even the
/// reverse-postorder re-lower) degrades the program by dropping that one
/// function, while any *unrecognised* lowering failure fails the
/// whole-program build (the coverage gate at the end of this function) so
/// a lowering regression cannot pass silently.
fn is_known_lowering_gap(msg: &str) -> bool {
    // The forward-reference shape: a body reads a MIR local on a path the
    // driver has not yet bound (`read of MIR local N before any Assign`).
    // `lower_fun_decl` first hits this under MIR-index order, then retries
    // the body in reverse-postorder — which orders the defining block
    // before the reading block and resolves every such read in the
    // current snapshot.  This predicate triggers that RPO retry, and
    // (defensively) if even RPO cannot bind the read it classifies the
    // residual failure as a tracked degradation (the function becomes a
    // residual call) rather than a build-failing regression.
    //
    // Loop-carried definitions (a local defined on a back-edge and read
    // at the loop header) are NOT a residual case: every block's live-in
    // locals already receive a block inputarg in `Lowering::new`, and
    // `edge_args` threads each predecessor's binding — including the
    // back-edge — into it, so the header read resolves to the inputarg
    // regardless of processing order.  The only requirement is that
    // `compute_mir_liveness` mark every local the lowering reads; see
    // `mark_projection_index_offset_use` for the `place[idx]` case.
    if msg.contains("uninitialised local") {
        return true;
    }
    // A scoped (`execute_*` family) Result-of-PyError wrapper whose body
    // is a pure tail-forward of an *unscoped* Result-of-PyError callee
    // (`let step = executor.method()?; Ok(step)` where `method` is not in
    // `RESULT_EXC_LOWERING_SCOPE`).  The forward collapses to a direct
    // returnblock link with no `Ok`/`Err` shell, so the callee rule finds
    // nothing to rewrite and the caller rule never saw a scoped call —
    // `result_exc::lower_result_exc_returns` reports "no rewritable
    // returns".  The exception-link lowering cannot model this shape, so
    // the wrapper degrades to a residual call (the trivial forward runs
    // unchanged at the interpreter level); the inner method still JITs
    // through its own scoped callees.
    msg.contains("no rewritable returns")
}

/// A discovered reference-payload generic-enum instantiation: the enum's
/// Charon def id and qualified name, the concrete type arguments in
/// generic-parameter order (`["Tuple", "PyError"]`), and the `<…>` suffix
/// the per-instantiation classdef carries (`<Tuple,PyError>`).
#[derive(Clone, PartialEq, Eq, Hash)]
struct RefEnumInst {
    def_id: u64,
    name_path: String,
    args: Vec<String>,
    suffix: String,
}

/// The [`RefEnumInst`] for the ADT descriptor `adt` (`{"id": …,
/// "generics": …}`) when it names a split-eligible reference-payload
/// generic enum, else `None`.  Shared by both instantiation scans below so
/// a constructor head and a type-node receiver seed the identical set and
/// spelling.
fn ref_enum_instantiation_of_adt(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
) -> Option<RefEnumInst> {
    let suffix = adt_head_instantiation_suffix(adt, llbc)?;
    let def_id = adt
        .get("id")
        .and_then(serde_json::Value::as_object)
        .and_then(|id| id.get("Adt"))
        .and_then(serde_json::Value::as_u64)?;
    let name_path = llbc.type_by_id(def_id)?.item_meta.name_path();
    let args = render_adt_type_args(adt, llbc, 0);
    Some(RefEnumInst {
        def_id,
        name_path,
        args,
        suffix,
    })
}

/// Collect every reference-payload generic-enum instantiation in the
/// program as `(name_path, "<…>")` pairs (`Result`, `<Tuple>`).  Two
/// sources, both filtered through [`adt_head_instantiation_suffix`] so the
/// discovered set agrees with what the constructor / field-read / receiver
/// sites project (a non-enum `Vec<T>` / `Box<T>` yields `None`; only the
/// intended `Option`/`Result`-family enums seed):
///   - CONSTRUCTOR heads — a `Result::Ok` / `Option::Some` payload
///     `setattr("__pos_0")` carries its concrete generics inline on the
///     `Rvalue::Aggregate(AggregateKind::Adt, …)` head.
///   - TYPE-NODE receivers — a function that only READS an instantiation
///     (a `__discriminant` match on a `Result<Tuple>` parameter, never a
///     constructor) carries it solely in a local's declared type.  Seeding
///     these pre-mints the variant subclasses before
///     `assign_inheritance_ids`; otherwise the discriminant narrowing mints
///     them lazily afterwards, unnumbered (per-graph Skip→legacy walker).
fn collect_ref_enum_instantiations(llbc: &Llbc) -> Vec<RefEnumInst> {
    let mut found: std::collections::HashSet<RefEnumInst> = std::collections::HashSet::new();
    for fd in llbc.iter_local_fns() {
        let Some(u) = fd.unstructured() else {
            continue;
        };
        for bb in &u.body {
            for st in &bb.statements {
                let Ok(StmtKind::Assign(_, Rvalue::Aggregate(kind, _))) = st.stmt_kind() else {
                    continue;
                };
                let Some(head) = kind
                    .get("Adt")
                    .and_then(serde_json::Value::as_array)
                    .and_then(|adt| adt.first())
                    .and_then(serde_json::Value::as_object)
                else {
                    continue;
                };
                if let Some(pair) = ref_enum_instantiation_of_adt(head, llbc) {
                    found.insert(pair);
                }
            }
        }
        for local in &u.locals.locals {
            let Some(adt) = tyref_node(&local.ty, llbc)
                .and_then(|n| strip_ty_wrappers(n, llbc))
                .and_then(serde_json::Value::as_object)
                .and_then(|n| n.get("Adt"))
                .and_then(serde_json::Value::as_object)
            else {
                continue;
            };
            if let Some(pair) = ref_enum_instantiation_of_adt(adt, llbc) {
                found.insert(pair);
            }
        }
    }
    found.into_iter().collect()
}

/// Substitute the instantiation's concrete type args into a variant
/// field's type: a bound type variable resolves to `args[idx]`; any other
/// shape (a concrete field, or a `??…` placeholder for an unresolved node)
/// renders as-is via [`tyref_to_ast_string`].  Only a bare type variable
/// is substituted — a `Box<T>` / `Vec<T>` field keeps its `??TypeVar`
/// inner rendering, which the caller treats as unresolved and skips.
fn substitute_field_type(ty: &TyRef, args: &[String], llbc: &Llbc) -> String {
    if let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) {
        if let Some(idx) = typevar_bound_index(node) {
            return args
                .get(idx as usize)
                .cloned()
                .unwrap_or_else(|| "??typevar_oob".to_string());
        }
    }
    tyref_to_ast_string(ty, llbc)
}

/// Whether a rendered payload type is an unboxed scalar rather than a
/// one-GC-word heap reference.  The textual descr path resolves a scalar's
/// bank/width from the row type string via `type_flag_from_str`
/// (`assembler.rs`) / `get_type_flag` (`call.rs`), so a scalar row is
/// layout-safe *as a variant's sole field* (byte offset 0).  In a
/// multi-field variant, mixed scalar widths could reorder under
/// `repr(Rust)` away from the textual offset accumulator, so a primitive
/// field there stays fail-closed (offset ambiguity).
fn is_primitive_payload_type(s: &str) -> bool {
    matches!(
        s.trim(),
        "i8" | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "f32"
            | "f64"
            | "bool"
            | "char"
            | "()"
            | ""
    )
}

/// Whether a rendered payload type is an inline multi-word aggregate — a
/// tuple `(A, B)`, array `[T; N]`, or slice `[T]` — rather than a single
/// GC-word reference or a scalar.  The suffixed-owner textual field descr
/// has no `StructId` for such a payload, so `type_flag_from_str` would fall
/// back to a one-GC-word `Ref` (correct for a reference, WRONG for an
/// inline aggregate that spans several words), so it must fail closed.  A
/// tuple/array whose inner type is itself unresolved already renders with a
/// `??` marker and is caught upstream; this only screens the resolved
/// inline-aggregate spellings.
fn is_inline_aggregate_type(s: &str) -> bool {
    let s = s.trim();
    (s.starts_with('(') && s != "()") || s.starts_with('[')
}

/// Register per-instantiation SUFFIXED variant payload rows for the
/// discovered reference-payload enum instantiations (#312 C4).
///
/// The unsuffixed template rows (`Result::Ok`, [`derive_program_metadata`])
/// carry the bare type variable, which `tyref_to_ast_string` renders as
/// `??TypeVar` → `project_pyre_field_type` → `Impossible`.  So a
/// narrowing-only receiver (`Result<Tuple>` matched + read, never
/// constructed in-graph) finds no concrete payload: its suffixed variant
/// classdef `__pos_0` stays `Impossible` (annotator bottom) and the graph
/// routes Skip → legacy walker.  A constructor instead flows the concrete
/// payload onto the classdef via `setattr`, so only the read-only case
/// gaps.
///
/// Substituting the instantiation's concrete type args into each variant
/// field's type-var position recovers the concrete row, keyed under the
/// same suffixed spellings (`{enum}{suffix}::{variant}`) that
/// `getuniqueclassdef_for_enum_variant` projects through.  A variant
/// registers when every payload field substitutes to either a heap
/// (one-GC-word) type or — for a single-field variant — a primitive scalar
/// (`Result<i64>`, `Option<bool>`): the textual descr path resolves the
/// scalar's bank/width from the row type string, and a sole field sits at
/// byte offset 0 so its layout is unambiguous.  An unresolved `??`
/// placeholder, an inline multi-word aggregate (tuple/array/slice), or a
/// primitive field in a multi-field variant (mixed scalar widths could
/// reorder under `repr(Rust)`) leaves the variant unregistered (fail-closed
/// Skip).  The qualified spelling always publishes; the bare-leaf / crate-
/// stripped spellings publish only while the leaf survived
/// `harden_duplicate_leaf_metadata`, so a duplicate enum leaf fails closed
/// instead of projecting whichever colliding decl was inserted first.  The
/// rows only ever fill an `Impossible`/force-shell attr, so a constructed
/// instantiation already carrying its concrete payload is left untouched
/// (monotonic).
fn register_ref_enum_instantiation_rows(
    llbc: &Llbc,
    insts: &[RefEnumInst],
    enum_variant_by_discriminant: &std::collections::HashMap<
        String,
        std::collections::HashMap<i64, String>,
    >,
    struct_fields: &mut crate::front::semantic::StructFieldRegistry,
) {
    for inst in insts {
        let Some(td) = llbc.type_by_id(inst.def_id) else {
            continue;
        };
        let TypeDeclKind::Enum(variants) = &td.kind else {
            continue;
        };
        let name = td.item_meta.name_path();
        let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
        let canon_base = strip_crate_prefix(&name);
        // The bare leaf survives in `enum_variant_by_discriminant` only when
        // `harden_duplicate_leaf_metadata` did not withdraw it on cross-decl
        // collision; gate the ambiguous bare spellings on that, same as the
        // discriminant-map mirror.
        let leaf_survived = enum_variant_by_discriminant.contains_key(&leaf);
        let mut bases: Vec<&str> = vec![name.as_str()];
        if leaf_survived {
            bases.push(leaf.as_str());
            bases.push(canon_base.as_str());
        }
        for v in variants {
            if v.fields.is_empty() {
                continue;
            }
            // A sole payload field sits at byte offset 0, so a resolved
            // primitive scalar is layout-safe there; a primitive in a
            // multi-field variant could reorder under `repr(Rust)`.
            let single_field = v.fields.len() == 1;
            let mut rows: Vec<(String, String)> = Vec::with_capacity(v.fields.len());
            let mut registrable = true;
            for (i, f) in v.fields.iter().enumerate() {
                let fname = f.name.clone().unwrap_or_else(|| format!("__pos_{i}"));
                let concrete = substitute_field_type(&f.ty, &inst.args, llbc);
                let trimmed = concrete.trim();
                // Register a field only when it is layout-safe for the
                // suffixed owner's textual descr: a single-GC-word heap
                // reference, or (sole field, offset 0) a real unboxed scalar.
                // Fail closed on an unresolved `??` placeholder, an inline
                // multi-word aggregate (tuple/array/slice), the unit / empty
                // render, or a scalar outside a single-field variant.
                let layout_safe = !concrete.contains("??")
                    && !is_inline_aggregate_type(trimmed)
                    && trimmed != "()"
                    && !trimmed.is_empty()
                    && (!is_primitive_payload_type(&concrete) || single_field);
                if !layout_safe {
                    registrable = false;
                    break;
                }
                rows.push((fname, concrete));
            }
            if !registrable {
                continue;
            }
            // Mirror the unsuffixed dual-publish so the suffixed variant path
            // the narrowing builds — `canonical_struct_name("{base}{suffix}")
            // ::{variant}` — hits regardless of which spelling the receiver
            // carries.  The qualified `name` is unambiguous and always
            // publishes; the bare-leaf / crate-stripped spellings collide
            // across modules sharing an enum leaf, so they publish only while
            // the bare leaf survived `harden_duplicate_leaf_metadata` (mirrors
            // the discriminant-map guard) — a withdrawn leaf fails closed
            // rather than projecting whichever colliding decl was inserted
            // first.
            for base in &bases {
                struct_fields
                    .fields
                    .entry(format!("{base}{}::{}", inst.suffix, v.name))
                    .or_insert_with(|| rows.clone());
            }
        }
    }
}

pub fn build_semantic_program_from_llbc(
    llbc: &Llbc,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    build_semantic_program_from_llbc_with_static_addrs(llbc, crate::HostStaticAddrs::default())
}

pub fn build_semantic_program_from_llbc_with_static_addrs(
    llbc: &Llbc,
    static_addrs: crate::HostStaticAddrs<'_>,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    build_semantic_program_from_llbc_with_static_addrs_filtered(llbc, static_addrs, None)
}

fn build_semantic_program_from_llbc_with_static_addrs_filtered(
    llbc: &Llbc,
    static_addrs: crate::HostStaticAddrs<'_>,
    module_filter: Option<&std::collections::HashSet<String>>,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    // ── Pass 1: walk type_decls + trait_decls ─────────────────────
    let (
        known_struct_names,
        known_trait_names,
        mut struct_fields,
        mut enum_variant_by_discriminant,
        mut struct_origins,
        struct_field_attrs,
        exact_layouts,
        struct_ids,
    ) = derive_program_metadata(llbc);
    harden_duplicate_leaf_metadata(
        &mut struct_fields,
        &mut struct_origins,
        &mut enum_variant_by_discriminant,
    );

    // Per-instantiation enum-variant pre-registration source (#100): a
    // reference-payload generic enum constructor (`Result<Tuple>::Ok`)
    // projects a per-instantiation variant class so its payload does not
    // union across instantiations.  Publish a discriminant-table entry
    // per such instantiation, keyed by the LEAF-suffixed `{leaf}{suffix}`
    // — the exact spelling the constructor carries as its owner tail
    // (`resolve_aggregate_adt`) — so the prologue pre-mint and the
    // constructor pass the identical string to `canonical_struct_name`
    // and resolve ONE variant classdef regardless of what
    // `STRUCT_ORIGIN_REGISTRY` maps the base to.  The cloned tag→variant
    // map is instantiation-invariant.  `pre_register_enum_variant_classes`
    // accepts the `<`-bearing key (its filter admits `::`-qualified OR
    // per-instantiation roots) and numbers the variant subclasses before
    // `assign_inheritance_ids`, so the split classes drain rather than
    // landing unnumbered (per-graph Skip).
    let ref_enum_insts = collect_ref_enum_instantiations(llbc);
    for inst in &ref_enum_insts {
        let leaf = inst
            .name_path
            .rsplit("::")
            .next()
            .unwrap_or(&inst.name_path);
        // Mirror the discriminant map to the `{leaf}{suffix}` spelling only
        // while the bare leaf survived `harden_duplicate_leaf_metadata`.  A
        // suffixed key carries no `::`, so it escapes the `::`-keyed
        // dup-leaf pass; re-arming it for a leaf the hardening withdrew on
        // cross-decl ambiguity would resurrect the silent-winner alias.
        // Fail-closed — a withheld alias misses to the qualified key or
        // routes the suffixed receiver to a per-graph Skip.
        if !enum_variant_by_discriminant.contains_key(leaf) {
            continue;
        }
        if let Some(bare) = enum_variant_by_discriminant.get(&inst.name_path).cloned() {
            enum_variant_by_discriminant
                .entry(format!("{leaf}{}", inst.suffix))
                .or_insert(bare);
        }
    }
    // Per-instantiation SUFFIXED variant payload rows (#312 C4): give a
    // narrowing-only reference-payload receiver its concrete payload type
    // so the suffixed variant classdef does not stay `Impossible` → Skip.
    register_ref_enum_instantiation_rows(
        llbc,
        &ref_enum_insts,
        &enum_variant_by_discriminant,
        &mut struct_fields,
    );

    // ── Pass 2: lower every function body and build SemanticFunctions ─
    let mut functions = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    for fd in llbc.iter_local_fns() {
        if fd.unstructured().is_none() {
            continue;
        }
        // Charon emits static / const initialiser bodies (e.g. the
        // body that builds `static NONE_SINGLETON`) as ordinary
        // `FunDecl` entries with `is_global_initializer` set to the
        // backing `GlobalDecl` id.  These bodies are not call targets
        // at the JIT level — skip them so they never surface as
        // call-registry entries the rest of the pipeline does not
        // model.  (Their unwind paths lower via `set_raise`,
        // `model.rs:4149`; the flowspace adapter converts only the
        // reachable block closure, so an unreachable unwind block's
        // orphan etype/evalue slots no longer reject the graph — this
        // skip is about call-target modelling, not adapter safety.)
        if fd.is_global_initializer.is_some() {
            continue;
        }
        // Key each SemanticFunction by bare leaf name plus a separate
        // `module_path` so `register_function_graph_alias` (lib.rs:444)
        // walks `{bare, crate::*, pyre_*::*}` correctly and the portal
        // lookup at lib.rs:1043 (`["eval_loop_jit"]`) resolves.
        let stripped = strip_crate_prefix(&fd.item_meta.name_path());
        let (module_path, name) = match stripped.rsplit_once("::") {
            Some((module, leaf)) => (module.to_string(), leaf.to_string()),
            None => (String::new(), stripped),
        };
        if !should_lower_module(module_filter, &module_path) {
            continue;
        }
        // A single function whose body the driver does not yet handle
        // should not abort the whole-program build.  Capture
        // per-function errors into a side bucket and continue; they are
        // surfaced via `PYRE_MIR_FRONTEND_DEBUG=1` for triage, but
        // production keeps going with a degraded SemanticProgram —
        // failing-loud on the single broken function rather than
        // erroring out at program-build time.
        let graph = match lower_fun_decl_with_static_addrs(llbc, fd, static_addrs) {
            Ok(g) => g,
            Err(e) => {
                skipped.push((name.clone(), e.to_string()));
                continue;
            }
        };
        // return_type is intentionally `None` until the Charon
        // dedup-table resolution can map a `TyRef::Deduplicated{id}` to
        // its primitive name. The codewriter's call-signature validator
        // at `codewriter/call.rs:4234` skips the check when declared
        // type is None, which is the right behaviour while the
        // resolution gap is open — TyRef labels (`ty#170`) would
        // otherwise be classified as `Type::Ref` and trip a spurious
        // mismatch panic against a real `Type::Int` callee result.
        // Surface the impl-method owner on the SemanticFunction so
        // `lib.rs:868` / `lib.rs:1086` and the
        // `extract_inherent_impl_methods` / `extract_trait_impls`
        // consumers see the same `self_ty_root` the MIR driver records.
        // Without this, every impl method built by the MIR driver looks
        // like a free function to the canonical registration loop and
        // the impl-key return-type / hint registrations get dropped.
        let self_ty_root = impl_method_owner_for_fundecl(llbc, fd).map(|(owner, _)| owner);
        // Surface trait identity for trait-impl methods so the
        // canonical registration loop can call `register_trait_method`
        // instead of routing through `extract_trait_impls`.  Inherent
        // impls leave `trait_root = None`; trait-impl methods carry the
        // trait's leaf name.
        //
        // Two sources feed `trait_root`:
        //   1. trait-impl bodies — penultimate NameSeg is `Impl{Trait:id}`
        //      indirecting through `trait_impls`.  `trait_impl_trait_path_for_fundecl`
        //      reads the id; `trait_qualified` keeps the full path so
        //      the unique-impl map can key on trait identity.
        //   2. trait-default bodies — Charon emits these as bare
        //      functions inside the trait's namespace; the penultimate
        //      NameSeg is `Ident{TraitLeaf}` with no `Impl` segment.
        //      Detect by matching the parent ident against
        //      `known_trait_names` (which derive_program_metadata seeds
        //      with both qualified path and bare leaf).
        let trait_qualified = trait_impl_trait_path_for_fundecl(llbc, fd);
        let trait_root = trait_qualified
            .as_ref()
            .and_then(|p| p.rsplit("::").next())
            .map(str::to_string)
            .or_else(|| trait_default_owner_for_fundecl(fd, &known_trait_names));
        functions.push(crate::front::semantic::SemanticFunction {
            name,
            graph,
            return_type: None,
            self_ty_root,
            module_path,
            hints: Vec::new(),
            access_directly: false,
            trait_root,
            trait_qualified,
        });
    }
    // Coverage gate. Every `skipped` entry is a function whose MIR shape
    // the driver could not lower — already after the reverse-postorder
    // retry in `lower_fun_decl`. The single known, tracked gap is an
    // "uninitialised local read" that even RPO could not bind (a genuine
    // loop-carried def — none in the current snapshot); such a function
    // would degrade the program by being dropped to a residual call,
    // never a correctness loss. Any *other* lowering failure is a coverage
    // regression that must not pass silently, so fail the whole-program
    // build with the offending list.
    if !skipped.is_empty() {
        let (tracked, regressions): (Vec<_>, Vec<_>) = skipped
            .iter()
            .partition(|(_, msg)| is_known_lowering_gap(msg));
        if std::env::var("PYRE_MIR_FRONTEND_DEBUG").is_ok() && !tracked.is_empty() {
            eprintln!(
                "[mir-frontend] {} function(s) skipped via the tracked \
                 uninitialised-local gap:",
                tracked.len()
            );
            for (name, msg) in tracked.iter().take(20) {
                eprintln!("  {name}: {msg}");
            }
        }
        if !regressions.is_empty() {
            let mut detail = String::new();
            for (name, msg) in &regressions {
                detail.push_str(&format!("\n  - {name}: {msg}"));
            }
            return Err(LowerError::Unsupported(format!(
                "MIR lowering coverage regression: {} function(s) failed to lower with \
                 an unrecognised error (not the tracked uninitialised-local gap). Fix the \
                 lowering, or extend `is_known_lowering_gap` if the new shape is \
                 intentionally unsupported:{detail}",
                regressions.len()
            )));
        }
    }
    Ok(crate::front::semantic::SemanticProgram {
        functions,
        known_struct_names,
        known_trait_names,
        struct_fields,
        // Immutable-field tracking depends on `#[majit_macros::immutable]`
        // attribute serialization that Charon does not currently surface
        // (the `attributes` array carries DocComment / Outer but not our
        // proc-macro hints).
        immutable_fields: std::collections::HashMap::new(),
        enum_variant_by_discriminant,
        struct_origins,
        struct_field_attrs,
        exact_layouts,
        struct_ids,
        // Populated post-build in `build_semantic_program_via_active_frontend`
        // (it iterates the full LLBC set), mirroring `merge_hints_from_llbcs`.
        unsafe_fn_stubs: Vec::new(),
        foreign_opaque_method_externals: Vec::new(),
    })
}

fn should_lower_module(
    module_filter: Option<&std::collections::HashSet<String>>,
    module_path: &str,
) -> bool {
    let Some(module_filter) = module_filter else {
        return true;
    };
    if module_filter.iter().any(|root| {
        module_path == root
            || module_path
                .strip_prefix(root)
                .is_some_and(|rest| rest.starts_with("::"))
    }) {
        return true;
    }
    // A fixture's `module_paths` names the interpreter/JIT roots it wants
    // to analyze.  Keep shared helper crates/modules available because
    // opcode graphs routinely call into `pyre_object`, `error`,
    // `baseobjspace`, etc.; only skip unrequested built-in extension modules
    // under `pyre_interpreter::module::*`, whose large dispatch/register
    // bodies are not part of the JIT portal closure.
    !(module_path == "module" || module_path.starts_with("module::"))
}

/// Derive whole-program type-metadata fields of `SemanticProgram` from
/// Charon's `type_decls` + `trait_decls` tables.
///
/// Returns `(known_struct_names, known_trait_names, struct_fields,
/// enum_variant_by_discriminant, struct_origins, struct_field_attrs,
/// exact_layouts)`.
/// Names are taken from `item_meta.name_path()`; struct field rows
/// resolve their type string via [`tyref_to_ast_string`] (Charon-resolved
/// types: references stripped, raw pointers kept, `Vec<T>` / `[T;N]`
/// generics, named structs by leaf).  `struct_origins` maps a bare
/// struct leaf to its defining module path with the crate prefix
/// stripped (so the value matches the runtime def-path convention).
/// `struct_field_attrs` maps the crate-stripped qualified struct name to
/// its declaration-ordered `(field, ValueType)` register classes.
/// Record a `name → StructId` mapping in the resolver table, collapsing
/// to `None` (ambiguous) if the same name already mapped to a different
/// identity — i.e. two distinct type definitions share that spelling
/// (only possible for a bare leaf across modules).
fn record_struct_id(
    table: &mut std::collections::HashMap<String, Option<majit_ir::descr::StructId>>,
    name: String,
    id: majit_ir::descr::StructId,
) {
    table
        .entry(name)
        .and_modify(|slot| {
            if *slot != Some(id) {
                *slot = None;
            }
        })
        .or_insert(Some(id));
}

fn derive_program_metadata(
    llbc: &Llbc,
) -> (
    std::collections::HashSet<String>,
    std::collections::HashSet<String>,
    crate::front::semantic::StructFieldRegistry,
    std::collections::HashMap<String, std::collections::HashMap<i64, String>>,
    std::collections::HashMap<String, String>,
    std::collections::HashMap<String, Vec<(String, ValueType)>>,
    std::collections::HashMap<majit_ir::descr::StructId, crate::front::semantic::ExactLayout>,
    std::collections::HashMap<String, Option<majit_ir::descr::StructId>>,
) {
    // Charon resolves layout per target; the LLBC carries a single
    // entry for the extraction target (build-script `TARGET`).  An
    // absent/non-matching target falls back to the sole entry inside
    // `layout_for_target`, so an empty value still resolves it.
    let target = std::env::var("TARGET").unwrap_or_default();
    let mut known_struct_names = std::collections::HashSet::new();
    let mut known_trait_names = std::collections::HashSet::new();
    let mut struct_fields = crate::front::semantic::StructFieldRegistry::default();
    let mut enum_variant_by_discriminant: std::collections::HashMap<
        String,
        std::collections::HashMap<i64, String>,
    > = std::collections::HashMap::new();
    let mut struct_origins: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut exact_layouts: std::collections::HashMap<
        majit_ir::descr::StructId,
        crate::front::semantic::ExactLayout,
    > = std::collections::HashMap::new();
    // name (any spelling) → canonical StructId; `None` marks a bare leaf
    // two distinct modules share.  Inserts go through `record_struct_id`
    // below so a cross-module bare-leaf clash collapses to `None`.
    let mut struct_ids: std::collections::HashMap<String, Option<majit_ir::descr::StructId>> =
        std::collections::HashMap::new();
    let mut struct_field_attrs: std::collections::HashMap<String, Vec<(String, ValueType)>> =
        std::collections::HashMap::new();

    for td in llbc.iter_type_decls() {
        let name = td.item_meta.name_path();
        match &td.kind {
            TypeDeclKind::Struct(fields) => {
                // Register the qualified path *and* the bare leaf name
                // so downstream lookups (`canonical_call_target`'s
                // bare-leaf fallback) resolve either spelling.
                let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
                let rows: Vec<(String, String)> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let fname = f.name.clone().unwrap_or_else(|| format!("__pos_{i}"));
                        (fname, tyref_to_ast_string(&f.ty, llbc))
                    })
                    .collect();
                struct_fields.fields.insert(name.clone(), rows.clone());
                struct_fields.fields.insert(leaf.clone(), rows);
                // Object identity for this struct type, minted from the
                // crate-stripped qualified path — the spelling the descr
                // layer keys `LLType::Struct` on and the runtime macro
                // publishes.  Every name spelling (full-crate, stripped,
                // bare leaf) resolves to it through the resolver table.
                let sid = majit_ir::descr::StructId::from_canonical(&strip_crate_prefix(&name));
                record_struct_id(&mut struct_ids, name.clone(), sid);
                record_struct_id(&mut struct_ids, strip_crate_prefix(&name), sid);
                record_struct_id(&mut struct_ids, leaf.clone(), sid);
                // Exact field byte offsets from Charon's resolved layout
                // (the true Rust layout, not the heuristic).  Field index
                // `i` matches the declaration order Charon offsets by.
                // Absent layout (opaque type) leaves the type unrecorded;
                // the heuristic provider covers it.  Keyed by the identity
                // token — one entry per type definition.
                if let Some(layout) = td.layout_for_target(&target) {
                    let mut field_offsets = std::collections::HashMap::new();
                    for (i, f) in fields.iter().enumerate() {
                        let fname = f.name.clone().unwrap_or_else(|| format!("__pos_{i}"));
                        if let Some(off) = layout.struct_field_offset(i) {
                            field_offsets.insert(fname, off);
                        }
                    }
                    let exact = crate::front::semantic::ExactLayout {
                        size: layout.size,
                        align: layout.align,
                        field_offsets,
                    };
                    exact_layouts.insert(sid, exact);
                }
                // `bare leaf → crate-relative module`: drop the crate
                // prefix (first segment) and the leaf (last segment) so
                // the value matches the runtime def-path
                // (`intobject::W_IntObject` ← `pyre_object::intobject::
                // W_IntObject`).  First-write-wins on duplicate leaves
                // defined in distinct modules; the loser's qualified
                // `name` key still resolves through the dual-publish.
                let segs: Vec<&str> = name.split("::").collect();
                let module = if segs.len() >= 2 {
                    segs[1..segs.len() - 1].join("::")
                } else {
                    String::new()
                };
                struct_origins.entry(leaf.clone()).or_insert(module);
                // Register-class rows for `FORCE_ATTRIBUTES_INTO_CLASSES`,
                // keyed by the crate-stripped defining path. This is the
                // closest Rust-side stand-in for RPython's class-object key:
                // same-leaf structs in distinct modules stay distinct, and
                // the spelling matches the def-path convention used by
                // `STRUCT_ORIGIN_REGISTRY`.
                let attr_rows: Vec<(String, ValueType)> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let fname = f.name.clone().unwrap_or_else(|| format!("__pos_{i}"));
                        (fname, tyref_to_attr_value_type(&f.ty, llbc))
                    })
                    .collect();
                struct_field_attrs.insert(strip_crate_prefix(&name), attr_rows);
                known_struct_names.insert(name);
                known_struct_names.insert(leaf);
            }
            TypeDeclKind::Enum(variants) => {
                // Enums register under their type name *and* under each
                // variant path (`Strategy::Empty`, `Strategy::IntKeyed`,
                // …) so a synthetic Aggregate(SyntheticTransparentCtor)
                // can be matched downstream.
                let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
                // The crate-stripped `module::Enum` spelling — the
                // canonical key `canonical_struct_name(enum_leaf)` resolves
                // to (`STRUCT_ORIGIN_REGISTRY` prepends the same
                // crate-stripped module path).  The annotator interns the
                // base and its variants under this spelling
                // (`intern_enum_variant_host`), so registering rows /
                // attrs here lets `project_struct_rows` and the FORCE
                // table find them on the one class object the narrowing
                // and the constructor share.
                let canon_base = strip_crate_prefix(&name);
                // Register the bare-leaf → crate-relative module origin, the
                // same way the struct arm does above.  `intern_enum_variant_host`
                // relies on `canonical_struct_name(leaf)` resolving to the
                // `module::Enum` spelling so the constructor side (owner tail =
                // bare leaf) and the discriminant-narrowing side (the base
                // classdef's canonical name) normalise to one cache key; without
                // the origin the bare leaf passes through unchanged and the two
                // sides mint sibling classdefs for one Rust enum.  First-write-
                // wins on duplicate leaves; `harden_duplicate_leaf_metadata`
                // empties the entry when distinct modules collide.
                let module = canon_base
                    .rsplit_once("::")
                    .map(|(m, _)| m.to_string())
                    .unwrap_or_default();
                struct_origins.entry(leaf.clone()).or_insert(module);
                known_struct_names.insert(name.clone());
                known_struct_names.insert(leaf.clone());
                // Register the enum base class in `struct_fields` with
                // only the synthetic `__discriminant` tag — the sum-type
                // base carries the discriminant, each variant subclass
                // carries its OWN payload fields (`rclass.py:82-88`,
                // registered below under `{enum}::{variant}` keys and
                // projected onto the variant classdef by
                // `getuniqueclassdef_for_enum_variant`).
                // `Rvalue::Discriminant` lowers to
                // `FieldRead("__discriminant")` on the base; a payload
                // projection resolves on the narrowed `SomeInstance(variant)`
                // receiver (`enum_variant_narrowing_knowntypedata`) and
                // reads at the variant-qualified runtime offset
                // (`resolve_adt_field` owner_root = `{enum_leaf}::{variant}`).
                // `__discriminant` width.  For a FIELDLESS enum the tag IS
                // the whole value, so model it at the enum's true byte size
                // — the heuristic layout then sizes an inline field of this
                // enum exactly (`W_ListObject.strategy` = 1 byte for a
                // `repr(u8)` `ListStrategy`), so a discriminant read of the
                // field reads only the tag, not 7 trailing padding bytes.  A
                // data-carrying enum's `layout.size` includes the variant
                // payload, which is not the tag width, so keep the wide
                // `i64` model there (the tag still sits in the low bytes; a
                // switch key compares equal under the wide read).
                // The fieldless tag is modeled at an UNSIGNED width (u8/u16/u32).
                // This is correct only while every variant discriminant is
                // non-negative: a signed-repr / negative-discriminant enum would
                // zero-extend the tag on read (e.g. `-1` → `255`) while Charon's
                // variant key stays the negative `discriminant_i64()`, so the
                // JIT switch/guard would take the wrong arm. All fieldless enums
                // lowered here today (ListStrategy / ArrayKind / DictViewKind /
                // StrategyKind / ExcKind …) use the default non-negative
                // discriminants, so unsigned is exact. A negative-discriminant
                // enum would need a signed width here AND a sign-extending field
                // read; revisit if one is introduced.
                let disc_ty = if variants.iter().all(|v| v.fields.is_empty()) {
                    match td.layout_for_target(&target).and_then(|l| l.size) {
                        Some(1) => "u8",
                        Some(2) => "u16",
                        Some(4) => "u32",
                        _ => "i64",
                    }
                } else {
                    "i64"
                };
                let rows: Vec<(String, String)> =
                    vec![("__discriminant".to_string(), disc_ty.to_string())];
                struct_fields.fields.insert(name.clone(), rows.clone());
                struct_fields.fields.insert(leaf.clone(), rows.clone());
                struct_fields.fields.insert(canon_base.clone(), rows);
                // Object identity for the enum base (the discriminant
                // carrier), minted from the crate-stripped `module::Enum`.
                let base_sid = majit_ir::descr::StructId::from_canonical(&canon_base);
                record_struct_id(&mut struct_ids, name.clone(), base_sid);
                record_struct_id(&mut struct_ids, leaf.clone(), base_sid);
                record_struct_id(&mut struct_ids, canon_base.clone(), base_sid);
                // Per-variant field rows + exact offsets under
                // `{enum}::{variant}` keys — the RPython sum-type subclass
                // layout where each variant carries its OWN fields, the base
                // only the discriminant (`rclass.py:82-88`).  These feed the
                // variant subclasses (their `getuniqueclassdef_for_enum_variant`
                // attr projection) and the per-variant runtime offsets.
                // Dual-published under the qualified and bare-leaf spellings
                // because `resolve_adt_field` emits `owner_root` =
                // `{enum_leaf}::{variant}`.  No cross-variant dedup: each
                // variant owns its field namespace.
                let enum_layout = td.layout_for_target(&target);
                // Register the enum BASE in `exact_layouts`: a single
                // `__discriminant` field at the tag's real byte position
                // (`discriminator.Branch.offset` via `discriminant_offset`).
                // `Rvalue::Discriminant` lowers to a `FieldRead("__discriminant")`
                // keyed to the base identity, so a niche/non-zero tag resolves
                // at its true offset instead of the heuristic 0.  A tag at
                // offset 0 (the common case) registers 0, matching the
                // heuristic exactly; a single-variant type has no `Branch`
                // tag (`discriminant_offset` → `None`) and also registers 0.
                if let Some(l) = enum_layout.as_ref() {
                    let mut base_offsets = std::collections::HashMap::new();
                    base_offsets.insert(
                        "__discriminant".to_string(),
                        l.discriminant_offset().unwrap_or(0),
                    );
                    let base_exact = crate::front::semantic::ExactLayout {
                        size: l.size,
                        align: l.align,
                        field_offsets: base_offsets,
                    };
                    exact_layouts.insert(base_sid, base_exact);
                }
                for (vidx, v) in variants.iter().enumerate() {
                    let variant_qual = format!("{name}::{}", v.name);
                    let variant_leaf = format!("{leaf}::{}", v.name);
                    let variant_canon = format!("{canon_base}::{}", v.name);
                    let mut vrows: Vec<(String, String)> = Vec::with_capacity(v.fields.len());
                    let mut vattrs: Vec<(String, ValueType)> = Vec::with_capacity(v.fields.len());
                    let mut voffsets: std::collections::HashMap<String, u64> =
                        std::collections::HashMap::new();
                    for (i, f) in v.fields.iter().enumerate() {
                        let fname = f.name.clone().unwrap_or_else(|| format!("__pos_{i}"));
                        // A bytecode-arg marker reads as a `u32` at runtime
                        // and annotates as an integer — keep the row string
                        // and the FORCE attr type in agreement so the
                        // projection refines the same shell.
                        let (row_ty, attr_ty) = if tyref_is_bytecode_arg_marker(&f.ty, llbc) {
                            ("u32".to_string(), ValueType::Int)
                        } else {
                            (
                                tyref_to_ast_string(&f.ty, llbc),
                                tyref_to_attr_value_type(&f.ty, llbc),
                            )
                        };
                        if let Some(off) =
                            enum_layout.as_ref().and_then(|l| l.field_offset(vidx, i))
                        {
                            voffsets.insert(fname.clone(), off);
                        }
                        vattrs.push((fname.clone(), attr_ty));
                        vrows.push((fname, row_ty));
                    }
                    struct_fields
                        .fields
                        .insert(variant_qual.clone(), vrows.clone());
                    struct_fields
                        .fields
                        .insert(variant_leaf.clone(), vrows.clone());
                    struct_fields.fields.insert(variant_canon.clone(), vrows);
                    // Object identity for the variant subclass, minted from
                    // the crate-stripped `module::Enum::Variant`.
                    let vsid = majit_ir::descr::StructId::from_canonical(&variant_canon);
                    record_struct_id(&mut struct_ids, variant_qual.clone(), vsid);
                    record_struct_id(&mut struct_ids, variant_leaf.clone(), vsid);
                    record_struct_id(&mut struct_ids, variant_canon.clone(), vsid);
                    if let Some(l) = enum_layout.as_ref() {
                        let exact = crate::front::semantic::ExactLayout {
                            size: l.size,
                            align: l.align,
                            field_offsets: voffsets,
                        };
                        exact_layouts.insert(vsid, exact);
                    }
                    // Variant payload attrs for `FORCE_ATTRIBUTES_INTO_CLASSES`,
                    // keyed by the canonical `module::Enum::Variant` — the
                    // key `_init_classdef` derives for the variant classdef
                    // whether the narrowing or the constructor minted it, so
                    // the payload attrs are forced onto the one variant
                    // class.  Mirrors the struct arm above.
                    struct_field_attrs.insert(variant_canon, vattrs);
                }
                // discriminant → variant name, published under both the
                // qualified path and the bare leaf so the opcode-dispatch
                // extractor can resolve by either spelling.
                let mut by_discr: std::collections::HashMap<i64, String> =
                    std::collections::HashMap::new();
                for v in variants {
                    let variant_path = format!("{name}::{}", v.name);
                    known_struct_names.insert(variant_path.clone());
                    if let Some(d) = v.discriminant_i64() {
                        by_discr.insert(d, v.name.clone());
                    }
                }
                if !by_discr.is_empty() {
                    enum_variant_by_discriminant.insert(name.clone(), by_discr.clone());
                    enum_variant_by_discriminant.insert(leaf, by_discr);
                }
            }
            TypeDeclKind::Alias(_) | TypeDeclKind::Opaque | TypeDeclKind::Unknown => {}
        }
    }

    for td in llbc.iter_trait_decls() {
        let name = td.item_meta.name_path();
        let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
        known_trait_names.insert(name);
        known_trait_names.insert(leaf);
    }

    (
        known_struct_names,
        known_trait_names,
        struct_fields,
        enum_variant_by_discriminant,
        struct_origins,
        struct_field_attrs,
        exact_layouts,
        struct_ids,
    )
}

/// Withdraw the bare-leaf convenience aliases for struct leaves shared
/// by two or more distinct type declarations.
///
/// The dual-publish in [`derive_program_metadata`] makes the bare-leaf
/// channels silent-winner maps on a duplicate leaf — and the two
/// winners disagree (`struct_fields.fields` insert = last-decl-wins,
/// `struct_origins` `or_insert` = first-decl-wins), so a bare lookup
/// could denote one struct while `canonical_struct_name` names another.
/// RPython cannot express this state at all: every classdef and
/// `FORCE_ATTRIBUTES_INTO_CLASSES` key is a live class OBJECT
/// (bookkeeper.py:361, classdesc.py:957), so a name is never an
/// identity.  The string-carrier stand-in therefore keeps leaf
/// resolution only while it is injective:
///
/// - `struct_fields.fields`: the bare alias is removed when the
///   colliding declarations disagree on field shape (equal-shape
///   duplicates keep the alias — any winner answers field-type lookups
///   identically).  Lookups then miss and fall to the qualified key or
///   fail conservatively (`SomeValue::Impossible` / residual call),
///   mirroring `MirGraphLookup::insert_or_mark_ambiguous`.
/// - `struct_origins`: the entry is emptied when the colliding
///   declarations live in different crate-stripped modules, which
///   `canonical_struct_name` (descr.rs:342) already treats as
///   unresolvable — the bare spelling passes through unchanged instead
///   of canonicalising to whichever module registered first.
/// - variant-leaf aliases: an enum variant dual-publishes a bare
///   `Enum::Variant` 2-segment convenience alias alongside its qualified
///   spellings.  The bare struct-leaf pass keys on the LAST segment
///   (`Variant`), which mixes distinct enums' same-named variants and
///   drops a `Variant` key that was never published, so a cross-module
///   `Enum::Variant` collision survives.  A dedicated pass groups by the
///   `Enum::Variant` tail and withdraws that alias off the same
///   field-shape-divergence signal.
///
/// The exact-layout channel is no longer hardened here: it is keyed by
/// `StructId` object identity, so two distinct definitions sharing a leaf
/// are distinct entries by construction and there is no bare alias to
/// withdraw.
///
/// Derived purely from the current qualified (`::`-containing) keys, so
/// the pass is idempotent and safe to re-run after the cross-LLBC
/// merge re-introduces a per-crate-unique alias.
fn harden_duplicate_leaf_metadata(
    struct_fields: &mut crate::front::semantic::StructFieldRegistry,
    struct_origins: &mut std::collections::HashMap<String, String>,
    enum_variant_by_discriminant: &mut std::collections::HashMap<
        String,
        std::collections::HashMap<i64, String>,
    >,
) {
    let mut by_leaf: std::collections::HashMap<&str, Vec<&str>> = std::collections::HashMap::new();
    for key in struct_fields.fields.keys() {
        if let Some((_, leaf)) = key.rsplit_once("::") {
            by_leaf.entry(leaf).or_default().push(key);
        }
    }
    let mut drop_field_aliases: Vec<String> = Vec::new();
    let mut tombstone_origins: Vec<String> = Vec::new();
    for (leaf, quals) in &by_leaf {
        if quals.len() < 2 {
            continue;
        }
        let first_rows = &struct_fields.fields[quals[0]];
        if quals[1..]
            .iter()
            .any(|q| &struct_fields.fields[*q] != first_rows)
        {
            drop_field_aliases.push((*leaf).to_string());
        }
        let first_module = strip_crate_prefix(quals[0])
            .rsplit_once("::")
            .map(|(m, _)| m.to_string())
            .unwrap_or_default();
        if quals[1..].iter().any(|q| {
            strip_crate_prefix(q)
                .rsplit_once("::")
                .map(|(m, _)| m)
                .unwrap_or_default()
                != first_module
        }) {
            tombstone_origins.push((*leaf).to_string());
        }
    }
    drop(by_leaf);
    for leaf in drop_field_aliases {
        struct_fields.fields.remove(&leaf);
    }
    for leaf in tombstone_origins {
        if let Some(module) = struct_origins.get_mut(&leaf) {
            module.clear();
        }
    }
    // Variant-leaf aliases.  The enum-variant rows (`derive_program_metadata`)
    // dual-publish each variant under qualified spellings
    // (`module::Enum::Variant`, `crate::…::Enum::Variant`) AND a bare
    // `Enum::Variant` 2-segment convenience alias.  The struct-leaf pass above
    // groups by the LAST segment (`Variant`), which mixes distinct enums'
    // same-named variants and tries to drop a bare `Variant` key that was never
    // published — so a cross-module `Enum::Variant` collision (last-decl-wins on
    // the bare alias) survives.  Group by the `Enum::Variant` tail so only
    // genuine same-enum-leaf duplicates meet, then withdraw the bare alias off
    // the same field-shape-divergence signal.  `resolve_adt_field` emits
    // `owner_root = {enum_leaf}::{variant}`, so a withdrawn alias makes an
    // ambiguous lookup miss to the qualified key or fail conservatively.
    let mut variant_by_alias: std::collections::HashMap<String, Vec<&str>> =
        std::collections::HashMap::new();
    for key in struct_fields.fields.keys() {
        if let Some((head, var)) = key.rsplit_once("::") {
            if let Some((_, enm)) = head.rsplit_once("::") {
                variant_by_alias
                    .entry(format!("{enm}::{var}"))
                    .or_default()
                    .push(key);
            }
        }
    }
    let mut drop_variant_aliases: Vec<String> = Vec::new();
    for (alias, quals) in &variant_by_alias {
        if quals.len() < 2 {
            continue;
        }
        let first_rows = &struct_fields.fields[quals[0]];
        if quals[1..]
            .iter()
            .any(|q| &struct_fields.fields[*q] != first_rows)
        {
            drop_variant_aliases.push(alias.clone());
        }
    }
    drop(variant_by_alias);
    for alias in drop_variant_aliases {
        struct_fields.fields.remove(&alias);
    }
    // `enum_variant_by_discriminant` dual-publishes the same bare-leaf
    // alias (qualified path + leaf), so a cross-decl leaf collision
    // leaves a silent-winner discriminant map there too.  Same rule as
    // `struct_fields.fields`: the bare alias survives only while every
    // qualified decl answers discriminant lookups identically.
    let mut enum_by_leaf: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    for key in enum_variant_by_discriminant.keys() {
        if let Some((_, leaf)) = key.rsplit_once("::") {
            enum_by_leaf.entry(leaf).or_default().push(key);
        }
    }
    let mut drop_enum_aliases: Vec<String> = Vec::new();
    for (leaf, quals) in &enum_by_leaf {
        if quals.len() < 2 {
            continue;
        }
        let first_map = &enum_variant_by_discriminant[quals[0]];
        if quals[1..]
            .iter()
            .any(|q| &enum_variant_by_discriminant[*q] != first_map)
        {
            drop_enum_aliases.push((*leaf).to_string());
        }
    }
    drop(enum_by_leaf);
    for leaf in drop_enum_aliases {
        enum_variant_by_discriminant.remove(&leaf);
        // Every enum base row is the `__discriminant`-only sentinel
        // (`derive_program_metadata`'s enum arm), so the struct-shape
        // check above can never tell two distinct enums sharing a leaf
        // apart — their bare base alias survives and
        // `pyre_struct_root_names` pre-mints ONE merged base class for
        // both.  Withdraw it here off the same discriminant-divergence
        // signal, gated on the sentinel so a same-named real struct
        // (whose bare alias the shape check already adjudicated) is left
        // untouched.
        if struct_fields
            .fields
            .get(&leaf)
            .is_some_and(|rows| rows.len() == 1 && rows[0].0 == "__discriminant")
        {
            struct_fields.fields.remove(&leaf);
        }
    }
}

/// Lower a single Charon [`FunDecl`] to a [`FunctionGraph`].
pub fn lower_fun_decl(llbc: &Llbc, fd: &FunDecl) -> Result<FunctionGraph, LowerError> {
    lower_fun_decl_with_static_addrs(llbc, fd, crate::HostStaticAddrs::default())
}

/// Whether the framestate-threaded lowering runs for acyclic bodies.
/// Default-on; `PYRE_MIR_FRAMESTATE=0` / `=false` is the rollback escape
/// hatch to the monotonic lowering.
fn framestate_enabled() -> bool {
    !matches!(
        std::env::var("PYRE_MIR_FRAMESTATE").as_deref(),
        Ok("0") | Ok("false")
    )
}

pub fn lower_fun_decl_with_static_addrs(
    llbc: &Llbc,
    fd: &FunDecl,
    static_addrs: crate::HostStaticAddrs<'_>,
) -> Result<FunctionGraph, LowerError> {
    let u = fd.unstructured().ok_or_else(|| {
        LowerError::Unsupported(format!(
            "{}: no Unstructured body (extracted with --ullbc?)",
            fd.item_meta.name_path()
        ))
    })?;
    let name = fd.item_meta.name_path();
    // The Result-of-PyError exception-link lowering's callee rule
    // applies when this body is a scoped callee (see
    // `front::result_exc`); the caller rule applies to the diamond
    // sites the body lowering captured.  Both run before
    // `simplify_lowered_graph` so the freed shell ops feed the same
    // dead-op sweep the Abort → RaiseImplicit fold uses.
    let result_exc_callee = crate::front::result_exc::in_result_exc_scope(&name)
        && crate::front::result_exc::tyref_is_result_of_pyerror(&fd.signature.output, llbc);
    // A `Result<(), PyError>` scoped callee returns void after the
    // exception-link lowering; widen its returnblock so the call
    // descriptor's `FUNC.RESULT` is `v`, not the `Ref`-typed unit shell.
    let result_exc_ok_is_unit = result_exc_callee
        && crate::front::result_exc::tyref_result_ok_is_unit(&fd.signature.output, llbc);
    let finish = |lo: &mut Lowering<'_>| -> Result<(), LowerError> {
        if !lo.result_exc_call_results.is_empty()
            || result_exc_callee
            || !lo.next_call_results.is_empty()
            || !lo.checked_arith_call_results.is_empty()
        {
            // The exception-link transforms run on a simplified graph,
            // as exceptiontransform.py does (graphs reach it after
            // `simplify_graph`, simplify.py:1075): the discriminant
            // switch's `default → Abort` else-unreachable arm must be
            // pruned by `remove_assertion_errors` before the diamond
            // matcher sees the switch, leaving the plain 0/1 pair.
            // Untouched graphs skip this and keep their single
            // end-of-lowering simplify, byte-identical.
            simplify_lowered_graph(&mut lo.graph);
        }
        let mut tail_forwarded_returns = 0usize;
        if !lo.result_exc_call_results.is_empty() {
            let outcome = crate::front::result_exc::rewire_result_exc_call_sites(
                &mut lo.graph,
                &lo.result_exc_call_results,
                result_exc_callee,
            )
            .map_err(LowerError::Unsupported)?;
            tail_forwarded_returns = outcome.tail_forwards;
        }
        if result_exc_callee {
            crate::front::result_exc::lower_result_exc_returns(
                &mut lo.graph,
                tail_forwarded_returns,
            )
            .map_err(LowerError::Unsupported)?;
            if result_exc_ok_is_unit {
                // Stamp `FUNC.RESULT = void`.  The exception-link lowering
                // already returns the unit `()` (the callee no longer
                // builds a `Result` shell), but `front::mir` types every
                // aggregate — including `()` — as `Ref`, so the CFG return
                // kind is still `r`.  The codewriter reconciles this by
                // collapsing the returnblock to a genuine void return
                // post-annotation (the `declared==v && cfg==r` gate),
                // mirroring `exceptiontransform.py` running after rtyping;
                // doing the structural collapse here, before the
                // whole-program annotation fixpoint, destabilises it.
                lo.graph.return_type = Some("()".to_string());
            }
        }
        // The `?`-diamond rewrite (`rewire_result_exc_call_sites`) detaches
        // the pre-rewrite branch / discriminant / break blocks: the call
        // block now exits straight to the continue arm and `exceptblock`,
        // so those blocks lose their only predecessor.  RPython graph
        // consumers only iterate blocks reachable from `startblock`
        // (`flowspace/model.py:66 iterblocks`), so drop the now-unreachable
        // blocks here — before `prune_dead_phis`, which would otherwise
        // treat a no-predecessor block as an extra root
        // (`transform_dead_op_vars`'s start set), and before the
        // `codewriter` consumers that scan `graph.blocks` directly.
        // The `next`-diamond rewrite (`front::iter_next`) runs on the same
        // simplified graph: the Option discriminant switch's default→Abort
        // arm must be pruned first, identically to the `?` diamond.  It is
        // fail-safe — a non-for-loop `Option` match is left as the residual
        // call — so it runs over every recorded site and reports how many
        // it actually rewrote.  Only an actual rewrite detaches blocks (the
        // discriminant switch), so the unreachable-block sweep is gated on
        // that count, leaving a declined graph byte-identical.
        let next_rewritten = if lo.next_call_results.is_empty() {
            0
        } else {
            crate::front::iter_next::rewire_next_call_sites(&mut lo.graph, &lo.next_call_results)
        };
        // The checked-arith rewrite (`front::checked_arith`) runs on the
        // same simplified graph as the `next`-diamond rewrite: the Option
        // discriminant switch's default→Abort arm must be pruned first,
        // identically.  It is fail-safe — a non-overflow-fallback `Option`
        // match is left as the residual call — so it runs over every
        // recorded site and reports how many it actually rewrote.  Only an
        // actual rewrite detaches the discriminant switch's block, so the
        // unreachable-block sweep is gated on that count.
        let checked_arith_rewritten = if lo.checked_arith_call_results.is_empty() {
            0
        } else {
            crate::front::checked_arith::rewire_checked_arith_call_sites(
                &mut lo.graph,
                &lo.checked_arith_call_results,
            )
        };
        if !lo.result_exc_call_results.is_empty()
            || result_exc_callee
            || next_rewritten > 0
            || checked_arith_rewritten > 0
        {
            crate::model::clear_unreachable_blocks(&mut lo.graph);
        }
        simplify_lowered_graph(&mut lo.graph);
        // `format!`-chain expansion (#131): rewrite the recognized
        // `Argument::new_display`/`Arguments::new`/`alloc::fmt::format`
        // chain into native `str` + `ll_strconcat` ops so the graph-less
        // fmt externs stop blocking the rtyper.  All emitted ops are ones
        // the legacy walker and codewriter already handle, so it runs
        // unconditionally.
        let mut fmt_collapsed = collapse_fmt_chains(&mut lo.graph);
        // Multi-argument `format!("{a}…{b}", …)` chains build an N-field
        // argument tuple and N `Argument::new_display` ctors across several
        // blocks — a shape the single-argument collapser above does not
        // recognize.  This phase rewrites each `new_display` to `str` in
        // place and folds the rendered values with the literal pieces at
        // the `Arguments::new` block, so multi-arg Display chains lower to
        // native `str` + `ll_strconcat` like the single-arg case.
        fmt_collapsed += collapse_fmt_chains_multi(&mut lo.graph);
        // Re-threading a rendered value onto a forwarding link and deleting
        // the chain's intermediate ops can leave a block with no remaining
        // predecessor (the pre-collapse branch arm) whose `Link.args` still
        // reference a now-deleted chain var; under framestate threading such
        // an orphan also pins the deleted var into a merge phi.  Drop the
        // orphaned blocks and the dead phis so the adapter does not later
        // reject the graph on an undefined `Link.args` operand — the same
        // post-collapse cleanup the panic-message phase below performs.
        if fmt_collapsed > 0 {
            crate::model::clear_unreachable_blocks(&mut lo.graph);
            crate::model::prune_dead_phis(&mut lo.graph);
        }
        // `panic!` / `assert!` message-block chains end in an implicit
        // `AssertionError` raise but route through graph-less `fmt`
        // message externs the rtyper can't type; collapse them to the
        // bare raise so `remove_assertion_errors` prunes the branch as it
        // does for a direct implicit raise.  Gated on an actual collapse
        // so untouched graphs keep their single end-of-lowering simplify.
        if collapse_panic_message_chains(&mut lo.graph) > 0 {
            crate::model::clear_unreachable_blocks(&mut lo.graph);
            simplify_lowered_graph(&mut lo.graph);
        }
        Ok(())
    };
    // Framestate-threaded lowering (the GAP-B path that threads locals
    // as block inputargs / phis — the orthodox flowspace shape).
    // Default-on; `PYRE_MIR_FRAMESTATE=0` rolls back to the monotonic
    // lowering.  Acyclic bodies thread via the two-pass RPO walk; cyclic
    // bodies additionally pre-seed each loop header's entry framestate
    // with the live-in phis `new` pre-bound, so the back-edge threads
    // into them in pass 2 instead of skipping the body to the monotonic
    // single-slot scheme.  On any framestate failure the body falls back
    // to the monotonic path — so the threaded path is never worse than
    // the monotonic one — unless `PYRE_MIR_FRAMESTATE_STRICT` is set,
    // which propagates the error for debugging.
    if framestate_enabled() {
        let mut lo = Lowering::new(llbc, name.clone(), &u, static_addrs, fd.generics.as_ref())?;
        // Back-edge targets (loop headers); empty for an acyclic body, in
        // which case `lower_framestate` reduces exactly to the two-pass
        // RPO walk.  Treat the threaded lowering and its shared
        // post-lowering stage (`finish`) as one attempt.  `finish` runs
        // uniformly, same as the linear / RPO paths below — it folds
        // constant exitswitches, drops `Abort` arms, runs
        // `simplify_lowered_graph` and installs the exception-link ABI
        // (raise-through vs Result return); gating it on result-exc
        // activity would let the framestate path keep dead arms the other
        // strategies prune.  A post-pass (e.g. the result-exc diamond
        // rewrite) can reject a body the threading itself accepted, and
        // the threading can fail on a CFG shape it does not yet model, so
        // fold both into one fallback: on any error fall through to the
        // monotonic path below, which is known to lower the body — the
        // threaded path is never worse than the monotonic one.
        let loop_headers = lo.mir_model_loop_headers();
        let attempt = lo
            .lower_framestate(&loop_headers)
            .and_then(|()| finish(&mut lo));
        match attempt {
            Ok(()) => return Ok(lo.graph),
            Err(e) => {
                if std::env::var_os("PYRE_MIR_FRAMESTATE_DEBUG").is_some() {
                    eprintln!("[FRAMESTATE fallback] {:?}: {e:?}", name);
                }
                if std::env::var_os("PYRE_MIR_FRAMESTATE_STRICT").is_some() {
                    return Err(e);
                }
            }
        }
    }
    let mut lo = Lowering::new(llbc, name.clone(), &u, static_addrs, fd.generics.as_ref())?;
    match lo.lower(BlockOrder::Linear) {
        Ok(()) => {
            finish(&mut lo)?;
            Ok(lo.graph)
        }
        // A forward-referenced definition — typically a `TermKind::Call`
        // dest at a higher MIR index than the block that reads it — reads
        // as an uninitialised local under MIR-index order.  Re-lower the
        // whole body in reverse-postorder, which visits the defining
        // block first.  This is scoped to exactly the bodies that fail
        // linearly (every other body keeps its linear-order bindings
        // untouched).  Loop-carried locals do not need RPO at all: they
        // ride the per-block live-in inputargs minted in `new` and
        // threaded by `edge_args` across the back-edge (phi / block
        // inputargs), which is order-independent.  RPO only resolves the
        // acyclic forward-reference case above.
        Err(LowerError::Unsupported(msg)) if is_known_lowering_gap(&msg) => {
            let mut lo = Lowering::new(llbc, name, &u, static_addrs, fd.generics.as_ref())?;
            lo.lower(BlockOrder::ReversePostorder)?;
            finish(&mut lo)?;
            Ok(lo.graph)
        }
        Err(e) => Err(e),
    }
}

/// Per-graph simplification after lowering — the model-layer slice of
/// RPython `simplify_graph(graph)` (`simplify.py:1075-1081`), which
/// upstream runs on every freshly built flow graph.  Only the passes
/// the Abort → `RaiseImplicit` fold needs are wired:
///
/// - `eliminate_empty_blocks` (simplify.py:52-69) collapses the empty
///   raise block between a discriminant switch's `else
///   unreachable!()` exit and `exceptblock`, exposing the
///   `[Constant(AssertionError), …]` link to the next pass.
/// - `remove_assertion_errors` (simplify.py:321-346) prunes the
///   shouldn't-occur branch and promotes the surviving exit to an
///   unconditional link.
/// - `prune_dead_phis` (`transform_dead_op_vars`, simplify.py:422-524)
///   melts the now-dead condition ops — the `__discriminant`
///   FieldRead feeding the dropped exitswitch
///   (`removeassert.py:35-37` "now melt away the (hopefully) dead
///   operation that compute the condition").  Upstream leaves this to
///   the later backendopt sweep (`backendopt/all.py`); the model
///   layer has no later sweep, so it runs here, gated on an actual
///   removal to keep untouched graphs byte-identical.
fn simplify_lowered_graph(graph: &mut FunctionGraph) {
    crate::model::eliminate_empty_blocks(graph);
    // Lower the boxing-constructor idiom `malloc_typed(W_FloatObject{…})`
    // to a native `NewWithVtable` + payload store before the dead-aggregate
    // sweep, which then reclaims the orphaned construct-on-stack ctor and
    // header field writes.
    let mut dirty = crate::model::fuse_boxing_alloc(graph) > 0;
    // Drop dead aggregate constructions (malloc + field stores whose
    // result is never read) before the dead-op sweep — `prune_dead_phis`
    // keeps them because a `FieldWrite` is side-effecting, so its `base`
    // pins the aggregate (`remove_simple_mallocs`, malloc.py).
    dirty |= crate::model::remove_dead_aggregates(graph) > 0;
    dirty |= crate::model::remove_assertion_errors(graph) > 0;
    // Constant-condition arms (`if WITHPREBUILTINT { … }` with the
    // config const folded by `const_eval_global`) collapse to the
    // taken link and the dead arm is emptied — the registry lift
    // (`translate_op`) walks blocks by index, so a disconnected arm
    // reading an unliftable static (`SMALL_INTS`) would otherwise
    // still fail the whole graph.  `constant_fold_graph` link
    // folding (backendopt/constfold.py).
    if crate::model::fold_constant_exitswitch(graph) > 0 {
        crate::model::clear_unreachable_blocks(graph);
        dirty = true;
    }
    if dirty {
        crate::model::prune_dead_phis(graph);
    }
}

/// Order in which [`Lowering::lower`] walks the MIR basic blocks.
#[derive(Clone, Copy)]
enum BlockOrder {
    /// Plain MIR index order (`0..len`) — the default.
    Linear,
    /// Reverse-postorder of the CFG — the fallback used only for bodies
    /// whose linear lowering hits a forward-referenced (uninitialised)
    /// local.
    ReversePostorder,
}

/// Errors the driver fails with. The driver fails loud — `Unsupported`
/// surfaces a precise variant + the MIR shape that prompted it so
/// each widening can be a small targeted change.
#[derive(Debug)]
pub enum LowerError {
    FunctionNotFound(String),
    /// A MIR construct the current driver does not yet handle.
    Unsupported(String),
    /// A failure to project raw JSON into the typed ULLBC subset.
    Schema(String),
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::FunctionNotFound(n) => write!(f, "function not found: {n}"),
            LowerError::Unsupported(s) => write!(f, "unsupported MIR: {s}"),
            LowerError::Schema(s) => write!(f, "schema decode: {s}"),
        }
    }
}

impl std::error::Error for LowerError {}

// ---------------------------------------------------------------------------
// Lowering state
// ---------------------------------------------------------------------------

/// The `(base, index)` operands of a devirtualized workspace index call,
/// recorded for the paired `*p = v` write.  Each operand keeps the
/// resolving-block Variable plus its source MIR local (`None` for a
/// constant operand) so the write site can re-resolve through
/// `local_var` after the operand is rebound across the call's block
/// split.  See [`Lowering::index_elem_alias`].
#[derive(Clone)]
struct IndexElemAlias {
    base_local: Option<usize>,
    base_var: Variable,
    index_local: Option<usize>,
    index_var: Variable,
}

struct Lowering<'a> {
    graph: FunctionGraph,
    llbc: &'a Llbc,
    body: &'a Unstructured,
    static_addrs: crate::HostStaticAddrs<'a>,
    arg_count: usize,
    /// `local_var[i] = Some(var)` once MIR local `i` has been bound to
    /// a flowspace Variable. Slot 0 is the return value, 1..arg_count
    /// are arguments, the rest are introduced lazily by the first
    /// `Assign` that writes them. Local 0 stays `None` until a Return
    /// terminator wires it up — the Return path reads MIR local 0 and
    /// drops a `Link([value], returnblock)` so we never need to mint
    /// a Variable for it.
    local_var: Vec<Option<Variable>>,
    /// `block_id[i]` = FunctionGraph BlockId for MIR basic block `i`.
    block_id: Vec<BlockId>,
    /// MIR locals that are live when entering each block. Non-entry
    /// blocks receive these through `Block.inputargs`, and predecessor
    /// edges pass the matching current Variables via `Link.args`.
    block_live_in: Vec<Vec<bool>>,
    block_entry_local_var: Vec<Vec<Option<Variable>>>,
    block_entry_positional_aggregate_locals: Vec<std::collections::HashMap<usize, String>>,
    block_positional_seen: Vec<Vec<bool>>,
    block_positional_conflict: Vec<Vec<bool>>,
    /// Maps each MIR local whose current binding was produced by a
    /// positional [`Rvalue::Aggregate`] (tuple / array / closure — any
    /// kind for which [`Lowering::resolve_aggregate_adt`] returns
    /// `None`) to the `owner_root` its construction-side `FieldWrite`
    /// chain used.  Such a local holds a synthetic transparent-ctor
    /// base with a `__pos_<i>` `FieldWrite` chain, so its `.N` reads
    /// must resolve to a symmetric `FieldRead __pos_<N>` in
    /// [`Lowering::resolve_place`] — carrying the *same* `owner_root` —
    /// rather than collapsing to the base Variable.  The stored owner
    /// is required because Charon's tuple `Aggregate` kind serialises
    /// as `{"Adt": [{"id": "Tuple", ..}, ..]}` (owner_root `"Adt"`)
    /// while the matching `Field` projection container serialises as
    /// `{"Tuple": N}`, so the read side cannot re-derive the
    /// construction owner from its own payload.  Excludes the
    /// `*Checked` `(value, bool)`-as-`BinOp` locals (those are bound by
    /// [`Rvalue::BinaryOp`], never an Aggregate), so their `.0` reads
    /// still fall through.
    positional_aggregate_locals: std::collections::HashMap<usize, String>,
    /// MIR locals bound by a scalar [`Rvalue::BinaryOp`] anywhere in the
    /// body.  A `*Checked (value, bool)` operation lowers to a single
    /// scalar `BinOp` (the [`Rvalue::BinaryOp`] arm), so the destination
    /// local — though MIR-typed `(numeric, bool)` — holds one scalar
    /// Variable, not a tuple.  Its `.0` projection therefore collapses to
    /// that scalar instead of extracting a tuple element; a `.1` read is
    /// the Rust overflow bit, which the JIT IR does not model, so it fails
    /// loud in [`Lowering::resolve_place`] rather than aliasing the
    /// overflow bool to the arithmetic value.  A MIR local's type is
    /// fixed, so a local bound by `BinaryOp` is a scalar at every read
    /// site (its `(numeric, bool)` type can never alias a genuine data
    /// tuple); a single function-wide set needs no per-block propagation.
    /// Distinguishes the collapse case from a genuine Ref tuple `.N` read
    /// in [`Lowering::resolve_place`].
    binop_result_locals: std::collections::HashSet<usize>,
    /// MIR locals bound by a devirtualized workspace `Index::index` /
    /// `IndexMut::index_mut` call, mapped to the `(base, index)`
    /// operand pair.  Those impls bottom out at raw-slice
    /// construction (`as_mut_slice` → `from_raw_parts`), which has no
    /// graph lowering, so the callsite is lowered as RPython's
    /// getarrayitem instead and the paired `*p = v` write
    /// ([`Lowering::emit_projection_write`] `Deref` arm) consults
    /// this map to emit `ArrayWrite` rather than the opaque
    /// `__deref_write` marker.  The base/index are kept as both the
    /// resolving-block Variable and their source MIR local: the write
    /// usually lands in a later block (the index call terminates its
    /// own block), where the operands have been rebound to fresh
    /// inputarg Variables, so the consumer re-resolves through
    /// `local_var` by local and only falls back to the recorded
    /// Variable for a base/index without a backing local (a constant).
    index_elem_alias: std::collections::HashMap<usize, IndexElemAlias>,
    /// MIR locals whose enum discriminant is a translation-time
    /// constant: single-assignment locals bound by an always-`Ok`
    /// decomposed conversion ([`Lowering::try_lower_usize_try_from`]).
    /// `Rvalue::Discriminant` on such a local emits `ConstInt(tag)`
    /// instead of the synthetic FieldRead, which lets the
    /// `lower_switch` Constant fold drop the statically dead
    /// `Err`/panic arm even though MIR calls terminate their block
    /// (the switch always sits in a successor block).  Only locals
    /// outside [`Lowering::multi_assigned_locals`] enter this map, so
    /// a re-bound local can never carry a stale tag.
    const_discriminant_locals: std::collections::HashMap<usize, i64>,
    /// MIR locals assigned more than once anywhere in the body
    /// (statement assigns + call destinations).  Guard set for
    /// [`Lowering::const_discriminant_locals`].
    multi_assigned_locals: std::collections::HashSet<usize>,
    /// Result `Variable`s of calls to `RESULT_EXC_LOWERING_SCOPE`
    /// callees whose declared result is `Result<T, PyError>`.  Each
    /// heads a `Try::branch` diamond that
    /// [`crate::front::result_exc::rewire_result_exc_call_sites`]
    /// rewires into `ExitSwitch::LastException` exits after the body
    /// lowering completes.  The paired `Option<String>` is the callee's
    /// per-instantiation `<…>` suffix (Ref-shaped payloads only) keying
    /// the rebuilt `Ok`/`Err` shells' ClassDef per instantiation.
    result_exc_call_results: Vec<(Variable, Option<String>)>,
    /// `Iterator::next()` call results (`Option<T>`-typed) recorded for
    /// the `next`-diamond rewiring pass (`front::iter_next`) that runs
    /// after the body lowering completes.
    next_call_results: Vec<Variable>,
    /// `i64::checked_{add,sub,mul}()` call results (`Option<i64>`-typed)
    /// recorded for the checked-arith rewiring pass
    /// (`front::checked_arith`) that runs after the body lowering
    /// completes, rewriting each into an `*_ovf` op + OverflowError edge.
    checked_arith_call_results: Vec<Variable>,
}

impl<'a> Lowering<'a> {
    fn new(
        llbc: &'a Llbc,
        name: String,
        body: &'a Unstructured,
        static_addrs: crate::HostStaticAddrs<'a>,
        generics: Option<&serde_json::Value>,
    ) -> Result<Self, LowerError> {
        let mut graph = FunctionGraph::new(name);
        let n_locals = body.locals.locals.len();
        let mut local_var: Vec<Option<Variable>> = vec![None; n_locals];

        let arg_count = body.locals.arg_count as usize;
        // Arguments become startblock inputargs in source order
        // (RPython parity: `flowcontext.py:333` populates `locals_w[:argcount]`
        // from `flowmodel.py:130` `Block(inputargs)`).
        //
        // Each parameter is also emitted as a paired `OpKind::Input { name,
        // ty }` op into the startblock.  Downstream consumers
        // — `flowspace_adapter::derive_subject_inputcells`
        // (`translator/rtyper/flowspace_adapter.rs:1464+`),
        // `graph_non_void_arg_types` (`codewriter/call.rs:2748+`),
        // `type_state` (`codewriter/type_state.rs:131`) — locate
        // each inputarg's declared `ValueType` by scanning the leading
        // `OpKind::Input` ops with `op.result == &arg`.  Without the
        // Input op, `derive_subject_inputcells` fails-loud at
        // `flowspace_adapter.rs:1504` for any MIR-built graph that
        // reaches the real-rtyper dual-gate.
        let mut startblock_args: Vec<Variable> = Vec::with_capacity(arg_count);
        let mut input_ops: Vec<SpaceOperation> = Vec::with_capacity(arg_count);
        for i in 1..=arg_count {
            let local = &body.locals.locals[i];
            let name = local.name.clone().unwrap_or_else(|| format!("arg{i}"));
            let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            // Register a stable name so canonical comparison can spot
            // arg-renames.  Names live on the value via `name_value_var`
            // (mirrors the `parse.rs` arg-binding path).
            graph.name_value_var(&var, name.clone());
            local_var[i] = Some(var.clone());
            let ty = tyref_to_value_type(&local.ty, llbc);
            // `class_root` carries the param's named-ADT leaf so
            // `derive_subject_inputcells` can seed the receiver's
            // `ClassDef`; only `Ref`-typed params consume it there.  A
            // generic param (`&T` where `T: Trait`, incl. trait default
            // bodies' `&Self`) has no ADT leaf — carry the bound
            // trait's qualified path instead, which the adapter
            // resolves through the unique-impl map
            // (`pyre_trait_unique_impls`, keyed by qualified path).
            let class_root = match &ty {
                ValueType::Ref(_) => tyref_class_root(&local.ty, llbc)
                    // A `&str` / `str` param strips to the `str` builtin
                    // (not an ADT), so `tyref_class_root` answers `None`;
                    // name it `"str"` so `derive_subject_inputcells` seeds
                    // the byte `SomeString` (`s_str0`) instead of the
                    // abstract `SomeInstance(None)` a `Ref(None)` projects
                    // to.  A string param compared against a string literal
                    // then rtypes as `pair(StringRepr, StringRepr)` rather
                    // than walling at `pair(InstanceRepr, StringRepr)`.
                    .or_else(|| tyref_strips_to_str(&local.ty, llbc).then(|| "str".to_string()))
                    .or_else(|| tyref_generic_trait_bound_root(&local.ty, llbc, generics))
                    // A list-typed param (`Vec<T>`, `&[T]`, …) has no
                    // named-ADT leaf — `tyref_class_root` answers `None`
                    // because `adt_node_class_root` excludes the
                    // core/std/alloc container family from classdef
                    // minting.  Carry its full monomorphic spelling so
                    // `derive_subject_inputcells` projects it through the
                    // annotator's list model (`project_pyre_field_type`)
                    // instead of the classdef-less `SomeInstance(None)`
                    // shell, on which a `len()` / iteration would wall at
                    // `getattr` over a classdef-less instance.
                    .or_else(|| {
                        let spelling = tyref_to_ast_string(&local.ty, llbc);
                        majit_ir::descr::is_list_container_spelling(&spelling).then_some(spelling)
                    }),
                _ => None,
            };
            input_ops.push(SpaceOperation {
                result: Some(var.clone()),
                kind: OpKind::Input {
                    name,
                    ty,
                    class_root,
                },
            });
            startblock_args.push(var);
        }
        // Startblock gets the args as its inputargs. The startblock is
        // BlockId(0), already created by `FunctionGraph::new`.
        for var in &startblock_args {
            graph.push_inputarg_var(graph.startblock, var.clone());
        }
        // Push the paired `OpKind::Input` ops into the startblock so
        // `derive_subject_inputcells` can project each inputarg's
        // declared ValueType to a SomeValue shell.
        graph
            .block_mut(graph.startblock)
            .operations
            .extend(input_ops);

        // Pre-allocate a Block for each MIR basic block so terminators
        // can refer to successors via stable BlockId. MIR bb0 maps to
        // the FunctionGraph startblock (already exists); the rest are
        // freshly created.
        let mut block_id: Vec<BlockId> = Vec::with_capacity(body.body.len());
        block_id.push(graph.startblock);
        for _ in 1..body.body.len() {
            block_id.push(graph.create_block());
        }
        let index_write_extra_live = compute_index_write_extra_live(body, llbc);
        let block_live_in = compute_mir_liveness(body, &index_write_extra_live);
        let mut block_entry_local_var = vec![vec![None; n_locals]; body.body.len()];
        let block_entry_positional_aggregate_locals =
            vec![std::collections::HashMap::new(); body.body.len()];
        if !block_entry_local_var.is_empty() {
            block_entry_local_var[0] = local_var.clone();
        }
        for mir_bb in 1..body.body.len() {
            for local_idx in 0..n_locals {
                if !block_live_in
                    .get(mir_bb)
                    .and_then(|locals| locals.get(local_idx))
                    .copied()
                    .unwrap_or(false)
                {
                    continue;
                }
                let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                graph.push_inputarg_var(block_id[mir_bb], var.clone());
                block_entry_local_var[mir_bb][local_idx] = Some(var);
            }
        }

        Ok(Self {
            graph,
            llbc,
            body,
            static_addrs,
            arg_count,
            local_var,
            block_id,
            block_live_in,
            block_entry_local_var,
            block_entry_positional_aggregate_locals,
            block_positional_seen: vec![vec![false; n_locals]; body.body.len()],
            block_positional_conflict: vec![vec![false; n_locals]; body.body.len()],
            positional_aggregate_locals: std::collections::HashMap::new(),
            binop_result_locals: compute_binop_result_locals(body),
            index_elem_alias: std::collections::HashMap::new(),
            const_discriminant_locals: std::collections::HashMap::new(),
            multi_assigned_locals: compute_multi_assigned_locals(body),
            result_exc_call_results: Vec::new(),
            next_call_results: Vec::new(),
            checked_arith_call_results: Vec::new(),
        })
    }

    fn lower(&mut self, order: BlockOrder) -> Result<(), LowerError> {
        // Each MIR basic block is a FlowGraph block.  Locals live across
        // a successor edge are explicit `Link.args` into the target
        // block's `inputargs`, mirroring FlowContext.mergeblock rather
        // than relying on a function-wide slot table.
        for mir_bb in self.block_processing_order(order) {
            self.lower_block(mir_bb)?;
        }
        Ok(())
    }

    /// Block processing order.  `Linear` is plain MIR index order.
    /// `ReversePostorder` is the reverse-postorder of the MIR CFG rooted
    /// at bb0, followed by any blocks unreachable from bb0 (ascending
    /// index, so the graph stays complete — every block is still
    /// lowered).  Successors mirror the `lower_terminator` edges exactly
    /// (normal target *and* `on_unwind` for `Call`/`Assert`/`Drop`; both
    /// arms of an `If`; every arm plus the default of a `SwitchInt`) so
    /// this order matches the CFG the classifier diagnostic validated.
    fn block_processing_order(&self, order: BlockOrder) -> Vec<usize> {
        let n = self.body.body.len();
        if matches!(order, BlockOrder::Linear) {
            return (0..n).collect();
        }
        if n == 0 {
            return vec![];
        }
        let succs = |bb: usize| -> Vec<usize> {
            let Ok(term) = self.body.body[bb].term() else {
                return vec![];
            };
            let raw: Vec<u64> = match term {
                TermKind::Goto { target } => vec![target],
                TermKind::Call {
                    target, on_unwind, ..
                }
                | TermKind::Assert {
                    target, on_unwind, ..
                }
                | TermKind::Drop {
                    target, on_unwind, ..
                } => vec![target, on_unwind],
                TermKind::Switch { targets, .. } => match targets {
                    SwitchTargets::If(a, b) => vec![a, b],
                    SwitchTargets::SwitchInt(_, arms, default) => {
                        let mut v: Vec<u64> = arms.iter().map(|(_, bb)| *bb).collect();
                        v.push(default);
                        v
                    }
                },
                TermKind::Return
                | TermKind::UnwindResume
                | TermKind::Abort(_)
                | TermKind::Unknown => vec![],
            };
            raw.into_iter()
                .map(|t| t as usize)
                .filter(|&t| t < n)
                .collect()
        };

        // Iterative DFS recording postorder; reverse-postorder is its
        // reverse.  `state`: 0 = white (unvisited), 1 = grey (on stack),
        // 2 = black (done).  Stack entries are `(node, next-succ-index)`.
        let mut state = vec![0u8; n];
        let mut postorder: Vec<usize> = Vec::with_capacity(n);
        let mut stack: Vec<(usize, usize)> = Vec::new();
        state[0] = 1;
        stack.push((0, 0));
        while let Some(&(node, idx)) = stack.last() {
            let s = succs(node);
            if idx < s.len() {
                stack.last_mut().unwrap().1 += 1;
                let nxt = s[idx];
                if state[nxt] == 0 {
                    state[nxt] = 1;
                    stack.push((nxt, 0));
                }
            } else {
                state[node] = 2;
                postorder.push(node);
                stack.pop();
            }
        }
        let mut order: Vec<usize> = postorder.into_iter().rev().collect();
        // Blocks unreachable from bb0 are still lowered (kept complete),
        // last and in MIR order — after every reachable def is seeded, so
        // they can only see *more* bindings than linear order did.
        for bb in 0..n {
            if state[bb] != 2 {
                order.push(bb);
            }
        }
        order
    }

    // -----------------------------------------------------------------------
    // Framestate-threaded lowering (acyclic GAP-B path)
    // -----------------------------------------------------------------------

    /// Snapshot the current `local_var` table as a [`FrameState`].  Only
    /// the locals projection is populated — MIR has no value stack /
    /// pending exception / block-stack, so those stay at the `Default`
    /// (empty) shape.  `entries` is indexed by MIR local index; every
    /// framestate produced in this lowering uses the same indexing, so
    /// [`FrameState::union`] / [`FrameState::getvariables`] /
    /// [`FrameState::getoutputargs`] — all purely positional over
    /// `entries` / `locals_w` — line up across predecessors and merge
    /// targets without consulting any external slot table.
    fn getstate(&self) -> FrameState {
        FrameState {
            entries: self.local_var.clone(),
            ..Default::default()
        }
    }

    /// Restore `local_var` from a [`FrameState`] produced by
    /// [`Self::getstate`] or [`FrameState::union`].  Slots are sized to
    /// the fixed local count; a `None` slot marks a local undefined on at
    /// least one predecessor path (union None-kill), so a later read
    /// fails-loud through [`Self::resolve_place`] as "uninitialised
    /// local" — the same use-before-def signal the monotonic path emits.
    fn setstate(&mut self, fs: &FrameState) {
        let n = self.local_var.len();
        self.local_var = (0..n)
            .map(|i| fs.entries.get(i).cloned().flatten())
            .collect();
    }

    /// Successor MIR blocks along the edges [`Self::lower_terminator`]
    /// actually materialises in the model graph.  This EXCLUDES
    /// `on_unwind` (dropped by `lower_call` and the `Assert` / `Drop`
    /// arms): the reachability / acyclicity / RPO that drive framestate
    /// threading must follow only the edges that exist in the produced
    /// `FunctionGraph`, otherwise an orphan cleanup chain would appear as
    /// a live merge predecessor.
    fn model_succs(&self, mir_bb: usize) -> Vec<usize> {
        let n = self.body.body.len();
        let Ok(term) = self.body.body[mir_bb].term() else {
            return vec![];
        };
        let raw: Vec<u64> = match term {
            TermKind::Goto { target } => vec![target],
            TermKind::Call { target, .. }
            | TermKind::Assert { target, .. }
            | TermKind::Drop { target, .. } => vec![target],
            TermKind::Switch { targets, .. } => match targets {
                SwitchTargets::If(a, b) => vec![a, b],
                SwitchTargets::SwitchInt(_, arms, default) => {
                    let mut v: Vec<u64> = arms.iter().map(|(_, bb)| *bb).collect();
                    if !self.switch_default_targets_panic_abort(default) {
                        v.push(default);
                    }
                    v
                }
            },
            TermKind::Return | TermKind::UnwindResume | TermKind::Abort(_) | TermKind::Unknown => {
                vec![]
            }
        };
        raw.into_iter()
            .map(|t| t as usize)
            .filter(|&t| t < n)
            .collect()
    }

    /// True when MIR block `bb`'s terminator is a panic/abort stub
    /// (`Abort` / `UnwindResume`).  rustc lowers the out-of-range
    /// `default` arm of an enum-discriminant `SwitchInt` to such a block
    /// — an unreachable UB stub with no flowgraph analogue.  Excluding it
    /// from the switch's successors keeps the orphan `set_raise`
    /// cleanup chain (with its undefined etype/evalue) out of the produced
    /// graph instead of leaving it as a live, undefined-operand block.
    fn switch_default_targets_panic_abort(&self, bb: u64) -> bool {
        matches!(
            self.body.body.get(bb as usize).and_then(|b| b.term().ok()),
            Some(TermKind::Abort(_)) | Some(TermKind::UnwindResume)
        )
    }

    /// Blocks reachable from bb0 over [`Self::model_succs`].
    fn mir_model_reachable(&self) -> Vec<bool> {
        let n = self.body.body.len();
        let mut reached = vec![false; n];
        if n == 0 {
            return reached;
        }
        reached[0] = true;
        let mut stack = vec![0usize];
        while let Some(bb) = stack.pop() {
            for s in self.model_succs(bb) {
                if !reached[s] {
                    reached[s] = true;
                    stack.push(s);
                }
            }
        }
        reached
    }

    /// Loop headers of the model-reachable component (from bb0 over
    /// [`Self::model_succs`]): the targets of back-edges.  Indexed by MIR
    /// block; `true` iff the block is the target of at least one retreating
    /// edge in the DFS — i.e. a successor still grey (on the DFS stack)
    /// when its predecessor reaches it.  Returns all-`false` for an
    /// acyclic body, so passing this to [`Self::lower_framestate`] reduces
    /// that path to the plain two-pass RPO walk.
    ///
    /// For a reducible body (every retreating edge's target dominates its
    /// source — the shape structured Rust control flow always produces)
    /// these are exactly the natural loop headers, independent of DFS
    /// order.  The framestate path uses them to pre-seed each loop
    /// header's entry state with its pre-bound live-in phis so the
    /// back-edge can thread into them; an irreducible body may mis-seed
    /// and produce an undefined-operand graph, which the pass-2 / final
    /// self-validation guards reject into the monotonic fallback.
    fn mir_model_loop_headers(&self) -> Vec<bool> {
        let n = self.body.body.len();
        let mut headers = vec![false; n];
        if n == 0 {
            return headers;
        }
        // 0 = white, 1 = grey (on stack), 2 = black (done).
        let mut state = vec![0u8; n];
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        state[0] = 1;
        while let Some(&(node, idx)) = stack.last() {
            let succs = self.model_succs(node);
            if idx < succs.len() {
                stack.last_mut().unwrap().1 += 1;
                let nxt = succs[idx];
                match state[nxt] {
                    0 => {
                        state[nxt] = 1;
                        stack.push((nxt, 0));
                    }
                    1 => headers[nxt] = true, // grey successor ⇒ back-edge target
                    _ => {}
                }
            } else {
                state[node] = 2;
                stack.pop();
            }
        }
        headers
    }

    /// Reverse-postorder of the model-reachable component over
    /// [`Self::model_succs`].  Only blocks reachable from bb0 appear;
    /// unreachable blocks are handled separately as dead stubs.  In an
    /// acyclic graph RPO visits every predecessor of a block before the
    /// block itself, which is exactly the order the two-pass framestate
    /// threading relies on.
    fn model_rpo(&self) -> Vec<usize> {
        let n = self.body.body.len();
        if n == 0 {
            return vec![];
        }
        let mut state = vec![0u8; n];
        let mut postorder: Vec<usize> = Vec::with_capacity(n);
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        state[0] = 1;
        while let Some(&(node, idx)) = stack.last() {
            let succs = self.model_succs(node);
            if idx < succs.len() {
                stack.last_mut().unwrap().1 += 1;
                let nxt = succs[idx];
                if state[nxt] == 0 {
                    state[nxt] = 1;
                    stack.push((nxt, 0));
                }
            } else {
                state[node] = 2;
                postorder.push(node);
                stack.pop();
            }
        }
        postorder.into_iter().rev().collect()
    }

    /// Stub `bb_id` as a dead bare-raise block: clear its body / exits /
    /// exitswitch, close it with a `set_raise`, and mark it `dead`.  Used
    /// for blocks the framestate threading must not lower — model-
    /// unreachable orphan `on_unwind` chains and `If`-arms a const-bool
    /// discriminant folded away (see [`Self::lower_framestate`]).  The
    /// real-path `function_graph_to_flowspace` prunes `dead` blocks
    /// (`remove_dead_blocks` parity), so the stub's orphan etype/evalue
    /// never reach the rtyper as undefined operands.
    fn stub_dead_block(&mut self, bb_id: BlockId) {
        let blk = self.graph.block_mut(bb_id);
        blk.operations.clear();
        blk.exits.clear();
        blk.exitswitch = None;
        self.graph.set_raise(bb_id, "mir-dead");
        self.graph.block_mut(bb_id).dead = true;
    }

    /// Lower the body threading locals as block inputargs / phis via
    /// per-block [`FrameState`]s, instead of the function-wide monotonic
    /// `local_var` table.  Handles both acyclic and cyclic bodies; this
    /// drains the GAP-B "undefined operand" census skips where a
    /// reassigned local reaches a merge with path-dependent values the
    /// monotonic single-slot scheme cannot represent, and puts cyclic
    /// bodies on the same orthodox flowspace shape as acyclic ones.
    ///
    /// `loop_headers` (from [`Self::mir_model_loop_headers`]) marks the
    /// back-edge targets.  For an acyclic body it is all-`false` and this
    /// reduces exactly to the two-pass RPO walk.
    ///
    /// Two passes over the model-reachable blocks in reverse-postorder:
    ///
    ///   Pass 0 (cyclic only) — pre-seed every loop header's entry
    ///   framestate with the live-in phis `new` pre-bound for it
    ///   (`block_entry_local_var[H]`).  The RPO walk visits a loop header
    ///   before its back-edge predecessor (latch), so without a seed the
    ///   header's entry would miss the back-edge's contribution; seeding
    ///   it with the live-in Variables makes the header's inputargs real
    ///   phis that both the forward predecessor(s) and the latch thread
    ///   into during pass 2.
    ///
    ///   Pass 1 — for each block, `setstate` to its accumulated entry
    ///   framestate, set its (non-startblock) inputargs to the entry
    ///   framestate's variables, lower its statements + terminator
    ///   (reusing the monotonic per-op lowering, which closes
    ///   return / raise correctly and emits placeholder empty-args
    ///   goto / branch links because every successor still has empty
    ///   inputargs at close time), snapshot the exit framestate, and
    ///   union that exit into each model successor's entry framestate —
    ///   except a successor that is a loop header, whose pre-seeded entry
    ///   is fixed (the predecessor still threads into it in pass 2).
    ///
    ///   Pass 2 — re-argument each goto / branch link from the
    ///   predecessor exit / target entry framestates via `getoutputargs`,
    ///   preserving the link's target, exitcase, and the block's
    ///   `exitswitch`.
    ///
    /// Model-unreachable blocks (orphan `on_unwind` cleanup chains) are
    /// stubbed as dead raises so the graph stays complete without
    /// threading dead state — leaving their original content would
    /// reference the monotonic carry-on `local_var`.  bb0 (the
    /// startblock) keeps its `OpKind::Input`-paired parameter inputargs
    /// untouched — the opcode-dispatch arm extractor depends on that
    /// shape — so a back-edge into bb0 (which would demand reseeding the
    /// parameter slots as phis) declines to the monotonic fallback.
    fn lower_framestate(&mut self, loop_headers: &[bool]) -> Result<(), LowerError> {
        let n = self.body.body.len();
        if n == 0 {
            return Ok(());
        }
        // A back-edge into bb0 would force the parameter slots to become
        // phis, but bb0 must keep its `OpKind::Input`-paired parameter
        // inputargs (the opcode-dispatch arm extractor depends on that
        // shape).  Decline such bodies to the monotonic fallback.
        if loop_headers.first().copied().unwrap_or(false) {
            return Err(LowerError::Unsupported(
                "framestate: back-edge into startblock bb0 — declines to monotonic".to_string(),
            ));
        }
        let reachable = self.mir_model_reachable();
        let rpo = self.model_rpo();
        let returnblock = self.graph.returnblock;
        let exceptblock = self.graph.exceptblock;

        // BlockId → MIR bb inverse.  returnblock / exceptblock (and any
        // non-MIR block) map to `usize::MAX` — they are merge sinks the
        // accumulation skips.
        let mut block_to_mir = vec![usize::MAX; self.graph.blocks.len()];
        for (mir, bid) in self.block_id.iter().enumerate() {
            block_to_mir[bid.0] = mir;
        }

        let mut entry_state: Vec<Option<FrameState>> = vec![None; n];
        let mut exit_state: Vec<Option<FrameState>> = vec![None; n];
        // bb0 enters with the parameter bindings established in `new`.
        entry_state[0] = Some(self.getstate());

        // Pass 0 (cyclic only) — pre-seed each loop header's entry
        // framestate with the live-in phis `new` pre-bound for it
        // (`block_entry_local_var[H]`, a fresh `Variable` per live-in
        // local already pushed as a `block_id[H]` inputarg).  The RPO
        // walk reaches a loop header before its latch, so this seed — not
        // a forward-predecessor union — is what makes the header's
        // inputargs the loop phis; the forward predecessor(s) and the
        // latch both thread into them in pass 2 via `getoutputargs`.
        // bb0 is never reseeded (guarded above + it is not in the merge
        // set), so its parameter inputargs / `Input` ops stay intact.
        for (h, &is_header) in loop_headers.iter().enumerate().take(n) {
            if is_header && h != 0 {
                entry_state[h] = Some(FrameState {
                    entries: self.block_entry_local_var[h].clone(),
                    ..Default::default()
                });
            }
        }

        // Pass 1 — RPO walk: setstate, inputargs, lower, snapshot, union.
        for &bb in &rpo {
            let st = match entry_state[bb].clone() {
                Some(st) => st,
                None => {
                    // No live predecessor edge reached this block.  RPO
                    // over `model_succs` visits every model-predecessor
                    // before `bb`, and each unions its *lowered* exits
                    // into `bb`'s entry; a still-empty entry here means
                    // every model edge into `bb` was an `If`-arm that
                    // `lower_switch` folded away on a const-bool
                    // discriminant (a translation-time `const` gate such
                    // as `if WITHPREBUILTINT`), so `bb` is unreachable in
                    // the produced graph even though `mir_model_reachable`
                    // — which reads the raw terminator, pre-fold — marks
                    // it reachable.  Stub it dead exactly like the model-
                    // unreachable orphan chains below; the real-path
                    // adapter's reachability prune removes it, and
                    // threading dead state would only mint phis the legacy
                    // fallback cannot type.  `bb` is never bb0 (the
                    // startblock is seeded and visited first), so a bare
                    // raise stub is well-formed.
                    self.stub_dead_block(self.block_id[bb]);
                    continue;
                }
            };
            self.setstate(&st);
            let bb_id = self.block_id[bb];
            // The startblock keeps its parameter inputargs + Input ops;
            // it is never a merge target in an acyclic body.
            if bb != 0 {
                let inputargs = st.getvariables(&self.graph);
                self.graph.block_mut(bb_id).inputargs = inputargs;
                // Anchor the block body to the framestate locals.
                // `lower_block` reloads `self.local_var` from
                // `block_entry_local_var[bb]` (the construction-time
                // per-block-entry Variables), so without this the body
                // would bind locals to those construction identities
                // while the inputargs above (and the `getoutputargs`
                // link args in Pass 2) carry the union-minted framestate
                // identities.  The mismatch leaves the body's operand
                // Variables defined as neither an inputarg nor an op
                // result, so `perform_register_allocation` assigns them
                // no colour and `assembler::lookup_coloring` aborts.
                // `self.local_var` here is exactly `setstate(&st)`'s
                // positional projection of the framestate entries, whose
                // Variable cells are the same identities `getvariables`
                // threaded into `inputargs`.
                self.block_entry_local_var[bb] = self.local_var.clone();
            }
            self.lower_block(bb)?;
            let mut ex = self.getstate();
            // Scrub phantom locals before threading.  A slot bound to a
            // Variable that is neither an inputarg nor an op result of this
            // block has no definition in the produced graph — e.g. a MIR
            // local the body never assigns, conservatively kept live by
            // `compute_mir_liveness` and prebound in `new`, that the
            // fully-inlined source leaves dead (the unread `MaybeUninit`
            // scratch a monomorphized `core::ptr::swap` carries).  The
            // monotonic path threads only `target_input_locals` (live-in)
            // via per-block prebind inputargs, so its copy is always a
            // defined block inputarg; the framestate threads every `Some`
            // slot positionally, so an undefined phantom would reach Pass 2
            // as a `Link.arg` defined at no source.  Drop it to `None`: a
            // dead local is irrelevant, and a later read of a genuinely
            // live-but-undefined slot still fails loud as an uninitialised
            // local through `resolve_place` — the same use-before-def
            // signal the monotonic `edge_args` raises.
            for slot in ex.entries.iter_mut() {
                if let Some(v) = slot
                    && !self.graph.variable_defined_in_block(bb_id, v)
                {
                    *slot = None;
                }
            }
            exit_state[bb] = Some(ex.clone());
            // Union this exit into each model successor's entry state.
            // Successors are read off the just-closed exits (the model
            // edges), skipping the return / except sinks and any
            // non-MIR target.
            let succ_targets: Vec<BlockId> = self
                .graph
                .block(bb_id)
                .exits
                .iter()
                .map(|l| l.target)
                .collect();
            for tgt in succ_targets {
                if tgt == returnblock || tgt == exceptblock {
                    continue;
                }
                let tmir = block_to_mir[tgt.0];
                if tmir == usize::MAX {
                    continue;
                }
                // A loop-header successor keeps its pre-seeded entry
                // (Pass 0): its live-in phis are fixed, and this
                // predecessor threads into them in pass 2.  Unioning the
                // exit in would re-mint the phi Variables on the forward
                // edge and lose the seed the latch must thread into.
                if loop_headers.get(tmir).copied().unwrap_or(false) {
                    continue;
                }
                let merged = match entry_state[tmir].take() {
                    None => ex.clone(),
                    Some(prev) => prev.union(&ex, &mut self.graph).ok_or_else(|| {
                        LowerError::Unsupported(format!(
                            "framestate: union of predecessors failed at bb{tmir}"
                        ))
                    })?,
                };
                entry_state[tmir] = Some(merged);
            }
        }

        // Model-unreachable blocks: orphan `on_unwind` cleanup chains that
        // no lowered exit targets — `lower_terminator` strips every unwind
        // edge (Goto / Assert / Drop / Call all forward to the success
        // continuation only).  `model_succs` is a *superset* of the
        // lowered exits — it reads the raw terminator and so still lists an
        // `If`-arm a const-bool discriminant later folds away — but it is
        // never a subset, so `reachable[bb]` false (outside the
        // `model_succs` closure) is a sufficient deadness signal: such a
        // block is genuinely unreferenced.  (The folded-arm case, where a
        // block IS in the closure yet has no live lowered predecessor, is
        // caught in Pass 1 by the empty-entry stub above.)  Mark them
        // `dead` so the graph stays closed for the legacy fallback path,
        // which consumes this same `FunctionGraph`.
        for bb in 0..n {
            if reachable[bb] {
                continue;
            }
            self.stub_dead_block(self.block_id[bb]);
        }

        // Pass 2 — re-argument the goto / branch links from framestates.
        for &bb in &rpo {
            let bb_id = self.block_id[bb];
            // A block Pass 1 stubbed dead (empty-entry const-fold orphan)
            // has no exit state and a bare-raise body — its links carry no
            // framestate args to re-argument.  Skip it; the final guard
            // and the real-path dead-block prune both ignore it too.
            if self.graph.block(bb_id).dead {
                continue;
            }
            let ex = exit_state[bb].clone().ok_or_else(|| {
                LowerError::Unsupported(format!("framestate: bb{bb} missing exit state in pass 2"))
            })?;
            let exits_meta: Vec<(usize, BlockId)> = self
                .graph
                .block(bb_id)
                .exits
                .iter()
                .enumerate()
                .map(|(i, l)| (i, l.target))
                .collect();
            for (idx, tgt) in exits_meta {
                if tgt == returnblock || tgt == exceptblock {
                    continue;
                }
                let tmir = block_to_mir[tgt.0];
                if tmir == usize::MAX {
                    continue;
                }
                let tgt_state = entry_state[tmir].clone().ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "framestate: bb{tmir} missing entry state in pass 2"
                    ))
                })?;
                // `try_getoutputargs` (not the panicking `getoutputargs`):
                // a loop header pre-seeded with live-in phis bypasses the
                // union's None-kill, so a phantom slot scrubbed to `None`
                // in this predecessor's exit could leave a target
                // `Variable` slot unbound in `self`.  Decline such a
                // mismatch to the monotonic fallback instead of panicking.
                let outputargs =
                    ex.try_getoutputargs(&tgt_state, &self.graph)
                        .ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "framestate: getoutputargs phantom-slot mismatch on edge bb{bb} -> \
                         bb{tmir} in graph {:?} — declines to monotonic",
                                self.graph.name,
                            ))
                        })?;
                // Self-validation: every output arg is an exit-state cell
                // of `bb`, hence must be defined in `bb_id` (as an
                // inputarg or op result).  If the threading would emit a
                // Link.arg undefined at its source block, bail to the
                // monotonic path rather than hand a malformed graph to
                // the downstream rtyper / legacy fallback (which both
                // consume this same graph and would fault on the
                // undefined operand).  This is the `flowspace_adapter`
                // "every referenced operand must be defined as a block
                // inputarg or op result" invariant, checked at the
                // threading site.
                for arg in &outputargs {
                    if let LinkArg::Value(var) = arg {
                        if !self.graph.variable_defined_in_block(bb_id, var) {
                            if std::env::var_os("PYRE_MIR_FRAMESTATE_DEBUG").is_some() {
                                self.debug_dump_undefined_link_arg(
                                    bb, bb_id, tgt, tmir, var, &ex, &tgt_state,
                                );
                            }
                            return Err(LowerError::Unsupported(format!(
                                "framestate: threaded Link.arg (var id={}) undefined in source \
                                 bb{bb} (BlockId {:?}) for edge -> bb{tmir} (BlockId {:?}) \
                                 in graph {:?}",
                                var.id(),
                                bb_id,
                                tgt,
                                self.graph.name,
                            )));
                        }
                    }
                }
                self.graph.block_mut(bb_id).exits[idx].args = outputargs;
            }
        }

        // Final adapter-rejection guard.  The real-path `flowspace_adapter`
        // rejects a graph if any non-dead block has a `Link.args` operand
        // that is not defined in its source block (an inputarg or op
        // result).  A model-reachable raise lowers to `set_raise`, whose
        // orphan etype/evalue are exactly such operands, so the graph is a
        // guaranteed real-path Skip — Pass 2 leaves those exceptblock /
        // returnblock links untouched, so scan every reachable block's
        // links here.  Threading framestate phis into a graph that will
        // Skip buys no census drain (it cannot Match) and hands the phis to
        // the legacy fallback, which cannot type non-startblock inputargs
        // and miscolours them `GcRef` (assembler Ref/Int mismatch).
        // Decline such graphs to the monotonic lowering, which the legacy
        // path consumes exactly as today; the decline is drain-neutral
        // because a Match is impossible while the orphan operand stands.
        for &bb in &rpo {
            let bb_id = self.block_id[bb];
            if self.graph.block(bb_id).dead {
                continue;
            }
            let undefined: Option<u64> = self
                .graph
                .block(bb_id)
                .exits
                .iter()
                .flat_map(|l| l.args.iter())
                .find_map(|arg| match arg {
                    LinkArg::Value(var) if !self.graph.variable_defined_in_block(bb_id, var) => {
                        Some(var.id())
                    }
                    _ => None,
                });
            if let Some(id) = undefined {
                return Err(LowerError::Unsupported(format!(
                    "framestate: declines graph {:?} — reachable bb{bb} (BlockId {:?}) has a \
                     Link.args operand (var id={id}) undefined at its source (real-path adapter \
                     would Skip; monotonic lowering is legacy-safe)",
                    self.graph.name, bb_id,
                )));
            }
        }

        Ok(())
    }

    /// Diagnostic dump for a framestate threading that would emit a
    /// `Link.arg` undefined at its source block.  Gated behind
    /// `PYRE_MIR_FRAMESTATE_DEBUG`.  Prints the source / target block
    /// shapes and both framestates so the alignment bug can be located.
    #[allow(clippy::too_many_arguments)]
    fn debug_dump_undefined_link_arg(
        &self,
        bb: usize,
        bb_id: BlockId,
        tgt: BlockId,
        tmir: usize,
        var: &Variable,
        ex: &FrameState,
        tgt_state: &FrameState,
    ) {
        let id_list = |entries: &[Option<Variable>]| -> String {
            entries
                .iter()
                .map(|e| match e {
                    Some(v) => v.id().to_string(),
                    None => "_".to_string(),
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        let src = self.graph.block(bb_id);
        let src_inputargs: Vec<u64> = src.inputargs.iter().map(|v| v.id()).collect();
        let src_op_results: Vec<u64> = src
            .operations
            .iter()
            .filter_map(|op| op.result.as_ref().map(|v| v.id()))
            .collect();
        eprintln!(
            "[FRAMESTATE undefined-link-arg] graph={:?}\n  edge bb{bb}({:?}) -> bb{tmir}({:?})\n  \
             undefined var id={}\n  src.inputargs ids=[{:?}]\n  src.op_result ids=[{:?}]\n  \
             ex.entries ids=[{}]\n  tgt_state.entries ids=[{}]\n  \
             tgt_state.locals_w.len={} ex.locals_w.len={}",
            self.graph.name,
            bb_id,
            tgt,
            var.id(),
            src_inputargs,
            src_op_results,
            id_list(&ex.entries),
            id_list(&tgt_state.entries),
            tgt_state.locals_w.len(),
            ex.locals_w.len(),
        );
    }

    fn lower_block(&mut self, mir_bb: usize) -> Result<(), LowerError> {
        let bb: &BasicBlock = &self.body.body[mir_bb];
        self.local_var = self.block_entry_local_var[mir_bb].clone();
        self.positional_aggregate_locals =
            self.block_entry_positional_aggregate_locals[mir_bb].clone();

        // 1. Statements -> SpaceOperations on the corresponding block.
        for (s_idx, st) in bb.statements.iter().enumerate() {
            let kind = st.stmt_kind().map_err(LowerError::Schema)?;
            self.lower_statement(mir_bb, s_idx, kind)?;
        }

        // 2. Terminator -> block exits (close the block).
        let term = bb.term().map_err(LowerError::Schema)?;
        self.lower_terminator(mir_bb, term)
    }

    // -----------------------------------------------------------------------
    // Statements
    // -----------------------------------------------------------------------

    fn lower_statement(
        &mut self,
        mir_bb: usize,
        s_idx: usize,
        kind: StmtKind,
    ) -> Result<(), LowerError> {
        match kind {
            // `StorageLive` / `StorageDead` carry no IR — RPython has
            // no lifetime markers and the JIT does not benefit from
            // them.
            StmtKind::StorageLive(_) | StmtKind::StorageDead(_) => Ok(()),

            // `let _ = place` — read for side-effect tracking only.
            // The JIT does not need to materialize anything.
            StmtKind::PlaceMention(_) => Ok(()),

            // Statement-level Rust overflow / bounds assertion. Charon
            // emits every assert as a *terminator* (`TermKind::Assert`,
            // handled in `lower_terminator`), so this arm is not reached
            // by the current corpus; it is kept as defensive handling
            // for the paired `Assign(AddChecked) + Assert(!overflow)`
            // shape. Stripping is correct either way — same rationale as
            // the terminator arm: a Rust-debug check with no
            // Python-observable meaning.
            StmtKind::Assert(_) => Ok(()),

            StmtKind::Assign(place, rvalue) => self.lower_assign(mir_bb, place, rvalue),

            StmtKind::Unknown => Err(LowerError::Unsupported(format!(
                "bb{mir_bb} stmt#{s_idx}: unknown StmtKind"
            ))),
        }
    }

    fn lower_assign(
        &mut self,
        mir_bb: usize,
        dest: Place,
        rvalue: Rvalue,
    ) -> Result<(), LowerError> {
        // The destination place's post-projection type is the rvalue's
        // result type (for both a `Local` slot and a `place.field`
        // write).  `build_rvalue` reads it to pick a cast's result bank.
        let dest_ty = clone_tyref(&dest.ty);
        match dest.kind {
            PlaceKind::Local(i) => {
                // Capture the construction `owner_root` if this binding
                // is a positional aggregate, before `build_rvalue`
                // consumes the rvalue, so `.N` reads of the local can
                // later emit a symmetric `FieldRead __pos_<N>` carrying
                // the same owner (see `resolve_place`).
                let positional_owner = self.positional_aggregate_owner(&rvalue);
                let (op, result_var) = self.build_rvalue(mir_bb, rvalue, &dest_ty)?;
                // The destination local takes on the freshly-minted
                // result Variable. Subsequent reads of the local
                // resolve to this Variable until the next Assign
                // overwrites the slot.
                self.local_var[i as usize] = Some(result_var.clone());
                // Keep the aggregate-local map in sync with the
                // last-write-wins slot: a non-aggregate rebind clears
                // the marker so the slot's reads collapse again.
                match positional_owner {
                    Some(owner) => {
                        self.positional_aggregate_locals.insert(i as usize, owner);
                    }
                    None => {
                        self.positional_aggregate_locals.remove(&(i as usize));
                    }
                }
                if let Some(op) = op {
                    let bb_id = self.block_id[mir_bb];
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(result_var),
                        kind: op,
                    });
                }
                Ok(())
            }
            PlaceKind::Projection(inner, elem) => {
                // `place.field = val` / `*p = val` / `p[i] = val`.
                // Compute the rvalue, then emit a write op keyed by the
                // projection element. The destination local is NOT
                // updated — the write goes through indirection, the
                // base local remains the same Variable.
                let value = match self.field_write_inline_const(&inner, &elem, &rvalue, &dest_ty) {
                    // An int / bool / float field write keeps its constant
                    // operand inline (`LinkArg::Const`) instead of
                    // materialising a `ConstInt` / `ConstFloat` + copy into
                    // a register: the assembler then takes the short `c`
                    // byte (`setfield_gc_i/rcd`) or a pool slot
                    // (`/rid` for a wide int, `/rfd` for a float).
                    // Mirrors RPython keeping the `Constant` box as a
                    // `setfield_gc` argument and deferring the
                    // short-vs-pool choice to `assembler.py:99-107`.
                    // `Constant::new` defers `concretetype` to the rtyper
                    // like every front-end synthesized constant; the
                    // assembler recovers the value kind from the
                    // self-describing Int / Bool / Float `ConstValue`
                    // variant via `constant_kind` (`getkind` fallback), so
                    // the `setfield_gc_<kind>` opname is keyed correctly.
                    Some(const_value) => {
                        LinkArg::Const(crate::flowspace::model::Constant::new(const_value))
                    }
                    None => {
                        let (op, value_var) = self.build_rvalue(mir_bb, rvalue, &dest_ty)?;
                        // If `build_rvalue` produced an op, emit it first
                        // so `value_var` is bound before the write reads
                        // it.
                        if let Some(op) = op {
                            let bb_id = self.block_id[mir_bb];
                            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                                result: Some(value_var.clone()),
                                kind: op,
                            });
                        }
                        LinkArg::Value(value_var)
                    }
                };
                self.emit_projection_write(mir_bb, *inner, elem, value, &dest_ty)
            }
            _ => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: Assign to {:?} destination not yet supported",
                place_kind_label(&dest.kind)
            ))),
        }
    }

    /// Re-resolve a recorded [`IndexElemAlias`] operand to the Variable
    /// live in the current block: prefer the current `local_var` binding
    /// of its source MIR local (rebound to a fresh inputarg if the
    /// operand was threaded across the index call's block split), and
    /// fall back to the Variable captured at the index call for a
    /// constant operand with no backing local.
    fn realias_operand(&self, local: Option<usize>, recorded: Variable) -> Variable {
        local
            .and_then(|l| self.local_var.get(l).cloned().flatten())
            .unwrap_or(recorded)
    }

    /// If `elem` writes a struct field and `rvalue` is a plain pooled
    /// constant matching the field's value bank — integer / bool for an
    /// int-kind field, float for a float-kind field — return the matching
    /// [`ConstValue`] so the `FieldWrite` carries it inline
    /// (`setfield_gc_i/rcd` short form or `/rid` pool slot for an int,
    /// `setfield_gc_f/rfd` pool slot for a float) rather than forcing it
    /// into a register with a `ConstInt` / `ConstFloat` + copy.
    ///
    /// RPython keeps the `Constant` box as a `setfield_gc` argument
    /// (codewriter args are `AbstractValue`, never pre-materialised) and
    /// leaves the short-vs-pool encoding to the assembler
    /// (`assembler.py:99-107`).  The assembler `FieldWrite` arm derives
    /// the value kind from the constant via `constant_kind` and routes a
    /// float to a pooled `f` slot (`assembler.rs` `emit_const_f`), which
    /// the walker resolves through the constants window of `registers_f`.
    ///
    /// Ref-kind constants are not pooled at this layer: a string / char /
    /// fn-pointer constant lowers to a runtime `Call`
    /// (`build_rvalue`'s `DecodedConst::Str` / `FnPath` arms), not a
    /// poolable `ConstPtr`, so it keeps the materialised path.  A plain
    /// `Deref` (`*p = v`) also keeps it, lowering to a `__deref_write`
    /// Call whose args must be Variables; an `index_mut`-aliased `Deref`
    /// and a `[i]` subscript both lower to `setarrayitem_gc_*`, whose
    /// value operand can stay an inline `Constant` like `setfield_gc_*`
    /// (jtransform.py:803 passes `op.args[2]` verbatim).
    fn field_write_inline_const(
        &self,
        inner: &Place,
        elem: &ProjectionElem,
        rvalue: &Rvalue,
        dest_ty: &TyRef,
    ) -> Option<ConstValue> {
        // A `Field`, a `[i]` subscript, or an `index_mut`-aliased `Deref`
        // lowers to a `setfield_gc_*` / `setarrayitem_gc_*` whose value
        // operand can stay an inline `Constant`.  A plain `Deref` lowers
        // to a `__deref_write` Call (Variable args only) and is excluded.
        let inlinable_target = match elem {
            ProjectionElem::Tagged(v) => v
                .as_object()
                .is_some_and(|m| m.contains_key("Field") || m.contains_key("Index")),
            ProjectionElem::Atom(s) if s == "Deref" => matches!(
                &inner.kind,
                PlaceKind::Local(i) if self.index_elem_alias.contains_key(&(*i as usize))
            ),
            _ => false,
        };
        if !inlinable_target {
            return None;
        }
        // Only a bare constant operand qualifies; a computed rvalue still
        // flows through `build_rvalue`.
        let Rvalue::Use(Operand::Const(value)) = rvalue else {
            return None;
        };
        // Inline only a genuinely pooled constant matched to the target's
        // value bank: int / bool into an int-kind target (`setfield_gc_i`
        // / `setarrayitem_gc_i`; `getkind(Bool) == 'int'`), float into a
        // float-kind target (`setfield_gc_f`).  A `Bool`-typed target
        // (e.g. a `bool` array element) banks `i` just like an `Int`
        // field.  `Str` / `FnPath` constants lower to a `Call`, not a
        // poolable `ConstPtr`, so they fall through to the materialised
        // path.
        match (
            tyref_to_value_type(dest_ty, self.llbc),
            decode_constant(self.llbc, value).ok()?,
        ) {
            (ValueType::Int, DecodedConst::Int(n)) => Some(ConstValue::Int(n)),
            (ValueType::Int | ValueType::Bool, DecodedConst::Bool(b)) => Some(ConstValue::Bool(b)),
            (ValueType::Float, DecodedConst::Float(bits)) => Some(ConstValue::Float(bits)),
            _ => None,
        }
    }

    /// Emit the side-effectful write op for an `Assign` whose dest is
    /// a `Projection(inner, elem)`. `value` is the freshly computed
    /// rvalue.  `dest_ty` is the projected place's own `TyRef` — the
    /// field type AFTER generic substitution, mirroring the typed
    /// `FieldRead` arm in `resolve_place` (the declaration-side field
    /// ty is the generic param for generic ADTs, which
    /// `tyref_to_value_type` can only degrade to `Ref(None)`).
    fn emit_projection_write(
        &mut self,
        mir_bb: usize,
        inner: Place,
        elem: ProjectionElem,
        value: LinkArg,
        dest_ty: &TyRef,
    ) -> Result<(), LowerError> {
        // A plain `*p = v` (`__deref_write` Call below) reads a
        // materialised register: its Call args must be Variables, so a
        // constant operand must already have been forced to a Variable.
        // `field_write_inline_const` only mints a `LinkArg::Const` for the
        // setfield / setarrayitem targets (Field, `[i]`, aliased Deref),
        // never the plain-Deref Call, so this expect never trips.
        let value_var = |v: &LinkArg| {
            v.as_variable()
                .expect("__deref_write carries a materialised Variable value")
                .clone()
        };
        let inner_local = match &inner.kind {
            PlaceKind::Local(i) => Some(*i as usize),
            _ => None,
        };
        let base = self.resolve_place(mir_bb, inner)?;
        let bb_id = self.block_id[mir_bb];
        let op = match &elem {
            ProjectionElem::Atom(s) if s == "Deref" => {
                if let Some(alias) = inner_local
                    .and_then(|i| self.index_elem_alias.get(&i))
                    .cloned()
                {
                    // `*p = val` where `p` was bound by a
                    // devirtualized workspace `index_mut` call is the
                    // write half of `arr[i] = val` — emit the array
                    // write directly (setarrayitem).  The index call
                    // terminates its own block, so the base/index
                    // operands are typically rebound to fresh inputarg
                    // Variables here; re-resolve them through
                    // `local_var` by their source local and fall back
                    // to the recorded Variable for a constant operand.
                    let arr = self.realias_operand(alias.base_local, alias.base_var);
                    let idx = self.realias_operand(alias.index_local, alias.index_var);
                    OpKind::ArrayWrite {
                        base: arr,
                        index: idx,
                        value: value.clone(),
                        item_ty: tyref_to_value_type(dest_ty, self.llbc),
                        array_type_id: None,
                        nolength: false,
                    }
                } else {
                    // `*p = val` — no IR-level FieldWrite/ArrayWrite
                    // fits.  Emit a synthetic 2-arg Call so the write
                    // remains visible to the downstream side-effect
                    // tracking.
                    //
                    // The write produces no value (`result` below is `None`),
                    // so the declared result kind must be Void: jtransform's
                    // `resolve_call_result` reads `result_ty` when the op has
                    // no result Variable, and a non-void kind there assembles
                    // a `residual_call_r_<kind>` key with no `>` result tail
                    // — a malformed opname nothing wires (`getkind(Void)`
                    // keeps result-less calls on the `residual_call_*_v`
                    // row).
                    OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: vec!["__deref_write".to_string()],
                        },
                        args: vec![base, value_var(&value)],
                        result_ty: ValueType::Void,
                    }
                }
            }
            ProjectionElem::Tagged(v) => {
                if let Some(field_payload) = v.as_object().and_then(|m| m.get("Field")) {
                    // Resolve the field through its TypeDecl exactly
                    // like the read side (`resolve_place` Field arm):
                    // the descriptor must carry the same
                    // (`field_name`, `owner_root`) shape or no
                    // downstream owner-keyed consumer can match it —
                    // jtransform's vable matcher rejects a `None`
                    // owner against an owner-rooted config, so e.g.
                    // `PyFrame.valuestackdepth` writes stayed plain
                    // `setfield` while the paired reads rewrote to
                    // `getfield_vable`.  The Adt case derives the
                    // written value's kind from `dest_ty` (a Ref field
                    // must not be stamped `Int`); the bare-label
                    // fallback keeps non-Adt containers lowering as
                    // before.
                    let (field, ty) = match self.resolve_adt_field(field_payload) {
                        Some((owner_root, field_name, _field_ty, owner_id)) => (
                            FieldDescriptor::new(field_name, Some(owner_root))
                                .with_owner_id(owner_id),
                            tyref_to_value_type(dest_ty, self.llbc),
                        ),
                        None => (
                            FieldDescriptor::new(field_label_from_payload(field_payload), None),
                            ValueType::Int,
                        ),
                    };
                    OpKind::FieldWrite {
                        base,
                        field,
                        value,
                        ty,
                    }
                } else if let Some(index_payload) = v.as_object().and_then(|m| m.get("Index")) {
                    let idx_var = self.index_offset_var(mir_bb, index_payload)?;
                    OpKind::ArrayWrite {
                        base,
                        index: idx_var,
                        value: value.clone(),
                        item_ty: ValueType::Int,
                        array_type_id: None,
                        nolength: false,
                    }
                } else {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: ProjectionElem::Tagged write not handled: {v}"
                    )));
                }
            }
            ProjectionElem::Atom(s) => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: ProjectionElem::Atom({s}) write not handled"
                )));
            }
        };
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: None,
            kind: op,
        });
        Ok(())
    }

    /// Extract the `offset` operand from an `Index { offset, from_end }`
    /// projection element and resolve it to a Variable. `from_end` is
    /// ignored: backwards-from-end indexing only appears in slice patterns
    /// at the moment, and the lowering uses the offset Variable directly.
    fn index_offset_var(
        &mut self,
        mir_bb: usize,
        index_payload: &serde_json::Value,
    ) -> Result<Variable, LowerError> {
        let offset = index_payload
            .as_object()
            .and_then(|m| m.get("offset"))
            .ok_or_else(|| {
                LowerError::Schema(format!(
                    "bb{mir_bb}: Index projection missing offset: {index_payload}"
                ))
            })?
            .clone();
        let op: Operand = serde_json::from_value(offset)
            .map_err(|e| LowerError::Schema(format!("bb{mir_bb}: Index offset decode: {e}")))?;
        self.resolve_operand(mir_bb, op)
    }

    /// Build the IR for an Rvalue. Returns `(op, result_var)` — `op` is
    /// the `OpKind` to push onto the current block, `result_var` is the
    /// Variable the destination local should be bound to. `op` is
    /// `None` for trivial copies (no op pushed, the existing Variable
    /// is reused).
    fn build_rvalue(
        &mut self,
        mir_bb: usize,
        rvalue: Rvalue,
        dest_ty: &TyRef,
    ) -> Result<(Option<OpKind>, Variable), LowerError> {
        match rvalue {
            Rvalue::Use(operand) => {
                let v = self.resolve_operand(mir_bb, operand)?;
                // Plain use — reuse the operand's Variable without
                // emitting a copy op. RPython does the same: a flow
                // graph never has a redundant `same_as` between two
                // Variables that already alias.
                Ok((None, v))
            }
            Rvalue::BinaryOp(op_json, lhs, rhs) => {
                let lhs_v = self.resolve_operand(mir_bb, lhs)?;
                let rhs_v = self.resolve_operand(mir_bb, rhs)?;
                let op_label = binop_label(&op_json)?;
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                // f64 arithmetic stays in the Float bank (`float_add`/
                // `float_mod`...); everything else — comparisons (bool),
                // integer math, and checked-arithmetic `(int, bool)`
                // tuple destinations — is Int.
                let result_ty = match tyref_to_value_type(dest_ty, self.llbc) {
                    ValueType::Float => ValueType::Float,
                    _ => ValueType::Int,
                };
                Ok((
                    Some(OpKind::BinOp {
                        op: op_label,
                        lhs: lhs_v,
                        rhs: rhs_v,
                        result_ty,
                    }),
                    res,
                ))
            }
            // `UnaryOp(op, operand)` — `Neg`, `Not`, casts.  Arithmetic
            // `Neg` / `Not` lower to `OpKind::UnaryOp` with a canonical
            // snake_case label so the assembler reaches the wired
            // `int_neg` / `int_invert` handlers instead of inventing a
            // synthetic `int_unary.*` opname.  `Cast(...)` is handled
            // separately below: same-bank casts alias the operand, and a
            // bank-crossing cast lowers to a `simple_call` against the
            // matching host cast callable (see the in-arm comment).
            Rvalue::UnaryOp(op_json, operand) => {
                // A `Cast(..)` reinterprets the operand. When the operand
                // and the destination share a register bank (ptr→ptr,
                // int→int of any width, float→float, `Unsize`, `FnPtr` —
                // every cast that keeps the i64/f64 carrier in place) the
                // JIT models it as `same_as`, so alias the operand without
                // emitting an op.  A bank-CHANGING cast (`int↔ptr`,
                // `int↔float`) must move the value into the destination
                // bank.  The rtyper retired every typed cast opname from
                // the unary-op path (`normalize_unary_op_name` accepts only
                // `neg` / `bool` / `invert` / `same_as`), so a bank
                // crossing lowers to `simple_call(<host_callable>, v)` —
                // `lltype.cast_int_to_ptr` / `lltype.cast_ptr_to_int` for
                // `int↔ptr`, the bare `float` / `int` builtins for
                // `int↔float` — whose rtyper hooks emit the low-level
                // `cast_*` op.  Pure-aliasing those would leave e.g. an
                // `as_usize() as *mut T` value in the Int bank where a
                // later GcRef merge expects a Ref, tripping the assembler's
                // per-bank cross-check.  The bank decision reads the
                // operand's place type and the destination type directly,
                // so it is independent of which Charon `CastKind` tag
                // encodes the conversion.  Genuine `Neg` / `Not` arithmetic
                // keeps a real scalar `OpKind::UnaryOp`.
                if unary_op_is_cast(&op_json) {
                    // Signedness-aware source classification — `operand_value_kind`
                    // (`tyref_to_value_type`) collapses `uN` and `iN` both to
                    // `Int`, hiding a `usize as i64` flip; `tyref_to_attr_value_type`
                    // keeps the signed/unsigned split.
                    let src_attr = match &operand {
                        Operand::Copy(p) | Operand::Move(p) => {
                            Some(tyref_to_attr_value_type(&p.ty, self.llbc))
                        }
                        Operand::Const(_) => None,
                    };
                    let src_kind = self.operand_value_kind(&operand);
                    let arg = self.resolve_operand(mir_bb, operand)?;
                    let dst_kind = tyref_to_value_type(dest_ty, self.llbc);
                    // Signedness-flipping int cast (`w_tuple_len(obj) as i64`)
                    // — aliasing keeps the source `r_uint` annotation on the
                    // signed destination, tripping the SomeInteger signedness
                    // `UnionError` (`binaryop.py:178-202`) when the length
                    // meets a signed index.  Route the unsigned→signed flip
                    // through `rarithmetic.intmask` (the RPython spelling of
                    // this re-type): `rtype_intmask` coerces to `lltype.Signed`
                    // — identity on the i64 carrier — so the value is unchanged
                    // and the result re-types Signed.  Resolves via the Layer-3
                    // `HOST_ENV.import_module(rpython.rlib.rarithmetic)
                    // .module_get(intmask)` path (`flowspace_adapter.rs`).
                    if matches!(src_attr, Some(ValueType::Unsigned))
                        && matches!(tyref_to_attr_value_type(dest_ty, self.llbc), ValueType::Int)
                    {
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        return Ok((
                            Some(OpKind::Call {
                                target: CallTarget::FunctionPath {
                                    segments: ["rpython", "rlib", "rarithmetic", "intmask"]
                                        .into_iter()
                                        .map(str::to_string)
                                        .collect(),
                                },
                                args: vec![arg],
                                result_ty: ValueType::Int,
                            }),
                            res,
                        ));
                    }
                    return Ok(
                        match src_kind
                            .as_ref()
                            .and_then(|s| cast_call_segments(s, &dst_kind))
                        {
                            Some(segments) => {
                                let res = self
                                    .graph
                                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                                (
                                    Some(OpKind::Call {
                                        target: CallTarget::FunctionPath { segments },
                                        args: vec![arg],
                                        result_ty: dst_kind,
                                    }),
                                    res,
                                )
                            }
                            // Same bank (or a bank pair with no host cast
                            // callable): alias the operand — except a
                            // ptr→ptr cast to a registered struct root,
                            // which narrows to `SomeInstance(root)` so a
                            // field read on the pointee resolves (#298;
                            // see the `Rvalue::Cast` arm for the full
                            // rationale).  The SOURCE must already be a Ref
                            // for this to be a genuine `cast_pointer`
                            // (ptr→ptr); an int→ptr or unknown-source cast
                            // is `cast_int_to_ptr` territory and aliases
                            // instead of narrowing to an instance.
                            None => {
                                if matches!(src_kind, Some(ValueType::Ref(_)))
                                    && let ValueType::Ref(_) = dst_kind
                                    && let Some(root) = tyref_class_root(dest_ty, self.llbc)
                                {
                                    let res = self.graph.alloc_value_var_with_type(
                                        crate::model::ConcreteType::Unknown,
                                    );
                                    (
                                        Some(OpKind::Call {
                                            target: CallTarget::FunctionPath {
                                                segments: vec![
                                                    "__pyre_cast_instance".to_string(),
                                                    root.clone(),
                                                ],
                                            },
                                            args: vec![arg],
                                            result_ty: ValueType::Ref(Some(root)),
                                        }),
                                        res,
                                    )
                                } else {
                                    (None, arg)
                                }
                            }
                        },
                    );
                }
                // Rust's `!` is bitwise complement on integers (`invert`)
                // but LOGICAL negation on `bool`.  `BoolRepr` has no
                // logical invert — it inherits `IntegerRepr`'s bitwise
                // `~` — so `!bool` lowered to `invert` is an op the rtyper
                // rejects.  Lower `!bool` to `eq(b, False)`, the orthodox
                // logical negation, leaving integer `!` on the `invert`
                // path.
                let operand_is_bool =
                    matches!(self.operand_value_kind(&operand), Some(ValueType::Bool));
                let op_label = unary_op_label(&op_json)?;
                let arg = self.resolve_operand(mir_bb, operand)?;
                if op_label == "invert" && operand_is_bool {
                    let false_var = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let bb_id = self.block_id[mir_bb];
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(false_var.clone()),
                        kind: OpKind::ConstBool(false),
                    });
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    return Ok((
                        Some(OpKind::BinOp {
                            op: "eq".to_string(),
                            lhs: arg,
                            rhs: false_var,
                            result_ty: ValueType::Int,
                        }),
                        res,
                    ));
                }
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                // `Neg` on f64 stays in the Float bank (`float_neg`);
                // everything else is Int.  A hardcoded Int here mistyped
                // float negation and produced cross-bank link renamings
                // downstream.
                let result_ty = match tyref_to_value_type(dest_ty, self.llbc) {
                    ValueType::Float => ValueType::Float,
                    _ => ValueType::Int,
                };
                Ok((
                    Some(OpKind::UnaryOp {
                        op: op_label,
                        operand: arg,
                        result_ty,
                    }),
                    res,
                ))
            }
            // `Ref { place, ... }` — references in MIR are pointer-typed
            // aliases of the referent. The JIT does not model lifetimes,
            // and downstream consumers (codewriter, regalloc) operate on
            // the value flowing through the reference, not the reference
            // itself. Aliasing the dest local to the referent Variable
            // keeps the IR small, treating `&x` as a same-Variable copy.
            Rvalue::Ref { place, .. } => {
                let v = self.resolve_place(mir_bb, place)?;
                Ok((None, v))
            }
            // `RawPtr { place, ... }` — `&raw const x` / `&raw mut x`.
            // Same aliasing model as `Ref`: the JIT treats raw pointers
            // and references identically at the IR level (lifetime
            // tracking lives outside the JIT).
            Rvalue::RawPtr { place, .. } => {
                let v = self.resolve_place(mir_bb, place)?;
                Ok((None, v))
            }
            // `Repeat(elem, ty, count)` — `[v; N]` literal. Modeled as
            // a synthetic Call so the IR shape stays uniform; downstream
            // consumers see a 1-arg array construction call.
            Rvalue::Repeat(elem, _ty, _count) => {
                let arg = self.resolve_operand(mir_bb, elem)?;
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                Ok((
                    Some(OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: vec!["__array_repeat".to_string()],
                        },
                        args: vec![arg],
                        result_ty: ValueType::Int,
                    }),
                    res,
                ))
            }
            // `ShallowInitBox(elem, ty)` — `Box::new` half-construction
            // marker. The MIR emits this followed by an `Assign(*box,
            // value)` that fills the box contents. Modeled as a
            // synthetic 1-arg constructor call carrying the element.
            Rvalue::ShallowInitBox(elem, _ty) => {
                let arg = self.resolve_operand(mir_bb, elem)?;
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                Ok((
                    Some(OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor("Box"),
                        args: vec![arg],
                        result_ty: ValueType::Int,
                    }),
                    res,
                ))
            }
            // `Cast(kind, operand, target_ty)` — numeric/pointer
            // coercion. The JIT does not track narrow integer widths,
            // so reuse the alias path: the cast result Variable is the
            // same as the operand Variable. `as` casts that do not
            // change the JIT-visible kind collapse this way.
            Rvalue::Cast(kind, operand, ty) => {
                // Classify the source BEFORE `resolve_operand` consumes
                // `operand` (mirrors the `UnaryOp::Cast` twin above): only a
                // Ref source may narrow to an instance downcast below.
                let src_kind = self.operand_value_kind(&operand);
                let v = self.resolve_operand(mir_bb, operand)?;
                // #298: a same-bank ptr→ptr cast to a registered struct
                // root (`obj as *const PyCode`) keeps the i64
                // pointer carrier in place, so it would alias like above
                // — but the result is then read like an instance of that
                // struct (`(*p).code_ptr`), and aliasing leaves the
                // pointer classdef-less so the field read blocks at the
                // annotator getattr arm.  `tyref_class_root` returns
                // `Some` only for a named-ADT pointee (None for
                // primitives / builtin containers / generics / multi-impl
                // type-vars), so emit a `__pyre_cast_instance` narrow
                // whose annotator types the result `SomeInstance(root)`
                // and whose typer folds to a `cast_pointer`.  Gate on the
                // raw-pointer cast kind: `lltype.cast_pointer`
                // (`lltype.py:964-975`) is pointer-to-pointer only, and
                // int-to-pointer is the separate `cast_int_to_ptr`
                // analyzer, so an `addr_usize as *const Struct` reinterpret
                // must NOT be narrowed to an instance downcast — the
                // `src_kind` Ref guard enforces exactly that.
                if cast_kind_is_raw_ptr(&kind)
                    && matches!(src_kind, Some(ValueType::Ref(_)))
                    && let ValueType::Ref(_) = tyref_to_value_type(&ty, self.llbc)
                    && let Some(root) = tyref_class_root(&ty, self.llbc)
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    return Ok((
                        Some(OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec!["__pyre_cast_instance".to_string(), root.clone()],
                            },
                            args: vec![v],
                            result_ty: ValueType::Ref(Some(root)),
                        }),
                        res,
                    ));
                }
                Ok((None, v))
            }
            // `Len(place)` — slice / array length. Synthetic 1-arg
            // call; needs no descriptor for now.
            Rvalue::Len(place) => {
                let base = self.resolve_place(mir_bb, place)?;
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                Ok((
                    Some(OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: vec!["__len".to_string()],
                        },
                        args: vec![base],
                        result_ty: ValueType::Int,
                    }),
                    res,
                ))
            }
            // `NullaryOp(op, ty)` — `SizeOf(T)`, `AlignOf(T)`, etc.
            // 0-arg synthetic Call carrying the op name.
            Rvalue::NullaryOp(op_json, _ty) => {
                let op_name = if let Some(s) = op_json.as_str() {
                    s.to_string()
                } else if let Some(obj) = op_json.as_object() {
                    obj.keys()
                        .next()
                        .cloned()
                        .unwrap_or_else(|| "nullary".into())
                } else {
                    "nullary".into()
                };
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                Ok((
                    Some(OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: vec![format!("__nullary_{op_name}")],
                        },
                        args: vec![],
                        result_ty: ValueType::Int,
                    }),
                    res,
                ))
            }
            // `Aggregate(kind, operands)` — tuple / struct / enum-variant
            // / array construction. Modeled as a synthetic constructor
            // call (`CallTarget::SyntheticTransparentCtor`), the
            // CallTarget variant explicitly carved out for "constructors
            // RPython's rtyper erases before jtransform" — the MIR
            // driver fits that description (Charon has already resolved
            // types, so the call is post-frontend-resolution by
            // construction).  Operands flow as call arguments; the
            // synthetic name is best-effort from the AggregateKind tag.
            Rvalue::Aggregate(kind, operands) => {
                // Resolve operand Variables up front; they flow into the
                // synthesised FieldWrite chain rather than the ctor's
                // arg list.
                let mut arg_vars: Vec<Variable> = Vec::with_capacity(operands.len());
                for op in operands {
                    arg_vars.push(self.resolve_operand(mir_bb, op)?);
                }
                // Resolve the user-defined owner + field names from the
                // Adt kind's `type_id` when possible.  Charon encodes
                // `AggregateKind::Adt(type_id, variant_idx, ..)` as
                // `{"Adt": [type_id, variant_idx, ..]}`; struct variants
                // use `variant_idx = null`, enum variants index into the
                // `TypeDeclKind::Enum` variant list.
                let resolved = self.resolve_aggregate_adt(&kind);
                let (owner_path, ctor_name, field_names) = match resolved {
                    Some((owner_path, ctor_name, field_names)) => {
                        (owner_path, ctor_name, field_names)
                    }
                    None => {
                        let leaf = aggregate_ctor_name(&kind);
                        // Synthetic placeholders for non-Adt aggregates
                        // (`Tuple`, `Array`, `Closure`) — they have no
                        // user-defined class to resolve into.
                        let positional =
                            (0..arg_vars.len()).map(|i| format!("__pos_{i}")).collect();
                        (Vec::new(), leaf, positional)
                    }
                };
                let result_ty_owner = if owner_path.is_empty() {
                    ctor_name.clone()
                } else {
                    format!("{}::{}", owner_path.join("::"), ctor_name)
                };
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                // Emit the transparent ctor with empty args so the
                // annotator's `ClassDesc::pycall` `args.fixedunpack(0)`
                // check (`classdesc.rs:1247`, mirroring upstream
                // `classdesc.py:705`) succeeds for classes whose
                // `__init__` is not registered with the bookkeeper —
                // the operand values flow through the FieldWrite chain
                // below instead.  `SyntheticTransparentCtor` survives
                // as the marker that downstream jtransform unwraps to
                // the underlying `SomeInstance(classdef)`.
                let ctor_target = if owner_path.is_empty() {
                    CallTarget::synthetic_transparent_ctor(ctor_name.clone())
                } else {
                    CallTarget::synthetic_transparent_ctor_with_owner(
                        owner_path.clone(),
                        ctor_name.clone(),
                    )
                };
                let bb_id = self.block_id[mir_bb];
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(res.clone()),
                    kind: OpKind::Call {
                        target: ctor_target,
                        args: Vec::new(),
                        result_ty: ValueType::Ref(Some(result_ty_owner.clone())),
                    },
                });
                // Surface every operand through a separate FieldWrite so
                // the field-to-value binding survives into the
                // codewriter / annotator.  Field names default to
                // `__pos_<i>` when the resolver could not project a real
                // schema entry (tuple aggregates, deduplicated types
                // not in the LLBC's local table).
                for (i, value) in arg_vars.into_iter().enumerate() {
                    let name = field_names
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("__pos_{i}"));
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: None,
                        kind: OpKind::FieldWrite {
                            base: res.clone(),
                            field: crate::model::FieldDescriptor {
                                name,
                                owner_root: Some(result_ty_owner.clone()),
                                owner_id: None,
                            },
                            value: crate::model::LinkArg::Value(value),
                            ty: ValueType::Ref(None),
                        },
                    });
                }
                Ok((None, res))
            }
            // `Discriminant(place)` — read the integer tag of an enum
            // value. Modeled as a synthetic `FieldRead` of an
            // `__discriminant` field: tag access is morally a pure
            // field read at the bit level, and reusing the existing
            // `FieldRead` shape keeps the IR closed under the opkind
            // catalogue (per `front/mod.rs` rule — no new OpKinds in
            // this layer).  The read is keyed to the enum base identity
            // (`owner_root = module::Enum`, `owner_id = StructId`) so it
            // resolves at the tag's real byte position — the enum base is
            // registered in `exact_layouts` with `__discriminant` at
            // `discriminant_offset()`.  A tag at offset 0 (the common
            // case) registers 0, identical to the prior heuristic.  An
            // unresolvable place type falls back to the unowned read.
            Rvalue::Discriminant(place) => {
                let (owner_root, owner_id) = match self.tyref_adt_name_path(&place.ty) {
                    Some(name_path) => {
                        let canon = strip_crate_prefix(&name_path);
                        let sid = majit_ir::descr::StructId::from_canonical(&canon);
                        (Some(canon), Some(sid))
                    }
                    None => (None, None),
                };
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                // A local bound directly to its payload by a decomposed
                // always-`Ok` conversion (`try_lower_usize_try_from`)
                // carries no runtime enum object — no `__discriminant`
                // field exists to read.  Its tag is the recorded
                // constant, so fold to `ConstInt(tag)` (mirrors
                // `expect_on_const_ok`, which aliases the payload).
                if let PlaceKind::Local(i) = &place.kind
                    && let Some(&tag) = self.const_discriminant_locals.get(&(*i as usize))
                {
                    return Ok((Some(OpKind::ConstInt(tag)), res));
                }
                // `Discriminant` of an INLINE enum field (`self.strategy`):
                // the field is stored by value, so the container holds the
                // enum's bytes directly — there is no pointer to follow.
                // Reading `&self.strategy` as a `getfield_gc_r` would load
                // the field's first word as a bogus pointer (the deref then
                // faults on the tag read).  When the tag sits at the field's
                // base (the C-like repr, `inline_enum_field_disc_offset ==
                // Some(0)`), the field's own offset already addresses the
                // tag, so read the tag in ONE getfield from the container at
                // the field offset — matching the runtime `strategy` field
                // descr (offset 32, size 1, Int) the production fold uses.
                // A non-zero tag offset falls through to the generic read.
                let inline_enum_field = if let PlaceKind::Projection(_, ProjectionElem::Tagged(v)) =
                    &place.kind
                    && let Some(field_payload) = v.as_object().and_then(|m| m.get("Field"))
                    && let Some((f_owner_root, f_name, f_ty, f_owner_id)) =
                        self.resolve_adt_field(field_payload)
                    && self.inline_fieldless_enum_field_tag0(&f_ty)
                {
                    Some((f_owner_root, f_name, f_owner_id))
                } else {
                    None
                };
                if let Some((f_owner_root, f_name, f_owner_id)) = inline_enum_field {
                    let PlaceKind::Projection(inner, _) = place.kind else {
                        unreachable!("inline_enum_field implies a Projection place")
                    };
                    let base = self.resolve_place(mir_bb, *inner)?;
                    return Ok((
                        Some(OpKind::FieldRead {
                            base,
                            field: FieldDescriptor::new(f_name, Some(f_owner_root))
                                .with_owner_id(f_owner_id),
                            ty: ValueType::Int,
                            pure: true,
                        }),
                        res,
                    ));
                }
                let base = self.resolve_place(mir_bb, place)?;
                Ok((
                    Some(OpKind::FieldRead {
                        base,
                        field: crate::model::FieldDescriptor {
                            name: "__discriminant".to_string(),
                            owner_root,
                            owner_id,
                        },
                        ty: ValueType::Int,
                        pure: true,
                    }),
                    res,
                ))
            }
            other => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: Rvalue::{} not yet supported",
                rvalue_variant_name(&other)
            ))),
        }
    }

    /// Resolve an [`Operand`] to the Variable the IR should reference.
    fn resolve_operand(&mut self, mir_bb: usize, op: Operand) -> Result<Variable, LowerError> {
        match op {
            Operand::Copy(place) | Operand::Move(place) => self.resolve_place(mir_bb, place),
            Operand::Const(value) => self.emit_constant(mir_bb, &value),
        }
    }

    /// The [`ValueType`] of a `Copy` / `Move` operand, read from the
    /// operand place's post-projection type.  Used by the `Cast`
    /// lowering to decide whether the cast crosses a register bank.
    /// Returns `None` for `Const` operands (a constant carries no place
    /// type here; a const-source cast aliases its operand).
    fn operand_value_kind(&self, op: &Operand) -> Option<ValueType> {
        match op {
            Operand::Copy(place) | Operand::Move(place) => {
                Some(tyref_to_value_type(&place.ty, self.llbc))
            }
            Operand::Const(_) => None,
        }
    }

    /// Decode a Charon `Operand::Const` value and emit the matching
    /// `OpKind::Const*` (or synthetic `Call` for non-primitive
    /// constants) operation on the current block, returning the fresh
    /// Variable that holds it.
    fn emit_constant(
        &mut self,
        mir_bb: usize,
        value: &serde_json::Value,
    ) -> Result<Variable, LowerError> {
        let op = match decode_constant(self.llbc, value)? {
            DecodedConst::Int(n) => OpKind::ConstInt(n),
            DecodedConst::Bool(b) => OpKind::ConstBool(b),
            DecodedConst::Float(bits) => OpKind::ConstFloat(bits),
            // String / char / byte-string constants — no
            // ConstStr opkind exists; synthesise a 0-arg `Call` whose
            // path encodes the literal text so the IR stays stable.
            DecodedConst::Str(s) => OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__str_const".to_string(), s],
                },
                args: vec![],
                // A `&str` / `&[u8]` literal lowers to `Ptr(STR)` (getkind
                // `r`), so the synthetic call's result kind is a Ref, not an
                // Int.  The `__str_const` path is never registered: on the
                // trace pipeline the call residualises, and the flowspace
                // adapter pre-folds it to the upstream `Constant('text')`
                // shape (`flowspace_adapter.rs::is_str_const_define`).
                result_ty: ValueType::Ref(None),
            },
            DecodedConst::FnPath(segments) => OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                args: vec![],
                result_ty: ValueType::Int,
            },
        };
        let var = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let bb_id = self.block_id[mir_bb];
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(var.clone()),
            kind: op,
        });
        Ok(var)
    }

    fn resolve_place(&mut self, mir_bb: usize, place: Place) -> Result<Variable, LowerError> {
        let place_ty = clone_tyref(&place.ty);
        match place.kind {
            PlaceKind::Local(i) => self.local_var[i as usize].clone().ok_or_else(|| {
                LowerError::Unsupported(format!(
                    "bb{mir_bb}: read of MIR local {i} before any Assign — \
                     uninitialised local, not yet supported"
                ))
            }),
            PlaceKind::Projection(inner, elem) => {
                // Adt-container `Field` projections emit a typed
                // `OpKind::FieldRead` carrying the field name and
                // `owner_root` so downstream consumers (codewriter
                // inlining + annotator GetAttr dispatch on
                // cross-procedural callers) get a resolvable
                // field/owner_root shape.
                //
                // Tuple-container `Field` projections split three ways.
                // A local bound by a positional `Rvalue::Aggregate`
                // (`positional_aggregate_locals`) carries a synthetic
                // ctor base with a `__pos_<N>` `FieldWrite` chain, so
                // its `.N` reads emit a symmetric `FieldRead __pos_<N>`.
                // A genuine Ref tuple (`__pos_<N>` block below) likewise
                // emits a typed `FieldRead`.  The `straight_line_add` /
                // AddChecked `(value, bool)` shape is the exception: it
                // lowers to a scalar `BinOp` (not an Aggregate), so its
                // `.0` collapses to the base Variable while the paired
                // `.1` Assert is dropped in `lower_statement` (a live
                // `.1` read fails loud — the overflow bit is unmodeled).
                //
                // Atom projections (`Deref` and others) still
                // collapse: `Deref` is a no-op for typed refs at the
                // JIT IR level, and any other Atom variant has no
                // typed analogue today.
                if let ProjectionElem::Tagged(v) = &elem
                    && let Some(field_payload) = v.as_object().and_then(|m| m.get("Field"))
                    && let Some((owner_root, field_name, field_ty, owner_id)) =
                        self.resolve_adt_field(field_payload)
                {
                    let base = self.resolve_place(mir_bb, *inner)?;
                    let bb_id = self.block_id[mir_bb];
                    // The field's DECLARED ty is the polymorphic decl's
                    // (monomorphize=false): for a generic container
                    // (`ControlFlow<B, C>`, `Result<T, E>`) it is a bare
                    // type variable, which projects to the `Ref`
                    // fallback even when the instantiated payload is an
                    // `i64`.  The place's post-projection type carries
                    // the substituted use-site type, so prefer it; keep
                    // the decl ty for the rare place shapes whose
                    // post-projection type the reader cannot resolve.
                    let ty = match tyref_to_value_type(&place_ty, self.llbc) {
                        ValueType::Ref(None) => tyref_to_value_type(&field_ty, self.llbc),
                        resolved => resolved,
                    };
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::FieldRead {
                            base,
                            field: FieldDescriptor::new(field_name, Some(owner_root))
                                .with_owner_id(owner_id),
                            ty,
                            pure: false,
                        },
                    });
                    return Ok(res);
                }
                // `xs[i]` element read — the symmetric counterpart of
                // the `ArrayWrite` Index arm in
                // `emit_projection_write`.  Collapsing to the base
                // (the previous behaviour for every non-Field Tagged
                // projection) aliased the element to the sequence
                // itself, so a method call on the element resolved
                // against the list annotation.
                if let ProjectionElem::Tagged(v) = &elem
                    && let Some(index_payload) = v.as_object().and_then(|m| m.get("Index"))
                {
                    let idx_var = self.index_offset_var(mir_bb, index_payload)?;
                    let base = self.resolve_place(mir_bb, *inner)?;
                    let bb_id = self.block_id[mir_bb];
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::ArrayRead {
                            base,
                            index: idx_var,
                            item_ty: tyref_to_value_type(&place_ty, self.llbc),
                            array_type_id: None,
                            nolength: false,
                            pure: false,
                        },
                    });
                    return Ok(res);
                }
                // Positional aggregate `.N` read: the base local was
                // bound by a non-Adt `Rvalue::Aggregate`, so emit the
                // `FieldRead __pos_<N>` that pairs with the
                // construction-side `FieldWrite __pos_<N>` instead of
                // aliasing the base.
                if let ProjectionElem::Tagged(v) = &elem
                    && let PlaceKind::Local(i) = inner.kind
                    && let Some(owner_root) =
                        self.positional_aggregate_locals.get(&(i as usize)).cloned()
                    && let Some(field_payload) = v.as_object().and_then(|m| m.get("Field"))
                    && let Some(idx) = self.positional_field_index(field_payload)
                {
                    let base = self.local_var[i as usize].clone().ok_or_else(|| {
                        LowerError::Unsupported(format!(
                            "bb{mir_bb}: read of MIR local {i} before any Assign — \
                             uninitialised local, not yet supported"
                        ))
                    })?;
                    let bb_id = self.block_id[mir_bb];
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::FieldRead {
                            base,
                            field: FieldDescriptor::new(format!("__pos_{idx}"), Some(owner_root)),
                            ty: ValueType::Ref(None),
                            pure: false,
                        },
                    });
                    return Ok(res);
                }
                // A `Field` projection whose base is a genuine Ref tuple
                // (`inner`'s post-projection type is a non-unit
                // `(A, B, ...)`) extracts element N: emit a typed
                // `FieldRead __pos_<N>` carrying the element type, the
                // same shape the positional-aggregate read above
                // produces.  This covers tuples the lowering does not
                // build inline — function-return tuples, enum-variant
                // payloads read through an `Option`/`Result` downcast —
                // whose base is an opaque Ref rather than a
                // transparent-ctor aggregate, so the base flows through
                // `inner` as a Ref while element N may be an `Int`.
                // Without it, `tuple.1` aliases the whole tuple (Ref) and
                // a later merge with an `Int`-typed sibling value trips
                // the assembler's kind cross-check.
                //
                // The `*Checked (value, bool)` local is field-dependent.
                // It lowered to a scalar `BinOp` (`binop_result_locals`):
                // field `.0` is that scalar, so it collapses to the base
                // Variable below.  Field `.1` is the Rust overflow bit,
                // which the JIT IR does not model — the paired overflow
                // `Assert` is dropped in `lower_statement`, and ovfcheck
                // is carried by separate `int_*_ovf` + guard ops, never a
                // boolean tuple field.  A live read of `.1` therefore has
                // no lowering: fail loud rather than silently alias the
                // overflow bool to the arithmetic value.
                if let ProjectionElem::Tagged(v) = &elem
                    && self.place_is_tuple(&inner)
                    && let Some(field_payload) = v.as_object().and_then(|m| m.get("Field"))
                    && let Some(idx) = self.positional_field_index(field_payload)
                {
                    if self.place_is_binop_scalar(&inner) {
                        if idx != 0 {
                            return Err(LowerError::Unsupported(format!(
                                "bb{mir_bb}: live read of field .{idx} of a \
                                 checked-binop `(value, bool)` local — the \
                                 overflow bit is not modeled (ovfcheck uses \
                                 separate guard ops, not a tuple field)"
                            )));
                        }
                        // idx == 0: fall through to the base-collapse below.
                    } else {
                        let base = self.resolve_place(mir_bb, *inner)?;
                        let bb_id = self.block_id[mir_bb];
                        let ty = tyref_to_value_type(&place_ty, self.llbc);
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(res.clone()),
                            kind: OpKind::FieldRead {
                                base,
                                field: FieldDescriptor::new(
                                    format!("__pos_{idx}"),
                                    // Same spelling the construction-side
                                    // FieldWrite chain records for builtin
                                    // tuple aggregates (`aggregate_ctor_name`
                                    // id atom), so read and write attrs key
                                    // under one owner.
                                    Some("Tuple".to_string()),
                                ),
                                ty,
                                pure: false,
                            },
                        });
                        return Ok(res);
                    }
                }
                match elem {
                    ProjectionElem::Tagged(_) | ProjectionElem::Atom(_) => {
                        self.resolve_place(mir_bb, *inner)
                    }
                }
            }
            // `Global { id, .. }` — static/const item reference.
            // The production trace supplies host addresses for the
            // object-space singletons pyre reads from statics. Preserve
            // those as constants; a synthetic 0-arg call would invent a
            // callable that neither Rust nor RPython has.
            PlaceKind::Global { id, .. } => {
                // A `NamedConst` (Rust `const`, not `static`) has no
                // address — the value is inlined at every use site.
                // Charon still emits a `Global` read, so fold the
                // trivial literal initializer to a constant rather than
                // calling a non-existent accessor.  Statics keep the
                // address/`FunctionPath` path.
                if let Some(const_op) = self.fold_named_const_global(id) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let bb_id = self.block_id[mir_bb];
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: const_op,
                    });
                    return Ok(res);
                }
                let segments = self.global_segments(mir_bb, id)?;
                // A `PyType` singleton static (`&SLICE_TYPE`): narrow the
                // raw address through `__pyre_cast_instance["PyType"]` so
                // the read types `SomeInstance("PyType")`, matching the
                // `(*obj).ob_type` field-read.  The bare `ConstInt` address
                // would pair `IntegerRepr` against that field's
                // `InstanceRepr` and block `rtype_is_` on the
                // `ob_type == &TYPE` pointer-identity chain — the same
                // narrow `obj as *const RegisteredStruct` already uses
                // (#298).
                if let Some(addr) = self.pytype_static_addr(&segments) {
                    let bb_id = self.block_id[mir_bb];
                    let raw = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(raw.clone()),
                        kind: OpKind::ConstRefAddr(addr),
                    });
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec![
                                    "__pyre_cast_instance".to_string(),
                                    "PyType".to_string(),
                                ],
                            },
                            args: vec![raw],
                            result_ty: ValueType::Ref(Some("PyType".to_string())),
                        },
                    });
                    return Ok(res);
                }
                let op = self
                    .static_addr_op(&segments)
                    .or_else(|| self.static_int_value_op(&segments))
                    .or_else(|| self.const_eval_global(id))
                    .or_else(|| self.fold_size_const_global(id))
                    .or_else(|| primitive_float_const(&segments))
                    .unwrap_or_else(|| OpKind::Call {
                        target: CallTarget::FunctionPath { segments },
                        args: vec![],
                        result_ty: tyref_to_value_type(&place_ty, self.llbc),
                    });
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                let bb_id = self.block_id[mir_bb];
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(res.clone()),
                    kind: op,
                });
                Ok(res)
            }
            PlaceKind::Unknown => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: Place::Unknown"
            ))),
        }
    }

    /// Resolve a Charon `Field` projection payload to the
    /// `(owner_root_leaf, field_name, field_ty)` triple suitable for
    /// `OpKind::FieldRead` emission.
    ///
    /// Charon encodes a Field as `[{"Adt": [type_id, variant_idx]}, idx]`
    /// where `variant_idx` is `null` for structs and the variant
    /// position for enums.  Returns `None` when:
    ///
    /// - the container is not `Adt` (Tuple etc. — caller falls back
    ///   to the existing collapse-to-base behaviour);
    /// - the `type_id` is missing from the LLBC's type table
    ///   (forward-decl / opaque);
    /// - the resolved TypeDecl is not `Struct(_)` / `Enum(_)`;
    /// - the field index is out of range for the resolved variant.
    ///
    /// Resolve a Charon `AggregateKind::Adt` payload to the
    /// `(owner_path, ctor_leaf, field_names)` triple suitable for a
    /// transparent-ctor + FieldWrite chain emission.
    ///
    /// Charon encodes `Aggregate(AggregateKind::Adt(type_id,
    /// variant_idx, ..), operands)` as `{"Adt": [type_id, variant_idx,
    /// ..]}`.  Struct aggregates use `variant_idx = null` and pull
    /// field names straight from the `TypeDeclKind::Struct(fields)`
    /// list; enum aggregates use a non-null `variant_idx` to select
    /// the right `VariantDecl` and emit the qualified ctor leaf
    /// (`Variant`) under the enum's `owner_path` (everything up to but
    /// not including the leaf in the resolved `name_path()`).
    ///
    /// Returns `None` when the kind is not Adt or the LLBC has no
    /// `TypeDecl` for `type_id`; the caller then falls back to the
    /// generic-tag ctor name with positional `__pos_<i>` fields.
    /// The construction-side `owner_root` when `rvalue` is an
    /// [`Rvalue::Aggregate`] that the lowering models as a synthetic
    /// transparent-ctor + positional `__pos_<i>` `FieldWrite` chain —
    /// i.e. a non-Adt aggregate (tuple / array / closure) for which
    /// [`Self::resolve_aggregate_adt`] returns `None`.  The owner is
    /// exactly what the [`Rvalue::Aggregate`] arm uses as
    /// `result_ty_owner` (`aggregate_ctor_name`, since `owner_path` is
    /// empty for the unresolved branch), so storing it lets `.N` reads
    /// emit a `FieldRead __pos_<N>` with the matching `owner_root`.
    /// Returns `None` for Adt aggregates: their `.field` reads already
    /// take the typed [`Self::resolve_adt_field`] path and never reach
    /// the collapse fallback.
    fn positional_aggregate_owner(&self, rvalue: &Rvalue) -> Option<String> {
        match rvalue {
            Rvalue::Aggregate(kind, _) if self.resolve_aggregate_adt(kind).is_none() => {
                Some(aggregate_ctor_name(kind))
            }
            _ => None,
        }
    }

    /// Decode a non-Adt `Field` projection payload — Charon encodes it
    /// as `[{"Tuple"|"Array"|"Closure": ..}, idx]` — to its field
    /// index.  Returns `None` for Adt containers (handled by
    /// [`Self::resolve_adt_field`]) and malformed payloads, so the
    /// caller only emits a positional `FieldRead` for genuine
    /// tuple/array/closure reads.
    /// True when `place`'s post-projection type is a non-unit tuple
    /// `(A, B, ...)` — a genuine Ref tuple whose `.N` reads extract
    /// element N rather than aliasing the base.
    fn place_is_tuple(&self, place: &Place) -> bool {
        tyref_is_tuple(&place.ty, self.llbc)
    }

    /// True when `place` is a bare local bound by a scalar
    /// [`Rvalue::BinaryOp`] (a `*Checked (value, bool)` result lowered to
    /// a single scalar Variable): its `.0` read must collapse to that
    /// scalar, not extract a tuple element.  See
    /// [`Lowering::binop_result_locals`].
    fn place_is_binop_scalar(&self, place: &Place) -> bool {
        matches!(
            &place.kind,
            PlaceKind::Local(i) if self.binop_result_locals.contains(&(*i as usize))
        )
    }

    fn positional_field_index(&self, payload: &serde_json::Value) -> Option<usize> {
        let arr = payload.as_array()?;
        if arr.len() != 2 {
            return None;
        }
        let label = arr[0].as_object()?.keys().next()?;
        if label == "Adt" {
            return None;
        }
        Some(arr[1].as_u64()? as usize)
    }

    fn resolve_aggregate_adt(
        &self,
        kind: &serde_json::Value,
    ) -> Option<(Vec<String>, String, Vec<String>)> {
        let adt = kind.as_object()?.get("Adt")?.as_array()?;
        // `AggregateKind::Adt` head: either a bare `type_id` u64 or a
        // full `TypeDeclRef` object `{"generics": …, "id": {"Adt":
        // <type_id>} | "Tuple" | {"Builtin": …}}` (the object shape is
        // what Charon emits for generic-instantiated types such as
        // `Result<T, E>`).  The bare-u64 read alone made every generic
        // aggregate fall through to the `"Adt"` ctor-name fallback,
        // collapsing all such constructors onto one identity (variant +
        // owner lost).  A `"Tuple"` / builtin atom `id` has no
        // user-defined class and falls through to `None` (the
        // Tuple/Array placeholder).
        let head = adt.first()?;
        let type_id = match head.as_u64() {
            Some(id) => id,
            None => head.get("id")?.get("Adt")?.as_u64()?,
        };
        let variant_idx = adt.get(1).and_then(serde_json::Value::as_u64);
        let td = self.llbc.type_by_id(type_id)?;
        let name_path = td.item_meta.name_path();
        let mut segments: Vec<String> = name_path.split("::").map(str::to_string).collect();
        let type_leaf = segments.pop().unwrap_or_default();
        let owner_path = segments;
        match (&td.kind, variant_idx) {
            (TypeDeclKind::Struct(fields), None) | (TypeDeclKind::Struct(fields), Some(_)) => {
                let field_names: Vec<String> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| f.name.clone().unwrap_or_else(|| format!("__pos_{i}")))
                    .collect();
                Some((owner_path, type_leaf, field_names))
            }
            (TypeDeclKind::Enum(variants), Some(idx)) => {
                let v = variants.get(idx as usize)?;
                let mut variant_owner = owner_path;
                // A reference-payload workspace enum instantiation
                // constructs into its per-instantiation variant class
                // (`Result<Tuple>::Ok`), the same spelling the receiver
                // type projects and the discriminant narrowing mints, so
                // the constructor's `setattr` and the narrowing land on
                // one classdef per instantiation (no cross-instantiation
                // payload union).
                let leaf = match head
                    .as_object()
                    .and_then(|h| adt_head_instantiation_suffix(h, self.llbc))
                {
                    Some(suffix) => format!("{type_leaf}{suffix}"),
                    None => type_leaf,
                };
                variant_owner.push(leaf);
                let field_names: Vec<String> = v
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| f.name.clone().unwrap_or_else(|| format!("__pos_{i}")))
                    .collect();
                Some((variant_owner, v.name.clone(), field_names))
            }
            _ => None,
        }
    }

    /// For a struct the owner_root is the LLBC TypeDecl's leaf name
    /// (`PyFrame` from `pyre_interpreter::pyframe::PyFrame`) so the
    /// downstream `struct_fields` registry resolves with the same leaf
    /// key.  For an enum the read is a variant downcast, so the
    /// owner_root is the variant-qualified `{enum_leaf}::{variant}` key
    /// (the variant subclass carries its own fields at the exact
    /// per-variant offset).
    fn resolve_adt_field(
        &self,
        payload: &serde_json::Value,
    ) -> Option<(String, String, TyRef, Option<majit_ir::descr::StructId>)> {
        let arr = payload.as_array()?;
        if arr.len() != 2 {
            return None;
        }
        let container = arr[0].as_object()?;
        let adt = container.get("Adt")?.as_array()?;
        // The projection container head is either a bare `type_id` u64 or
        // the full `TypeDeclRef` object a generic-instantiated downcast
        // carries — the same two shapes `resolve_aggregate_adt` decodes.
        // The bare-u64 read alone returned `None` for every generic
        // variant field read, so the per-instantiation variant class the
        // constructor and receiver project had no matching field read.
        let head = adt.first()?;
        let type_id = match head.as_u64() {
            Some(id) => id,
            None => head.get("id")?.get("Adt")?.as_u64()?,
        };
        let variant_idx = adt.get(1).and_then(serde_json::Value::as_u64);
        let field_idx = arr[1].as_u64()? as usize;
        let td = self.llbc.type_by_id(type_id)?;
        // The full `name_path()` is in hand here, so mint the owning
        // type's object-identity token from the crate-stripped qualified
        // path (`module::Type` / `module::Enum::Variant`) — the same key
        // the layout layer is registered under.  `owner_root` stays the
        // bare leaf its non-layout consumers expect; `owner_id` carries
        // the collision-free identity for the layout layer.
        let name_path = td.item_meta.name_path();
        // `owner_root` is the annotation-side classdef key: a
        // reference-payload workspace enum instantiation reads its field
        // off the per-instantiation variant class (`Result<Tuple>::Ok`),
        // matching the receiver / constructor projection.  `owner_id`
        // (the layout-side `StructId` minted below) stays on the bare
        // template name so every instantiation shares the one template
        // variant layout — sound because the split is scoped to
        // reference payloads, which all share that word-slot layout.
        let owner_leaf = name_path.rsplit("::").next().unwrap_or("").to_string();
        let owner_root = match head
            .as_object()
            .and_then(|h| adt_head_instantiation_suffix(h, self.llbc))
        {
            Some(suffix) => format!("{owner_leaf}{suffix}"),
            None => owner_leaf,
        };
        match (&td.kind, variant_idx) {
            (TypeDeclKind::Struct(fields), None) => {
                let f = fields.get(field_idx)?;
                let name = f
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("__pos_{field_idx}"));
                let ty = clone_tyref(&f.ty);
                let owner_id = Some(majit_ir::descr::StructId::from_canonical(
                    &strip_crate_prefix(&name_path),
                ));
                Some((owner_root, name, ty, owner_id))
            }
            (TypeDeclKind::Enum(variants), Some(vidx)) => {
                let variant = variants.get(vidx as usize)?;
                let f = variant.fields.get(field_idx)?;
                let name = f
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("__pos_{field_idx}"));
                let ty = clone_tyref(&f.ty);
                // owner_root = the variant subclass `{enum_leaf}::{variant}`
                // — the read resolves the variant's own field at its exact
                // enum-relative offset (`variant_layouts[vidx].field_offsets`,
                // registered under this key).  The downcast statically fixes
                // the variant.
                let variant_owner = format!("{owner_root}::{}", variant.name);
                let owner_id = Some(majit_ir::descr::StructId::from_canonical(&format!(
                    "{}::{}",
                    strip_crate_prefix(&name_path),
                    variant.name
                )));
                Some((variant_owner, name, ty, owner_id))
            }
            _ => None,
        }
    }

    /// Resolve a global `def_id` to its fully-qualified path segments
    /// via the reader's `global_decls` table.
    fn global_segments(&self, mir_bb: usize, def_id: u64) -> Result<Vec<String>, LowerError> {
        self.llbc
            .global_by_id(def_id)
            .map(|g| {
                g.item_meta
                    .name_path()
                    .split("::")
                    .map(|s| s.to_string())
                    .collect()
            })
            .ok_or_else(|| {
                LowerError::Schema(format!(
                    "bb{mir_bb}: Place::Global references unknown GlobalDecl id {def_id}"
                ))
            })
    }

    /// Fold a `NamedConst` global (Rust `const`) whose initializer is a
    /// single literal assignment to its `ConstInt` / `ConstBool` /
    /// `ConstFloat` value.  `None` for statics (`global_kind` ≠
    /// `NamedConst`), absent initializers, or any non-trivial init body
    /// (a computed const keeps the accessor path so it is not
    /// mis-evaluated here).
    fn fold_named_const_global(&self, def_id: u64) -> Option<OpKind> {
        let gd = self.llbc.global_by_id(def_id)?;
        if gd
            .rest
            .get("global_kind")
            .and_then(serde_json::Value::as_str)
            != Some("NamedConst")
        {
            return None;
        }
        let init_id = gd.rest.get("init")?.as_u64()?;
        let init = self.llbc.fn_by_id(init_id)?;
        let body = init.unstructured()?;
        // The initializer must be exactly one literal assignment to the
        // return local (`_0 = const <lit>`); anything else (arithmetic,
        // calls, multiple assigns) is a computed const left to the
        // accessor path.
        let mut found: Option<&serde_json::Value> = None;
        for blk in &body.body {
            for st in &blk.statements {
                let Some(assign) = st.kind.get("Assign").and_then(|a| a.as_array()) else {
                    continue;
                };
                let is_local0 = assign
                    .first()
                    .and_then(|p| p.get("kind"))
                    .and_then(|k| k.get("Local"))
                    .and_then(serde_json::Value::as_u64)
                    == Some(0);
                let lit = assign
                    .get(1)
                    .and_then(|rv| rv.get("Use"))
                    .and_then(|u| u.get("Const"))
                    .and_then(|c| c.get("kind"))
                    .and_then(|k| k.get("Literal"));
                match lit {
                    Some(l) if is_local0 => {
                        if found.is_some() {
                            return None;
                        }
                        found = Some(l);
                    }
                    // A non-literal write to _0 (computed const) — bail.
                    _ if is_local0 => return None,
                    _ => {}
                }
            }
        }
        match decode_literal(found?).ok()? {
            DecodedConst::Int(n) => Some(OpKind::ConstInt(n)),
            DecodedConst::Bool(b) => Some(OpKind::ConstBool(b)),
            DecodedConst::Float(bits) => Some(OpKind::ConstFloat(bits)),
            _ => None,
        }
    }

    fn static_addr_op(&self, segments: &[String]) -> Option<OpKind> {
        let full = segments.join("::");
        let stripped = strip_crate_prefix(&full);
        for (key, addr) in self.static_addrs.refs {
            if static_key_matches(&full, &stripped, key) {
                return Some(OpKind::ConstRefAddr(*addr));
            }
        }
        None
    }

    /// Value of an immutable size `const` baked at build time
    /// (`HostStaticAddrs.int_values`).  The initializer is a
    /// `size_of::<T>()` the front-end cannot evaluate from the LLBC
    /// (Charon leaves the target-dependent layout symbolic), so the
    /// driver captures its compile-time value and the read folds to the
    /// same `ConstInt` an inline integer literal produces — rather than a
    /// 0-arg accessor call no registry resolves.  The value is identical
    /// at the call site (the JIT is native: host target == runtime
    /// target).
    fn static_int_value_op(&self, segments: &[String]) -> Option<OpKind> {
        let full = segments.join("::");
        let stripped = strip_crate_prefix(&full);
        for (key, value) in self.static_addrs.int_values {
            if static_key_matches(&full, &stripped, key) {
                return Some(OpKind::ConstInt(*value));
            }
        }
        None
    }

    /// Address of a `pytypes`-bucket host static — a `PyType` singleton
    /// (`&SLICE_TYPE`, `&INT_TYPE`, …).  The `Global` reader lowers these
    /// to a `__pyre_cast_instance["PyType"]` narrow of the raw address
    /// (a typed instance pointer) rather than the bare `ConstInt` the
    /// `refs` siblings avoid: a `PyType` static is the same kind of value
    /// as the `(*obj).ob_type` field-read it is compared against, so it
    /// must type `SomeInstance("PyType")` for `rtype_is_` (pointer
    /// identity) to lower the `ob_type == &TYPE` chain
    /// (`is_slice` / `is_cell` / `is_range`).  `jit_static_pytype_addrs`
    /// puts only `PyType` statics in this bucket, so the root is always
    /// `"PyType"`.
    fn pytype_static_addr(&self, segments: &[String]) -> Option<i64> {
        let full = segments.join("::");
        let stripped = strip_crate_prefix(&full);
        self.static_addrs
            .pytypes
            .iter()
            .find(|(key, _)| static_key_matches(&full, &stripped, key))
            .map(|(_, addr)| *addr)
    }

    /// Evaluate a global's initializer to its literal when the body is
    /// the trivial `_0 = <literal>; return` shape.  The read then
    /// lowers to the same `Const*` op an inline literal produces —
    /// flow graphs carry module-level constants as `Constant(value)`,
    /// so config bools like `WITHPREBUILTINT` constant-fold their
    /// guarded branches instead of minting a synthetic 0-arg call no
    /// registry can resolve.  Non-trivial initializers (multi-block,
    /// calls, aggregates) return `None` and keep the Call fallback.
    fn const_eval_global(&self, def_id: u64) -> Option<OpKind> {
        let g = self.llbc.global_by_id(def_id)?;
        // Only an immutable, non-thread-local global folds to its init
        // literal.  A `static mut` is written at runtime and a
        // thread-local holds a per-thread value, so a post-init read of
        // either must reach the live accessor, not the initialiser.  (A
        // by-value read carries no address identity, so an immutable
        // `static`'s literal is safe to inline here even when its address
        // is taken elsewhere.)  `global_kind` does not distinguish
        // `static mut` from `static`, so the mutability comes from the
        // `static mut` keyword in Charon's recorded `source_text`.
        if g.rest
            .get("global_kind")
            .and_then(serde_json::Value::as_str)
            == Some("ThreadLocal")
        {
            return None;
        }
        let is_static_mut = g
            .rest
            .get("item_meta")
            .and_then(|m| m.get("source_text"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|s| s.contains("static mut"));
        if is_static_mut {
            return None;
        }
        let init_id = g.rest.get("init")?.as_u64()?;
        let fd = self.llbc.fn_by_id(init_id)?;
        let u = fd.unstructured()?;
        let [block] = u.body.as_slice() else {
            return None;
        };
        if !matches!(block.term(), Ok(TermKind::Return)) {
            return None;
        }
        let mut assigned: Option<serde_json::Value> = None;
        for stmt in &block.statements {
            match stmt.stmt_kind() {
                Ok(StmtKind::StorageLive(_)) | Ok(StmtKind::StorageDead(_)) => {}
                Ok(StmtKind::Assign(place, Rvalue::Use(Operand::Const(value))))
                    if matches!(place.kind, PlaceKind::Local(0)) && assigned.is_none() =>
                {
                    assigned = Some(value);
                }
                _ => return None,
            }
        }
        match decode_constant(self.llbc, &assigned?).ok()? {
            DecodedConst::Int(n) => Some(OpKind::ConstInt(n)),
            DecodedConst::Bool(b) => Some(OpKind::ConstBool(b)),
            DecodedConst::Float(bits) => Some(OpKind::ConstFloat(bits)),
            // Strings / fn pointers keep the existing Call shapes the
            // operand-constant lowering uses; folding them here would
            // diverge from the `Rvalue::Use(Const)` treatment.
            DecodedConst::Str(_) | DecodedConst::FnPath(_) => None,
        }
    }

    /// Fold a `NamedConst` global whose initializer is exactly
    /// `size_of::<T>()` / `align_of::<T>()` to the concrete byte size /
    /// alignment Charon resolved for `T`'s layout.  The const's value is
    /// a build-time constant of the extraction target, so folding it to
    /// a `ConstInt` removes the residual accessor call the layout-size
    /// consts (`FUNCTION_OBJECT_SIZE`, `W_DICT_OBJECT_SIZE`, …) otherwise
    /// lower to (a `FunctionPath` the rtyper cannot register).
    ///
    /// Accepts only the *exact* shape — `_0` (the const's value) is written
    /// once, by a single `size_of`/`align_of` call, unconditionally — and
    /// returns `None` for anything richer so it keeps the residual accessor
    /// path, mirroring the strict single-statement acceptance in
    /// [`Self::const_eval_global`].  Rejected:
    ///   * a `Switch` terminator (data-dependent control flow), so the call
    ///     can never be made conditional;
    ///   * any statement that assigns `_0` (a computed value such as
    ///     `size_of::<A>() + size_of::<B>()`, whose `+` writes `_0`);
    ///   * a second `_0`-defining call, or a `_0`-defining call to any
    ///     function other than `size_of`/`align_of`;
    ///   * a body with no `Return`.
    /// Linear `Goto`s and panic-cleanup terminators (`Abort`,
    /// `UnwindResume`, `Drop`, `Assert`), plus calls and assignments that
    /// do *not* write `_0`, are permitted: none of them define the const
    /// value, so with no `Switch` the single `size_of` call is its
    /// unconditional sole definer.  Also `None` for a non-ADT type argument
    /// (primitive / pointer / tuple, which has no `TypeDecl` layout to
    /// read) or a layout Charon left unresolved.
    fn fold_size_const_global(&self, def_id: u64) -> Option<OpKind> {
        let gd = self.llbc.global_by_id(def_id)?;
        if gd
            .rest
            .get("global_kind")
            .and_then(serde_json::Value::as_str)
            != Some("NamedConst")
        {
            return None;
        }
        let init_id = gd.rest.get("init")?.as_u64()?;
        let body = self.llbc.fn_by_id(init_id)?.unstructured()?;

        // The single `size_of`/`align_of` call that defines `_0`, captured
        // as `(want_align, type_argument)`.  `term()`/`stmt_kind()` return
        // owned values, so the type expression is cloned out of the call.
        let mut found: Option<(bool, serde_json::Value)> = None;
        let mut saw_return = false;
        for block in &body.body {
            for stmt in &block.statements {
                match stmt.stmt_kind() {
                    Ok(StmtKind::StorageLive(_))
                    | Ok(StmtKind::StorageDead(_))
                    | Ok(StmtKind::PlaceMention(_)) => {}
                    // A statement writing `_0` means the value is computed,
                    // not the bare `size_of`/`align_of` call result.
                    Ok(StmtKind::Assign(place, _)) if matches!(place.kind, PlaceKind::Local(0)) => {
                        return None;
                    }
                    // An assignment to a temporary cannot reach `_0` without
                    // an `_0` write we already reject, so it is inert here.
                    Ok(StmtKind::Assign(_, _)) => {}
                    // `Assert` statements and anything unparsed: bail.
                    _ => return None,
                }
            }
            match block.term() {
                // Data-dependent control flow or an unreadable terminator:
                // the const value would not be the unconditional size_of.
                Ok(TermKind::Switch { .. }) | Ok(TermKind::Unknown) | Err(_) => {
                    return None;
                }
                Ok(TermKind::Return) => saw_return = true,
                Ok(TermKind::Call { call, .. })
                    if matches!(call.dest.kind, PlaceKind::Local(0)) =>
                {
                    let CallFunc::Regular(reg) = &call.func else {
                        return None;
                    };
                    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
                        return None;
                    };
                    let want_align = match self.llbc.fn_by_id(*id)?.item_meta.name_path().as_str() {
                        "core::mem::size_of" => false,
                        "core::mem::align_of" => true,
                        // `_0` defined by some other function: not a bare
                        // size_of/align_of const.
                        _ => return None,
                    };
                    // A second `_0`-defining call makes the value
                    // order-dependent; only a single one is foldable.
                    if found.is_some() {
                        return None;
                    }
                    let ty = reg.generics.get("types")?.as_array()?.first()?.clone();
                    found = Some((want_align, ty));
                }
                // `Goto` / `Abort` / `UnwindResume` / `Drop` / `Assert`, and
                // any call that does not define `_0`: linear continuation or
                // panic cleanup, none of which write the const value.
                Ok(_) => {}
            }
        }
        let (want_align, ty) = found?;
        if !saw_return {
            return None;
        }
        let adt = self.resolve_tyexpr_to_adt_def_id(&ty)?;
        let layout = self.llbc.type_by_id(adt)?.layout_for_target("")?;
        let value = if want_align {
            layout.align
        } else {
            layout.size
        }?;
        Some(OpKind::ConstInt(value as i64))
    }

    // -----------------------------------------------------------------------
    // Terminators
    // -----------------------------------------------------------------------

    fn lower_terminator(&mut self, mir_bb: usize, term: TermKind) -> Result<(), LowerError> {
        let bb_id = self.block_id[mir_bb];
        match term {
            TermKind::Return => {
                // A `-> ()` body materializes its implicit return as a
                // unit aggregate (`_0 = ()`), which lowers to a
                // Ref-typed transparent ctor.  Feeding that into the
                // return block colors the result kind 'r', contradicting
                // the declared void kind ('v').  RPython filters void
                // out of return links (NON_VOID); mirror that by routing
                // an empty void return.
                if is_unit_type(&self.body.locals.locals[0].ty, self.llbc) {
                    self.graph.set_return(bb_id, None);
                    return Ok(());
                }
                let ret = self.local_var[0].clone().ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: Return without any Assign to MIR local 0"
                    ))
                })?;
                self.graph.set_return(bb_id, Some(ret));
                Ok(())
            }
            TermKind::Abort(_) => {
                // A Rust panic-abort (`unreachable!()`, `panic!`,
                // failed `unwrap`).  Python-level exceptions never
                // reach here — they ride the `Result<_, PyError>`
                // Switch/Return edges as ordinary control flow — so
                // an Abort marks a "shouldn't occur at run-time"
                // path, exactly the implicit-exception raise of
                // `RaiseImplicit.nomoreblocks`
                // (`flowcontext.py:1271-1284`).  Closing the block
                // with `[Constant(AssertionError),
                // Constant(AssertionError(msg))]` lets
                // `remove_assertion_errors` (simplify.py:321-346)
                // prune the branch — e.g. the `else unreachable!()`
                // arm of a per-variant `let Instruction::X {..} =`
                // re-match folds away together with its discriminant
                // switch.
                self.graph.set_raise_implicit(bb_id, "AssertionError");
                Ok(())
            }
            TermKind::UnwindResume => {
                // Unwind-table cleanup resume.  Its only inbound edges
                // are `on_unwind` edges, all of which this lowering
                // drops, so the block is unreachable — close it as a
                // bare exception propagation; the flowspace adapter
                // converts only the reachable closure and never sees
                // it.  Python-level exceptions never reach here: they
                // ride the `Result<_, PyError>` Switch/Return edges as
                // ordinary control flow.
                self.graph.set_raise(bb_id, "mir-unwind");
                Ok(())
            }
            TermKind::Goto { target } => {
                let target_bb = self.block_id[target as usize];
                let args = self.edge_args(mir_bb, target as usize)?;
                self.graph.set_goto(bb_id, target_bb, args);
                Ok(())
            }
            TermKind::Assert {
                target, on_unwind, ..
            } => {
                // A Rust-level overflow / bounds / division-by-zero check
                // whose `on_unwind` successor is a bare UnwindResume panic
                // path. These are debug-build artifacts release builds
                // elide, with no Python-observable meaning: Python ints are
                // arbitrary-precision (no machine OverflowError), and any
                // IndexError / ZeroDivisionError is produced by an explicit
                // value-level guard that lowers to a `Result` Switch and is
                // already carried by the ArrayRead / BinOp op's canraise.
                // Strip the check — branch unconditionally to the success
                // continuation — leaving the panic path unreachable.
                // RPython does the same (`backendopt/removeassert.py`).
                let _ = on_unwind;
                let target_bb = self.block_id[target as usize];
                let args = self.edge_args(mir_bb, target as usize)?;
                self.graph.set_goto(bb_id, target_bb, args);
                Ok(())
            }
            TermKind::Switch { discr, targets } => self.lower_switch(mir_bb, discr, targets),
            TermKind::Call {
                call,
                target,
                on_unwind,
            } => self.lower_call(mir_bb, call, target as usize, on_unwind as usize),
            // `Drop` is a destructor invocation — the JIT does not model
            // destructor semantics (RPython lacks them entirely), so
            // forward unconditionally to the success continuation and
            // ignore the unwind path. Any side effects worth tracing
            // (heap mutation by a `Drop` impl) become visible through
            // the field/array ops the destructor body itself emits at
            // a deeper inlining level.
            TermKind::Drop { target, .. } => {
                let target_bb = self.block_id[target as usize];
                let args = self.edge_args(mir_bb, target as usize)?;
                self.graph.set_goto(bb_id, target_bb, args);
                Ok(())
            }
            TermKind::Unknown => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: unknown TermKind"
            ))),
        }
    }

    fn lower_call(
        &mut self,
        mir_bb: usize,
        call: CallPayload,
        target: usize,
        on_unwind: usize,
    ) -> Result<(), LowerError> {
        let bb_id = self.block_id[mir_bb];

        // Destination must be a plain `Local(i)` — projection-typed
        // destinations are not produced for monomorphized calls in any
        // body we extract today; fail-loud if Charon surfaces one.
        let dest_local = match call.dest.kind {
            PlaceKind::Local(i) => i as usize,
            _ => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: Call with projection-destination not supported"
                )));
            }
        };

        // The call result kind is the MIR-declared type of the
        // destination place. RPython `call.py:222` reads `FUNC.RESULT`
        // off the callee funcptr; the destination local's declared type
        // is that same value at the call site, so deriving it here keeps
        // `getcalldescr`'s `RESULT == FUNC.RESULT` check (`call.py:230`)
        // satisfied for non-`Int` returns such as
        // `new_for_call_with_closure_and_globals_obj` (Ref).
        //
        // A `-> ()` callee's graph reports a void result kind (its
        // `Return` lowers via `set_return(None)`, see [`is_unit_type`]),
        // so the call site must declare the result Void too — otherwise
        // `tyref_to_value_type`'s `Ref` projection for unit contradicts
        // the callee's `FUNC.RESULT=Void` and trips `call.rs:4268`
        // (e.g. `ExecutionContext.force_all_frames`).
        let result_ty = if is_unit_type(&call.dest.ty, self.llbc) {
            ValueType::Void
        } else {
            tyref_to_value_type(&call.dest.ty, self.llbc)
        };

        // Resolve arguments before deciding the call shape so receiver
        // resolution and `dyn` operand handling share the same path.
        let mut args: Vec<Variable> = Vec::with_capacity(call.args.len());
        // MIR local index behind each plain-local argument, kept
        // alongside the resolved Variables so call intercepts can
        // consult per-local lowering state
        // (`const_discriminant_locals`).
        let arg_locals: Vec<Option<usize>> = call
            .args
            .iter()
            .map(|op| match op {
                Operand::Copy(p) | Operand::Move(p) => match p.kind {
                    PlaceKind::Local(i) => Some(i as usize),
                    _ => None,
                },
                Operand::Const(_) => None,
            })
            .collect();
        // First argument's MIR-declared type, captured before the
        // operands are consumed — `reflexive_into_alias` compares it
        // against the destination type.
        let first_arg_ty: Option<TyRef> = call.args.first().and_then(|op| match op {
            Operand::Copy(p) | Operand::Move(p) => Some(clone_tyref(&p.ty)),
            Operand::Const(_) => None,
        });
        for op in call.args {
            args.push(self.resolve_operand(mir_bb, op)?);
        }

        let class = call.func.classify();
        // Full `name_path()` of a `CallKind::Fun` callee, captured for the
        // `Result<T, PyError>` scope decision at the `?`-diamond capture
        // site below: the built `CallTarget::Method` keeps only the leaf,
        // losing the module path the scope predicate keys on.  It is the
        // same string the callee-side gate sees (`fd.item_meta.name_path()`,
        // `lower_fun_decl_with_static_addrs`), so caller and callee agree on
        // whether a given callee is scoped.
        let mut callee_name_path: Option<String> = None;
        let op_kind = match (class, call.func) {
            (CallClass::Direct, CallFunc::Regular(reg))
            | (CallClass::Trait, CallFunc::Regular(reg)) => {
                // Reflexive blanket `into` — the callsite selected
                // `impl<T> From<T> for T`, a pure `T -> T` identity
                // conversion.  Bind the destination local to the
                // argument directly (same shape as a transparent
                // ctor alias) instead of emitting a call to core's
                // identity body, which is not a registered callee.
                //
                // The clause-bound variant — `msg.into()` inside a
                // generic body with `T: Into<String>` — has no
                // resolved impl to devirtualize through; for a
                // string-family target the lifted value model treats
                // the conversion as identity too (Rust `String` and
                // `&str` both lower to the immutable rpy_string), so
                // it takes the same alias path.
                if args.len() == 1
                    && (matches!(self.blanket_into_devirt(&reg), Some(IntoDevirt::Identity))
                        || self.trait_clause_into_string_identity(&reg, &call.dest.ty)
                        || self.is_noop_ptr_cast(&reg)
                        || self.is_reflexive_into_iter(&reg)
                        || self.is_hint_must_use(&reg)
                        || self.is_arguments_from_str(&reg))
                {
                    self.local_var[dest_local] = Some(args[0].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<String>::deref` / `<str>::deref` and the Wtf8 string
                // family: a `Deref` between two string-value types is a
                // pointer-follow with no transformation.  `String` / `&str`
                // / `Wtf8` / `Wtf8Buf` all project to the single immutable
                // `s_unicode0` value, so the deref is identity — alias the
                // destination to the argument directly (the same zero-op
                // shape as the reflexive `into` above) instead of falling
                // through to the residual `method("deref", …)` build, an
                // unregistered callee.  `deref_cast_root` returns `None` for
                // these (the `&str` dest resolves no struct root), so the
                // cast arm below declines; gate on the receiver AND the
                // destination both being string values so slice / `Vec`
                // derefs (`&[T]`, root `Builtin::Slice`) keep their ordinary
                // path.
                if args.len() == 1
                    && is_deref_call(&reg, self.llbc)
                    && first_arg_ty
                        .as_ref()
                        .is_some_and(|t| tyref_is_string_value(t, self.llbc))
                    && tyref_is_string_value(&call.dest.ty, self.llbc)
                {
                    self.local_var[dest_local] = Some(args[0].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<Box<T>>::deref` / `<Rc<T>>::deref` / `<Arc<T>>::deref`
                // / the workspace `FrameBox::deref` (+ their `deref_mut`)
                // whose `&T` is a registered struct.  The handle is one
                // pointer word, so `*p` is a typed pointer reinterpret:
                // emit the `cast_pointer(T, p)` downcast marker
                // (`cast_pointer_marker_op`) the flowspace adapter rebuilds
                // into `simple_call(lltype.cast_pointer, T, p)`, yielding
                // `SomeInstance(T)` regardless of the receiver's
                // classdef-less annotation.  Slice / `str` derefs resolve
                // no struct root and keep their ordinary lowering.
                if args.len() == 1
                    && let Some(root) = self.deref_cast_root(&reg, &call.dest.ty)
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: cast_pointer_marker_op(root, args[0].clone()),
                    });
                    self.local_var[dest_local] = Some(res);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<String>::deref` / `<Vec<T>>::deref` (+ `deref_mut`) —
                // identity in the lifted value model (String/&str and
                // Vec/&[T] share one repr), so alias the destination to
                // the receiver instead of emitting a `deref` method call
                // the rtyper cannot route on the classdef-less receiver.
                if args.len() == 1 && self.is_container_identity_deref(&reg) {
                    self.local_var[dest_local] = Some(args[0].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<String as AsRef<str>>::as_ref` / `String::as_str` /
                // `<String as Borrow<str>>::borrow` — a `&str` view of the
                // same string, identity in the lifted value model, so alias
                // the destination to the receiver instead of emitting an
                // `as_ref` method call the rtyper cannot route on the
                // classdef-less string receiver.
                if args.len() == 1 && self.is_string_to_str_identity(&reg, &call.dest.ty) {
                    self.local_var[dest_local] = Some(args[0].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<str|String as ToString>::to_string` on a string-family
                // receiver — a `String` clone that is an identity in the
                // lifted value model, so alias the destination to the
                // receiver instead of emitting an unregistered `to_string`.
                if args.len() == 1 && self.is_to_string_identity(&reg, first_arg_ty.as_ref()) {
                    self.local_var[dest_local] = Some(args[0].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // Workspace `Index::index` / `IndexMut::index_mut`
                // impls (`FixedObjectArray` and friends) bottom out at
                // raw-slice construction (`as_mut_slice` →
                // `from_raw_parts`), which has no graph lowering.  The
                // callsite is RPython's getarrayitem: lower it as an
                // eager `ArrayRead` for value uses (`x = arr[i]`
                // desugars to `x = *index(&arr, i)` and the `Deref`
                // read collapses to the bound element), and record the
                // `(base, index)` pair so the paired `*p = v` write
                // (`arr[i] = v` desugar) emits `ArrayWrite` from the
                // `emit_projection_write` `Deref` arm.
                if args.len() == 2 && self.is_workspace_index_call(&reg) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::ArrayRead {
                            base: args[0].clone(),
                            index: args[1].clone(),
                            item_ty: ValueType::Ref(None),
                            array_type_id: None,
                            nolength: false,
                            pure: false,
                        },
                    });
                    self.index_elem_alias.insert(
                        dest_local,
                        IndexElemAlias {
                            base_local: arg_locals.first().copied().flatten(),
                            base_var: args[0].clone(),
                            index_local: arg_locals.get(1).copied().flatten(),
                            index_var: args[1].clone(),
                        },
                    );
                    self.local_var[dest_local] = Some(res);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `*items_block_items_base(block).add(idx)` — a list /
                // tuple element load or store through an `ItemsBlock`
                // items pointer.  The same getarrayitem / setarrayitem
                // decomposition as the workspace `Index` arm above, but
                // reached through a raw `.add` whose base traces to the
                // header-returning accessor (brick 1) rather than
                // `Index::index`: emit an `ArrayRead` for `x = *p` (the
                // `Deref` read collapses to the bound element) and record
                // the `(base, index)` pair so a paired `*p = v` write
                // emits `ArrayWrite` from `emit_projection_write`'s
                // `Deref` arm.  The gate ([`is_list_items_elem_ptr_add`])
                // requires the `.add` result be dereferenced exactly once
                // and never escape as a raw pointer, so the residual
                // pointer-walking callers (`object_insert` / `_remove` /
                // `_splice`) keep their `add` and fall to the legacy
                // walker.
                if self.is_list_items_elem_ptr_add(
                    &reg,
                    args.len(),
                    &arg_locals,
                    first_arg_ty.as_ref(),
                    dest_local,
                ) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::ArrayRead {
                            base: args[0].clone(),
                            index: args[1].clone(),
                            item_ty: ValueType::Ref(None),
                            array_type_id: None,
                            nolength: false,
                            pure: false,
                        },
                    });
                    self.index_elem_alias.insert(
                        dest_local,
                        IndexElemAlias {
                            base_local: arg_locals.first().copied().flatten(),
                            base_var: args[0].clone(),
                            index_local: arg_locals.get(1).copied().flatten(),
                            index_var: args[1].clone(),
                        },
                    );
                    self.local_var[dest_local] = Some(res);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<[T]>::swap(s, a, b)` over a `&mut [T]` whose base is
                // the same `FixedObjectArray` shape the workspace index
                // path reads.  Lower to the getarrayitem/setarrayitem
                // decomposition: read both elements, then write each
                // back to the other's index.  The base operand feeds the
                // synthetic `ArrayWrite`s the MIR never spells, but every
                // arg here is live in the call block (no cross-block
                // rebind like the deferred `index_mut` write), so no
                // extra liveness threading is needed.  The call returns
                // `()`; its dead destination binds to a fresh Void var.
                if args.len() == 3 && self.is_slice_swap_call(&reg) {
                    let base = args[0].clone();
                    let idx_a = args[1].clone();
                    let idx_b = args[2].clone();
                    let elem_a = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    let elem_b = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(elem_a.clone()),
                        kind: OpKind::ArrayRead {
                            base: base.clone(),
                            index: idx_a.clone(),
                            item_ty: ValueType::Ref(None),
                            array_type_id: None,
                            nolength: false,
                            pure: false,
                        },
                    });
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(elem_b.clone()),
                        kind: OpKind::ArrayRead {
                            base: base.clone(),
                            index: idx_b.clone(),
                            item_ty: ValueType::Ref(None),
                            array_type_id: None,
                            nolength: false,
                            pure: false,
                        },
                    });
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: None,
                        kind: OpKind::ArrayWrite {
                            base: base.clone(),
                            index: idx_a,
                            value: LinkArg::Value(elem_b),
                            item_ty: ValueType::Ref(None),
                            array_type_id: None,
                            nolength: false,
                        },
                    });
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: None,
                        kind: OpKind::ArrayWrite {
                            base,
                            index: idx_b,
                            value: LinkArg::Value(elem_a),
                            item_ty: ValueType::Ref(None),
                            array_type_id: None,
                            nolength: false,
                        },
                    });
                    self.local_var[dest_local] = Some(
                        self.graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Void),
                    );
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `*const T::cast_mut()` / `*mut T::cast_const()` /
                // `<ptr>::cast()` — address-preserving pointer
                // reinterprets the JIT models as identity (the
                // receiver-method twin of the `CastKind::RawPtr` Rvalue
                // → `same_as`).  Alias the destination to the receiver so
                // the method name never reaches the rtyper as a
                // `ptr.getattr`.
                if args.len() == 1 && self.is_ptr_identity_cast(&reg) {
                    // When the cast target names a registered struct root
                    // (`ptr.cast::<W_SRE_Pattern>()` then a field read),
                    // narrow the result to `SomeInstance(root)` exactly
                    // like the `Rvalue::Cast` arm: a bare alias leaves the
                    // pointer classdef-less and the downstream `getattr`
                    // blocks at the annotator.  The receiver is already a
                    // raw pointer (`is_ptr_identity_cast`), so this is a
                    // genuine `cast_pointer` (ptr→ptr).
                    if let ValueType::Ref(_) = tyref_to_value_type(&call.dest.ty, self.llbc)
                        && let Some(root) = tyref_class_root(&call.dest.ty, self.llbc)
                    {
                        let res = self
                            .graph
                            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(res.clone()),
                            kind: OpKind::Call {
                                target: CallTarget::FunctionPath {
                                    segments: vec![
                                        "__pyre_cast_instance".to_string(),
                                        root.clone(),
                                    ],
                                },
                                args: vec![args[0].clone()],
                                result_ty: ValueType::Ref(Some(root)),
                            },
                        });
                        self.local_var[dest_local] = Some(res);
                    } else {
                        self.local_var[dest_local] = Some(args[0].clone());
                    }
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<Atomic*>::load(&self, ordering)` — a relaxed read of a
                // layout-transparent atomic.  `&self` already aliases the
                // inner field read, so alias the destination to it (the
                // `ordering` arg is discarded); the `load` name never
                // reaches the rtyper as a `ptr.getattr`.
                if args.len() == 2 && self.is_atomic_load(&reg) {
                    self.local_var[dest_local] = Some(args[0].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `(block as *mut u8).add(ITEMS_BLOCK_ITEMS_OFFSET)` inside
                // an `ItemsBlock` items-base accessor — the interior items
                // pointer the runtime reads through a `base_size`-bearing
                // gcarray descr, so the JIT's items base is the header
                // pointer itself.  Alias the destination to the receiver,
                // dropping the unregistered raw-pointer `add`
                // ([`is_items_block_base_ptr_add`] keys on the enclosing
                // accessor so a dereferenced `.add` elsewhere is untouched).
                if args.len() == 2 && self.is_items_block_base_ptr_add(&reg) {
                    self.local_var[dest_local] = Some(args[0].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `f64::is_nan(x)` is `x != x` (`rfloat.isnan`) — emit the
                // reflexive `ne` BinOp instead of an unresolved call.
                if args.len() == 1 && self.is_f64_is_nan(&reg) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::BinOp {
                            op: "ne".to_string(),
                            lhs: args[0].clone(),
                            rhs: args[0].clone(),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(res);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // The concrete container `IntoIterator::into_iter` impls
                // (`&[T]`/`Vec`/`[T;N]`) construct a container iterator.
                // Emit the `iter` operation on the container receiver — the
                // `("slice","iter")` bridge routes it to `Repr::rtype_iter`
                // (`ListIteratorRepr` via `make_iterator_repr`) — instead of
                // the unregistered concrete-impl `FunctionPath` callee.  The
                // canonical `core::slice::iter` segments are the bridge's
                // recognised token; the receiver annotation (a `SomeList`)
                // supplies the actual element repr.
                if args.len() == 1 && self.is_concrete_iter_constructor(&reg) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec![
                                    "core".to_string(),
                                    "slice".to_string(),
                                    "iter".to_string(),
                                ],
                            },
                            args: vec![args[0].clone()],
                            result_ty: ValueType::Ref(None),
                        },
                    });
                    self.local_var[dest_local] = Some(res);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<[T]>::len` / `Vec::len` returns the container element
                // count.  Emit the `__len` operation on the receiver — the
                // rtyper routes it through the `len` op
                // (`flowspace_adapter`), which on a `SomeList` receiver
                // lowers to `AbstractBaseListRepr.rtype_len` — instead of
                // the `getattr("len")` the generic method fallback emits,
                // which dead-ends at `Cannot find attribute "len"` on the
                // list annotation.  Same routing
                // [`is_concrete_iter_constructor`] gives the container
                // `iter`.
                if args.len() == 1 && self.is_container_len(&reg) {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec!["__len".to_string()],
                            },
                            args: vec![args[0].clone()],
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(res);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `Vec::as_slice` / `<[T]>::as_slice` borrows the same
                // elements as a slice — identity on the list model.  Alias
                // the result to the receiver so the slice consumer reads
                // the list directly, instead of the `getattr("as_slice")`
                // the generic method fallback emits, which dead-ends at
                // `Cannot find attribute "as_slice"` on the list
                // annotation.  Same shape as the reflexive identity
                // aliases below.
                if args.len() == 1 && self.is_container_as_slice(&reg) {
                    self.local_var[dest_local] = Some(args[0].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `alloc::fmt::format` of a no-placeholder constant
                // message — `format!("literal")`, whose `format_args!`
                // lowered to `Arguments::from_str` (aliased to its
                // `__str_const` operand by the identity fold above).  The
                // render of a constant string is that string, so alias the
                // result to the constant instead of leaving an unregistered
                // `alloc::fmt::format` call.
                if args.len() == 1
                    && self.is_fmt_format_call(&reg)
                    && self.traces_to_str_const(&args[0])
                {
                    self.local_var[dest_local] = Some(args[0].clone());
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // Resolve the target function's fully-qualified path
                // through the FunId → FunDecl table. `Trait` here is
                // Charon's "trait-bound generic resolved at extraction
                // time", which is itself a direct call once the impl
                // is selected — same OpKind shape as Direct.
                //
                // When the FunDecl's name path encodes an `Impl`
                // segment whose owner type is resolvable, emit
                // `CallTarget::Method` instead of `FunctionPath` so the
                // annotator's `MethodDesc.func_args`
                // (`annotator/description.rs:2278`) prepends a
                // classdef-bound `SomeInstance` for `self`.  Without it,
                // the callee body's `self` lands with `classdef=None`
                // and any `.field` projection on it panics at
                // `unaryop.rs:3587` (lib test
                // `generic_handler_graphs_keep_symbolic_fnaddr_surface`).
                let (segments, method_hint) = self.call_target_segments(mir_bb, &reg)?;
                // For a method/direct callee this equals the callee's
                // `name_path()`; the scope predicate keys on the module
                // path, which the built `CallTarget::Method` drops.
                callee_name_path = Some(segments.join("::"));
                if self.try_lower_checked_neg(
                    mir_bb,
                    &reg.kind,
                    &segments,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_usize_try_from(
                    mir_bb,
                    &reg.kind,
                    &segments,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                if self.try_lower_num_from(
                    mir_bb,
                    &reg.kind,
                    &segments,
                    &args,
                    dest_local,
                    &call.dest.ty,
                    target,
                )? {
                    return Ok(());
                }
                // `<str as PartialEq>::eq(a, b)` is the string-equality
                // `BinOp("eq")` (pairtype `rtype_eq` → `ll_streq`) — emit it
                // instead of leaving the graph-less trait-method extern.
                // Both operands already type `SomeString`; the comparison
                // ops share the reflexive `BinOp` dispatch `f64::is_nan`
                // uses for `ne`.
                if args.len() == 2
                    && fmt_path_ends_with(&segments, &["str", "traits", "<Impl>", "eq"])
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::BinOp {
                            op: "eq".to_string(),
                            lhs: args[0].clone(),
                            rhs: args[1].clone(),
                            result_ty: ValueType::Int,
                        },
                    });
                    self.local_var[dest_local] = Some(res);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<T as ToString>::to_string(x)` renders `x` to an owned
                // String — the same `str(x)` (`ll_str`) the format!
                // expansion emits for a Display placeholder.  Lower it to
                // `UnaryOp("str")` instead of leaving the graph-less
                // `to_string` extern; the rtyper routes `str` to the
                // operand repr's `ll_str` (string = identity).
                if args.len() == 1
                    && fmt_path_ends_with(&segments, &["string", "<Impl>", "to_string"])
                {
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::UnaryOp {
                            op: "str".to_string(),
                            operand: args[0].clone(),
                            result_ty: ValueType::Ref(None),
                        },
                    });
                    self.local_var[dest_local] = Some(res);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                // `<str>::is_empty(s)` is `len(s) == 0` (`ll_strlen` then a
                // `ConstInt(0)` compare) — emit the `__len` + `BinOp("eq")`
                // decomposition instead of leaving the graph-less
                // trait-method extern.  `__len` routes through the rtyper's
                // `len` op (`flowspace_adapter`), which on a `&str` operand
                // lowers to `StringRepr.rtype_len` → `ll_strlen` → the
                // `strlen` blackhole op; the `eq` of the two `Signed`
                // operands lowers to `int_eq`.
                if args.len() == 1 && fmt_path_ends_with(&segments, &["str", "<Impl>", "is_empty"])
                {
                    let push_op = |graph: &mut FunctionGraph, kind: OpKind| {
                        let res =
                            graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                        graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(res.clone()),
                            kind,
                        });
                        res
                    };
                    let len = push_op(
                        &mut self.graph,
                        OpKind::Call {
                            target: CallTarget::FunctionPath {
                                segments: vec!["__len".to_string()],
                            },
                            args: vec![args[0].clone()],
                            result_ty: ValueType::Int,
                        },
                    );
                    let zero = push_op(&mut self.graph, OpKind::ConstInt(0));
                    let res = push_op(
                        &mut self.graph,
                        OpKind::BinOp {
                            op: "eq".to_string(),
                            lhs: len,
                            rhs: zero,
                            result_ty: ValueType::Int,
                        },
                    );
                    self.local_var[dest_local] = Some(res);
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                let alias =
                    if let Some(payload) = self.expect_on_const_ok(&segments, &args, &arg_locals) {
                        // Identity unwrap: the receiver variable was bound
                        // directly to the `Ok` payload, so the result is
                        // that variable — bind and close the block with no
                        // op emitted.
                        Some(payload)
                    } else {
                        self.reflexive_into_alias(
                            &segments,
                            &args,
                            first_arg_ty.as_ref(),
                            &call.dest.ty,
                        )
                        .or_else(|| {
                            self.reflexive_into_iter_alias(
                                &segments,
                                &args,
                                first_arg_ty.as_ref(),
                                &call.dest.ty,
                            )
                        })
                        .or_else(|| self.trait_into_string_alias(&segments, &args, &call.dest.ty))
                        .or_else(|| self.wtf8_string_identity_alias(&segments, &args))
                        .or_else(|| {
                            self.oparg_arg_get_alias(&reg.kind, &segments, &args, &call.dest.ty)
                        })
                        .or_else(|| self.oparg_value_alias(&segments, &args))
                        .or_else(|| self.identity_passthrough_alias(&segments, &args))
                    };
                if let Some(value) = alias {
                    self.local_var[dest_local] = Some(value);
                    let bb_id = self.block_id[mir_bb];
                    let target_bb = self.block_id[target];
                    let link_args = self.edge_args(mir_bb, target)?;
                    self.graph.set_goto(bb_id, target_bb, link_args);
                    return Ok(());
                }
                {
                    // `jit::promote(x)` (and its `promote_string` /
                    // `promote_unicode` siblings) rewrites to the synthesised
                    // `hint_promote*` marker so the residual `OpKind::Call`
                    // reaches `jtransform::rewrite_op_hint`, which emits
                    // `[-live-, <kind>_guard_value(x)]`
                    // (`codewriter/jtransform.py:608-614`).  The rtyper
                    // lowers the marker to `same_as` for the dual-gate type
                    // projection (`flowspace_adapter`), and jtransform aliases
                    // the result back to `x`.  Same single-segment marker
                    // shape as the `elidable_promote` wrapper's
                    // `hint_promote_or_string`.
                    let promote_marker = self.jit_promote_marker(&reg);
                    let target = if args.len() == 1
                        && let Some(marker) = promote_marker
                    {
                        CallTarget::FunctionPath {
                            segments: vec![marker.to_string()],
                        }
                    } else {
                        // `CallTarget::Method` requires a receiver in `args[0]`
                        // (the flowspace adapter lowers it to `getattr(recv,
                        // method_leaf) → simple_call(bound_method, …)`).
                        // Charon's `impl_method_owner` matches both inherent
                        // methods (which carry `&self`) *and* associated
                        // functions (e.g. `RootScope::new()` — no `self` arg).
                        // Only the former actually has a receiver in `args[0]`;
                        // routing a 0-arg associated function through `Method`
                        // panics at `flowspace_adapter.rs:1045` ("Call::Method
                        // has empty args").  Fall back to the `FunctionPath`
                        // segments when there is no receiver to thread.
                        match method_hint {
                            Some((owner_root, leaf)) if !args.is_empty() => {
                                CallTarget::method(leaf, Some(owner_root))
                            }
                            _ => CallTarget::FunctionPath { segments },
                        }
                    };
                    OpKind::Call {
                        target,
                        args,
                        result_ty: result_ty.clone(),
                    }
                }
            }
            (CallClass::Dynamic, CallFunc::Dynamic(dyn_operand)) => {
                // `dyn Trait` virtual call. The fat-pointer receiver
                // is carried in `dyn_operand`; thread it into `args[0]`
                // and emit a synthetic `__dyn_call` path so the
                // codewriter sees a uniform `Call` shape.  A faithful
                // lowering would emit `VtableMethodPtr` + `IndirectCall`;
                // that needs the trait_root/method_name pair Charon does
                // not yet surface.
                let recv = self.resolve_operand(mir_bb, dyn_operand)?;
                let mut full_args = Vec::with_capacity(args.len() + 1);
                full_args.push(recv);
                full_args.extend(args);
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__dyn_call".to_string()],
                    },
                    args: full_args,
                    result_ty,
                }
            }
            (CallClass::Ptr, _) => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: Call CallClass::Ptr (fn pointer) not yet supported"
                )));
            }
            (CallClass::Unknown, _) | (_, CallFunc::Unknown) => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: Call with unknown CallFunc/CallClass"
                )));
            }
            // Class/payload mismatches shouldn't happen — `classify`
            // is total over the typed variants — but cover the arm so
            // the match is exhaustive without `_`.
            (CallClass::Dynamic, _) | (CallClass::Direct, _) | (CallClass::Trait, _) => {
                return Err(LowerError::Schema(format!(
                    "bb{mir_bb}: CallClass / CallFunc mismatch"
                )));
            }
        };

        // A hint-marker call (`jit::promote(x)` → `hint_promote`,
        // `#[elidable_promote]` → `hint_promote_or_string`) lowers to the
        // distinct `OpKind::Hint` op (RPython `flowspace/operation.py:521
        // add_operator('hint', None, dispatch=1)`) carrying the structured
        // hint `kind`, instead of a synthesised `Call` marker classified by
        // name downstream.  The flowspace oracle types it as `same_as(value)`
        // and `jtransform::rewrite_op_hint` rewrites it to the
        // `<kind>_guard_value` family.
        let op_kind = if let OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } = &op_kind
            && !args.is_empty()
            && let Some(kind) =
                crate::hints::classify_hint_segments(segments.iter().map(String::as_str))
        {
            OpKind::Hint {
                value: args[0].clone(),
                kind,
            }
        } else {
            op_kind
        };

        // Allocate the result Variable and bind it to the destination
        // local before pushing the op, so subsequent reads see the
        // freshly-minted Variable.
        let result_var = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.local_var[dest_local] = Some(result_var.clone());
        // Capture scoped `Result<T, PyError>` call results for the
        // `?`-diamond rewiring pass (`front::result_exc`) that runs
        // after the body lowering completes.
        if let OpKind::Call { .. } = &op_kind
            && callee_name_path
                .as_deref()
                .is_some_and(crate::front::result_exc::in_result_exc_scope)
            && crate::front::result_exc::tyref_is_result_of_pyerror(&call.dest.ty, self.llbc)
        {
            // The per-instantiation suffix of the callee's `Result<T,
            // PyError>` keys the shells `result_exc` rebuilds at the
            // rewrap site, matching the front aggregate path's suffix so
            // both writers share one ClassDef per instantiation.
            let suffix = crate::front::result_exc::tyref_result_instantiation_suffix(
                &call.dest.ty,
                self.llbc,
            );
            self.result_exc_call_results
                .push((result_var.clone(), suffix));
        }
        // Capture `Iterator::next()` results (`Option<T>`-typed) for the
        // `next`-diamond rewiring pass (`front::iter_next`).  Recognition
        // is liberal — any `next`-leaf call returning `Option` — because
        // the rewrite itself validates the surrounding for-loop match and
        // declines (leaving the residual call) on any other shape.
        if let OpKind::Call { target, .. } = &op_kind
            && crate::front::iter_next::is_iterator_next_target(target)
            && crate::front::result_exc::tyref_is_option(&call.dest.ty, self.llbc)
        {
            self.next_call_results.push(result_var.clone());
        }
        // Capture `i64::checked_{add,sub,mul}()` results (`Option<i64>`-
        // typed) for the checked-arith rewiring pass
        // (`front::checked_arith`), which rewrites each into the native
        // `*_ovf` op + OverflowError edge.  Recognition is liberal — any
        // `core::num::<Impl>::checked_*` call returning `Option` — because
        // the rewrite itself validates the surrounding overflow-fallback
        // match and declines (leaving the residual call) on any other
        // shape.
        if let OpKind::Call { target, .. } = &op_kind
            && crate::front::checked_arith::is_checked_arith_target(target)
            && crate::front::result_exc::tyref_is_option(&call.dest.ty, self.llbc)
        {
            self.checked_arith_call_results.push(result_var.clone());
        }
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(result_var),
            kind: op_kind,
        });

        // Close the block: forward to the success target. The call's
        // `on_unwind` successor is a Rust panic-cleanup path (destructor
        // drop-glue terminating in UnwindResume / Abort) with no Python
        // meaning — a Python exception raised by the callee rides the
        // SUCCESS edge as a `Result::Err` value, matched downstream as
        // ordinary control flow, not this unwind edge. The residual-call
        // `guard_no_exception` is re-derived op-locally from the callee
        // graph (`codewriter/call.rs` `_canraise`), so dropping the
        // front-graph unwind edge keeps the can-raise signal. A real
        // try/except handler would need a `LastException` edge here; the
        // interpreter expresses exceptions as `Result`, so none arises.
        let _ = on_unwind;
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }

    /// Resolve a Charon `CallKind` to a flattened path segment list the
    /// codewriter consumes as `CallTarget::FunctionPath`, plus an
    /// optional `(owner_root_leaf, method_leaf)` pair for impl methods,
    /// plus the `[Owner, leaf]` registration spelling when the callee is
    /// an inherent-impl *associated function* (no receiver).
    ///
    /// The method hint is `Some` when the FunDecl's raw name segments
    /// encode an `Impl` block immediately before the leaf `Ident` —
    /// the standard Charon shape for inherent / trait-impl methods
    /// (e.g. `pyre_interpreter::pyframe::<Impl>::locals_w_mut`).  The
    /// caller uses the hint to pick `CallTarget::Method` over
    /// `CallTarget::FunctionPath` so the annotator can prepend a
    /// classdef-bound `SomeInstance` for `self`; see the comment at
    /// the use site in [`Self::lower_call`].
    ///
    /// The associated-function spelling is computed only when the
    /// method hint is `None` (a receiver-shaped callee never consumes
    /// it), and only the `CallTarget::FunctionPath` construction in
    /// `lower_call` applies it — the raw `name_path()` segments stay
    /// untouched for the std special-case matchers (`checked_neg` /
    /// `try_from` / `into` / `expect`) that key on the
    /// `[.., "<Impl>", leaf]` shape.
    fn call_target_segments(
        &self,
        mir_bb: usize,
        reg: &RegularCall,
    ) -> Result<(Vec<String>, Option<(String, String)>), LowerError> {
        match &reg.kind {
            CallKind::Fun(FunId::Regular { id }) => self
                .llbc
                .fn_by_id(*id)
                .map(|fd| {
                    // Blanket `impl<T, U: From<T>> Into<U> for T`
                    // (core::convert) — `x.into()` is `U::from(x)`.
                    // The callsite's resolved `U: From<T>` obligation
                    // names the concrete From impl, so devirtualize to
                    // that impl's `from` the way rustc's
                    // monomorphization does.  The blanket body itself
                    // is generic-trait shaped and never lifts.  (The
                    // reflexive `Identity` outcome is intercepted at
                    // `lower_call` before reaching here.)
                    if let Some(IntoDevirt::Target(segments)) = self.blanket_into_devirt(reg) {
                        return (segments, None);
                    }
                    let method_hint = self.impl_method_owner(fd);
                    // An impl-block associated function (the method
                    // gate rejected it — no `self` receiver) is
                    // spelled `[<qualified owner>, <fn>]`, the key the
                    // canonical registration loop derives from
                    // `self_ty_root`; the raw `name_path()` carries an
                    // `<Impl>` segment that never matches a registry
                    // entry.
                    let segments: Vec<String> = if method_hint.is_none()
                        && let Some((owner_qualified, leaf)) =
                            impl_method_owner_for_fundecl(self.llbc, fd)
                    {
                        // Split like `CallPath::for_impl_method` so the
                        // segment vectors compare equal.
                        let mut v: Vec<String> =
                            owner_qualified.split("::").map(str::to_string).collect();
                        v.push(leaf);
                        v
                    } else {
                        fd.item_meta
                            .name_path()
                            .split("::")
                            .map(|s| s.to_string())
                            .collect()
                    };
                    (segments, method_hint)
                })
                .ok_or_else(|| {
                    LowerError::Schema(format!(
                        "bb{mir_bb}: Call references unknown FunDecl id {id}"
                    ))
                }),
            CallKind::Fun(FunId::Other(v)) => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: CallKind::Fun(Other) not yet supported: {v}"
            ))),
            // `CallKind::Trait([trait_ref, method_idx, fn_decl_id])` —
            // generic-trait method call.  Charon's `arr[2]` is the
            // `def_id` of the trait method declaration itself
            // (e.g. `pyre_interpreter::shared_opcode::SharedOpcodeHandler::
            // push_value`).
            //
            // `extract_trait_impls` parses the trait declaration's
            // default-body and registers it under BOTH
            // `["<default methods of <Trait>>", <method>]` (the
            // selfclassdef-bound `register_trait_method` path) and the
            // direct path `[<Trait>, <method>]` (lib.rs:957-969 —
            // `register_function_graph(direct_path, …)`).  The direct
            // path is the call-site shape Rust code emits when calling
            // `<Trait>::<method>(receiver, …)` and the BFS-driven
            // `find_all_graphs` reaches it as a regular candidate.
            //
            // To stay PyPy-orthodox for generic-trait dispatch, route
            // the call through that same `[<Trait>, <method>]` path so:
            //   1. BFS discovers the trait default body as a
            //      candidate, which transitively pulls in the helpers
            //      it calls (e.g. `opcode_load_const`).
            //   2. `flowspace_adapter` emits a `simple_call(<
            //      callable>, args…)` shape (no `getattr` surface) so
            //      the classdef-less receiver does not surface as a
            //      panicking `SomeInstance.getattr`.
            //
            // Falls back to the `["__trait_method", <label>]` synthetic
            // path when the fn_decl cannot be resolved or does not have
            // the trait-method shape (e.g. when arr[2] is missing or
            // points at an `Impl` block).
            CallKind::Trait(v) => {
                let fn_id = v
                    .as_array()
                    .and_then(|a| a.get(2))
                    .and_then(serde_json::Value::as_u64);
                let direct = fn_id
                    .and_then(|id| self.llbc.fn_by_id(id))
                    .and_then(trait_method_owner);
                if let Some((trait_leaf, method_leaf)) = direct {
                    Ok((vec![trait_leaf, method_leaf], None))
                } else {
                    let label = trait_call_label(v);
                    Ok((vec!["__trait_method".to_string(), label], None))
                }
            }
            CallKind::Ptr(v) => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: CallKind::Ptr not yet supported: {v}"
            ))),
            CallKind::Unknown => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: CallKind::Unknown"
            ))),
        }
    }

    /// Return `(owner_root_leaf, method_leaf)` when the FunDecl's name
    /// path encodes an impl block (inherent or trait-impl) whose owner
    /// type is resolvable through the LLBC tables.
    ///
    /// Charon serialises an impl method's name as:
    ///   `[Ident("crate"), Ident("mod"), Other({"Impl": ...}), Ident("method_name")]`
    /// where the `Impl` segment carries either
    ///   `{"Ty": {"skip_binder": {"Deduplicated": <type_id>}, "kind": "InherentImplBlock"}}`
    /// for inherent impls or `{"Trait": <trait_impl_id>}` for trait-impls.
    /// Trait-impl lookups indirect through the top-level `trait_impls`
    /// table, kept opaque (`schema::Translated.rest["trait_impls"]`)
    /// because no other consumer needs it typed.
    fn impl_method_owner(&self, fd: &FunDecl) -> Option<(String, String)> {
        let segs = &fd.item_meta.name;
        let last_idx = segs
            .iter()
            .rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
        let leaf = match &segs[last_idx] {
            NameSeg::Ident { ident: (s, _) } => s.clone(),
            _ => return None,
        };
        if last_idx == 0 {
            return None;
        }
        let impl_payload = match &segs[last_idx - 1] {
            NameSeg::Other(v) => v.as_object()?.get("Impl")?,
            _ => return None,
        };
        let owner_leaf = match self.resolve_impl_owner_adt_def_id(impl_payload) {
            Some(adt_def_id) => {
                // The hint is only valid when the fn actually receives
                // the impl owner as `self` (`Owner` / `&Owner` /
                // `*mut Owner` first input).  Associated functions —
                // `PyError::type_error(msg)` — have no receiver;
                // routing them as `Method` makes the adapter getattr
                // the method name off `args[0]` (the message string).
                if !self.first_input_is_adt(fd, adt_def_id) {
                    return None;
                }
                let td = self.llbc.type_by_id(adt_def_id)?;
                // A foreign opaque owner (`malachite_bigint::BigInt`,
                // `Sign`, …) has no extracted body, so the annotator never
                // mints a `ClassDef` for it — the receiver lands as a
                // classdef-less `SomeInstance` and a `CallTarget::Method`
                // getattr panics ("SomeInstance.getattr on classdef-less
                // instance").  The interpreter's overflow→long arms are the
                // cold `@jit.dont_look_inside` bailouts that operate on this
                // opaque value (`bigint_add`/`bigint_sub`/… call
                // `<BigInt as Add>::add` etc.), so the faithful treatment is
                // to residualize the call, not trace into the foreign body —
                // the `register_external` analog.  Declining the Method hint
                // routes the call through the `FunctionPath` form, which the
                // call registry resolves to an opaque external
                // (`register_foreign_opaque_method_externals`) and the
                // codewriter emits as a residual fnaddr call.
                if matches!(td.kind, TypeDeclKind::Opaque) {
                    return None;
                }
                let owner = td
                    .item_meta
                    .name_path()
                    .rsplit("::")
                    .next()
                    .unwrap_or("")
                    .to_string();
                // Only an actual method (first signature input is the
                // owner ADT, possibly behind `&`/`&mut`) may route as
                // `CallTarget::Method`.  An associated function with
                // arguments (`PyError::type_error(msg)`) would
                // otherwise thread its first argument as the getattr
                // receiver and the annotator resolves the method name
                // against that argument's type.  Compared by ADT
                // def_id, not name leaf, so generic owners
                // (`Result::branch` — `?`'s Try::branch) still match.
                let first_is_self = fd
                    .signature
                    .inputs
                    .first()
                    .and_then(|t| tyref_node(t, self.llbc))
                    .and_then(|n| strip_ty_wrappers(n, self.llbc))
                    .and_then(adt_node_def_id)
                    .is_some_and(|id| id == adt_def_id);
                if !first_is_self {
                    return None;
                }
                owner
            }
            // Non-ADT `Self` (primitive / raw pointer / slice): Charon leaves
            // the impl owner type unresolved, so the ADT table has no entry.
            // Fall back to the module Ident immediately preceding the `Impl`
            // NameSeg, which Charon names after the primitive's impl module
            // (`core::ptr::mut_ptr::<Impl>::is_null` → `mut_ptr`).  Restricted
            // to `(module, method)` pairs that have a classdef-less analyzer
            // reachable through the `getattr` → bound-method path
            // (`unaryop.rs::ptr_method_is_null`); analyzer-less primitive
            // methods stay on the `FunctionPath` form so they do not surface a
            // new panicking `SomeInstance.getattr`.
            None => {
                if last_idx < 2 {
                    return None;
                }
                let module_leaf = match &segs[last_idx - 2] {
                    NameSeg::Ident { ident: (s, _) } => s.as_str(),
                    _ => return None,
                };
                if !NON_ADT_OWNER_METHOD_ALLOWLIST
                    .iter()
                    .any(|&(m, f)| m == module_leaf && f == leaf)
                {
                    return None;
                }
                module_leaf.to_string()
            }
        };
        if owner_leaf.is_empty() {
            return None;
        }
        Some((owner_leaf, leaf))
    }

    /// `arr[i]` / `arr[i] = v` on a workspace fixed-array type —
    /// resolves to its `Index::index` / `IndexMut::index_mut` impl,
    /// whose body bottoms out at raw-slice construction
    /// (`as_mut_slice` → `from_raw_parts`) with no graph lowering.
    /// The structs are length-prefixed GcArray layouts (see
    /// `FixedObjectArray`), so the callsite IS RPython's
    /// getarrayitem/setarrayitem on the receiver and is devirtualized
    /// to `ArrayRead`/`ArrayWrite` by the caller.
    fn is_workspace_index_call(&self, reg: &RegularCall) -> bool {
        is_workspace_index_regular(reg, self.llbc)
    }

    /// `<[T]>::swap(s, a, b)` (`core::slice::<Impl>::swap`) — an
    /// in-place element exchange through a `&mut [T]`.  The slice base
    /// is the same `FixedObjectArray` shape `is_workspace_index_call`
    /// reads, so the callsite lowers to the getarrayitem/setarrayitem
    /// decomposition rather than residualizing the Opaque core body.
    fn is_slice_swap_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::slice::<Impl>::swap")
    }

    /// Pointer reinterprets `*const T::cast_mut` / `*mut T::cast_const`
    /// / `<ptr>::cast` — address-preserving const↔mut flips and pointee
    /// retypes.  The same i64 machine repr as the receiver, so the JIT
    /// models them as identity, the receiver-method twin of the
    /// `CastKind::RawPtr` Rvalue (`same_as`, [`cast_label_from_payload`]).
    /// Gating on a `RawPtr` self excludes unrelated inherent `cast`
    /// methods on non-pointer owners.
    fn is_ptr_identity_cast(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let leaf = fd
            .item_meta
            .name_path()
            .rsplit("::")
            .next()
            .unwrap_or("")
            .to_string();
        if !matches!(leaf.as_str(), "cast" | "cast_mut" | "cast_const") {
            return false;
        }
        fd.signature.inputs.first().is_some_and(|t| {
            tyref_node(t, self.llbc)
                .and_then(|n| strip_ty_wrappers(n, self.llbc))
                .and_then(serde_json::Value::as_object)
                .is_some_and(|o| o.contains_key("RawPtr"))
        })
    }

    /// `<core::sync::atomic::Atomic*>::load(&self, ordering)` — a relaxed
    /// read of a std atomic.  The atomic types are layout-transparent
    /// over their inner scalar/pointer (asserted for the `PyType`
    /// `subclassrange_*` / `instantiate` vtable fields), so the JIT
    /// models the load as that inner value: [`tyref_atomic_inner_value_type`]
    /// types the `Atomic*` field as its inner [`ValueType`], the `&self`
    /// `Ref` rvalue aliases to that field read, and this aliases the load
    /// destination to the receiver.  Gating on an atomic receiver
    /// excludes unrelated inherent `load` methods, and the method name
    /// never reaches the rtyper as a `ptr.getattr`.
    fn is_atomic_load(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let name = fd.item_meta.name_path();
        if name.rsplit("::").next() != Some("load") {
            return false;
        }
        fd.signature.inputs.first().is_some_and(|t| {
            adt_path_of_tyref(t, self.llbc).is_some_and(|p| {
                p.contains("::sync::atomic::")
                    && p.rsplit("::")
                        .next()
                        .is_some_and(|leaf| leaf.starts_with("Atomic"))
            })
        })
    }

    /// `<*mut T>::add` / `<*const T>::add` inside an `ItemsBlock`
    /// items-base accessor (`items_block_items_base` /
    /// `items_block_items_ptr`), whose whole body is
    /// `(block as *mut u8).add(ITEMS_BLOCK_ITEMS_OFFSET)` — the items
    /// pointer of a list's backing block.
    ///
    /// The runtime reads list items through a `gcarray` descr whose
    /// `base_size` already folds in the header offset
    /// (`pyobject_gcarray_descr` base_size = `ITEMS_BLOCK_ITEMS_OFFSET`),
    /// so the JIT's items base *is* the header pointer.  The accessor
    /// body therefore collapses to its receiver, dropping the
    /// `core::ptr::mut_ptr::<Impl>::add` call (no graph lowering, not a
    /// registered callee).
    ///
    /// The soundness boundary is the *enclosing accessor*, not the
    /// `.add` callee or its constant offset: a `.add(NAMED_OFFSET)`
    /// whose interior pointer is dereferenced in place — `runtime_ops`
    /// reading `*(p.add(PYFRAME_W_BUILTIN_OFFSET))`, or
    /// `w_tuple_getitem_known`'s `*base.add(idx)` — must keep its offset.
    /// Only an accessor that *returns* the interior pointer for
    /// descr-based consumption is safe to alias to its receiver.
    fn is_items_block_base_ptr_add(&self, reg: &RegularCall) -> bool {
        graph_is_items_block_base_accessor(&self.graph.name)
            && regular_call_is_ptr_add(reg, self.llbc)
    }

    /// `*items_block_items_base(block).add(idx)` — a list / tuple
    /// element load (`x = *p`) or store (`*p = v`) through an
    /// `ItemsBlock` items pointer.  brick 1 rewrites the accessor to
    /// return the block *header*, so the runtime reaches `items[idx]`
    /// through a `gcarray` descr whose `base_size` folds in
    /// `ITEMS_BLOCK_ITEMS_OFFSET`; this lowers the dereferenced `.add`
    /// as that getarrayitem / setarrayitem — the same decomposition as
    /// the workspace [`Lowering::is_workspace_index_call`] path, but
    /// reached through a raw `.add` rather than `Index::index`.
    ///
    /// Five conditions, cheapest first (so the body scans run only on
    /// genuine candidates): the callee is `<ptr>::add`; two arguments;
    /// the receiver is `*mut PyObjectRef`; the index is a runtime local
    /// (not the constant offset brick 1 collapses); the base traces to
    /// an items-base accessor ([`base_traces_to_items_block_accessor`])
    /// — so the header `base_size` re-add lands on `items[0]`; and the
    /// `.add` result is dereferenced exactly once and never escapes as a
    /// raw pointer ([`add_dest_used_only_as_single_deref`]).
    fn is_list_items_elem_ptr_add(
        &self,
        reg: &RegularCall,
        args_len: usize,
        arg_locals: &[Option<usize>],
        first_arg_ty: Option<&TyRef>,
        dest_local: usize,
    ) -> bool {
        is_list_items_elem_ptr_add_parts(
            reg,
            args_len,
            arg_locals.first().copied().flatten(),
            first_arg_ty,
            arg_locals.get(1).copied().flatten(),
            dest_local,
            self.body,
            self.llbc,
        )
    }

    /// Devirtualize a callsite of the blanket
    /// `impl<T, U: From<T>> Into<U> for T` (`core::convert::<Impl>::into`).
    ///
    /// The callsite's `generics.trait_refs` carries the resolved
    /// `U: From<T>` obligation as a trait ref whose `trait_decl_ref`
    /// names `core::convert::From` and whose `kind` is
    /// `TraitImpl { id }` — the def_id of the selected `impl From<T>
    /// for U`.  Two outcomes:
    ///
    /// - The obligation's decl-ref type args are equal (`T == U`):
    ///   the reflexive `impl<T> From<T> for T` was selected and the
    ///   whole conversion is a `T -> T` identity —
    ///   [`IntoDevirt::Identity`].
    /// - Otherwise the impl's `methods` table binds the single `From`
    ///   method to the concrete `from` FunDecl, whose path is the
    ///   devirtualized call target — [`IntoDevirt::Target`].  `from`
    ///   is an associated function (no `self` receiver), so the
    ///   caller must keep the `FunctionPath` shape (a
    ///   `CallTarget::Method` hint would bind the *argument* as a
    ///   receiver).
    ///
    /// Returns `None` (caller keeps the blanket-into path) when the
    /// obligation is unresolved (`kind` is a clause/builtin rather
    /// than `TraitImpl`) or any table lookup misses.
    /// `<*const T>::cast_mut` / `<*mut T>::cast_const` — pointer casts that
    /// change only const/mut, never the pointee type.  The JIT does not
    /// model the mut/const distinction (`Ref` / `RawPtr` lower to a
    /// same-Variable alias, mir.rs:50), so a `p.cast_mut()` callsite binds
    /// its destination straight to the pointer argument instead of
    /// emitting a call to core's raw-pointer method (which has no graph
    /// lowering and is not a registered callee).
    ///
    /// The pointee-changing `<ptr>::cast::<U>()` is NOT matched here: it
    /// routes to [`is_ptr_identity_cast`], which narrows a
    /// `cast::<RegisteredStruct>()` to `SomeInstance(root)` (the same
    /// `cast_pointer` shape as `obj as *const W_Foo`) instead of leaving
    /// the destination classdef-less and blocking the downstream getattr.
    fn is_noop_ptr_cast(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::ptr::const_ptr::<Impl>::cast_mut" | "core::ptr::mut_ptr::<Impl>::cast_const"
            )
        })
    }

    /// A thin-pointer `Deref::deref` / `DerefMut::deref_mut` whose
    /// dereferenced `&T` is a registered struct, resolved to the pointee
    /// struct root.  Valid only when the handle's single pointer word *is*
    /// the pointee address, so `*p` is a zero-offset reinterpret: `Box<T>`
    /// (no header — `Unique<T>` points straight at `T`), the workspace
    /// `FrameBox` (`{ptr: *mut PyFrame}`), and single-field transparent
    /// wrappers (`{UnsafeCell<T>}`).  For these the caller lowers the
    /// deref to the `cast_pointer(T, p)` downcast marker
    /// (`cast_pointer_marker_op`), which yields `SomeInstance(T)`
    /// (lltype.py:964-974) independent of the receiver's (classdef-less)
    /// annotation — the same shape pyre emits for `obj as *const W_Foo`.
    ///
    /// `Rc<T>` / `Arc<T>` are the exception: their word points at a
    /// refcount header, so the pointee sits at a non-zero offset and
    /// `cast_pointer` would reinterpret the header.  They are subtracted
    /// by owner-type leaf; an unresolved owner keeps the ordinary
    /// thin-pointer treatment, since the dereferenced `&T` must still
    /// resolve a registered struct root below.
    ///
    /// Returns `None` when the call is not a `deref` / `deref_mut` leaf or
    /// the dereferenced type is not a named ADT (slice / `str` derefs
    /// resolve no struct root — their `&[T]` / `&str` value model is the
    /// receiver's, narrowed by the list / string reprs, not a pointer
    /// downcast).
    fn deref_cast_root(&self, reg: &RegularCall, dest_ty: &TyRef) -> Option<String> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        let fd = self.llbc.fn_by_id(*id)?;
        let np = fd.item_meta.name_path();
        if !(np.ends_with("::deref") || np.ends_with("::deref_mut")) {
            return None;
        }
        if let Some(leaf) = deref_impl_owner_leaf(self.llbc, fd) {
            if matches!(leaf.as_str(), "Rc" | "Arc") {
                return None;
            }
        }
        tyref_class_root(dest_ty, self.llbc)
    }

    /// `<String as Deref>::deref(&self) -> &str` / `<Vec<T> as
    /// Deref>::deref(&self) -> &[T]` (and their `deref_mut`).  Unlike the
    /// `Box`/`FrameBox` handles [`Self::deref_cast_root`] reinterprets as a
    /// typed struct pointer, the owning-container deref returns the same
    /// value the lifted model already carries: Rust `String`/`&str` both
    /// lower to the immutable rpy_string and `Vec<T>`/`&[T]` both to the
    /// rpy_list, so `*s` is identity.  `deref_cast_root` returns `None`
    /// here (a `str`/slice target has no class root), so without this the
    /// callsite falls through to a `CallTarget::Method` `deref` getattr the
    /// rtyper cannot route on the classdef-less receiver.  Bind the
    /// destination to the argument instead, the same alias shape as the
    /// blanket-`into` identity above.
    fn is_container_identity_deref(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let np = fd.item_meta.name_path();
        if !(np.ends_with("::deref") || np.ends_with("::deref_mut")) {
            return false;
        }
        deref_impl_owner_leaf(self.llbc, fd)
            .is_some_and(|leaf| matches!(leaf.as_str(), "String" | "Vec"))
    }

    /// `<String as AsRef<str>>::as_ref(&self) -> &str`,
    /// `String::as_str(&self) -> &str`, and
    /// `<String as Borrow<str>>::borrow(&self) -> &str` — every one
    /// returns a `&str` view of the same string, an identity in the
    /// lifted value model (Rust `String`/`&str`/`str` all lower to the
    /// immutable rpy_string).  Without the intercept the call keeps a
    /// `CallTarget::Method` `as_ref` getattr the rtyper cannot route on
    /// the classdef-less string receiver (the `Cannot find attribute
    /// "as_ref" on UnicodeString` wall).  Bind the destination to the
    /// receiver instead, the same alias shape as the container deref
    /// above.  Gated on the `str`-typed result so `String`'s sibling
    /// `AsRef<[u8]>` / `AsRef<OsStr>` / `AsRef<Path>` impls — whose
    /// `&[u8]`/etc. result is a *different* value-model family — keep
    /// their ordinary lowering.
    fn is_string_to_str_identity(&self, reg: &RegularCall, dest_ty: &TyRef) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        let np = fd.item_meta.name_path();
        let Some(leaf) = np.rsplit("::").next() else {
            return false;
        };
        if !matches!(leaf, "as_ref" | "as_str" | "borrow") {
            return false;
        }
        if deref_impl_owner_leaf(self.llbc, fd).as_deref() != Some("String") {
            return false;
        }
        tyref_strips_to_str(dest_ty, self.llbc)
    }

    /// `<str as ToString>::to_string` / `<String as ToString>::to_string`
    /// (`alloc::string::<Impl>::to_string`) — a `String` clone of a value
    /// that is already a string in the lifted model (`String`/`&str`/`str`
    /// all lower to the immutable rpy_string), so it is an identity.  The
    /// path alone cannot tell the string specialization from the blanket
    /// `impl<T: Display> ToString for T` (both monomorphize to the same
    /// `<Impl>::to_string`), so gate on the receiver type: fold only when
    /// the argument is itself string-family.  A non-string `to_string`
    /// (an integer's `ll_int2dec`) has a receiver outside the family and
    /// keeps its ordinary lowering.
    fn is_to_string_identity(&self, reg: &RegularCall, first_arg_ty: Option<&TyRef>) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return false;
        };
        if !fd.item_meta.name_path().ends_with("::to_string") {
            return false;
        }
        first_arg_ty.is_some_and(|ty| {
            tyref_is_string_adt(ty, self.llbc) || tyref_strips_to_str(ty, self.llbc)
        })
    }

    /// `alloc::fmt::format(args)` (the `format!` macro's String producer)
    /// — the call whose `format_args!` argument the #277 recognizer
    /// back-traces.  `write!`/`writeln!` lower to `fmt::Write::write_fmt`
    /// (a `Result`), never `fmt::format`, so the two-segment tail keeps
    /// the match precise.
    fn is_fmt_format_call(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            let np = fd.item_meta.name_path();
            let segs: Vec<String> = np.split("::").map(str::to_string).collect();
            fmt_path_ends_with(&segs, &["fmt", "format"])
        })
    }

    /// `fmt::Arguments::from_str(s)` — the no-placeholder `format_args!`
    /// constructor a constant message lowers to (`debug_assert!` /
    /// `assert!` / `panic!` with a literal, and `format!("literal")`).
    /// Its sole argument is the `&'static str` literal itself (an already
    /// string-family value in the lifted model), so the `Arguments` is the
    /// constant string: the caller aliases the destination to that operand
    /// instead of emitting the unregistered ctor.  The result then flows
    /// either to a residualized panic (`Abort` → `RaiseImplicit`) or to
    /// `alloc::fmt::format` (handled by [`Self::traces_to_str_const`]).
    fn is_arguments_from_str(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        // `from_str` is an inherent-impl associated function on
        // `core::fmt::Arguments`, so its `name_path()` carries an `<Impl>`
        // placeholder, not the owner name — resolve the owner the same way
        // `call_target_segments` spells the `["fmt", "Arguments",
        // "from_str"]` `FunctionPath` (via `impl_method_owner_for_fundecl`).
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            impl_method_owner_for_fundecl(self.llbc, fd).is_some_and(|(owner, leaf)| {
                leaf == "from_str" && owner.ends_with("fmt::Arguments")
            })
        })
    }

    /// Whether `var` is produced by a `__str_const` synthetic call (a
    /// string literal — see [`Lowering::emit_constant`]), following the
    /// cross-block back-trace.  Used to recognize `alloc::fmt::format`
    /// applied to a constant message (a folded `Arguments::from_str`),
    /// whose render is the constant itself.
    fn traces_to_str_const(&self, var: &Variable) -> bool {
        use crate::model::{CallTarget, OpKind};
        resolve_to_producer_op(&self.graph, var)
            .and_then(|(b, i)| {
                self.graph
                    .blocks
                    .iter()
                    .find(|blk| blk.id == b)
                    .and_then(|blk| blk.operations.get(i))
                    .map(|op| &op.kind)
            })
            .is_some_and(|kind| {
                matches!(
                    kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments },
                        ..
                    } if segments.first().map(String::as_str) == Some("__str_const")
                )
            })
    }

    /// The blanket `impl<I: Iterator> IntoIterator for I`
    /// (`core::iter::traits::collect`) — its `into_iter(self) -> I`
    /// returns the receiver unchanged, so a `for` desugar's `into_iter`
    /// callsite aliases its argument instead of calling core's identity
    /// body (an unregistered callee).  Container impls
    /// (`Vec`/array/`Range`) live under other module paths, so the exact
    /// path match selects only the reflexive blanket.
    fn is_reflexive_into_iter(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            fd.item_meta.name_path() == "core::iter::traits::collect::<Impl>::into_iter"
        })
    }

    /// The concrete container `IntoIterator` impls — `<&[T] as
    /// IntoIterator>::into_iter` and the `Vec` / array forms.  Unlike the
    /// reflexive blanket (`is_reflexive_into_iter`), the receiver type
    /// (the container) differs from the destination (a fresh iterator), so
    /// they cannot be identity-aliased; the caller lowers them to the
    /// `iter` operation on the container instead of an unregistered
    /// `FunctionPath` callee.  The explicit `<[T]>::iter` already routes to
    /// `iter` through the `("slice","iter")` bridge
    /// (`flowspace_adapter::nonraising_core_bridge_opname`), so it is not
    /// matched here.
    fn is_concrete_iter_constructor(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::slice::iter::<Impl>::into_iter"
                    | "alloc::vec::<Impl>::into_iter"
                    | "core::array::<Impl>::into_iter"
            )
        })
    }

    /// `<[T]>::len` / `Vec::len` — the container element count.  Lowered
    /// to the `__len` operation so the receiver's list annotation
    /// supplies the length through the rtyper's `len` op, the same
    /// routing [`is_concrete_iter_constructor`] gives the container
    /// `iter`.
    fn is_container_len(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::slice::<Impl>::len" | "alloc::vec::<Impl>::len"
            )
        })
    }

    /// `Vec::as_slice` / `<[T]>::as_slice` — a borrowed slice view of the
    /// same elements.  Identity on the list model, so the callsite aliases
    /// its receiver instead of leaving the unregistered method callee.
    fn is_container_as_slice(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc.fn_by_id(*id).is_some_and(|fd| {
            matches!(
                fd.item_meta.name_path().as_str(),
                "core::slice::<Impl>::as_slice" | "alloc::vec::<Impl>::as_slice"
            )
        })
    }

    /// `core::hint::must_use(value)` — the identity wrapper
    /// (`pub const fn must_use<T>(value: T) -> T`) the compiler inserts to
    /// carry an `unused_must_use` lint through macro-generated code.  It
    /// returns its argument unchanged and `core` has no graph body for it
    /// (an unregistered callee), so the callsite aliases its argument
    /// instead of calling the missing identity body.
    fn is_hint_must_use(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::hint::must_use")
    }

    /// `f64::is_nan(self)` — `core` has no graph body (Opaque), so the
    /// callsite would skip as an unregistered `FunctionPath`.  `is_nan`
    /// is `value != value` (`rfloat.isnan`), so the caller lowers it to
    /// a reflexive `ne` `BinOp`; the float operand makes the rtyper pick
    /// `float_ne`, which (unlike `int_ne`) carries no `n(x, x) => 0`
    /// reflexive fold, preserving the NaN-only truth value.
    fn is_f64_is_nan(&self, reg: &RegularCall) -> bool {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return false;
        };
        self.llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::f64::<Impl>::is_nan")
    }

    /// `majit_metainterp::jit::promote(x)` = `hint(x, promote=True)`
    /// (`rlib/jit.py:101`), with the `promote_string` (`:118`) and
    /// `promote_unicode` (`:124`) siblings.  All three wrappers carry their
    /// flag by name (each body is a bare `hint(x)`), so the callsite is
    /// recognised by the wrapper path, not the body.  Returns the
    /// synthesised `hint_*` marker leaf for the matched wrapper so the
    /// residual `OpKind::Call` reaches the matching `jtransform`
    /// `rewrite_op_hint` arm.
    ///
    /// `promote_string`/`promote_unicode` route to their own
    /// `hint_promote_string`/`hint_promote_unicode` markers (preserving the
    /// upstream hint-kind distinction in the IR) even though `jtransform`
    /// lowers all three through the same `<kind>_guard_value` family: pyre
    /// interpreter strings are `W_UnicodeObject` GC refs, not
    /// `Ptr(rstr.STR)`, so the string `guard_value` collapses to the
    /// ref-kind `r_guard_value` (see `jtransform::rewrite_op_hint`).
    fn jit_promote_marker(&self, reg: &RegularCall) -> Option<&'static str> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        match self.llbc.fn_by_id(*id)?.item_meta.name_path().as_str() {
            "majit_metainterp::jit::promote" => Some("hint_promote"),
            "majit_metainterp::jit::promote_string" => Some("hint_promote_string"),
            "majit_metainterp::jit::promote_unicode" => Some("hint_promote_unicode"),
            _ => None,
        }
    }

    fn blanket_into_devirt(&self, reg: &RegularCall) -> Option<IntoDevirt> {
        let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
            return None;
        };
        let is_blanket_into = self
            .llbc
            .fn_by_id(*id)
            .is_some_and(|fd| fd.item_meta.name_path() == "core::convert::<Impl>::into");
        if !is_blanket_into {
            return None;
        }
        let trait_refs = reg.generics.get("trait_refs")?.as_array()?;
        for tref in trait_refs {
            let Some(tref) = traitref_unwrap(tref, self.llbc, 0) else {
                continue;
            };
            let Some(decl) = tref
                .get("trait_decl_ref")
                .and_then(|d| d.get("skip_binder"))
            else {
                continue;
            };
            let Some(decl_id) = decl.get("id").and_then(serde_json::Value::as_u64) else {
                continue;
            };
            let is_from = self
                .llbc
                .trait_by_id(decl_id)
                .is_some_and(|td| td.item_meta.name_path() == "core::convert::From");
            if !is_from {
                continue;
            }
            // `U: From<T>` decl-ref generics carry `[U, T]`; equal
            // args select the reflexive blanket impl.  Compare by
            // hash-cons id so an inline `HashConsedValue: [id, …]`
            // matches its `Deduplicated: id` reference.
            let types = decl.get("generics")?.get("types")?.as_array()?;
            if types.len() == 2 {
                let reflexive = match (ty_dedup_key(&types[0]), ty_dedup_key(&types[1])) {
                    (Some(a), Some(b)) => a == b,
                    _ => types[0] == types[1],
                };
                if reflexive {
                    return Some(IntoDevirt::Identity);
                }
            }
            let impl_id = traitref_impl_id(tref, self.llbc, 0)?;
            let ti = self.llbc.trait_impls_raw().get(impl_id as usize)?;
            let fn_id = ti.get("methods")?.as_array()?.iter().find_map(|m| {
                let tm = m.get("kind")?.get("TraitMethod")?.as_array()?;
                if tm.first()?.as_u64()? != decl_id {
                    return None;
                }
                m.get("skip_binder")?.get("id")?.as_u64()
            })?;
            let fd = self.llbc.fn_by_id(fn_id)?;
            let segments: Vec<String> = fd
                .item_meta
                .name_path()
                .split("::")
                .map(|s| s.to_string())
                .collect();
            return Some(IntoDevirt::Target(segments));
        }
        None
    }

    /// `msg.into()` on a generic parameter bound `T: Into<String>` —
    /// a `CallKind::Trait` whose trait ref is a *clause* (no resolved
    /// impl for [`Self::blanket_into_devirt`] to read).  The blanket
    /// `impl<T, U: From<T>> Into<U> for T` makes the result
    /// `U::from(self)`; for a string-family target the conversion is
    /// identity in the lifted value model (Rust `String` and `&str`
    /// both lower to the immutable rpy_string), so the caller may
    /// alias the destination to the argument.  The callsite's `dest`
    /// type *is* the trait ref's target type argument, so it is the
    /// only payload field consulted besides the trait identity.
    fn trait_clause_into_string_identity(&self, reg: &RegularCall, dest_ty: &TyRef) -> bool {
        let CallKind::Trait(v) = &reg.kind else {
            return false;
        };
        let Some(traitref) = v.as_array().and_then(|a| a.first()) else {
            return false;
        };
        let Some(trait_id) = traitref_decl_id(traitref, self.llbc, 0) else {
            return false;
        };
        let is_into = self
            .llbc
            .trait_by_id(trait_id)
            .is_some_and(|td| td.item_meta.name_path() == "core::convert::Into");
        is_into && tyref_is_string_adt(dest_ty, self.llbc)
    }

    /// Whether the FunDecl's first signature input is the given ADT,
    /// possibly behind reference / raw-pointer layers (`self: Owner`,
    /// `&Owner`, `&mut Owner`, `*const/mut Owner`).  Distinguishes a
    /// real method from an associated function of the same impl block.
    fn first_input_is_adt(&self, fd: &FunDecl, adt_def_id: u64) -> bool {
        let Some(first) = fd.signature.inputs.first() else {
            return false;
        };
        let Some(mut v) = self.tyref_body(first) else {
            return false;
        };
        loop {
            let Some(obj) = v.as_object() else {
                return false;
            };
            if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
                match self.llbc.dedup_body(id) {
                    Some(b) => {
                        v = b;
                        continue;
                    }
                    None => return false,
                }
            }
            if let Some(arr) = obj
                .get("HashConsedValue")
                .and_then(serde_json::Value::as_array)
                && arr.len() == 2
            {
                v = &arr[1];
                continue;
            }
            // `{"Ref": [region, ty, kind]}` / `{"RawPtr": [ty, kind]}`.
            if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
                match arr.get(1) {
                    Some(inner) => {
                        v = inner;
                        continue;
                    }
                    None => return false,
                }
            }
            if let Some(arr) = obj.get("RawPtr").and_then(serde_json::Value::as_array) {
                match arr.first() {
                    Some(inner) => {
                        v = inner;
                        continue;
                    }
                    None => return false,
                }
            }
            return inline_adt_def_id(v) == Some(adt_def_id);
        }
    }

    /// Decode the receiver type's ADT `def_id` from an `Impl` NameSeg
    /// payload.  Two shapes:
    ///
    /// - **InherentImplBlock**: `{"Ty": {"skip_binder": <TyExpr>}}` where
    ///   `<TyExpr>` is the type expression of `Self` in the impl block.
    ///   It can be inline (`HashConsedValue: [id, body]`) or
    ///   deduplicated (`Deduplicated: id`).  When inline, the body
    ///   carries the ADT def_id directly (`{"Adt": {"id": {"Adt": <def_id>}}}`);
    ///   when deduplicated, we consult [`Self::dedup_to_adt_def_id`]
    ///   which lazy-builds a per-LLBC `dedup_id → adt_def_id` index
    ///   from the inline forms scattered across the LLBC.
    ///
    /// - **TraitImplBlock**: `{"Trait": <trait_impl_id>}` — indirect
    ///   through the opaque `trait_impls` array to find the impl's
    ///   first concrete type argument, then resolve through the same
    ///   inline-or-dedup path.
    fn resolve_impl_owner_adt_def_id(&self, impl_payload: &serde_json::Value) -> Option<u64> {
        let obj = impl_payload.as_object()?;
        if let Some(ty) = obj.get("Ty") {
            let sb = ty.as_object()?.get("skip_binder")?;
            return self.resolve_tyexpr_to_adt_def_id(sb);
        }
        if let Some(trait_impl_id) = obj.get("Trait").and_then(serde_json::Value::as_u64) {
            let trait_impls = self.llbc.trait_impls_raw();
            let ti = trait_impls.get(trait_impl_id as usize)?;
            let first_ty = ti
                .as_object()?
                .get("impl_trait")?
                .as_object()?
                .get("generics")?
                .as_object()?
                .get("types")?
                .as_array()?
                .first()?;
            return self.resolve_tyexpr_to_adt_def_id(first_ty);
        }
        None
    }

    /// Resolve a Charon type expression to the underlying ADT
    /// `def_id`, whether the expression is inline
    /// (`HashConsedValue: [_, body]`) or deduplicated
    /// (`Deduplicated: id`).  Returns `None` for non-ADT shapes
    /// (primitives, references, tuples).
    fn resolve_tyexpr_to_adt_def_id(&self, ty: &serde_json::Value) -> Option<u64> {
        let obj = ty.as_object()?;
        if let Some(arr) = obj
            .get("HashConsedValue")
            .and_then(serde_json::Value::as_array)
            && let Some(body) = arr.get(1)
        {
            return inline_adt_def_id(body);
        }
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            return self.llbc.dedup_to_adt_def_id(id);
        }
        None
    }

    /// Alias `Arg::<T>::get(self, arg: OpArg) -> T`
    /// (`rustpython_compiler_core::bytecode::instruction`, instruction.rs:1286)
    /// to its `OpArg` argument.  `Arg<T>` is the zero-sized oparg marker
    /// ([`tyref_is_bytecode_arg_marker`]) and `OpArg` is the transparent
    /// `struct OpArg(u32)` newtype, both modeled as a plain integer (the
    /// raw operand) — see [`tyref_is_oparg`].
    /// The body is `T::try_from(u32::from(arg)).unwrap()`, and every
    /// `OpArgType` (`VarNum` / `VarNums` / `u32`) is a transparent newtype
    /// over `u32`, so the result's bits equal `u32::from(arg)` — the
    /// argument's own integer value.  Returning `arg` (`args[1]`) directly
    /// is the identity the marker `self` is dropped around.
    ///
    /// Without this the `.get` method call lowers to `getattr(Integer,
    /// "get")` on the integer `self` marker — an attribute the annotator
    /// cannot type (the dominant `Cannot find attribute "get" on Integer`
    /// wall behind `complete_pending_blocks` for the opcode-dispatch
    /// handler family, which all read their operand via `<field>.get(op_arg)`).
    ///
    /// Identified by the second input being the `oparg::OpArg` type: no
    /// other `.get` (slice / `Vec` / map) takes an `OpArg`, and the
    /// concrete `OpArg` resolves robustly where the generic `self: Arg<T>`
    /// marker would not.  Returns `None` for any other `.get` so those
    /// keep the `Call` form.
    fn oparg_arg_get_alias(
        &self,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        if segments.last().map(String::as_str) != Some("get") {
            return None;
        }
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return None;
        };
        let fd = self.llbc.fn_by_id(*id)?;
        // The `OpArg` argument (`inputs[1]` — `args[1]`, after the
        // `self` receiver) is the operand value.
        let oparg_ty = fd.signature.inputs.get(1)?;
        // `OpArg` is `Opaque` in the LLBC (external crate), so resolve it
        // by qualified name rather than structural shape.
        if !adt_path_of_tyref(oparg_ty, self.llbc).is_some_and(|p| p.ends_with("oparg::OpArg")) {
            return None;
        }
        // A fieldless-enum result (`Arg::<SpecialMethod>::get`) must keep
        // its `Ref` enum shape.  Aliasing to the bare integer operand
        // would make the downstream `match`'s `Rvalue::Discriminant` read
        // a `__discriminant` field off an integer base — a
        // `getfield_gc_i_pure/id>i` opname no blackhole handler covers.
        // The generic `fd.signature.output` is the type parameter `T`, so
        // key off the call's concrete destination type instead.  Only the
        // int-newtype results (`VarNum` / `u32`, whose bits *are* the
        // operand) alias; the enum keeps the canonical `…/rd>i` read.
        if self.tyref_is_fieldless_enum(dest_ty) {
            return None;
        }
        args.get(1).cloned()
    }

    /// Alias the transparent `OpArgType` conversions to their single
    /// argument.  Each oparg newtype is `#[repr(transparent)]` over
    /// `u32` and the fieldless oparg enums carry their operand as the
    /// discriminant, so — once the operand is modeled as an integer
    /// ([`tyref_is_oparg`], [`oparg_arg_get_alias`]) — every conversion
    /// below is the identity on that integer:
    ///   * the inherent `as_u32` / `as_usize` extractors and the
    ///     `from_u32` constructor (`newtype_oparg!`), and
    ///   * the `From` conversions (`u32::from(oparg)` for a newtype, the
    ///     `__discriminant` read for a fieldless enum).
    /// Gated on the defining impl living in `bytecode::oparg` so the
    /// generic `from` / `as_u32` / `as_usize` names cannot match
    /// unrelated types.
    fn oparg_value_alias(&self, segments: &[String], args: &[Variable]) -> Option<Variable> {
        let [arg] = args else {
            return None;
        };
        let [first, .., module, impl_seg, leaf] = segments else {
            return None;
        };
        if first.as_str() != "rustpython_compiler_core"
            || module.as_str() != "oparg"
            || impl_seg.as_str() != "<Impl>"
            || !matches!(leaf.as_str(), "as_u32" | "as_usize" | "from_u32" | "from")
        {
            return None;
        }
        Some(arg.clone())
    }

    /// Identity-passthrough stdlib wrappers that return their sole
    /// argument unchanged: `core::hint::must_use` (the `#[must_use]` lint
    /// shim, `must_use<T>(x: T) -> T { x }`) and `core::convert::identity`.
    /// Aliasing the destination to the argument keeps the result kind
    /// following the operand — both are generic over `T` — where a
    /// residual `Call` Skips "not registered" and a fixed-type stub would
    /// mistype a non-ref argument.  Mirrors the `Rvalue::Use` no-copy
    /// alias.  `must_use` is the immediate consumer of every
    /// `alloc::fmt::format` result, so leaving it residual blocks the
    /// whole error-message helper from rtyping.
    fn identity_passthrough_alias(
        &self,
        segments: &[String],
        args: &[Variable],
    ) -> Option<Variable> {
        let [arg] = args else {
            return None;
        };
        // `fmt::Arguments::from_str(s)` is the no-placeholder `format_args!`
        // — the constructed `Arguments` renders to exactly its `&str`
        // argument, so the rendered value *is* that string.  Alias it to
        // `s`, dropping the graph-less `Arguments::from_str` extern: the
        // literal-message chains it heads dead-end in synthetic panic
        // `Option`s pyre replaces with host exception constants, or thread
        // into `Write::write_fmt` (whose own graph-less extern keeps those
        // subjects residual) — no `alloc::fmt::format` consumes one.  This
        // is the same transparent passthrough `must_use` / `identity` take.
        if fmt_path_ends_with(segments, &["Arguments", "from_str"]) {
            return Some(arg.clone());
        }
        let [a, b, c] = segments else {
            return None;
        };
        let is_identity = a == "core"
            && ((b == "hint" && c == "must_use") || (b == "convert" && c == "identity"));
        is_identity.then(|| arg.clone())
    }

    /// The ADT `def_id` behind a signature [`TyRef`], whether inline
    /// or routed through the dedup table.  `None` for non-ADT shapes.
    fn tyref_adt_def_id(&self, ty: &TyRef) -> Option<u64> {
        match ty {
            TyRef::Inline { value: (_, v) } | TyRef::Other(v) => inline_adt_def_id(v),
            TyRef::Dedup { id } => self.llbc.dedup_to_adt_def_id(*id),
        }
    }

    /// `true` when `ty` resolves to a fieldless (C-like) enum — at least
    /// one variant and every variant carrying zero payload fields.  Such
    /// an enum is represented by-value as its discriminant integer, so
    /// `Rvalue::Discriminant` on it is the identity on that value (see the
    /// `Discriminant` lowering).  Payload-carrying enums return `false`
    /// and keep the `__discriminant` field read against their aggregate
    /// `Ref` base.
    fn tyref_is_fieldless_enum(&self, ty: &TyRef) -> bool {
        let Some(def_id) = self.tyref_adt_def_id(ty) else {
            return false;
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return false;
        };
        match &td.kind {
            TypeDeclKind::Enum(variants) => {
                !variants.is_empty() && variants.iter().all(|v| v.fields.is_empty())
            }
            _ => false,
        }
    }

    /// Lower `i64::checked_neg()` (`core::num::<Impl>::checked_neg` —
    /// core fn bodies are Opaque in the LLBC, so the `Call` form is
    /// permanently unliftable) to a decomposed ovfcheck shape.
    /// Upstream `translator/simplify.py:70-108 transform_ovfcheck`
    /// rewrites `ovfcheck(-x)` into the op's `_ovf` variant
    /// (`flowspace/operation.py:195-200 ovfchecked`, registered by
    /// `operation.py:466 add_operator('neg', ..., ovf=True)`), whose
    /// OverflowError edge the caller branches on.  Rust spells that
    /// ovfcheck as `checked_neg()` + a `Some`/`None` match, so the
    /// equivalent decomposition writes the destination `Option<i64>`
    /// as a synthetic aggregate (the same transparent-ctor +
    /// `FieldWrite` chain `Rvalue::Aggregate` emits):
    /// `__discriminant = ne(v, i64::MIN)` — negation overflows only
    /// at `i64::MIN`, and `Option`'s `None`/`Some` tags are 0/1 — and
    /// payload `__pos_0 = neg(v)` (wrapping; the `None` arm never
    /// reads it).  The downstream discriminant switch and payload
    /// downcast then lower through the ordinary enum FieldRead paths.
    /// The `i64::MIN` sentinel is only correct at word width, so the
    /// lowering is gated on a word-sized signed operand (`i64`/`isize`)
    /// — a narrower `checked_neg` overflows at its own narrower `MIN`
    /// and keeps the generic `Call` form (none arise today; the live
    /// callers are `neg`'s `int_value` and `functional`'s `step`,
    /// both `i64`).
    /// Returns `Ok(false)` when the call is not `checked_neg` (or the
    /// destination's `Option` decl cannot be resolved) so the generic
    /// `Call` lowering proceeds.
    fn try_lower_checked_neg(
        &mut self,
        mir_bb: usize,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return Ok(false);
        };
        if first.as_str() != "core"
            || module.as_str() != "num"
            || impl_seg.as_str() != "<Impl>"
            || leaf.as_str() != "checked_neg"
        {
            return Ok(false);
        }
        let [arg] = args else {
            return Ok(false);
        };
        // The decomposition compares against `i64::MIN`; restrict it to
        // word-sized signed operands so a narrower `checked_neg` (which
        // overflows at its own `MIN`) is not miscompiled.  The operand
        // is the receiver — `checked_neg(self)` — so read `inputs[0]`.
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        let Some(src) = fd.signature.inputs.first() else {
            return Ok(false);
        };
        if !matches!(self.tyref_literal_int_atom(src), Some("I64" | "Isize")) {
            return Ok(false);
        }
        // Resolve the destination `Option` decl so the FieldWrite owner
        // matches what `resolve_aggregate_adt` would record for a real
        // `Some(..)` construction site of the same type.
        let Some(def_id) = self.tyref_adt_def_id(dest_ty) else {
            return Ok(false);
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return Ok(false);
        };
        // Same owner-path / ctor-leaf split `resolve_aggregate_adt`
        // performs, so the ctor target and FieldWrite owner carry the
        // spellings the rest of the aggregate machinery expects.
        let owner = td.item_meta.name_path();
        let arg = arg.clone();
        let bb_id = self.block_id[mir_bb];

        let push_op = |graph: &mut FunctionGraph, kind: OpKind| {
            let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(res.clone()),
                kind,
            });
            res
        };
        let payload = push_op(
            &mut self.graph,
            OpKind::UnaryOp {
                op: "neg".to_string(),
                operand: arg.clone(),
                result_ty: ValueType::Int,
            },
        );
        let min = push_op(&mut self.graph, OpKind::ConstInt(i64::MIN));
        let disc = push_op(
            &mut self.graph,
            OpKind::BinOp {
                op: "ne".to_string(),
                lhs: arg,
                rhs: min,
                result_ty: ValueType::Int,
            },
        );
        // Success (`arg != MIN`) is the `Some` variant (discriminant 1).
        let payload_owner =
            Self::tagged_pair_payload_owner(td, &owner, 1).unwrap_or_else(|| owner.clone());
        self.emit_tagged_pair_aggregate(
            mir_bb,
            &owner,
            &payload_owner,
            disc,
            payload,
            dest_local,
            target,
        )?;
        Ok(true)
    }

    /// Lower the infallible `usize::try_from(<u8|u16|u32>)`
    /// (`core::convert::num::ptr_try_from_impls::<Impl>::try_from`,
    /// Opaque in the LLBC like every core fn) to its decomposed
    /// always-`Ok` shape: `__discriminant = 0` (`Result`'s `Ok` tag)
    /// and `__pos_0 = arg` (the widening is an identity on the i64
    /// carrier).  Upstream `rarithmetic.py:140-145 widen` performs
    /// the same smaller-than-word unsigned → Signed widening as a
    /// no-op; pyre spells it `usize::try_from(x).expect(..)`
    /// (`pyopcode.rs` `u32_as_usize` / `raise_kind_as_usize`), whose
    /// `Err` arm is statically dead on the 64-bit-only targets pyre
    /// supports.  Impls with word-sized-or-wider inputs — the
    /// genuinely fallible directions of the same impl group — keep
    /// the `Call` form.
    fn try_lower_usize_try_from(
        &mut self,
        mir_bb: usize,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return Ok(false);
        };
        if first.as_str() != "core"
            || module.as_str() != "ptr_try_from_impls"
            || impl_seg.as_str() != "<Impl>"
            || leaf.as_str() != "try_from"
        {
            return Ok(false);
        }
        let [arg] = args else {
            return Ok(false);
        };
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        let Some(src) = fd.signature.inputs.first() else {
            return Ok(false);
        };
        if !matches!(
            self.tyref_literal_uint_atom(src),
            Some("U8" | "U16" | "U32")
        ) {
            return Ok(false);
        }
        let Some(def_id) = self.tyref_adt_def_id(dest_ty) else {
            return Ok(false);
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return Ok(false);
        };
        let owner = td.item_meta.name_path();
        let arg = arg.clone();
        if self.multi_assigned_locals.contains(&dest_local) {
            // A re-bindable local may later carry a runtime `Result`,
            // so the constant tag can't be recorded and consumers
            // can't be folded — materialize the aggregate so field
            // reads on the local stay type-consistent.
            let bb_id = self.block_id[mir_bb];
            let disc = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(disc.clone()),
                kind: OpKind::ConstInt(0),
            });
            // Always-`Ok`: the payload belongs to the `Ok` variant (tag 0).
            let payload_owner =
                Self::tagged_pair_payload_owner(td, &owner, 0).unwrap_or_else(|| owner.clone());
            self.emit_tagged_pair_aggregate(
                mir_bb,
                &owner,
                &payload_owner,
                disc,
                arg,
                dest_local,
                target,
            )?;
            return Ok(true);
        }
        // Identity widening: bind the destination local directly to
        // the operand — no `Result` object is materialized at all,
        // exactly as upstream `widen` leaves no op in the graph.
        // The discriminant switch always sits in a successor block
        // (a MIR call terminates its own block), so record the
        // constant tag per-local: `Rvalue::Discriminant` folds to
        // `ConstInt(0)` and `expect`/`unwrap` aliases the payload
        // (`expect_on_const_ok`) without touching the variable.
        self.const_discriminant_locals.insert(dest_local, 0);
        self.local_var[dest_local] = Some(arg);
        let bb_id = self.block_id[mir_bb];
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// Lower an infallible numeric widening `<i64 as From<u32>>::from(x)`
    /// (`core::convert::num::<Impl>::from`, Opaque in the LLBC) to an
    /// identity bind on the destination local.  `From` is implemented in
    /// core only for value-preserving conversions, and a
    /// smaller-than-word unsigned source widens into the word-sized
    /// carrier with no change of value — the same no-op `rarithmetic.py
    /// widen` performs — so no op is emitted, exactly as the always-`Ok`
    /// `usize::try_from` payload binds directly
    /// ([`Lowering::try_lower_usize_try_from`]).  `pyopcode.rs`
    /// `u32_as_i64` is `i64::from(x: u32)`.  Word-or-wider sources keep
    /// the `Call` form (no smaller-than-word identity to fold).
    fn try_lower_num_from(
        &mut self,
        mir_bb: usize,
        kind: &CallKind,
        segments: &[String],
        args: &[Variable],
        dest_local: usize,
        dest_ty: &TyRef,
        target: usize,
    ) -> Result<bool, LowerError> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return Ok(false);
        };
        if first.as_str() != "core"
            || module.as_str() != "num"
            || impl_seg.as_str() != "<Impl>"
            || leaf.as_str() != "from"
        {
            return Ok(false);
        }
        let [arg] = args else {
            return Ok(false);
        };
        let CallKind::Fun(FunId::Regular { id }) = kind else {
            return Ok(false);
        };
        let Some(fd) = self.llbc.fn_by_id(*id) else {
            return Ok(false);
        };
        let Some(src) = fd.signature.inputs.first() else {
            return Ok(false);
        };
        // Only smaller-than-word unsigned sources widen as a
        // value-preserving identity in the word carrier (the same gate
        // `try_lower_usize_try_from` uses).
        if !matches!(
            self.tyref_literal_uint_atom(src),
            Some("U8" | "U16" | "U32")
        ) {
            return Ok(false);
        }
        // Destination must be a word-sized integer carrier.  A signed
        // word destination needs the same annotation re-type the cast
        // path applies (the `Rvalue::Cast` unsigned→signed arm): aliasing
        // a `uN` source straight into an `iN` carrier preserves the source
        // `r_uint` annotation and trips the SomeInteger signedness
        // `UnionError` (binaryop.py:178-202) when the value later meets a
        // signed operand.  Route signed destinations through
        // `rarithmetic.intmask` (identity on the i64 carrier, re-types
        // Signed); alias unsigned destinations directly, as upstream
        // `widen` leaves no op.
        let dest_signed_word =
            matches!(self.tyref_literal_int_atom(dest_ty), Some("I64" | "Isize"));
        let dest_unsigned_word =
            matches!(self.tyref_literal_uint_atom(dest_ty), Some("U64" | "Usize"));
        if !dest_signed_word && !dest_unsigned_word {
            return Ok(false);
        }
        let arg = arg.clone();
        let bb_id = self.block_id[mir_bb];
        if dest_signed_word {
            let res = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: Some(res.clone()),
                kind: OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: ["rpython", "rlib", "rarithmetic", "intmask"]
                            .into_iter()
                            .map(str::to_string)
                            .collect(),
                    },
                    args: vec![arg],
                    result_ty: ValueType::Int,
                },
            });
            self.local_var[dest_local] = Some(res);
        } else {
            // Identity widen: bind the destination directly to the operand —
            // no op materialized, exactly as upstream `widen` leaves none.
            self.local_var[dest_local] = Some(arg);
        }
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(true)
    }

    /// Resolve `Result::expect` / `Result::unwrap` on a receiver local
    /// recorded always-`Ok` (`const_discriminant_locals`, tag 0) to
    /// the receiver variable itself.  Such locals are bound directly
    /// to the `Ok` payload by [`Lowering::try_lower_usize_try_from`]
    /// (no `Result` object is materialized), so the unwrap is an
    /// alias, not an op.  The match these methods perform lives
    /// inside the Opaque core body, so without the intercept the call
    /// keeps its panic-message `&str` argument and the graph walls on
    /// the `__str_const` lowering even though the `Err` arm is
    /// statically dead.  Upstream has no such call: `ovfcheck`-free
    /// widening is a plain identity (`rarithmetic.py:140-145 widen`),
    /// so the operand *is* the whole operation.  Returns `None` for
    /// any other callee or a receiver without the always-`Ok` record,
    /// keeping the generic `Call` form.
    fn expect_on_const_ok(
        &self,
        segments: &[String],
        args: &[Variable],
        arg_locals: &[Option<usize>],
    ) -> Option<Variable> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return None;
        };
        if first.as_str() != "core"
            || module.as_str() != "result"
            || impl_seg.as_str() != "<Impl>"
            || !matches!(leaf.as_str(), "expect" | "unwrap")
        {
            return None;
        }
        let recv = args.first()?;
        let recv_local = (*arg_locals.first()?)?;
        if *self.const_discriminant_locals.get(&recv_local)? != 0 {
            return None;
        }
        Some(recv.clone())
    }

    /// Resolve the reflexive blanket `Into::into`
    /// (`core::convert::<Impl>::into` where the argument's declared
    /// type equals the destination's) to its operand.  `impl<T>
    /// From<T> for T` makes `x.into()` an identity, but the blanket
    /// fn's body is Opaque in the LLBC so the `Call` form is
    /// permanently unliftable.  Upstream has no counterpart: an
    /// identity conversion never appears as an op in RPython graphs.
    /// Non-reflexive `into` calls (source ≠ destination type) keep
    /// the generic `Call` form.
    fn reflexive_into_alias(
        &self,
        segments: &[String],
        args: &[Variable],
        first_arg_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        let [first, .., module, impl_seg, leaf] = segments else {
            return None;
        };
        if first.as_str() != "core"
            || module.as_str() != "convert"
            || impl_seg.as_str() != "<Impl>"
            || leaf.as_str() != "into"
        {
            return None;
        }
        self.identity_self_call_alias(args, first_arg_ty, dest_ty)
    }

    /// Resolve the reflexive blanket `IntoIterator::into_iter`
    /// (`core::iter::traits::collect::<Impl>::into_iter`) to its
    /// operand.  `impl<I: Iterator> IntoIterator for I` sets
    /// `IntoIter = Self` and `into_iter(self) -> Self { self }`, so the
    /// call is an identity whenever the receiver type equals the
    /// destination type.  The blanket fn's body is Opaque in the LLBC,
    /// leaving the `Call` form permanently unliftable; an
    /// already-an-iterator `into_iter` never appears as an op in RPython
    /// graphs.  Concrete `IntoIterator` impls (`&[T]` → `slice::Iter`,
    /// `Vec` → `vec::IntoIter`) have receiver ≠ destination and keep the
    /// generic `Call` form.
    fn reflexive_into_iter_alias(
        &self,
        segments: &[String],
        args: &[Variable],
        first_arg_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        let [.., iter_seg, traits_seg, collect_seg, impl_seg, leaf] = segments else {
            return None;
        };
        if iter_seg.as_str() != "iter"
            || traits_seg.as_str() != "traits"
            || collect_seg.as_str() != "collect"
            || impl_seg.as_str() != "<Impl>"
            || leaf.as_str() != "into_iter"
        {
            return None;
        }
        self.identity_self_call_alias(args, first_arg_ty, dest_ty)
    }

    /// Shared resolution for an Opaque blanket-impl call whose receiver
    /// type equals its result type: the call is a no-op, so it resolves
    /// to its sole operand.  Used by the reflexive `Into::into` and
    /// `IntoIterator::into_iter` blanket-impl aliases above.
    fn identity_self_call_alias(
        &self,
        args: &[Variable],
        first_arg_ty: Option<&TyRef>,
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        let [arg] = args else {
            return None;
        };
        let src = self.tyref_body(first_arg_ty?)?;
        let dst = self.tyref_body(dest_ty)?;
        (src == dst).then(|| arg.clone())
    }

    /// Resolve the trait-spelled `Into::into` (`["Into", "into"]` — a
    /// generic-parameter receiver, so Charon cannot select the impl at
    /// the call site) to its operand when the destination type is
    /// `alloc::string::String`.  `impl Into<String>` message parameters
    /// (the `PyError` constructor family) reach `msg.into()` inside the
    /// generic body; the annotation model maps `String` and `str` to
    /// the same string value (`project_pyre_field_type` — `s_unicode0`),
    /// matching upstream's single string type (`rstr.py`), so the
    /// conversion is an identity at the annotation level.  Other
    /// destination types keep the generic `Call` form.
    fn trait_into_string_alias(
        &self,
        segments: &[String],
        args: &[Variable],
        dest_ty: &TyRef,
    ) -> Option<Variable> {
        let [trait_seg, leaf] = segments else {
            return None;
        };
        if trait_seg.as_str() != "Into" || leaf.as_str() != "into" {
            return None;
        }
        let [arg] = args else {
            return None;
        };
        let dest_path = self.tyref_adt_name_path(dest_ty)?;
        (dest_path == "alloc::string::String").then(|| arg.clone())
    }

    /// Resolve the WTF-8 string wrappers `Wtf8::new(&str) -> &Wtf8` and
    /// `Wtf8Buf::from_string(String) -> Wtf8Buf` to their sole string
    /// argument.  Rust's `&str` / `String` / `Wtf8` / `Wtf8Buf` all map
    /// to the single immutable rpy_string value (`project_pyre_field_type`
    /// — `s_unicode0`, matching upstream's one string type in `rstr.py`),
    /// so the wrap is an identity at the annotation level; the boxing the
    /// callers want (`box_str_constant`) happens downstream on the bound
    /// value.  Both bodies are Opaque in the LLBC (external
    /// `rustpython_wtf8` crate), leaving the generic `Call` permanently
    /// unliftable.
    fn wtf8_string_identity_alias(
        &self,
        segments: &[String],
        args: &[Variable],
    ) -> Option<Variable> {
        let [arg] = args else {
            return None;
        };
        matches!(
            segments,
            [a, b] if matches!(
                (a.as_str(), b.as_str()),
                ("Wtf8", "new") | ("Wtf8Buf", "from_string")
            )
        )
        .then(|| arg.clone())
    }

    /// The fully-qualified `name_path()` of the ADT a [`TyRef`]
    /// resolves to, following `Deduplicated` / `HashConsedValue`
    /// wrapper layers.  `None` for non-ADT shapes.
    fn tyref_adt_name_path(&self, ty: &TyRef) -> Option<String> {
        let mut v = self.tyref_body(ty)?;
        loop {
            let obj = v.as_object()?;
            if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
                v = self.llbc.dedup_body(id)?;
                continue;
            }
            if let Some(arr) = obj
                .get("HashConsedValue")
                .and_then(serde_json::Value::as_array)
                && arr.len() == 2
            {
                v = &arr[1];
                continue;
            }
            break;
        }
        let def_id = inline_adt_def_id(v)?;
        let td = self.llbc.type_by_id(def_id)?;
        Some(td.item_meta.name_path())
    }

    /// `true` when `ty` resolves to a FIELDLESS enum whose discriminant
    /// tag sits at the value's base (byte 0).  Mirrors the
    /// [`Lowering::tyref_adt_name_path`] resolution (dedup / hash-consed
    /// bodies), then checks the enum layout's tag position.  The
    /// `Discriminant` lowering folds the tag read of such an inline enum
    /// field directly into the container at the field offset: a
    /// fieldless enum's whole value IS the tag, so reading the field's
    /// bytes (sized to the enum at `disc_ty` above) yields the
    /// discriminant.  A data-carrying enum or a non-zero tag offset is
    /// rejected so the read never picks up variant payload bytes.
    fn inline_fieldless_enum_field_tag0(&self, ty: &TyRef) -> bool {
        let Some(mut v) = self.tyref_body(ty) else {
            return false;
        };
        loop {
            let Some(obj) = v.as_object() else {
                return false;
            };
            if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
                let Some(next) = self.llbc.dedup_body(id) else {
                    return false;
                };
                v = next;
                continue;
            }
            if let Some(arr) = obj
                .get("HashConsedValue")
                .and_then(serde_json::Value::as_array)
                && arr.len() == 2
            {
                v = &arr[1];
                continue;
            }
            break;
        }
        let Some(def_id) = inline_adt_def_id(v) else {
            return false;
        };
        let Some(td) = self.llbc.type_by_id(def_id) else {
            return false;
        };
        let TypeDeclKind::Enum(variants) = &td.kind else {
            return false;
        };
        variants.iter().all(|variant| variant.fields.is_empty())
            && td
                .layout_for_target("")
                .and_then(|l| l.discriminant_offset())
                .unwrap_or(0)
                == 0
    }

    /// The resolved JSON body of a [`TyRef`], following the dedup
    /// table.  Two `TyRef`s denote the same type iff their bodies are
    /// structurally equal (the dedup table only dedupes identical
    /// bodies, so mixed inline/dedup spellings still compare equal).
    fn tyref_body<'t>(&self, ty: &'t TyRef) -> Option<&'t serde_json::Value>
    where
        'a: 't,
    {
        match ty {
            TyRef::Inline { value: (_, v) } => Some(v),
            TyRef::Other(v) => Some(v),
            TyRef::Dedup { id } => self.llbc.dedup_body(*id),
        }
    }

    /// The `UInt` width atom (`"U8"` / `"U32"` / `"Usize"` …) of a
    /// scalar-typed [`TyRef`], `None` for any non-`UInt` shape.
    fn tyref_literal_uint_atom<'t>(&self, ty: &'t TyRef) -> Option<&'t str>
    where
        'a: 't,
    {
        let value = match ty {
            TyRef::Inline { value: (_, v) } => v,
            TyRef::Other(v) => v,
            TyRef::Dedup { id } => self.llbc.dedup_body(*id)?,
        };
        value
            .as_object()?
            .get("Literal")?
            .as_object()?
            .get("UInt")?
            .as_str()
    }

    /// The variant-qualified owner key `{owner}::{variant}` for the
    /// SUCCESS variant a tagged-pair `__pos_0` payload belongs to
    /// (`Option::Some` at disc 1, `Result::Ok` at disc 0).
    ///
    /// Resolution prefers the variant carrying the declared discriminant.
    /// For a niche-optimised layout where the success variant has no
    /// explicit discriminant (`discriminant_i64()` is `None`), it falls
    /// back to the SOLE payload-carrying variant — the `__pos_0` payload
    /// belongs to it by definition, never to the empty `None`/`Err`
    /// sibling or the enum root (whose only field after the base/variant
    /// split is `__discriminant`).  Resolving to the variant keeps the
    /// write owner in agreement with the variant-qualified read owner
    /// (`resolve_adt_field`).  `None` only when `td` is not an enum or no
    /// single payload variant can be identified.
    fn tagged_pair_payload_owner(td: &TypeDecl, owner: &str, disc: i64) -> Option<String> {
        let TypeDeclKind::Enum(variants) = &td.kind else {
            return None;
        };
        let v = variants
            .iter()
            .find(|v| v.discriminant_i64() == Some(disc))
            .or_else(|| {
                let mut payload_variants = variants.iter().filter(|v| !v.fields.is_empty());
                match (payload_variants.next(), payload_variants.next()) {
                    (Some(only), None) => Some(only),
                    _ => None,
                }
            })?;
        Some(format!("{owner}::{}", v.name))
    }

    /// The `Int` width atom (`"I8"` / `"I32"` / `"Isize"` …) of a
    /// signed-integer literal type, mirroring [`tyref_literal_uint_atom`]
    /// for the `{"Literal": {"Int": "Isize"}}` shell.  `None` for any
    /// non-signed-literal type.
    fn tyref_literal_int_atom<'t>(&self, ty: &'t TyRef) -> Option<&'t str>
    where
        'a: 't,
    {
        let value = match ty {
            TyRef::Inline { value: (_, v) } => v,
            TyRef::Other(v) => v,
            TyRef::Dedup { id } => self.llbc.dedup_body(*id)?,
        };
        value
            .as_object()?
            .get("Literal")?
            .as_object()?
            .get("Int")?
            .as_str()
    }

    /// Shared tail for the decomposed checked-arithmetic /
    /// infallible-conversion call lowerings: write `disc` and
    /// `payload` into the destination `Option`/`Result` local as a
    /// synthetic aggregate — the same transparent-ctor + `FieldWrite`
    /// chain [`Rvalue::Aggregate`] emits — then bind the destination
    /// local and close the block toward the call's success target.
    ///
    /// Unlike `resolve_aggregate_adt`'s enum arm, which constructs
    /// the VARIANT identity (`Option::Some`), the CTOR here constructs
    /// the enum TYPE root (`Option`).  That is deliberate: `disc` may
    /// be a runtime value (`checked_neg`'s `ne(v, MIN)`), so no single
    /// variant identity annotates the destination, and the root is the
    /// `SomeInstance(enum)` that multi-assigned locals union against
    /// (`<other> ∪ int` UnionError in `mergeinputargs` otherwise).
    /// The `__discriminant` write keys the root too (the tag sits at
    /// offset 0 of every variant).  The `__pos_0` write keys
    /// `payload_owner` — the SUCCESS variant (`Option::Some` /
    /// `Result::Ok`) — so its runtime offset matches the
    /// `resolve_adt_field` read, which is variant-qualified
    /// (`{enum_leaf}::{variant}`).
    fn emit_tagged_pair_aggregate(
        &mut self,
        mir_bb: usize,
        owner: &str,
        payload_owner: &str,
        disc: Variable,
        payload: Variable,
        dest_local: usize,
        target: usize,
    ) -> Result<(), LowerError> {
        let bb_id = self.block_id[mir_bb];
        let mut owner_path: Vec<String> = owner.split("::").map(str::to_string).collect();
        let ctor_name = owner_path.pop().unwrap_or_default();
        let ctor_target = if owner_path.is_empty() {
            CallTarget::synthetic_transparent_ctor(ctor_name.clone())
        } else {
            CallTarget::synthetic_transparent_ctor_with_owner(owner_path, ctor_name)
        };
        let res = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::Call {
                target: ctor_target,
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(owner.to_string())),
            },
        });
        // Both decomposed fields carry integers: the `__discriminant`
        // tag is an `i64` (matching the `Rvalue::Discriminant`
        // `FieldRead` and the `i64` field registration) and the
        // `__pos_0` payload is the negated / widened integer the
        // `checked_neg` / `usize::try_from` callers materialize.  A
        // `Ref` field type here would disagree with that registration.
        // `__discriminant` keys the root (tag offset 0); `__pos_0` keys
        // the success variant so its exact offset matches the read.
        for (name, value, field_owner) in [
            ("__discriminant", disc, owner),
            ("__pos_0", payload, payload_owner),
        ] {
            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: res.clone(),
                    field: crate::model::FieldDescriptor {
                        name: name.to_string(),
                        owner_root: Some(field_owner.to_string()),
                        owner_id: None,
                    },
                    value: LinkArg::Value(value),
                    ty: ValueType::Int,
                },
            });
        }
        self.local_var[dest_local] = Some(res);
        let target_bb = self.block_id[target];
        let link_args = self.edge_args(mir_bb, target)?;
        self.graph.set_goto(bb_id, target_bb, link_args);
        Ok(())
    }

    fn lower_switch(
        &mut self,
        mir_bb: usize,
        discr: Operand,
        targets: SwitchTargets,
    ) -> Result<(), LowerError> {
        let bb_id = self.block_id[mir_bb];
        let discr_var = self.resolve_operand(mir_bb, discr)?;
        match targets {
            SwitchTargets::If(then_bb, else_bb) => {
                // Constant-bool discriminant: take the known arm
                // unconditionally.  `flowcontext.py:364-367
                // FlowContext.guessbool` returns a Constant condition's
                // value directly instead of forking the recorder, so a
                // translation-time `const` gate like
                // `if WITHPREBUILTINT { ... }` leaves no branch in the
                // upstream graph.  The discriminant here is a fresh var
                // whose defining op sits in this same block when the
                // read folded to a constant (`static_addr_op` /
                // `global_literal_init_op` chain); the untaken target
                // stays in `block_id` but drops out via the adapter's
                // reachability prune.
                let const_cond = self
                    .graph
                    .block(bb_id)
                    .operations
                    .iter()
                    .rev()
                    .find(|op| op.result.as_ref() == Some(&discr_var))
                    .and_then(|op| match op.kind {
                        OpKind::ConstBool(b) => Some(b),
                        _ => None,
                    });
                if let Some(cond) = const_cond {
                    let taken = if cond { then_bb } else { else_bb };
                    let args = self.edge_args(mir_bb, taken as usize)?;
                    self.graph
                        .set_goto(bb_id, self.block_id[taken as usize], args);
                    return Ok(());
                }
                // Route through `set_branch` so the cond gets the
                // upstream `bool` UnaryOp wrap before becoming the
                // exitswitch (flowcontext.py:756
                // `Variable.bool().eval(self)`).  Necessary because the
                // MIR discriminant for an `If` target can be a Ref
                // (e.g. a SyntheticTransparentCtor result) whereas
                // codewriter/assembler.rs::FlatOp::GotoIfNot expects
                // `cond.kind == RegKind::Int`.  `true_args` / `false_args`
                // carry each target block's input arguments; `set_branch`
                // asserts their arity against the block's `inputargs`.
                let then_args = self.edge_args(mir_bb, then_bb as usize)?;
                let else_args = self.edge_args(mir_bb, else_bb as usize)?;
                self.graph.set_branch(
                    bb_id,
                    discr_var,
                    self.block_id[then_bb as usize],
                    then_args,
                    self.block_id[else_bb as usize],
                    else_args,
                );
                Ok(())
            }
            SwitchTargets::SwitchInt(_int_ty, arms, default) => {
                let mut links: Vec<Link> = Vec::new();
                for (scalar, bb) in arms {
                    let case = scalar_to_const_value(&scalar).ok_or_else(|| {
                        LowerError::Unsupported(format!(
                            "bb{mir_bb}: SwitchInt case scalar shape not yet supported: {scalar}"
                        ))
                    })?;
                    let args = self.edge_args(mir_bb, bb as usize)?;
                    links.push(
                        Link::from_variables(
                            &self.graph,
                            args,
                            self.block_id[bb as usize],
                            Some(ExitCase::Const(case)),
                        )
                        .with_prevblock(bb_id)
                        .with_llexitcase_from_exitcase(),
                    );
                }
                if !self.switch_default_targets_panic_abort(default) {
                    let default_args = self.edge_args(mir_bb, default as usize)?;
                    links.push(
                        Link::from_variables(
                            &self.graph,
                            default_args,
                            self.block_id[default as usize],
                            Some(ExitCase::Const(ConstValue::UniStr("default".into()))),
                        )
                        .with_prevblock(bb_id),
                    );
                }
                self.graph.block_mut(bb_id).exitswitch = Some(ExitSwitch::Value(discr_var));
                self.graph.closeblock(bb_id, links);
                Ok(())
            }
        }
    }

    fn edge_args(&mut self, from_bb: usize, target_bb: usize) -> Result<Vec<Variable>, LowerError> {
        let local_indices = self.target_input_locals(target_bb)?;
        let mut args = Vec::with_capacity(local_indices.len());
        for local_idx in local_indices {
            let var = self
                .local_var
                .get(local_idx)
                .and_then(Clone::clone)
                .ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{from_bb}: edge to bb{target_bb} needs live MIR local {local_idx}, \
                         but it is uninitialised"
                    ))
                })?;
            self.merge_positional_aggregate_state(target_bb, local_idx);
            args.push(var);
        }
        Ok(args)
    }

    fn target_input_locals(&self, target_bb: usize) -> Result<Vec<usize>, LowerError> {
        if target_bb >= self.block_id.len() {
            return Err(LowerError::Schema(format!(
                "edge references unknown target bb{target_bb}"
            )));
        }
        if target_bb == 0 {
            let mut locals = Vec::with_capacity(self.arg_count);
            for local_idx in 1..=self.arg_count {
                locals.push(local_idx);
            }
            for (local_idx, live) in self
                .block_live_in
                .get(target_bb)
                .into_iter()
                .flat_map(|v| v.iter())
                .copied()
                .enumerate()
            {
                if live && (local_idx == 0 || local_idx > self.arg_count) {
                    return Err(LowerError::Unsupported(format!(
                        "edge to startblock bb0 requires non-argument MIR local {local_idx}"
                    )));
                }
            }
            return Ok(locals);
        }
        Ok(self
            .block_live_in
            .get(target_bb)
            .map(|locals| {
                locals
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(idx, live)| live.then_some(idx))
                    .collect()
            })
            .unwrap_or_default())
    }

    fn merge_positional_aggregate_state(&mut self, target_bb: usize, local_idx: usize) {
        if target_bb >= self.block_positional_seen.len()
            || local_idx >= self.block_positional_seen[target_bb].len()
            || self.block_positional_conflict[target_bb][local_idx]
        {
            return;
        }
        let incoming = self.positional_aggregate_locals.get(&local_idx).cloned();
        if !self.block_positional_seen[target_bb][local_idx] {
            self.block_positional_seen[target_bb][local_idx] = true;
            if let Some(owner) = incoming {
                self.block_entry_positional_aggregate_locals[target_bb].insert(local_idx, owner);
            }
            return;
        }
        let current = self.block_entry_positional_aggregate_locals[target_bb]
            .get(&local_idx)
            .cloned();
        if current != incoming {
            self.block_positional_conflict[target_bb][local_idx] = true;
            self.block_entry_positional_aggregate_locals[target_bb].remove(&local_idx);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect the MIR locals bound by a scalar [`Rvalue::BinaryOp`]
/// anywhere in `body`.  See [`Lowering::binop_result_locals`] for why a
/// single function-wide set is sound (a local's MIR type is fixed, so a
/// local bound by `BinaryOp` is a `*Checked` scalar at every read site).
fn compute_binop_result_locals(body: &Unstructured) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    for bb in &body.body {
        for stmt in &bb.statements {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind() else {
                continue;
            };
            if matches!(rvalue, Rvalue::BinaryOp(..))
                && let PlaceKind::Local(i) = place.kind
            {
                set.insert(i as usize);
            }
        }
    }
    set
}

/// MIR locals assigned more than once anywhere in `body` — statement
/// assigns plus call destinations.  See
/// [`Lowering::multi_assigned_locals`].
fn compute_multi_assigned_locals(body: &Unstructured) -> std::collections::HashSet<usize> {
    let mut counts: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    let mut bump = |place: &Place| {
        if let PlaceKind::Local(i) = place.kind {
            *counts.entry(i as usize).or_insert(0) += 1;
        }
    };
    for bb in &body.body {
        for stmt in &bb.statements {
            if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind() {
                bump(&place);
            }
        }
        if let Ok(TermKind::Call { call, .. }) = bb.term() {
            bump(&call.dest);
        }
    }
    counts
        .into_iter()
        .filter(|(_, c)| *c > 1)
        .map(|(i, _)| i)
        .collect()
}

/// Whether a statically-resolved [`RegularCall`] is a workspace
/// `Index::index` / `IndexMut::index_mut` impl (the `FixedObjectArray`
/// family) — those bottom out at raw-slice construction and are lowered
/// as RPython's getarrayitem rather than a residual call.  Shared by the
/// call-lowering intercept and the liveness pre-pass.
fn is_workspace_index_regular(reg: &RegularCall, llbc: &Llbc) -> bool {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    let Some(fd) = llbc.fn_by_id(*id) else {
        return false;
    };
    let path = fd.item_meta.name_path();
    let leaf = path.rsplit("::").next().unwrap_or("");
    (leaf == "index" || leaf == "index_mut") && path.starts_with("pyre_")
}

/// Whether a statically-resolved [`RegularCall`] is `<*mut T>::add` /
/// `<*const T>::add` (`core::ptr::{mut_ptr,const_ptr}::<Impl>::add`) —
/// the only `.add` spellings the items-base accessor collapse
/// (brick 1, [`Lowering::is_items_block_base_ptr_add`]) and the list
/// element-access lowering (brick 3,
/// [`Lowering::is_list_items_elem_ptr_add`]) intercept.
fn regular_call_is_ptr_add(reg: &RegularCall, llbc: &Llbc) -> bool {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    llbc.fn_by_id(*id).is_some_and(|fd| {
        matches!(
            fd.item_meta.name_path().as_str(),
            "core::ptr::mut_ptr::<Impl>::add" | "core::ptr::const_ptr::<Impl>::add"
        )
    })
}

/// Whether a statically-resolved [`RegularCall`] is one of the two
/// `ItemsBlock` items-base accessors brick 1 rewrites to return the
/// block *header* pointer (`items_block_items_base` /
/// `items_block_items_ptr`).  brick 3's `*base.add(idx)` lowering only
/// fires when the base traces to one of these (their header return is
/// what the `base_size = ITEMS_BLOCK_ITEMS_OFFSET` array descr re-adds);
/// a `*mut PyObjectRef` from any other producer already points at
/// `items[0]` and must keep its residual `add`.
fn regular_call_is_items_block_accessor(reg: &RegularCall, llbc: &Llbc) -> bool {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    llbc.fn_by_id(*id).is_some_and(|fd| {
        matches!(
            fd.item_meta.name_path().as_str(),
            "pyre_object::object_array::items_block_items_base"
                | "pyre_object::object_array::items_block_items_ptr"
        )
    })
}

/// The [`TyRef`] of a plain-place [`Operand`], or `None` for a
/// constant.  The borrow tracks the operand, so callers reading a
/// receiver type from a raw `CallPayload` (the deferred-write liveness
/// pre-pass) need not clone it like [`Lowering::lower_call`] does.
fn operand_tyref(op: &Operand) -> Option<&TyRef> {
    match op {
        Operand::Copy(p) | Operand::Move(p) => Some(&p.ty),
        Operand::Const(_) => None,
    }
}

/// The decomposed brick-3 element-`add` gate
/// ([`Lowering::is_list_items_elem_ptr_add`]), taking pieces both the
/// call-lowering intercept (which has already moved `call.func` /
/// `call.args` out of the payload) and the deferred-write liveness
/// pre-pass (which still holds the raw [`CallPayload`]) can supply, so
/// both agree on exactly which `.add` calls become array ops.
#[allow(clippy::too_many_arguments)]
fn is_list_items_elem_ptr_add_parts(
    reg: &RegularCall,
    args_len: usize,
    base_local: Option<usize>,
    base_ty: Option<&TyRef>,
    index_local: Option<usize>,
    dest_local: usize,
    body: &Unstructured,
    llbc: &Llbc,
) -> bool {
    args_len == 2
        && regular_call_is_ptr_add(reg, llbc)
        && base_ty.is_some_and(|ty| is_pyobjectref_items_ptr(ty, llbc))
        && index_local.is_some()
        && base_local.is_some_and(|base| base_traces_to_items_block_accessor(body, base, llbc))
        && add_dest_used_only_as_single_deref(body, dest_local)
}

/// Whether MIR local `base` — the receiver of a `*base.add(idx)` over
/// `*mut PyObjectRef` — traces, through plain `Copy` / `Move` aliases,
/// to the result of an `items_block_items_base` / `items_block_items_ptr`
/// call.  Those accessors are the only `*mut PyObjectRef` producers
/// brick 1 rewrites to return the `ItemsBlock` *header*, so the matching
/// `ArrayRead` / `ArrayWrite` `base_size = ITEMS_BLOCK_ITEMS_OFFSET`
/// re-adds the header offset to reach `items[0]`.  A pointer from any
/// other source (`FixedObjectArray::items_mut_ptr`, `null_mut`, a
/// chained `.add`) already points at the items, so re-adding the offset
/// would overshoot by one element — such a base keeps its residual
/// `add` and falls to the legacy walker.  A multiply-defined or
/// non-`Copy`-produced local is ambiguous and rejected.
fn base_traces_to_items_block_accessor(body: &Unstructured, base: usize, llbc: &Llbc) -> bool {
    let mut cur = base;
    for _ in 0..32 {
        let mut producers = 0usize;
        let mut copy_src: Option<usize> = None;
        let mut is_accessor = false;
        for bb in &body.body {
            for stmt in &bb.statements {
                if let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind()
                    && matches!(place.kind, PlaceKind::Local(i) if i as usize == cur)
                {
                    producers += 1;
                    if let Rvalue::Use(op) = &rvalue {
                        copy_src = operand_local(Some(op));
                    }
                }
            }
            if let Ok(TermKind::Call { call, .. }) = bb.term()
                && matches!(call.dest.kind, PlaceKind::Local(i) if i as usize == cur)
            {
                producers += 1;
                if let CallFunc::Regular(reg) = &call.func
                    && regular_call_is_items_block_accessor(reg, llbc)
                {
                    is_accessor = true;
                }
            }
        }
        if producers != 1 {
            return false;
        }
        if is_accessor {
            return true;
        }
        match copy_src {
            Some(src) => cur = src,
            None => return false,
        }
    }
    false
}

/// How a place / operand references the brick-3 `add`-result local
/// `dest`, as seen by [`add_dest_used_only_as_single_deref`].
enum DestRef {
    /// Does not reference `dest`.
    None,
    /// `*dest` exactly — the element load / store the array op models.
    Deref,
    /// References `dest` some other way (passed by value, re-borrowed,
    /// a deeper projection) — the raw pointer escapes the load / store.
    Other,
}

/// Whether a place's projection chain bottoms out at `Local(dest)`.
fn place_references_local(place: &Place, dest: usize) -> bool {
    match &place.kind {
        PlaceKind::Local(i) => *i as usize == dest,
        PlaceKind::Projection(inner, _) => place_references_local(inner, dest),
        PlaceKind::Global { .. } | PlaceKind::Unknown => false,
    }
}

/// Whether a place is exactly `*Local(dest)` — `Projection(Local(dest),
/// Deref)` with nothing layered on top.
fn place_is_immediate_deref_of(place: &Place, dest: usize) -> bool {
    let PlaceKind::Projection(inner, elem) = &place.kind else {
        return false;
    };
    matches!(elem, ProjectionElem::Atom(s) if s == "Deref")
        && matches!(inner.kind, PlaceKind::Local(i) if i as usize == dest)
}

/// Classify how a read-position operand references `dest`.
fn operand_dest_ref(op: &Operand, dest: usize) -> DestRef {
    let (Operand::Copy(p) | Operand::Move(p)) = op else {
        return DestRef::None;
    };
    if place_is_immediate_deref_of(p, dest) {
        DestRef::Deref
    } else if place_references_local(p, dest) {
        DestRef::Other
    } else {
        DestRef::None
    }
}

fn bump_dest_ref(r: DestRef, derefs: &mut usize, other: &mut usize) {
    match r {
        DestRef::Deref => *derefs += 1,
        DestRef::Other => *other += 1,
        DestRef::None => {}
    }
}

/// Classify a write-target place's reference to `dest`: a bare
/// `Local(dest)` is the `add` definition, `*dest` a deref store, and any
/// other reference an escape.
fn classify_write_place(
    place: &Place,
    dest: usize,
    defs: &mut usize,
    derefs: &mut usize,
    other: &mut usize,
) {
    if matches!(place.kind, PlaceKind::Local(i) if i as usize == dest) {
        *defs += 1;
    } else if place_is_immediate_deref_of(place, dest) {
        *derefs += 1;
    } else if place_references_local(place, dest) {
        *other += 1;
    }
}

fn scan_rvalue_dest_ref(rvalue: &Rvalue, dest: usize, derefs: &mut usize, other: &mut usize) {
    match rvalue {
        Rvalue::Use(op)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Cast(_, op, _)
        | Rvalue::Repeat(op, _, _)
        | Rvalue::ShallowInitBox(op, _) => bump_dest_ref(operand_dest_ref(op, dest), derefs, other),
        Rvalue::BinaryOp(_, lhs, rhs) => {
            bump_dest_ref(operand_dest_ref(lhs, dest), derefs, other);
            bump_dest_ref(operand_dest_ref(rhs, dest), derefs, other);
        }
        // A re-borrow / `Len` / `Discriminant` of `dest` is not a plain
        // element load, so any reference to `dest` here escapes.
        Rvalue::Ref { place, .. }
        | Rvalue::RawPtr { place, .. }
        | Rvalue::Len(place)
        | Rvalue::Discriminant(place) => {
            if place_references_local(place, dest) {
                *other += 1;
            }
        }
        Rvalue::Aggregate(_, operands) => {
            for op in operands {
                bump_dest_ref(operand_dest_ref(op, dest), derefs, other);
            }
        }
        Rvalue::NullaryOp(_, _) | Rvalue::Unknown => {}
    }
}

/// The brick-3 escape guard: whether the `.add`-result local `dest` is
/// used in the body *only* as a single dereference — exactly one
/// definition (the `add` itself), exactly one `*dest` read **or** `*dest
/// = v` write, and no other appearance.  A raw pointer that is passed by
/// value (`ptr::copy(p, ..)`), used as the base of a further `.add`,
/// re-borrowed, or read twice leaves the element load / store model the
/// `ArrayRead` / `ArrayWrite` captures, so the `add` must stay residual.
/// `StorageLive` / `StorageDead` / `PlaceMention` are borrow-ck markers,
/// not loads, so they are ignored.
fn add_dest_used_only_as_single_deref(body: &Unstructured, dest: usize) -> bool {
    let mut defs = 0usize;
    let mut derefs = 0usize;
    let mut other = 0usize;
    for bb in &body.body {
        for stmt in &bb.statements {
            match stmt.stmt_kind() {
                Ok(StmtKind::Assign(place, rvalue)) => {
                    scan_rvalue_dest_ref(&rvalue, dest, &mut derefs, &mut other);
                    classify_write_place(&place, dest, &mut defs, &mut derefs, &mut other);
                }
                Ok(StmtKind::Assert(assert)) => bump_dest_ref(
                    operand_dest_ref(&assert.cond, dest),
                    &mut derefs,
                    &mut other,
                ),
                _ => {}
            }
        }
        match bb.term() {
            Ok(TermKind::Switch { discr, .. }) => {
                bump_dest_ref(operand_dest_ref(&discr, dest), &mut derefs, &mut other)
            }
            Ok(TermKind::Call { call, .. }) => {
                if let CallFunc::Dynamic(op) = &call.func {
                    bump_dest_ref(operand_dest_ref(op, dest), &mut derefs, &mut other);
                }
                for arg in &call.args {
                    bump_dest_ref(operand_dest_ref(arg, dest), &mut derefs, &mut other);
                }
                classify_write_place(&call.dest, dest, &mut defs, &mut derefs, &mut other);
            }
            Ok(TermKind::Assert { assert, .. }) => bump_dest_ref(
                operand_dest_ref(&assert.cond, dest),
                &mut derefs,
                &mut other,
            ),
            _ => {}
        }
    }
    defs == 1 && derefs == 1 && other == 0
}

/// The MIR local behind a plain-local [`Operand`], or `None` for a
/// constant or a projected place.
fn operand_local(op: Option<&Operand>) -> Option<usize> {
    match op? {
        Operand::Copy(p) | Operand::Move(p) => match p.kind {
            PlaceKind::Local(i) => Some(i as usize),
            _ => None,
        },
        Operand::Const(_) => None,
    }
}

/// The base MIR local of a `*p = v` deref write (`Projection(Local(p),
/// Deref)`), or `None` for any other place shape — the same destination
/// shape [`Lowering::emit_projection_write`] re-expresses as an
/// `ArrayWrite` when `p` was bound by a workspace index call.
fn deref_write_base_local(place: &Place) -> Option<usize> {
    let PlaceKind::Projection(inner, elem) = &place.kind else {
        return None;
    };
    let ProjectionElem::Atom(s) = elem else {
        return None;
    };
    if s != "Deref" {
        return None;
    }
    match inner.kind {
        PlaceKind::Local(i) => Some(i as usize),
        _ => None,
    }
}

/// Per-block extra-live MIR locals for the deferred array-write base /
/// index of a workspace `index` / `index_mut` call.
///
/// `arr[i] = v` lowers to `_p = index_mut(_s, _i)` (recorded as an
/// `index_elem_alias`, [`Lowering::lower_call`]) followed — across the
/// call's own block split — by `*_p = v`, which
/// [`Lowering::emit_projection_write`] re-expresses as `ArrayWrite {
/// base: _s, index: _i, .. }`.  That `ArrayWrite` is a synthetic use of
/// `_s` / `_i` the MIR never spells, so plain liveness drops them before
/// the write block and the base reaches the rtyper as an undefined
/// operand.  Return, per block, the base/index locals of every such
/// `_p` deref-written in it, for [`compute_mir_liveness`] to mark live;
/// the backward fixpoint then threads them from their definition (which
/// dominates the `_p` use).
fn compute_index_write_extra_live(body: &Unstructured, llbc: &Llbc) -> Vec<Vec<usize>> {
    let mut index_call: std::collections::HashMap<usize, (Option<usize>, Option<usize>)> =
        std::collections::HashMap::new();
    for bb in &body.body {
        let Ok(TermKind::Call { call, .. }) = bb.term() else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        let PlaceKind::Local(p) = call.dest.kind else {
            continue;
        };
        // Both the workspace `index_mut` call and brick 3's
        // `*base.add(idx)` defer their `*p = v` write to a later block,
        // where the base / index operands are no longer spelled (the
        // `ArrayWrite` is synthesised by `emit_projection_write`).
        let is_deferred_write_producer = is_workspace_index_regular(reg, llbc)
            || is_list_items_elem_ptr_add_parts(
                reg,
                call.args.len(),
                operand_local(call.args.first()),
                call.args.first().and_then(operand_tyref),
                operand_local(call.args.get(1)),
                p as usize,
                body,
                llbc,
            );
        if !is_deferred_write_producer {
            continue;
        }
        index_call.insert(
            p as usize,
            (
                operand_local(call.args.first()),
                operand_local(call.args.get(1)),
            ),
        );
    }
    let mut extra = vec![Vec::new(); body.body.len()];
    if index_call.is_empty() {
        return extra;
    }
    for (bb_idx, bb) in body.body.iter().enumerate() {
        for stmt in &bb.statements {
            let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind() else {
                continue;
            };
            if let Some(p) = deref_write_base_local(&place)
                && let Some((base, idx)) = index_call.get(&p)
            {
                extra[bb_idx].extend(base.iter().copied());
                extra[bb_idx].extend(idx.iter().copied());
            }
        }
    }
    extra
}

fn compute_mir_liveness(body: &Unstructured, extra_live: &[Vec<usize>]) -> Vec<Vec<bool>> {
    use bit_set::BitSet;

    let n_blocks = body.body.len();
    let n_locals = body.locals.locals.len();
    let mut uses = vec![BitSet::with_capacity(n_locals); n_blocks];
    let mut defs = vec![BitSet::with_capacity(n_locals); n_blocks];
    let mut succs = vec![Vec::<usize>::new(); n_blocks];
    let mut preds = vec![Vec::<usize>::new(); n_blocks];

    for (bb_idx, bb) in body.body.iter().enumerate() {
        for stmt in &bb.statements {
            let Ok(kind) = stmt.stmt_kind() else {
                continue;
            };
            match kind {
                StmtKind::Assign(place, rvalue) => {
                    mark_rvalue_uses(&rvalue, &mut uses[bb_idx], &defs[bb_idx], n_locals);
                    mark_place_write(&place, &mut uses[bb_idx], &mut defs[bb_idx], n_locals);
                }
                StmtKind::PlaceMention(place) => {
                    mark_place_use(&place, &mut uses[bb_idx], &defs[bb_idx], n_locals)
                }
                StmtKind::Assert(assert) => {
                    mark_operand_use(&assert.cond, &mut uses[bb_idx], &defs[bb_idx], n_locals)
                }
                StmtKind::StorageLive(_) | StmtKind::StorageDead(_) | StmtKind::Unknown => {}
            }
        }
        let Ok(term) = bb.term() else {
            continue;
        };
        match term {
            TermKind::Return => mark_local_use(0, &mut uses[bb_idx], &defs[bb_idx], n_locals),
            TermKind::Goto { target } => {
                push_successor(&mut succs[bb_idx], &mut preds, bb_idx, target, n_blocks)
            }
            TermKind::Switch { discr, targets } => {
                mark_operand_use(&discr, &mut uses[bb_idx], &defs[bb_idx], n_locals);
                match targets {
                    SwitchTargets::If(a, b) => {
                        push_successor(&mut succs[bb_idx], &mut preds, bb_idx, a, n_blocks);
                        push_successor(&mut succs[bb_idx], &mut preds, bb_idx, b, n_blocks);
                    }
                    SwitchTargets::SwitchInt(_, arms, default) => {
                        for (_, bb) in arms {
                            push_successor(&mut succs[bb_idx], &mut preds, bb_idx, bb, n_blocks);
                        }
                        push_successor(&mut succs[bb_idx], &mut preds, bb_idx, default, n_blocks);
                    }
                }
            }
            TermKind::Call { call, target, .. } => {
                mark_call_uses(&call, &mut uses[bb_idx], &defs[bb_idx], n_locals);
                mark_place_write(&call.dest, &mut uses[bb_idx], &mut defs[bb_idx], n_locals);
                push_successor(&mut succs[bb_idx], &mut preds, bb_idx, target, n_blocks);
            }
            TermKind::Assert { assert, target, .. } => {
                mark_operand_use(&assert.cond, &mut uses[bb_idx], &defs[bb_idx], n_locals);
                push_successor(&mut succs[bb_idx], &mut preds, bb_idx, target, n_blocks);
            }
            TermKind::Drop { target, .. } => {
                push_successor(&mut succs[bb_idx], &mut preds, bb_idx, target, n_blocks)
            }
            TermKind::UnwindResume | TermKind::Abort(_) | TermKind::Unknown => {}
        }
    }

    // Synthetic deferred-array-write uses: the base/index of a
    // workspace index call feed an `ArrayWrite` the MIR never spells
    // (see `compute_index_write_extra_live`).  Mark them live-used in
    // the block that holds the `*p = v` write — unless that block also
    // defines them — so the backward fixpoint threads them in from
    // their definition, which dominates the `_p` use.
    for (bb_idx, locals) in extra_live.iter().enumerate().take(n_blocks) {
        for &local_idx in locals {
            if local_idx < n_locals && !defs[bb_idx].contains(local_idx) {
                uses[bb_idx].insert(local_idx);
            }
        }
    }

    let mut live_in = vec![BitSet::with_capacity(n_locals); n_blocks];
    let mut worklist: std::collections::VecDeque<usize> = (0..n_blocks).rev().collect();
    let mut in_worklist = vec![true; n_blocks];
    while let Some(bb_idx) = worklist.pop_front() {
        in_worklist[bb_idx] = false;
        let mut new_in = BitSet::with_capacity(n_locals);
        for &succ in &succs[bb_idx] {
            new_in.union_with(&live_in[succ]);
        }
        new_in.difference_with(&defs[bb_idx]);
        new_in.union_with(&uses[bb_idx]);
        if new_in != live_in[bb_idx] {
            live_in[bb_idx] = new_in;
            for &pred in &preds[bb_idx] {
                if !in_worklist[pred] {
                    worklist.push_back(pred);
                    in_worklist[pred] = true;
                }
            }
        }
    }

    let mut result = vec![vec![false; n_locals]; n_blocks];
    for (bb_idx, locals) in live_in.into_iter().enumerate() {
        for local_idx in locals.iter() {
            if local_idx < n_locals {
                result[bb_idx][local_idx] = true;
            }
        }
    }
    result
}

fn push_successor(
    out: &mut Vec<usize>,
    preds: &mut [Vec<usize>],
    source: usize,
    target: u64,
    n_blocks: usize,
) {
    let target = target as usize;
    if target < n_blocks && !out.contains(&target) {
        out.push(target);
        preds[target].push(source);
    }
}

fn mark_call_uses(
    call: &CallPayload,
    uses: &mut bit_set::BitSet,
    defs: &bit_set::BitSet,
    n_locals: usize,
) {
    if let CallFunc::Dynamic(op) = &call.func {
        mark_operand_use(op, uses, defs, n_locals);
    }
    for arg in &call.args {
        mark_operand_use(arg, uses, defs, n_locals);
    }
}

fn mark_rvalue_uses(
    rvalue: &Rvalue,
    uses: &mut bit_set::BitSet,
    defs: &bit_set::BitSet,
    n_locals: usize,
) {
    match rvalue {
        Rvalue::Use(op)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Cast(_, op, _)
        | Rvalue::Repeat(op, _, _)
        | Rvalue::ShallowInitBox(op, _) => mark_operand_use(op, uses, defs, n_locals),
        Rvalue::BinaryOp(_, lhs, rhs) => {
            mark_operand_use(lhs, uses, defs, n_locals);
            mark_operand_use(rhs, uses, defs, n_locals);
        }
        Rvalue::Ref { place, .. }
        | Rvalue::RawPtr { place, .. }
        | Rvalue::Len(place)
        | Rvalue::Discriminant(place) => mark_place_use(place, uses, defs, n_locals),
        Rvalue::Aggregate(_, operands) => {
            for op in operands {
                mark_operand_use(op, uses, defs, n_locals);
            }
        }
        Rvalue::NullaryOp(_, _) | Rvalue::Unknown => {}
    }
}

fn mark_operand_use(
    op: &Operand,
    uses: &mut bit_set::BitSet,
    defs: &bit_set::BitSet,
    n_locals: usize,
) {
    match op {
        Operand::Copy(place) | Operand::Move(place) => mark_place_use(place, uses, defs, n_locals),
        Operand::Const(_) => {}
    }
}

fn mark_place_use(
    place: &Place,
    uses: &mut bit_set::BitSet,
    defs: &bit_set::BitSet,
    n_locals: usize,
) {
    match &place.kind {
        PlaceKind::Local(i) => mark_local_use(*i as usize, uses, defs, n_locals),
        PlaceKind::Projection(inner, elem) => {
            mark_place_use(inner, uses, defs, n_locals);
            mark_projection_index_offset_use(elem, uses, defs, n_locals);
        }
        PlaceKind::Global { .. } | PlaceKind::Unknown => {}
    }
}

fn mark_place_write(
    place: &Place,
    uses: &mut bit_set::BitSet,
    defs: &mut bit_set::BitSet,
    n_locals: usize,
) {
    match &place.kind {
        PlaceKind::Local(i) => mark_local_def(*i as usize, defs, n_locals),
        PlaceKind::Projection(inner, elem) => {
            mark_place_use(inner, uses, defs, n_locals);
            mark_projection_index_offset_use(elem, uses, defs, n_locals);
        }
        PlaceKind::Global { .. } | PlaceKind::Unknown => {}
    }
}

/// A place whose projection chain contains an `Index { offset, .. }`
/// reads the offset operand's local at lowering time — `resolve_place`
/// (read) and `emit_projection_write` (write) both route the offset
/// through `index_offset_var` → `resolve_operand`.  The offset lives in
/// the projection element, not in `inner`, so the surrounding
/// `mark_place_use` / `mark_place_write` recursion (which descends only
/// into `inner`) would miss it.  Mark it as a use so a loop-carried
/// index local receives a block inputarg and threads across the
/// back-edge, instead of failing loud as an uninitialised local at the
/// loop header.  `from_end` is ignored, matching `index_offset_var`.
fn mark_projection_index_offset_use(
    elem: &ProjectionElem,
    uses: &mut bit_set::BitSet,
    defs: &bit_set::BitSet,
    n_locals: usize,
) {
    let ProjectionElem::Tagged(v) = elem else {
        return;
    };
    let Some(offset) = v
        .as_object()
        .and_then(|m| m.get("Index"))
        .and_then(|idx| idx.as_object())
        .and_then(|m| m.get("offset"))
    else {
        return;
    };
    if let Ok(op) = serde_json::from_value::<Operand>(offset.clone()) {
        mark_operand_use(&op, uses, defs, n_locals);
    }
}

fn mark_local_use(
    local_idx: usize,
    uses: &mut bit_set::BitSet,
    defs: &bit_set::BitSet,
    n_locals: usize,
) {
    if local_idx >= n_locals || defs.contains(local_idx) {
        return;
    }
    uses.insert(local_idx);
}

fn mark_local_def(local_idx: usize, defs: &mut bit_set::BitSet, n_locals: usize) {
    if local_idx < n_locals {
        defs.insert(local_idx);
    }
}

/// Free-function version of [`Lowering::impl_method_owner`] for callers
/// that only have the `Llbc` + `FunDecl` and do not want to instantiate
/// a full `Lowering` context just to ask the question.  Used by
/// `build_semantic_program_from_llbc` to populate
/// `SemanticFunction.self_ty_root` on the canonical SemanticProgram
/// produced by the MIR driver.
///
/// Mirrors the instance method line-for-line; any change here must be
/// kept in sync with the `&self` version.
fn impl_method_owner_for_fundecl(llbc: &Llbc, fd: &FunDecl) -> Option<(String, String)> {
    let segs = &fd.item_meta.name;
    let last_idx = segs
        .iter()
        .rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
    let leaf = match &segs[last_idx] {
        NameSeg::Ident { ident: (s, _) } => s.clone(),
        _ => return None,
    };
    if last_idx == 0 {
        return None;
    }
    let impl_payload = match &segs[last_idx - 1] {
        NameSeg::Other(v) => v.as_object()?.get("Impl")?,
        _ => return None,
    };
    let owner_qualified = match resolve_impl_owner_adt_def_id_free(llbc, impl_payload) {
        Some(adt_def_id) => {
            let td = llbc.type_by_id(adt_def_id)?;
            // Owner-qualification convention: bare ident qualified by the
            // type's defining module path (e.g. `gc_roots::RootScope`).  Strip
            // the crate name from the TypeDecl's full name_path so the
            // `self_ty_root` keys land on a `[module::Owner, method]` CallPath.
            // Without this qualification the canonical registration loop at
            // `lib.rs:864-902` cannot find the graph keyed by
            // `[qualified_owner, method]`.
            strip_crate_prefix(&td.item_meta.name_path())
        }
        // Non-ADT `Self` allowlist fallback — same arm as the instance
        // method.  An allowlisted method has no TypeDecl, so the module
        // Ident is the only owner name available; using the same bare
        // leaf on both sides keeps the call-target key
        // (`CallTarget::Method { owner, .. }`) and the registration key
        // (`self_ty_root`) identical should an allowlisted pair ever
        // match a local fn with a body.
        None => {
            if last_idx < 2 {
                return None;
            }
            let module_leaf = match &segs[last_idx - 2] {
                NameSeg::Ident { ident: (s, _) } => s.as_str(),
                _ => return None,
            };
            if !NON_ADT_OWNER_METHOD_ALLOWLIST
                .iter()
                .any(|&(m, f)| m == module_leaf && f == leaf)
            {
                return None;
            }
            module_leaf.to_string()
        }
    };
    if owner_qualified.is_empty() {
        return None;
    }
    Some((owner_qualified, leaf))
}

/// For a `Deref` / `DerefMut` trait-impl method, resolve the leaf
/// identifier of the implementing `Self` ADT (`Box`, `Rc`, `Arc`,
/// `FrameBox`, …) directly from the impl's `Self` type, bypassing the
/// registry-keyed `impl_method_owner_for_fundecl` (which resolves only
/// self-receiver methods it can key into `PyreCallRegistry`).  Used to
/// subtract the `cast_pointer` thin-pointer rewrite for the
/// header-offset handles whose word is not the pointee address.
/// Returns `None` when the owner cannot be resolved, in which case the
/// caller keeps the ordinary thin-pointer treatment.
fn deref_impl_owner_leaf(llbc: &Llbc, fd: &FunDecl) -> Option<String> {
    let segs = &fd.item_meta.name;
    let last_idx = segs
        .iter()
        .rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
    if last_idx == 0 {
        return None;
    }
    let impl_payload = match &segs[last_idx - 1] {
        NameSeg::Other(v) => v.as_object()?.get("Impl")?,
        _ => return None,
    };
    let adt_def_id = resolve_impl_owner_adt_def_id_free(llbc, impl_payload)?;
    let td = llbc.type_by_id(adt_def_id)?;
    let path = td.item_meta.name_path();
    Some(path.rsplit("::").next().unwrap_or(&path).to_string())
}

/// Collect, from the lowered MIR,
/// `(path-segments, Signature, return-lltype)` for every local `unsafe
/// fn` / unsafe impl-method whose return type projects to `Void` (unit)
/// or `Bool`.  These callees cannot lower their bodies (raw-pointer
/// access the flowspace adapter does not model), but downstream
/// `OpKind::Call::FunctionPath` sites still need their signature
/// registered so the dual gate does not Skip with "not registered in
/// PyreCallRegistry".
///
/// The single registration key must equal the call-site lookup. A free
/// fn keys as the crate-included `name_path()` split on every `::`
/// (`["pyre_interpreter", "objspace", "std", "mapdict", "fn"]`) — the
/// exact segment vector the Call terminator and `FnDef`-constant
/// call-sites emit (`call_target_segments` / `decode_constant` both
/// `name_path().split("::")` without stripping the crate).
/// `register_unsafe_fn_stubs` registers a single verbatim key with no
/// alias fan-out (unlike `free_function_alias_paths`), and three-plus-
/// segment paths are excluded from the `lookup_with_leaf_match`
/// fallback, so a crate-stripped or module-collapsed key would miss the
/// nested call site.  Free functions and impl-owned functions are both
/// collected and keyed on `name_path()`: an impl method usually lowers
/// to `CallTarget::Method` (resolved through the receiver classdef), but
/// a receiver-less associated function and any impl method reached
/// through an `FnDef` constant fall back to `CallTarget::FunctionPath {
/// name_path }`, whose lookup is served only by this registry.  Argument
/// names come from the Charon body locals, falling back to `arg{N}`.
/// Return types other than unit / bool surface no entry, preserving the
/// original "not registered" Skip for those fns — matches
/// `simple_return_type_to_lltype`'s Void/Bool-only projection.
pub(crate) fn collect_unsafe_fn_stubs_from_llbc(
    llbc: &Llbc,
) -> Vec<(
    Vec<String>,
    crate::flowspace::argument::Signature,
    crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
)> {
    use crate::flowspace::argument::Signature;
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    let mut out = Vec::new();
    for fd in llbc.iter_local_fns() {
        if !fd.signature.is_unsafe {
            continue;
        }
        // Global initializers are synthetic, not user-callable fns; the
        // MIR-driver lowering loop skips them too.
        if fd.is_global_initializer.is_some() {
            continue;
        }
        // Reference returns (`&bool`, `&()`, …) are not plain unit/bool
        // stubs: `tyref_to_ast_string` strips the reference to its
        // referent, which would misclassify `&bool` as `bool`.  The syn
        // extractor's `simple_return_type_to_lltype` rejects
        // `syn::Type::Reference`, so skip references here to match it.
        if output_type_is_ref(&fd.signature.output, llbc) {
            continue;
        }
        let lltype = match tyref_to_ast_string(&fd.signature.output, llbc).as_str() {
            "()" => LowLevelType::Void,
            "bool" => LowLevelType::Bool,
            _ => continue,
        };
        // Both free functions and impl-owned functions are collected,
        // keyed on `name_path()` — the segment vector
        // `call_target_segments` emits for a `CallKind::Fun(Regular)`
        // (mir.rs:2186).  An impl method usually lowers to
        // `CallTarget::Method` (resolved via the receiver classdef), but a
        // receiver-less associated function (`Owner::new() -> ()/bool`) and
        // any impl method reached through an `FnDef` constant fall back to
        // `CallTarget::FunctionPath { name_path }` (mir.rs:2082, 3995),
        // whose lookup is served only by this registry — skipping impl
        // owners would leave those call sites "not registered".
        let segments: Vec<String> = fd
            .item_meta
            .name_path()
            .split("::")
            .map(String::from)
            .collect();
        // Prefer the Charon body's declared parameter names
        // (`locals[1..=argc]`, the same source the regular lowering reads
        // at `local.name`); fall back to positional `arg{N}` when the body
        // is opaque or a local is unnamed.
        let body = fd.unstructured();
        let argnames: Vec<String> = (0..fd.signature.inputs.len())
            .map(|i| {
                body.as_ref()
                    .and_then(|u| u.locals.locals.get(i + 1))
                    .and_then(|l| l.name.clone())
                    .unwrap_or_else(|| format!("arg{i}"))
            })
            .collect();
        out.push((segments, Signature::new(argnames, None, None), lltype));
    }
    out
}

/// Collect, from the lowered MIR, `(path-segments, Signature, result
/// ValueType)` for every method whose impl-owner is a **foreign opaque**
/// ADT (`malachite_bigint::bigint::BigInt`, …) and whose result type can
/// be modeled faithfully.
///
/// An opaque owner has no extracted body, so the annotator never mints a
/// `ClassDef` for it; the receiver lands as a classdef-less
/// `SomeInstance` and a `CallTarget::Method` getattr panics.  These
/// methods are the foreign helpers the interpreter's cold
/// `@jit.dont_look_inside` overflow→long arms operate on
/// (`<BigInt as Add>::add`, `…::clone`, …), so the faithful treatment is
/// the `register_external` analog: residualize the call rather than trace
/// into the foreign body.  [`Lowering::impl_method_owner`] already
/// declines the `CallTarget::Method` hint for an opaque owner (routing the
/// call through `CallTarget::FunctionPath`); this collection feeds the
/// matching registry entries so the `FunctionPath` lookup resolves instead
/// of raising "not registered in PyreCallRegistry".
///
/// The registration key is derived through the SAME
/// [`impl_method_owner_for_fundecl`] the declined-Method
/// `call_target_segments` arm uses, so the key equals the call-site
/// lookup (`[strip_crate(owner.name_path).split("::"), leaf]` —
/// e.g. `["bigint", "BigInt", "add"]`).
///
/// **Faithful result shell — no blanket Ref.**  The result `ValueType` is
/// read from the method's LLBC output signature:
///
/// - output is a scalar literal (`i64` / `f64` / `bool`) → that
///   `ValueType` (the residual really produces an integer / float / bool);
/// - output is itself a foreign **opaque** ADT (`BigInt` → `BigInt`,
///   the `Add`/`Sub`/`Mul`/`clone` cluster) → `Ref(None)`, the
///   classdef-less `SomeInstance` shell `bigint_from` produces;
/// - anything else — an `Option<i64>` (`to_i64`), an enum (`sign`), a
///   tuple, a reference, a non-opaque ADT — is **declined** (no entry),
///   leaving the method at the original "not registered" Skip.  Modeling
///   an `Option<i64>` return as a bare integer or as `Ref(None)` would
///   mis-type the value and only migrate the failure to a deeper wall, so
///   those methods stay residual until their result type can be modeled.
pub(crate) fn collect_foreign_opaque_method_externals(
    llbc: &Llbc,
) -> Vec<(
    Vec<String>,
    crate::flowspace::argument::Signature,
    ValueType,
)> {
    use crate::flowspace::argument::Signature;
    let mut out = Vec::new();
    for fd in llbc.iter_local_fns() {
        if fd.is_global_initializer.is_some() {
            continue;
        }
        // Owner must be an impl-block method on an opaque ADT, with the
        // owner ADT as the first (`self`) input.  `impl_method_owner_for_fundecl`
        // resolves the owner's qualified name; the explicit opaque-kind +
        // self-receiver checks here mirror the gate `impl_method_owner`
        // applies before declining the Method hint.
        let Some((owner_qualified, leaf)) = impl_method_owner_for_fundecl(llbc, fd) else {
            continue;
        };
        let Some(owner_def_id) = impl_owner_adt_def_id_for_fundecl(llbc, fd) else {
            continue;
        };
        let Some(owner_td) = llbc.type_by_id(owner_def_id) else {
            continue;
        };
        if !matches!(owner_td.kind, TypeDeclKind::Opaque) {
            continue;
        }
        if !first_input_is_adt_free(llbc, fd, owner_def_id) {
            continue;
        }
        // Faithful result shell read from the LLBC output signature; a
        // result type that cannot be modeled (Option / enum / tuple /
        // reference / non-opaque ADT) declines the method.
        let Some(result_ty) = foreign_opaque_method_result_valuetype(&fd.signature.output, llbc)
        else {
            continue;
        };
        let mut segments: Vec<String> = owner_qualified.split("::").map(str::to_string).collect();
        segments.push(leaf);
        let body = fd.unstructured();
        let argnames: Vec<String> = (0..fd.signature.inputs.len())
            .map(|i| {
                body.as_ref()
                    .and_then(|u| u.locals.locals.get(i + 1))
                    .and_then(|l| l.name.clone())
                    .unwrap_or_else(|| format!("arg{i}"))
            })
            .collect();
        out.push((segments, Signature::new(argnames, None, None), result_ty));
    }
    out
}

/// Resolve the impl-owner ADT `def_id` of an impl-block method `fd`
/// directly from its `<Impl>` NameSeg, the free-function twin of
/// [`Lowering::resolve_impl_owner_adt_def_id`] over the `<Impl>` segment.
fn impl_owner_adt_def_id_for_fundecl(llbc: &Llbc, fd: &FunDecl) -> Option<u64> {
    let segs = &fd.item_meta.name;
    let last_idx = segs
        .iter()
        .rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
    if last_idx == 0 {
        return None;
    }
    let impl_payload = match &segs[last_idx - 1] {
        NameSeg::Other(v) => v.as_object()?.get("Impl")?,
        _ => return None,
    };
    resolve_impl_owner_adt_def_id_free(llbc, impl_payload)
}

/// Free-function twin of [`Lowering::first_input_is_adt`]: true when the
/// method's first input (`self`, possibly behind `&`/`&mut`/`*`) is the
/// owner ADT.
fn first_input_is_adt_free(llbc: &Llbc, fd: &FunDecl, adt_def_id: u64) -> bool {
    fd.signature
        .inputs
        .first()
        .and_then(|t| tyref_node(t, llbc))
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(adt_node_def_id)
        .is_some_and(|id| id == adt_def_id)
}

/// Faithful result `ValueType` for a residualized foreign-opaque method,
/// or `None` to decline (see [`collect_foreign_opaque_method_externals`]).
/// A scalar literal output keeps its `ValueType`; an opaque-ADT output
/// projects to `Ref(None)`; every other shape (`Option`, enum, tuple,
/// reference, non-opaque ADT) is declined.
fn foreign_opaque_method_result_valuetype(output: &TyRef, llbc: &Llbc) -> Option<ValueType> {
    // A reference return (`&T`) is not the owned residual result the
    // stub models; decline.
    if output_type_is_ref(output, llbc) {
        return None;
    }
    match tyref_to_value_type(output, llbc) {
        // Scalar literal results are produced directly by the residual.
        vt @ (ValueType::Int | ValueType::Unsigned | ValueType::Float | ValueType::Bool) => {
            Some(vt)
        }
        // A `Ref` projection covers every non-scalar ADT shape (`BigInt`,
        // `Option<i64>`, tuples, …).  Accept it ONLY when the result ADT
        // is itself a foreign opaque type (the `BigInt`-returning
        // arithmetic cluster), which the classdef-less `SomeInstance`
        // shell models faithfully.  A non-opaque ADT (`Option`, an enum)
        // would be mis-typed as an opaque GcRef, so decline it.
        ValueType::Ref(_) => {
            let def_id = output_adt_def_id_free(output, llbc)?;
            let td = llbc.type_by_id(def_id)?;
            matches!(td.kind, TypeDeclKind::Opaque).then_some(ValueType::Ref(None))
        }
        _ => None,
    }
}

/// Free-function twin of [`Lowering::tyref_adt_def_id`]: resolve a
/// `TyRef`'s ADT `def_id`, following the dedup index for a `Dedup` shape.
fn output_adt_def_id_free(ty: &TyRef, llbc: &Llbc) -> Option<u64> {
    match ty {
        TyRef::Inline { value: (_, v) } | TyRef::Other(v) => inline_adt_def_id(v),
        TyRef::Dedup { id } => llbc.dedup_to_adt_def_id(*id),
    }
}

/// Free-function version of [`Lowering::resolve_impl_owner_adt_def_id`].
fn resolve_impl_owner_adt_def_id_free(
    llbc: &Llbc,
    impl_payload: &serde_json::Value,
) -> Option<u64> {
    let obj = impl_payload.as_object()?;
    if let Some(ty) = obj.get("Ty") {
        let sb = ty.as_object()?.get("skip_binder")?;
        return resolve_tyexpr_to_adt_def_id_free(llbc, sb);
    }
    if let Some(trait_impl_id) = obj.get("Trait").and_then(serde_json::Value::as_u64) {
        let trait_impls = llbc.trait_impls_raw();
        let ti = trait_impls.get(trait_impl_id as usize)?;
        let first_ty = ti
            .as_object()?
            .get("impl_trait")?
            .as_object()?
            .get("generics")?
            .as_object()?
            .get("types")?
            .as_array()?
            .first()?;
        return resolve_tyexpr_to_adt_def_id_free(llbc, first_ty);
    }
    None
}

/// When `fd` is a trait-impl method (i.e. its NameSeg's penultimate
/// segment is an `Impl` with a `{"Trait": <trait_impl_id>}` payload),
/// return the implemented trait's leaf identifier.  Returns `None`
/// for free functions, inherent impl methods, and trait default
/// bodies (those carry the trait name directly in `name_path()`'s
/// penultimate segment, so the caller can read it through
/// [`trait_method_owner`] without a `trait_impls` indirection).
///
/// Used by `build_semantic_program_from_llbc` to populate
/// `SemanticFunction.trait_root` (leaf) and `trait_qualified` (this
/// fn's return value, the full `name_path()`) so the canonical
/// registration loop can call `CallControl::register_trait_method`
/// instead of routing through `extract_trait_impls`, and so the
/// unique-impl map can key on trait identity rather than a bare leaf
/// (two distinct traits may share a final segment).
fn trait_impl_trait_path_for_fundecl(llbc: &Llbc, fd: &FunDecl) -> Option<String> {
    let segs = &fd.item_meta.name;
    let last_idx = segs
        .iter()
        .rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
    if last_idx == 0 {
        return None;
    }
    let impl_payload = match &segs[last_idx - 1] {
        NameSeg::Other(v) => v.as_object()?.get("Impl")?,
        _ => return None,
    };
    let trait_impl_id = impl_payload
        .as_object()?
        .get("Trait")
        .and_then(serde_json::Value::as_u64)?;
    let trait_impls = llbc.trait_impls_raw();
    let ti = trait_impls.get(trait_impl_id as usize)?;
    // `impl_trait` is a TraitDeclRef `{"id": <trait_decl_id>,
    // "generics": {...}}` — same shape `resolve_impl_owner_adt_def_id`
    // reads `generics.types[0]` from.
    let trait_id = ti
        .as_object()?
        .get("impl_trait")?
        .as_object()?
        .get("id")?
        .as_u64()?;
    let td = llbc.trait_by_id(trait_id)?;
    // Only source-local traits participate: std/core trait impls
    // (`FnOnce` closure shims, `Destruct`/`Drop` glue, `PartialEq`, …)
    // are host machinery, not translated-program classes, and keep
    // their inherent classification.
    if !td.item_meta.is_local {
        return None;
    }
    let trait_path = td.item_meta.name_path();
    if trait_path.is_empty() {
        return None;
    }
    Some(trait_path)
}

/// Detect a trait-default body — a function whose penultimate NameSeg
/// is a bare `Ident` matching a known trait leaf (no `Impl` segment).
/// Charon emits trait default impls inline in the trait's namespace,
/// so they look like `pyre_interpreter::pyopcode::LocalOpcodeHandler::
/// load_local_checked_value` with the trait leaf as the parent ident.
///
/// Returns the trait leaf so `build_semantic_program_from_llbc` can
/// populate `SemanticFunction.trait_root` and the canonical
/// registration loop (`lib.rs:985-1141`) can find the body without
/// going through `extract_trait_impls`'s `<default methods of <T>>`
/// pseudo-impl-type detour.
fn trait_default_owner_for_fundecl(
    fd: &FunDecl,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    let (parent, _leaf) = trait_method_owner(fd)?;
    if known_trait_names.contains(&parent) {
        Some(parent)
    } else {
        None
    }
}

/// Free-function version of [`Lowering::resolve_tyexpr_to_adt_def_id`].
fn resolve_tyexpr_to_adt_def_id_free(llbc: &Llbc, ty: &serde_json::Value) -> Option<u64> {
    let obj = ty.as_object()?;
    if let Some(arr) = obj
        .get("HashConsedValue")
        .and_then(serde_json::Value::as_array)
        && let Some(body) = arr.get(1)
    {
        return inline_adt_def_id(body);
    }
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return llbc.dedup_to_adt_def_id(id);
    }
    None
}

/// Canonicalise a Charon `BinaryOp` tag (PascalCase + JSON-tagged
/// variants) to the RPython-style snake_case label the codewriter
/// expects.  After this the assembler's `op_kind_to_opname` reaches the
/// already-wired `int_{label}` / `ptr_{label}` keys instead of inventing
/// `int_AddChecked` / `int_BitAnd` shapes that have no blackhole handler.
///
/// Mapping reflects RPython's `jtransform` / `rint` /  `rptr` rewrites:
///   - `Add` / `Sub` / `Mul` plain → `add`/`sub`/`mul` (wrapping arith).
///   - `*Checked` variants → `*_ovf` (overflow-guarded arith, paired
///     with `guard_no_overflow` downstream).
///   - Shift `*Wrap` / `*Checked` collapse onto the canonical
///     `lshift`/`rshift` (RPython treats them identically because
///     shifts cannot overflow into a different repr).
///   - `BitAnd` / `BitOr` / `BitXor` → `and`/`or`/`xor` to match
///     `blackhole.py:500` canonical bitwise opnames.
///   - Comparisons `Eq` / `Ne` / `Lt` / `Le` / `Gt` / `Ge` pass through
///     as lowercase; the assembler later branches on operand kind
///     (`ii` → `int_eq`, `rr` → `ptr_eq`, …).
fn binop_label(v: &serde_json::Value) -> Result<String, LowerError> {
    if let Some(s) = v.as_str() {
        return Ok(canonical_binop_label(s, None));
    }
    if let Some(obj) = v.as_object() {
        if let Some((k, payload)) = obj.iter().next() {
            let suffix = match payload {
                serde_json::Value::String(s) => Some(s.as_str()),
                _ => None,
            };
            return Ok(canonical_binop_label(k, suffix));
        }
    }
    Err(LowerError::Schema(format!(
        "BinaryOp op label has unexpected shape: {v}"
    )))
}

/// Charon's `UnaryOp` tag → canonical RPython unary opname.  Plain
/// atoms (`"Neg"`, `"Not"`) share `binop_label`'s mapping; tagged
/// `{"Cast": {...}}` payloads encode the source/dest scalar shape and
/// project onto `cast_int_to_float` / `cast_float_to_int` /
/// `cast_int_to_ptr` / `cast_ptr_to_int` per `blackhole.py:603-816`.
/// Cast shapes the JIT models as identity (RawPtr→RawPtr,
/// Scalar Int↔UInt of the same width, Unsize) collapse to `same_as`
/// so the assembler emits the per-kind copy op instead of an unwired
/// `int_unary.*` shape.
/// `true` when a `Rvalue::UnaryOp` op payload is a `Cast` (the JSON
/// object `{"Cast": ..}`) rather than `Neg` / `Not`. Casts alias the
/// operand instead of emitting an `OpKind::UnaryOp`.
fn unary_op_is_cast(v: &serde_json::Value) -> bool {
    v.as_object()
        .and_then(|o| o.keys().next())
        .is_some_and(|k| k == "Cast")
}

fn unary_op_label(v: &serde_json::Value) -> Result<String, LowerError> {
    if let Some(s) = v.as_str() {
        return Ok(canonical_binop_label(s, None));
    }
    let Some(obj) = v.as_object() else {
        return Err(LowerError::Schema(format!(
            "UnaryOp op label has unexpected shape: {v}"
        )));
    };
    let Some((tag, payload)) = obj.iter().next() else {
        return Err(LowerError::Schema("UnaryOp object is empty".into()));
    };
    match tag.as_str() {
        "Cast" => Ok(cast_label_from_payload(payload)),
        _ => {
            let suffix = payload.as_str();
            Ok(canonical_binop_label(tag, suffix))
        }
    }
}

/// Translate a Charon `CastKind` JSON payload into a canonical RPython
/// cast opname.  `Scalar([Int, Float])` (and the float-to-int reverse)
/// drive `bhimpl_cast_int_to_float` / `bhimpl_cast_float_to_int`; ptr
/// casts go through `bhimpl_cast_{int_to_ptr,ptr_to_int}`.  Same-repr
/// casts (RawPtr→RawPtr, same-width Int↔UInt, Unsize) are JIT-no-ops
/// → `same_as` (the assembler's per-kind copy fallback).  Variants the
/// JIT does not model (`VTable` / `VTableUpcast`) remain identifiable
/// in the unwired diagnostic via the lower-cased default.
fn cast_label_from_payload(payload: &serde_json::Value) -> String {
    let Some(obj) = payload.as_object() else {
        return "same_as".into();
    };
    let Some((kind, inner)) = obj.iter().next() else {
        return "same_as".into();
    };
    match kind.as_str() {
        // `Scalar([src, dst])` — int↔float crossings surface as the
        // canonical RPython cast opnames; int↔uint of any width is a
        // JIT-no-op (`same_as` copies the i64 carrier).
        "Scalar" => {
            let arr = match inner.as_array() {
                Some(a) if a.len() == 2 => a,
                _ => return "same_as".into(),
            };
            let src_is_float = scalar_is_float(&arr[0]);
            let dst_is_float = scalar_is_float(&arr[1]);
            match (src_is_float, dst_is_float) {
                (true, false) => "cast_float_to_int".into(),
                (false, true) => "cast_int_to_float".into(),
                _ => "same_as".into(),
            }
        }
        // `RawPtr([_, _])` — pointer-to-pointer reinterpret; same i64
        // machine repr, so the JIT copies through `same_as`.
        "RawPtr" => "same_as".into(),
        // `Unsize` produces a fat pointer at the source level; the JIT
        // models the array head as a single Ref so this is a no-op.
        "Unsize" => "same_as".into(),
        // `FnPtr` / `Transmute` / `VTable*` etc. — preserve a stable
        // identifier so the unwired diagnostic surfaces the shape.
        _ => kind.to_lowercase(),
    }
}

/// `true` for the Charon `CastKind::RawPtr` payload (atom or
/// single-key object form) — the pointer-to-pointer reinterpret
/// `expr as *const T` / `as *mut T`.
fn cast_kind_is_raw_ptr(kind: &serde_json::Value) -> bool {
    kind.as_str() == Some("RawPtr") || kind.as_object().is_some_and(|o| o.contains_key("RawPtr"))
}

fn scalar_is_float(v: &serde_json::Value) -> bool {
    if let Some(s) = v.as_str() {
        return matches!(s, "F32" | "F64");
    }
    if let Some(obj) = v.as_object() {
        if obj.contains_key("Float") {
            return true;
        }
    }
    false
}

fn canonical_binop_label(tag: &str, subkind: Option<&str>) -> String {
    // Charon emits `*Checked` (Rust debug-mode trap-on-overflow) and
    // `*Wrap` (release-mode wrapping) variants either as single
    // PascalCase atoms (`"AddChecked"`, `"ShrWrap"`) or as tagged
    // objects (`{"Add": "Checked"}`); both forms collapse onto the
    // plain RPython opname because the JIT does not model Rust's
    // debug-trap semantics — overflow guarding belongs to the
    // optimizer / `guard_no_overflow` level (`pure.rs:int_add_ovf`)
    // and is not emitted from MIR rvalues.
    match (tag, subkind) {
        // Arithmetic (atomic + tagged).
        ("Add" | "AddChecked" | "AddWrap", _) => "add".into(),
        ("Sub" | "SubChecked" | "SubWrap", _) => "sub".into(),
        ("Mul" | "MulChecked" | "MulWrap", _) => "mul".into(),
        ("Div", _) => "floordiv".into(),
        ("Rem", _) => "mod".into(),
        // Bitwise.  The canonical pyre labels carry the `bit` prefix so
        // `codewriter::jtransform` (`bitand`/`bitor`/`bitxor` arm) and
        // the rtyper adapter `normalize_binop_name` (`bitand`->`and_`,
        // `bitor`->`or_`, `bitxor`->`xor`) recognise them.  Bare `and`/`or`
        // are reserved for short-circuit control flow, which never reaches
        // here: rustc lowers `&&`/`||` to branches before charon.
        ("BitAnd", _) => "bitand".into(),
        ("BitOr", _) => "bitor".into(),
        ("BitXor", _) => "bitxor".into(),
        // Shifts.
        ("Shl" | "ShlChecked" | "ShlWrap", _) => "lshift".into(),
        ("Shr" | "ShrChecked" | "ShrWrap", _) => "rshift".into(),
        // Comparisons.
        ("Eq", _) => "eq".into(),
        ("Ne", _) => "ne".into(),
        ("Lt", _) => "lt".into(),
        ("Le", _) => "le".into(),
        ("Gt", _) => "gt".into(),
        ("Ge", _) => "ge".into(),
        // Unary tags surface here through `Rvalue::UnaryOp`.
        ("Neg", _) => "neg".into(),
        ("Not", _) => "invert".into(),
        // Default: lower-case the tag + subkind so unknown shapes
        // remain identifiable in `unwired` diagnostics.
        _ => match subkind {
            Some(s) => format!("{}_{}", tag.to_lowercase(), s.to_lowercase()),
            None => tag.to_lowercase(),
        },
    }
}

/// Best-effort name for an [`Rvalue::Aggregate`]'s constructor, used as
/// the [`CallTarget::SyntheticTransparentCtor::name`] string.  Shape is
/// either an enum-tag object (`{"Adt": {...}}`, `{"Tuple": null}`,
/// `{"Array": null}`) or a bare string.  We project a stable label per
/// kind so debug output is readable; the codewriter does not yet route
/// on these names.
fn aggregate_ctor_name(kind: &serde_json::Value) -> String {
    if let Some(s) = kind.as_str() {
        return s.to_string();
    }
    if let Some(obj) = kind.as_object() {
        // `{"Adt": [{"id": "Tuple" | {"Builtin": "Array"}}, ...]}` — an
        // aggregate whose type id is a builtin container atom rather
        // than a resolvable ADT def_id.  Name the placeholder after the
        // id atom so every tuple/array site shares one spelling; the
        // wrapper key "Adt" would mint a fresh same-named class per
        // site (the adapter's bare-leaf arm does not intern), and two
        // `()` values meeting at a join then fail to union ("RPython
        // cannot unify instances with no common base class").
        if let Some(id) = obj
            .get("Adt")
            .and_then(serde_json::Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(serde_json::Value::as_object)
            .and_then(|m| m.get("id"))
        {
            if let Some(atom) = id.as_str() {
                return atom.to_string();
            }
            if let Some(builtin) = id
                .as_object()
                .and_then(|m| m.get("Builtin"))
                .and_then(serde_json::Value::as_str)
            {
                return builtin.to_string();
            }
        }
        if let Some(k) = obj.keys().next() {
            return k.clone();
        }
    }
    "ctor".to_string()
}

/// Project a `HashConsedValue` body to the underlying ADT
/// `def_id` when the body has shape `{"Adt": {"id": {"Adt": <def_id>}}}`.
/// Mirrors the reader's private helper used to build
/// `Llbc::dedup_to_adt_def_id`; reproduced here because the inline
/// arm of [`Lowering::resolve_tyexpr_to_adt_def_id`] decodes the
/// same body shape without going through the dedup cache.
fn inline_adt_def_id(body: &serde_json::Value) -> Option<u64> {
    body.as_object()?
        .get("Adt")?
        .as_object()?
        .get("id")?
        .as_object()?
        .get("Adt")?
        .as_u64()
}

/// Clone a [`TyRef`] (no `Clone` impl on the schema enum).  Used by
/// [`Lowering::resolve_adt_field`] when handing the resolved field's
/// type to [`tyref_to_value_type`].
fn clone_tyref(ty: &TyRef) -> TyRef {
    match ty {
        TyRef::Dedup { id } => TyRef::Dedup { id: *id },
        TyRef::Inline { value: (id, v) } => TyRef::Inline {
            value: (*id, v.clone()),
        },
        TyRef::Other(v) => TyRef::Other(v.clone()),
    }
}

/// Project a Charon [`TyRef`] to the JIT-visible [`ValueType`].
///
/// Numeric scalars → `Int` / `Float`, bool → `Bool`, unit → `Void`,
/// everything else (structs, pointers, references) → `Ref`.  The
/// TyRef's serialized form is the source of truth —
/// `TyRef::label()` produces a compact short form
/// (`"ty#170"`, `"ty<Adt>"`) for opaque IDs, while the underlying
/// JSON carries the primitive name for literal types.
///
/// For `TyRef::Deduplicated{id}`, the projection consults
/// `llbc.dedup_body(id)` to recover the inline body shape and runs
/// the same primitive-pattern match.  Required so FunDecl return
/// types serialized as `Deduplicated` (≈92% in `pyre-interpreter.ullbc`)
/// resolve to `Int` / `Bool` / `Float` instead of falling back to
/// `Ref`.
/// The JIT register bank a [`ValueType`] occupies, mirroring
/// `flatten.py getkind`: the integer family (`Int` / `Unsigned` /
/// `Bool`) shares the `'int'` bank, `Ref` the `'ref'` bank, `Float` the
/// `'float'` bank.  Non-value kinds (`Void` / `State` / `Unknown`) get a
/// distinct discriminant so they never compare equal to a real bank.
fn value_type_bank(ty: &ValueType) -> u8 {
    match ty {
        ValueType::Int | ValueType::Unsigned | ValueType::Bool => 0,
        ValueType::Ref(_) => 1,
        ValueType::Float => 2,
        ValueType::Void => 3,
        ValueType::State => 4,
        ValueType::Unknown => 5,
    }
}

/// The fully-qualified host-callable path for a bank-crossing cast, or
/// `None` when `src` and `dst` share a bank (a JIT no-op aliased to the
/// operand) or the bank pair has no host cast callable.  A bank crossing
/// lowers to `simple_call(<host_callable>, v)`, never an `OpKind::UnaryOp`:
/// the rtyper retired every typed cast opname from the unary-op path
/// (`flowspace_adapter::normalize_unary_op_name` accepts only
/// `neg` / `bool` / `invert` / `same_as`), so the only surface that reaches
/// `rtype_cast_int_to_ptr` / `rtype_cast_ptr_to_int` (rbuiltin.py:543-557)
/// and `rtype_builtin_float` / `rtype_builtin_int` (rbuiltin.py:178-189,
/// which delegate to `rtype_float` / `rtype_int` and emit the low-level
/// `cast_int_to_float` / `cast_float_to_int`) is a `simple_call`.  `int →
/// ptr` / `ptr → int` resolve the `lltype.cast_*` module attr
/// (`["rpython", "rtyper", "lltypesystem", "lltype", …]` per
/// `flowspace_adapter` Branch 3b); `int → float` / `float → int` resolve
/// the bare `float` / `int` builtins (single-segment `HOST_ENV.\
/// lookup_builtin`).
fn cast_call_segments(src: &ValueType, dst: &ValueType) -> Option<Vec<String>> {
    let (s, d) = (value_type_bank(src), value_type_bank(dst));
    let lltype = |name: &str| -> Vec<String> {
        ["rpython", "rtyper", "lltypesystem", "lltype", name]
            .into_iter()
            .map(str::to_string)
            .collect()
    };
    match (s, d) {
        _ if s == d => None,
        // int → ptr / ptr → int — `lltype.cast_int_to_ptr` /
        // `lltype.cast_ptr_to_int`.
        (0, 1) => Some(lltype("cast_int_to_ptr")),
        (1, 0) => Some(lltype("cast_ptr_to_int")),
        // int → float / float → int — `float(v)` / `int(v)`, whose
        // rtyper delegates to `rtype_float` / `rtype_int`.
        (0, 2) => Some(vec!["float".to_string()]),
        (2, 0) => Some(vec!["int".to_string()]),
        // No host callable for the remaining pairs (e.g. ref↔float, or any
        // pair touching Void/State/Unknown): alias the operand.
        _ => None,
    }
}

fn tyref_to_value_type(ty: &TyRef, llbc: &Llbc) -> ValueType {
    // The HashConsedValue arm carries the body inline; primitives
    // typically land here.  The Deduplicated arm carries only an
    // ID; consult the dedup-body index to recover the inline shape
    // when it was recorded.  Ids never seen inline (or scanned out
    // of order by the reader) fall back to `Ref` — the same
    // projection downstream uses for any non-primitive shape.
    let value = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return ValueType::Ref(None),
        },
    };
    // Primitive shapes Charon emits inline.  The literal-type schema
    // splits across two forms:
    //
    //   - atom: `{"Literal": "Bool"}`, `{"Literal": "Char"}`.
    //   - object: `{"Literal": {"Int": "Isize"}}`,
    //     `{"Literal": {"UInt": "Usize"}}`,
    //     `{"Literal": {"Float": "F64"}}`.
    //
    // A single `{"Literal": {"Integer": …}}` shape is also accepted so
    // .ullbc artefacts that use it still resolve.
    //
    // Unit type `()` serializes as `{"Adt": {"id": "Tuple",
    // "generics": {"types": []}}}` and routes through the final `Ref`
    // fallback here.  A `-> ()` function's *return* is special-cased
    // separately by [`is_unit_type`] at the `Return` terminator so the
    // result kind comes out void ('v'); in operand position a unit
    // value stays Ref like any other transparent-ctor result.
    if let Some(obj) = value.as_object()
        && let Some(lit) = obj.get("Literal")
    {
        if let Some(lit_atom) = lit.as_str() {
            return match lit_atom {
                "Bool" => ValueType::Bool,
                "Char" => ValueType::Int,
                _ => ValueType::Ref(None),
            };
        }
        if let Some(lit_obj) = lit.as_object() {
            if lit_obj.contains_key("Int")
                || lit_obj.contains_key("UInt")
                || lit_obj.contains_key("Integer")
                || lit_obj.contains_key("Char")
            {
                return ValueType::Int;
            }
            if lit_obj.contains_key("Bool") {
                return ValueType::Bool;
            }
            if let Some(f) = lit_obj.get("Float").and_then(serde_json::Value::as_str) {
                // `getkind(SingleFloat) == 'int'` (history.py:53): single
                // floats live in the int register bank; only `f64`
                // (`lltype.Float`) keeps the float kind.
                return if f == "F32" {
                    ValueType::Int
                } else {
                    ValueType::Float
                };
            }
        }
    }
    // Non-`Literal` (ADT / pointer / tuple) shapes only.  Atomic wrappers
    // type as their inner value; integer widths collapse to `Int` here,
    // matching the literal-int handling above.  Checked after the cheap
    // `Literal` fast-path so primitive operands never pay the lookup.
    if let Some(inner) = tyref_atomic_inner_value_type(ty, llbc) {
        return match inner {
            ValueType::Unsigned => ValueType::Int,
            other => other,
        };
    }
    // `OpArg` is the transparent `struct OpArg(u32)` raw-operand wrapper;
    // its external decl is `Opaque`, so it would otherwise fall to
    // `Ref(None)` and shell to a classdef-less instance.  Model it as the
    // bare integer operand it carries (`tyref_is_oparg`).
    if tyref_is_oparg(ty, llbc) {
        return ValueType::Int;
    }
    ValueType::Ref(None)
}

/// Classify a struct field [`TyRef`] into the RPython `lltype` register
/// class the annotator pre-fills into `FORCE_ATTRIBUTES_INTO_CLASSES`.
///
/// Unlike [`tyref_to_value_type`] (which collapses every integer width to
/// `Int`), this keeps the signed/unsigned split: `{"Literal": {"UInt":
/// …}}` shells to [`ValueType::Unsigned`] so `valuetype_to_someshell`
/// picks `SomeInteger { unsigned: true }`, matching the per-field shells
/// the syn classifier produced for `u8`..`usize`.  `char` and every
/// signed width fold to `Int`; `bool`/`float` keep their classes; every
/// non-primitive shape (named struct/enum, reference, raw pointer, tuple,
/// slice, array, `Box`/`Rc`/`Arc` wrapper) folds to `Ref(None)` whose
/// someshell ignores the payload.
fn tyref_to_attr_value_type(ty: &TyRef, llbc: &Llbc) -> ValueType {
    let value = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return ValueType::Ref(None),
        },
    };
    if let Some(obj) = value.as_object()
        && let Some(lit) = obj.get("Literal")
    {
        if let Some(lit_atom) = lit.as_str() {
            return match lit_atom {
                "Bool" => ValueType::Bool,
                "Char" => ValueType::Int,
                _ => ValueType::Ref(None),
            };
        }
        if let Some(lit_obj) = lit.as_object() {
            if lit_obj.contains_key("UInt") {
                return ValueType::Unsigned;
            }
            if lit_obj.contains_key("Int")
                || lit_obj.contains_key("Integer")
                || lit_obj.contains_key("Char")
            {
                return ValueType::Int;
            }
            if lit_obj.contains_key("Bool") {
                return ValueType::Bool;
            }
            if let Some(f) = lit_obj.get("Float").and_then(serde_json::Value::as_str) {
                // `getkind(SingleFloat) == 'int'`: single floats are
                // int-banked; only `f64` keeps the float kind.
                return if f == "F32" {
                    ValueType::Int
                } else {
                    ValueType::Float
                };
            }
        }
    }
    // Non-`Literal` (ADT / pointer / tuple) shapes only.  Atomic wrappers
    // type as their inner value; unlike [`tyref_to_value_type`] the
    // signed/unsigned split is kept here (`AtomicUsize` → `Unsigned`) so
    // the per-field someshell matches the inner scalar.  Checked after
    // the cheap `Literal` fast-path so primitive fields never pay it.
    if let Some(inner) = tyref_atomic_inner_value_type(ty, llbc) {
        return inner;
    }
    ValueType::Ref(None)
}

/// The bare leaf name of `ty`'s named-ADT root, after stripping
/// reference wrappers (`&T` / `&mut T` → `T`, the same contract as
/// [`tyref_to_ast_string`]).  This is the value `OpKind::Input.class_root`
/// carries so `derive_subject_inputcells`
/// (`flowspace_adapter.rs:1860-1885`) can seed a `Ref` parameter with
/// its cached struct-root `ClassDef` instead of the classdef-less
/// `SomeInstance` shell.
///
/// Raw pointers (`*const T` / `*mut T`) resolve to their pointee's
/// ADT root: a `*mut W_Foo` parameter is the same instance-lattice
/// value as a `&W_Foo` one (upstream `SomePtr(PTRTYPE)` carries the
/// pointee type either way).  The pointer-method answer (`is_null`)
/// stays intact — `SomeInstance.getattr` resolves it as a bound
/// method BEFORE projecting the classdef (`unaryop.rs:3664`), so a
/// seeded classdef no longer bypasses it.
///
/// Returns `None` for:
///   - primitives / tuples / builtin containers (no class root);
///   - generic ADT instantiations (`Arg<u32>`) — the registry rows for
///     a generic decl carry unresolved type-variable field strings, so
///     a seeded classdef would project bogus attr shells.
fn tyref_class_root(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    let node = strip_ty_wrappers(tyref_node(ty, llbc)?, llbc)?;
    adt_node_class_root(node, llbc).or_else(|| raw_ptr_pointee_class_root(node, llbc))
}

/// The underlying JSON type node of a `TyRef`, resolving the `Dedup`
/// indirection through the LLBC dedup-body index.
fn tyref_node<'l>(ty: &'l TyRef, llbc: &'l Llbc) -> Option<&'l serde_json::Value> {
    match ty {
        TyRef::Inline { value: (_, v) } => Some(v),
        TyRef::Other(v) => Some(v),
        TyRef::Dedup { id } => llbc.dedup_body(*id),
    }
}

/// Whether a `TyRef` resolves (behind the usual wrappers) to the
/// `alloc::string::String` ADT.
fn tyref_is_string_adt(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(adt_node_def_id)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| td.item_meta.name_path() == "alloc::string::String")
}

/// Whether a `TyRef` resolves (behind `Ref`/dedup/hash-cons wrappers) to
/// the `str` builtin — the unsized string slice (`{"Builtin": "Str"}`).
/// Distinct from `[u8]` (the `Slice` builtin) and from `[T]`/`Box`/named
/// ADTs, so it pins the string-family result of a `String -> &str` view
/// apart from `String`'s sibling `AsRef<[u8]>` / `AsRef<OsStr>` impls.
fn tyref_strips_to_str(ty: &TyRef, llbc: &Llbc) -> bool {
    tyref_node(ty, llbc)
        .and_then(|n| strip_ty_wrappers(n, llbc))
        .and_then(|n| {
            n.as_object()?
                .get("Adt")?
                .as_object()?
                .get("id")?
                .as_object()
        })
        .and_then(|id| id.get("Builtin"))
        .and_then(serde_json::Value::as_str)
        == Some("Str")
}

/// Whether a `TyRef` resolves (behind `Ref` / dedup wrappers) to a
/// string-family value: the `alloc::string::String` ADT, the
/// `rustpython_wtf8` `Wtf8` / `Wtf8Buf` wrappers, or the `{Builtin: "Str"}`
/// node (the bare `str` deref destination).  All four project to the
/// single immutable `s_unicode0` value, so a `deref` between any two of
/// them is value-identity.
fn tyref_is_string_value(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) else {
        return false;
    };
    // `{Builtin: "Str"}` — the bare `str` value (no ADT def-id).
    if node
        .get("Adt")
        .and_then(|a| a.get("id"))
        .and_then(|id| id.get("Builtin"))
        .and_then(serde_json::Value::as_str)
        == Some("Str")
    {
        return true;
    }
    // Named string ADTs: `alloc::string::String` and the WTF-8 wrappers.
    adt_node_def_id(node)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|td| {
            let np = td.item_meta.name_path();
            np == "alloc::string::String"
                || matches!(np.rsplit("::").next(), Some("Wtf8" | "Wtf8Buf"))
        })
}

/// Whether `reg` resolves to a `Deref::deref` / `DerefMut::deref_mut`
/// leaf, by the callee's `name_path` suffix.  Unlike `deref_cast_root`
/// this makes no claim about the dereferenced type — the caller pairs it
/// with a `tyref_is_string_value` receiver / dest gate.
fn is_deref_call(reg: &RegularCall, llbc: &Llbc) -> bool {
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return false;
    };
    llbc.fn_by_id(*id).is_some_and(|fd| {
        let np = fd.item_meta.name_path();
        np.ends_with("::deref") || np.ends_with("::deref_mut")
    })
}

/// The qualified declaration path of a `TyRef`'s base ADT, after
/// stripping `Ref` / hash-cons wrappers.  `None` for non-ADT types.
fn adt_path_of_tyref(ty: &TyRef, llbc: &Llbc) -> Option<String> {
    let node = tyref_node(ty, llbc)?;
    let node = strip_ty_wrappers(node, llbc)?;
    let id = adt_node_def_id(node)?;
    Some(llbc.type_by_id(id)?.item_meta.name_path())
}

/// The inner [`ValueType`] of a `core::sync::atomic` atomic type
/// (`AtomicI64` → `Int`, `AtomicUsize` → `Unsigned`, `AtomicBool` →
/// `Bool`, `AtomicPtr<T>` → `Ref(None)`), or `None` when `ty` is not a
/// std atomic.  The atomic wrappers are layout-transparent over their
/// inner scalar/pointer (asserted at pyobject.rs for the `PyType` vtable
/// `subclassrange_*` / `instantiate` fields), and the upstream typeptr
/// ranges are plain int fields, so a field of one types as that inner
/// value and its `load`/`store` fold to it (`is_atomic_load`).
fn tyref_atomic_inner_value_type(ty: &TyRef, llbc: &Llbc) -> Option<ValueType> {
    let node = strip_ty_wrappers(tyref_node(ty, llbc)?, llbc)?;
    let id = adt_node_def_id(node)?;
    let name = &llbc.type_by_id(id)?.item_meta.name;
    // Cheap leaf check first — no path-string allocation (unlike
    // `name_path` / `adt_path_of_tyref`).  This runs on the hot
    // `tyref_to_value_type` fallback path, so it must stay
    // allocation-free: bail before the module scan unless the type's
    // last segment is an `Atomic*` ident.
    let leaf = match name.last()? {
        NameSeg::Ident { ident: (s, _) } => s.as_str(),
        NameSeg::Other(_) => return None,
    };
    if !leaf.starts_with("Atomic") {
        return None;
    }
    // Confirm std's `core::sync::atomic` module so a user type
    // coincidentally named `Atomic*` does not match.
    let in_atomic_mod = name
        .iter()
        .any(|s| matches!(s, NameSeg::Ident { ident: (id, _) } if id == "atomic"));
    if !in_atomic_mod {
        return None;
    }
    match leaf {
        "AtomicPtr" => Some(ValueType::Ref(None)),
        "AtomicBool" => Some(ValueType::Bool),
        l if l.starts_with("AtomicI") => Some(ValueType::Int),
        l if l.starts_with("AtomicU") => Some(ValueType::Unsigned),
        _ => None,
    }
}

/// `Arg<T>` from `rustpython_compiler_core::bytecode::instruction` —
/// the zero-sized oparg marker (`pub struct Arg<T: OpArgType>
/// (PhantomData<T>)`, instruction.rs:1262).  The external decl is
/// Opaque in the LLBC, so a payload row spelled through it would
/// project to an attr the annotator cannot type; the lifted model
/// carries the marker as a plain integer instead.  Its consumer
/// `Arg::get` keeps its ordinary (residual) call lowering — the ZST
/// marker is never dereferenced.
fn tyref_is_bytecode_arg_marker(ty: &TyRef, llbc: &Llbc) -> bool {
    adt_path_of_tyref(ty, llbc)
        .is_some_and(|p| p == "rustpython_compiler_core::bytecode::instruction::Arg")
}

/// Whether a `TyRef` resolves (behind reference wrappers) to the
/// `rustpython_compiler_core::bytecode::oparg::OpArg` newtype.  `OpArg`
/// is the transparent `struct OpArg(u32)` raw-operand wrapper; its
/// external decl is `Opaque` in the LLBC, so a payload row spelled
/// through it would project to an attr the annotator cannot type.  The
/// value is only ever the raw u32 operand (constructed from a `u32`,
/// consumed by `Arg::get` / `u32::from`), so the lifted model carries it
/// as a plain integer wherever it appears — matching upstream, where the
/// bytecode operand is a bare int with no wrapper type.
fn tyref_is_oparg(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(node) = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) else {
        return false;
    };
    let Some(id) = adt_node_def_id(node) else {
        return false;
    };
    let Some(td) = llbc.type_by_id(id) else {
        return false;
    };
    // Cheap leaf check before the full path-string comparison: bail
    // unless the type's last segment is the `OpArg` ident.
    let leaf_is_oparg = matches!(
        td.item_meta.name.last(),
        Some(NameSeg::Ident { ident: (s, _) }) if s == "OpArg"
    );
    leaf_is_oparg && td.item_meta.name_path().ends_with("oparg::OpArg")
}

/// The ADT def_id of an (already wrapper-stripped) type node, or
/// `None` for non-ADT nodes.
fn adt_node_def_id(node: &serde_json::Value) -> Option<u64> {
    node.as_object()?
        .get("Adt")?
        .as_object()?
        .get("id")?
        .as_object()?
        .get("Adt")?
        .as_u64()
}

/// The monomorphic-ADT class root of an (already wrapper-stripped)
/// type node, or `None` for non-ADTs and generic instantiations.
fn adt_node_class_root(node: &serde_json::Value, llbc: &Llbc) -> Option<String> {
    let adt = node.as_object()?.get("Adt")?.as_object()?;
    let def_id = adt_node_def_id(node)?;
    let has_type_args = adt
        .get("generics")
        .and_then(|g| g.as_object())
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .is_some_and(|t| !t.is_empty());
    let name = llbc.type_by_id(def_id)?.item_meta.name_path();
    if has_type_args {
        // A parameterised workspace ADT (e.g. `CodeObject<C>` used at
        // its one `ConstantData` instantiation) registers in
        // `struct_fields` under its ungeneric name like any other
        // decl, so it resolves to that flat classdef — the same
        // generics collapse `derive_program_metadata` applies.  The
        // core/std/alloc container family (`Vec<T>`, `Option<T>`,
        // `Box<T>`, …) stays excluded: those map to dedicated
        // annotator models (lists, options, wrappers), never to a
        // classdef.  Same crate-root convention as the trait-bound
        // resolver above.
        let crate_root = name.split("::").next().unwrap_or(&name);
        if matches!(crate_root, "core" | "std" | "alloc") {
            return None;
        }
    }
    let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
    // A reference-payload workspace enum instantiation projects to a
    // per-instantiation base class (`Result<Tuple>`) so its variant
    // payloads do not union across instantiations.  The discriminant
    // narrowing reads this base class name back to mint the matching
    // `Result<Tuple>::Ok` variant subclass, so the suffix here and the
    // constructor / field-read sites must agree — they share
    // `adt_head_instantiation_suffix`.  Non-enum and primitive-payload
    // heads return `None` and keep collapsing to the bare leaf.
    if let Some(suffix) = adt_head_instantiation_suffix(adt, llbc) {
        return Some(format!("{leaf}{suffix}"));
    }
    Some(leaf)
}

/// The pointee's monomorphic-ADT class root of an (already
/// wrapper-stripped) `RawPtr` type node, or `None` when the node is
/// not a raw pointer onto a plain ADT.
fn raw_ptr_pointee_class_root(node: &serde_json::Value, llbc: &Llbc) -> Option<String> {
    let raw = node.as_object()?.get("RawPtr")?.as_array()?;
    adt_node_class_root(strip_ty_wrappers(raw.first()?, llbc)?, llbc)
}

/// Whether `ty` is a `*mut PyObjectRef` / `*const PyObjectRef`
/// (`RawPtr` onto a `RawPtr` onto `PyObject`) — the pointer-to-pointer
/// signature of the object-element store an `ItemsBlock` lays out after
/// its length header, and the receiver type of brick 3's
/// `*base.add(idx)` ([`Lowering::is_list_items_elem_ptr_add`]).  A typed
/// primitive array carries a scalar pointee (`*mut u8` / `*mut i64`),
/// excluded by the inner-`RawPtr` test; the `PyObject` pointee root pins
/// it to the object family rather than any `*mut *mut T`.
fn is_pyobjectref_items_ptr(ty: &TyRef, llbc: &Llbc) -> bool {
    let Some(outer) = tyref_node(ty, llbc).and_then(|n| strip_ty_wrappers(n, llbc)) else {
        return false;
    };
    let Some(inner) = outer
        .as_object()
        .and_then(|m| m.get("RawPtr"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|n| strip_ty_wrappers(n, llbc))
    else {
        return false;
    };
    raw_ptr_pointee_class_root(inner, llbc).as_deref() == Some("PyObject")
}

/// The `__cast_pointer/<Root>` marker call — front::mir's carrier for
/// the upstream `cast_pointer(PTRTYPE, ptr)` op (lltype.py:964).  The
/// target class travels in the path (same `Vec<Variable>`-carrier
/// constraint as the `simple_call(<exc class>)` raise marker,
/// `front/exc_from_raise.rs:120-126`); the flowspace adapter rebuilds the
/// 2-arg upstream shape, and jtransform re-aliases the call to its
/// operand (`rewrite_op_cast_pointer` → `same_as`,
/// jtransform.py:254-257) so the jitcode shape stays identical to the
/// plain alias lowering.
fn cast_pointer_marker_op(root: String, arg: Variable) -> OpKind {
    OpKind::Call {
        target: CallTarget::FunctionPath {
            segments: vec!["__cast_pointer".to_string(), root.clone()],
        },
        args: vec![arg],
        result_ty: ValueType::Ref(Some(root)),
    }
}

/// Strip the indirection wrappers a Charon type node can carry —
/// `{"Deduplicated": id}` / `{"HashConsedValue": [id, ty]}` /
/// `{"Ref": [region, ty, kind]}` — and return the underlying type node
/// (`Adt`, `TypeVar`, `Literal`, …).
fn strip_ty_wrappers<'l>(
    mut node: &'l serde_json::Value,
    llbc: &'l Llbc,
) -> Option<&'l serde_json::Value> {
    for _ in 0..24 {
        let obj = node.as_object()?;
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            node = llbc.dedup_body(id)?;
            continue;
        }
        if let Some(arr) = obj
            .get("HashConsedValue")
            .and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            node = &arr[1];
            continue;
        }
        if let Some(arr) = obj.get("Ref").and_then(serde_json::Value::as_array) {
            node = arr.get(1)?;
            continue;
        }
        return Some(node);
    }
    None
}

/// De Bruijn *index* of a `{"TypeVar": {"Bound": [depth, index]}}` node.
/// The binder depth differs between a parameter-type position and a
/// trait-clause subject position (the clause subject sits one binder
/// deeper), so only the index participates in matching the two.
fn typevar_bound_index(node: &serde_json::Value) -> Option<u64> {
    node.get("TypeVar")?
        .get("Bound")?
        .as_array()?
        .get(1)?
        .as_u64()
}

/// True when fn-level type variable `var_index` carries an
/// `Into<String>` trait clause.  The clause's trait generics spell
/// `[<subject>, <target>]`; the subject must be our variable and the
/// target must resolve to the `alloc::string::String` ADT, with the
/// trait decl's leaf name `Into` (a `From<String>` bound has the same
/// generics shape but means the *opposite* conversion).
fn typevar_bounded_by_into_string(
    var_index: u64,
    fn_generics: &serde_json::Value,
    llbc: &Llbc,
) -> bool {
    let Some(clauses) = fn_generics
        .as_object()
        .and_then(|g| g.get("trait_clauses"))
        .and_then(|c| c.as_array())
    else {
        return false;
    };
    fn strip<'a>(llbc: &'a Llbc, mut v: &'a serde_json::Value) -> Option<&'a serde_json::Value> {
        loop {
            let obj = v.as_object()?;
            if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
                v = llbc.dedup_body(id)?;
                continue;
            }
            if let Some(arr) = obj
                .get("HashConsedValue")
                .and_then(serde_json::Value::as_array)
                && arr.len() == 2
            {
                v = &arr[1];
                continue;
            }
            return Some(v);
        }
    }
    for clause in clauses {
        let Some(sb) = clause
            .as_object()
            .and_then(|c| c.get("trait_"))
            .and_then(|t| t.as_object())
            .and_then(|t| t.get("skip_binder"))
            .and_then(|s| s.as_object())
        else {
            continue;
        };
        let is_into = sb
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .and_then(|id| llbc.trait_by_id(id))
            .map(|td| td.item_meta.name_path())
            .is_some_and(|n| n.rsplit("::").next() == Some("Into"));
        if !is_into {
            continue;
        }
        let Some(types) = sb
            .get("generics")
            .and_then(|g| g.get("types"))
            .and_then(|t| t.as_array())
        else {
            continue;
        };
        let subject_is_var = types
            .first()
            .and_then(|s| strip(llbc, s))
            .and_then(typevar_bound_index)
            == Some(var_index);
        if !subject_is_var {
            continue;
        }
        let target_is_string = types
            .get(1)
            .and_then(|t| resolve_tyexpr_to_adt_def_id_free(llbc, t))
            .and_then(|id| llbc.type_by_id(id))
            .is_some_and(|td| td.item_meta.name_path() == "alloc::string::String");
        if target_is_string {
            return true;
        }
    }
    false
}

/// Resolve a generic parameter type (`&T` where `T: Trait`, including a
/// trait default body's `&Self`) to the bound trait's qualified
/// `name_path()`.
///
/// [`tyref_class_root`] answers `None` for such a parameter — a
/// `TypeVar` has no ADT decl — so `OpKind::Input.class_root` stayed
/// empty and the subject graph annotated the receiver as the
/// classdef-less `SomeInstance(None)` shell, which fails on the first
/// `getattr`.  The bound trait names the receiver's only possible shape
/// when the analyzed world has exactly one concrete impl;
/// `derive_subject_inputcells` resolves the returned trait path through
/// `Bookkeeper::pyre_trait_unique_impls` and only seeds a classdef on a
/// unique hit, so carrying a multi-impl (or foreign) trait path here is
/// inert.  The qualified path (not the leaf) is the map key so two
/// distinct traits sharing a final segment cannot seed each other's
/// impl type.
///
/// Bounds declared in `core`/`std`/`alloc` (`MetaSized`, `Sized`, …)
/// are skipped: marker/stdlib traits never name a project struct, and
/// they precede the user bound in `trait_clauses` order.
fn tyref_generic_trait_bound_root(
    ty: &TyRef,
    llbc: &Llbc,
    generics: Option<&serde_json::Value>,
) -> Option<String> {
    let generics = generics?;
    let node = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => llbc.dedup_body(*id)?,
    };
    let param_index = typevar_bound_index(strip_ty_wrappers(node, llbc)?)?;
    // `T: Into<String>` — the conventional message-parameter bound
    // (`PyError::type_error(msg: impl Into<String>)`).  Such a variable
    // is a string at the annotation level (the model maps `String` and
    // `str` to one string type), so it resolves to the `"String"` root
    // and the input-cell derivation seeds `s_unicode0` instead of a
    // classdef-less instance shell whose field writes would poison
    // classdef attr cells.
    if typevar_bounded_by_into_string(param_index, generics, llbc) {
        return Some("String".to_string());
    }
    for clause in generics.get("trait_clauses")?.as_array()? {
        let Some(pred) = clause.get("trait_").and_then(|t| t.get("skip_binder")) else {
            continue;
        };
        let subject_index = pred
            .get("generics")
            .and_then(|g| g.get("types"))
            .and_then(serde_json::Value::as_array)
            .and_then(|t| t.first())
            .and_then(|s| strip_ty_wrappers(s, llbc))
            .and_then(typevar_bound_index);
        if subject_index != Some(param_index) {
            continue;
        }
        let Some(trait_id) = pred.get("id").and_then(serde_json::Value::as_u64) else {
            continue;
        };
        let Some(td) = llbc.trait_by_id(trait_id) else {
            continue;
        };
        let name = td.item_meta.name_path();
        let crate_root = name.split("::").next().unwrap_or(&name);
        if matches!(crate_root, "core" | "std" | "alloc") {
            continue;
        }
        return Some(name);
    }
    None
}

/// True when `ty` is a non-unit tuple `(A, B, ...)` — Charon's
/// synthetic `Tuple` Adt with a non-empty type-argument list.  A local
/// of this type that is not a scalar `*Checked` `BinOp` result is a
/// genuine Ref tuple whose `.N` reads extract element N via a typed
/// `FieldRead`.  The inverse-emptiness check of [`is_unit_type`]: a
/// `()` (empty `types`) is the void unit, never a field-projectable
/// tuple.
fn tyref_is_tuple(ty: &TyRef, llbc: &Llbc) -> bool {
    let value = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return false,
        },
    };
    let Some(adt) = value
        .as_object()
        .and_then(|m| m.get("Adt"))
        .and_then(|a| a.as_object())
    else {
        return false;
    };
    let is_tuple = adt.get("id").and_then(|i| i.as_str()) == Some("Tuple");
    let non_empty = adt
        .get("generics")
        .and_then(|g| g.as_object())
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .is_some_and(|t| !t.is_empty());
    is_tuple && non_empty
}

/// True when `ty` is Charon's unit type `()`.
///
/// Unit serializes as an `Adt` carrying the synthetic `"Tuple"`
/// type-id with zero type arguments:
/// `{"Adt": {"id": "Tuple", "generics": {"types": [], …}}}`.  A
/// non-empty `types` array is a real tuple (`(A, B)`) — a genuine
/// aggregate that is NOT void — so the emptiness check matters.
///
/// Used by the `Return` terminator to route `-> ()` bodies through
/// the void return path ([`FunctionGraph::set_return`] with `None`),
/// which drops a `Const(None, VOID)` return link.  Without it the
/// implicit `_0 = ()` unit aggregate lowers to a Ref-typed
/// transparent ctor and colors the result kind 'r', contradicting the
/// declared void kind and tripping the codewriter cross-check
/// (`codewriter.rs:585`).
fn is_unit_type(ty: &TyRef, llbc: &Llbc) -> bool {
    let value = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return false,
        },
    };
    let Some(adt) = value
        .as_object()
        .and_then(|m| m.get("Adt"))
        .and_then(|a| a.as_object())
    else {
        return false;
    };
    let is_tuple = adt.get("id").and_then(|i| i.as_str()) == Some("Tuple");
    let empty_types = adt
        .get("generics")
        .and_then(|g| g.as_object())
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .is_some_and(|t| t.is_empty());
    is_tuple && empty_types
}

/// True when `ty`'s top-level constructor — after the dedup /
/// hash-cons indirections [`charon_type_value_to_ast_string`] itself
/// follows — is a reference (`&T` / `&mut T`).
///
/// `tyref_to_ast_string` strips references to their referent, so a
/// `-> &bool` return would otherwise classify as a plain `bool` stub.
/// `simple_return_type_to_lltype` rejects `syn::Type::Reference` (only
/// a bare `bool` / unit projects), so the unsafe-stub collector skips
/// reference returns to keep the stub set parity-exact.
fn output_type_is_ref(ty: &TyRef, llbc: &Llbc) -> bool {
    let mut node = match ty {
        TyRef::Inline { value: (_, v) } => v,
        TyRef::Other(v) => v,
        TyRef::Dedup { id } => match llbc.dedup_body(*id) {
            Some(v) => v,
            None => return false,
        },
    };
    for _ in 0..24 {
        let Some(obj) = node.as_object() else {
            return false;
        };
        if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            match llbc.dedup_body(id) {
                Some(body) => {
                    node = body;
                    continue;
                }
                None => return false,
            }
        }
        if let Some(arr) = obj
            .get("HashConsedValue")
            .and_then(serde_json::Value::as_array)
            && arr.len() == 2
        {
            node = &arr[1];
            continue;
        }
        return obj.contains_key("Ref");
    }
    false
}

/// Resolve a Charon [`TyRef`] to the Rust type STRING the
/// `struct_fields` registry consumers expect, so
/// `derive_program_metadata` can fill `struct_fields` with real type
/// strings instead of `TyRef::label()` placeholders.
///
/// Format contract:
///   - references are STRIPPED (`&T` / `&mut T` -> `T`);
///   - raw pointers keep `*mut ` / `*const ` prefixes;
///   - integer / float / bool / char primitives use their Rust spelling;
///   - `Vec<T>` / `Option<T>` / `HashMap<K,V>` etc. are angle-bracketed
///     with comma-joined args (no spaces);
///   - slices `[T]`, arrays `[T;N]`, tuples `(A,B)` / `()`;
///   - named structs/enums use their leaf name (the registry publishes
///     both the qualified path and the bare leaf, and every consumer
///     keys on a leaf-ish form after stripping wrappers).
///
/// Shapes the resolver does not yet recognise produce a `??<key>:<json>`
/// marker so the differential gate (`PYRE_STRUCT_DIFF`) surfaces them
/// for a follow-up widening rather than silently mislabelling a field.
fn tyref_to_ast_string(ty: &TyRef, llbc: &Llbc) -> String {
    let body = match ty {
        TyRef::Inline { value: (_, v) } => Some(v),
        TyRef::Other(v) => Some(v),
        TyRef::Dedup { id } => llbc.dedup_body(*id),
    };
    match body {
        Some(v) => charon_type_value_to_ast_string(v, llbc, 0),
        None => match ty {
            TyRef::Dedup { id } => format!("??unresolved_dedup#{id}"),
            _ => "??no_body".to_string(),
        },
    }
}

/// Recursive worker for [`tyref_to_ast_string`] operating on a raw
/// Charon type-expression `Value` (a TyRef body or a nested
/// generic-argument type).  `depth` guards against pathological cycles.
pub(crate) fn charon_type_value_to_ast_string(
    v: &serde_json::Value,
    llbc: &Llbc,
    depth: usize,
) -> String {
    if depth > 24 {
        return "??deep".to_string();
    }
    let Some(obj) = v.as_object() else {
        return "??scalar".to_string();
    };
    // Indirections — follow the dedup table / inline hash-cons one hop.
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return match llbc.dedup_body(id) {
            Some(body) => charon_type_value_to_ast_string(body, llbc, depth + 1),
            None => format!("??unresolved_dedup#{id}"),
        };
    }
    if let Some(arr) = obj
        .get("HashConsedValue")
        .and_then(serde_json::Value::as_array)
        && arr.len() == 2
    {
        return charon_type_value_to_ast_string(&arr[1], llbc, depth + 1);
    }
    // Primitive literals.
    if let Some(lit) = obj.get("Literal") {
        return charon_literal_to_ast_string(lit);
    }
    // References are stripped to their referent (`&T` / `&mut T` -> `T`).
    if let Some(r) = obj.get("Ref") {
        if let Some(arr) = r.as_array() {
            // `{"Ref": [region, ty, kind]}`.
            if let Some(inner) = arr.get(1) {
                return charon_type_value_to_ast_string(inner, llbc, depth + 1);
            }
        }
        return "??ref_shape".to_string();
    }
    // Raw pointers keep the mutability prefix.
    if let Some(rp) = obj.get("RawPtr") {
        if let Some(arr) = rp.as_array()
            && arr.len() == 2
        {
            let inner = charon_type_value_to_ast_string(&arr[0], llbc, depth + 1);
            let mutbl = arr[1].as_str().unwrap_or("");
            let prefix = if mutbl.eq_ignore_ascii_case("Mut") {
                "*mut "
            } else {
                "*const "
            };
            return format!("{prefix}{inner}");
        }
        return "??rawptr_shape".to_string();
    }
    // ADTs: tuples, builtins (Box/Slice/Str/Array), and named types.
    if let Some(adt) = obj.get("Adt").and_then(|a| a.as_object()) {
        return charon_adt_to_ast_string(adt, llbc, depth);
    }
    // Top-level array `{"Array": [elem, len]}` -> `[elem;len]`.
    if let Some(arr) = obj.get("Array").and_then(serde_json::Value::as_array)
        && arr.len() == 2
    {
        let elem = charon_type_value_to_ast_string(&arr[0], llbc, depth + 1);
        let len = charon_const_generic_to_string(&arr[1]);
        return format!("[{elem};{len}]");
    }
    // Top-level slice `{"Slice": elem}` -> `[elem]`.
    if let Some(elem) = obj.get("Slice") {
        return format!(
            "[{}]",
            charon_type_value_to_ast_string(elem, llbc, depth + 1)
        );
    }
    // Trait associated-type projections (`C::Name`).  The decl-level
    // type cannot spell a concrete type, but the program-level
    // resolution is recoverable when the LLBC carries exactly ONE impl
    // of the trait: the impl's `types[]` binds each associated type to
    // its concrete value (e.g. `impl Constant for ConstantData { type
    // Name = String }` → `CodeObject<C>`'s `varnames: Box<[C::Name]>`
    // row renders `Box<[String]>`).  Ambiguous (multi-impl) or
    // unresolvable projections keep the `??TraitType` fallback below,
    // so a lookup miss stays conservative.
    if let Some(arr) = obj.get("TraitType").and_then(serde_json::Value::as_array)
        && arr.len() == 2
        && let Some(rendered) = resolve_trait_assoc_type(&arr[0], &arr[1], llbc, depth)
    {
        return rendered;
    }
    // `dyn Trait` -> `dyn <trait-root>`; recover the trait's leaf name
    // from the first trait-ref's resolved decl when present.
    if obj.contains_key("DynTrait") {
        return charon_dyn_trait_to_ast_string(&obj["DynTrait"], llbc);
    }
    // Function pointers — the JIT consumers only ever wrapper-strip and
    // struct-name-match field types, so a coarse `fn` marker is
    // sufficient (no consumer parses the `fn(..) -> ..` arrow form).
    if obj.contains_key("FnPtr") {
        return "fn".to_string();
    }
    let key = obj.keys().next().cloned().unwrap_or_else(|| "?".into());
    format!("??{key}")
}

/// Resolve a `TraitType [traitref, assoc]` projection through the
/// trait's unique impl, rendering the bound concrete type.
///
/// `traitref` names the trait via `trait_decl_ref.skip_binder.id`
/// (possibly behind `HashConsedValue`/`Deduplicated` indirections);
/// `assoc` selects the associated item (an index in current Charon
/// output).  Returns `None` — keeping the caller's `??TraitType`
/// fallback — when the trait id cannot be recovered, when zero or
/// more than one impl of the trait exists in this LLBC (a multi-impl
/// projection is genuinely instantiation-dependent), or when the
/// unique impl carries no binding for the selected item.
fn resolve_trait_assoc_type(
    traitref: &serde_json::Value,
    assoc: &serde_json::Value,
    llbc: &Llbc,
    depth: usize,
) -> Option<String> {
    let trait_id = traitref_decl_id(traitref, llbc, 0)?;
    let mut unique: Option<&serde_json::Value> = None;
    for ti in llbc.trait_impls_raw() {
        let Some(impl_trait) = ti.get("impl_trait") else {
            continue;
        };
        if impl_trait.get("id").and_then(serde_json::Value::as_u64) != Some(trait_id) {
            continue;
        }
        if unique.is_some() {
            return None;
        }
        unique = Some(ti);
    }
    let entries = unique?.get("types")?.as_array()?;
    for entry in entries {
        let Some(kind) = entry
            .get("kind")
            .and_then(|k| k.get("TraitType"))
            .and_then(serde_json::Value::as_array)
        else {
            continue;
        };
        if kind.len() == 2 && &kind[1] == assoc {
            let value = entry.get("skip_binder")?.get("value")?;
            return Some(charon_type_value_to_ast_string(value, llbc, depth + 1));
        }
    }
    None
}

/// Recover the trait decl id a `TraitRef` names —
/// `trait_decl_ref.skip_binder.id`, behind the usual
/// `HashConsedValue` / `Deduplicated` indirections.
fn traitref_decl_id(v: &serde_json::Value, llbc: &Llbc, depth: usize) -> Option<u64> {
    if depth > 8 {
        return None;
    }
    let obj = v.as_object()?;
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return traitref_decl_id(llbc.dedup_body(id)?, llbc, depth + 1);
    }
    if let Some(arr) = obj
        .get("HashConsedValue")
        .and_then(serde_json::Value::as_array)
        && arr.len() == 2
    {
        return traitref_decl_id(&arr[1], llbc, depth + 1);
    }
    obj.get("trait_decl_ref")?
        .get("skip_binder")?
        .get("id")?
        .as_u64()
}

/// Recover the trait *impl* id a resolved `TraitRef` selected —
/// `kind.TraitImpl.id`, behind the usual `HashConsedValue` /
/// `Deduplicated` indirections.  `None` when the ref is still a
/// clause/builtin obligation rather than a selected impl.
fn traitref_impl_id(v: &serde_json::Value, llbc: &Llbc, depth: usize) -> Option<u64> {
    if depth > 8 {
        return None;
    }
    let obj = v.as_object()?;
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return traitref_impl_id(llbc.dedup_body(id)?, llbc, depth + 1);
    }
    if let Some(arr) = obj
        .get("HashConsedValue")
        .and_then(serde_json::Value::as_array)
        && arr.len() == 2
    {
        return traitref_impl_id(&arr[1], llbc, depth + 1);
    }
    obj.get("kind")?.get("TraitImpl")?.get("id")?.as_u64()
}

/// Unwrap a `TraitRef`'s `HashConsedValue` / `Deduplicated` indirections
/// to the underlying trait-ref object.
fn traitref_unwrap<'a>(
    v: &'a serde_json::Value,
    llbc: &'a Llbc,
    depth: usize,
) -> Option<&'a serde_json::Value> {
    if depth > 8 {
        return None;
    }
    let obj = v.as_object()?;
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return traitref_unwrap(llbc.dedup_body(id)?, llbc, depth + 1);
    }
    if let Some(arr) = obj
        .get("HashConsedValue")
        .and_then(serde_json::Value::as_array)
        && arr.len() == 2
    {
        return traitref_unwrap(&arr[1], llbc, depth + 1);
    }
    Some(v)
}

/// Hash-cons identity of a type expression: the `Deduplicated` id or
/// the inline `HashConsedValue: [id, …]` id.  Two type refs with the
/// same key denote the same monomorphized type.
fn ty_dedup_key(v: &serde_json::Value) -> Option<u64> {
    let obj = v.as_object()?;
    if let Some(id) = obj.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return Some(id);
    }
    obj.get("HashConsedValue")?.as_array()?.first()?.as_u64()
}

/// Outcome of devirtualizing a blanket `core::convert::<Impl>::into`
/// callsite — see `Lowering::blanket_into_devirt`.
enum IntoDevirt {
    /// The reflexive `impl<T> From<T> for T` was selected — the call
    /// is a `T -> T` identity conversion.
    Identity,
    /// A concrete `impl From<T> for U` was selected; the segments are
    /// its `from` function's path.
    Target(Vec<String>),
}

/// Render a Charon `DynTrait` body to `dyn <trait-leaf>`.  Falls back to
/// `dyn` when the predicate shape does not expose a resolvable trait id.
fn charon_dyn_trait_to_ast_string(dynt: &serde_json::Value, llbc: &Llbc) -> String {
    // Charon nests the principal trait id a few ways across revisions;
    // scan for the first `{"trait_decl_id": <id>}` (or bare `id`) and
    // resolve it to the trait's leaf name.
    fn find_trait_id(v: &serde_json::Value) -> Option<u64> {
        match v {
            serde_json::Value::Object(m) => {
                if let Some(id) = m.get("trait_decl_id").and_then(serde_json::Value::as_u64) {
                    return Some(id);
                }
                m.values().find_map(find_trait_id)
            }
            serde_json::Value::Array(a) => a.iter().find_map(find_trait_id),
            _ => None,
        }
    }
    match find_trait_id(dynt).and_then(|id| llbc.trait_by_id(id)) {
        Some(td) => {
            let name = td.item_meta.name_path();
            let leaf = name.rsplit("::").next().unwrap_or(&name);
            format!("dyn {leaf}")
        }
        None => "dyn".to_string(),
    }
}

/// Map a Charon `Literal` type body to its Rust spelling.
fn charon_literal_to_ast_string(lit: &serde_json::Value) -> String {
    if let Some(atom) = lit.as_str() {
        return match atom {
            "Bool" => "bool",
            "Char" => "char",
            other => return format!("??lit_atom_{other}"),
        }
        .to_string();
    }
    if let Some(obj) = lit.as_object() {
        if let Some(int) = obj.get("Int").and_then(serde_json::Value::as_str) {
            return charon_int_kind_to_rust(int, true);
        }
        if let Some(uint) = obj.get("UInt").and_then(serde_json::Value::as_str) {
            return charon_int_kind_to_rust(uint, false);
        }
        if let Some(int) = obj.get("Integer").and_then(serde_json::Value::as_str) {
            // Single-`Integer` form: kind string is already signed/unsigned.
            let signed = !int.starts_with('U');
            return charon_int_kind_to_rust(int, signed);
        }
        if let Some(float) = obj.get("Float").and_then(serde_json::Value::as_str) {
            return match float {
                "F16" => "f16",
                "F32" => "f32",
                "F64" => "f64",
                "F128" => "f128",
                other => return format!("??float_{other}"),
            }
            .to_string();
        }
    }
    "??lit".to_string()
}

/// Translate a Charon integer-kind tag (`"I64"`, `"Usize"`, `"U8"`, …)
/// to its Rust spelling.  `signed` disambiguates the `Isize`/`Usize`
/// spelling when the kind tag itself omits the sign.
fn charon_int_kind_to_rust(kind: &str, signed: bool) -> String {
    let lowered = kind.to_ascii_lowercase();
    // Kind tags already carry the leading `i`/`u` for most widths
    // (`I64` -> `i64`, `U8` -> `u8`, `Usize` -> `usize`).  The single
    // `Integer` form may hand back a bare width — fall back to `signed`.
    if lowered.starts_with('i') || lowered.starts_with('u') {
        return lowered;
    }
    if signed {
        format!("i{lowered}")
    } else {
        format!("u{lowered}")
    }
}

/// Render an ADT head's generic type-arguments (`generics.types[]`) to
/// their AST strings, dropping the default allocator / hasher type-args
/// Charon makes explicit (`Vec<T, Global>`, `HashMap<K, V, RandomState,
/// Global>`).  Empty when the head carries no type-arguments.  Shared by
/// [`charon_adt_to_ast_string`] (full type printer) and
/// [`adt_head_instantiation_suffix`] (variant-class projection) so a
/// rendered type and its `<…>` class suffix spell one instantiation
/// identically.
fn render_adt_type_args(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
    depth: usize,
) -> Vec<String> {
    adt.get("generics")
        .and_then(|g| g.as_object())
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .map(|t| charon_type_value_to_ast_string(t, llbc, depth + 1))
                .filter(|s| s != "Global" && s != "RandomState")
                .collect()
        })
        .unwrap_or_default()
}

/// A type-argument drives a per-instantiation variant-class split unless
/// it is in the deferred-or-degenerate set below.  A split mints a
/// distinct `<…>`-suffixed variant classdef (`Result<bool>::Ok` vs
/// `Result<i64>::Ok` vs `Result<Tuple>::Ok`) whose payload field sees only
/// that one concrete type, so the field annotates to that type's own repr
/// (`BoolRepr`, `IntegerRepr`, the GC-pointer `InstanceRepr`).  Without the
/// split every instantiation collapses onto one bare variant class whose
/// `__pos_0` unions the whole program — a cross-category `int ∪ float ∪
/// char ∪ ()` merge that generalises to a generic `InstanceRepr` and walls
/// the rtyper on the materialised store.
///
/// Reference payloads (a heap class `Tuple`/`W_Object`/`str`, a nested
/// generic enum `Option<…>`, a tuple of references) are one word-sized GC
/// pointer; the scalar integer/bool/char primitives are unboxed but each
/// carries a well-defined int-banked repr, so all of these split.
/// Excluded:
///   - `f32`/`f64` — float bank; deferred until the codewriter field-descr
///     carries the concrete bank (a suffixed-owner `setfield_gc_f`
///     otherwise falls back to a GC-word descr).
///   - `()` — the unit-`Ok` return widens to a genuine void return
///     (`widen_unit_return_to_void`) and never materialises a payload field.
///   - `""` — a degenerate/absent type-arg that would render a malformed
///     `<>` suffix.
fn type_arg_splits_per_instantiation(arg: &str) -> bool {
    const DEFERRED: &[&str] = &["f32", "f64", "()", ""];
    !DEFERRED.contains(&arg)
}

/// The `<…>` generic-argument suffix of an `AggregateKind::Adt` /
/// type-node head (`<Tuple>` for a `Result<Tuple, PyError>`
/// instantiation), or `None` when the head is not a reference-payload
/// `enum` instantiation.  Returned `Some` only when the head names an
/// `enum` (struct generics keep collapsing to their flat classdef) whose
/// every type-argument [`type_arg_splits_per_instantiation`] — the scope
/// under which the per-instantiation variant class carries a sound payload
/// repr.  Every projection site (receiver type,
/// variant constructor, variant field read, numbering pre-scan) routes
/// through this one predicate so they all agree on which instantiations
/// split and how they spell the suffix.
///
/// `core`/`std`/`alloc` is NOT excluded here: the variant classes that
/// collide are `Result::Ok` / `Option::Some`, minted by the constructor
/// path, so the split must reach `core::result::Result` /
/// `core::option::Option`.  The receiver-type projection
/// [`adt_node_class_root`] keeps its own container exclusion so
/// `Vec<T>` / `Box<T>` still map to their annotator models.
pub(crate) fn adt_head_instantiation_suffix(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
) -> Option<String> {
    let def_id = adt.get("id")?.as_object()?.get("Adt")?.as_u64()?;
    let td = llbc.type_by_id(def_id)?;
    if !matches!(td.kind, TypeDeclKind::Enum(_)) {
        return None;
    }
    let type_args = render_adt_type_args(adt, llbc, 0);
    if type_args.is_empty()
        || !type_args
            .iter()
            .all(|a| type_arg_splits_per_instantiation(a))
    {
        return None;
    }
    Some(format!("<{}>", type_args.join(",")))
}

/// Format a Charon `Adt` type body (`{"id": …, "generics": {"types": […]}}`).
fn charon_adt_to_ast_string(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
    depth: usize,
) -> String {
    let id = adt.get("id");
    let type_args: Vec<String> = render_adt_type_args(adt, llbc, depth);
    // `id` is either a string atom (`"Tuple"`), or an object
    // (`{"Adt": <def_id>}`, `{"Builtin": "Box"|"Slice"|"Str"|"Array"}`).
    if let Some(atom) = id.and_then(serde_json::Value::as_str) {
        return match atom {
            "Tuple" => {
                if type_args.is_empty() {
                    "()".to_string()
                } else {
                    format!("({})", type_args.join(","))
                }
            }
            other => format!("??adt_atom_{other}"),
        };
    }
    if let Some(id_obj) = id.and_then(|i| i.as_object()) {
        if let Some(def_id) = id_obj.get("Adt").and_then(serde_json::Value::as_u64) {
            let name = llbc
                .type_by_id(def_id)
                .map(|td| td.item_meta.name_path())
                .unwrap_or_else(|| format!("??adt#{def_id}"));
            let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
            if type_args.is_empty() {
                return leaf;
            }
            return format!("{leaf}<{}>", type_args.join(","));
        }
        if let Some(builtin) = id_obj.get("Builtin") {
            return charon_builtin_adt_to_ast_string(builtin, &type_args, adt);
        }
    }
    let key = id
        .and_then(|i| i.as_object())
        .and_then(|m| m.keys().next().cloned())
        .or_else(|| id.and_then(|i| i.as_str()).map(str::to_string))
        .unwrap_or_else(|| "?".into());
    format!("??adt_id_{key}")
}

/// Format a Charon builtin ADT id (`Box`/`Slice`/`Str`/`Array`).
fn charon_builtin_adt_to_ast_string(
    builtin: &serde_json::Value,
    type_args: &[String],
    adt: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let name = builtin
        .as_str()
        .or_else(|| {
            builtin
                .as_object()
                .and_then(|m| m.keys().next().map(String::as_str))
        })
        .unwrap_or("?");
    match name {
        // Charon's `Box` builtin maps to the `Box<T>` spelling.
        "Box" => match type_args.first() {
            Some(inner) => format!("Box<{inner}>"),
            None => "Box".to_string(),
        },
        "Slice" => match type_args.first() {
            Some(inner) => format!("[{inner}]"),
            None => "??slice_noelem".to_string(),
        },
        "Str" => "str".to_string(),
        "Array" => {
            let elem = type_args.first().cloned().unwrap_or_default();
            // Array length lives in the ADT's const-generic args; when
            // absent fall back to the `N` placeholder for non-literal
            // lengths.
            let len = adt
                .get("generics")
                .and_then(|g| g.as_object())
                .and_then(|g| g.get("const_generics"))
                .and_then(|c| c.as_array())
                .and_then(|c| c.first())
                .map(charon_const_generic_to_string)
                .unwrap_or_else(|| "N".to_string());
            format!("[{elem};{len}]")
        }
        other => format!("??builtin_{other}"),
    }
}

/// Best-effort render of a Charon const-generic (array length) value.
fn charon_const_generic_to_string(cg: &serde_json::Value) -> String {
    if let Some(s) = cg.as_str() {
        return s.to_string();
    }
    if let Some(obj) = cg.as_object() {
        if let Some(val) = obj.get("Value") {
            if let Some(scalar) = val
                .as_object()
                .and_then(|m| m.get("Scalar"))
                .and_then(|s| s.as_object())
            {
                if let Some(n) = scalar
                    .values()
                    .find_map(|v| v.as_object())
                    .and_then(|m| m.get("value"))
                    .and_then(serde_json::Value::as_u64)
                {
                    return n.to_string();
                }
            }
        }
    }
    "N".to_string()
}

/// Stable short label for an [`Rvalue::Aggregate`]'s [`Field`]
/// projection payload. Charon encodes `Field` as `[{"Adt"|"Tuple": ...}, idx]`,
/// where `idx` is the field's position. We project to
/// `<container>_<idx>` so synthetic FieldDescriptors stay readable.
fn field_label_from_payload(payload: &serde_json::Value) -> String {
    if let Some(arr) = payload.as_array() {
        if arr.len() == 2 {
            let container = arr[0]
                .as_object()
                .and_then(|m| m.keys().next().cloned())
                .unwrap_or_else(|| "Field".into());
            let idx = arr[1].as_u64().unwrap_or(u64::MAX);
            return format!("{container}_{idx}");
        }
    }
    "field".into()
}

/// `(module_leaf, method_leaf)` pairs whose primitive/raw-pointer impl
/// method has a classdef-less analyzer reachable through the `getattr` →
/// bound-method path, so [`Lowering::impl_method_owner`] may route them as
/// `CallTarget::Method` even though Charon leaves the `Self` type unresolved
/// (non-ADT, no entry in the type table).  `*_ptr::is_null` resolves to
/// `unaryop.rs::ptr_method_is_null` (yielding `SomeBool`), lowered to
/// `ptr_iszero`; `const_ptr` and `mut_ptr` share the analyzer since the
/// receiver mutability does not affect the null test.  Pairs absent here
/// keep the `FunctionPath` form rather than surface a new panicking
/// `SomeInstance.getattr`.
const NON_ADT_OWNER_METHOD_ALLOWLIST: &[(&str, &str)] =
    &[("mut_ptr", "is_null"), ("const_ptr", "is_null")];

/// Return `(trait_leaf_ident, method_leaf_ident)` when the FunDecl's
/// raw `NameSeg` vec ends in two consecutive `Ident` segments — the
/// Charon shape for a trait method declaration (e.g.
/// `pyre_interpreter::shared_opcode::SharedOpcodeHandler::push_value`).
/// The penultimate Ident is the trait name, the leaf the method
/// name.
///
/// Distinct from [`Lowering::impl_method_owner`], which looks for an
/// `Impl` `NameSeg::Other` segment preceding the leaf — that arm
/// fires for inherent / trait-impl methods Charon already resolved
/// at extraction time.  Trait method declarations have no `Impl`
/// segment because the body is the trait's default impl.
///
/// Used by the `CallKind::Trait` arm of
/// [`Lowering::call_target_segments`] to emit
/// `CallTarget::FunctionPath { segments: [trait_leaf, method_leaf]
/// }`, matching the direct-path key
/// `register_function_graph(direct_path, …)` at `lib.rs:957-969`
/// (`extract_trait_impls`'s `<default methods of <Trait>>` branch).
fn trait_method_owner(fd: &FunDecl) -> Option<(String, String)> {
    let segs = &fd.item_meta.name;
    if segs.len() < 2 {
        return None;
    }
    let leaf = match segs.last()? {
        NameSeg::Ident { ident: (s, _) } => s.clone(),
        _ => return None,
    };
    let parent = match &segs[segs.len() - 2] {
        NameSeg::Ident { ident: (s, _) } => s.clone(),
        _ => return None,
    };
    Some((parent, leaf))
}

/// Compact identifier for a `CallKind::Trait` payload — the triple
/// `[trait_ref, method_idx, decl_id]`. We project to
/// `trait<decl_id>::m<method_idx>` so the synthesised path is small
/// and deterministic.  Falls back to `unknown` if the shape is
/// unexpected; callers should fail-loud on `unknown` if downstream
/// dispatch needs the actual impl.
fn trait_call_label(v: &serde_json::Value) -> String {
    if let Some(arr) = v.as_array() {
        let method_idx = arr.get(1).and_then(Value::as_u64).unwrap_or(u64::MAX);
        let decl_id = arr.get(2).and_then(Value::as_u64).unwrap_or(u64::MAX);
        return format!("trait{decl_id}::m{method_idx}");
    }
    "unknown".to_string()
}

/// Strip the leading crate-name segment from a Charon `name_path()`.
/// Charon prefixes every fully-qualified path with the crate name
/// (`pyre_interpreter::frame::eval_loop_jit`); functions are named
/// relative to their module root instead (`frame::eval_loop_jit` for a
/// non-empty `module_path`, or the bare leaf for `module_path == ""`)
/// so `register_function_graph_alias` (lib.rs:444) can walk
/// `{bare, crate::*, pyre_interpreter::*, pyre_object::*, pyre_jit::*}`
/// aliases off the same `func.name`.
fn strip_crate_prefix(path: &str) -> String {
    match path.split_once("::") {
        Some((_crate, rest)) => rest.to_string(),
        // single-segment name (rare — top-level item without crate
        // prefix in some Charon outputs): leave as-is.
        None => path.to_string(),
    }
}

/// True for the `ItemsBlock` items-base accessor bodies whose
/// `.add(ITEMS_BLOCK_ITEMS_OFFSET)` the front-end collapses to the
/// receiver (see [`Lowering::is_items_block_base_ptr_add`]).  Matched on
/// the module-qualified path so a leaf collision in another module
/// cannot widen the gate, and crate-prefix-independent so the same
/// accessor matches whether reached from `pyre_object`, `pyre_interpreter`,
/// or `pyre_jit`'s monomorphized copy.
fn graph_is_items_block_base_accessor(name: &str) -> bool {
    name.ends_with("object_array::items_block_items_base")
        || name.ends_with("object_array::items_block_items_ptr")
}

fn static_key_matches(full: &str, stripped: &str, key: &str) -> bool {
    full == key
        || stripped == key
        || full
            .strip_suffix(key)
            .is_some_and(|prefix| prefix.ends_with("::"))
        || stripped
            .strip_suffix(key)
            .is_some_and(|prefix| prefix.ends_with("::"))
}

/// Supply the value of a primitive `f64` associated constant whose
/// initializer Charon records as an `Opaque` body — `core` defines
/// `f64::INFINITY` as `1.0_f64 / 0.0_f64`, so no in-LLBC init survives
/// for [`Lowering::const_eval_global`] to evaluate.  The value is a
/// fixed IEEE-754 bit pattern the host (rustc) already computed, so
/// emit it as the same by-value `ConstFloat` an inline float literal
/// lowers to (mirroring `rfloat.INFINITY` reaching the flow graph as a
/// float `Constant`).  Matches on the `f64::<Impl>::<NAME>` tail so a
/// `core`- or `std`-rooted path resolves identically.
fn primitive_float_const(segments: &[String]) -> Option<OpKind> {
    let tail: Vec<&str> = segments
        .iter()
        .rev()
        .take(3)
        .rev()
        .map(String::as_str)
        .collect();
    let bits = match tail.as_slice() {
        ["f64", "<Impl>", "INFINITY"] => f64::INFINITY.to_bits(),
        _ => return None,
    };
    Some(OpKind::ConstFloat(bits))
}

fn place_kind_label(k: &PlaceKind) -> &'static str {
    match k {
        PlaceKind::Local(_) => "Local",
        PlaceKind::Projection(_, _) => "Projection",
        PlaceKind::Global { .. } => "Global",
        PlaceKind::Unknown => "Unknown",
    }
}

fn rvalue_variant_name(rv: &Rvalue) -> &'static str {
    match rv {
        Rvalue::Use(_) => "Use",
        Rvalue::BinaryOp(..) => "BinaryOp",
        Rvalue::UnaryOp(..) => "UnaryOp",
        Rvalue::Ref { .. } => "Ref",
        Rvalue::Aggregate(..) => "Aggregate",
        Rvalue::Discriminant(_) => "Discriminant",
        Rvalue::Cast(..) => "Cast",
        Rvalue::Len(_) => "Len",
        Rvalue::Repeat(..) => "Repeat",
        Rvalue::RawPtr { .. } => "RawPtr",
        Rvalue::NullaryOp(..) => "NullaryOp",
        Rvalue::ShallowInitBox(..) => "ShallowInitBox",
        Rvalue::Unknown => "Unknown",
    }
}

/// Subset of MIR constant kinds the driver currently knows how to
/// emit. Widen as the corpus grows past `straight_line_add`.
enum DecodedConst {
    Int(i64),
    Bool(bool),
    Float(u64),
    /// String / char / byte-string literals. The IR has no dedicated
    /// string constant opkind; the codewriter treats these as opaque
    /// pointer-typed values. We carry the textual representation as a
    /// unique-string `ConstValue` so the generated IR is stable across
    /// runs.
    Str(String),
    /// Constant function pointer (`FnDef`). Encoded as a synthetic
    /// `FunctionPath` so it shares the existing `Call` lowering path
    /// when threaded into an indirect call site.
    FnPath(Vec<String>),
}

/// Decode `Operand::Const`'s value field. Possible shapes:
///   - `{kind: {Literal: {Scalar: {Signed|Unsigned|Isize|Usize: [ty, "v"]}}}}`
///   - `{kind: {Literal: {Bool: bool}}}`
///   - `{kind: {Literal: {Float: {value: "v", ty: "F32|F64"}}}}`
///   - `{kind: {Literal: {Str: "..."}}}`
///   - `{kind: {Literal: {Char: "c"}}}`
///   - `{kind: {Literal: {ByteStr: "..."}}}`
///   - `{kind: {FnDef: {kind: {Fun: {Regular: id}}, generics: ...}}}`
fn decode_constant(llbc: &Llbc, value: &serde_json::Value) -> Result<DecodedConst, LowerError> {
    let kind = value
        .as_object()
        .and_then(|m| m.get("kind"))
        .and_then(|k| k.as_object())
        .ok_or_else(|| {
            LowerError::Unsupported(format!("Operand::Const value missing object kind: {value}"))
        })?;
    if let Some(lit) = kind.get("Literal") {
        return decode_literal(lit);
    }
    // `Opaque "<reason>"` — Charon itself bailed on the constant.
    // Forward the reason so it ends up in the synthetic path; the
    // codewriter sees a 0-arg Call it can ignore for analysis.
    if let Some(reason) = kind.get("Opaque").and_then(Value::as_str) {
        return Ok(DecodedConst::Str(format!("opaque:{reason}")));
    }
    // `VTableRef { ... }` — vtable pointer for dynamic dispatch.
    // Treat as an opaque pointer-typed value; covering it faithfully
    // requires the trait dispatch widening.
    if kind.contains_key("VTableRef") {
        return Ok(DecodedConst::Str("__vtable_ref".to_string()));
    }
    // `TraitConst` — trait-associated const. Opaque for now; covering
    // it faithfully requires trait/impl resolution.
    if kind.contains_key("TraitConst") {
        return Ok(DecodedConst::Str("__trait_const".to_string()));
    }
    if let Some(fn_def) = kind.get("FnDef") {
        // `FnDef.kind = Fun(Regular id)` carries the function the
        // constant references; resolve it to a path via the same
        // FunId lookup the Call terminator uses.
        let inner = fn_def
            .as_object()
            .and_then(|m| m.get("kind"))
            .and_then(|m| m.get("Fun"))
            .and_then(|m| m.get("Regular"))
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                LowerError::Unsupported(format!("FnDef shape not yet handled: {fn_def}"))
            })?;
        let fd = llbc.fn_by_id(inner).ok_or_else(|| {
            LowerError::Schema(format!(
                "FnDef constant references unknown FunDecl id {inner}"
            ))
        })?;
        return Ok(DecodedConst::FnPath(
            fd.item_meta
                .name_path()
                .split("::")
                .map(|s| s.to_string())
                .collect(),
        ));
    }
    Err(LowerError::Unsupported(format!(
        "Operand::Const kind not yet handled: {value}"
    )))
}

fn decode_literal(lit: &serde_json::Value) -> Result<DecodedConst, LowerError> {
    let lit_obj = lit
        .as_object()
        .ok_or_else(|| LowerError::Schema(format!("Literal not object: {lit}")))?;
    if let Some(scalar_obj) = lit_obj.get("Scalar").and_then(Value::as_object) {
        for (k, payload) in scalar_obj {
            let arr = payload.as_array().ok_or_else(|| {
                LowerError::Schema(format!("Scalar {k}: payload not array: {payload}"))
            })?;
            if arr.len() != 2 {
                continue;
            }
            let v_str = arr[1].as_str().ok_or_else(|| {
                LowerError::Schema(format!("Scalar {k}: value not a string: {payload}"))
            })?;
            return Ok(match k.as_str() {
                "Signed" | "Isize" => DecodedConst::Int(
                    v_str
                        .parse()
                        .map_err(|e| LowerError::Schema(format!("Scalar Signed parse: {e}")))?,
                ),
                "Unsigned" | "Usize" => DecodedConst::Int(
                    v_str
                        .parse::<u64>()
                        .map_err(|e| LowerError::Schema(format!("Scalar Unsigned parse: {e}")))?
                        as i64,
                ),
                _ => {
                    return Err(LowerError::Unsupported(format!(
                        "Scalar kind {k} not yet decoded"
                    )));
                }
            });
        }
    }
    if let Some(b) = lit_obj.get("Bool").and_then(Value::as_bool) {
        return Ok(DecodedConst::Bool(b));
    }
    if let Some(f) = lit_obj.get("Float") {
        if let Some(s) = f
            .as_object()
            .and_then(|m| m.get("value"))
            .and_then(Value::as_str)
        {
            if let Ok(v) = s.parse::<f64>() {
                return Ok(DecodedConst::Float(v.to_bits()));
            }
        }
        return Err(LowerError::Schema(format!("Float shape: {f}")));
    }
    if let Some(s) = lit_obj.get("Str").and_then(Value::as_str) {
        return Ok(DecodedConst::Str(s.to_string()));
    }
    if let Some(s) = lit_obj.get("Char").and_then(Value::as_str) {
        return Ok(DecodedConst::Str(s.to_string()));
    }
    if let Some(s) = lit_obj.get("ByteStr").and_then(Value::as_str) {
        return Ok(DecodedConst::Str(s.to_string()));
    }
    Err(LowerError::Unsupported(format!(
        "Literal shape not yet decoded: {lit}"
    )))
}

use serde_json::Value;

fn scalar_to_const_value(v: &serde_json::Value) -> Option<ConstValue> {
    let obj = v.as_object()?;
    // `{Scalar: {Signed|Unsigned|Isize|Usize: [ty, value]}}`
    if let Some(scalar) = obj.get("Scalar").and_then(Value::as_object) {
        for (_k, payload) in scalar {
            let arr = payload.as_array()?;
            if arr.len() != 2 {
                continue;
            }
            let n: i64 = scalar_value_to_i64(&arr[1])?;
            return Some(ConstValue::Int(n));
        }
    }
    // `{Char: "c"}` — character matched as a SwitchInt arm.
    if let Some(c) = obj.get("Char").and_then(Value::as_str) {
        return Some(ConstValue::Int(c.chars().next()? as i64));
    }
    // `{Bool: true}` — boolean matched as a SwitchInt arm.
    if let Some(b) = obj.get("Bool").and_then(Value::as_bool) {
        return Some(ConstValue::Int(b as i64));
    }
    None
}

/// Extract a scalar value as `i64`. Accepts both string ("0") and
/// JSON-numeric (0) representations: Charon emits `["Char", "97"]`
/// for `'a'` but `["Bool", true]` for boolean discriminants.
fn scalar_value_to_i64(v: &serde_json::Value) -> Option<i64> {
    if let Some(s) = v.as_str() {
        return s.parse().ok();
    }
    if let Some(b) = v.as_bool() {
        return Some(b as i64);
    }
    if let Some(n) = v.as_i64() {
        return Some(n);
    }
    if let Some(n) = v.as_u64() {
        return Some(n as i64);
    }
    None
}

/// Resolve a `Variable` used as a call argument back to the operation
/// that produced it, following block-input `Link`s across blocks.
///
/// The `fmt`-chain recognizer (#277) fires at `alloc::fmt::format(v)`
/// where `v` is the `Arguments` value — but `v` reaches that block as an
/// `inputarg` threaded from the predecessor's `Arguments::new` result via
/// the outgoing `Link`, not as a direct op result in the current block.
/// Existing `try_lower_*` folds only read a call's direct args; the fmt
/// recognizer needs this cross-block back-trace.
///
/// Returns `(block, op_index)` of the producing `SpaceOperation`, or
/// `None` when `var` traces to a `Const`, a function input (no producing
/// op), a phi merge (a block with more than one incoming `Link`, so no
/// single producer), or a producer not yet emitted into `graph`.
fn resolve_to_producer_op(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> Option<(BlockId, usize)> {
    let mut current = var.clone();
    let mut visited: Vec<u64> = Vec::new();
    loop {
        if visited.contains(&current.id()) {
            return None; // cycle guard
        }
        visited.push(current.id());

        // (1) Produced directly by an op in some block?
        for block in &graph.blocks {
            for (idx, op) in block.operations.iter().enumerate() {
                if op.result.as_ref().is_some_and(|r| r.id() == current.id()) {
                    return Some((block.id, idx));
                }
            }
        }

        // (2) Otherwise it is a block inputarg — follow the single
        // incoming Link's matching positional arg back one block.
        let (owner, pos) = graph.blocks.iter().find_map(|block| {
            block
                .inputargs
                .iter()
                .position(|a| a.id() == current.id())
                .map(|pos| (block.id, pos))
        })?;
        let mut incoming = graph
            .blocks
            .iter()
            .flat_map(|b| b.exits.iter())
            .filter(|link| link.target == owner);
        let first = incoming.next()?;
        if incoming.next().is_some() {
            return None; // phi merge: no single producer
        }
        match first.args.get(pos)? {
            crate::model::LinkArg::Value(v) => current = v.clone(),
            crate::model::LinkArg::Const(_) => return None,
        }
    }
}

/// Whether `dom` dominates `target`: every path from the graph entry to
/// `target` passes through `dom`.  Computed as "with `dom` removed, is
/// `target` unreachable from `startblock`" (the standard reachability
/// formulation; a block dominates itself).  Used to prove a value defined
/// in `dom` is available at every op in `target`.
fn block_dominates(graph: &FunctionGraph, dom: BlockId, target: BlockId) -> bool {
    if dom == target {
        return true;
    }
    let mut seen: Vec<BlockId> = vec![graph.startblock];
    let mut stack = vec![graph.startblock];
    while let Some(b) = stack.pop() {
        if b == dom {
            continue; // cut: do not traverse through the candidate dominator
        }
        if b == target {
            return false; // reached target without passing through dom
        }
        if let Some(block) = graph.blocks.iter().find(|x| x.id == b) {
            for l in &block.exits {
                if !seen.contains(&l.target) {
                    seen.push(l.target);
                    stack.push(l.target);
                }
            }
        }
    }
    // `target` unreachable once `dom` is cut ⇒ `dom` dominates `target`.
    // Guard against an unreachable `target` (vacuously "dominated"): only
    // treat as dominated when `target` is genuinely reachable in the full
    // graph.
    block_reachable(graph, target)
}

/// Whether `target` is reachable from `startblock` following exits.
fn block_reachable(graph: &FunctionGraph, target: BlockId) -> bool {
    let mut seen: Vec<BlockId> = vec![graph.startblock];
    let mut stack = vec![graph.startblock];
    while let Some(b) = stack.pop() {
        if b == target {
            return true;
        }
        if let Some(block) = graph.blocks.iter().find(|x| x.id == b) {
            for l in &block.exits {
                if !seen.contains(&l.target) {
                    seen.push(l.target);
                    stack.push(l.target);
                }
            }
        }
    }
    false
}

/// Decode the packed format-string template that `format_args!` lowers
/// its `&[&str]` pieces into.  The pieces argument to `fmt::Arguments::new`
/// is a `[u8; N]` byte buffer (charon lowers it as an `Array` of `U8`
/// constant elements); this reconstructs the literal `pieces` and counts
/// the argument placeholders.
///
/// Grammar (verified against the real LLBC for several handler graphs):
/// - literal segment: a length byte `L` with `L < 0x80`, then `L` bytes
///   of UTF-8 text appended to the current piece;
/// - placeholder: the single byte `0xC0` — closes the current piece and
///   begins the next (the Display-vs-Debug choice lives in the parallel
///   args array, not in this template, so a placeholder carries no kind);
/// - terminator: a `0x00` byte (only at a segment boundary; `0`/`0xC0`
///   bytes inside a literal are consumed by its length prefix).
///
/// Returns `(pieces, placeholder_count)` with `pieces.len() ==
/// placeholder_count + 1`.  Returns `None` (bail, leaving the graph
/// untouched) on any high-bit control byte other than `0xC0` — i.e. a
/// format spec with width/precision/fill or an explicit positional/named
/// argument — and on non-UTF-8 literal bytes.
fn decode_packed_format_pieces(bytes: &[u8]) -> Option<(Vec<String>, usize)> {
    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut placeholders = 0usize;
    let mut i = 0;
    let mut terminated = false;
    while i < bytes.len() {
        let b = bytes[i];
        if b == 0x00 {
            // terminator — must be the final byte
            if i + 1 != bytes.len() {
                return None;
            }
            terminated = true;
            break;
        } else if b == 0xC0 {
            // plain sequential placeholder
            pieces.push(std::mem::take(&mut current));
            placeholders += 1;
            i += 1;
        } else if b < 0x80 {
            // literal segment of length `b`
            let start = i + 1;
            let end = start.checked_add(b as usize)?;
            let seg = bytes.get(start..end)?;
            current.push_str(std::str::from_utf8(seg).ok()?);
            i = end;
        } else {
            // any other control byte = format spec / positional arg: bail
            return None;
        }
    }
    // The grammar requires the `0x00` terminator at a segment boundary;
    // running out of bytes without it is a truncated or wrong array, not
    // a valid template — bail rather than accept it once wired.
    if !terminated {
        return None;
    }
    pieces.push(current);
    Some((pieces, placeholders))
}

/// Read an `Array` aggregate literal: given the Variable holding a
/// `SyntheticTransparentCtor { name: "Array" }` result, collect the
/// values written to its `__pos_0..__pos_{n-1}` fields in index order.
/// The ctor and its element `FieldWrite`s are emitted into one block by
/// the `Rvalue::Aggregate` array lowering, so the search is block-local
/// once the ctor is found. Returns `None` if `array_var` is not produced
/// by an `Array` ctor, or its `__pos_i` writes are not the contiguous
/// range `0..n`.
fn read_array_literal_elements(
    graph: &FunctionGraph,
    array_var: &crate::flowspace::model::Variable,
) -> Option<Vec<crate::flowspace::model::Variable>> {
    use crate::model::{CallTarget, OpKind};
    let (block_id, ctor_idx) = resolve_to_producer_op(graph, array_var)?;
    let block = graph.blocks.iter().find(|b| b.id == block_id)?;
    match &block.operations.get(ctor_idx)?.kind {
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { name, .. },
            ..
        } if name == "Array" => {}
        _ => return None,
    }
    let mut by_index: Vec<(usize, crate::flowspace::model::Variable)> = Vec::new();
    for op in &block.operations {
        if let OpKind::FieldWrite {
            base, field, value, ..
        } = &op.kind
        {
            if base.id() == array_var.id() {
                let idx = field.name.strip_prefix("__pos_")?.parse::<usize>().ok()?;
                // A `setfield_gc` inline `Const` element carries no SSA
                // Variable; the recognizer needs a register operand per slot.
                let write_var = value.as_variable()?;
                by_index.push((idx, write_var.clone()));
            }
        }
    }
    by_index.sort_by_key(|(i, _)| *i);
    for (expected, (idx, _)) in by_index.iter().enumerate() {
        if *idx != expected {
            return None;
        }
    }
    Some(by_index.into_iter().map(|(_, v)| v).collect())
}

/// Resolve a Variable to the `i64` of the `ConstInt` op that produces it,
/// following cross-block links via [`resolve_to_producer_op`]. Used to
/// read the packed-format pieces byte array (each `__pos_i` element is a
/// `ConstInt` byte).
fn resolve_const_int(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> Option<i64> {
    use crate::model::OpKind;
    let (block_id, idx) = resolve_to_producer_op(graph, var)?;
    let block = graph.blocks.iter().find(|b| b.id == block_id)?;
    match &block.operations.get(idx)?.kind {
        OpKind::ConstInt(n) => Some(*n),
        _ => None,
    }
}

/// Whether a `Display` (`{}`) or `Debug` (`{:?}`) placeholder rendered an
/// argument. Carried by the `fmt::rt::Argument::new_display` /
/// `new_debug` constructor in the parallel args array (the packed pieces
/// template does not encode the choice).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FmtArgKind {
    Display,
    Debug,
}

/// One placeholder argument recovered from a `format_args!` chain: the
/// value Variable the placeholder renders and its Display/Debug flavour.
#[derive(Debug, Clone)]
struct FmtArg {
    value: crate::flowspace::model::Variable,
    kind: FmtArgKind,
}

/// The decoded contents of a `format_args!` chain — the literal string
/// pieces interleaved with the rendered placeholder arguments
/// (`pieces.len() == args.len() + 1`).
#[derive(Debug, Clone)]
struct FmtChain {
    pieces: Vec<String>,
    args: Vec<FmtArg>,
}

/// Match a `FunctionPath`'s trailing segments against `tail`, so a
/// crate-qualified spelling (`core::fmt::Arguments::new`) and the
/// crate-stripped front-end spelling (`fmt::Arguments::new`) both
/// resolve.
fn fmt_path_ends_with(segments: &[String], tail: &[&str]) -> bool {
    segments.len() >= tail.len()
        && segments[segments.len() - tail.len()..]
            .iter()
            .zip(tail)
            .all(|(s, t)| s.as_str() == *t)
}

/// The `fmt::Arguments::new(pieces, args)` constructor that `format_args!`
/// builds from the on-stack pieces+args arrays.
fn is_arguments_new_path(segments: &[String]) -> bool {
    fmt_path_ends_with(segments, &["Arguments", "new"])
}

/// Classify a `fmt::rt::Argument::new_display` / `new_debug` constructor
/// path. Any other path (positional/named/width-bearing argument ctor) is
/// not in the recognized subset.
fn fmt_argument_ctor_kind(segments: &[String]) -> Option<FmtArgKind> {
    if fmt_path_ends_with(segments, &["Argument", "new_display"]) {
        Some(FmtArgKind::Display)
    } else if fmt_path_ends_with(segments, &["Argument", "new_debug"]) {
        Some(FmtArgKind::Debug)
    } else {
        None
    }
}

/// Unwrap the `format_args!` argument tuple-ref. Each Display/Debug
/// argument reaches `Argument::new_display(&v)` as a `FieldRead` off the
/// on-stack argument `Tuple` aggregate (`&(v,).0`); follow it back to the
/// value written into that tuple field. Returns `var` unchanged when it
/// is not produced by a `FieldRead` (value passed directly), and `None`
/// when the tuple field has conflicting writers.
fn unwrap_fmt_arg_tuple_ref(
    graph: &FunctionGraph,
    var: &crate::flowspace::model::Variable,
) -> Option<crate::flowspace::model::Variable> {
    use crate::model::OpKind;
    let (block_id, idx) = resolve_to_producer_op(graph, var)?;
    let block = graph.blocks.iter().find(|b| b.id == block_id)?;
    let (base, field_name) = match &block.operations.get(idx)?.kind {
        OpKind::FieldRead { base, field, .. } => (base.clone(), field.name.clone()),
        _ => return Some(var.clone()),
    };
    // The aggregate `FieldWrite`s live in the block that built the tuple,
    // but a `FieldRead` for the 2nd+ placeholder reads a *threaded copy*
    // of the tuple ref (forwarded across a block boundary as an inputarg),
    // so its `base` Variable id differs from the write `base`'s id even
    // though both denote the same tuple ctor.  Resolve the read base to
    // its producing op so the write base can be matched by producer
    // identity rather than raw Variable id.
    let base_producer = resolve_to_producer_op(graph, &base);
    let mut found: Option<crate::flowspace::model::Variable> = None;
    for b in &graph.blocks {
        for op in &b.operations {
            if let OpKind::FieldWrite {
                base: write_base,
                field,
                value,
                ..
            } = &op.kind
            {
                let base_matches = write_base.id() == base.id()
                    || (base_producer.is_some()
                        && resolve_to_producer_op(graph, write_base) == base_producer);
                if base_matches && field.name == field_name {
                    // A `setfield_gc` inline `Const` carries no SSA Variable;
                    // the back-trace needs a register operand to follow.
                    let write_var = value.as_variable()?;
                    if found.as_ref().is_some_and(|f| f.id() != write_var.id()) {
                        return None; // ambiguous: distinct values written
                    }
                    found = Some(write_var.clone());
                }
            }
        }
    }
    found
}

/// Recover one placeholder argument from an args-array element: the
/// element is the result of a `fmt::rt::Argument::new_display` /
/// `new_debug` constructor, whose sole argument back-traces (through the
/// tuple-ref wrap) to the rendered value.
fn extract_fmt_arg(
    graph: &FunctionGraph,
    arg_elem: &crate::flowspace::model::Variable,
) -> Option<FmtArg> {
    use crate::model::{CallTarget, OpKind};
    let (block_id, idx) = resolve_to_producer_op(graph, arg_elem)?;
    let block = graph.blocks.iter().find(|b| b.id == block_id)?;
    let (kind, inner) = match &block.operations.get(idx)?.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } => (fmt_argument_ctor_kind(segments)?, args.first()?.clone()),
        _ => return None,
    };
    let value = unwrap_fmt_arg_tuple_ref(graph, &inner)?;
    Some(FmtArg { value, kind })
}

/// Back-trace a recognized `format_args!` chain from the Variable passed
/// to `alloc::fmt::format(args)` (the String producer) to the literal
/// pieces and the Display/Debug argument values. Composes the staged
/// primitives:
///   fmt-args → [`resolve_to_producer_op`] → `fmt::Arguments::new(pieces, args)`
///     · pieces: [`read_array_literal_elements`] → [`resolve_const_int`] per
///       byte → [`decode_packed_format_pieces`]
///     · args: [`read_array_literal_elements`] → per element
///       [`extract_fmt_arg`]
/// Returns `None` (the recognizer leaves the graph untouched) on any
/// shape outside the recognized subset, so it never fires on an
/// unsupported chain.
fn extract_fmt_chain(
    graph: &FunctionGraph,
    fmt_args_var: &crate::flowspace::model::Variable,
) -> Option<FmtChain> {
    use crate::model::{CallTarget, OpKind};
    // The format(args) argument is the result of `Arguments::new(pieces, args)`.
    let (block_id, idx) = resolve_to_producer_op(graph, fmt_args_var)?;
    let block = graph.blocks.iter().find(|b| b.id == block_id)?;
    let (pieces_var, args_var) = match &block.operations.get(idx)?.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } if is_arguments_new_path(segments) => (args.first()?.clone(), args.get(1)?.clone()),
        _ => return None,
    };
    // Pieces: an `Array` of `ConstInt` bytes → decode the packed template.
    let piece_byte_vars = read_array_literal_elements(graph, &pieces_var)?;
    let mut bytes = Vec::with_capacity(piece_byte_vars.len());
    for v in &piece_byte_vars {
        bytes.push(u8::try_from(resolve_const_int(graph, v)?).ok()?);
    }
    let (pieces, placeholder_count) = decode_packed_format_pieces(&bytes)?;
    // Args: an `Array` of `Argument::new_display|new_debug(&v)` ctors, one
    // per placeholder.
    let arg_elems = read_array_literal_elements(graph, &args_var)?;
    if arg_elems.len() != placeholder_count {
        return None;
    }
    let mut args = Vec::with_capacity(arg_elems.len());
    for elem in &arg_elems {
        args.push(extract_fmt_arg(graph, elem)?);
    }
    Some(FmtChain { pieces, args })
}

/// Emit a `__str_const` constant of `text` into `bb_id` and return its
/// Variable — the same synthetic `Call(["__str_const", text])` shape the
/// constant lowering uses (see [`Lowering::emit_constant`]); the flowspace
/// adapter pre-folds it to the upstream string `Constant` and types it as
/// a String.
fn emit_str_const(graph: &mut FunctionGraph, bb_id: BlockId, text: &str) -> Variable {
    use crate::model::{CallTarget, OpKind, ValueType};
    let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(var.clone()),
        kind: OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["__str_const".to_string(), text.to_string()],
            },
            args: vec![],
            result_ty: ValueType::Ref(None),
        },
    });
    var
}

/// Emit a string concatenation `lhs + rhs` into `bb_id` and return its
/// Variable. The `"add"` opname passes through the flowspace adapter
/// unchanged; when both operands type as String the rtyper routes it to
/// `pair(StringRepr, StringRepr).rtype_add` → `direct_call(ll_strconcat)`.
fn emit_str_add(
    graph: &mut FunctionGraph,
    bb_id: BlockId,
    lhs: &Variable,
    rhs: &Variable,
) -> Variable {
    use crate::model::{OpKind, ValueType};
    let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
    graph.block_mut(bb_id).operations.push(SpaceOperation {
        result: Some(var.clone()),
        kind: OpKind::BinOp {
            op: "add".to_string(),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            result_ty: ValueType::Ref(None),
        },
    });
    var
}

/// Emit the StringRepr-add left fold that materializes a recognized
/// `format_args!` chain as a runtime String:
///
/// ```text
/// acc = strconst(pieces[0])
/// for each arg i:
///     acc = acc + args[i].value          // render(&str) == identity
///     acc = acc + strconst(pieces[i+1])
/// ```
///
/// Returns the final String Variable. Each literal piece becomes a
/// `__str_const` call and each concatenation a `BinOp("add")` the rtyper
/// lowers through `ll_strconcat`. The caller must have verified every
/// argument renders by identity (a `&str` `Display`); `Debug` rendering
/// and non-`&str` `Display` are outside the recognized subset and are
/// rejected before emission.
fn emit_fmt_concat(graph: &mut FunctionGraph, bb_id: BlockId, chain: &FmtChain) -> Variable {
    let mut acc = emit_str_const(graph, bb_id, &chain.pieces[0]);
    for (i, arg) in chain.args.iter().enumerate() {
        acc = emit_str_add(graph, bb_id, &acc, &arg.value);
        let next_piece = emit_str_const(graph, bb_id, &chain.pieces[i + 1]);
        acc = emit_str_add(graph, bb_id, &acc, &next_piece);
    }
    acc
}

/// Build the orthodox string-build expansion for a recognized
/// single-argument `format!("{pre}{}{post}", value)` chain: the literal
/// pieces become `__str_const`s, the `{}` placeholder renders its value
/// with `str(value)` (`OpKind::UnaryOp { op: "str" }` → the rtyper's
/// `ll_str`), and the parts fold left through `BinOp("add")` (→
/// `ll_strconcat`).  Empty literal pieces are skipped.  The final op
/// reuses `result` (the displaced `alloc::fmt::format` result var) so the
/// downstream consumer is untouched.  Mirrors how RPython lowers a
/// constant-template `%`/`+` string build into native rstr operations,
/// rather than leaving the graph-less `fmt::rt` externs residual.
fn emit_fmt_expansion_ops(
    graph: &mut FunctionGraph,
    pieces: &[String],
    value: &Variable,
    result: Variable,
) -> Vec<SpaceOperation> {
    use crate::model::{CallTarget, OpKind, ValueType};
    let str_const = |graph: &mut FunctionGraph, ops: &mut Vec<SpaceOperation>, text: &str| {
        let v = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        ops.push(SpaceOperation {
            result: Some(v.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__str_const".to_string(), text.to_string()],
                },
                args: vec![],
                result_ty: ValueType::Ref(None),
            },
        });
        v
    };
    let mut ops: Vec<SpaceOperation> = Vec::new();
    let mut parts: Vec<Variable> = Vec::new();
    if !pieces[0].is_empty() {
        parts.push(str_const(graph, &mut ops, &pieces[0]));
    }
    // `str(value)` — render the single Display placeholder.
    let rendered = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
    ops.push(SpaceOperation {
        result: Some(rendered.clone()),
        kind: OpKind::UnaryOp {
            op: "str".to_string(),
            operand: value.clone(),
            result_ty: ValueType::Ref(None),
        },
    });
    parts.push(rendered);
    if pieces.len() > 1 && !pieces[1].is_empty() {
        parts.push(str_const(graph, &mut ops, &pieces[1]));
    }
    // Left-fold the parts through `add` (`ll_strconcat`).
    let mut acc = parts[0].clone();
    for part in &parts[1..] {
        let sum = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        ops.push(SpaceOperation {
            result: Some(sum.clone()),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: acc.clone(),
                rhs: part.clone(),
                result_ty: ValueType::Ref(None),
            },
        });
        acc = sum;
    }
    // Reuse the displaced format result var on the final op so the
    // downstream link still forwards the rendered String.
    if let Some(last) = ops.last_mut() {
        last.result = Some(result);
    }
    ops
}

/// The unique `Link` feeding `target`'s inputarg at `pos`: the source
/// block id, its exit index, and the threaded value Variable. Returns
/// `None` when `target` has more than one predecessor (a phi merge, so
/// no single thread to rewrite) or the position carries a `Const`.
fn single_incoming_link(
    graph: &FunctionGraph,
    target: BlockId,
    pos: usize,
) -> Option<(BlockId, usize, Variable)> {
    let mut found: Option<(BlockId, usize, Variable)> = None;
    for src in &graph.blocks {
        for (ei, link) in src.exits.iter().enumerate() {
            if link.target == target {
                if found.is_some() {
                    return None; // phi merge: ambiguous predecessor
                }
                let v = link.args.get(pos)?.as_variable()?.clone();
                found = Some((src.id, ei, v));
            }
        }
    }
    found
}

/// Position of `var` in `block`'s inputargs.
fn inputarg_pos(graph: &FunctionGraph, block: BlockId, var: &Variable) -> Option<usize> {
    graph
        .blocks
        .iter()
        .find(|b| b.id == block)?
        .inputargs
        .iter()
        .position(|a| a.id() == var.id())
}

/// A recognized single-argument `format!` chain ready to collapse: the
/// terminal `alloc::fmt::format` op, the literal pieces, the forwarding
/// links to rewrite (so the rendered value threads straight to the format
/// block in place of the `new_display`/`Arguments` values), and the
/// now-dead chain ops to delete.
struct FmtCollapse {
    format_block: BlockId,
    format_result: u64,
    pieces: Vec<String>,
    /// `(block, exit_index, arg_pos, replacement)` — replace the chain
    /// value the link forwarded with the threaded rendered value.
    link_rewrites: Vec<(BlockId, usize, usize, Variable)>,
    /// Op result var ids to delete (chain ctors / `new_display` / pieces
    /// bytes / `Arguments::new`).
    dead_results: Vec<u64>,
    /// Aggregate base var ids whose `FieldWrite`s are deleted.
    dead_bases: Vec<u64>,
}

/// Recognize the single-argument `format!` chain terminating at the
/// `alloc::fmt::format` op `(bf, fi)` and collect the rewrite plan, or
/// `None` for any shape outside the recognized subset (multi-argument,
/// non-threaded, or phi-merged) so the collapse leaves the graph
/// untouched. The recognized shape is the one charon lowers for
/// `f(format!("…{x}"))`: block B0 builds `Argument::new_display(&x)` off
/// the argument Tuple and forwards it; block Bp builds the args + pieces
/// arrays and `Arguments::new`, forwarding the `Arguments`; block Bf
/// calls `alloc::fmt::format`.
fn collect_fmt_collapse(graph: &FunctionGraph, bf: BlockId, fi: usize) -> Option<FmtCollapse> {
    use crate::model::{CallTarget, OpKind};
    let block_f = graph.blocks.iter().find(|b| b.id == bf)?;
    let format_op = block_f.operations.get(fi)?;
    let (fmt_args, format_result) = match &format_op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } if fmt_path_ends_with(segments, &["fmt", "format"]) => {
            (args.first()?.clone(), format_op.result.as_ref()?.id())
        }
        _ => return None,
    };
    let chain = extract_fmt_chain(graph, &fmt_args)?;
    if chain.args.len() != 1 {
        return None; // scope: single-argument chains
    }
    if chain.args[0].kind != FmtArgKind::Display {
        // `str(value)` renders Display; `{:?}` Debug has no native rstr
        // counterpart, so leave a Debug chain residual.
        return None;
    }
    let pieces = chain.pieces.clone();

    // `fmt_args` reaches Bf as an inputarg threaded from Bp's
    // `Arguments::new` result.
    let pf = inputarg_pos(graph, bf, &fmt_args)?;
    let (bp, ei_bf, arguments_var) = single_incoming_link(graph, bf, pf)?;
    if bp == bf {
        return None;
    }
    let block_p = graph.blocks.iter().find(|b| b.id == bp)?;
    let (pieces_var, args_var) = block_p.operations.iter().find_map(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } if op.result.as_ref().map(|r| r.id()) == Some(arguments_var.id())
            && is_arguments_new_path(segments) =>
        {
            Some((args.first()?.clone(), args.get(1)?.clone()))
        }
        _ => None,
    })?;
    let piece_byte_vars = read_array_literal_elements(graph, &pieces_var)?;
    let arg_elems = read_array_literal_elements(graph, &args_var)?;
    if arg_elems.len() != 1 {
        return None;
    }
    let arg_elem = arg_elems.into_iter().next()?;

    // `arg_elem` reaches Bp as an inputarg threaded from B0's
    // `new_display` result.
    let pe = inputarg_pos(graph, bp, &arg_elem)?;
    let (b0, ei_bp, new_display_var) = single_incoming_link(graph, bp, pe)?;
    if b0 == bp {
        return None;
    }
    let block_0 = graph.blocks.iter().find(|b| b.id == b0)?;
    let (arg_ref, tuple_var) = block_0.operations.iter().find_map(|op| match &op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } if op.result.as_ref().map(|r| r.id()) == Some(new_display_var.id())
            && fmt_argument_ctor_kind(segments).is_some() =>
        {
            let arg_ref = args.first()?.clone();
            let (fr_block, fr_idx) = resolve_to_producer_op(graph, &arg_ref)?;
            let frb = graph.blocks.iter().find(|b| b.id == fr_block)?;
            match &frb.operations.get(fr_idx)?.kind {
                OpKind::FieldRead { base, .. } => Some((arg_ref.clone(), base.clone())),
                _ => None,
            }
        }
        _ => None,
    })?;
    // The rendered value written into the argument tuple field.
    let context = unwrap_fmt_arg_tuple_ref(graph, &arg_ref)?;

    // Thread `context` straight through the slots the chain values used:
    // B0→Bp forwards `context` where it forwarded `new_display`, Bp→Bf
    // forwards Bp's now-`context`-bound inputarg where it forwarded
    // `Arguments`.  The format op then reads `context` in place of the
    // `Arguments` value.
    let link_rewrites = vec![
        (b0, ei_bp, pe, context.clone()),
        (bp, ei_bf, pf, arg_elem.clone()),
    ];
    let mut dead_results = vec![
        new_display_var.id(),
        arg_ref.id(),
        tuple_var.id(),
        arguments_var.id(),
        pieces_var.id(),
        args_var.id(),
    ];
    dead_results.extend(piece_byte_vars.iter().map(|v| v.id()));
    let dead_bases = vec![tuple_var.id(), pieces_var.id(), args_var.id()];

    Some(FmtCollapse {
        format_block: bf,
        format_result,
        pieces,
        link_rewrites,
        dead_results,
        dead_bases,
    })
}

/// Expand every recognized single-argument `format!` chain into the
/// rtyper's native string-build operations: the chain's
/// `Argument::new_display` ctor, on-stack pieces/args arrays, argument
/// Tuple and `Arguments::new` are deleted, the rendered value is threaded
/// straight to the `alloc::fmt::format` block, and that op is replaced by
/// `str(value)` (`ll_str`) folded with the literal pieces through
/// `add` (`ll_strconcat`).  This is the orthodox lowering — every emitted
/// op (`__str_const`, `str`, `add`) the rtyper natively types `SomeString`
/// and the legacy walker / codewriter / runtime already handle — so the
/// graph-less `fmt::rt::Argument` / `fmt::Arguments` externs no longer
/// block the rtyper, with no opaque residual or fresh runtime helper.
fn collapse_fmt_chains(graph: &mut FunctionGraph) -> usize {
    use crate::model::{CallTarget, LinkArg, OpKind};
    let sites: Vec<FmtCollapse> = graph
        .blocks
        .iter()
        .flat_map(|block| {
            block
                .operations
                .iter()
                .enumerate()
                .filter_map(move |(fi, op)| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments },
                        ..
                    } if fmt_path_ends_with(segments, &["fmt", "format"]) => Some((block.id, fi)),
                    _ => None,
                })
        })
        .filter_map(|(bid, fi)| collect_fmt_collapse(graph, bid, fi))
        .collect();
    if sites.is_empty() {
        return 0;
    }
    // 1. Re-thread the rendered values onto the forwarding links.
    for site in &sites {
        for (bid, ei, pos, repl) in &site.link_rewrites {
            if let Some(link) = graph.block_mut(*bid).exits.get_mut(*ei) {
                if let Some(arg) = link.args.get_mut(*pos) {
                    *arg = LinkArg::Value(repl.clone());
                }
            }
        }
    }
    // 2. Delete the now-dead chain ops across all blocks.
    let dead_results: std::collections::HashSet<u64> = sites
        .iter()
        .flat_map(|s| s.dead_results.iter().copied())
        .collect();
    let dead_bases: std::collections::HashSet<u64> = sites
        .iter()
        .flat_map(|s| s.dead_bases.iter().copied())
        .collect();
    for block in &mut graph.blocks {
        block.operations.retain(|op| {
            if let Some(r) = &op.result {
                if dead_results.contains(&r.id()) {
                    return false;
                }
            }
            if let OpKind::FieldWrite { base, .. } = &op.kind {
                if dead_bases.contains(&base.id()) {
                    return false;
                }
            }
            true
        });
    }
    // 3. Replace each `alloc::fmt::format` op (now reading the threaded
    //    rendered value) with the orthodox `str(value)` + `ll_strconcat`
    //    expansion, reusing the format result var.
    for site in &sites {
        let Some((idx, value, result)) = graph
            .blocks
            .iter()
            .find(|b| b.id == site.format_block)
            .and_then(|b| {
                let idx = b.operations.iter().position(|op| {
                    op.result.as_ref().map(|r| r.id()) == Some(site.format_result)
                })?;
                let value = match &b.operations[idx].kind {
                    OpKind::Call { args, .. } => args.first()?.clone(),
                    _ => return None,
                };
                Some((idx, value, b.operations[idx].result.clone()?))
            })
        else {
            continue;
        };
        let expansion = emit_fmt_expansion_ops(graph, &site.pieces, &value, result);
        graph
            .block_mut(site.format_block)
            .operations
            .splice(idx..idx + 1, expansion);
    }
    sites.len()
}

/// A recognized multi-argument `format!` chain ready to collapse.  Unlike
/// the single-argument [`FmtCollapse`], a multi-arg chain builds an N-field
/// argument tuple and one `Argument::new_display` ctor per placeholder
/// (spread across several blocks), and the rendered values already thread
/// to the `Arguments::new` block as the args-array elements.  So the
/// collapse renders each placeholder in place (`new_display(inner)` →
/// `str(inner)`) and folds the rendered values with the literal pieces at
/// the `Arguments::new` block, threading the single resulting String to the
/// `alloc::fmt::format` block.
struct FmtCollapseMulti {
    /// Block holding `Arguments::new` — where the concat fold is emitted
    /// (every rendered value is live there, as an args-array element).
    args_block: BlockId,
    /// `Arguments::new` result var id (re-threaded to the folded String,
    /// then deleted).
    arguments_var: u64,
    /// The rendered-value vars in placeholder order (the args-array
    /// elements, live as inputargs of `args_block`).
    arg_elem_vars: Vec<Variable>,
    pieces: Vec<String>,
    /// `(block, op_index, inner)` per `Argument::new_display` ctor to
    /// rewrite into `str(inner)` in place.
    new_display_ops: Vec<(BlockId, usize, Variable)>,
    format_block: BlockId,
    format_idx: usize,
    /// The var the `alloc::fmt::format` op reads (an `Arguments`-threaded
    /// inputarg that, after the link rewrite, carries the folded String).
    fmt_args: Variable,
    /// Op result var ids to delete (args/pieces array ctors, packed byte
    /// consts, `Arguments::new`, and — when the argument-Tuple round-trip is
    /// eliminated — the Tuple ctor + its `__pos_N` `FieldRead`s).
    dead_results: Vec<u64>,
    /// Aggregate base var ids whose `FieldWrite`s are deleted (the args and
    /// pieces arrays, plus the argument-Tuple base when eliminated).
    dead_bases: Vec<u64>,
    /// `(block, exit_idx, arg_pos, value)` link-arg rebinds applied before
    /// deletion: a `format_args!` Tuple ref threaded to a placeholder's block
    /// is replaced by the rendered value, so the threaded inputarg `str`
    /// reads now carries the value rather than the Tuple field. Empty unless
    /// the Tuple round-trip is eliminated ([`attempt_fmt_tuple_elimination`]).
    link_rebinds: Vec<(BlockId, usize, usize, Variable)>,
}

/// True if `op` reads `var` in any operand position.  Covers the front-end
/// operand-bearing `OpKind` variants present during `finish` — the fmt
/// collapse runs pre-`jtransform`, so the JIT-only call / vable / guard-value
/// families never appear and cannot carry the on-stack `format_args!` Tuple
/// ref; other variants default to `false`.
fn op_reads_var(op: &crate::model::SpaceOperation, var: &Variable) -> bool {
    use crate::model::{LinkArg, OpKind};
    let v = var.id();
    let is = |x: &Variable| x.id() == v;
    let arg_is = |a: &LinkArg| matches!(a, LinkArg::Value(x) if x.id() == v);
    match &op.kind {
        OpKind::FieldRead { base, .. } => is(base),
        OpKind::FieldWrite { base, value, .. } => is(base) || arg_is(value),
        OpKind::ArrayRead { base, index, .. } => is(base) || is(index),
        OpKind::ArrayWrite {
            base, index, value, ..
        } => is(base) || is(index) || arg_is(value),
        OpKind::InteriorFieldRead { base, index, .. } => is(base) || is(index),
        OpKind::InteriorFieldWrite {
            base, index, value, ..
        } => is(base) || is(index) || is(value),
        OpKind::Call { args, .. } => args.iter().any(is),
        OpKind::BinOp { lhs, rhs, .. } => is(lhs) || is(rhs),
        OpKind::UnaryOp { operand, .. } => is(operand),
        OpKind::GuardTrue { cond } | OpKind::GuardFalse { cond } => is(cond),
        _ => false,
    }
}

/// The plan to also delete the `format_args!` argument-Tuple round-trip in a
/// multi-arg chain — the deletion the single-arg [`collect_fmt_collapse`]
/// always performs but the multi-arg path historically skipped.  Without it
/// the Tuple ctor + its `__pos_N` `FieldWrite`/`FieldRead`s survive, and a
/// placeholder value (a `StringRepr`) written into the Tuple's
/// PyObject-erased `__pos_N` field walls the rtyper (`convertvar(StringRepr →
/// InstanceRepr PyObject)`, no pairtype arm) when the Tuple is pinned by a
/// field read-back.  The Tuple is a Rust `format_args!` artifact with no PyPy
/// counterpart (PyPy joins the message flat, then boxes once at the exception
/// boundary), so deleting it is the parity-correct collapse.
///
/// Resolves each placeholder to its rendered value (`chain.args[i].value`,
/// already unwrapped through the Tuple by `extract_fmt_chain`) so `str` reads
/// the value, not the Tuple field.  A placeholder whose `new_display` sits in
/// the Tuple-construction block reads the value directly; one in a successor
/// block (the common Charon call-boundary split) reads it through a slot
/// rebind on the single incoming link that forwards the Tuple ref.  Returns
/// `None` (leaving the historic in-place `str(field-read)` collapse, which
/// keeps the residual Tuple but never reintroduces the `fmt` extern) for any
/// shape outside this set, so the change is strictly additive and a hard
/// shape bails rather than half-deleting.
struct FmtTupleElim {
    /// Per placeholder (chain order): the var `str` should read.
    str_inputs: Vec<Variable>,
    /// `(block, exit_idx, arg_pos, value)` Tuple-ref slot rebinds.
    link_rebinds: Vec<(BlockId, usize, usize, Variable)>,
    /// Tuple ctor + per-placeholder `FieldRead` result ids to delete.
    dead_results: Vec<u64>,
    /// Tuple base id whose `FieldWrite`s are deleted.
    dead_bases: Vec<u64>,
}

/// Whether every `FieldWrite` storing into the argument `Tuple` `tv` lives
/// in `bc` (the ctor block).  Matches a write by Variable id or by producer
/// identity (a threaded copy of the ref carries a different id but the same
/// producing ctor op), mirroring [`unwrap_fmt_arg_tuple_ref`].  When true,
/// every field value is in scope at `bc`, hence at any block `bc`
/// dominates — so an in-place `str(value)` rewrite at a cross-block reader
/// is sound.
fn all_tuple_writes_in_block(graph: &FunctionGraph, tv: &Variable, bc: BlockId) -> bool {
    use crate::model::OpKind;
    let tv_producer = resolve_to_producer_op(graph, tv);
    for b in &graph.blocks {
        for op in &b.operations {
            if let OpKind::FieldWrite { base, .. } = &op.kind {
                let denotes_tuple = base.id() == tv.id()
                    || (tv_producer.is_some()
                        && resolve_to_producer_op(graph, base) == tv_producer);
                if denotes_tuple && b.id != bc {
                    return false;
                }
            }
        }
    }
    true
}

fn attempt_fmt_tuple_elimination(
    graph: &FunctionGraph,
    chain: &FmtChain,
    new_display_ops: &[(BlockId, usize, Variable)],
) -> Option<FmtTupleElim> {
    use crate::model::{ExitSwitch, LinkArg, OpKind};
    if chain.args.len() != new_display_ops.len() {
        return None;
    }

    // The shared argument-Tuple ctor, resolved from each placeholder's
    // `new_display` arg `FieldRead` base; all placeholders must read the same
    // Tuple, or the chain is not the single private `format_args!` Tuple.
    let mut tuple_var: Option<Variable> = None;
    let mut ctor_block: Option<BlockId> = None;

    enum Plan {
        Local {
            value: Variable,
            getattr_result: u64,
        },
        Threaded {
            carrier: Variable,
            getattr_result: u64,
            rebind: (BlockId, usize, usize, Variable),
        },
    }
    let mut plans: Vec<Plan> = Vec::with_capacity(new_display_ops.len());

    for (i, (nd_block, _nd_idx, inner)) in new_display_ops.iter().enumerate() {
        let value = chain.args.get(i)?.value.clone();
        // `inner` is a `FieldRead` of the Tuple's `__pos_i` field; co-located
        // with its `new_display` (the Charon array lowering keeps them in one
        // block).
        let (fr_block, fr_idx) = resolve_to_producer_op(graph, inner)?;
        if fr_block != *nd_block {
            return None;
        }
        let frb = graph.blocks.iter().find(|b| b.id == fr_block)?;
        let carrier = match &frb.operations.get(fr_idx)?.kind {
            OpKind::FieldRead { base, .. } => base.clone(),
            _ => return None,
        };
        let getattr_result = inner.id();

        // The Tuple ctor: the carrier resolves to it (directly for the local
        // placeholder, through the forwarding link for a threaded copy).
        let (cb, ci) = resolve_to_producer_op(graph, &carrier)?;
        let ctor_result = graph
            .blocks
            .iter()
            .find(|b| b.id == cb)?
            .operations
            .get(ci)?
            .result
            .as_ref()?
            .clone();
        match (&tuple_var, &ctor_block) {
            (Some(tv), Some(bc)) if tv.id() != ctor_result.id() || *bc != cb => return None,
            _ => {
                tuple_var = Some(ctor_result.clone());
                ctor_block = Some(cb);
            }
        }
        let tv = tuple_var.clone()?;
        let bc = ctor_block?;

        if carrier.id() == tv.id() {
            // Local: the value is in scope where the Tuple is built.  A
            // cross-block reader (`format_args!` splits the placeholder
            // field reads across a straight-line block boundary) is still
            // sound when `bc` dominates the reader and every Tuple write
            // lives in `bc`, so each field value is available at the reader.
            if *nd_block != bc
                && !(block_dominates(graph, bc, *nd_block)
                    && all_tuple_writes_in_block(graph, &tv, bc))
            {
                return None;
            }
            plans.push(Plan::Local {
                value,
                getattr_result,
            });
            continue;
        }

        // Threaded: the carrier is an inputarg at position `pos`, fed by a
        // single incoming link from the ctor block that forwards the Tuple
        // ref at that position; rebind that slot to the value.  The carrier
        // must be used in this block only by this one `FieldRead` (never
        // forwarded onward / read again / used in the exit switch) so the
        // rebind is sound.  `LinkArg`s hold the predecessor's source vars,
        // which correspond positionally to the target's inputargs.
        let nd_blk = graph.blocks.iter().find(|b| b.id == *nd_block)?;
        let pos = nd_blk
            .inputargs
            .iter()
            .position(|a| a.id() == carrier.id())?;
        let incoming: Vec<(BlockId, usize)> = graph
            .blocks
            .iter()
            .flat_map(|b| {
                b.exits
                    .iter()
                    .enumerate()
                    .filter_map(move |(ei, l)| (l.target == *nd_block).then_some((b.id, ei)))
            })
            .collect();
        if incoming.len() != 1 {
            return None; // phi merge: no single forwarding slot to rebind
        }
        let (pred, ei) = incoming[0];
        if pred != bc {
            return None;
        }
        let forwards_tuple = matches!(
            graph
                .blocks
                .iter()
                .find(|b| b.id == pred)?
                .exits
                .get(ei)
                .and_then(|l| l.args.get(pos)),
            Some(LinkArg::Value(x)) if x.id() == tv.id()
        );
        if !forwards_tuple {
            return None;
        }
        let used_elsewhere = nd_blk
            .operations
            .iter()
            .enumerate()
            .any(|(idx, op)| idx != fr_idx && op_reads_var(op, &carrier))
            || nd_blk.exits.iter().any(|l| {
                l.args
                    .iter()
                    .any(|a| matches!(a, LinkArg::Value(x) if x.id() == carrier.id()))
            })
            || matches!(&nd_blk.exitswitch, Some(ExitSwitch::Value(x)) if x.id() == carrier.id());
        if used_elsewhere {
            return None;
        }
        plans.push(Plan::Threaded {
            carrier: carrier.clone(),
            getattr_result,
            rebind: (pred, ei, pos, value.clone()),
        });
    }

    let tuple_var = tuple_var?;
    let link_rebinds: Vec<(BlockId, usize, usize, Variable)> = plans
        .iter()
        .filter_map(|p| match p {
            Plan::Threaded { rebind, .. } => Some(rebind.clone()),
            Plan::Local { .. } => None,
        })
        .collect();

    let mut dead_results = vec![tuple_var.id()];
    let mut str_inputs = Vec::with_capacity(plans.len());
    for p in &plans {
        match p {
            Plan::Local {
                value,
                getattr_result,
            } => {
                str_inputs.push(value.clone());
                dead_results.push(*getattr_result);
            }
            Plan::Threaded {
                carrier,
                getattr_result,
                ..
            } => {
                str_inputs.push(carrier.clone());
                dead_results.push(*getattr_result);
            }
        }
    }

    // Deleting the ctor, its `FieldWrite`s, and the placeholder `FieldRead`s
    // leaves the framestate forwards of the (now dead) Tuple ref for the
    // post-collapse `prune_dead_phis` in `finish` to clean up (it trims dead
    // `Link.args` + inputargs).  That is sound only when the Tuple ref has no
    // *surviving* op reader: every op that reads it must be one we delete (a
    // placeholder `FieldRead` whose result is in `dead_results`, or a
    // `FieldWrite` into it).  Threaded slots are redirected to the value by
    // `link_rebinds`; any remaining forward of a fully-dead Tuple ref is what
    // `prune_dead_phis` removes.  A surviving reader (the ref escapes the
    // chain) would dangle, so bail to the historic in-place `str(field-read)`
    // collapse (residual Tuple, no regression).
    let tuple_has_live_reader = graph.blocks.iter().any(|b| {
        b.operations.iter().any(|op| {
            let deleted = op
                .result
                .as_ref()
                .is_some_and(|r| dead_results.contains(&r.id()))
                || matches!(&op.kind, OpKind::FieldWrite { base, .. } if base.id() == tuple_var.id());
            !deleted
                && crate::inline::op_variable_refs(&op.kind)
                    .iter()
                    .any(|v| v.id() == tuple_var.id())
        }) || matches!(&b.exitswitch, Some(ExitSwitch::Value(x)) if x.id() == tuple_var.id())
    });
    if tuple_has_live_reader {
        return None;
    }

    Some(FmtTupleElim {
        str_inputs,
        link_rebinds,
        dead_results,
        dead_bases: vec![tuple_var.id()],
    })
}

/// Recognize a multi-argument `format!` chain terminating at the
/// `alloc::fmt::format` op `(bf, fi)` and collect the rewrite plan, or
/// `None` for any shape outside the recognized subset (single-argument,
/// Debug, non-threaded, phi-merged) so the collapse leaves the graph
/// untouched.
fn collect_fmt_collapse_multi(
    graph: &FunctionGraph,
    bf: BlockId,
    fi: usize,
) -> Option<FmtCollapseMulti> {
    use crate::model::{CallTarget, OpKind};
    let block_f = graph.blocks.iter().find(|b| b.id == bf)?;
    let format_op = block_f.operations.get(fi)?;
    let fmt_args = match &format_op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } if fmt_path_ends_with(segments, &["fmt", "format"]) => args.first()?.clone(),
        _ => return None,
    };
    let chain = extract_fmt_chain(graph, &fmt_args)?;
    if chain.args.len() < 2 {
        return None; // single-argument chains handled by `collapse_fmt_chains`
    }
    if chain.args.iter().any(|a| a.kind != FmtArgKind::Display) {
        // `{:?}` Debug over an enum has no native rstr render (no
        // `rtype_str` on enum reprs); leave Debug chains residual.
        return None;
    }
    let pieces = chain.pieces.clone();

    // The `Arguments::new(pieces, args)` producer block: where the concat
    // fold is emitted (all rendered values are live there as the args-array
    // elements).  Require it distinct from the format block so a forwarding
    // link exists to re-thread the folded String onto.
    let (args_block, args_idx) = resolve_to_producer_op(graph, &fmt_args)?;
    if args_block == bf {
        return None;
    }
    let block_p = graph.blocks.iter().find(|b| b.id == args_block)?;
    let args_new_op = block_p.operations.get(args_idx)?;
    let (pieces_var, args_var) = match &args_new_op.kind {
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            args,
            ..
        } if is_arguments_new_path(segments) => (args.first()?.clone(), args.get(1)?.clone()),
        _ => return None,
    };
    let arguments_var = args_new_op.result.as_ref()?.id();
    let piece_byte_vars = read_array_literal_elements(graph, &pieces_var)?;
    let arg_elem_vars = read_array_literal_elements(graph, &args_var)?;
    if arg_elem_vars.len() != chain.args.len() {
        return None;
    }

    // Each args-array element traces back to a `new_display` ctor; record it
    // for the in-place `→ str` rewrite.
    let mut new_display_ops = Vec::with_capacity(arg_elem_vars.len());
    for elem in &arg_elem_vars {
        let (b_i, idx_i) = resolve_to_producer_op(graph, elem)?;
        let blk = graph.blocks.iter().find(|b| b.id == b_i)?;
        let inner = match &blk.operations.get(idx_i)?.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                args,
                ..
            } if fmt_argument_ctor_kind(segments) == Some(FmtArgKind::Display) => {
                args.first()?.clone()
            }
            _ => return None,
        };
        new_display_ops.push((b_i, idx_i, inner));
    }

    let mut dead_results = vec![arguments_var, pieces_var.id(), args_var.id()];
    dead_results.extend(piece_byte_vars.iter().map(|v| v.id()));
    let mut dead_bases = vec![pieces_var.id(), args_var.id()];

    // Also delete the `format_args!` argument-Tuple round-trip when the shape
    // permits (mirroring the single-arg path).  On success each placeholder's
    // `str` reads the rendered value instead of the Tuple field, so the Tuple
    // ctor + its writes/reads become dead.  On `None` the historic in-place
    // `str(field-read)` collapse stands (residual Tuple, no `fmt` extern).
    let mut link_rebinds = Vec::new();
    if let Some(elim) = attempt_fmt_tuple_elimination(graph, &chain, &new_display_ops) {
        for (slot, value) in new_display_ops.iter_mut().zip(elim.str_inputs) {
            slot.2 = value;
        }
        dead_results.extend(elim.dead_results);
        dead_bases.extend(elim.dead_bases);
        link_rebinds = elim.link_rebinds;
    }

    Some(FmtCollapseMulti {
        args_block,
        arguments_var,
        arg_elem_vars,
        pieces,
        new_display_ops,
        format_block: bf,
        format_idx: fi,
        fmt_args,
        dead_results,
        dead_bases,
        link_rebinds,
    })
}

/// Expand every recognized multi-argument `format!` chain into native
/// `str` + `ll_strconcat` ops: each `Argument::new_display(inner)` becomes
/// `str(inner)` in place (rendering the placeholder; `str(&str)` folds to
/// identity), the rendered values are folded with the literal pieces at the
/// `Arguments::new` block, and the resulting String is threaded to the
/// `alloc::fmt::format` block (whose op becomes `same_as`).  The on-stack
/// args/pieces arrays and `Arguments::new` are deleted.  Every emitted op
/// the rtyper / codewriter / runtime already handle, so the graph-less
/// `fmt::rt::Argument` / `fmt::Arguments` externs no longer block the
/// rtyper.
fn collapse_fmt_chains_multi(graph: &mut FunctionGraph) -> usize {
    use crate::model::{CallTarget, LinkArg, OpKind, ValueType};
    let sites: Vec<FmtCollapseMulti> = graph
        .blocks
        .iter()
        .flat_map(|block| {
            block
                .operations
                .iter()
                .enumerate()
                .filter_map(move |(fi, op)| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments },
                        ..
                    } if fmt_path_ends_with(segments, &["fmt", "format"]) => Some((block.id, fi)),
                    _ => None,
                })
        })
        .filter_map(|(bid, fi)| collect_fmt_collapse_multi(graph, bid, fi))
        .collect();
    if sites.is_empty() {
        return 0;
    }
    for site in &sites {
        // 1. Render each placeholder in place: `new_display(inner)` →
        //    `str(inner)`.  The result var is unchanged, so the rendered
        //    &str threads to the args block exactly where the opaque
        //    `Argument` value did.
        for (b_i, idx_i, inner) in &site.new_display_ops {
            if let Some(op) = graph.block_mut(*b_i).operations.get_mut(*idx_i) {
                op.kind = OpKind::UnaryOp {
                    op: "str".to_string(),
                    operand: inner.clone(),
                    result_ty: ValueType::Ref(None),
                };
            }
        }
        // 1b. Rebind any `format_args!` Tuple-ref slots to the rendered value
        //     (Tuple round-trip elimination): the threaded inputarg a `str`
        //     now reads carries the value instead of the deleted Tuple field.
        for (bid, ei, pos, value) in &site.link_rebinds {
            if let Some(link) = graph.block_mut(*bid).exits.get_mut(*ei) {
                if let Some(arg) = link.args.get_mut(*pos) {
                    *arg = LinkArg::Value(value.clone());
                }
            }
        }
        // 2. Replace the `alloc::fmt::format` op with `same_as(fmt_args)`:
        //    after the link re-thread below, `fmt_args` carries the folded
        //    String, so the format result forwards it unchanged.
        if let Some(op) = graph
            .block_mut(site.format_block)
            .operations
            .get_mut(site.format_idx)
        {
            op.kind = OpKind::UnaryOp {
                op: "same_as".to_string(),
                operand: site.fmt_args.clone(),
                result_ty: ValueType::Ref(None),
            };
        }
        // 3. Fold the rendered values with the literal pieces at the args
        //    block (`piece0 ++ rendered0 ++ piece1 ++ … ++ pieceN`).  The
        //    args-array elements are already rendered (step 1), so they are
        //    concatenated directly.
        let fold_chain = FmtChain {
            pieces: site.pieces.clone(),
            args: site
                .arg_elem_vars
                .iter()
                .map(|v| FmtArg {
                    value: v.clone(),
                    kind: FmtArgKind::Display,
                })
                .collect(),
        };
        let folded = emit_fmt_concat(graph, site.args_block, &fold_chain);
        // 4. Re-thread the folded String onto the link that forwarded the
        //    `Arguments` value out of the args block.
        for link in &mut graph.block_mut(site.args_block).exits {
            for arg in &mut link.args {
                if let LinkArg::Value(v) = arg {
                    if v.id() == site.arguments_var {
                        *arg = LinkArg::Value(folded.clone());
                    }
                }
            }
        }
    }
    // 5. Delete the now-dead chain ops across all blocks (args/pieces
    //    arrays + their `FieldWrite`s, packed byte consts, `Arguments::new`).
    let dead_results: std::collections::HashSet<u64> = sites
        .iter()
        .flat_map(|s| s.dead_results.iter().copied())
        .collect();
    let dead_bases: std::collections::HashSet<u64> = sites
        .iter()
        .flat_map(|s| s.dead_bases.iter().copied())
        .collect();
    for block in &mut graph.blocks {
        block.operations.retain(|op| {
            if let Some(r) = &op.result {
                if dead_results.contains(&r.id()) {
                    return false;
                }
            }
            if let OpKind::FieldWrite { base, .. } = &op.kind {
                if dead_bases.contains(&base.id()) {
                    return false;
                }
            }
            true
        });
    }
    sites.len()
}

/// A block is collapsible into a bare implicit-`AssertionError` raise when
/// every op is a pure value / ctor / field op or a recognised `fmt`
/// message extern — the message-building work a Rust `panic!` / `assert!`
/// emits before its diverging call.  A `FieldWrite` is allowed only when
/// its base aggregate was constructed earlier in the same block (the
/// `(a, b)` tuple / `Some(_)` / pieces array the message builder fills),
/// so a write into an escaped object (`self.field = …; panic!()`) keeps
/// the block non-collapsible and its effect preserved.  Any other `Call`
/// — a real side-effecting function such as `Write::write_fmt` — likewise
/// blocks the collapse.
fn panic_block_is_pure_message(block: &crate::model::Block) -> bool {
    use crate::model::{CallTarget, OpKind};
    let is_message_extern = |segments: &[String]| -> bool {
        if segments.first().map(String::as_str) == Some("__str_const") {
            return true;
        }
        let n = segments.len();
        n >= 2
            && segments.first().map(String::as_str) == Some("fmt")
            && matches!(segments[n - 2].as_str(), "Argument" | "Arguments")
    };
    let mut produced = std::collections::HashSet::new();
    for op in &block.operations {
        let pure = match &op.kind {
            OpKind::ConstInt(_)
            | OpKind::ConstBool(_)
            | OpKind::ConstFloat(_)
            | OpKind::ConstRef(_)
            | OpKind::ConstRefNull
            | OpKind::ConstRefAddr(_)
            | OpKind::ConstSymbolic { .. }
            | OpKind::FieldRead { .. }
            | OpKind::BinOp { .. }
            | OpKind::UnaryOp { .. } => true,
            OpKind::FieldWrite { base, .. } => produced.contains(&base.id()),
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { .. },
                ..
            } => true,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                ..
            } => is_message_extern(segments),
            _ => false,
        };
        if !pure {
            return false;
        }
        if let Some(r) = &op.result {
            produced.insert(r.id());
        }
    }
    true
}

/// Collapse a Rust `panic!` / `assert!` message-building block chain into
/// a bare implicit-`AssertionError` raise, matching RPython's `_implicit_`
/// exception form (`flowcontext.py`; `remove_assertion_errors`,
/// simplify.py:321-346): RPython's implicit exceptions are direct raises
/// carrying no computed message, whereas Rust lowers `panic!("…{}", x)`
/// into a chain of message blocks (`fmt::rt::Argument::new_display`,
/// `fmt::Arguments::new`, the on-stack pieces / args arrays) ending in the
/// `Abort` → `set_raise_implicit(AssertionError)` exit.  Those message
/// blocks are graph-less host externs the rtyper can't type, so they wall
/// the dual gate.
///
/// Each message block has a single exit and only pure message ops
/// (`panic_block_is_pure_message`); the chain tail exits to `exceptblock`
/// carrying the `[AssertionError, value]` pair `set_raise_implicit`
/// installs.  Growing that property backward over single-exit edges yields
/// the set of collapsible blocks; the deciding block at the head (a `bool`
/// switch whose panic arm enters the chain) is not collapsible.  Its panic
/// edge is retargeted straight to `exceptblock` with the chain's
/// `[AssertionError, value]` args, leaving the message blocks unreachable
/// for `clear_unreachable_blocks`, after which `remove_assertion_errors`
/// prunes the now-direct AssertionError edge exactly as for an
/// already-direct implicit raise.
///
/// Returns the number of retargeted deciding edges so the caller can gate
/// the follow-up `clear_unreachable_blocks` / `simplify_lowered_graph`.
fn collapse_panic_message_chains(graph: &mut FunctionGraph) -> usize {
    use crate::flowspace::model::{ConstValue, HOST_ENV};
    use crate::model::LinkArg;
    use std::collections::HashMap;
    let assert_err = HOST_ENV
        .lookup_builtin("AssertionError")
        .expect("HOST_ENV missing AssertionError");
    let exceptblock = graph.exceptblock;
    let exit_raises_assert = |link: &crate::model::Link| -> bool {
        link.target == exceptblock
            && matches!(
                link.args.first(),
                Some(LinkArg::Const(c))
                    if matches!(&c.value, ConstValue::HostObject(h) if *h == assert_err)
            )
    };
    // Grow the collapsible set backward over single-exit pure-message
    // edges, keyed to the `[AssertionError, value]` args the tail raise
    // carries.
    let mut collapsible: HashMap<crate::model::BlockId, Vec<LinkArg>> = HashMap::new();
    loop {
        let mut changed = false;
        for block in &graph.blocks {
            if collapsible.contains_key(&block.id) || block.exits.len() != 1 {
                continue;
            }
            if !panic_block_is_pure_message(block) {
                continue;
            }
            let exit = &block.exits[0];
            let raise_args = if exit_raises_assert(exit) {
                Some(exit.args.clone())
            } else {
                collapsible.get(&exit.target).cloned()
            };
            if let Some(args) = raise_args {
                collapsible.insert(block.id, args);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Retarget every edge from a non-collapsible block into a collapsible
    // block straight to `exceptblock` with the chain's raise args; the
    // collapsible blocks themselves keep their edges and fall out as
    // unreachable.
    let mut redirected = 0usize;
    for block_idx in 0..graph.blocks.len() {
        if collapsible.contains_key(&graph.blocks[block_idx].id) {
            continue;
        }
        for ei in 0..graph.blocks[block_idx].exits.len() {
            let target = graph.blocks[block_idx].exits[ei].target;
            let Some(args) = collapsible.get(&target) else {
                continue;
            };
            let args = args.clone();
            let exit = &mut graph.blocks[block_idx].exits[ei];
            exit.target = exceptblock;
            exit.args = args;
            redirected += 1;
        }
    }
    redirected
}

#[cfg(test)]
mod tests {
    use super::harden_duplicate_leaf_metadata;
    use super::{cast_kind_is_raw_ptr, cast_pointer_marker_op, charon_type_value_to_ast_string};
    use majit_charon_reader::Llbc;

    #[test]
    fn type_arg_splits_per_instantiation_defers_only_float_unit_empty() {
        use super::type_arg_splits_per_instantiation;
        // Bare named heap classes split (1-word GC pointer in the
        // erased model).
        assert!(type_arg_splits_per_instantiation("Tuple"));
        assert!(type_arg_splits_per_instantiation("W_Object"));
        assert!(type_arg_splits_per_instantiation("str"));
        // Nested generics and tuples are also 1-word GC pointers in the
        // erased model (`Result<StepResult<…>, PyError>` etc.).
        assert!(type_arg_splits_per_instantiation("Option<i32>"));
        assert!(type_arg_splits_per_instantiation("(A,B)"));
        // Scalar integer/bool/char primitives split too: each carries a
        // well-defined int-banked repr, so the per-instantiation `__pos_0`
        // annotates to that repr instead of a program-wide union.
        for p in [
            "bool", "char", "u8", "u32", "u64", "u128", "usize", "i8", "i64", "i128", "isize",
        ] {
            assert!(type_arg_splits_per_instantiation(p), "{p} must split");
        }
        // Deferred: floats (float bank, codewriter descr not yet bank-aware)
        // and the unit/degenerate atoms (no materialised payload field).
        for p in ["f32", "f64", "()", ""] {
            assert!(!type_arg_splits_per_instantiation(p), "{p} must not split");
        }
    }

    #[test]
    fn resolve_to_producer_op_follows_cross_block_inputarg_link() {
        use super::resolve_to_producer_op;
        use crate::flowspace::model::Variable;
        use crate::model::{FunctionGraph, Link, OpKind, SpaceOperation};

        let mut graph = FunctionGraph::new("xblock");
        let a = graph.create_block();
        let b = graph.create_block();

        // block A produces `x` via a ConstInt op.
        let x = Variable::new();
        graph.block_mut(a).operations.push(SpaceOperation {
            result: Some(x.clone()),
            kind: OpKind::ConstInt(7),
        });

        // block B takes `w` as its single inputarg; A links to B passing
        // `x` for `w` (the cross-block threading the recognizer sees).
        let w = Variable::new();
        graph.block_mut(b).inputargs = vec![w.clone()];
        let link = Link::from_variables(&graph, vec![x.clone()], b, None).with_prevblock(a);
        graph.block_mut(a).exits = vec![link];

        // `w` (block B inputarg) back-traces to A's ConstInt op.
        assert_eq!(resolve_to_producer_op(&graph, &w), Some((a, 0)));
        // A direct op result resolves to itself.
        assert_eq!(resolve_to_producer_op(&graph, &x), Some((a, 0)));
        // An unrelated free Variable has no producer.
        assert_eq!(resolve_to_producer_op(&graph, &Variable::new()), None);
    }

    #[test]
    fn items_block_base_accessor_gate_excludes_deref_in_place() {
        use super::graph_is_items_block_base_accessor;

        // The two accessors whose `.add(ITEMS_BLOCK_ITEMS_OFFSET)` body
        // returns the interior items pointer for descr-based consumption —
        // safe to collapse to the receiver, regardless of crate prefix.
        assert!(graph_is_items_block_base_accessor(
            "pyre_object::object_array::items_block_items_base"
        ));
        assert!(graph_is_items_block_base_accessor(
            "pyre_object::object_array::items_block_items_ptr"
        ));
        assert!(graph_is_items_block_base_accessor(
            "pyre_jit::object_array::items_block_items_base"
        ));

        // Bodies that dereference a `.add(NAMED_OFFSET)` interior pointer
        // in place must NOT be aliased — the offset is load-bearing.
        assert!(!graph_is_items_block_base_accessor(
            "pyre_object::tupleobject::w_tuple_getitem_known"
        ));
        assert!(!graph_is_items_block_base_accessor(
            "pyre_interpreter::runtime_ops::load_global_str_extern"
        ));
        // A leaf collision in another module must not widen the gate.
        assert!(!graph_is_items_block_base_accessor(
            "pyre_object::other_mod::items_block_items_base_helper"
        ));
    }

    #[test]
    fn decode_packed_format_pieces_matches_real_llbc_templates() {
        use super::decode_packed_format_pieces;

        // The four fixtures below are the verbatim `[u8; N]` pieces buffers
        // charon lowers for these handler graphs (captured from the real
        // pyre-interpreter.ullbc). Each asserts the reconstructed pieces +
        // placeholder count, i.e. the original format string.

        // `format!("stack underflow during {}", context)`
        let (pieces, n) = decode_packed_format_pieces(&[
            23, 115, 116, 97, 99, 107, 32, 117, 110, 100, 101, 114, 102, 108, 111, 119, 32, 100,
            117, 114, 105, 110, 103, 32, 192, 0,
        ])
        .unwrap();
        assert_eq!(
            pieces,
            vec!["stack underflow during ".to_string(), String::new()]
        );
        assert_eq!(n, 1);

        // `format!("{} indices must be integers or slices, not {}", ..)`
        let (pieces, n) = decode_packed_format_pieces(&[
            192, 41, 32, 105, 110, 100, 105, 99, 101, 115, 32, 109, 117, 115, 116, 32, 98, 101, 32,
            105, 110, 116, 101, 103, 101, 114, 115, 32, 111, 114, 32, 115, 108, 105, 99, 101, 115,
            44, 32, 110, 111, 116, 32, 192, 0,
        ])
        .unwrap();
        assert_eq!(
            pieces,
            vec![
                String::new(),
                " indices must be integers or slices, not ".to_string(),
                String::new(),
            ]
        );
        assert_eq!(n, 2);

        // `format!("'{}' object does not support item assignment", ..)`
        let (pieces, n) = decode_packed_format_pieces(&[
            1, 39, 192, 41, 39, 32, 111, 98, 106, 101, 99, 116, 32, 100, 111, 101, 115, 32, 110,
            111, 116, 32, 115, 117, 112, 112, 111, 114, 116, 32, 105, 116, 101, 109, 32, 97, 115,
            115, 105, 103, 110, 109, 101, 110, 116, 0,
        ])
        .unwrap();
        assert_eq!(
            pieces,
            vec![
                "'".to_string(),
                "' object does not support item assignment".to_string(),
            ]
        );
        assert_eq!(n, 1);

        // `format!("__init__() should return None, not '{}'", ..)`
        let (pieces, n) = decode_packed_format_pieces(&[
            36, 95, 95, 105, 110, 105, 116, 95, 95, 40, 41, 32, 115, 104, 111, 117, 108, 100, 32,
            114, 101, 116, 117, 114, 110, 32, 78, 111, 110, 101, 44, 32, 110, 111, 116, 32, 39,
            192, 1, 39, 0,
        ])
        .unwrap();
        assert_eq!(
            pieces,
            vec![
                "__init__() should return None, not '".to_string(),
                "'".to_string(),
            ]
        );
        assert_eq!(n, 1);
    }

    #[test]
    fn decode_packed_format_pieces_bails_and_handles_edges() {
        use super::decode_packed_format_pieces;

        // A control byte other than 0xC0 (e.g. a format-spec / positional
        // placeholder) must bail so the recognizer leaves the graph alone.
        assert_eq!(decode_packed_format_pieces(&[0xC1, 0]), None);
        assert_eq!(decode_packed_format_pieces(&[0x80, 0]), None);

        // A literal length that overruns the buffer bails.
        assert_eq!(decode_packed_format_pieces(&[5, 65, 66, 0]), None);

        // A 0 byte that is not the final byte bails (terminator is last).
        assert_eq!(decode_packed_format_pieces(&[0, 1, 65, 0]), None);

        // A buffer that runs out without the 0x00 terminator is truncated
        // or wrong — bail rather than accept a malformed template.
        assert_eq!(decode_packed_format_pieces(&[2, 104, 105]), None);
        assert_eq!(decode_packed_format_pieces(&[2, 104, 105, 192]), None);

        // Literal-only template (no placeholders) → single piece.
        let (pieces, n) = decode_packed_format_pieces(&[2, 104, 105, 0]).unwrap();
        assert_eq!(pieces, vec!["hi".to_string()]);
        assert_eq!(n, 0);

        // Two consecutive literal segments accumulate into one piece
        // (how a >127-byte literal is split); one placeholder follows.
        let (pieces, n) = decode_packed_format_pieces(&[1, 97, 1, 98, 192, 0]).unwrap();
        assert_eq!(pieces, vec!["ab".to_string(), String::new()]);
        assert_eq!(n, 1);
    }

    #[test]
    fn read_array_literal_then_decode_pieces_end_to_end() {
        use super::{decode_packed_format_pieces, read_array_literal_elements, resolve_const_int};
        use crate::flowspace::model::Variable;
        use crate::model::{
            CallTarget, FieldDescriptor, FunctionGraph, OpKind, SpaceOperation, ValueType,
        };

        // Build the pieces array the way the front-end lowers it: an
        // `Array` ctor followed by `__pos_i` FieldWrites of ConstInt
        // bytes. Template for `format!("hi{}")` → pieces ["hi", ""],
        // one placeholder = bytes [2, 'h', 'i', 0xC0, 0].
        let mut graph = FunctionGraph::new("arraylit");
        let a = graph.create_block();
        let arr = Variable::new();
        graph.block_mut(a).operations.push(SpaceOperation {
            result: Some(arr.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Array".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Array".to_string())),
            },
        });
        let bytes = [2i64, 104, 105, 0xC0, 0];
        for (i, b) in bytes.iter().enumerate() {
            let v = Variable::new();
            graph.block_mut(a).operations.push(SpaceOperation {
                result: Some(v.clone()),
                kind: OpKind::ConstInt(*b),
            });
            graph.block_mut(a).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: arr.clone(),
                    field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                    value: crate::model::LinkArg::Value(v),
                    ty: ValueType::Ref(None),
                },
            });
        }

        let elements = read_array_literal_elements(&graph, &arr).expect("array elements");
        assert_eq!(elements.len(), 5);
        let decoded: Vec<u8> = elements
            .iter()
            .map(|v| resolve_const_int(&graph, v).expect("const int") as u8)
            .collect();
        assert_eq!(decoded, vec![2, 104, 105, 0xC0, 0]);
        let (pieces, n) = decode_packed_format_pieces(&decoded).unwrap();
        assert_eq!(pieces, vec!["hi".to_string(), String::new()]);
        assert_eq!(n, 1);

        // A non-Array producer (plain ConstInt) is rejected.
        assert_eq!(read_array_literal_elements(&graph, &elements[0]), None);
    }

    #[test]
    fn extract_fmt_chain_recovers_pieces_and_args_cross_block() {
        use super::{FmtArgKind, extract_fmt_chain};
        use crate::flowspace::model::Variable;
        use crate::model::{
            CallTarget, FieldDescriptor, FunctionGraph, Link, OpKind, SpaceOperation, ValueType,
        };

        // Reconstruct the real `format!("hi{}", ctx)` front-end shape:
        // block A builds the `Argument::new_display(&ctx)` through the
        // argument Tuple (`&(ctx,).0`); block B builds the args + pieces
        // arrays and `Arguments::new`. The args-array element is the
        // new_display result threaded across the A→B link, so extraction
        // must follow the cross-block inputarg back to it.
        let mut graph = FunctionGraph::new("fmt_chain");
        let a = graph.create_block();
        let b = graph.create_block();

        // ── block A: Argument::new_display(&ctx) via the arg Tuple ──
        let ctx = Variable::new();
        let tuple = Variable::new();
        graph.block_mut(a).operations.push(SpaceOperation {
            result: Some(tuple.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Tuple".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Tuple".to_string())),
            },
        });
        graph.block_mut(a).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: tuple.clone(),
                field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
                value: crate::model::LinkArg::Value(ctx.clone()),
                ty: ValueType::Ref(None),
            },
        });
        let arg_ref = Variable::new();
        graph.block_mut(a).operations.push(SpaceOperation {
            result: Some(arg_ref.clone()),
            kind: OpKind::FieldRead {
                base: tuple,
                field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        let argument = Variable::new();
        graph.block_mut(a).operations.push(SpaceOperation {
            result: Some(argument.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![
                        "fmt".to_string(),
                        "rt".to_string(),
                        "Argument".to_string(),
                        "new_display".to_string(),
                    ],
                },
                args: vec![arg_ref],
                result_ty: ValueType::Ref(Some("Argument".to_string())),
            },
        });

        // block B takes the new_display result as its single inputarg.
        let arg_in = Variable::new();
        graph.block_mut(b).inputargs = vec![arg_in.clone()];
        let link = Link::from_variables(&graph, vec![argument], b, None).with_prevblock(a);
        graph.block_mut(a).exits = vec![link];

        // ── block B: args array, pieces array, Arguments::new ──
        let args_arr = Variable::new();
        graph.block_mut(b).operations.push(SpaceOperation {
            result: Some(args_arr.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Array".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Array".to_string())),
            },
        });
        graph.block_mut(b).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: args_arr.clone(),
                field: FieldDescriptor::new("__pos_0", Some("Array".to_string())),
                value: crate::model::LinkArg::Value(arg_in),
                ty: ValueType::Ref(None),
            },
        });
        let pieces_arr = Variable::new();
        graph.block_mut(b).operations.push(SpaceOperation {
            result: Some(pieces_arr.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Array".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Array".to_string())),
            },
        });
        // `format!("hi{}")` packed template: [2, 'h', 'i', 0xC0, 0].
        for (i, byte) in [2i64, 104, 105, 0xC0, 0].iter().enumerate() {
            let v = Variable::new();
            graph.block_mut(b).operations.push(SpaceOperation {
                result: Some(v.clone()),
                kind: OpKind::ConstInt(*byte),
            });
            graph.block_mut(b).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: pieces_arr.clone(),
                    field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                    value: crate::model::LinkArg::Value(v),
                    ty: ValueType::Int,
                },
            });
        }
        let fmt_args = Variable::new();
        graph.block_mut(b).operations.push(SpaceOperation {
            result: Some(fmt_args.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![
                        "fmt".to_string(),
                        "Arguments".to_string(),
                        "new".to_string(),
                    ],
                },
                args: vec![pieces_arr.clone(), args_arr],
                result_ty: ValueType::Ref(Some("Arguments".to_string())),
            },
        });

        let chain = extract_fmt_chain(&graph, &fmt_args).expect("recognized fmt chain");
        assert_eq!(chain.pieces, vec!["hi".to_string(), String::new()]);
        assert_eq!(chain.args.len(), 1);
        assert_eq!(chain.args[0].kind, FmtArgKind::Display);
        // The recovered value is `ctx`, unwrapped through the arg Tuple.
        assert_eq!(chain.args[0].value.id(), ctx.id());

        // A non-`Arguments::new` producer is not recognized.
        assert!(extract_fmt_chain(&graph, &pieces_arr).is_none());
    }

    #[test]
    fn collapse_fmt_chains_expands_single_arg_chain_to_str_concat() {
        use super::collapse_fmt_chains;
        use crate::flowspace::model::Variable;
        use crate::model::{
            CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
            ValueType,
        };

        // Reconstruct the full `f(format!("hi{}", ctx))` shape charon
        // lowers across three blocks: B0 builds `new_display(&ctx)` off the
        // arg Tuple, Bp builds the args/pieces arrays + `Arguments::new`,
        // Bf calls `alloc::fmt::format`, then returns the String.
        let mut graph = FunctionGraph::new("fmt_collapse");
        let b0 = graph.create_block();
        let bp = graph.create_block();
        let bf = graph.create_block();
        let bret = graph.create_block();

        // ── B0: Argument::new_display(&ctx) via the arg Tuple ──
        let ctx = Variable::new();
        graph.block_mut(b0).inputargs = vec![ctx.clone()];
        let tuple = Variable::new();
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: Some(tuple.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Tuple".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Tuple".to_string())),
            },
        });
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: tuple.clone(),
                field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
                value: LinkArg::Value(ctx.clone()),
                ty: ValueType::Ref(None),
            },
        });
        let arg_ref = Variable::new();
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: Some(arg_ref.clone()),
            kind: OpKind::FieldRead {
                base: tuple,
                field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        let new_display = Variable::new();
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: Some(new_display.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: ["fmt", "rt", "Argument", "new_display"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                },
                args: vec![arg_ref],
                result_ty: ValueType::Ref(None),
            },
        });
        let arg_in = Variable::new();
        graph.block_mut(bp).inputargs = vec![arg_in.clone()];
        graph.block_mut(b0).exits =
            vec![Link::from_variables(&graph, vec![new_display], bp, None).with_prevblock(b0)];

        // ── Bp: args array, pieces array, Arguments::new ──
        let args_arr = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(args_arr.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Array".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Array".to_string())),
            },
        });
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: args_arr.clone(),
                field: FieldDescriptor::new("__pos_0", Some("Array".to_string())),
                value: LinkArg::Value(arg_in.clone()),
                ty: ValueType::Ref(None),
            },
        });
        let pieces_arr = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(pieces_arr.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Array".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Array".to_string())),
            },
        });
        for (i, byte) in [2i64, 104, 105, 0xC0, 0].iter().enumerate() {
            let v = Variable::new();
            graph.block_mut(bp).operations.push(SpaceOperation {
                result: Some(v.clone()),
                kind: OpKind::ConstInt(*byte),
            });
            graph.block_mut(bp).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: pieces_arr.clone(),
                    field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                    value: LinkArg::Value(v),
                    ty: ValueType::Int,
                },
            });
        }
        let fmt_args = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(fmt_args.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: ["fmt", "Arguments", "new"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                },
                args: vec![pieces_arr, args_arr],
                result_ty: ValueType::Ref(None),
            },
        });
        let fmt_args_in = Variable::new();
        graph.block_mut(bf).inputargs = vec![fmt_args_in.clone()];
        graph.block_mut(bp).exits =
            vec![Link::from_variables(&graph, vec![fmt_args], bf, None).with_prevblock(bp)];

        // ── Bf: alloc::fmt::format(args) → String ──
        let formatted = Variable::new();
        graph.block_mut(bf).operations.push(SpaceOperation {
            result: Some(formatted.clone()),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: ["alloc", "fmt", "format"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                },
                args: vec![fmt_args_in.clone()],
                result_ty: ValueType::Ref(None),
            },
        });
        let ret = Variable::new();
        graph.block_mut(bret).inputargs = vec![ret];
        graph.block_mut(bf).exits = vec![
            Link::from_variables(&graph, vec![formatted.clone()], bret, None).with_prevblock(bf),
        ];

        collapse_fmt_chains(&mut graph);

        // Bf's `alloc::fmt::format` is replaced by the orthodox
        // `"hi" + str(value)` expansion (pieces ["hi", ""]; the empty
        // trailing piece is skipped), reusing `formatted` on the final
        // `add` so the return link still forwards the String.
        let bf_block = graph.blocks.iter().find(|b| b.id == bf).unwrap();
        assert_eq!(bf_block.operations.len(), 3);
        match &bf_block.operations[0].kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                args,
                ..
            } => {
                assert_eq!(segments, &vec!["__str_const".to_string(), "hi".to_string()]);
                assert!(args.is_empty());
            }
            other => panic!("Bf op[0] not a __str_const: {other:?}"),
        }
        match &bf_block.operations[1].kind {
            OpKind::UnaryOp { op, operand, .. } => {
                assert_eq!(op, "str");
                assert_eq!(operand.id(), fmt_args_in.id());
            }
            other => panic!("Bf op[1] not a str UnaryOp: {other:?}"),
        }
        match &bf_block.operations[2].kind {
            OpKind::BinOp { op, lhs, rhs, .. } => {
                assert_eq!(op, "add");
                assert_eq!(
                    lhs.id(),
                    bf_block.operations[0].result.as_ref().unwrap().id()
                );
                assert_eq!(
                    rhs.id(),
                    bf_block.operations[1].result.as_ref().unwrap().id()
                );
            }
            other => panic!("Bf op[2] not an add BinOp: {other:?}"),
        }
        assert_eq!(
            bf_block.operations[2].result.as_ref().unwrap().id(),
            formatted.id()
        );

        // The chain ops are gone: B0 keeps no Tuple/new_display ops, Bp
        // keeps no array/Arguments ops.
        let b0_block = graph.blocks.iter().find(|b| b.id == b0).unwrap();
        assert!(b0_block.operations.is_empty(), "B0 chain ops not deleted");
        let bp_block = graph.blocks.iter().find(|b| b.id == bp).unwrap();
        assert!(bp_block.operations.is_empty(), "Bp chain ops not deleted");

        // The rendered value threads straight through: B0→Bp forwards
        // `ctx`, Bp→Bf forwards Bp's inputarg (now bound to `ctx`).
        let b0_exit = &b0_block.exits[0];
        assert_eq!(b0_exit.args[0].as_variable().unwrap().id(), ctx.id());
        let bp_exit = &bp_block.exits[0];
        assert_eq!(bp_exit.args[0].as_variable().unwrap().id(), arg_in.id());
    }

    #[test]
    fn collapse_fmt_chains_multi_expands_two_arg_chain() {
        use super::{collapse_fmt_chains_multi, fmt_path_ends_with, is_arguments_new_path};
        use crate::flowspace::model::Variable;
        use crate::model::{
            CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
            ValueType,
        };

        // Reconstruct the `f(format!("a{}b{}c", x, y))` shape charon lowers:
        // B0 builds the 2-field arg Tuple + `new_display(&x)`, B1 reads the
        // 2nd field off a *threaded copy* of the tuple ref + `new_display(&y)`
        // (exercising the cross-block `unwrap_fmt_arg_tuple_ref` match), Bp
        // builds the args/pieces arrays + `Arguments::new`, Bf calls
        // `alloc::fmt::format`.
        let mut graph = FunctionGraph::new("fmt_collapse_multi");
        let b0 = graph.create_block();
        let b1 = graph.create_block();
        let bp = graph.create_block();
        let bf = graph.create_block();
        let bret = graph.create_block();

        let fpath = |segs: &[&str]| CallTarget::FunctionPath {
            segments: segs.iter().map(|s| s.to_string()).collect(),
        };

        // ── B0: Tuple{x, y} + FieldRead __pos_0 + new_display(&x) ──
        let x = Variable::new();
        let y = Variable::new();
        graph.block_mut(b0).inputargs = vec![x.clone(), y.clone()];
        let tuple = Variable::new();
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: Some(tuple.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Tuple".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Tuple".to_string())),
            },
        });
        for (i, v) in [&x, &y].iter().enumerate() {
            graph.block_mut(b0).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: tuple.clone(),
                    field: FieldDescriptor::new(format!("__pos_{i}"), Some("Tuple".to_string())),
                    value: LinkArg::Value((*v).clone()),
                    ty: ValueType::Ref(None),
                },
            });
        }
        let ar0 = Variable::new();
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: Some(ar0.clone()),
            kind: OpKind::FieldRead {
                base: tuple.clone(),
                field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        let nd0 = Variable::new();
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: Some(nd0.clone()),
            kind: OpKind::Call {
                target: fpath(&["fmt", "rt", "Argument", "new_display"]),
                args: vec![ar0.clone()],
                result_ty: ValueType::Ref(None),
            },
        });
        // B0 → B1 forwards the tuple ref and the first new_display.
        let tuple_in = Variable::new();
        let nd0_in = Variable::new();
        graph.block_mut(b1).inputargs = vec![tuple_in.clone(), nd0_in.clone()];
        graph.block_mut(b0).exits = vec![
            Link::from_variables(&graph, vec![tuple.clone(), nd0], b1, None).with_prevblock(b0),
        ];

        // ── B1: FieldRead __pos_1 (off the threaded tuple copy) + new_display(&y) ──
        let ar1 = Variable::new();
        graph.block_mut(b1).operations.push(SpaceOperation {
            result: Some(ar1.clone()),
            kind: OpKind::FieldRead {
                base: tuple_in.clone(),
                field: FieldDescriptor::new("__pos_1", Some("Tuple".to_string())),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        let nd1 = Variable::new();
        graph.block_mut(b1).operations.push(SpaceOperation {
            result: Some(nd1.clone()),
            kind: OpKind::Call {
                target: fpath(&["fmt", "rt", "Argument", "new_display"]),
                args: vec![ar1.clone()],
                result_ty: ValueType::Ref(None),
            },
        });
        let nd0_p = Variable::new();
        let nd1_p = Variable::new();
        graph.block_mut(bp).inputargs = vec![nd0_p.clone(), nd1_p.clone()];
        graph.block_mut(b1).exits =
            vec![Link::from_variables(&graph, vec![nd0_in, nd1], bp, None).with_prevblock(b1)];

        // ── Bp: args array, pieces array, Arguments::new ──
        let args_arr = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(args_arr.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Array".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Array".to_string())),
            },
        });
        for (i, v) in [&nd0_p, &nd1_p].iter().enumerate() {
            graph.block_mut(bp).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: args_arr.clone(),
                    field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                    value: LinkArg::Value((*v).clone()),
                    ty: ValueType::Ref(None),
                },
            });
        }
        let pieces_arr = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(pieces_arr.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Array".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Array".to_string())),
            },
        });
        // Template for "a{}b{}c": 'a', ph, 'b', ph, 'c', term.
        for (i, byte) in [1i64, 97, 0xC0, 1, 98, 0xC0, 1, 99, 0].iter().enumerate() {
            let v = Variable::new();
            graph.block_mut(bp).operations.push(SpaceOperation {
                result: Some(v.clone()),
                kind: OpKind::ConstInt(*byte),
            });
            graph.block_mut(bp).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: pieces_arr.clone(),
                    field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                    value: LinkArg::Value(v),
                    ty: ValueType::Int,
                },
            });
        }
        let fmt_args = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(fmt_args.clone()),
            kind: OpKind::Call {
                target: fpath(&["fmt", "Arguments", "new"]),
                args: vec![pieces_arr, args_arr],
                result_ty: ValueType::Ref(None),
            },
        });
        let fmt_args_in = Variable::new();
        graph.block_mut(bf).inputargs = vec![fmt_args_in.clone()];
        graph.block_mut(bp).exits =
            vec![Link::from_variables(&graph, vec![fmt_args.clone()], bf, None).with_prevblock(bp)];

        // ── Bf: alloc::fmt::format(args) → String ──
        let formatted = Variable::new();
        graph.block_mut(bf).operations.push(SpaceOperation {
            result: Some(formatted.clone()),
            kind: OpKind::Call {
                target: fpath(&["alloc", "fmt", "format"]),
                args: vec![fmt_args_in.clone()],
                result_ty: ValueType::Ref(None),
            },
        });
        let ret = Variable::new();
        graph.block_mut(bret).inputargs = vec![ret];
        graph.block_mut(bf).exits = vec![
            Link::from_variables(&graph, vec![formatted.clone()], bret, None).with_prevblock(bf),
        ];

        collapse_fmt_chains_multi(&mut graph);

        // Both `new_display` ctors are rendered in place to `str`.
        let find_str = |bid| {
            graph
                .blocks
                .iter()
                .find(|b| b.id == bid)
                .unwrap()
                .operations
                .iter()
                .find_map(|op| match &op.kind {
                    OpKind::UnaryOp { op: o, operand, .. } if o == "str" => Some(operand.id()),
                    _ => None,
                })
        };
        // The argument-Tuple round-trip is eliminated: the local placeholder
        // reads its value `x` directly; the threaded placeholder reads the
        // rebound carrier (`tuple_in`, now forwarding `y`), not the Tuple
        // field.
        assert_eq!(find_str(b0), Some(x.id()), "B0 new_display→str(x)");
        assert_eq!(
            find_str(b1),
            Some(tuple_in.id()),
            "B1 new_display→str(rebound carrier)"
        );

        // No `fmt` externs survive anywhere in the graph.
        for b in &graph.blocks {
            for op in &b.operations {
                if let OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } = &op.kind
                {
                    let bad = fmt_path_ends_with(segments, &["Argument", "new_display"])
                        || is_arguments_new_path(segments)
                        || fmt_path_ends_with(segments, &["fmt", "format"]);
                    assert!(!bad, "residual fmt extern survived: {segments:?}");
                }
            }
        }

        // The Tuple ctor, its `__pos_N` writes, and the field read-backs are
        // all deleted — no StringRepr→PyObject-erased-field setattr survives.
        for b in &graph.blocks {
            for op in &b.operations {
                if let Some(r) = &op.result {
                    assert_ne!(r.id(), tuple.id(), "Tuple ctor survived");
                    assert_ne!(r.id(), ar0.id(), "__pos_0 FieldRead survived");
                    assert_ne!(r.id(), ar1.id(), "__pos_1 FieldRead survived");
                }
                if let OpKind::FieldWrite { base, .. } = &op.kind {
                    assert_ne!(base.id(), tuple.id(), "Tuple __pos_N FieldWrite survived");
                }
            }
        }
        // The B0→B1 link slot that forwarded the Tuple ref now forwards `y`.
        let b0_block = graph.blocks.iter().find(|b| b.id == b0).unwrap();
        assert_eq!(
            b0_block.exits[0].args[0].as_variable().map(|v| v.id()),
            Some(y.id()),
            "B0→B1 Tuple slot rebound to y"
        );

        // Bf's format op became `same_as(fmt_args_in)`, keeping `formatted`.
        let bf_block = graph.blocks.iter().find(|b| b.id == bf).unwrap();
        let same_as = bf_block
            .operations
            .iter()
            .find(|op| matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "same_as"))
            .expect("Bf same_as op");
        match &same_as.kind {
            OpKind::UnaryOp { operand, .. } => assert_eq!(operand.id(), fmt_args_in.id()),
            _ => unreachable!(),
        }
        assert_eq!(same_as.result.as_ref().unwrap().id(), formatted.id());

        // Bp now folds the rendered values with the pieces (str_const + add)
        // and forwards the folded String where it forwarded `Arguments`.
        let bp_block = graph.blocks.iter().find(|b| b.id == bp).unwrap();
        let add_count = bp_block
            .operations
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::BinOp { op, .. } if op == "add"))
            .count();
        assert!(
            add_count >= 2,
            "Bp must fold both rendered args, got {add_count} adds"
        );
        let bp_exit_val = bp_block.exits[0].args[0].as_variable().unwrap().id();
        assert_ne!(
            bp_exit_val,
            fmt_args.id(),
            "Bp must forward the folded String, not Arguments"
        );
    }

    #[test]
    fn collapse_fmt_chains_multi_eliminates_cross_block_local_tuple() {
        use super::{collapse_fmt_chains_multi, fmt_path_ends_with, is_arguments_new_path};
        use crate::flowspace::model::Variable;
        use crate::model::{
            CallTarget, FieldDescriptor, FunctionGraph, Link, LinkArg, OpKind, SpaceOperation,
            ValueType,
        };

        // The `index_type_error` shape: `format!` splits the two placeholder
        // field reads across a straight-line block boundary, but B1 reads the
        // 2nd field off the *same* Tuple ref id (referenced cross-block by
        // dominance), not a threaded inputarg copy.  Both placeholders are
        // Local; the cross-block reader is admitted because B0 dominates B1
        // and every Tuple write lives in B0.
        let mut graph = FunctionGraph::new("fmt_collapse_cross_block_local");
        let b0 = graph.create_block();
        let b1 = graph.create_block();
        let bp = graph.create_block();
        let bf = graph.create_block();
        let bret = graph.create_block();
        // B0 is the entry so `block_dominates(b0, b1)` holds (b1 reachable
        // only through b0), mirroring the real `index_type_error` CFG.
        graph.startblock = b0;

        let fpath = |segs: &[&str]| CallTarget::FunctionPath {
            segments: segs.iter().map(|s| s.to_string()).collect(),
        };

        // ── B0: Tuple{x, y} + both FieldWrites + FieldRead __pos_0 + new_display ──
        let x = Variable::new();
        let y = Variable::new();
        graph.block_mut(b0).inputargs = vec![x.clone(), y.clone()];
        let tuple = Variable::new();
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: Some(tuple.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Tuple".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Tuple".to_string())),
            },
        });
        for (i, v) in [&x, &y].iter().enumerate() {
            graph.block_mut(b0).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: tuple.clone(),
                    field: FieldDescriptor::new(format!("__pos_{i}"), Some("Tuple".to_string())),
                    value: LinkArg::Value((*v).clone()),
                    ty: ValueType::Ref(None),
                },
            });
        }
        let ar0 = Variable::new();
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: Some(ar0.clone()),
            kind: OpKind::FieldRead {
                base: tuple.clone(),
                field: FieldDescriptor::new("__pos_0", Some("Tuple".to_string())),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        let nd0 = Variable::new();
        graph.block_mut(b0).operations.push(SpaceOperation {
            result: Some(nd0.clone()),
            kind: OpKind::Call {
                target: fpath(&["fmt", "rt", "Argument", "new_display"]),
                args: vec![ar0.clone()],
                result_ty: ValueType::Ref(None),
            },
        });
        // B0 → B1 forwards only the first new_display; the Tuple ref is NOT a
        // link arg — B1 references it directly by id (dominance carry).
        let nd0_in = Variable::new();
        graph.block_mut(b1).inputargs = vec![nd0_in.clone()];
        graph.block_mut(b0).exits =
            vec![Link::from_variables(&graph, vec![nd0], b1, None).with_prevblock(b0)];

        // ── B1: FieldRead __pos_1 off the *same* tuple id + new_display(&y) ──
        let ar1 = Variable::new();
        graph.block_mut(b1).operations.push(SpaceOperation {
            result: Some(ar1.clone()),
            kind: OpKind::FieldRead {
                base: tuple.clone(),
                field: FieldDescriptor::new("__pos_1", Some("Tuple".to_string())),
                ty: ValueType::Ref(None),
                pure: false,
            },
        });
        let nd1 = Variable::new();
        graph.block_mut(b1).operations.push(SpaceOperation {
            result: Some(nd1.clone()),
            kind: OpKind::Call {
                target: fpath(&["fmt", "rt", "Argument", "new_display"]),
                args: vec![ar1.clone()],
                result_ty: ValueType::Ref(None),
            },
        });
        let nd0_p = Variable::new();
        let nd1_p = Variable::new();
        graph.block_mut(bp).inputargs = vec![nd0_p.clone(), nd1_p.clone()];
        graph.block_mut(b1).exits =
            vec![Link::from_variables(&graph, vec![nd0_in, nd1], bp, None).with_prevblock(b1)];

        // ── Bp: args array, pieces array, Arguments::new ──
        let args_arr = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(args_arr.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Array".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Array".to_string())),
            },
        });
        for (i, v) in [&nd0_p, &nd1_p].iter().enumerate() {
            graph.block_mut(bp).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: args_arr.clone(),
                    field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                    value: LinkArg::Value((*v).clone()),
                    ty: ValueType::Ref(None),
                },
            });
        }
        let pieces_arr = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(pieces_arr.clone()),
            kind: OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor {
                    name: "Array".to_string(),
                    owner_path: vec![],
                },
                args: vec![],
                result_ty: ValueType::Ref(Some("Array".to_string())),
            },
        });
        for (i, byte) in [1i64, 97, 0xC0, 1, 98, 0xC0, 1, 99, 0].iter().enumerate() {
            let v = Variable::new();
            graph.block_mut(bp).operations.push(SpaceOperation {
                result: Some(v.clone()),
                kind: OpKind::ConstInt(*byte),
            });
            graph.block_mut(bp).operations.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: pieces_arr.clone(),
                    field: FieldDescriptor::new(format!("__pos_{i}"), Some("Array".to_string())),
                    value: LinkArg::Value(v),
                    ty: ValueType::Int,
                },
            });
        }
        let fmt_args = Variable::new();
        graph.block_mut(bp).operations.push(SpaceOperation {
            result: Some(fmt_args.clone()),
            kind: OpKind::Call {
                target: fpath(&["fmt", "Arguments", "new"]),
                args: vec![pieces_arr, args_arr],
                result_ty: ValueType::Ref(None),
            },
        });
        let fmt_args_in = Variable::new();
        graph.block_mut(bf).inputargs = vec![fmt_args_in.clone()];
        graph.block_mut(bp).exits =
            vec![Link::from_variables(&graph, vec![fmt_args.clone()], bf, None).with_prevblock(bp)];

        // ── Bf: alloc::fmt::format(args) → String ──
        let formatted = Variable::new();
        graph.block_mut(bf).operations.push(SpaceOperation {
            result: Some(formatted.clone()),
            kind: OpKind::Call {
                target: fpath(&["alloc", "fmt", "format"]),
                args: vec![fmt_args_in.clone()],
                result_ty: ValueType::Ref(None),
            },
        });
        let ret = Variable::new();
        graph.block_mut(bret).inputargs = vec![ret];
        graph.block_mut(bf).exits = vec![
            Link::from_variables(&graph, vec![formatted.clone()], bret, None).with_prevblock(bf),
        ];

        collapse_fmt_chains_multi(&mut graph);

        let find_str = |bid| {
            graph
                .blocks
                .iter()
                .find(|b| b.id == bid)
                .unwrap()
                .operations
                .iter()
                .find_map(|op| match &op.kind {
                    OpKind::UnaryOp { op: o, operand, .. } if o == "str" => Some(operand.id()),
                    _ => None,
                })
        };
        // Both placeholders are Local and read their values directly — the
        // cross-block reader in B1 reads `y`, not a Tuple field.
        assert_eq!(find_str(b0), Some(x.id()), "B0 new_display→str(x)");
        assert_eq!(
            find_str(b1),
            Some(y.id()),
            "B1 cross-block new_display→str(y)"
        );

        // No `fmt` externs survive.
        for b in &graph.blocks {
            for op in &b.operations {
                if let OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } = &op.kind
                {
                    let bad = fmt_path_ends_with(segments, &["Argument", "new_display"])
                        || is_arguments_new_path(segments)
                        || fmt_path_ends_with(segments, &["fmt", "format"]);
                    assert!(!bad, "residual fmt extern survived: {segments:?}");
                }
            }
        }

        // Tuple ctor, both writes, and both field read-backs are gone.
        for b in &graph.blocks {
            for op in &b.operations {
                if let Some(r) = &op.result {
                    assert_ne!(r.id(), tuple.id(), "Tuple ctor survived");
                    assert_ne!(r.id(), ar0.id(), "__pos_0 FieldRead survived");
                    assert_ne!(r.id(), ar1.id(), "__pos_1 FieldRead survived");
                }
                if let OpKind::FieldWrite { base, .. } = &op.kind {
                    assert_ne!(base.id(), tuple.id(), "Tuple __pos_N FieldWrite survived");
                }
            }
        }
    }

    #[test]
    fn emit_fmt_concat_builds_interleaved_str_add_fold() {
        use super::{FmtArg, FmtArgKind, FmtChain, emit_fmt_concat};
        use crate::flowspace::model::Variable;
        use crate::model::{CallTarget, FunctionGraph, OpKind};

        // `format!("a{}b{}c", x, y)` → pieces ["a","b","c"], two args.
        let mut graph = FunctionGraph::new("concat");
        let bb = graph.create_block();
        let x = Variable::new();
        let y = Variable::new();
        let chain = FmtChain {
            pieces: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            args: vec![
                FmtArg {
                    value: x.clone(),
                    kind: FmtArgKind::Display,
                },
                FmtArg {
                    value: y.clone(),
                    kind: FmtArgKind::Display,
                },
            ],
        };
        let result = emit_fmt_concat(&mut graph, bb, &chain);

        let ops = &graph.blocks.iter().find(|b| b.id == bb).unwrap().operations;
        // 3 literal `__str_const`s + 4 `add`s = 7 ops.
        assert_eq!(ops.len(), 7);

        let str_const_text = |i: usize| -> String {
            match &ops[i].kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments.first().map(String::as_str) == Some("__str_const") => {
                    segments[1].clone()
                }
                other => panic!("op[{i}] not a __str_const: {other:?}"),
            }
        };
        let add_operands = |i: usize| -> (u64, u64) {
            match &ops[i].kind {
                OpKind::BinOp { op, lhs, rhs, .. } if op == "add" => (lhs.id(), rhs.id()),
                other => panic!("op[{i}] not an add: {other:?}"),
            }
        };
        let result_id = |i: usize| ops[i].result.as_ref().unwrap().id();

        assert_eq!(str_const_text(0), "a");
        assert_eq!(str_const_text(2), "b");
        assert_eq!(str_const_text(5), "c");
        // Fold chain: ("a" + x) + "b", then (+ y) + "c".
        assert_eq!(add_operands(1), (result_id(0), x.id()));
        assert_eq!(add_operands(3), (result_id(1), result_id(2)));
        assert_eq!(add_operands(4), (result_id(3), y.id()));
        assert_eq!(add_operands(6), (result_id(4), result_id(5)));
        assert_eq!(result.id(), result_id(6));
    }

    /// Anchor [`extract_fmt_chain`] to the real lowered IR of
    /// `stack_underflow_error` (= `type_error(format!("stack underflow
    /// during {context}"))`). Ignored by default (loads the 242MB real
    /// LLBC); run with `cargo test -p majit-translate --lib
    /// extract_fmt_chain_matches_real -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn extract_fmt_chain_matches_real_stack_underflow() {
        use super::{FmtArgKind, extract_fmt_chain};
        use crate::model::{CallTarget, OpKind};

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../build/llbc/pyre-interpreter.ullbc"
        );
        let llbc = Llbc::load(path).expect("load real LLBC");
        let graph = super::lower_function(&llbc, "stack_underflow_error")
            .expect("lower stack_underflow_error");

        // Find the `alloc::fmt::format(args)` call and extract its arg.
        let fmt_args = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    args,
                    ..
                } if super::fmt_path_ends_with(segments, &["fmt", "format"]) => {
                    args.first().cloned()
                }
                _ => None,
            })
            .expect("alloc::fmt::format call present");

        let chain = extract_fmt_chain(&graph, &fmt_args).expect("recognized real fmt chain");
        assert_eq!(
            chain.pieces,
            vec!["stack underflow during ".to_string(), String::new()]
        );
        assert_eq!(chain.args.len(), 1);
        assert_eq!(chain.args[0].kind, FmtArgKind::Display);
    }

    #[test]
    fn cast_pointer_marker_carries_root_in_path_and_result_type() {
        use crate::model::{CallTarget, OpKind, ValueType};
        let arg = crate::flowspace::model::Variable::new();
        let op = cast_pointer_marker_op("W_CastTarget".to_string(), arg.clone());
        let OpKind::Call {
            target,
            args,
            result_ty,
        } = op
        else {
            panic!("marker must be an OpKind::Call");
        };
        assert_eq!(
            target,
            CallTarget::FunctionPath {
                segments: vec!["__cast_pointer".to_string(), "W_CastTarget".to_string()],
            }
        );
        assert_eq!(args, vec![arg]);
        assert_eq!(result_ty, ValueType::Ref(Some("W_CastTarget".to_string())));
    }

    /// Anchor [`Lowering::fold_size_const_global`] to the real lowered
    /// IR of `function_new_impl` (= reads `FUNCTION_OBJECT_SIZE`, a
    /// `const usize = size_of::<Function>()`).  The global read must fold
    /// to `Function`'s concrete byte size (144) rather than residualize
    /// as an unregisterable `FunctionPath` accessor call.  Ignored by
    /// default (loads the 249MB real LLBC); run with `cargo test -p
    /// majit-translate --lib fold_size_const_real -- --ignored`.
    #[test]
    #[ignore]
    fn fold_size_const_real_function_object_size() {
        use crate::model::{CallTarget, OpKind};

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../build/llbc/pyre-interpreter.ullbc"
        );
        let llbc = Llbc::load(path).expect("load real LLBC");
        let graph =
            super::lower_function(&llbc, "function_new_impl").expect("lower function_new_impl");

        let ops: Vec<&OpKind> = graph
            .blocks
            .iter()
            .flat_map(|b| b.operations.iter())
            .map(|op| &op.kind)
            .collect();

        // The `size_of::<Function>()` const read folded to its byte size.
        assert!(
            ops.iter().any(|k| matches!(k, OpKind::ConstInt(144))),
            "expected a ConstInt(144) for the folded FUNCTION_OBJECT_SIZE"
        );
        // No residual accessor call to the const remains.
        let residual = ops.iter().any(|k| {
            matches!(
                k,
                OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                    if segments.last().is_some_and(|s| s.ends_with("FUNCTION_OBJECT_SIZE"))
            )
        });
        assert!(
            !residual,
            "FUNCTION_OBJECT_SIZE must not residualize as a FunctionPath call"
        );
    }

    /// Minimal `Llbc` carrying only `trait_impls` — the surface
    /// [`resolve_trait_assoc_type`] consults.
    fn llbc_with_trait_impls(trait_impls: serde_json::Value) -> Llbc {
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [],
                "fun_decls": [],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": trait_impls,
            }
        });
        Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc parses")
    }

    #[test]
    fn trait_assoc_type_resolves_via_unique_impl() {
        // `C::Name` with `impl Trait#1 for X { type Name = bool }` as
        // the LLBC's only impl of trait 1 renders the bound type.
        let llbc = llbc_with_trait_impls(serde_json::json!([
            null,
            {
                "impl_trait": { "id": 1 },
                "types": [{
                    "kind": { "TraitType": [1, 0] },
                    "skip_binder": { "value": { "Literal": "Bool" } }
                }]
            }
        ]));
        let projection = serde_json::json!({
            "TraitType": [
                { "trait_decl_ref": { "skip_binder": { "id": 1 } } },
                0
            ]
        });
        assert_eq!(
            charon_type_value_to_ast_string(&projection, &llbc, 0),
            "bool"
        );
    }

    #[test]
    fn trait_assoc_type_keeps_fallback_when_impl_ambiguous_or_missing() {
        // Two impls of trait 1 → instantiation-dependent → fallback.
        let two_impls = llbc_with_trait_impls(serde_json::json!([
            {
                "impl_trait": { "id": 1 },
                "types": [{
                    "kind": { "TraitType": [1, 0] },
                    "skip_binder": { "value": { "Literal": "Bool" } }
                }]
            },
            {
                "impl_trait": { "id": 1 },
                "types": [{
                    "kind": { "TraitType": [1, 0] },
                    "skip_binder": { "value": { "Literal": "Char" } }
                }]
            }
        ]));
        let projection = serde_json::json!({
            "TraitType": [
                { "trait_decl_ref": { "skip_binder": { "id": 1 } } },
                0
            ]
        });
        assert_eq!(
            charon_type_value_to_ast_string(&projection, &two_impls, 0),
            "??TraitType"
        );
        // No impl at all → fallback too.
        let no_impls = llbc_with_trait_impls(serde_json::json!([]));
        assert_eq!(
            charon_type_value_to_ast_string(&projection, &no_impls, 0),
            "??TraitType"
        );
    }

    #[test]
    fn cast_kind_raw_ptr_recognizes_atom_and_object_forms() {
        assert!(cast_kind_is_raw_ptr(&serde_json::json!("RawPtr")));
        assert!(cast_kind_is_raw_ptr(
            &serde_json::json!({"RawPtr": ["x", "y"]})
        ));
        assert!(!cast_kind_is_raw_ptr(&serde_json::json!("Unsize")));
        assert!(!cast_kind_is_raw_ptr(&serde_json::json!({"Scalar": []})));
    }

    fn rows(spec: &[(&str, &str)]) -> Vec<(String, String)> {
        spec.iter()
            .map(|(n, t)| (n.to_string(), t.to_string()))
            .collect()
    }

    #[test]
    fn harden_withdraws_shape_divergent_bare_alias_and_tombstones_origin() {
        let mut reg = crate::front::semantic::StructFieldRegistry::default();
        let a = rows(&[("handlerposition", "usize")]);
        let b = rows(&[
            ("valuestackdepth", "usize"),
            ("previous", "*mut FrameBlock"),
        ]);
        reg.fields.insert(
            "pyre_interpreter::pyopcode::FrameBlock".to_string(),
            a.clone(),
        );
        reg.fields
            .insert("pyre_interpreter::pyframe::FrameBlock".to_string(), b);
        // last-decl-wins bare alias as the dual-publish would leave it
        reg.fields.insert("FrameBlock".to_string(), a);
        let mut origins = std::collections::HashMap::new();
        // first-decl-wins origin as `or_insert` would leave it
        origins.insert("FrameBlock".to_string(), "pyopcode".to_string());
        let mut enums = std::collections::HashMap::new();

        harden_duplicate_leaf_metadata(&mut reg, &mut origins, &mut enums);

        assert!(
            !reg.fields.contains_key("FrameBlock"),
            "shape-divergent duplicate leaf must lose its bare alias"
        );
        assert!(
            reg.fields
                .contains_key("pyre_interpreter::pyopcode::FrameBlock")
        );
        assert!(
            reg.fields
                .contains_key("pyre_interpreter::pyframe::FrameBlock")
        );
        assert_eq!(
            origins.get("FrameBlock").map(String::as_str),
            Some(""),
            "module-divergent duplicate leaf origin must be tombstoned"
        );
    }

    #[test]
    fn harden_keeps_alias_for_equal_shape_same_module_duplicates() {
        let mut reg = crate::front::semantic::StructFieldRegistry::default();
        let shape = rows(&[("x", "i64")]);
        reg.fields
            .insert("pyre_object::eval::Point".to_string(), shape.clone());
        reg.fields
            .insert("pyre_jit::eval::Point".to_string(), shape.clone());
        reg.fields.insert("Point".to_string(), shape.clone());
        let mut origins = std::collections::HashMap::new();
        origins.insert("Point".to_string(), "eval".to_string());
        let mut enums = std::collections::HashMap::new();

        harden_duplicate_leaf_metadata(&mut reg, &mut origins, &mut enums);

        assert_eq!(reg.fields.get("Point"), Some(&shape));
        assert_eq!(origins.get("Point").map(String::as_str), Some("eval"));
    }

    #[test]
    fn harden_withdraws_discriminant_divergent_bare_enum_alias() {
        let mut reg = crate::front::semantic::StructFieldRegistry::default();
        let mut origins = std::collections::HashMap::new();
        let map_a: std::collections::HashMap<i64, String> =
            [(0, "Continue".to_string()), (1, "Break".to_string())].into();
        let map_b: std::collections::HashMap<i64, String> = [(0, "Return".to_string())].into();
        let same_as_a = map_a.clone();
        let mut enums = std::collections::HashMap::new();
        enums.insert("pyre_interpreter::eval::StepResult".to_string(), map_a);
        enums.insert("pyre_jit::eval::StepResult".to_string(), map_b);
        // silent-winner bare alias as the dual-publish would leave it
        enums.insert("StepResult".to_string(), same_as_a.clone());
        enums.insert("pyre_object::flow::Verdict".to_string(), same_as_a.clone());
        enums.insert("pyre_jit::flow::Verdict".to_string(), same_as_a.clone());
        enums.insert("Verdict".to_string(), same_as_a.clone());
        // The `__discriminant`-only base rows as the enum arm registers
        // them — shape-identical across enums, so only the discriminant
        // divergence above can adjudicate the bare alias.
        let disc = rows(&[("__discriminant", "i64")]);
        reg.fields.insert("StepResult".to_string(), disc.clone());
        reg.fields.insert("Verdict".to_string(), disc.clone());

        harden_duplicate_leaf_metadata(&mut reg, &mut origins, &mut enums);

        assert!(
            !enums.contains_key("StepResult"),
            "discriminant-divergent duplicate leaf must lose its bare alias"
        );
        assert!(enums.contains_key("pyre_interpreter::eval::StepResult"));
        assert!(enums.contains_key("pyre_jit::eval::StepResult"));
        assert_eq!(
            enums.get("Verdict"),
            Some(&same_as_a),
            "equal-map duplicates keep the alias"
        );
        // The bare enum-base row follows the discriminant signal: dropped
        // for the divergent leaf, kept for the equal-map one.
        assert!(
            !reg.fields.contains_key("StepResult"),
            "divergent enum base must lose its bare alias so it is not pre-minted as one merged class"
        );
        assert_eq!(
            reg.fields.get("Verdict"),
            Some(&disc),
            "equal-map enum base keeps its bare alias"
        );
    }

    #[test]
    fn harden_leaves_unique_leaves_untouched() {
        let mut reg = crate::front::semantic::StructFieldRegistry::default();
        let shape = rows(&[("ob_value", "i64")]);
        reg.fields.insert(
            "pyre_object::intobject::W_IntObject".to_string(),
            shape.clone(),
        );
        reg.fields.insert("W_IntObject".to_string(), shape.clone());
        let mut origins = std::collections::HashMap::new();
        origins.insert("W_IntObject".to_string(), "intobject".to_string());
        let mut enums = std::collections::HashMap::new();

        harden_duplicate_leaf_metadata(&mut reg, &mut origins, &mut enums);

        assert_eq!(reg.fields.get("W_IntObject"), Some(&shape));
        assert_eq!(
            origins.get("W_IntObject").map(String::as_str),
            Some("intobject")
        );
    }

    #[test]
    fn harden_withdraws_shape_divergent_variant_leaf_alias() {
        let mut reg = crate::front::semantic::StructFieldRegistry::default();
        let shape_a = rows(&[("__pos_0", "i64")]);
        let shape_b = rows(&[("__pos_0", "*mut Foo"), ("__pos_1", "u32")]);
        // Two `Outcome` enums in different modules whose `Ok` variant rows
        // diverge — the bare `Outcome::Ok` alias is last-decl-wins (shape_b),
        // as the variant dual-publish leaves it.
        reg.fields
            .insert("crateX::moduleA::Outcome::Ok".to_string(), shape_a.clone());
        reg.fields
            .insert("moduleA::Outcome::Ok".to_string(), shape_a.clone());
        reg.fields
            .insert("crateY::moduleB::Outcome::Ok".to_string(), shape_b.clone());
        reg.fields
            .insert("moduleB::Outcome::Ok".to_string(), shape_b.clone());
        reg.fields.insert("Outcome::Ok".to_string(), shape_b);
        // A shape-identical `Status::Active` duplicate keeps its bare alias.
        let shape_s = rows(&[("__pos_0", "u8")]);
        reg.fields
            .insert("moduleC::Status::Active".to_string(), shape_s.clone());
        reg.fields
            .insert("moduleD::Status::Active".to_string(), shape_s.clone());
        reg.fields
            .insert("Status::Active".to_string(), shape_s.clone());

        let mut origins = std::collections::HashMap::new();
        let mut enums = std::collections::HashMap::new();

        harden_duplicate_leaf_metadata(&mut reg, &mut origins, &mut enums);

        assert!(
            !reg.fields.contains_key("Outcome::Ok"),
            "shape-divergent variant-leaf alias must be withdrawn (bare last-segment pass missed it)"
        );
        assert!(reg.fields.contains_key("moduleA::Outcome::Ok"));
        assert!(reg.fields.contains_key("moduleB::Outcome::Ok"));
        assert_eq!(
            reg.fields.get("Status::Active"),
            Some(&shape_s),
            "shape-identical variant duplicate keeps its bare alias"
        );
    }

    #[test]
    fn liveness_marks_loop_carried_index_local_as_live_in() {
        use majit_charon_reader::ullbc::Unstructured;
        // A self-loop whose only read of the loop-carried local `_2` is
        // as an `Index` offset (`_3 = _1[_2]`); `_2` is redefined on the
        // back edge from `_3`, so it is never read as a plain operand.
        // The index offset lives in the projection element, not in the
        // projected place's `inner`, so a liveness pass that descends
        // only into `inner` misses it: `_2` is not marked live-in at the
        // loop block, the monotonic lowering mints no block inputarg for
        // it, and the loop-header read fails loud as an uninitialised
        // local (#176).  `mark_projection_index_offset_use` closes the
        // gap — assert `_2` is live-in at bb1.
        let span = || {
            serde_json::json!({
                "data": {
                    "file_id": 0,
                    "beg": {"line": 0, "col": 0},
                    "end": {"line": 0, "col": 0}
                },
                "generated_from_span": null
            })
        };
        let ty = || serde_json::json!({"Deduplicated": 0});
        let place_local = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
        let copy_local = |i: u64| serde_json::json!({"Copy": place_local(i)});
        let local =
            |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
        let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
        // bb0:  _2 = const 0;  goto bb1
        let bb0 = serde_json::json!({
            "statements": [stmt(serde_json::json!({
                "Assign": [place_local(2), {"Use": {"Const": null}}]
            }))],
            "terminator": {"kind": {"Goto": {"target": 1}}}
        });
        // bb1:  _3 = _1[_2];  _2 = _3;  goto bb1   (self-loop)
        let read_index = stmt(serde_json::json!({
            "Assign": [
                place_local(3),
                {"Use": {"Copy": {
                    "kind": {"Projection": [
                        place_local(1),
                        {"Index": {"offset": copy_local(2), "from_end": false}}
                    ]},
                    "ty": ty()
                }}}
            ]
        }));
        let carry = stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": copy_local(3)}]
        }));
        let bb1 = serde_json::json!({
            "statements": [read_index, carry],
            "terminator": {"kind": {"Goto": {"target": 1}}}
        });
        let body_json = serde_json::json!({
            "span": span(),
            "locals": {
                "arg_count": 1,
                "locals": [local(0), local(1), local(2), local(3)]
            },
            "body": [bb0, bb1]
        });
        let body: Unstructured =
            serde_json::from_value(body_json).expect("fixture Unstructured parses");
        // Base dataflow only: this case is closed by
        // `mark_projection_index_offset_use` in the statement scan, not by
        // the index-write `extra_live` set, so pass an empty extra_live.
        let live = super::compute_mir_liveness(&body, &[]);
        assert!(
            live[1][2],
            "loop-carried index local _2 must be live-in at the loop block bb1"
        );
        // Sanity: the array base _1 (read as the projection inner) is
        // also live-in, and the throwaway temp _3 (defined before its
        // only use within bb1) is not.
        assert!(live[1][1], "array base _1 must be live-in at bb1");
        assert!(!live[1][3], "temp _3 is block-local, not live-in at bb1");
    }

    #[test]
    fn add_dest_single_deref_guard_classifies_uses() {
        use majit_charon_reader::ullbc::Unstructured;
        // brick 3's escape guard: the `.add`-result local `_1` may be
        // dereferenced once and never escape as a raw pointer.  Each
        // fixture defines `_1` with one bare-local write (standing in for
        // the `add` call, which the guard counts the same as any other
        // bare-local definition) and varies how `_1` is then used.
        let span = || {
            serde_json::json!({
                "data": {"file_id": 0, "beg": {"line": 0, "col": 0}, "end": {"line": 0, "col": 0}},
                "generated_from_span": null
            })
        };
        let ty = || serde_json::json!({"Deduplicated": 0});
        let place_local = |i: u64| serde_json::json!({"kind": {"Local": i}, "ty": ty()});
        let deref_place = |i: u64| {
            serde_json::json!({
                "kind": {"Projection": [place_local(i), "Deref"]}, "ty": ty()
            })
        };
        let local =
            |i: u64| serde_json::json!({"index": i, "name": null, "span": span(), "ty": ty()});
        let stmt = |kind: serde_json::Value| serde_json::json!({"kind": kind, "comments_before": [], "span": span()});
        // A single-block body: `_1 = const` (the def) followed by `extra`,
        // returning.  `dest` = `_1`.
        let body_of = |extra: Vec<serde_json::Value>| -> Unstructured {
            let mut statements = vec![stmt(
                serde_json::json!({"Assign": [place_local(1), {"Use": {"Const": null}}]}),
            )];
            statements.extend(extra);
            let bb = serde_json::json!({
                "statements": statements,
                "terminator": {"kind": "Return"}
            });
            let body_json = serde_json::json!({
                "span": span(),
                "locals": {"arg_count": 0, "locals": [local(0), local(1), local(2)]},
                "body": [bb]
            });
            serde_json::from_value(body_json).expect("fixture Unstructured parses")
        };

        // `_2 = *_1` — one deref read: accepted.
        let read = body_of(vec![stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": {"Copy": deref_place(1)}}]
        }))]);
        assert!(super::add_dest_used_only_as_single_deref(&read, 1));

        // `*_1 = _2` — one deref write: accepted.
        let write = body_of(vec![stmt(serde_json::json!({
            "Assign": [deref_place(1), {"Use": {"Copy": place_local(2)}}]
        }))]);
        assert!(super::add_dest_used_only_as_single_deref(&write, 1));

        // `_2 = _1` — the pointer escapes by value: rejected.
        let escape = body_of(vec![stmt(serde_json::json!({
            "Assign": [place_local(2), {"Use": {"Copy": place_local(1)}}]
        }))]);
        assert!(!super::add_dest_used_only_as_single_deref(&escape, 1));

        // `_2 = *_1; _3 = *_1` — two derefs: rejected.
        let twice = body_of(vec![
            stmt(
                serde_json::json!({"Assign": [place_local(2), {"Use": {"Copy": deref_place(1)}}]}),
            ),
            stmt(
                serde_json::json!({"Assign": [place_local(2), {"Use": {"Copy": deref_place(1)}}]}),
            ),
        ]);
        assert!(!super::add_dest_used_only_as_single_deref(&twice, 1));

        // No use at all (just the def): rejected (no element load/store).
        let dead = body_of(vec![]);
        assert!(!super::add_dest_used_only_as_single_deref(&dead, 1));
    }
}
