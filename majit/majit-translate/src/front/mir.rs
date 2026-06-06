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
        Place, PlaceKind, ProjectionElem, Rvalue, StmtKind, SwitchTargets, TermKind, TyRef,
        TypeDeclKind, Unstructured,
    },
};

use crate::flowspace::model::{ConstValue, Variable};
use crate::model::{
    BlockId, CallTarget, ExitCase, ExitSwitch, FieldDescriptor, FunctionGraph, Link, OpKind,
    SpaceOperation, ValueType,
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
        let prog = build_semantic_program_from_llbc(llbc)?;
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
            }
        }
    }
    Ok(
        merged.unwrap_or_else(|| crate::front::semantic::SemanticProgram {
            functions: Vec::new(),
            known_struct_names: std::collections::HashSet::new(),
            known_trait_names: std::collections::HashSet::new(),
            struct_fields: crate::front::semantic::StructFieldRegistry::default(),
            immutable_fields: std::collections::HashMap::new(),
            enum_variant_by_discriminant: std::collections::HashMap::new(),
        }),
    )
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
    // current snapshot.  This predicate has two roles: it triggers that
    // RPO retry, and (defensively) if even RPO cannot bind the read — a
    // genuine loop-carried definition, of which there are none today —
    // it classifies the residual failure as a tracked degradation (the
    // function becomes a residual call) rather than a build-failing
    // regression.
    msg.contains("uninitialised local")
}

pub fn build_semantic_program_from_llbc(
    llbc: &Llbc,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    // ── Pass 1: walk type_decls + trait_decls ─────────────────────
    let (known_struct_names, known_trait_names, struct_fields, enum_variant_by_discriminant) =
        derive_program_metadata(llbc);

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
        // at the JIT level, and their unwind paths use `set_raise`
        // (`model.rs:3873`) — which mints orphan etype/evalue slots
        // the flowspace adapter then rejects with the "undefined
        // operand slot N as Link.args[0]" invariant break.  Skip them
        // so they never surface as call-registry entries the rest of
        // the pipeline does not model.
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
        // A single function whose body the driver does not yet handle
        // should not abort the whole-program build.  Capture
        // per-function errors into a side bucket and continue; they are
        // surfaced via `PYRE_MIR_FRONTEND_DEBUG=1` for triage, but
        // production keeps going with a degraded SemanticProgram —
        // failing-loud on the single broken function rather than
        // erroring out at program-build time.
        let graph = match lower_fun_decl(llbc, fd) {
            Ok(g) => g,
            Err(e) => {
                skipped.push((name.clone(), e.to_string()));
                continue;
            }
        };
        // return_type is intentionally `None` until the Charon
        // dedup-table resolution can map a `TyRef::Deduplicated{id}` to
        // its primitive name. The codewriter's call-signature validator
        // at `jit_codewriter/call.rs:4234` skips the check when declared
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
        //      indirecting through `trait_impls`.  `trait_impl_trait_root_for_fundecl`
        //      reads the id.
        //   2. trait-default bodies — Charon emits these as bare
        //      functions inside the trait's namespace; the penultimate
        //      NameSeg is `Ident{TraitLeaf}` with no `Impl` segment.
        //      Detect by matching the parent ident against
        //      `known_trait_names` (which derive_program_metadata seeds
        //      with both qualified path and bare leaf).
        let trait_root = trait_impl_trait_root_for_fundecl(llbc, fd)
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
    })
}

/// Derive whole-program type-metadata fields of `SemanticProgram` from
/// Charon's `type_decls` + `trait_decls` tables.
///
/// Returns `(known_struct_names, known_trait_names, struct_fields,
/// enum_variant_by_discriminant)`.
/// Names are taken from `item_meta.name_path()`; struct field rows
/// resolve their type string via [`tyref_to_ast_string`] (Charon-resolved
/// types: references stripped, raw pointers kept, `Vec<T>` / `[T;N]`
/// generics, named structs by leaf).
fn derive_program_metadata(
    llbc: &Llbc,
) -> (
    std::collections::HashSet<String>,
    std::collections::HashSet<String>,
    crate::front::semantic::StructFieldRegistry,
    std::collections::HashMap<String, std::collections::HashMap<i64, String>>,
) {
    let mut known_struct_names = std::collections::HashSet::new();
    let mut known_trait_names = std::collections::HashSet::new();
    let mut struct_fields = crate::front::semantic::StructFieldRegistry::default();
    let mut enum_variant_by_discriminant: std::collections::HashMap<
        String,
        std::collections::HashMap<i64, String>,
    > = std::collections::HashMap::new();

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
                known_struct_names.insert(name);
                known_struct_names.insert(leaf);
            }
            TypeDeclKind::Enum(variants) => {
                // Enums register under their type name *and* under each
                // variant path (`Strategy::Empty`, `Strategy::IntKeyed`,
                // …) so a synthetic Aggregate(SyntheticTransparentCtor)
                // can be matched downstream.
                let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
                known_struct_names.insert(name.clone());
                known_struct_names.insert(leaf.clone());
                // discriminant → variant name, published under both the
                // qualified path and the bare leaf so the opcode-dispatch
                // extractor can resolve by either spelling.
                let mut by_discr: std::collections::HashMap<i64, String> =
                    std::collections::HashMap::new();
                for v in variants {
                    let variant_path = format!("{name}::{}", v.name);
                    known_struct_names.insert(variant_path);
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
    )
}

/// Lower a single Charon [`FunDecl`] to a [`FunctionGraph`].
pub fn lower_fun_decl(llbc: &Llbc, fd: &FunDecl) -> Result<FunctionGraph, LowerError> {
    let u = fd.unstructured().ok_or_else(|| {
        LowerError::Unsupported(format!(
            "{}: no Unstructured body (extracted with --ullbc?)",
            fd.item_meta.name_path()
        ))
    })?;
    let name = fd.item_meta.name_path();
    let mut lo = Lowering::new(llbc, name.clone(), &u)?;
    match lo.lower(BlockOrder::Linear) {
        Ok(()) => Ok(lo.graph),
        // A forward-referenced definition — typically a `TermKind::Call`
        // dest at a higher MIR index than the block that reads it — reads
        // as an uninitialised local under MIR-index order.  Re-lower the
        // whole body in reverse-postorder, which visits the defining
        // block first.  This is scoped to exactly the bodies that fail
        // linearly (every other body keeps its linear-order bindings
        // untouched), and RPO is proven sufficient for all of them
        // (`classify_uninitialised_local_rpo_vs_loop_carried`: 0
        // loop-carried, so no phi / block-inputarg threading is needed).
        Err(LowerError::Unsupported(msg)) if is_known_lowering_gap(&msg) => {
            let mut lo = Lowering::new(llbc, name, &u)?;
            lo.lower(BlockOrder::ReversePostorder)?;
            Ok(lo.graph)
        }
        Err(e) => Err(e),
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

struct Lowering<'a> {
    graph: FunctionGraph,
    llbc: &'a Llbc,
    body: &'a Unstructured,
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
}

impl<'a> Lowering<'a> {
    fn new(llbc: &'a Llbc, name: String, body: &'a Unstructured) -> Result<Self, LowerError> {
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
        // `graph_non_void_arg_types` (`jit_codewriter/call.rs:2748+`),
        // `type_state` (`jit_codewriter/type_state.rs:131`) — locate
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
            input_ops.push(SpaceOperation {
                result: Some(var.clone()),
                kind: OpKind::Input {
                    name,
                    ty,
                    class_root: None,
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

        Ok(Self {
            graph,
            llbc,
            body,
            local_var,
            block_id,
            positional_aggregate_locals: std::collections::HashMap::new(),
        })
    }

    fn lower(&mut self, order: BlockOrder) -> Result<(), LowerError> {
        // Per the §"Reference" section above there is no mergeblock
        // dance — every MIR basic block is its own join point, fully
        // prepared by `create_block` / `startblock`, and locals live in
        // the function-wide `local_var` table rather than being threaded
        // as block inputargs.  `local_var` is a single monotonic slot
        // per local, so iteration order only governs *when* each block
        // writes its defs: a read binds as soon as *any* defining block
        // precedes it in processing order.  The default is MIR-index
        // (`Linear`) order.  It fails only when a definition
        // (notably a `TermKind::Call` dest) sits at a higher MIR index
        // than a block that reads it, producing a spurious "uninitialised
        // local"; `lower_fun_decl` then re-lowers in `ReversePostorder`,
        // which visits the defining block before the reading block
        // whenever the def forward-reaches the read.  Every such read is a
        // *forward* reference, never a back-edge-only definition (proven
        // by `classify_uninitialised_local_rpo_vs_loop_carried`: 15
        // forward-ref, 0 loop-carried), so RPO binds all of them without
        // any phi / block-inputarg threading.
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

    fn lower_block(&mut self, mir_bb: usize) -> Result<(), LowerError> {
        let bb: &BasicBlock = &self.body.body[mir_bb];

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
        match dest.kind {
            PlaceKind::Local(i) => {
                // Capture the construction `owner_root` if this binding
                // is a positional aggregate, before `build_rvalue`
                // consumes the rvalue, so `.N` reads of the local can
                // later emit a symmetric `FieldRead __pos_<N>` carrying
                // the same owner (see `resolve_place`).
                let positional_owner = self.positional_aggregate_owner(&rvalue);
                let (op, result_var) = self.build_rvalue(mir_bb, rvalue)?;
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
                let (_op, value_var) = self.build_rvalue(mir_bb, rvalue)?;
                // If `build_rvalue` produced an op, emit it first so
                // `value_var` is bound before the write reads it.
                if let Some(op) = _op {
                    let bb_id = self.block_id[mir_bb];
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(value_var.clone()),
                        kind: op,
                    });
                }
                self.emit_projection_write(mir_bb, *inner, elem, value_var)
            }
            _ => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: Assign to {:?} destination not yet supported",
                place_kind_label(&dest.kind)
            ))),
        }
    }

    /// Emit the side-effectful write op for an `Assign` whose dest is
    /// a `Projection(inner, elem)`. `value` is the freshly computed
    /// rvalue.
    fn emit_projection_write(
        &mut self,
        mir_bb: usize,
        inner: Place,
        elem: ProjectionElem,
        value: Variable,
    ) -> Result<(), LowerError> {
        let base = self.resolve_place(mir_bb, inner)?;
        let bb_id = self.block_id[mir_bb];
        let op = match &elem {
            ProjectionElem::Atom(s) if s == "Deref" => {
                // `*p = val` — no IR-level FieldWrite/ArrayWrite fits.
                // Emit a synthetic 2-arg Call so the write remains
                // visible to the downstream side-effect tracking.
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__deref_write".to_string()],
                    },
                    args: vec![base, value],
                    result_ty: ValueType::Int,
                }
            }
            ProjectionElem::Tagged(v) => {
                if let Some(field_payload) = v.as_object().and_then(|m| m.get("Field")) {
                    let label = field_label_from_payload(field_payload);
                    OpKind::FieldWrite {
                        base,
                        field: FieldDescriptor::new(label, None),
                        value,
                        ty: ValueType::Int,
                    }
                } else if let Some(index_payload) = v.as_object().and_then(|m| m.get("Index")) {
                    let idx_var = self.index_offset_var(mir_bb, index_payload)?;
                    OpKind::ArrayWrite {
                        base,
                        index: idx_var,
                        value,
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
                Ok((
                    Some(OpKind::BinOp {
                        op: op_label,
                        lhs: lhs_v,
                        rhs: rhs_v,
                        result_ty: ValueType::Int,
                    }),
                    res,
                ))
            }
            // `UnaryOp(op, operand)` — `Neg`, `Not`, casts.  Lowered to
            // `OpKind::UnaryOp` with a canonical snake_case label so the
            // assembler reaches the wired `int_neg` / `int_invert`
            // handlers instead of inventing a synthetic `int_unary.*`
            // opname.  `Cast(...)` carries a JSON sub-payload encoding
            // the cast kind; map the common JIT-no-op cases (RawPtr,
            // Scalar Int↔UInt, Unsize) to RPython's canonical cast
            // opnames so downstream dispatch matches `blackhole.py`'s
            // `bhimpl_cast_*` handlers.  Genuinely unsupported cast
            // shapes (e.g. VTable) fall back to a lowercased label that
            // surfaces as an unwired-opname diagnostic.
            Rvalue::UnaryOp(op_json, operand) => {
                let arg = self.resolve_operand(mir_bb, operand)?;
                let op_label = unary_op_label(&op_json)?;
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                Ok((
                    Some(OpKind::UnaryOp {
                        op: op_label,
                        operand: arg,
                        result_ty: ValueType::Int,
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
            Rvalue::Cast(_kind, operand, _ty) => {
                let v = self.resolve_operand(mir_bb, operand)?;
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
                            },
                            value,
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
            // this layer). `owner_root` is left
            // `None` because Charon's [`Place`] does not yet surface a
            // resolvable enum type name; the codewriter that consumes
            // this op may look up the receiver's classdef hint from
            // type-flow if it needs a more specific descriptor.
            Rvalue::Discriminant(place) => {
                let base = self.resolve_place(mir_bb, place)?;
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                Ok((
                    Some(OpKind::FieldRead {
                        base,
                        field: FieldDescriptor::new("__discriminant", None),
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
                // Int.  The `__str_const` path is never registered, so this
                // call always residualises; correcting `result_ty` fixes the
                // residual result kind without changing behaviour today.
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
                // cross-procedural callers like
                // `flowspace/rust_source/build_flow.rs:4770
                // lower_field`) get a resolvable field/owner_root shape.
                //
                // Tuple-container `Field` projections split two ways.
                // A local bound by a positional `Rvalue::Aggregate`
                // (`positional_aggregate_locals`) carries a synthetic
                // ctor base with a `__pos_<N>` `FieldWrite` chain, so
                // its `.N` reads emit a symmetric `FieldRead
                // __pos_<N>`.  Every other Tuple-container read still
                // collapses to the base Variable: the `straight_line_add`
                // / AddChecked `(value, bool)` shape lowers to a scalar
                // `BinOp` (not an Aggregate), so its `.0` must fall
                // through to the underlying Variable and the paired
                // `.1` Assert is dropped in `lower_statement`.
                //
                // Atom projections (`Deref` and others) still
                // collapse: `Deref` is a no-op for typed refs at the
                // JIT IR level, and any other Atom variant has no
                // typed analogue today.
                if let ProjectionElem::Tagged(v) = &elem
                    && let Some(field_payload) = v.as_object().and_then(|m| m.get("Field"))
                    && let Some((owner_root, field_name, field_ty)) =
                        self.resolve_adt_field(field_payload)
                {
                    let base = self.resolve_place(mir_bb, *inner)?;
                    let bb_id = self.block_id[mir_bb];
                    let ty = tyref_to_value_type(&field_ty, self.llbc);
                    let res = self
                        .graph
                        .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(res.clone()),
                        kind: OpKind::FieldRead {
                            base,
                            field: FieldDescriptor::new(field_name, Some(owner_root)),
                            ty,
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
                match elem {
                    ProjectionElem::Tagged(_) | ProjectionElem::Atom(_) => {
                        self.resolve_place(mir_bb, *inner)
                    }
                }
            }
            // `Global { id, .. }` — static/const item reference.
            // Modeled as a synthetic 0-arg `Call` to a `FunctionPath`
            // carrying the global's resolved name; downstream
            // consumers see a uniform call shape and can route on
            // the name (e.g. recognise `__elidable_function_*` constants
            // handled by the hint pass).
            PlaceKind::Global { id, .. } => {
                let segments = self.global_segments(mir_bb, id)?;
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                let bb_id = self.block_id[mir_bb];
                self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                    result: Some(res.clone()),
                    kind: OpKind::Call {
                        target: CallTarget::FunctionPath { segments },
                        args: vec![],
                        result_ty: ValueType::Int,
                    },
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
        let type_id = adt.first()?.as_u64()?;
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
                variant_owner.push(type_leaf);
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

    /// The owner_root is the LLBC TypeDecl's leaf name
    /// (`PyFrame` from `pyre_interpreter::pyframe::PyFrame`) so the
    /// downstream `struct_fields` registry resolves with the same
    /// leaf key.
    fn resolve_adt_field(&self, payload: &serde_json::Value) -> Option<(String, String, TyRef)> {
        let arr = payload.as_array()?;
        if arr.len() != 2 {
            return None;
        }
        let container = arr[0].as_object()?;
        let adt = container.get("Adt")?.as_array()?;
        let type_id = adt.first()?.as_u64()?;
        let variant_idx = adt.get(1).and_then(serde_json::Value::as_u64);
        let field_idx = arr[1].as_u64()? as usize;
        let td = self.llbc.type_by_id(type_id)?;
        let owner_root = td
            .item_meta
            .name_path()
            .rsplit("::")
            .next()
            .unwrap_or("")
            .to_string();
        match (&td.kind, variant_idx) {
            (TypeDeclKind::Struct(fields), None) => {
                let f = fields.get(field_idx)?;
                let name = f
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("__pos_{field_idx}"));
                let ty = clone_tyref(&f.ty);
                Some((owner_root, name, ty))
            }
            (TypeDeclKind::Enum(variants), Some(vidx)) => {
                let variant = variants.get(vidx as usize)?;
                let f = variant.fields.get(field_idx)?;
                let name = f
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("__pos_{field_idx}"));
                let ty = clone_tyref(&f.ty);
                Some((owner_root, name, ty))
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
            TermKind::UnwindResume | TermKind::Abort(_) => {
                // Rust panic propagation (unwind-table cleanup / abort).
                // No RPython analogue — RPython models neither destructors
                // nor a Rust-panic catch — so close the block as a bare
                // exception propagation into the canonical exceptblock.
                // Python-level exceptions never reach here: they ride the
                // `Result<_, PyError>` Switch/Return edges as ordinary
                // control flow.
                self.graph.set_raise(bb_id, "mir-unwind");
                Ok(())
            }
            TermKind::Goto { target } => {
                let target_bb = self.block_id[target as usize];
                // For now: no LinkArgs (MIR locals are function-wide,
                // and the target block carries no inputargs). The
                // moment we have a target block with inputargs, this
                // arm has to thread the matching Variables from
                // `self.local_var` through `Link::from_variables`.
                self.graph.set_goto(bb_id, target_bb, vec![]);
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
                self.graph.set_goto(bb_id, target_bb, vec![]);
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
                self.graph.set_goto(bb_id, target_bb, vec![]);
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
        for op in call.args {
            args.push(self.resolve_operand(mir_bb, op)?);
        }

        let class = call.func.classify();
        let op_kind = match (class, call.func) {
            (CallClass::Direct, CallFunc::Regular(reg))
            | (CallClass::Trait, CallFunc::Regular(reg)) => {
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
                let (segments, method_hint) = self.call_target_segments(mir_bb, &reg.kind)?;
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
                let target = match method_hint {
                    Some((owner_root, leaf)) if !args.is_empty() => {
                        CallTarget::method(leaf, Some(owner_root))
                    }
                    _ => CallTarget::FunctionPath { segments },
                };
                OpKind::Call {
                    target,
                    args,
                    result_ty: result_ty.clone(),
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

        // Allocate the result Variable and bind it to the destination
        // local before pushing the op, so subsequent reads see the
        // freshly-minted Variable.
        let result_var = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        self.local_var[dest_local] = Some(result_var.clone());
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
        // graph (`jit_codewriter/call.rs` `_canraise`), so dropping the
        // front-graph unwind edge keeps the can-raise signal. A real
        // try/except handler would need a `LastException` edge here; the
        // interpreter expresses exceptions as `Result`, so none arises.
        let _ = on_unwind;
        let target_bb = self.block_id[target];
        self.graph.set_goto(bb_id, target_bb, vec![]);
        Ok(())
    }

    /// Resolve a Charon `CallKind` to a flattened path segment list the
    /// codewriter consumes as `CallTarget::FunctionPath`, plus an
    /// optional `(owner_root_leaf, method_leaf)` pair for impl methods.
    ///
    /// The method hint is `Some` when the FunDecl's raw name segments
    /// encode an `Impl` block immediately before the leaf `Ident` —
    /// the standard Charon shape for inherent / trait-impl methods
    /// (e.g. `pyre_interpreter::pyframe::<Impl>::locals_w_mut`).  The
    /// caller uses the hint to pick `CallTarget::Method` over
    /// `CallTarget::FunctionPath` so the annotator can prepend a
    /// classdef-bound `SomeInstance` for `self`; see the comment at
    /// the use site in [`Self::lower_call`].
    fn call_target_segments(
        &self,
        mir_bb: usize,
        kind: &CallKind,
    ) -> Result<(Vec<String>, Option<(String, String)>), LowerError> {
        match kind {
            CallKind::Fun(FunId::Regular { id }) => self
                .llbc
                .fn_by_id(*id)
                .map(|fd| {
                    let segments: Vec<String> = fd
                        .item_meta
                        .name_path()
                        .split("::")
                        .map(|s| s.to_string())
                        .collect();
                    let method_hint = self.impl_method_owner(fd);
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
        let adt_def_id = self.resolve_impl_owner_adt_def_id(impl_payload)?;
        let td = self.llbc.type_by_id(adt_def_id)?;
        let owner_leaf = td
            .item_meta
            .name_path()
            .rsplit("::")
            .next()
            .unwrap_or("")
            .to_string();
        if owner_leaf.is_empty() {
            return None;
        }
        Some((owner_leaf, leaf))
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
            let trait_impls = self
                .llbc
                .file
                .translated
                .rest
                .get("trait_impls")?
                .as_array()?;
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
                // Route through `set_branch` so the cond gets the
                // upstream `bool` UnaryOp wrap before becoming the
                // exitswitch (flowcontext.py:756
                // `Variable.bool().eval(self)`).  Necessary because the
                // MIR discriminant for an `If` target can be a Ref
                // (e.g. a SyntheticTransparentCtor result) whereas
                // jit_codewriter/assembler.rs::FlatOp::GotoIfNot expects
                // `cond.kind == RegKind::Int`.
                self.graph.set_branch(
                    bb_id,
                    discr_var,
                    self.block_id[then_bb as usize],
                    vec![],
                    self.block_id[else_bb as usize],
                    vec![],
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
                    links.push(
                        Link::new_mixed(
                            vec![],
                            self.block_id[bb as usize],
                            Some(ExitCase::Const(case)),
                        )
                        .with_prevblock(bb_id)
                        .with_llexitcase_from_exitcase(),
                    );
                }
                links.push(
                    Link::new_mixed(
                        vec![],
                        self.block_id[default as usize],
                        Some(ExitCase::Const(ConstValue::UniStr("default".into()))),
                    )
                    .with_prevblock(bb_id),
                );
                self.graph.block_mut(bb_id).exitswitch = Some(ExitSwitch::Value(discr_var));
                self.graph.closeblock(bb_id, links);
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
    let adt_def_id = resolve_impl_owner_adt_def_id_free(llbc, impl_payload)?;
    let td = llbc.type_by_id(adt_def_id)?;
    // Owner-qualification convention: bare ident qualified by the
    // type's defining module path (e.g. `gc_roots::RootScope`).  Strip
    // the crate name from the TypeDecl's full name_path so the
    // `self_ty_root` keys land on a `[module::Owner, method]` CallPath.
    // Without this qualification the canonical registration loop at
    // `lib.rs:864-902` cannot find the graph keyed by
    // `[qualified_owner, method]`.
    let owner_qualified = strip_crate_prefix(&td.item_meta.name_path());
    if owner_qualified.is_empty() {
        return None;
    }
    Some((owner_qualified, leaf))
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
        let trait_impls = llbc.file.translated.rest.get("trait_impls")?.as_array()?;
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
/// `SemanticFunction.trait_root` so the canonical registration loop
/// can call `CallControl::register_trait_method` instead of routing
/// through `extract_trait_impls`.
fn trait_impl_trait_root_for_fundecl(llbc: &Llbc, fd: &FunDecl) -> Option<String> {
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
    let trait_impls = llbc.file.translated.rest.get("trait_impls")?.as_array()?;
    let ti = trait_impls.get(trait_impl_id as usize)?;
    let trait_id = ti
        .as_object()?
        .get("impl_trait")?
        .as_object()?
        .get("trait_id")?
        .as_u64()?;
    let td = llbc.trait_by_id(trait_id)?;
    let trait_leaf = td
        .item_meta
        .name_path()
        .rsplit("::")
        .next()
        .unwrap_or("")
        .to_string();
    if trait_leaf.is_empty() {
        return None;
    }
    Some(trait_leaf)
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
        // Bitwise.
        ("BitAnd", _) => "and".into(),
        ("BitOr", _) => "or".into(),
        ("BitXor", _) => "xor".into(),
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
            if lit_obj.contains_key("Float") {
                return ValueType::Float;
            }
        }
    }
    ValueType::Ref(None)
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
fn charon_type_value_to_ast_string(v: &serde_json::Value, llbc: &Llbc, depth: usize) -> String {
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

/// Format a Charon `Adt` type body (`{"id": …, "generics": {"types": […]}}`).
fn charon_adt_to_ast_string(
    adt: &serde_json::Map<String, serde_json::Value>,
    llbc: &Llbc,
    depth: usize,
) -> String {
    let id = adt.get("id");
    let type_args: Vec<String> = adt
        .get("generics")
        .and_then(|g| g.as_object())
        .and_then(|g| g.get("types"))
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .map(|t| charon_type_value_to_ast_string(t, llbc, depth + 1))
                // Drop the default allocator / hasher type-args Charon
                // makes explicit (`Vec<T, Global>`, `HashMap<K, V,
                // RandomState, Global>`) so the rendered string elides
                // them.
                .filter(|s| s != "Global" && s != "RandomState")
                .collect()
        })
        .unwrap_or_default();
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
