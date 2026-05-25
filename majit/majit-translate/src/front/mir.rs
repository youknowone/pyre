//! MIR-driven flowspace driver — issue #97 Step 3.
//!
//! This module is the **new** front-end the Charon migration is building
//! toward: it consumes Charon's ULLBC (a basic-block CFG derived from
//! rustc MIR) and produces the same [`FunctionGraph`] shape the rest of
//! the codewriter pipeline already consumes.
//!
//! It is structurally simpler than the AST-based driver in `front/ast.rs`
//! because the input is already in CFG form. The four mechanisms the AST
//! driver carries to *reconstruct* a CFG from a recursive walk —
//! `lazy_install_local_at_current_block_var`,
//! `can_thread_variable_to_block`, the `lower_if_expr` fallback branch,
//! and the per-scope binding tracking in `GraphBuildContext` — are
//! unnecessary here and have no analog in this driver. (Issue #97
//! Step 5 retires them once this driver covers the production target
//! set; that retirement is a *consequence* of this driver landing, not
//! a precondition.)
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
//! ## Scope as of issue #97 Step 3 — production coverage
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
//!     edge; matches `prototype/README.md` §"Deltas worth calling out"
//!     §4).
//!
//! ### Rvalues
//!   - `Use(operand)` — same-Variable alias.
//!   - `BinaryOp` / `UnaryOp` — `OpKind::BinOp` (unary uses a
//!     `unary.` prefix on the op label).
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
/// The lookup is the same `ends_with("::<name>")` rule the spike's
/// `local_fn` uses. Replace with a fully-qualified-path lookup once
/// the call graph plumbing makes it useful.
pub fn lower_function(llbc: &Llbc, function_name: &str) -> Result<FunctionGraph, LowerError> {
    let fd = llbc
        .local_fn(function_name)
        .ok_or_else(|| LowerError::FunctionNotFound(function_name.to_string()))?;
    lower_fun_decl(llbc, fd)
}

/// Step 4.6 multi-LLBC variant — merge functions and metadata from a
/// slice of LLBCs into one `SemanticProgram`.  When `pyre-jit-trace`
/// parses pyre-object + pyre-interpreter together, the cutover needs
/// each crate's `.ullbc` so cross-crate calls in the merged
/// SemanticProgram resolve.  Per-LLBC duplicates (a function defined
/// in both, e.g. via dependency closure) keep the first occurrence —
/// matches the AST builder's first-wins semantics under repeated
/// `parsed_files` entries.
pub fn build_semantic_program_from_llbcs(
    llbcs: &[Llbc],
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    let mut merged: Option<crate::front::semantic::SemanticProgram> = None;
    // Dedup key is the full qualified path (`{module_path}::{name}` or
    // just `name` for module-less entries), NOT the bare leaf.  Keying
    // on the leaf alone collapsed every same-named method across the
    // pyre-object → pyre-interpreter merge (`Vec::push` dropping
    // `PyFrame::push`, `Code::new` dropping `Code::new` from the other
    // crate, …), which surfaced as ≈20 audit gaps in
    // `audit_ast_extract_coverage` under MIR cutover.
    let mut seen_function_keys = std::collections::HashSet::new();
    let mut seen_struct_names = std::collections::HashSet::new();
    let mut seen_trait_names = std::collections::HashSet::new();
    let dedup_key = |f: &crate::front::semantic::SemanticFunction| -> String {
        if f.module_path.is_empty() {
            f.name.clone()
        } else {
            format!("{}::{}", f.module_path, f.name)
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
            }
        }
    }
    Ok(merged.unwrap_or_else(|| crate::front::semantic::SemanticProgram {
        functions: Vec::new(),
        known_struct_names: std::collections::HashSet::new(),
        known_trait_names: std::collections::HashSet::new(),
        struct_fields: crate::front::semantic::StructFieldRegistry::default(),
        fn_return_types: std::collections::HashMap::new(),
        immutable_fields: std::collections::HashMap::new(),
        module_statics: std::collections::HashMap::new(),
    }))
}

/// Build a [`SemanticProgram`] by lowering every local function
/// declaration in `llbc`.
///
/// This is the MIR-driven analog of
/// [`crate::front::ast::build_semantic_program_from_parsed_files`]
/// and the entry point Step 4 swaps into the production pipeline at
/// `lib.rs:134`.
///
/// **Whole-program metadata** (`known_struct_names`,
/// `known_trait_names`, `struct_fields`) is populated from
/// `type_decls` / `trait_decls` (Step 4.3.b). `fn_return_types` and
/// `immutable_fields` remain empty until Step 4.3.c.ext widens the
/// dedup table; the cutover at `lib.rs::build_semantic_program_via_active_frontend`
/// fills these in via a syn-AST merge pass over the same
/// parsed_files (Step 4.5.c).
///
/// Functions whose body Charon could not extract (extraction error,
/// opaque body, `null` entry) are skipped silently — the production
/// pipeline relies on `lib.rs`-level coverage audits to flag missing
/// graphs.  Functions whose MIR shape the driver does not yet handle
/// produce a [`LowerError`] that propagates out: the function-level
/// failure model from [`lower_function`] applies unchanged.
pub fn build_semantic_program_from_llbc(
    llbc: &Llbc,
) -> Result<crate::front::semantic::SemanticProgram, LowerError> {
    // ── Pass 1: walk type_decls + trait_decls (Step 4.3.b) ────────
    let (known_struct_names, known_trait_names, struct_fields) =
        derive_program_metadata(llbc);

    // ── Pass 2: lower every function body and build SemanticFunctions ─
    let mut functions = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    for fd in llbc.iter_local_fns() {
        if fd.unstructured().is_none() {
            continue;
        }
        // Step 4.5 naming alignment: AST front-end keys SemanticFunction
        // by bare leaf name plus a separate `module_path`.  Mirror the
        // shape so `register_function_graph_alias` (lib.rs:444) walks
        // `{bare, crate::*, pyre_*::*}` correctly for both front-ends
        // and the portal lookup at lib.rs:1043 (`["eval_loop_jit"]`)
        // resolves.
        let stripped = strip_crate_prefix(&fd.item_meta.name_path());
        let (module_path, name) = match stripped.rsplit_once("::") {
            Some((module, leaf)) => (module.to_string(), leaf.to_string()),
            None => (String::new(), stripped),
        };
        // Step 4.5: a single function whose body the driver does not
        // yet handle should not abort the whole-program build.
        // Capture per-function errors into a side bucket and continue;
        // the cutover surfaces them via `PYRE_MIR_FRONTEND_DEBUG=1`
        // for triage, but production keeps going with a degraded
        // SemanticProgram. This matches the AST driver's policy of
        // failing-loud on the single broken function rather than
        // erroring out at program-build time.
        let graph = match lower_fun_decl(llbc, fd) {
            Ok(g) => g,
            Err(e) => {
                skipped.push((name.clone(), e.to_string()));
                continue;
            }
        };
        // Step 4.5: return_type is intentionally `None` until the
        // Charon dedup-table widening (Step 4.3.c.ext) can resolve a
        // `TyRef::Deduplicated{id}` to its primitive name. The
        // codewriter's call-signature validator at
        // `jit_codewriter/call.rs:4234` skips the check when declared
        // type is None, which is the right behaviour while the
        // resolution gap is open — TyRef labels (`ty#170`) would
        // otherwise be classified as `Type::Ref` and trip a spurious
        // mismatch panic against a real `Type::Int` callee result.
        // Step 6.E prerequisite: surface impl-method owner on the
        // SemanticFunction so `lib.rs:868` / `lib.rs:1086` and the
        // (still-AST-side) `extract_inherent_impl_methods` /
        // `extract_trait_impls` consumers see the same
        // `self_ty_root` MIR records.  Without this, every impl method
        // built by MIR looks like a free function to the canonical
        // registration loop and the impl-key return-type / hint
        // registrations get dropped.
        let self_ty_root = impl_method_owner_for_fundecl(llbc, fd).map(|(owner, _)| owner);
        // Step 6.E follow-up: surface trait identity for trait-impl
        // methods so the canonical registration loop can call
        // `register_trait_method` instead of routing through
        // `extract_trait_impls`.  Inherent impls leave `trait_root =
        // None`; trait-impl methods carry the trait's leaf name.
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
    if std::env::var("PYRE_MIR_FRONTEND_DEBUG").is_ok() && !skipped.is_empty() {
        eprintln!(
            "[mir-frontend] {} function(s) skipped during lowering:",
            skipped.len()
        );
        for (name, msg) in skipped.iter().take(20) {
            eprintln!("  {name}: {msg}");
        }
    }
    Ok(crate::front::semantic::SemanticProgram {
        functions,
        known_struct_names,
        known_trait_names,
        struct_fields,
        // Step 4.5: fn_return_types empty until Step 4.3.c.ext (Charon
        // dedup-table widening) lets us resolve TyRef→primitive name.
        // Empty is type-validator-safe; TyRef labels are not.
        fn_return_types: std::collections::HashMap::new(),
        // Immutable-field tracking depends on `#[majit_macros::immutable]`
        // attribute serialization that Charon does not currently surface
        // (the `attributes` array carries DocComment / Outer but not our
        // proc-macro hints). Tracked under Step 4.3.d.
        immutable_fields: std::collections::HashMap::new(),
        // Module-static literals come from the AST pre-walk
        // (collect_program_metadata_pub via merge_module_statics_from_
        // parsed_files); LLBC carries the encoded const values without
        // their source spelling, so MIR leaves this empty and the AST
        // merge populates it.
        module_statics: std::collections::HashMap::new(),
    })
}

/// Step 4.3.b: derive whole-program type-metadata fields of
/// `SemanticProgram` from Charon's `type_decls` + `trait_decls`
/// tables.
///
/// Returns `(known_struct_names, known_trait_names, struct_fields)`.
/// Names are taken from `item_meta.name_path()`; struct field rows use
/// `TyRef::label()` as the field type string (matching Step 4.3.a's
/// label-as-placeholder convention).
fn derive_program_metadata(
    llbc: &Llbc,
) -> (
    std::collections::HashSet<String>,
    std::collections::HashSet<String>,
    crate::front::semantic::StructFieldRegistry,
) {
    let mut known_struct_names = std::collections::HashSet::new();
    let mut known_trait_names = std::collections::HashSet::new();
    let mut struct_fields = crate::front::semantic::StructFieldRegistry::default();

    for td in llbc.iter_type_decls() {
        let name = td.item_meta.name_path();
        match &td.kind {
            TypeDeclKind::Struct(fields) => {
                // Register the qualified path *and* the bare leaf name
                // so downstream lookups (`canonical_call_target`'s
                // bare-leaf fallback) resolve either spelling. Matches
                // the AST driver's dual-publish convention.
                let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
                let rows: Vec<(String, String)> = fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let fname = f
                            .name
                            .clone()
                            .unwrap_or_else(|| format!("__pos_{i}"));
                        (fname, f.ty.label())
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
                // …) so synthetic Aggregate(SyntheticTransparentCtor)
                // emitted by Step 3.9 can be matched downstream.
                let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
                known_struct_names.insert(name.clone());
                known_struct_names.insert(leaf);
                for v in variants {
                    let variant_path = format!("{name}::{}", v.name);
                    known_struct_names.insert(variant_path);
                }
            }
            TypeDeclKind::Alias(_)
            | TypeDeclKind::Opaque
            | TypeDeclKind::Unknown => {}
        }
    }

    for td in llbc.iter_trait_decls() {
        let name = td.item_meta.name_path();
        let leaf = name.rsplit("::").next().unwrap_or(&name).to_string();
        known_trait_names.insert(name);
        known_trait_names.insert(leaf);
    }

    (known_struct_names, known_trait_names, struct_fields)
}

/// Lower a single Charon [`FunDecl`] to a [`FunctionGraph`].
pub fn lower_fun_decl(llbc: &Llbc, fd: &FunDecl) -> Result<FunctionGraph, LowerError> {
    let u = fd.unstructured().ok_or_else(|| {
        LowerError::Unsupported(format!(
            "{}: no Unstructured body (extracted with --ullbc?)",
            fd.item_meta.name_path()
        ))
    })?;
    let mut lo = Lowering::new(llbc, fd.item_meta.name_path(), &u)?;
    lo.lower()?;
    Ok(lo.graph)
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
        // ty }` op into the startblock, mirroring AST
        // (`front/ast.rs:1636-1646 / 1656-1665`).  Downstream consumers
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
            // arg-renames.
            graph.value_names.insert(graph.slot_of(&var).unwrap(), name.clone());
            local_var[i] = Some(var.clone());
            let ty = tyref_to_value_type(&local.ty, llbc);
            input_ops.push(SpaceOperation {
                result: Some(var.clone()),
                kind: OpKind::Input { name, ty },
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
        })
    }

    fn lower(&mut self) -> Result<(), LowerError> {
        // BFS over MIR basic blocks. Per the §"Reference" section
        // above, there is no mergeblock dance — every MIR basic block
        // is its own join point, fully prepared by `create_block` /
        // `startblock`. So iteration order is *only* about
        // deterministic processing; we use linear MIR order (BFS in
        // the trivial single-predecessor sense) and keep it that way
        // for reproducibility.
        for mir_bb in 0..self.body.body.len() {
            self.lower_block(mir_bb)?;
        }
        Ok(())
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

            // Inline overflow assertion — see issue #97 prototype README
            // §"Deltas worth calling out" §4. We strip these for now;
            // a future widening can honour them when interpreter
            // semantics require panic-on-overflow.
            StmtKind::Assert(_) => Ok(()),

            StmtKind::Assign(place, rvalue) => {
                self.lower_assign(mir_bb, place, rvalue)
            }

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
                let (op, result_var) = self.build_rvalue(mir_bb, rvalue)?;
                // The destination local takes on the freshly-minted
                // result Variable. Subsequent reads of the local
                // resolve to this Variable until the next Assign
                // overwrites the slot.
                self.local_var[i as usize] = Some(result_var.clone());
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
        let op: Operand = serde_json::from_value(offset).map_err(|e| {
            LowerError::Schema(format!("bb{mir_bb}: Index offset decode: {e}"))
        })?;
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
            // `UnaryOp(op, operand)` — `Neg`, `Not`, etc. Modeled as a
            // single-arg `BinOp` whose op label carries a `unary.` prefix
            // so downstream consumers can spot the arity mismatch without
            // re-parsing the label.  PRE-EXISTING-ADAPTATION: RPython has
            // distinct `int_neg`/`int_invert` ops, but pyre's `OpKind` does
            // not expose a typed unary form and front/mod.rs forbids
            // introducing one from this layer.
            Rvalue::UnaryOp(op_json, operand) => {
                let arg = self.resolve_operand(mir_bb, operand)?;
                let op_label = format!("unary.{}", binop_label(&op_json)?);
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                Ok((
                    Some(OpKind::BinOp {
                        op: op_label,
                        // Repeat the arg on both sides so the existing
                        // 2-operand shape stays usable; the `unary.`
                        // label tells consumers to disregard `rhs`.
                        lhs: arg.clone(),
                        rhs: arg,
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
            // keeps the IR small and matches the AST driver's behaviour
            // of treating `&x` as a same-Variable copy.
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
            // same as the operand Variable. PRE-EXISTING-ADAPTATION:
            // the AST front-end also collapses `as` casts that do not
            // change the JIT-visible kind.
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
                    obj.keys().next().cloned().unwrap_or_else(|| "nullary".into())
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
                let mut args: Vec<Variable> = Vec::with_capacity(operands.len());
                for op in operands {
                    args.push(self.resolve_operand(mir_bb, op)?);
                }
                let ctor_name = aggregate_ctor_name(&kind);
                let res = self
                    .graph
                    .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                Ok((
                    Some(OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor(ctor_name),
                        args,
                        result_ty: ValueType::Int,
                    }),
                    res,
                ))
            }
            // `Discriminant(place)` — read the integer tag of an enum
            // value. Modeled as a synthetic `FieldRead` of an
            // `__discriminant` field: tag access is morally a pure
            // field read at the bit level, and reusing the existing
            // `FieldRead` shape keeps the IR closed under the AST
            // front-end's opkind catalogue (per `front/mod.rs` rule —
            // no new OpKinds in this layer). `owner_root` is left
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
            // PRE-EXISTING-ADAPTATION: AST front-end resolves these
            // through `front/ast.rs::lower_expr` const folding and
            // never emits them as a free-standing op either.
            DecodedConst::Str(s) => OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["__str_const".to_string(), s],
                },
                args: vec![],
                result_ty: ValueType::Int,
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
                // `OpKind::FieldRead` so downstream consumers
                // (codewriter inlining + annotator GetAttr dispatch
                // on cross-procedural callers like
                // `flowspace/rust_source/build_flow.rs:4770
                // lower_field`) see the same field/owner_root shape
                // the AST front-end emits at `front/ast.rs:4923`.
                //
                // Tuple-container `Field` projections still collapse
                // (the `straight_line_add` / AddChecked `(value,
                // bool)` shape needs `.0` reads to fall through to
                // the underlying Variable; the paired `.1` Assert is
                // dropped in `lower_statement`).
                //
                // Atom projections (`Deref` and others) still
                // collapse: `Deref` is a no-op for typed refs at the
                // JIT IR level, and any other Atom variant has no
                // typed analogue today.
                if let ProjectionElem::Tagged(v) = &elem
                    && let Some(field_payload) =
                        v.as_object().and_then(|m| m.get("Field"))
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
            // already handled by the AST front-end's hint pass).
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
    /// The owner_root is the LLBC TypeDecl's leaf name
    /// (`PyFrame` from `pyre_interpreter::pyframe::PyFrame`) so the
    /// downstream `struct_fields` registry — populated in
    /// AST-format by `merge_fn_return_types_from_parsed_files`
    /// (lib.rs:286) — resolves with the same leaf key.
    fn resolve_adt_field(
        &self,
        payload: &serde_json::Value,
    ) -> Option<(String, String, TyRef)> {
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
                let name = f.name.clone().unwrap_or_else(|| format!("__pos_{field_idx}"));
                let ty = clone_tyref(&f.ty);
                Some((owner_root, name, ty))
            }
            (TypeDeclKind::Enum(variants), Some(vidx)) => {
                let variant = variants.get(vidx as usize)?;
                let f = variant.fields.get(field_idx)?;
                let name = f.name.clone().unwrap_or_else(|| format!("__pos_{field_idx}"));
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
                let ret = self.local_var[0].clone().ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "bb{mir_bb}: Return without any Assign to MIR local 0"
                    ))
                })?;
                self.graph.set_return(bb_id, Some(ret));
                Ok(())
            }
            TermKind::UnwindResume | TermKind::Abort(_) => {
                // Close the block as exception-propagating. The AST
                // driver uses `set_raise` with a static reason; mirror
                // that until the typed exception lowering pass lands.
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
            TermKind::Assert { target, on_unwind, .. } => {
                // Inline overflow assertion at terminator level —
                // strip it: branch unconditionally to the success
                // continuation. The unwind successor (which always
                // ends in UnwindResume in extracted bodies) becomes
                // unreachable from us. This mirrors the policy
                // documented in `prototype/README.md` §"Deltas
                // worth calling out" §4.
                let _ = on_unwind;
                let target_bb = self.block_id[target as usize];
                self.graph.set_goto(bb_id, target_bb, vec![]);
                Ok(())
            }
            TermKind::Switch { discr, targets } => {
                self.lower_switch(mir_bb, discr, targets)
            }
            TermKind::Call { call, target, on_unwind } => {
                self.lower_call(mir_bb, call, target as usize, on_unwind as usize)
            }
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
                // classdef-bound `SomeInstance` for `self`.  AST does
                // the same at `front/ast.rs:5205` for
                // `syn::Expr::MethodCall`; without it, the callee
                // body's `self` lands with `classdef=None` and any
                // `.field` projection on it panics at
                // `unaryop.rs:3587` (lib test
                // `generic_handler_graphs_keep_symbolic_fnaddr_surface`).
                let (segments, method_hint) =
                    self.call_target_segments(mir_bb, &reg.kind)?;
                let target = if let Some((owner_root, leaf)) = method_hint {
                    CallTarget::method(leaf, Some(owner_root))
                } else {
                    CallTarget::FunctionPath { segments }
                };
                OpKind::Call {
                    target,
                    args,
                    result_ty: ValueType::Int,
                }
            }
            (CallClass::Dynamic, CallFunc::Dynamic(dyn_operand)) => {
                // `dyn Trait` virtual call. The fat-pointer receiver
                // is carried in `dyn_operand`; thread it into `args[0]`
                // and emit a synthetic `__dyn_call` path so the
                // codewriter sees a uniform `Call` shape.
                // PRE-EXISTING-ADAPTATION: a faithful lowering would
                // emit `VtableMethodPtr` + `IndirectCall`; that needs
                // the trait_root/method_name pair Charon does not yet
                // surface (tracked under [[step3-dynamic-call-vtable]]).
                let recv = self.resolve_operand(mir_bb, dyn_operand)?;
                let mut full_args = Vec::with_capacity(args.len() + 1);
                full_args.push(recv);
                full_args.extend(args);
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["__dyn_call".to_string()],
                    },
                    args: full_args,
                    result_ty: ValueType::Int,
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

        // Close the block: forward to the success target. The unwind
        // continuation feeds the exceptblock — under current policy we
        // do not wire a separate exception edge because the AST driver
        // also collapses post-call unwinds to the function's single
        // `exceptblock` and we have not yet introduced typed exception
        // links here. Tracked under Step 3.6 follow-up.
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
            // The AST front-end's `extract_trait_impls` parses the
            // trait declaration's default-body and registers it under
            // BOTH `["<default methods of <Trait>>", <method>]` (the
            // selfclassdef-bound `register_trait_method` path) and the
            // direct path `[<Trait>, <method>]` (lib.rs:957-969 —
            // `register_function_graph(direct_path, …)`).  The direct
            // path is the call-site shape Rust code emits when calling
            // `<Trait>::<method>(receiver, …)` and the BFS-driven
            // `find_all_graphs` reaches it as a regular candidate.
            //
            // To stay PyPy-orthodox for generic-trait dispatch in MIR
            // mode, route the call through that same `[<Trait>,
            // <method>]` path so:
            //   1. BFS discovers the trait default body as a
            //      candidate, which transitively pulls in the helpers
            //      it calls (e.g. `opcode_load_const`).
            //   2. `flowspace_adapter` emits the same `simple_call(<
            //      callable>, args…)` shape AST does (no `getattr`
            //      surface) so the classdef-less receiver does not
            //      surface as a panicking `SomeInstance.getattr`.
            //
            // Falls back to the legacy `["__trait_method", <label>]`
            // synthetic path when the fn_decl cannot be resolved or
            // does not have the trait-method shape (e.g. when arr[2]
            // is missing or points at an `Impl` block).
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
        let last_idx = segs.iter().rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
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
    fn resolve_impl_owner_adt_def_id(
        &self,
        impl_payload: &serde_json::Value,
    ) -> Option<u64> {
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
        let mut links: Vec<Link> = Vec::new();
        match targets {
            SwitchTargets::If(then_bb, else_bb) => {
                links.push(
                    Link::new_mixed(vec![], self.block_id[else_bb as usize], Some(ExitCase::Bool(false)))
                        .with_prevblock(bb_id)
                        .with_llexitcase_from_exitcase(),
                );
                links.push(
                    Link::new_mixed(vec![], self.block_id[then_bb as usize], Some(ExitCase::Bool(true)))
                        .with_prevblock(bb_id)
                        .with_llexitcase_from_exitcase(),
                );
            }
            SwitchTargets::SwitchInt(_int_ty, arms, default) => {
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
            }
        }
        self.graph.block_mut(bb_id).exitswitch = Some(ExitSwitch::Value(discr_var));
        self.graph.closeblock(bb_id, links);
        Ok(())
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
    let last_idx = segs.iter().rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
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
    // Match the AST builder's owner-qualification convention: bare
    // ident qualified by the type's defining module path.  AST does
    // `qualify_type_name_with_imports(type_root, module_prefix, …)`
    // → `gc_roots::RootScope`; mirror that by stripping the crate
    // name from the TypeDecl's full name_path so MIR's
    // `self_ty_root` keys land on the same `[module::Owner, method]`
    // CallPath the AST extractors use.  Without this alignment the
    // canonical registration loop at `lib.rs:864-902` cannot find
    // the MIR-built graph keyed by `[qualified_owner, method]`.
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
        let trait_impls = llbc
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
/// through the AST-side `extract_trait_impls`.
fn trait_impl_trait_root_for_fundecl(llbc: &Llbc, fd: &FunDecl) -> Option<String> {
    let segs = &fd.item_meta.name;
    let last_idx = segs.iter().rposition(|s| matches!(s, NameSeg::Ident { .. }))?;
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
    let trait_impls = llbc
        .file
        .translated
        .rest
        .get("trait_impls")?
        .as_array()?;
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

fn binop_label(v: &serde_json::Value) -> Result<String, LowerError> {
    // Plain atom: `"Add"`, `"Eq"`, …
    if let Some(s) = v.as_str() {
        return Ok(s.to_string());
    }
    // Tagged form: `{"Add": "Wrap"}`, `{"Shr": "Wrap"}` etc — combine
    // into `"AddWrap"` style label. Distinct from the bare form so a
    // future widening can route them to different OpKinds if
    // semantics diverge (e.g. emitting wrapping vs. saturating math).
    if let Some(obj) = v.as_object() {
        if let Some((k, payload)) = obj.iter().next() {
            let suffix = match payload {
                serde_json::Value::String(s) => s.clone(),
                _ => payload.to_string(),
            };
            return Ok(format!("{k}{suffix}"));
        }
    }
    Err(LowerError::Schema(format!(
        "BinaryOp op label has unexpected shape: {v}"
    )))
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
        TyRef::Inline { value: (id, v) } => TyRef::Inline { value: (*id, v.clone()) },
        TyRef::Other(v) => TyRef::Other(v.clone()),
    }
}

/// Project a Charon [`TyRef`] to the JIT-visible [`ValueType`].
///
/// Mirrors AST's `type_string_to_value_type` (front/ast.rs:10310)
/// surface: numeric scalars → `Int` / `Float`, bool → `Bool`, unit
/// → `Void`, everything else (structs, pointers, references) →
/// `Ref`.  The TyRef's serialized form is the source of truth —
/// `TyRef::label()` produces a compact short form
/// (`"ty#170"`, `"ty<Adt>"`) for opaque IDs, while the underlying
/// JSON carries the primitive name for literal types.
///
/// For `TyRef::Deduplicated{id}`, the projection consults
/// `llbc.dedup_body(id)` to recover the inline body shape and runs
/// the same primitive-pattern match.  Required so FunDecl return
/// types serialized as `Deduplicated` (≈92% in `pyre-interpreter.ullbc`)
/// resolve to `Int` / `Bool` / `Float` instead of falling back to
/// `Ref` (Step 4.3.c.ext / Task #30).
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
    // (Older Charon revisions used a single `{"Literal": {"Integer":
    // …}}` shape, which we still accept for forward-compat with any
    // pre-extracted .ullbc artefacts still floating around.)
    //
    // Unit type `()` lowers to `{"Adt": [tuple_arity = 0, []]}` which
    // routes through the final `Ref` fallback; the codewriter treats
    // void-typed Variables uniformly via `getkind`'s 'v' arm.
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
/// (`pyre_interpreter::frame::eval_loop_jit`), while the AST front-end
/// names functions relative to their parsed-file root
/// (`frame::eval_loop_jit` for a non-empty `module_path`, or the bare
/// leaf for `module_path == ""`).  Matching the AST convention lets
/// `register_function_graph_alias` (lib.rs:444) walk
/// `{bare, crate::*, pyre_interpreter::*, pyre_object::*, pyre_jit::*}`
/// aliases off the same `func.name` for both front-ends.
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
    /// string constant opkind in the AST front-end either; the
    /// codewriter treats these as opaque pointer-typed values. We
    /// carry the textual representation as a unique-string `ConstValue`
    /// so the generated IR is stable across runs.
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
fn decode_constant(
    llbc: &Llbc,
    value: &serde_json::Value,
) -> Result<DecodedConst, LowerError> {
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
    // requires the trait dispatch widening (Step 3.11).
    if kind.contains_key("VTableRef") {
        return Ok(DecodedConst::Str("__vtable_ref".to_string()));
    }
    // `TraitConst` — trait-associated const. Opaque for now; covering
    // it faithfully requires trait/impl resolution (Step 3.11).
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
            LowerError::Schema(format!("FnDef constant references unknown FunDecl id {inner}"))
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
        if let Some(s) = f.as_object().and_then(|m| m.get("value")).and_then(Value::as_str) {
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

