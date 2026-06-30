//! `model::FunctionGraph` (pyre surface DSL) →
//! `flowspace::FunctionGraph` (RPython-orthodox) adapter.
//!
//! TODO(retire-this-adapter): this file has no upstream RPython
//! counterpart. RPython's pipeline has only one graph model — the
//! annotator builds its `FunctionGraph` (`rpython/flowspace/model.py`)
//! directly and the rtyper consumes it in place.  Pyre carries two
//! graph models in parallel: `crate::model::FunctionGraph` for the
//! surface DSL emitted by `parse → front → SemanticProgram`, and
//! `crate::flowspace::model::FunctionGraph` for the real
//! `translator/rtyper/` pipeline.  This adapter bridges the gap.
//! Retire when the surface DSL is replaced with `flowspace`-native
//! producers; until then the adapter remains the per-graph entry
//! into `RPythonAnnotator` / `RPythonTyper::specialize`.
//!
//! ## Why not `PyGraph`?
//!
//! `PyGraph` (`flowspace::pygraph::PyGraph`) wraps a `FunctionGraph`
//! with `GraphFunc` / `HostCode` / `Signature` / `defaults` —
//! Python-runtime function metadata that pyre's surface DSL does not
//! produce (`parse → front → SemanticProgram` operates on Rust source,
//! not CPython callables). `RPythonTyper::specialize`
//! (`rtyper.rs:1743`) does NOT consume `PyGraph` directly — it iterates
//! `RPythonAnnotator.annotated` / `all_blocks`, which
//! `specialize_legacy_graph` will populate with the
//! [`FlowspaceAdapterOutput`] this adapter returns. Skipping the PyGraph
//! wrapping avoids fabricating fake `GraphFunc` / `HostCode` instances.
//!
//! ## Layout
//!
//! The adapter performs three responsibilities, all line-by-line at
//! `function_graph_to_flowspace`:
//!
//! 1. **Annotation lift** — clone each legacy `Variable.annotation`
//!    cell (`Rc<RefCell<Option<Rc<SomeValue>>>>`) onto a
//!    freshly-allocated `flowspace::Variable`. Variable identity is
//!    block-local per `flowspace/model.py:checkgraph`; the adapter
//!    keeps a `legacy Variable → typed Variable` representative map
//!    (keyed on Variable object identity) for post-specialize readback.
//! 2. **Per-OpKind translation** — `translate_op` maps each pyre
//!    `model::SpaceOperation` to a `flowspace::SpaceOperation` over
//!    `Hlvalue` operands.  Pre-rtyper variants (`Input`, `ConstInt`,
//!    `ConstFloat`, `BinOp`, `Call`, `FieldRead`, `ArrayRead`, ...)
//!    have explicit arms; post-rtyper jtransform variants are
//!    classified by [`post_rtyper_jtransform_variant_name`] and
//!    fail-loud with a stage-mismatch message.
//! 3. **Block topology** — wires `flowspace::Block` per legacy
//!    `Block`, translates `exits` / `exitcase` / `exitswitch`,
//!    designates `startblock` / `returnblock` / `exceptblock`, and
//!    assembles a `flowspace::FunctionGraph`.  `getreturnvar`
//!    (`rtyper.rs:1633-1638`) is non-degenerate because the
//!    returnblock's inputarg is materialised as the canonical
//!    flowspace return `Variable`.
//!
//! [`crate::translator::rtyper::cutover::specialize_legacy_graph_with_registry_returning_value_to_var`]
//! drives this adapter, runs `RPythonTyper::specialize`, and returns
//! the legacy-`Variable`-keyed typed-`Variable` map + the per-`Variable`
//! `Constant.concretetype` `LowLevelType` table that consumers project to
//! `ConcreteType` on demand.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::flowspace::model::{
    self as flowspace_model, Block as FlowspaceBlock, BlockRef, ConstValue, Constant,
    FunctionGraph as FlowspaceGraph, HOST_ENV, Hlvalue, HostObject, Link as FlowspaceLink,
    SpaceOperation as FlowspaceOp, Variable, c_last_exception,
};
use crate::model::{BlockId, ExitCase, ExitSwitch, FunctionGraph, LinkArg, OpKind, SpaceOperation};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

/// Map from a legacy graph `Variable` to the representative typed
/// `flowspace::Variable` the adapter created for readback.
///
/// This is not the graph's identity model. RPython `checkgraph` requires
/// block inputargs and operation results to be defined in exactly one
/// block, so [`function_graph_to_flowspace`] uses block-local Variables
/// while translating the actual graph. The map is keyed by the **legacy
/// graph Variable's object identity** (RPython keys cross-graph dicts on
/// Variable objects, never on an integer slot): `legacy_var -> typed_var`
/// lets consumers read `typed_var.concretetype` / `.annotation` back onto
/// the legacy Variable they already hold, with no slot-index round-trip.
pub type LegacyToTyped = HashMap<Variable, Variable>;

pub use crate::codewriter::annotation_state::valuetype_to_someshell;

/// Allocate a fresh `flowspace::Variable` and attach the projected
/// `SomeValue` shell to its `annotation` slot.
///
/// The legacy `Variable.id` does NOT carry over to the fresh
/// Variable's id — `Variable::new` allocates a fresh process-wide
/// identity (`flowspace/model.rs:2042`). Identity correspondence is
/// preserved out-of-band by [`LegacyToTyped`].
fn seed_variable(legacy_var: &Variable) -> Variable {
    let var = Variable::new();
    // Copy the precise per-`Variable.annotation` `SomeValue` shell
    // from the legacy Variable onto the freshly minted one,
    // matching upstream `_setbinding(v, s_value)` semantics
    // (`rpython/annotator/annrpython.py:333-340`).
    //
    // Source: tests that hand-seed annotations call
    // `legacy_annotator::setbinding(&legacy_var, ty)` before
    // reaching here;
    // the production `addpendingblock` flowin path leaves
    // `legacy_var.annotation` empty so the fresh Variable
    // starts unannotated and flowin populates it via `setbinding`.
    // The dual-gate baseline that calls `legacy_annotator::annotate`
    // runs AFTER specialize completes (cutover.rs:dual_gate_check /
    // dual_gate_check_with_registry baseline section), so the wider
    // legacy lift never reaches this site.
    //
    // An empty `legacy_var.annotation` slot (unpopulated or
    // `ValueType::Unknown`) leaves the fresh `Variable.annotation`
    // empty — the rtyper then fails at `bindingrepr` with `KeyError:
    // no binding for arg` on first touch, surfacing the producer-
    // side gap rather than silently bridging to `GcRef` via a
    // fabricated `SomeInstance(None)` shell.
    if let Some(s) = legacy_var.annotation.borrow().as_ref() {
        *var.annotation.borrow_mut() = Some(s.clone());
    }
    var
}

/// Build the `legacy Variable → typed flowspace::Variable` map for every
/// value reachable from `legacy.blocks`.
///
/// Three reference-site classes seed the map:
///
/// 1. **Definitions** — `block.inputargs` (RPython-orthodox phi nodes)
///    and `op.result`. Every operand referenced via `op.args` /
///    `link.args` / `exitswitch` resolves to a definition site in the
///    same graph (legacy `FunctionGraph` is mostly SSA), so seeding
///    definitions covers most of the value set.
///
/// 2. **Link-side sentinels** — `link.args` / `link.last_exception` /
///    `link.last_exc_value`. RPython `flowspace/model.py:114` and
///    pyre's front (`front::mir`) allow a `Link.args` slot
///    to carry a *fresh* prevblock-side `Variable` whose only "defining
///    site" is the link itself — the value flows into the target
///    block's inputarg via this synthetic Variable. The adapter must
///    seed a `Variable` for each such slot so the link
///    translation can resolve the operand without tripping the
///    "undefined operand" invariant in `lookup_operand`.
///
/// 3. **Exitswitch values** — `block.exitswitch = Some(ExitSwitch::Value(vid))`
///    sometimes references a slot defined in a successor block's
///    inputarg context (rarely but legitimately in legacy graphs).
///    Seeded for the same reason.
///
/// Each legacy Variable is seeded exactly once via
/// `entry().or_insert_with`, preserving operand identity across
/// multiple readers — the op translator looks up the same Variable
/// instance for every reader of a given operand, matching upstream
/// Python's reference semantics where `op.args[i]` and `op2.args[j]`
/// may be the same `Variable` object.
///
/// **Restricted to the adapter / its tests.**  `function_graph_to_flowspace`
/// builds a *block-local* `legacy Variable -> typed Variable` map per
/// block in the topology assembly pass, mirroring `RPython` `checkgraph`'s
/// per-block Variable invariant
/// (`rpython/flowspace/model.py:585-590`: a Variable must be defined in
/// exactly one block).  Using this whole-graph helper as the source of
/// truth for the adapter's main path would violate that invariant by
/// reusing a single `Variable` across blocks.  The helper stays in the
/// crate solely to back the adapter's regression tests
/// (`build_value_to_variable_map_*`); production cutover code must use
/// the per-block maps owned by `function_graph_to_flowspace`.
#[cfg(test)]
pub(crate) fn build_value_to_variable_map(legacy: &FunctionGraph) -> LegacyToTyped {
    // Callers that hand-seed test fixtures must write directly to
    // `legacy.variable(vid).annotation` via
    // `legacy_annotator::setbinding(&var, ty)` before invoking this so
    // downstream `seed_variable` reads through the orthodox
    // `Variable.annotation` carrier.
    let mut map: LegacyToTyped = HashMap::new();
    for block in &legacy.blocks {
        // Class 1a — block-inputarg definitions.
        for var in &block.inputargs {
            map.entry(var.clone()).or_insert_with(|| seed_variable(var));
        }

        // Per-block name → inputarg-Variable lookup for `OpKind::Input`
        // rebind-aliasing. Pyre's surface front (`front::mir`) emits a
        // *leading* `Input{name, ty}` op for each named parameter whose
        // `op.result` matches a `block.inputargs` entry, and may emit
        // *additional* `Input{same name}` ops with fresh `op.result`
        // slots for body-side rebinds. RPython's flowspace has no
        // such Input op machinery — the parameter Variable IS the
        // inputarg. Without aliasing the rebind result to the
        // canonical inputarg Variable, `setup_block_entry` writes
        // `concretetype` only on the inputarg's Rc<RefCell> and the
        // body's BinOp lookup hits a fresh Variable with `None`
        // concretetype, tripping `genop`'s "wrong level!" assertion.
        let mut name_to_inputarg_var: HashMap<&str, Variable> = HashMap::new();
        for op in &block.operations {
            if let OpKind::Input { name, .. } = &op.kind
                && let Some(result_var) = op.result.as_ref()
                && block.inputargs.contains(result_var)
                && let Some(var) = map.get(result_var)
            {
                name_to_inputarg_var
                    .entry(name.as_str())
                    .or_insert_with(|| var.clone());
            }
        }

        // Class 1b — op-result definitions, with Input rebind aliasing.
        //
        // `OpKind::Abort` (pyre-only front-end marker for unsupported
        // expression forms) is intentionally NOT seeded into
        // `value_to_var`.  `translate_op`'s Abort arm emits no
        // flowspace op (`flowspace_adapter.rs:648 OpKind::Abort { .. }
        // => Ok(Vec::new())`), so seeding the result_var here would
        // hand consumer ops a `Hlvalue::Variable` that never gets
        // *defined* by any emitted flowspace op — `checkgraph`
        // (`flowspace/model.rs::checkgraph`) then panics with
        // "variable used before definition" at the consumer's arg
        // slot, NOT at the missing-operand site.  Skipping the seed
        // here forces the first consumer's `lookup_operand` to fail
        // with "undefined operand" instead (`is_known_unported`
        // already matches that substring; the dual gate Skip-classifies
        // the graph cleanly at the producer-adjacent site).
        for op in &block.operations {
            let Some(result_var) = op.result.as_ref() else {
                continue;
            };
            if matches!(op.kind, OpKind::Abort { .. }) {
                continue;
            }
            if map.contains_key(result_var) {
                continue;
            }
            let var = if let OpKind::Input { name, .. } = &op.kind {
                name_to_inputarg_var
                    .get(name.as_str())
                    .cloned()
                    .unwrap_or_else(|| seed_variable(result_var))
            } else {
                seed_variable(result_var)
            };
            map.insert(result_var.clone(), var);
        }
        // Class 3 — exitswitch-referenced values.
        if let Some(crate::model::ExitSwitch::Value(var)) = &block.exitswitch {
            map.entry(var.clone()).or_insert_with(|| seed_variable(var));
        }
        // Class 2 — link-side sentinels.
        for link in &block.exits {
            for arg in &link.args {
                if let Some(var) = arg.as_variable() {
                    map.entry(var.clone()).or_insert_with(|| seed_variable(var));
                }
            }
            if let Some(arg) = link.last_exception.as_ref()
                && let Some(var) = arg.as_variable()
            {
                map.entry(var.clone()).or_insert_with(|| seed_variable(var));
            }
            if let Some(arg) = link.last_exc_value.as_ref()
                && let Some(var) = arg.as_variable()
            {
                map.entry(var.clone()).or_insert_with(|| seed_variable(var));
            }
        }
    }
    map
}

/// result-`Variable` → `Hlvalue` map combining the [`LegacyToTyped`]
/// map with constant-inlining of `OpKind::ConstInt` / `ConstFloat`
/// define-ops.
///
/// RPython's flowspace inlines constants natively as `Hlvalue::Constant`
/// in `op.args` (`flowspace/operation.py:152` `simple_call(target,
/// *args)` — `target` and each `arg` is either a `Variable` or
/// `Constant`). Pyre's legacy graph splits constants into define-ops
/// (`OpKind::ConstInt(n)` produces a fresh result consumed
/// elsewhere). The adapter must recombine: every result defined as a
/// const becomes a `Hlvalue::Constant`; every other defined result
/// remains a `Hlvalue::Variable` from the variable map.
///
/// Constants are wrapped with their low-level concretetype attached,
/// matching RPython's `Constant.concretetype` shape.
///
/// Test-only: production const-inlining is carried by the `value_map`
/// / `value_to_var` Variable-keyed path; this result-Variable-keyed
/// builder survives solely as the mirror the
/// `build_value_to_hlvalue_map_inlines_const_defines` fixture diffs
/// against, so it is gated `#[cfg(test)]`.
#[cfg(test)]
pub fn build_value_to_hlvalue_map(
    legacy: &FunctionGraph,
    value_to_var: &LegacyToTyped,
) -> HashMap<crate::flowspace::model::Variable, Hlvalue> {
    let mut map: HashMap<crate::flowspace::model::Variable, Hlvalue> = value_to_var
        .iter()
        .map(|(legacy_var, var)| (legacy_var.clone(), Hlvalue::Variable(var.clone())))
        .collect();

    for block in &legacy.blocks {
        for op in &block.operations {
            let Some(result) = op.result.clone() else {
                continue;
            };
            match &op.kind {
                OpKind::ConstInt(n) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Int(*n),
                            LowLevelType::Signed,
                        )),
                    );
                }
                OpKind::ConstBool(b) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Bool(*b),
                            LowLevelType::Bool,
                        )),
                    );
                }
                OpKind::ConstFloat(bits) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(Constant::with_concretetype(
                            ConstValue::Float(*bits),
                            LowLevelType::Float,
                        )),
                    );
                }
                OpKind::ConstRefNull => {
                    map.insert(result, Hlvalue::Constant(const_ref_gcref_constant(None)));
                }
                OpKind::ConstRefAddr(addr) => {
                    map.insert(
                        result,
                        Hlvalue::Constant(const_ref_gcref_constant(Some(*addr))),
                    );
                }
                _ => {}
            }
        }
    }
    map
}

/// Fold a `ConstRefAddr` / `ConstRefNull` opkind to its constant.
///
/// Both opkinds carry a `PyObjectRef` (object-space singleton address
/// from `static_addrs.refs`, or the `PY_NULL` null reference) — a GC
/// reference in the walker's kind system.  The constant is a
/// `GCREF`-typed `_ptr` (`Ptr(GcOpaque("GCREF"))`, the lltype of
/// erased GC references) carrying the raw address as
/// `_ptr_obj::IntCast`, so `SomePtr(GCREF)` annotation and `PtrRepr`
/// rtyping give the slot its GcRef kind.  Folding to
/// `LLAddress`/`Address` instead made every graph returning such a
/// constant diverge at the dual-gate (`legacy=GcRef, real=Signed`):
/// `llmemory.Address` is int-kinded.  `cast_int_to_ptr` is not usable
/// here — it asserts odd (tagged) integers (lltype.py:2372-2377) and
/// host addresses are even — so the `_ptr` is built directly.
fn const_ref_gcref_constant(addr: Option<i64>) -> Constant {
    use crate::translator::rtyper::lltypesystem::lltype::{_ptr, _ptr_obj, GCREF, LowLevelType};
    let LowLevelType::Ptr(gcref_t) = GCREF.clone() else {
        panic!("GCREF must be a Ptr lowleveltype");
    };
    let p = match addr {
        None | Some(0) => _ptr::new(*gcref_t, Ok(None)),
        Some(a) => _ptr::new_with_solid(*gcref_t, Ok(Some(_ptr_obj::IntCast(a))), true),
    };
    Constant::with_concretetype(ConstValue::LLPtr(Box::new(p)), GCREF.clone())
}

/// Look up the `Hlvalue` for an operand `Variable`. Surfaces a
/// fail-loud `TyperError` when the operand is undefined (every
/// referenced operand `Variable` must have been seeded by
/// [`build_value_to_variable_map`] or shadowed by
/// `build_value_to_hlvalue_map`'s const inlining).
///
/// The error message embeds the enclosing `SpaceOperation` (variant
/// name + result `Variable`) and the role of the failing argument (e.g.
/// `"lhs"`, `"rhs"`, `"base"`, `"index"`, `"value"`, `"operand"`,
/// `"args[i]"`, `"result"`) so per-graph diagnosis can locate the
/// broken op without re-traversing the graph. The required substring
/// `"undefined operand"` is preserved verbatim so the dual
/// gate's `is_known_unported` predicate (`cutover.rs:441`) keeps
/// matching this category.
fn lookup_operand(
    value_map: &HashMap<Variable, Hlvalue>,
    operand: &Variable,
    op: &SpaceOperation,
    arg_role: &str,
) -> Result<Hlvalue, TyperError> {
    value_map.get(operand).cloned().ok_or_else(|| {
        let result_label = match op.result.as_ref() {
            Some(var) => format!("Some({var:?})"),
            None => "None".to_string(),
        };
        TyperError::message(format!(
            "translate_op: undefined operand {operand:?} as {arg_role} of {opkind} \
             (result {result_label}) — adapter invariant broken (every referenced \
             operand must be defined as a block inputarg or op result)",
            opkind = opkind_variant_name(&op.kind),
        ))
    })
}

/// Resolve the `Hlvalue` result slot for a legacy op. When the op has
/// no result (`Option::None`), allocate a fresh anonymous Variable per
/// RPython convention (every `SpaceOperation.result` slot is non-None
/// upstream — model.py:432-438; void-result ops use a throwaway
/// `Variable()`).
fn resolve_result_hlvalue(
    op: &SpaceOperation,
    value_map: &HashMap<Variable, Hlvalue>,
) -> Result<Hlvalue, TyperError> {
    match op.result.as_ref() {
        Some(var) => lookup_operand(value_map, var, op, "result"),
        None => Ok(Hlvalue::Variable(Variable::new())),
    }
}

/// Map a pyre-frontend unary op name (`front::mir::unary_op_label`)
/// onto the RPython flowspace operator name
/// (`rpython/flowspace/operation.py:465-474`).
///
/// `neg` and `bool` pass through (registered upstream as
/// `add_operator('neg', 1, ..)` at line 466 and `add_operator('bool',
/// 1, ..)` at line 467).
///
/// Typed numeric / ptr / Unsigned casts have no `OpKind::UnaryOp`
/// route — the frontend routes typed casts through
/// `simple_call(<host_callable>, v)` per
/// upstream `__builtin__.int/float/bool` /
/// `lltype.cast_ptr_to_int` / `lltype.cast_int_to_ptr` /
/// `rarithmetic.intmask` / `rarithmetic.r_uint`.  Only `same_as`
/// remains on the `OpKind::UnaryOp` route, emitted by the
/// identity / source-type-unknown cast fallback and
/// dispatched by `RPythonTyper::translate_operation` to
/// `rbuiltin::rtype_same_as` (verbatim port of `rtyper.py:478-481`).
/// `same_as` is also generated by `unsimplify::split_block` Void-
/// variable recreation and the backendopt pipeline.
///
/// `not` and `deref` are the only fail-loud arms: pyre's frontend
/// eliminates both at the source (`UnOp::Not` desugar / `Deref`
/// pass-through).  Reaching either arm means a synthetic graph
/// injected the op directly.  RPython distinguishes logical `not`
/// (UNARY_NOT, lowered as `bool(operand)` + branch —
/// `flowcontext.py:531-538`) from bitwise `invert` (UNARY_INVERT —
/// `flowcontext.py:190` → `op.invert`); without static type info,
/// the adapter cannot discriminate, so fail-loud is the only safe
/// choice.
fn normalize_unary_op_name(pyre_name: &str) -> Result<String, TyperError> {
    match pyre_name {
        "neg" => Ok("neg".to_string()),
        // RPython `bool` is registered as a unary op at
        // `operation.py:467 add_operator('bool', 1, ..)` and emitted
        // by `flowcontext.py:531-538 UNARY_NOT` /
        // `:766-777 JUMP_IF_*_OR_POP` as the discriminator before a
        // `guessbool` fork.  Pyre's frontend emits
        // `OpKind::UnaryOp { op: "bool", .. }` from the `&&` / `||`
        // short-circuit desugar,
        // mirroring `build_flow.rs:1191 lower_short_circuit`.
        // Pass through unchanged.
        "bool" => Ok("bool".to_string()),
        // `invert` — PyPy `add_operator('invert', 1, .., pure=True)` at
        // `operation.py:474`, emitted by `flowcontext.py:188-191
        // UNARY_INVERT` and dispatched through
        // `RPythonTyper::translate_op`'s `"invert"` arm
        // (`rtyper.rs:2025`) into `IntegerRepr::rtype_invert`
        // (`rint.py:107-110` → `rint.rs:284`). Pyre's frontend
        // emits `OpKind::UnaryOp { op: "invert", .. }` directly for the
        // integer bitwise-complement form (the case Rust's `!42_i64`
        // denotes).  Without this arm the
        // literal-int parity path Skip-classifies in the real rtyper.
        "invert" => Ok("invert".to_string()),
        // `same_as` — RPython's internal rtyper renaming op
        // (`rtyper.py:478-481`).  Defensively kept on the unary-op
        // dispatch path so the rtyper can re-enter `translate_operation`
        // on graphs that carry `same_as` from any source: identity /
        // source-type-unknown cast lowering in the frontend,
        // `unsimplify::split_block`'s
        // Void-variable recreation (`unsimplify.rs:280`), and the
        // backendopt pipeline's block-prefix `same_as` insertion
        // (`backendopt/constfold.rs:859`, `backendopt/all.rs:615`,
        // `removenoops.rs:86`, `storesink.rs:95`).  All other typed
        // `(source, target)` casts have no unary-op route — they
        // route through `simple_call(<host_callable>, v)` per upstream
        // `__builtin__.int/float/bool` / `lltype.cast_*` /
        // `rarithmetic.intmask` / `rarithmetic.r_uint`.
        "same_as" => Ok("same_as".to_string()),
        // `str` — RPython `add_operator('str', 1, ..)` (`operation.py`),
        // dispatched at `rtyper.rs "str" => rtype_str` into the per-repr
        // `ll_str` lowering and annotated `SomeString` (`unaryop.rs str =>
        // SomeString()`).  `front::mir`'s `format!`-chain expansion
        // (`collapse_fmt_chains`) renders each `{}` placeholder argument
        // with `OpKind::UnaryOp { op: "str", .. }` — the orthodox
        // string-build lowering (`str(arg)` ++ `ll_strconcat`) in place of
        // the graph-less `fmt::rt::Argument::new_display` chain.
        "str" => Ok("str".to_string()),
        other => Err(TyperError::missing_rtype_operation(format!(
            "normalize_unary_op_name: pyre UnaryOp `{other}` has no \
             flowspace counterpart (operation.py registers \
             `pos` / `neg` / `invert` / `bool` and the ported `str` \
             as unary ops; \
             `same_as` is rtyper's internal renaming op per \
             rtyper.py:478-481; all 13 typed cast names retired \
             across Slices A.3 / B.1 / A.4a / A.4b / A.4c — frontend \
             routes typed casts through \
             `simple_call(<host_callable>, v)` per upstream \
             `__builtin__.int/float/bool` / `lltype.cast_*` / \
             `rarithmetic.intmask` / `rarithmetic.r_uint`).  Frontend \
             must distinguish bitwise `invert` from logical `not` \
             and remove `deref` before reaching the rtyper."
        ))),
    }
}

/// Map a pyre-frontend binary op name (`front::mir::binop_label`)
/// onto the RPython flowspace operator name
/// (`rpython/flowspace/operation.py:485-507 add_operator(...)`).
///
/// Rust-side identifiers (`bitand`, `bitor`, `bitxor`, `add_assign`,
/// ...) become the trailing-underscore / `inplace_*` forms RPython
/// registers and `RPythonTyper::translate_op_with_map`
/// (`rtyper.rs:2023-2078`) dispatches on.  Names already matching
/// RPython (`add`, `sub`, `mul`, `mod`, `lshift`, `rshift`, `lt`, ...)
/// pass through unchanged.
///
/// A bare `and` / `or` arriving here is always the *bitwise* operator:
/// `front::mir::binop_label` maps Charon `BitAnd` / `BitOr` to
/// `"and"` / `"or"` (matching the trace pipeline's blackhole handler
/// vocabulary), and rustc lowers the short-circuit `&&` / `||` forms
/// into MIR branch forks before Charon ever sees them — Python's
/// short-circuit `and`/`or` are likewise control flow upstream
/// (`operation.py:475-510` registers no such binary operator), so the
/// keyword-suffixed `and_` / `or_` names below are unambiguously the
/// bitwise registrations.
fn normalize_binop_name(pyre_name: &str) -> Result<String, TyperError> {
    let normalized = match pyre_name {
        "and" | "bitand" => "and_",
        "or" | "bitor" => "or_",
        "bitxor" => "xor",
        "add_assign" => "inplace_add",
        "sub_assign" => "inplace_sub",
        "mul_assign" => "inplace_mul",
        "div_assign" => "inplace_div",
        "mod_assign" => "inplace_mod",
        "bitand_assign" => "inplace_and",
        "bitor_assign" => "inplace_or",
        "bitxor_assign" => "inplace_xor",
        "lshift_assign" => "inplace_lshift",
        "rshift_assign" => "inplace_rshift",
        other => other,
    };
    Ok(normalized.to_string())
}

/// Translate a single legacy `model::SpaceOperation` into zero or more
/// `flowspace::SpaceOperation`s.
///
/// Returns `Ok(Vec::new())` when the op is **fully consumed by other
/// adapter infrastructure** — `OpKind::Input` (handled by block
/// topology assembly, where the result `Variable` becomes a
/// `block.inputargs` entry) and `OpKind::ConstInt` / `ConstFloat`
/// (inlined by the `value_map` const-folding path at every consuming
/// op's args site, mirrored in tests by `build_value_to_hlvalue_map`).
///
/// Returns `Err(TyperError)` for variants that are not yet lowered.
/// The error message names the specific variant so the dual-gate
/// failure cleanly identifies which unported variant it hit.
/// Format `op.result` for diagnostic messages.  Renders the result
/// `Variable`'s identity name directly through its `Display`
/// (`Variable.__repr__`, the `{_name}{_nr}` shape) so fail-loud
/// output stays readable without a backing-slot projection.
fn fmt_op_result(op: &SpaceOperation) -> String {
    match op.result.as_ref() {
        Some(var) => format!("Some({var})"),
        None => "None".to_string(),
    }
}

/// `true` iff `kind` is a 0-arg unit-variant transparent ctor
/// (`StepResult::Continue`, `LoopResult::Done`, …) that [`translate_op`]
/// pre-folds to a `Constant` and emits no `FlowspaceOp` (the unit-variant
/// guard at the top of `translate_op`).
///
/// [`op_canraise`] and [`translate_op`] consult the SAME predicate —
/// `op_canraise` is false exactly when `translate_op` emits no op.
fn is_elided_unit_variant_ctor(kind: &OpKind) -> bool {
    if let OpKind::Call {
        target: crate::model::CallTarget::SyntheticTransparentCtor { name, owner_path },
        args,
        ..
    } = kind
        && args.is_empty()
    {
        let mut segments = owner_path.clone();
        segments.push(name.clone());
        return crate::translator::rtyper::unit_variant_fold::is_synthetic_unit_variant_path(
            &segments,
        );
    }
    false
}

/// `true` iff `kind` is `front::mir`'s synthetic string-literal
/// define-op (`Call(["__str_const", <text>])`, `mir.rs:1576`).
/// Upstream flowspace carries a string literal as a bare
/// `Constant('text')` SSA value, so the pre-pass
/// ([`legacy_const_define_hlvalue`]) folds the op to that Constant and
/// [`translate_op`] emits nothing; a Constant is not an operation and
/// raises nothing, so [`op_canraise`] is false — same contract as the
/// unit-variant ctor elision above.
fn is_str_const_define(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments },
            args,
            ..
        } if args.is_empty() && segments.len() == 2 && segments[0] == "__str_const"
    )
}

/// The pure, non-raising flowspace opname a `core::<family>::<leaf>`
/// method-call bridge lowers to in [`translate_op`], or `None` when the
/// path is not such a bridge.  The single source of truth shared by
/// [`translate_op`] (which emits the op) and [`op_canraise`] (which must
/// agree the emitted op closes no exception edge), so the two tables
/// cannot drift.  `core::cmp::{min,max}` is deliberately excluded: it
/// lowers to a `simple_call(<builtin>)` that keeps the ordinary raising
/// `CallOp` classification.
fn nonraising_core_bridge_opname(segments: &[String], arg_count: usize) -> Option<&str> {
    if segments.len() < 3 || segments[0] != "core" {
        return None;
    }
    let family = segments[1].as_str();
    let leaf = segments[segments.len() - 1].as_str();
    match (family, leaf) {
        // Rich comparisons map to the like-named flowspace op; plain
        // comparisons carry an empty `canraise` (operation.py).
        ("cmp", "eq" | "ne" | "lt" | "le" | "gt" | "ge") if arg_count == 2 => Some(leaf),
        ("slice", "len") if arg_count == 1 => Some("len"),
        ("slice", "iter") if arg_count == 1 => Some("iter"),
        // `wrapping_mul` is the Rust spelling of upstream's plain `*`
        // (lltype `int_mul` wraps); plain `mul` raises nothing.
        ("num", "wrapping_mul") if arg_count == 2 => Some("mul"),
        _ => None,
    }
}

/// `true` iff `segments` is the `core::slice::<Impl>::reverse` FunctionPath
/// that `front::mir` emits for a Rust `slice.reverse()` call (which has no
/// source body to register).  Unlike the bridge ops above, `reverse` has no
/// rtyper *operator* arm, so it is routed in [`translate_op`] to the
/// `getattr` + `simple_call` *method* shape that reaches
/// `FixedSizeListRepr.rtype_method("reverse")` (`rlist.py:138-143`
/// `rtype_method_reverse` → `ll_reverse`).  Shared with [`op_canraise`],
/// which classifies the originating Call non-raising
/// (`rlist.py:142 hop.exception_cannot_occur()`).
fn is_slice_reverse_segments(segments: &[String]) -> bool {
    segments.len() == 4
        && segments[0] == "core"
        && segments[1] == "slice"
        && segments[2] == "<Impl>"
        && segments[3] == "reverse"
}

/// `Vec::push(l, item)` (Rust MIR `vec::Vec::push`) — the resizable-list
/// append. Routed to the resized `ListRepr.rtype_method("append")`
/// (`rlist.py:185`) via the `getattr(recv, "append") + simple_call` method
/// shape, exactly like [`is_slice_reverse_segments`]; the Rust method name
/// `push` maps to the RPython list method `append`.
///
/// This recognizer fires only on the `CallTarget::FunctionPath` shape, and
/// that is complete: `vec::Vec` is a foreign type with no extracted LLBC ADT,
/// so `front::mir::impl_method_owner` cannot resolve it to a classdef-bound
/// owner and returns `None`, which forces every `Vec::push` call to the
/// `[vec, Vec, push]` FunctionPath segments rather than `CallTarget::Method`.
/// The generic Method arm (which would `getattr(recv, "push")` against a list
/// that has no `push`) is therefore unreachable for it — only user-defined
/// classdef-bound receivers route through Method.
fn is_vec_push_segments(segments: &[String]) -> bool {
    segments.len() == 3 && segments[0] == "vec" && segments[1] == "Vec" && segments[2] == "push"
}

/// `Vec::extend_from_slice(l, slice)` (Rust MIR `vec::Vec::extend_from_slice`)
/// — appends every element of `slice` to the resizable list. Routed to the
/// resized `ListRepr.rtype_method("extend")` (`rlist.py:204`) via the
/// `getattr(recv, "extend") + simple_call` method shape, exactly like
/// [`is_vec_push_segments`]; the Rust method name `extend_from_slice` maps to
/// the RPython list method `extend` (whose argument is the slice, a
/// non-resized list — `list_method_extend` takes the list-list path).
fn is_vec_extend_from_slice_segments(segments: &[String]) -> bool {
    segments.len() == 3
        && segments[0] == "vec"
        && segments[1] == "Vec"
        && segments[2] == "extend_from_slice"
}

/// Test-only mirror of whether the flowspace op(s) this `OpKind`
/// lowers to carry a non-empty `canraise` (`operation.py`).
///
/// The production path is `translate_op`, which emits real flowspace
/// operations and lets their upstream-modeled `canraise` metadata drive
/// exception edges.  This predicate is retained only for tests that keep
/// the local lowering table honest: a non-raising tail op (a transparent
/// ctor, `same_as`, a pure cast / binop, getattr / setattr, a const)
/// must not be classified as canraise.
///
/// KEEP IN SYNC with [`translate_op`]'s `OpKind` -> flowspace-opname arms.
#[cfg(test)]
pub(crate) fn op_canraise(kind: &OpKind) -> bool {
    match kind {
        // getitem / setitem -> `[IndexError, KeyError, Exception]`
        // (operation.py:727-730).
        OpKind::ArrayRead { .. } | OpKind::ArrayWrite { .. } => true,
        // `InteriorField*` unfolds in `translate_op` into a chained
        // `getitem(base, index)` followed by `getattr` / `setattr`, so it
        // carries the getitem's `[IndexError, KeyError, Exception]`
        // (operation.py:727-730).  The getattr / setattr step is itself
        // non-raising, but the getitem makes the op raise.
        OpKind::InteriorFieldRead { .. } | OpKind::InteriorFieldWrite { .. } => true,
        // A transparent `Ok(x)` / `Some(x)` / `Err(e)` ctor lowers to
        // `simple_call(HostClass(qualname), x…)` (the
        // `SyntheticTransparentCtor` arm in `translate_op`).  The `?` /
        // Result / Option transparent-ctor elision is a Rust-specific
        // adaptation with no RPython counterpart, but the simple_call it
        // emits is classified by the ordinary `CallOp.canraise` rule: a
        // Constant class callable outside `__builtin__` / `exceptions`
        // raises `[Exception]` (operation.py:648-661), so a non-unit ctor
        // raises like any non-builtin call.  The sole non-raising case is
        // the 0-arg unit-variant ctor, which `translate_op` pre-folds to a
        // `Constant` and emits no op — `op_canraise` is false exactly when
        // that happens.  Matched before the general `Call` arm.
        OpKind::Call {
            target: crate::model::CallTarget::SyntheticTransparentCtor { .. },
            ..
        } => !is_elided_unit_variant_ctor(kind),
        // A string-literal define-op pre-folds to `Constant('text')`
        // and emits no op — a Constant raises nothing.  Matched before
        // the general `Call` arm, same as the unit-variant elision.
        kind if is_str_const_define(kind) => false,
        // A `hint(x, **kwds)` op (`OpKind::Hint`) lowers to a non-raising
        // `same_as(value)` (`rtyper.py:478-481` internal renaming) in
        // `translate_op` — it emits no raising op.  Matched before the
        // general `Call` arm, same as the elisions above.
        OpKind::Hint { .. } => false,
        // `core::cmp::{eq..ge}` / `core::slice::{len,iter}` /
        // `core::num::wrapping_mul` lower to pure, non-raising flowspace
        // ops (see `nonraising_core_bridge_opname`); classify the
        // originating Call the same way so a `?` tail op does not install
        // an exception edge the lowered op cannot take.  Matched before
        // the general `Call` arm.
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments },
            args,
            ..
        } if nonraising_core_bridge_opname(segments, args.len()).is_some() => false,
        // `core::slice::<Impl>::reverse` lowers (in `translate_op`) to a
        // `getattr` + `simple_call` method shape whose `rtype_method_reverse`
        // does `hop.exception_cannot_occur()` (`rlist.py:142`); classify the
        // originating Call non-raising so a `?` tail op installs no exception
        // edge the lowered op cannot take.  (`reverse` returns `()`, so it is
        // never actually a `?` operand — this keeps the table faithful.)
        // Matched before the general `Call` arm.
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments },
            ..
        } if is_slice_reverse_segments(segments) => false,
        // `[__iter_next]` (`front::iter_next`) lowers to the raising
        // flowspace `next` op (`operation.rs` `OpKind::Next.can_only_throw
        // = [StopIteration, RuntimeError]`).  Matched before the general
        // `Call` arm, mirroring [`translate_op`]'s `[__iter_next]` arm.
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments },
            args,
            ..
        } if args.len() == 1 && crate::front::iter_next::is_iter_next_segments(segments) => true,
        // simple_call -> `CallOp.canraise` is `[Exception]` for a
        // non-builtin callable (operation.py:648-661).  Constant builtin
        // callables (int / float / chr / unicode) carry the narrower
        // `builtins_exceptions` set, but a `?` operand is Result/Option-
        // typed so over-approximating those few builtins is inert.  The
        // jtransform-generated call variants cannot appear as a front-end
        // `Expr::Try` tail op (they are produced by a later pass); listing
        // them only keeps the predicate faithful to `CallOp`.
        OpKind::Call { .. }
        | OpKind::IndirectCall { .. }
        | OpKind::CallElidable { .. }
        | OpKind::CallResidual { .. }
        | OpKind::CallMayForce { .. } => true,
        // Plain binops: div / mod / divmod / truediv / floordiv / pow carry
        // ZeroDivisionError, and pow / lshift / rshift carry ValueError,
        // even without the `_ovf` suffix (operation.py:751-756); plain
        // add / sub / mul / cmp / bitops are `[]`.  Compound-assign names
        // (`*_assign`) are seen here BEFORE `normalize_binop_name` maps
        // them to `inplace_*`, so they are classified by their `inplace_*`
        // canraise: `inplace_div/mod/lshift/rshift` keep the plain
        // ZeroDivisionError/ValueError, and `inplace_add/sub/mul` carry
        // OverflowError (they have no `_ovf` variant) (operation.py:751-756);
        // `inplace_and/or/xor` are `[]`.  The Rust front end emits the
        // plain div / mod / lshift / rshift and the `*_assign` family; the
        // other plain names are faithful but never produced.
        OpKind::BinOp { op, .. } => matches!(
            op.as_str(),
            "div"
                | "mod"
                | "divmod"
                | "truediv"
                | "floordiv"
                | "pow"
                | "lshift"
                | "rshift"
                // The `_ovf` arithmetic twins carry `[OverflowError]`
                // (operation.py:760-761 `_add_except_ovf`;
                // `OpKind::{Add,Sub,Mul}Ovf.canraise()`).  The front-end
                // emits them only at a `LastException` block whose
                // `raising_op` is the `_ovf` op (`front::checked_arith`),
                // so they must classify as raising.
                | "add_ovf"
                | "sub_ovf"
                | "mul_ovf"
                | "add_assign"
                | "sub_assign"
                | "mul_assign"
                | "div_assign"
                | "mod_assign"
                | "lshift_assign"
                | "rshift_assign"
        ),
        // getattr / setattr / neg / not / type / same_as / Const* /
        // Guard* / VableForce / Input / Abort (JIT-abort marker) -> all
        // `canraise = []`.
        _ => false,
    }
}

pub fn translate_op(
    op: &SpaceOperation,
    value_map: &HashMap<Variable, Hlvalue>,
    // The call registry is consulted by the `OpKind::Call::FunctionPath`
    // arm to resolve a registered `(HostObject, FunctionDesc)` pair
    // and emit a flowspace `simple_call` (`operation.py:152`,
    // `rpbc.rs:1621 FunctionRepr::rtype_simple_call`).  Empty registry
    // callsites surface a distinct fail-loud message; producers
    // must pre-register every reachable FunctionPath.
    call_registry: &crate::translator::rtyper::pyre_call_registry::PyreCallRegistry,
) -> Result<Vec<FlowspaceOp>, TyperError> {
    // Unit-variant ctors (`StepResult::Continue`, `LoopResult::Done`, …)
    // pre-fold to `Hlvalue::Constant(HostObject(prebuilt_instance))` in the
    // pre-pass (see `legacy_const_define_hlvalue`).  Skip translation here
    // so they do not double-emit as `simple_call(HostClass(qualname))` —
    // matches the `ConstInt`/`ConstBool`/`ConstFloat` pattern below (the
    // pre-pass owns the slot's `Hlvalue::Constant`, translate_op emits no
    // FlowspaceOp).  `op_canraise` consults the same predicate, so it
    // reports these — and only these — transparent ctors as non-raising.
    if is_elided_unit_variant_ctor(&op.kind) {
        return Ok(Vec::new());
    }
    // String-literal define-ops pre-fold to `Constant('text')` the same
    // way (see `is_str_const_define`); the pre-pass owns the slot's
    // `Hlvalue::Constant`, translate_op emits no FlowspaceOp.
    if is_str_const_define(&op.kind) {
        return Ok(Vec::new());
    }
    match &op.kind {
        // ─── Skipped: fully consumed by other adapter infrastructure ───
        OpKind::Input { .. } => Ok(Vec::new()),
        OpKind::ConstInt(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstFloat(_)
        | OpKind::ConstRefNull
        | OpKind::ConstRefAddr(_) => Ok(Vec::new()),
        // ─── Skipped: pyre JIT trace markers without a flowspace peer ───
        // `GuardTrue` / `GuardFalse` / `GuardValue` are JIT-side
        // assertions emitted by pyre's tracer — they constrain the
        // runtime value of an existing SSA operand and produce no new
        // SSA result.  `VableForce` is a virtualizable-flush hint
        // (`hint_force_virtualizable`, no operands, no result).
        // RPython's flowspace abstract interpreter does not have any
        // of these at the high-level rtyper input
        // (`rpython/flowspace/operation.py:475-510`); the equivalent
        // checks appear later in `pyjitpl` / `codewriter` after the
        // rtyper has already lowered to lltype.  For the type
        // resolution pass driven by `specialize_legacy_graph` they
        // are pure no-ops: skipping them preserves the SSA chain
        // (any operand they read is defined elsewhere; the absence
        // of a result means no consumer is left unsatisfied).
        OpKind::GuardTrue { .. }
        | OpKind::GuardFalse { .. }
        | OpKind::GuardValue { .. }
        | OpKind::VableForce { .. } => Ok(Vec::new()),

        // `hint(x, **kwds)` (`OpKind::Hint`) is an identity outside the JIT:
        // the flowspace oracle types its result as `same_as(value)` — the
        // `rtyper.py:478-481` internal-renaming op — exactly as RPython's
        // `hint` op is a no-op in the genc/non-JIT build, while the JIT
        // codewriter (`jtransform::rewrite_op_hint`) is what rewrites it to
        // the `<kind>_guard_value` family.  `jit::promote(x)` and
        // `#[elidable_promote]`'s `hint_promote_or_string` both land here.
        OpKind::Hint { value, .. } => {
            let value_hl = lookup_operand(value_map, value, op, "value")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new("same_as", vec![value_hl], result)])
        }

        // ─── pyre-only `OpKind::Abort` marker ───
        // Front-end `lower_expr::stop_unsupported` / `continue_with_unknown`
        // emit this when the surface Rust DSL hits an unsupported
        // expression form (unsupported lit, ForLoop, Closure, Macro,
        // …).  RPython upstream raises `FlowingError`
        // (`flowspace/flowcontext.py:258,417`) and drops the function
        // before annotator/rtyper see it, so there is no upstream
        // analogue.  A `SomeInstance(None) -> Ptr -> GcRef`
        // synthesis is not used because the `classdef=None` result
        // trips downstream `find_attribute` lookups.
        //
        // Post-`FORCE_ATTRIBUTES_INTO_CLASSES` pre-population (struct
        // field projection by `register_struct_fields`) impl-method
        // `self` narrows to a populated
        // ClassDef, so the original classdef:None cascade is no longer
        // triggered by the receiver projection.  Abort's own
        // result_var is still un-narrowed, but every front-end
        // emit-site falls into one of two shapes:
        //
        //   (a) `stop_unsupported` — pushes Abort and returns
        //       `Err(FlowingError::Unsupported)`; the parent `?`
        //       ladder aborts the body before any operand reads the
        //       result_var.  No downstream consumer ⇒ skipping is
        //       safe.
        //
        //   (b) `continue_with_unknown` — pushes Abort and returns
        //       the result_var as a `Lowered.value`.  Callers may
        //       consume it; for those, the absent translate output
        //       leaves the result_var unmapped in `value_to_var`,
        //       and the first consumer surfaces a fail-loud
        //       "undefined operand" message that
        //       `is_known_unported` classifies as Skip — same
        //       outcome as the prior "post-rtyper jtransform
        //       variant" Skip, just at a more localised site.
        //
        // Each emit-site is retired by lowering the
        // specific expression form properly (`ConstStr`, `Range`,
        // `Closure`, etc.); until each is lowered, this
        // arm absorbs the placeholder silently so the dual-gate
        // doesn't have to round-trip through a TyperError just to
        // re-classify as Skip.
        OpKind::Abort { .. } => Ok(Vec::new()),

        // ─── `newtuple` — RPython `BUILD_TUPLE` / `space.newtuple` ───
        // `PureOperation` (`operation.py:542-548`).  Each `args[i]`
        // Variable is routed through `value_map` so the legacy
        // flowspace SpaceOperation references the same Hlvalue
        // identities the graph validator (`checkgraph`) tracks; using
        // raw model-side Variables here would trip
        // "variable used before definition" when an earlier op
        // remapped its result to a different legacy Variable.
        OpKind::NewTuple { args } => {
            let mut hl_args: Vec<Hlvalue> = Vec::with_capacity(args.len());
            for (i, var) in args.iter().enumerate() {
                let role = format!("arg{i}");
                hl_args.push(lookup_operand(value_map, var, op, &role)?);
            }
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new("newtuple", hl_args, result)])
        }

        // ─── `newlist` — RPython `BUILD_LIST` / `space.newlist` ───
        // `PureOperation`.  Same operand-routing discipline as
        // `newtuple`: each `args[i]` Variable goes through `value_map`
        // so the legacy SpaceOperation references the Hlvalue identities
        // `checkgraph` tracks.
        OpKind::NewList { args } => {
            let mut hl_args: Vec<Hlvalue> = Vec::with_capacity(args.len());
            for (i, var) in args.iter().enumerate() {
                let role = format!("arg{i}");
                hl_args.push(lookup_operand(value_map, var, op, &role)?);
            }
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new("newlist", hl_args, result)])
        }

        // ─── `NewWithVtable` — boxing GC allocation (`fuse_boxing_alloc`) ───
        // The model-graph op carries the boxing struct leaf `owner` and flows
        // straight to the codewriter/assembler (`new_with_vtable`).  For the
        // ephemeral annotation / rtype type-oracle it mirrors the
        // `SyntheticTransparentCtor` struct path (flowspace_adapter.rs:1705): a
        // zero-arg `simple_call` against the interned class host annotates the
        // result as a fresh `SomeInstance(owner)`.  `getuniqueclassdef_for_
        // struct_root` first forces the struct's field rows to be projected so
        // the trailing payload `FieldWrite(result, …)` resolves.
        OpKind::NewWithVtable { owner, .. } => {
            let bk = call_registry.bookkeeper();
            bk.getuniqueclassdef_for_struct_root(owner).map_err(|e| {
                TyperError::message(format!(
                    "translate_op: NewWithVtable owner {owner:?} is not a known struct root: {e}"
                ))
            })?;
            let host = bk.intern_class_by_qualname(owner);
            let callable = Hlvalue::Constant(Constant::new(ConstValue::HostObject(host)));
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "simple_call",
                vec![callable],
                result,
            )])
        }

        // ─── `LoadStatic` — single-segment static lookup ─
        // Pyre-only marker emitted by the frontend when a path
        // expression resolves to a crate-level `static` declaration
        // (SHOUTY_CASE constant like `GC_WEAKREF_BOX_TYPE`).  RPython
        // peer: `LOAD_GLOBAL` (`flowspace/flowcontext.py:1098`)
        // resolves the name lookup to a `Constant(value)` directly
        // — no SpaceOperation is emitted, and the bound `Variable`
        // *is* the graph-level definition.  Pyre always emits an op
        // here so cross-block reads have a defined producer (the
        // `checkgraph` defining-var set requires every operand to
        // trace to an op result or `Block.inputargs`).
        //
        // When `extract_static_decls` could fold the static's
        // RHS to a `ConstValue` (`bool` / integer / float / string
        // literals + `const { LIT }` block wrapper), the adapter emits
        // `same_as(Constant(value))` — the concrete `Constant` shape PyPy
        // `LOAD_GLOBAL` pushes.  Unresolved RHS values are rejected here:
        // allowing a path-string sentinel to survive would create a
        // `same_as/*` JitCode opcode, but RPython's blackhole only has
        // `int_same_as/i>i` for test hints.
        OpKind::LoadStatic {
            segments, value, ..
        } => {
            let Some(v) = value else {
                return Err(TyperError::message(format!(
                    "translate_op: unresolved LoadStatic {segments:?} has no RPython \
                     JitCode counterpart; fold the static to a Constant before rtyper"
                )));
            };
            let constant = Hlvalue::Constant(Constant::new(v.clone()));
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new("same_as", vec![constant], result)])
        }

        // ─── Pre-rtyper opname normalization ───
        // `front::mir::binop_label` emits Rust-side
        // names (`bitand`, `bitor`, `bitxor`, `add_assign`, ...).
        // RPython flowspace registers operators via
        // `add_operator('and_', 2, ...)` etc.
        // (`rpython/flowspace/operation.py:485-507`): `and_`, `or_`,
        // `xor`, `inplace_add`, `inplace_sub`, ...  Translate the
        // pyre-side name to its RPython counterpart so the rtyper's
        // `translate_op` arm matching (`rtyper.rs:2023-2078`) finds
        // the proper `pair_rtype_*` dispatch.
        OpKind::BinOp {
            op: opname,
            lhs,
            rhs,
            ..
        } => {
            let l = lookup_operand(value_map, lhs, op, "lhs")?;
            let r = lookup_operand(value_map, rhs, op, "rhs")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                normalize_binop_name(opname)?,
                vec![l, r],
                result,
            )])
        }

        // ─── Pre-rtyper opname normalization for unary ops ───
        // `front::mir::unary_op_label` emits Rust-side
        // names (`neg`, `not`, `deref`).  RPython flowspace registers
        // unary operators via `add_operator('neg', 1, ..)` /
        // `add_operator('invert', 1, ..)` /
        // `add_operator('pos', 1, ..)` etc.
        // (`rpython/flowspace/operation.py:465-474`).  Translate the
        // pyre-side name to its RPython counterpart so the rtyper's
        // unary dispatch (`rtyper.rs:2023-2078 translate_op_*`) finds
        // the proper `unary_rtype_*` arm.
        OpKind::UnaryOp {
            op: opname,
            operand,
            ..
        } => {
            let v = lookup_operand(value_map, operand, op, "operand")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                normalize_unary_op_name(opname)?,
                vec![v],
                result,
            )])
        }

        // ─── `isinstance` — RPython `space.isinstance(obj, cls)` ───
        // `flowspace/operation.py:259 isinstance → OpKind::IsInstance`.
        // Emitted pre-rtyper by `front/ast.rs` at `TupleStruct` /
        // composite match-cascade payload sites where a unit-variant
        // ptr_eq does not suffice. The rtyper dispatches `"isinstance"`
        // at `rtyper.rs:2035 translate_unary_operation` →
        // `InstanceRepr::rtype_isinstance`, which mints either a
        // per-class `ll_isinstance_const_*` helper (Constant
        // `class_carrier`) or the generic `ll_isinstance` helper
        // (Variable `class_carrier`).
        OpKind::IsInstance {
            obj, class_carrier, ..
        } => {
            let obj_hl = lookup_operand(value_map, obj, op, "obj")?;
            let cls_hl = lookup_operand(value_map, class_carrier, op, "class_carrier")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "isinstance",
                vec![obj_hl, cls_hl],
                result,
            )])
        }

        // ─── FieldRead / FieldWrite ports ───
        // RPython `flowspace/operation.py:617 GetAttr.opname = 'getattr'`
        // and `setattr` (operation.py: same module). The high-level
        // attribute-access op carries the field name as a
        // `ConstValue::ByteStr` (Python 2 `str`), matching the rtyper's
        // `rtype_getattr` / `rtype_setattr` dispatch
        // (`rtyper.rs:2013-2014`). InstanceRepr later lowers the
        // `getattr`/`setattr` op into a `getfield_*` / `setfield_*`
        // bytecode keyed on the field's lltype kind.
        OpKind::FieldRead { base, field, .. } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "getattr",
                vec![
                    base_hl,
                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                ],
                result,
            )])
        }
        OpKind::FieldWrite {
            base, field, value, ..
        } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            // The stored value is an `AbstractValue` — a `Variable`
            // (resolve through the SSA map) or an inline `Constant`
            // (carries its own value), mirroring `Hlvalue` directly.
            let value_hl = match value {
                crate::model::LinkArg::Value(var) => lookup_operand(value_map, var, op, "value")?,
                crate::model::LinkArg::Const(c) => Hlvalue::Constant(c.clone()),
            };
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "setattr",
                vec![
                    base_hl,
                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                    value_hl,
                ],
                result,
            )])
        }

        // ─── ArrayRead / ArrayWrite ports ───
        // RPython `flowspace/operation.py: GetItem.opname = 'getitem'`
        // and `setitem`. The base[index] form maps directly to
        // `getitem(base, index)` / `setitem(base, index, value)`; the
        // rtyper's `rtype_getitem` / `rtype_setitem` later route through
        // ListRepr / TupleRepr / Fixed-array repr based on the receiver's
        // resolved type, lowering to `getarrayitem_gc_*` /
        // `setarrayitem_gc_*` bytecodes.
        OpKind::ArrayRead { base, index, .. } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let index_hl = lookup_operand(value_map, index, op, "index")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "getitem",
                vec![base_hl, index_hl],
                result,
            )])
        }
        OpKind::ArrayWrite {
            base, index, value, ..
        } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let index_hl = lookup_operand(value_map, index, op, "index")?;
            // The stored value is an `AbstractValue` — a `Variable`
            // (resolve through the SSA map) or an inline `Constant`.
            let value_hl = match value {
                crate::model::LinkArg::Value(var) => lookup_operand(value_map, var, op, "value")?,
                crate::model::LinkArg::Const(c) => Hlvalue::Constant(c.clone()),
            };
            let result = resolve_result_hlvalue(op, value_map)?;
            Ok(vec![FlowspaceOp::new(
                "setitem",
                vec![base_hl, index_hl, value_hl],
                result,
            )])
        }

        // ─── InteriorFieldRead / InteriorFieldWrite ports ───
        // RPython `effectinfo.py:313-340` notes that `getinteriorfield_gc`
        // implicitly carries both a `readarray` and a `readinteriorfield`
        // effect — the array-of-structs pattern is fundamentally a
        // chained `getitem(base, index) -> elem` followed by
        // `getattr(elem, field_name)` (or `setattr` for writes). Pyre's
        // legacy IR collapses these into a single `InteriorField*` op
        // for direct lowering convenience, but the rtyper sees the
        // chained form, so unfold here into two flowspace ops with an
        // intermediate `Variable` carrying the array element.
        OpKind::InteriorFieldRead {
            base, index, field, ..
        } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let index_hl = lookup_operand(value_map, index, op, "index")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            let elem_var = Hlvalue::Variable(Variable::new());
            Ok(vec![
                FlowspaceOp::new("getitem", vec![base_hl, index_hl], elem_var.clone()),
                FlowspaceOp::new(
                    "getattr",
                    vec![
                        elem_var,
                        Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                    ],
                    result,
                ),
            ])
        }
        OpKind::InteriorFieldWrite {
            base,
            index,
            field,
            value,
            ..
        } => {
            let base_hl = lookup_operand(value_map, base, op, "base")?;
            let index_hl = lookup_operand(value_map, index, op, "index")?;
            let value_hl = lookup_operand(value_map, value, op, "value")?;
            let result = resolve_result_hlvalue(op, value_map)?;
            let elem_var = Hlvalue::Variable(Variable::new());
            Ok(vec![
                FlowspaceOp::new("getitem", vec![base_hl, index_hl], elem_var.clone()),
                FlowspaceOp::new(
                    "setattr",
                    vec![
                        elem_var,
                        Hlvalue::Constant(Constant::new(ConstValue::byte_str(&field.name))),
                        value_hl,
                    ],
                    result,
                ),
            ])
        }

        // ─── Call port (CallTarget per variant) ───
        // RPython `flowspace/operation.py:663 SimpleCall.opname =
        // 'simple_call'`. The first arg is a Constant wrapping the
        // callable (or a Variable carrying a runtime function pointer).
        // Each `CallTarget` variant maps to a different shape:
        //
        //   FunctionPath { segments }     — direct call to `path::func`.
        //                                    Wrap the joined qualname in
        //                                    a `HostObject::new_opaque(...)`
        //                                    Constant; rtyper's
        //                                    `rtype_simple_call` dispatches
        //                                    on the callable's resolved
        //                                    repr (PBCRepr / etc.).
        //   SyntheticTransparentCtor      — Rust struct ctor `Class { .. }`.
        //                                    Same shape as FunctionPath:
        //                                    opaque host wrapping the type
        //                                    qualname; the rtyper-equivalent
        //                                    layer routes the ctor to its
        //                                    InstanceRepr.
        //   Method { name, .. }           — `obj.method(args)` — chains
        //                                    `getattr(args[0], name) → meth`
        //                                    into `simple_call(meth, args[1..])`,
        //                                    mirroring `flowspace/
        //                                    flowcontext.py:LOAD_ATTR +
        //                                    CALL_FUNCTION` shape.
        //   Indirect { trait_root, name } — `dyn Trait` dispatch. Pyre's
        //                                    `rclass.rs` rewrites this into
        //                                    a `VtableMethodPtr` followed
        //                                    by an `IndirectCall`; reaching
        //                                    the adapter means rclass.rs
        //                                    didn't run, so fail-loud.
        //   UnsupportedExpr               — frontend coverage gap; fail-loud
        //                                    surfaces the missing
        //                                    `front::mir` arm.
        OpKind::Call { target, args, .. } => {
            use crate::model::CallTarget;
            let arg_hls: Result<Vec<_>, _> = args
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let role = format!("args[{i}]");
                    lookup_operand(value_map, v, op, &role)
                })
                .collect();
            let arg_hls = arg_hls?;
            let result = resolve_result_hlvalue(op, value_map)?;
            match target {
                // `FunctionPath` resolves through
                // `PyreCallRegistry`, returning the registry entry's
                // `HostObject::UserFunction` instead of an opaque
                // wrapper. The rtyper's `pair_simple_call` then
                // short-circuits on `bookkeeper.descs` (pre-populated
                // by the registry) and routes through
                // `FunctionRepr::call(hop)` (`rpbc.py:199`).
                CallTarget::FunctionPath { segments } => {
                    // `core` method spellings of upstream operations:
                    // pyre source writes `a.min(b)` /
                    // `a != b`-via-`PartialEq::ne` / `v.len()` /
                    // `x.wrapping_mul(y)` where upstream Python
                    // writes `min(a, b)` / the flowspace ops `ne` /
                    // `len` / `mul` (flowspace/operation.py tables;
                    // lltype `int_mul` wraps, so `wrapping_mul` is
                    // the Rust spelling of upstream's plain `*` —
                    // e.g. the `x * 1000003` green-key hash).
                    // front::mir records the monomorphized core path
                    // when the call survives as a trait/inherent
                    // method call (primitive BinOps lower to
                    // `OpKind::BinOp` and never reach here); map the
                    // path back to the upstream spelling so the
                    // chain reaches unaryop.py / binaryop.py /
                    // rbuiltin.py instead of failing FunctionPath
                    // resolution.
                    // Pure bridges (cmp comparisons / slice len|iter /
                    // num wrapping_mul) → a like-named non-raising
                    // flowspace op.  `nonraising_core_bridge_opname` is
                    // the shared table `op_canraise` mirrors.
                    if let Some(opname) = nonraising_core_bridge_opname(segments, arg_hls.len()) {
                        return Ok(vec![FlowspaceOp::new(opname, arg_hls, result)]);
                    }
                    // `[__iter_next]` (`front::iter_next`) → the raising
                    // flowspace `next` op (`operation.rs` `OpKind::Next`,
                    // `can_only_throw = [StopIteration, RuntimeError]`).  The
                    // block's `LastException` exits — set by the front-end
                    // `next`-diamond rewrite — carry the exception edges, so
                    // no edge-building happens here (unlike a `?`-tail op).
                    if crate::front::iter_next::is_iter_next_segments(segments)
                        && arg_hls.len() == 1
                    {
                        return Ok(vec![FlowspaceOp::new("next", arg_hls, result)]);
                    }
                    // `min`/`max` are the exception: they lower to a
                    // `simple_call(<builtin>)` (raising like any builtin
                    // call), so they stay out of the pure-bridge table.
                    if segments.len() >= 3
                        && segments[0] == "core"
                        && segments[1] == "cmp"
                        && arg_hls.len() == 2
                    {
                        let leaf = segments[segments.len() - 1].as_str();
                        if matches!(leaf, "min" | "max") {
                            let builtin = HOST_ENV.lookup_builtin(leaf).ok_or_else(|| {
                                TyperError::message(format!(
                                    "builtin `{leaf}` missing from HOST_ENV"
                                ))
                            })?;
                            let callable =
                                Hlvalue::Constant(Constant::new(ConstValue::HostObject(builtin)));
                            let mut call_args = Vec::with_capacity(arg_hls.len() + 1);
                            call_args.push(callable);
                            call_args.extend(arg_hls);
                            return Ok(vec![FlowspaceOp::new("simple_call", call_args, result)]);
                        }
                    }
                    // `__cast_pointer/<Root>` marker (`front::mir`
                    // `cast_pointer_marker_op`) — pyre's carrier for the
                    // upstream `cast_pointer(PTRTYPE, ptr)` downcast
                    // (lltype.py:964-968).  Same path-encoded-constant
                    // reconstruction as the `simple_call(<exc class>)`
                    // raise marker (Branch 3c below): rebuild the 2-arg
                    // upstream shape with the target class as the
                    // constant first argument.  The class is interned by
                    // qualname so every cast site shares one `HostObject`
                    // Arc (`getdesc` dedups on Arc identity — fresh Arcs
                    // would mint one ClassDesc per cast site).
                    if segments.len() == 2 && segments[0] == "__cast_pointer" && arg_hls.len() == 1
                    {
                        let callable_host = HOST_ENV
                            .import_module("rpython.rtyper.lltypesystem.lltype")
                            .and_then(|m| m.module_get("cast_pointer"))
                            .ok_or_else(|| {
                                TyperError::message(
                                    "HOST_ENV lltype module must expose cast_pointer".to_string(),
                                )
                            })?;
                        let class_host = call_registry
                            .bookkeeper()
                            .intern_class_by_qualname(&segments[1]);
                        let mut call_args = Vec::with_capacity(arg_hls.len() + 2);
                        call_args.push(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                            callable_host,
                        ))));
                        call_args.push(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                            class_host,
                        ))));
                        call_args.extend(arg_hls);
                        return Ok(vec![FlowspaceOp::new("simple_call", call_args, result)]);
                    }
                    // `__pyre_cast_instance` — front-end pointer-downcast
                    // narrow (#298).  `front::mir` emits a synthetic
                    // `Call(["__pyre_cast_instance", <root>], [operand])`
                    // for `obj as *const RegisteredStruct`, stashing the
                    // target struct root in `segments[1]` because the
                    // `Vec<Variable>` arg carrier cannot hold a `Constant`
                    // (same carrier limitation as the Branch-3c
                    // `simple_call(<exc class>)` reconstruction below).
                    // Reconstruct it here as `simple_call(callable,
                    // operand, Constant(root))`: the analyzer reads the
                    // trailing `ByteStr` root to type the result
                    // `SomeInstance(root)`, and the typer lowers the call
                    // to a `cast_pointer`.  The callable resolves through
                    // the `__pyre_cast_instance` HOST_ENV singleton so its
                    // Arc identity matches the `BUILTIN_TYPER` key.
                    if segments.len() == 2 && segments[0] == "__pyre_cast_instance" {
                        if arg_hls.len() != 1 {
                            return Err(TyperError::message(format!(
                                "__pyre_cast_instance requires exactly one operand, got {}",
                                arg_hls.len()
                            )));
                        }
                        let callable_host = HOST_ENV
                            .lookup_builtin("__pyre_cast_instance")
                            .ok_or_else(|| {
                                TyperError::message(
                                    "__pyre_cast_instance missing from HOST_ENV bootstrap"
                                        .to_string(),
                                )
                            })?;
                        let callable =
                            Hlvalue::Constant(Constant::new(ConstValue::HostObject(callable_host)));
                        let mut call_args = Vec::with_capacity(arg_hls.len() + 2);
                        call_args.push(callable);
                        call_args.extend(arg_hls);
                        call_args.push(Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                            &segments[1],
                        ))));
                        return Ok(vec![FlowspaceOp::new("simple_call", call_args, result)]);
                    }
                    // The `len` operation in its three spellings: the
                    // `__len` synthetic `front/mir.rs` lowers `Rvalue::Len`
                    // (and the `<str>::is_empty` decomposition) to; the
                    // slice-receiver `core::slice::<Impl>::len`; and the
                    // `<str>::len` method.  Rust lowers `slice.len()` /
                    // `s.len()` to MIR calls to those intrinsics, which
                    // have no source body to register.  Route all three to
                    // the rtyper's `len` operation (`rtyper.rs:2016 "len"
                    // arm` → `Repr.rtype_len`), the same dispatch upstream
                    // `op.len(v)` reaches via `unaryop.py:867-870`.  The
                    // rtyper dispatches on the receiver repr: a slice maps
                    // to `SomeList` (`ll_length`), a `&str` to `SomeString`
                    // (`StringRepr.rtype_len` → `ll_strlen`).  The helper
                    // is registered as an opname graph and lowered to the
                    // `strlen`/`arraylen_gc` blackhole op
                    // (`codewriter::jtransform_opname::lower_graph`), so
                    // these are real `len` ops, not symbolic residuals.
                    let is_len_op = (segments.len() == 1 && segments[0] == "__len")
                        || (segments.len() == 4
                            && segments[0] == "core"
                            && segments[1] == "slice"
                            && segments[2] == "<Impl>"
                            && segments[3] == "len")
                        || (segments.len() >= 3
                            && segments[segments.len() - 3] == "str"
                            && segments[segments.len() - 2] == "<Impl>"
                            && segments[segments.len() - 1] == "len");
                    if is_len_op {
                        if arg_hls.len() != 1 {
                            return Err(TyperError::message(format!(
                                "len operation requires exactly one receiver arg, got {}",
                                arg_hls.len()
                            )));
                        }
                        let mut iter = arg_hls.into_iter();
                        let arg = iter.next().ok_or_else(|| {
                            TyperError::message("len operation requires a receiver arg".to_string())
                        })?;
                        return Ok(vec![FlowspaceOp::new("len", vec![arg], result)]);
                    }
                    // `slice.reverse()` lowers (in Rust MIR) to a call to
                    // `core::slice::<Impl>::reverse`, arriving as the
                    // FunctionPath with no source body to register.  Unlike
                    // `len`/`iter`, `reverse` has no rtyper operator arm, so
                    // emit the *method* shape — `getattr(recv, "reverse")` +
                    // `simple_call(bound_method)` — that the annotator's
                    // `find_method("reverse")` (`unaryop.rs`) and
                    // `BuiltinMethodRepr::rtype_simple_call` →
                    // `rtype_method("reverse")` consume, identical to the
                    // `CallTarget::Method` arm below.  `rtype_method_reverse`
                    // (`rlist.py:138-143`) `gendirectcall`s `ll_reverse`
                    // (`rlist.py:677-686`).
                    if is_slice_reverse_segments(segments) {
                        if arg_hls.len() != 1 {
                            return Err(TyperError::message(format!(
                                "slice.reverse requires exactly one receiver arg, got {}",
                                arg_hls.len()
                            )));
                        }
                        let mut iter = arg_hls.into_iter();
                        let receiver = iter.next().ok_or_else(|| {
                            TyperError::message("slice.reverse requires a receiver arg".to_string())
                        })?;
                        let bound_method = Hlvalue::Variable(Variable::new());
                        return Ok(vec![
                            FlowspaceOp::new(
                                "getattr",
                                vec![
                                    receiver,
                                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                                        "reverse",
                                    ))),
                                ],
                                bound_method.clone(),
                            ),
                            FlowspaceOp::new("simple_call", vec![bound_method], result),
                        ]);
                    }
                    // `Vec::push(recv, item)` lowers (in Rust MIR) to a call
                    // to `vec::Vec::push`. Emit the *method* shape
                    // `getattr(recv, "append") + simple_call(bound_method,
                    // item)` that the annotator's `find_method("append")`
                    // (`unaryop.rs`) and `BuiltinMethodRepr::rtype_simple_call`
                    // → `ListRepr::rtype_method("append")` consume —
                    // identical to the `slice.reverse` arm above, but the
                    // bound method takes the appended `item` arg.
                    if is_vec_push_segments(segments) {
                        if arg_hls.len() != 2 {
                            return Err(TyperError::message(format!(
                                "Vec::push requires exactly two args (receiver, item), got {}",
                                arg_hls.len()
                            )));
                        }
                        let mut iter = arg_hls.into_iter();
                        let receiver = iter.next().ok_or_else(|| {
                            TyperError::message("Vec::push requires a receiver arg".to_string())
                        })?;
                        let item = iter.next().ok_or_else(|| {
                            TyperError::message("Vec::push requires an item arg".to_string())
                        })?;
                        let bound_method = Hlvalue::Variable(Variable::new());
                        return Ok(vec![
                            FlowspaceOp::new(
                                "getattr",
                                vec![
                                    receiver,
                                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                                        "append",
                                    ))),
                                ],
                                bound_method.clone(),
                            ),
                            FlowspaceOp::new("simple_call", vec![bound_method, item], result),
                        ]);
                    }
                    // `Vec::extend_from_slice(recv, slice)` — same method
                    // shape as the `Vec::push` arm, but the bound method is
                    // `extend` and its arg is the source slice (a non-resized
                    // list). Reaches `ListRepr::rtype_method("extend")`.
                    if is_vec_extend_from_slice_segments(segments) {
                        if arg_hls.len() != 2 {
                            return Err(TyperError::message(format!(
                                "Vec::extend_from_slice requires exactly two args \
                                 (receiver, slice), got {}",
                                arg_hls.len()
                            )));
                        }
                        let mut iter = arg_hls.into_iter();
                        let receiver = iter.next().ok_or_else(|| {
                            TyperError::message(
                                "Vec::extend_from_slice requires a receiver arg".to_string(),
                            )
                        })?;
                        let source = iter.next().ok_or_else(|| {
                            TyperError::message(
                                "Vec::extend_from_slice requires a source-slice arg".to_string(),
                            )
                        })?;
                        let bound_method = Hlvalue::Variable(Variable::new());
                        return Ok(vec![
                            FlowspaceOp::new(
                                "getattr",
                                vec![
                                    receiver,
                                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                                        "extend",
                                    ))),
                                ],
                                bound_method.clone(),
                            ),
                            FlowspaceOp::new("simple_call", vec![bound_method, source], result),
                        ]);
                    }
                    let key =
                        crate::translator::rtyper::pyre_call_registry::FunctionPathKey::from_segments(
                            segments.iter().cloned(),
                        );
                    // Three resolution layers, matching upstream's three
                    // dispatch shapes for a dotted call site:
                    //
                    // 1. `PyreCallRegistry` — user functions registered
                    //    by the production builder.  Analogous to
                    //    `flowspace/flowcontext.py:LOAD_GLOBAL` reading
                    //    `frame.globals` (user globals) first.
                    //
                    // 2. Single-segment HOST_ENV builtin
                    //    (`HOST_ENV.lookup_builtin(name)`) — analogous
                    //    to `flowcontext.py:851 getattr(__builtin__,
                    //    varname)`, the second stage of
                    //    `find_global` (`flowcontext.py:845-853`).
                    //
                    // 3. Multi-segment HOST_ENV module attribute
                    //    (`HOST_ENV.import_module(prefix).module_get(\
                    //    attr)`) — analogous to the bytecode chain
                    //    `LOAD_GLOBAL <module>` (resolving via
                    //    `find_global`) followed by `LOAD_ATTR <attr>`
                    //    (`flowcontext.py:861-866`).  Pyre folds the
                    //    chain into a single `FunctionPath` carrier
                    //    because the frontend already records the
                    //    fully-qualified dotted path; the host
                    //    resolution still respects the same scope
                    //    boundary.
                    //
                    // Layer 3 — host module attribute lookup.  Upstream
                    // `LOAD_GLOBAL <module>` + `LOAD_ATTR <attr>` chain
                    // (`flowcontext.py:845-866`) consults the caller's
                    // per-function `frame.globals` first, falling back
                    // to `builtins` if absent.  Resolution order:
                    //
                    // 3b. `segments` already
                    //     spell out the fully-qualified Rust path
                    //     (`rpython::rtyper::lltypesystem::lltype::
                    //     cast_ptr_to_int`) without a matching `use`
                    //     statement — Rust compiles such paths
                    //     directly, so pyre source frequently writes
                    //     them inline.  Upstream has no exact analog
                    //     (Python source uniformly imports before
                    //     calling), so the HOST_ENV fallback
                    //     stays in place for existing pyre callsites.
                    //     A production callsite relies on this branch
                    //     (a tracing-time `OpKind::Call::FunctionPath`
                    //     with segments spelling a curated HOST_ENV
                    //     module), so removing the branch fails the
                    //     strict gate (cranelift fib_recursive +
                    //     fannkuch TIMEOUT).
                    //
                    // 3c. Unknown prefix — `TyperError` (caller must
                    //     register the path or import the prefix).
                    let callable_host = if let Some(entry) = call_registry.lookup(&key) {
                        entry.host_object.clone()
                    } else if segments.len() == 1
                        && let Some(builtin) = HOST_ENV.lookup_builtin(&segments[0])
                    {
                        builtin
                    } else if segments.len() >= 2
                        && let Some(module) =
                            HOST_ENV.import_module(&segments[..segments.len() - 1].join("."))
                        && let Some(attr) = module.module_get(&segments[segments.len() - 1])
                    {
                        // Branch 3b — fully-qualified inline path,
                        // PRE-EXISTING-ADAPTATION as documented above.
                        attr
                    } else if segments.len() == 2
                        && segments[0] == "simple_call"
                        && let Some(exc_class) = HOST_ENV.lookup_builtin(&segments[1])
                    {
                        // Branch 3c — PRE-EXISTING-ADAPTATION closure
                        // for `front::exc_from_raise::lower_exc_from_raise`
                        // (~`exc_from_raise.rs:153`).  Upstream RPython
                        // `flowcontext.py:614/623` emits
                        // `op.simple_call(const(exc_class), *args)`
                        // with the class as `args[0]`; pyre stashes
                        // the class name in `path[1]` of the
                        // `FunctionPath` because its `Vec<Variable>`
                        // arg carrier cannot hold a
                        // `Constant(HostObject(class))` alongside
                        // `Variable`s — holding it would require a
                        // `Vec<Variable>` → `Vec<LinkArg>` carrier (see the
                        // module-level "PRE-EXISTING-ADAPTATION"
                        // block in `front/exc_from_raise.rs:120-126` for the
                        // detailed rationale).  The downstream
                        // reconstruction is documented at
                        // `exc_from_raise.rs:122-123`:
                        // > any downstream reader can reconstruct
                        // > `(op, const_class, args…)` from
                        // > `(path[0], path[1], op.args)`
                        // This branch is exactly that
                        // reconstruction: resolve `path[1]`
                        // (the exception class name) as a builtin
                        // HostObject and use it as the simple_call
                        // callable, leaving `op.args` as the
                        // trailing message arguments.  No longer
                        // needed once the arg carrier can hold a
                        // `Constant` directly.
                        exc_class
                    } else if let Some(entry) = call_registry.lookup_with_leaf_match(&key) {
                        // Fuzzy leaf-match is the last registry fallback.
                        // Exact registry entries, HOST_ENV
                        // module paths, and the `simple_call(<exc class>)`
                        // raise reconstruction must win first so external
                        // stubs such as `BigInt::from`, `Vec::new`, and
                        // `Box::new` — and exception classes sharing a leaf —
                        // cannot be captured by an unrelated user function
                        // with the same leaf.
                        //
                        // This is the resting point for the former
                        // caller-scoped `use`-import resolution: MIR callsites
                        // carry the fully-qualified crate-relative path, so the
                        // exact `lookup(&key)` above is the primary binder and
                        // no per-caller import scope is needed to disambiguate.
                        // The leaf-scan itself is wrong-bind-safe —
                        // `lookup_with_leaf_match` returns `Some` only when the
                        // same-leaf matches converge on a single `host_object`
                        // identity, otherwise `None` falls through to the hard
                        // error below.
                        entry.host_object.clone()
                    } else {
                        return Err(TyperError::message(format!(
                            "translate_op: OpKind::Call::FunctionPath {{ segments: {:?} }} \
                             not registered in PyreCallRegistry, not in HOST_ENV \
                             `__builtin__`, and not a known module-qualified host attribute — \
                             the production builder (a SemanticProgram walker, or a test \
                             fixture building the registry directly) must register the path \
                             with its parameter Signature before specialize_legacy_graph \
                             consults the rtyper. Result slot = {}",
                            segments,
                            fmt_op_result(op),
                        )));
                    };
                    let callable =
                        Hlvalue::Constant(Constant::new(ConstValue::HostObject(callable_host)));
                    let mut call_args = Vec::with_capacity(arg_hls.len() + 1);
                    call_args.push(callable);
                    call_args.extend(arg_hls);
                    Ok(vec![FlowspaceOp::new("simple_call", call_args, result)])
                }
                CallTarget::SyntheticTransparentCtor { name, owner_path } => {
                    // RPython parity: tagged-union ctor `Foo(x)` annotates as
                    // `SomePBC([ClassDesc(Foo)])` then `pair_simple_call`
                    // constructs `SomeInstance(classdef)` (`bookkeeper.py:
                    // 315-316`).  Wrapping the ctor name as
                    // `HostObject::new_class(qualname, [])` routes through
                    // the existing `is_class()` arm in
                    // [`crate::annotator::bookkeeper::Bookkeeper::immutablevalue_hostobject`]
                    // (`bookkeeper.rs:1984`) → `getdesc` → `ClassDesc::new`
                    // (`classdesc.rs:708`) → `SomePBC([ClassDesc])`, instead
                    // of falling through to the "Don't know how to represent"
                    // error that `HostObject::new_opaque` produces.  The
                    // resulting `SomeInstance(classdef)` projects to
                    // `ConcreteType::GcRef`, matching legacy
                    // `resolve_types(Unknown) → GcRef`.  Post-jtransform
                    // [`crate::codewriter::jtransform`] still unwraps
                    // the simple_call to its inner value (the transparent
                    // semantics survive at the codewriter layer).
                    //
                    // `owner_path` qualifies the ctor identity so two
                    // distinct enums sharing a leaf (e.g.
                    // `StepResult::Continue` vs `JitAction::Continue`)
                    // produce different ClassDescs.  Falls back to the
                    // bare leaf when no owner was recorded (Ok/Err/Some).
                    // Intern the ctor class by its qualified qualname so
                    // every site shares one `HostObject` Arc — the
                    // singleton-class identity `getdesc` dedups on
                    // (`bookkeeper.rs:1040`, keyed by `Arc::ptr_eq`).
                    // Without interning each site mints a fresh Arc → a
                    // fresh `ClassDesc` per occurrence, so a class can
                    // never be numbered once (its `minid` would differ per
                    // site) nor instantiated consistently across graphs.
                    // The bare-leaf fallback (`owner_path` empty, only
                    // reached when the type failed to resolve) is NOT
                    // interned: a bare variant name like `Ok` is ambiguous
                    // across enums, so interning it would conflate distinct
                    // classes onto one `ClassDesc`.  Exception: the fixed
                    // aggregate-kind placeholder tags named by id atom
                    // (`Tuple` for tuples / unresolved ADTs, `Array`,
                    // `Closure` — see `front::mir::aggregate_ctor_name`) are
                    // not variant names; they name ONE universal placeholder
                    // each (no enum ambiguity), and the construction-side
                    // FieldWrite chain shares the same tag as `owner_root`
                    // with the field-projection side (which resolves it
                    // through `getuniqueclassdef_for_struct_root` →
                    // `intern_class_by_qualname`).  Minting a fresh Arc per
                    // site here splits the placeholder into one `ClassDef`
                    // per occurrence, so the value's classdef can never match
                    // the field repr's interned classdef (rtyper
                    // convert_from_to has no common base → typer error), and
                    // two `()` values meeting at a join can never union.
                    let host = if owner_path.is_empty() {
                        if matches!(name.as_str(), "Tuple" | "Array" | "Closure") {
                            call_registry.bookkeeper().intern_class_by_qualname(&name)
                        } else {
                            HostObject::new_class(name.clone(), Vec::new())
                        }
                    } else {
                        let bk = call_registry.bookkeeper();
                        // A variant ctor's owner tail is the enum type
                        // itself (`resolve_aggregate_adt` pushes the
                        // enum leaf onto the variant's owner path); a
                        // struct ctor's owner tail is its module.  The
                        // enum base registers as a flat class whose only
                        // row is the synthetic `__discriminant` tag
                        // (`front::mir` metadata collection) — no struct
                        // carries one — so `is_enum_base` discriminates an
                        // enum variant ctor from a struct ctor.
                        //
                        // An enum variant interns through the SAME
                        // primitive the discriminant-narrowing path uses
                        // ([`Bookkeeper::intern_enum_variant_host`]), keyed
                        // by `canonical_struct_name(enum_leaf)::variant`.
                        // A constructed `Some(x)` and a matched `Some(x)`
                        // then resolve to ONE class object (`rclass.py:
                        // 82-88` single-class-per-variant), so they union
                        // to that variant (not the base) and agree on the
                        // payload attr / field owner.  The variant is a
                        // subclass of the discriminant-only enum base, so
                        // sibling variants still union to
                        // `SomeInstance(enum)` through `ClassDef::commonbase`
                        // (classdesc.py:251-254) instead of failing
                        // `mergeinputargs` with "no common base class".
                        // `canonical_struct_name` keeps two enums sharing a
                        // leaf (`StepResult::Continue` vs
                        // `JitAction::Continue`) distinct.
                        //
                        // A struct ctor keeps its dotted qualname: the
                        // FORCE-attr projection in `_init_classdef`
                        // (`struct_force_key_from_dotted_qualname`) strips
                        // the crate to the `module::Type` key the
                        // struct-field FORCE table uses.  The bare-leaf
                        // case above (`owner_path` empty) is unchanged: a
                        // bare variant name like `Ok` is ambiguous across
                        // enums, so it is not interned.
                        // Probe the QUALIFIED enum-base spelling
                        // (`owner_path.join("::")` = the enum's full
                        // `name_path`) before the bare tail.  The qualified
                        // key survives `harden_duplicate_leaf_metadata`'s
                        // bare-leaf withdrawal, so a variant ctor of an enum
                        // whose leaf collides across modules
                        // (`StepResult::Continue` vs `JitAction::Continue`)
                        // still routes through `intern_enum_variant_host`
                        // instead of falling through to the dotted struct-ctor
                        // branch — which would mint a class unrelated to the
                        // matched variant's, breaking enum base/variant class
                        // identity (RPython keys identity on the live class
                        // object, never a name, so this cannot arise upstream).
                        // Interning still keys on the bare tail: a constructed
                        // value and a discriminant-narrowed value both reach
                        // the SAME base classdef (the narrowing reads back the
                        // classdef name the ctor minted), so they agree.
                        let owner_tail = owner_path.last();
                        let owner_qual = owner_path.join("::");
                        let is_enum_variant =
                            bk.pyre_struct_fields.borrow().as_ref().is_some_and(|reg| {
                                reg.is_enum_base(&owner_qual)
                                    || owner_tail.is_some_and(|tail| reg.is_enum_base(tail))
                            });
                        if is_enum_variant {
                            bk.intern_enum_variant_host(owner_tail.unwrap(), &name)
                        } else {
                            let qualname = format!("{}.{}", owner_path.join("."), name);
                            bk.intern_class_by_qualname(&qualname)
                        }
                    };
                    let callable = Hlvalue::Constant(Constant::new(ConstValue::HostObject(host)));
                    let mut call_args = Vec::with_capacity(arg_hls.len() + 1);
                    call_args.push(callable);
                    call_args.extend(arg_hls);
                    Ok(vec![FlowspaceOp::new("simple_call", call_args, result)])
                }
                CallTarget::Method { name, .. } => {
                    let mut iter = arg_hls.into_iter();
                    let receiver = iter.next().ok_or_else(|| {
                        TyperError::message(
                            "Call::Method has empty args: receiver must be args[0]".to_string(),
                        )
                    })?;
                    let bound_method = Hlvalue::Variable(Variable::new());
                    let mut call_args = Vec::with_capacity(iter.size_hint().0 + 1);
                    call_args.push(bound_method.clone());
                    call_args.extend(iter);
                    Ok(vec![
                        FlowspaceOp::new(
                            "getattr",
                            vec![
                                receiver,
                                Hlvalue::Constant(Constant::new(ConstValue::byte_str(name))),
                            ],
                            bound_method,
                        ),
                        FlowspaceOp::new("simple_call", call_args, result),
                    ])
                }
                CallTarget::Indirect { .. } => Err(TyperError::message(format!(
                    "translate_op: Call with CallTarget::Indirect at result={} \
                     must be lowered to VtableMethodPtr + IndirectCall by \
                     rclass.rs before reaching the flowspace adapter; \
                     reaching here means the rclass rewrite didn't run",
                    fmt_op_result(op),
                ))),
                CallTarget::UnsupportedExpr => Err(TyperError::message(format!(
                    "translate_op: Call with CallTarget::UnsupportedExpr at \
                     result={} — frontend coverage gap; the `front::mir` \
                     arm that emitted this Call must classify the call shape \
                     before the rtyper sees it",
                    fmt_op_result(op),
                ))),
            }
        }

        // ─── Pyre-internal: IndirectCall ───
        // RPython `rpython/rtyper/rpbc.py:216-217`:
        // ```python
        // vlist.append(hop.inputconst(Void, row_of_graphs.values()))
        // v = hop.genop('indirect_call', vlist, resulttype=rresult)
        // ```
        // The trailing `c_graphs` Constant must carry actual graph
        // identities — pyre's parity emits `ConstValue::Graphs(Vec<usize>)`
        // via `GraphKey::of(&g.graph).as_usize()` (see
        // `translator/rtyper/rpbc.rs:1481-1490`). The flowspace adapter
        // doesn't have access to the graph registry that resolves
        // `CallPath` segments to `Rc<RefCell<FunctionGraph>>` references,
        // so it cannot construct a faithful `ConstValue::Graphs`. A
        // synthetic `ConstValue::List(byte_str(qualname))` would silently
        // drop indirect-call analysis (`graphanalyze.rs:333` falls back
        // to `top_result()` for any non-Graphs ConstValue), so fail-loud
        // is the parity-correct behaviour: `IndirectCall` must be lowered
        // by `rpbc.rs` (the rtyper-equivalent layer that owns the graph
        // registry) before reaching the flowspace adapter.
        OpKind::IndirectCall { .. } => Err(TyperError::message(format!(
            "translate_op: IndirectCall at result={} must be lowered to \
             a flowspace `indirect_call` op with `ConstValue::Graphs(Vec<\
             usize>)` candidate-graph keys by `rpbc.rs:1481-1490` before \
             reaching the adapter; synthesising a `ConstValue::List` here \
             would break `graphanalyze.rs:333` indirect-call analysis",
            fmt_op_result(op),
        ))),

        // ─── Pyre-internal: VtableMethodPtr ───
        // TODO(rclass-vtable-rework): pyre-only adaptation of
        // `rclass.py:371-377 getclsfield()`.  Emitted by
        // `translator/rtyper/rclass.rs` to project the function
        // pointer out of a `dyn Trait` receiver's vtable. It exists
        // only *inside* the rtyper pipeline (rclass produces it;
        // jtransform consumes it), so reaching the flowspace adapter
        // input means an rtyper-stage layer missed its own emit/
        // consume invariant.
        OpKind::VtableMethodPtr { .. } => Err(TyperError::message(format!(
            "translate_op: VtableMethodPtr at result={} is rtyper-internal \
             (TODO(rclass-vtable-rework) of rclass.py:371-377); rclass.rs \
             emits it and the jtransform layer consumes it before flowspace \
             adapter input — reaching here means the rclass→jtransform \
             pipeline broke",
            fmt_op_result(op),
        ))),

        // ─── Stage-invariant fail-loud catch-all ───
        // No remaining variants reach here legitimately: every legitimate
        // pre-rtyper input shape has an explicit arm above, every
        // post-rtyper jtransform-emitted variant is enumerated in
        // `post_rtyper_jtransform_variant_name` and short-circuits with
        // a stage-mismatch message before this fall-through, and every
        // pyre-internal rtyper-cutover variant (`IndirectCall`,
        // `VtableMethodPtr`) has its own targeted fail-loud arm.  Hitting
        // this catch-all means a brand-new `OpKind` was added without
        // updating either the explicit translate arm OR the variant-name
        // table — fail-loud with a clear pointer at where the missing
        // arm should land.
        other => {
            let variant = opkind_variant_name(other);
            if let Some(stage_msg) = post_rtyper_jtransform_variant_name(other) {
                Err(TyperError::message(format!(
                    "translate_op: post-rtyper jtransform variant `{stage_msg}` \
                     reached the flowspace adapter at result={}.  RPython \
                     `rpython/jit/codewriter/jtransform.py` runs *after* the \
                     rtyper has lowered every high-level op, so this variant \
                     must NEVER appear at the rtyper input.  Source of the \
                     leak is upstream — check `rpbc.rs` / `rclass.rs` / the \
                     pre-rtyper graph builder for an emit site that should \
                     have produced a pre-rtyper shape (e.g. `FieldRead` / \
                     `ArrayRead` / `Call`) instead of `{variant}`.",
                    fmt_op_result(op),
                )))
            } else {
                Err(TyperError::message(format!(
                    "translate_op: OpKind variant `{variant}` has no \
                     translate arm and no stage-invariant classification.  \
                     A new pyre-internal variant was added to \
                     `model::OpKind` without updating \
                     `flowspace_adapter::translate_op` or \
                     `opkind_variant_name`.  Add an explicit translate arm \
                     above (lower to flowspace) or, if the variant is \
                     post-rtyper-only, list it in \
                     `post_rtyper_jtransform_variant_name` so the \
                     stage-mismatch message fires.  result={}",
                    fmt_op_result(op),
                )))
            }
        }
    }
}

/// Stable variant name for fail-loud messages. Matches the RPython
/// convention of identifying ops by their opname stem so dual-gate
/// failures are immediately greppable.
fn opkind_variant_name(kind: &OpKind) -> &'static str {
    #[allow(unreachable_patterns)]
    match kind {
        OpKind::Input { .. } => "Input",
        OpKind::ConstInt(_) => "ConstInt",
        OpKind::ConstBool(_) => "ConstBool",
        OpKind::ConstSymbolic { .. } => "ConstSymbolic",
        OpKind::ConstFloat(_) => "ConstFloat",
        OpKind::FieldRead { .. } => "FieldRead",
        OpKind::FieldWrite { .. } => "FieldWrite",
        OpKind::ArrayRead { .. } => "ArrayRead",
        OpKind::ArrayWrite { .. } => "ArrayWrite",
        OpKind::InteriorFieldRead { .. } => "InteriorFieldRead",
        OpKind::InteriorFieldWrite { .. } => "InteriorFieldWrite",
        OpKind::Call { .. } => "Call",
        OpKind::GuardTrue { .. } => "GuardTrue",
        OpKind::GuardFalse { .. } => "GuardFalse",
        OpKind::GuardValue { .. } => "GuardValue",
        OpKind::VtableMethodPtr { .. } => "VtableMethodPtr",
        OpKind::IndirectCall { .. } => "IndirectCall",
        OpKind::BinOp { .. } => "BinOp",
        OpKind::UnaryOp { .. } => "UnaryOp",
        OpKind::VableForce { .. } => "VableForce",
        OpKind::VableFieldRead { .. } => "VableFieldRead",
        OpKind::VableFieldWrite { .. } => "VableFieldWrite",
        OpKind::VableArrayRead { .. } => "VableArrayRead",
        OpKind::VableArrayWrite { .. } => "VableArrayWrite",
        OpKind::CallElidable { .. } => "CallElidable",
        OpKind::CallResidual { .. } => "CallResidual",
        OpKind::CallMayForce { .. } => "CallMayForce",
        OpKind::InlineCall { .. } => "InlineCall",
        OpKind::RecursiveCall { .. } => "RecursiveCall",
        OpKind::JitDebug { .. } => "JitDebug",
        OpKind::AssertGreen { .. } => "AssertGreen",
        OpKind::CurrentTraceLength => "CurrentTraceLength",
        OpKind::IsConstant { .. } => "IsConstant",
        OpKind::IsVirtual { .. } => "IsVirtual",
        OpKind::IsInstance { .. } => "IsInstance",
        OpKind::ConditionalCall { .. } => "ConditionalCall",
        OpKind::ConditionalCallValue { .. } => "ConditionalCallValue",
        OpKind::RecordKnownResult { .. } => "RecordKnownResult",
        OpKind::RecordQuasiImmutField { .. } => "RecordQuasiImmutField",
        OpKind::Live => "Live",
        OpKind::JitMergePoint { .. } => "JitMergePoint",
        OpKind::LoopHeader { .. } => "LoopHeader",
        OpKind::Abort { .. } => "Abort",
        // Catch-all for variants pyre may add without bumping this
        // table — surfaces as `<unknown>` in the fail-loud message
        // rather than a misleading variant tag.  The catch-all message
        // (above) instructs the reader to extend this table.
        _ => "<unknown OpKind variant>",
    }
}

/// Identify whether `kind` is a post-rtyper jtransform-emitted variant
/// (i.e., emitted by `rpython/jit/codewriter/jtransform.py` AFTER the
/// rtyper has lowered every high-level op).  These variants must NEVER
/// reach the flowspace adapter — a leak indicates the upstream pre-
/// rtyper graph builder (or pyre's `rpbc.rs` / `rclass.rs`) emitted the
/// post-rtyper shape directly instead of routing through the rtyper-
/// equivalent lowering step.  Returns `Some(name)` for fail-loud messages.
fn post_rtyper_jtransform_variant_name(kind: &OpKind) -> Option<&'static str> {
    Some(match kind {
        OpKind::VableFieldRead { .. } => "VableFieldRead (jtransform.py:651-927)",
        OpKind::VableFieldWrite { .. } => "VableFieldWrite (jtransform.py:651-927)",
        OpKind::VableArrayRead { .. } => "VableArrayRead (jtransform.py:651-927)",
        OpKind::VableArrayWrite { .. } => "VableArrayWrite (jtransform.py:651-927)",
        OpKind::CallElidable { .. } => "CallElidable (jtransform.py:414-435 rewrite_call)",
        OpKind::CallResidual { .. } => "CallResidual (jtransform.py:414-435 rewrite_call)",
        OpKind::CallMayForce { .. } => "CallMayForce (jtransform.py:414-435 rewrite_call)",
        OpKind::InlineCall { .. } => "InlineCall (jtransform.py:473-482)",
        OpKind::RecursiveCall { .. } => "RecursiveCall (jtransform.py:522-534)",
        OpKind::JitDebug { .. } => "JitDebug (jtransform.py:1731-1743)",
        OpKind::AssertGreen { .. } => "AssertGreen (jtransform.py:1731-1743)",
        OpKind::CurrentTraceLength => "CurrentTraceLength (jtransform.py:1731-1743)",
        OpKind::IsConstant { .. } => "IsConstant (jtransform.py:1731-1743)",
        OpKind::IsVirtual { .. } => "IsVirtual (jtransform.py:1731-1743)",
        OpKind::ConditionalCall { .. } => "ConditionalCall (jtransform.py:1665-1688)",
        OpKind::ConditionalCallValue { .. } => "ConditionalCallValue (jtransform.py:1665-1688)",
        OpKind::RecordKnownResult { .. } => "RecordKnownResult (jtransform.py:292-313)",
        OpKind::RecordQuasiImmutField { .. } => "RecordQuasiImmutField (jtransform.py:901-903)",
        OpKind::Live => "Live (jtransform.py:469,481,533)",
        OpKind::JitMergePoint { .. } => "JitMergePoint (jtransform.py:1690-1718)",
        OpKind::LoopHeader { .. } => "LoopHeader (jtransform.py:1690-1718)",
        // `OpKind::Abort` is pyre-only — RPython raises `FlowingError`
        // (`flowspace/flowcontext.py:258,417`) and drops the function
        // before reaching the rtyper.  Now handled by an explicit
        // `Ok(Vec::new())` arm in `translate_op`, but the diagnostic
        // name is retained here so the post-rtyper variant table can
        // still surface the marker if a leak ever reaches it.
        // Each front-end `stop_unsupported` / `continue_with_unknown`
        // emit-site is retired by lowering its specific expression
        // form (`ConstStr`, `Range`, `Closure`, etc.).
        OpKind::Abort { .. } => "Abort (pyre-only abort marker)",
        _ => return None,
    })
}

/// Output of [`function_graph_to_flowspace`] — the assembled
/// `flowspace::FunctionGraph` plus enough side tables for
/// `specialize_legacy_graph` to drive `RPythonTyper::specialize`
/// against pyre's annotator surface and read back per-`Variable`
/// concretetypes.
#[derive(Debug)]
pub struct FlowspaceAdapterOutput {
    /// Assembled `flowspace::FunctionGraph` carrying every legacy block
    /// translated to a `flowspace::Block` over `Hlvalue` operands.
    /// Wrapped in `Rc<RefCell<_>>` to match RPython's
    /// `FunctionDesc.cache` ownership shape — handed to
    /// `RPythonAnnotator` directly.
    pub graph: Rc<RefCell<FlowspaceGraph>>,
    /// Legacy graph `Variable` → typed flowspace `Variable`; each typed
    /// Variable's `concretetype` is read after `specialize` returns.
    pub value_to_var: LegacyToTyped,
    /// Const-define result `Variable` -> `Constant.concretetype`.
    /// Materialised at lift time from `OpKind::ConstInt` / `ConstFloat`
    /// via `Constant::with_concretetype` (`flowspace_adapter.rs:518-527`),
    /// matching RPython's `Constant.concretetype` ground truth.  The
    /// per-`Variable` `LowLevelType` is read directly so the projector
    /// does not have to reconstruct the kind from the reduced legacy
    /// `ValueType` view.
    pub constant_concretetypes: HashMap<Variable, LowLevelType>,
    /// `BlockId → flowspace::BlockRef` mapping. Includes the canonical
    /// `returnblock` and `exceptblock` (mapped to the
    /// `FunctionGraph::with_return_var`-allocated final blocks) so any
    /// legacy Link targeting them resolves correctly.
    #[cfg(test)]
    pub block_map: HashMap<BlockId, BlockRef>,
}

/// Translate a legacy `ExitCase` into the `Hlvalue` slot expected by
/// `flowspace::Link.exitcase`. RPython encodes the discriminating value
/// as a `Hlvalue::Constant` carrying the matched bool / Python value
/// (`flowspace/model.py:114-120`).
fn exitcase_to_hlvalue(exitcase: Option<&ExitCase>) -> Option<Hlvalue> {
    match exitcase {
        None => None,
        Some(ExitCase::Bool(b)) => Some(Hlvalue::Constant(constant_from_constvalue(
            ConstValue::Bool(*b),
        ))),
        Some(ExitCase::Const(cv)) => Some(Hlvalue::Constant(constant_from_constvalue(cv.clone()))),
    }
}

fn constant_from_constvalue(value: ConstValue) -> Constant {
    match value {
        ConstValue::Int(n) => Constant::with_concretetype(ConstValue::Int(n), LowLevelType::Signed),
        ConstValue::Bool(b) => Constant::with_concretetype(ConstValue::Bool(b), LowLevelType::Bool),
        ConstValue::Float(bits) => {
            Constant::with_concretetype(ConstValue::Float(bits), LowLevelType::Float)
        }
        other => Constant::new(other),
    }
}

fn legacy_const_define_hlvalue(op: &SpaceOperation) -> Option<Hlvalue> {
    // Const-define ops always carry a result Variable; bail on the
    // (malformed) result-less op so the caller can key the const
    // concretetype by `op.result` identity.
    op.result.as_ref()?;
    match &op.kind {
        OpKind::ConstInt(n) => Some(Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(*n),
            LowLevelType::Signed,
        ))),
        OpKind::ConstBool(b) => Some(Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Bool(*b),
            LowLevelType::Bool,
        ))),
        OpKind::ConstFloat(bits) => Some(Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Float(*bits),
            LowLevelType::Float,
        ))),
        OpKind::ConstRefNull => Some(Hlvalue::Constant(const_ref_gcref_constant(None))),
        OpKind::ConstRefAddr(addr) => {
            Some(Hlvalue::Constant(const_ref_gcref_constant(Some(*addr))))
        }
        // String-literal constant.  Upstream flowspace carries a string
        // literal as a bare `Constant('text')` SSA value (annotated
        // `SomeString` by `immutablevalue`); `front::mir` has no
        // ConstStr opkind and synthesises a 0-arg
        // `Call(["__str_const", <text>])` instead (`mir.rs:1576`).
        // Re-fold that define-op to the upstream Constant shape here.
        // The stamped lltype is `Ptr(STR)` — fixed for every string
        // constant, same as the primitive arms above (the rtyper's
        // `StringRepr.convert_const` re-derives the same `Ptr(STR)`
        // per use; `inputconst` always converts from `c.value`, so
        // the ByteStr payload is never read through this stamp).
        // Without the stamp the define-result Variable has no kind,
        // and the dual-gate projection reports the slot as a
        // `legacy=GcRef, real=Unknown` divergence on every graph
        // containing a string literal.
        OpKind::Call {
            target: crate::model::CallTarget::FunctionPath { segments },
            args,
            ..
        } if args.is_empty() && segments.len() == 2 && segments[0] == "__str_const" => {
            Some(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::ByteStr(segments[1].clone().into_bytes()),
                crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone(),
            )))
        }
        // RPython parity: unit-variant ctors (`StepResult::Continue`,
        // `LoopResult::Done`, …) are pre-built singleton instances at
        // the rtyper layer (`rclass.InstanceRepr.
        // get_reusable_prebuilt_instance`), so the codewriter never
        // sees a call op for them — the rtyper folds them to
        // `Constant(prebuilt_ptr)` before `jtransform` runs.
        //
        // Pyre's frontend (`front::mir`) lowers a unit-variant
        // path expression to `OpKind::Call { target:
        // SyntheticTransparentCtor, args: [], result_ty: Unknown }`;
        // without this pre-fold the args=[] call falls through to
        // `handle_residual_call` and leaves a `residual_call_r/d>r`
        // op in the walker arm body that breaks the walker's arm
        // dispatch.
        //
        // The `is_synthetic_unit_variant_path` allowlist
        // (StepResult, LoopResult, JitAction, CompareOp variants) is
        // the same set consulted here — both layers agree on which
        // paths are unit-variant singletons.
        OpKind::Call {
            target: crate::model::CallTarget::SyntheticTransparentCtor { name, owner_path },
            args,
            ..
        } if args.is_empty() => {
            let mut segments = owner_path.clone();
            segments.push(name.clone());
            if !crate::translator::rtyper::unit_variant_fold::is_synthetic_unit_variant_path(
                &segments,
            ) {
                return None;
            }
            // PyPy rtyper folds unit-variant PBC constructors into a
            // singleton instance pointer before jtransform sees them
            // (`rtyper/rpbc.py::SingleFrozenPBCRepr`).  The
            // pre-fold here materialises the same shape inside the
            // flowspace graph so the per-graph annotator surfaces a
            // `Hlvalue::Constant(HostObject(prebuilt_instance))` to
            // downstream rtyper passes.  This only
            // affects graphs that go through the rtyper Match arm
            // (`dual_gate_publish_concretetypes`).  Per-opcode arm
            // body graphs registered via `register_function_graph`
            // typically take the Skip arm and bypass this pre-fold;
            // the residual `OpKind::Call` survives into jtransform
            // and is emitted as a `residual_call_r_r` wrapper there.
            // Closing that gap requires either an early-pass on
            // `FunctionGraph` ahead of jtransform or extending
            // `is_synthetic_result_option_ctor` to handle the args=0
            // case.
            let qualname = segments.join(".");
            // Reuse the process-wide prebuilt-instance interner so this
            // legacy fold path produces the same `HostObject` Arc as the
            // pre-jtransform `fold_unit_variant_ctors` pass — mirrors
            // `InstanceRepr.get_reusable_prebuilt_instance` caching on
            // the per-rtyper `instance_reprs` map
            // (`rpython/rtyper/rclass.py:804`).  Without this, two
            // graphs that reach the same unit variant via different
            // gate arms would resolve to distinct singletons.
            let instance = crate::translator::rtyper::unit_variant_fold::intern_unit_variant_prebuilt_instance(
                &qualname,
            )?;
            Some(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::HostObject(instance),
                crate::translator::rtyper::rclass::OBJECTPTR.clone(),
            )))
        }
        _ => None,
    }
}

/// Translate a single legacy `LinkArg` into a `Hlvalue`. `LinkArg::Value`
/// resolves through `value_map` (which carries Variable identities for
/// regular operands and inlined constants for `OpKind::ConstInt` /
/// `ConstFloat` defines, mirrored in tests by
/// `build_value_to_hlvalue_map`). `LinkArg::Const` materialises a
/// fresh `Hlvalue::Constant`.
///
/// `source_block_id` / `target_block_id` / `arg_index` carry the
/// surrounding context for fail-loud diagnostics — when the lookup
/// misses, the message embeds the predecessor and successor block ids
/// plus the `arg_index` into `Link.args`, so per-graph diagnosis can
/// locate the broken link without re-traversing the graph.  Mirrors
/// the role-bearing enrichment of `lookup_operand` (variant name +
/// arg role).  The required substring `"undefined operand"`
/// is preserved verbatim for `is_known_unported`
/// (`cutover.rs:441`).
fn link_arg_to_hlvalue(
    arg: &LinkArg,
    value_map: &HashMap<Variable, Hlvalue>,
    source_block_id: BlockId,
    target_block_id: BlockId,
    arg_index: usize,
) -> Result<Hlvalue, TyperError> {
    match arg {
        // `LinkArg::Const` now carries the full upstream-orthodox
        // `Constant` (id + value + concretetype) directly — no need
        // to round-trip through `constant_from_constvalue` and
        // mint a fresh id.
        LinkArg::Const(cv) => Ok(Hlvalue::Constant(cv.clone())),
        LinkArg::Value(var) => value_map.get(var).cloned().ok_or_else(|| {
            TyperError::message(format!(
                "translate_op: undefined operand {var:?} as Link.args[{arg_index}] entry \
                 (source block {source_block_id:?} -> target block {target_block_id:?}) — \
                 adapter invariant broken (every referenced operand must be \
                 defined as a block inputarg or op result)",
            ))
        }),
    }
}

/// Translate an exception-link extra variable.
///
/// RPython `flowspace/model.py:636-642` defines `link.last_exception`
/// and `link.last_exc_value` in the link scope before checking
/// `link.args`; those same Variables may then appear in `link.args`.
/// Pyre's legacy graph represents them as fresh slots whose only
/// definition site is the link, so the adapter must materialise them in
/// a per-link map instead of requiring a block-local definition.
fn link_extravar_to_hlvalue(
    arg: &LinkArg,
    value_map: &mut HashMap<Variable, Hlvalue>,
    value_to_var: &mut LegacyToTyped,
) -> Result<Hlvalue, TyperError> {
    match arg {
        LinkArg::Value(legacy_var) => {
            if let Some(existing) = value_map.get(legacy_var).cloned() {
                return Ok(existing);
            }
            let var = seed_variable(legacy_var);
            value_to_var
                .entry(legacy_var.clone())
                .or_insert_with(|| var.clone());
            let hlvalue = Hlvalue::Variable(var);
            value_map.insert(legacy_var.clone(), hlvalue.clone());
            Ok(hlvalue)
        }
        // `LinkArg::Const` now carries the full upstream-orthodox
        // `Constant` (id + value + concretetype) directly — no need
        // to round-trip through `constant_from_constvalue` and
        // mint a fresh id.
        LinkArg::Const(cv) => Ok(Hlvalue::Constant(cv.clone())),
    }
}

/// Derive per-inputarg `SomeValue` cells for a subject's startblock,
/// preferring an explicit `Variable.annotation` seed (test-fixture
/// hand-built graphs without front-end Input ops) and falling through
/// to the `OpKind::Input { name, ty }` ops the front-end (`front::mir`)
/// emits for the receiver `self` and each typed param.
///
/// Returns one `SomeValue` per `startblock.inputargs` entry, in
/// position order.
///
/// Resolution order per inputarg `vid`:
/// 1. `legacy.variable(vid).annotation` — minimal fixtures supply
///    Variable-shape annotations explicitly via
///    `legacy_annotator::setbinding(&var, ty)`.
/// 2. Matching `OpKind::Input { ty }` op result == `vid` at the
///    startblock — production graphs from `front::mir`.
///
/// Errors:
///
/// - Both sources miss for an inputarg — front-end producer
///   divergence (every typed param emits the Input op alongside the
///   inputargs registration in the front pass; a missing Input op
///   means the producer wired the inputarg without declaring its
///   type and no `Variable.annotation` shell was supplied either).
/// - `valuetype_to_someshell(ty)` returns `None` for the resolved
///   `ValueType` (only `ValueType::Unknown`) — the inputarg's type
///   is an annotation gap; the helper surfaces it the same way
///   `seed_variable` does (`flowspace_adapter.rs:99-115`).
pub(crate) fn derive_subject_inputcells(
    legacy: &FunctionGraph,
    // Used to seed a `Ref` receiver with its cached, identity-stable
    // struct-root `ClassDef` via `getuniqueclassdef_for_struct_root`
    // (see the `OpKind::Input` arm below).  `None` (test fixtures) keeps
    // every `Ref` classdef-less, narrowed later by annotation.
    bookkeeper: Option<&Rc<crate::annotator::bookkeeper::Bookkeeper>>,
) -> Result<Vec<crate::annotator::model::SomeValue>, TyperError> {
    // Id-keyed lookup, not the dense `blocks[id.0]` projection — block
    // ids need not be index-aligned (see `reachable_block_ids`).
    let startblock = legacy
        .blocks
        .iter()
        .find(|b| b.id == legacy.startblock)
        .ok_or_else(|| {
            TyperError::message(format!(
                "derive_subject_inputcells: startblock {:?} not present in graph blocks",
                legacy.startblock
            ))
        })?;
    let mut input_by_result: HashMap<
        crate::flowspace::model::Variable,
        (&crate::model::ValueType, &Option<String>),
    > = HashMap::new();
    for op in &startblock.operations {
        if let (Some(result), OpKind::Input { ty, class_root, .. }) = (op.result.as_ref(), &op.kind)
        {
            input_by_result.insert(result.clone(), (ty, class_root));
        }
    }
    let mut cells = Vec::with_capacity(startblock.inputargs.len());
    for (idx, var) in startblock.inputargs.iter().enumerate() {
        // 1. Explicit SomeValue seed published onto
        //    `var.annotation` (test fixtures seed via
        //    `legacy_annotator::setbinding(&var, ty)` before invoking
        //    this function).
        if let Some(rc) = var.annotation.borrow().as_ref() {
            cells.push((**rc).clone());
            continue;
        }
        // 2. Front-end Input op at the startblock.
        if let Some(&(ty, class_root)) = input_by_result.get(var) {
            let shell = valuetype_to_someshell(ty).ok_or_else(|| {
                TyperError::message(format!(
                    "derive_subject_inputcells: startblock.inputargs[{idx}] \
                     ({var:?}) has `ValueType::{ty:?}` (from Input op) whose \
                     `valuetype_to_someshell` projection is `None` (annotation gap — \
                     only `ValueType::Unknown` lacks a SomeValue shell)"
                ))
            })?;
            // A `Ref` inputarg projects to the abstract `SomeInstance(None)`
            // shell from `valuetype_to_someshell`.  When the front-end
            // resolved the param's struct root (`OpKind::Input.class_root`,
            // populated from `type_root_ident` at front/ast.rs:2107-2184)
            // and the struct-field registry knows that root, seed the
            // receiver with the *cached* `ClassDef` from
            // `getuniqueclassdef_for_struct_root` (bookkeeper.rs:1522). The
            // cache makes the classdef identity-stable across repeated
            // lookups (`Rc::ptr_eq`), so the annotation fixpoint stays
            // independent of graph-processing order — unlike the earlier
            // eager seed that minted a fresh struct-root ClassDef per site
            // (identity divergence → order-dependent fixpoint).  Host structs
            // with no RPython classdef (e.g. `PyFrame::locals_cells_stack_w:
            // *mut FixedObjectArray`) are never a call argument, so
            // call-propagation never narrows them; the seed is the only path
            // that gives them a populated receiver.  Other `Ref` variants
            // (no `class_root`, unknown root, or no bookkeeper) keep the
            // classdef-less shell, narrowed by call-propagation as before
            // (`description.py:283-305 FunctionDesc.pycall`).
            if matches!(ty, crate::model::ValueType::Ref(_)) {
                // A list-typed param (`Vec<T>`, `&[T]`, …) carries its
                // full monomorphic spelling as `class_root` (the named-ADT
                // root resolver excludes the core/std/alloc container
                // family precisely so the receiver projects to the
                // annotator's list model here, not a minted classdef).
                // `project_pyre_field_type` maps the spelling to
                // `SomeList(elem)` so a `len()` / iteration on the receiver
                // resolves as a list op instead of `getattr` over the
                // classdef-less `SomeInstance(None)` shell.
                if let (Some(bk), Some(root)) = (bookkeeper, class_root.as_deref()) {
                    if majit_ir::descr::is_list_container_spelling(root) {
                        cells.push(bk.project_pyre_field_type(root));
                        continue;
                    }
                }
                // String-typed params are string values, not class
                // instances: `String` and `str` both map to the byte
                // string type (`s_str0` = `SomeString(no_nul=True)`,
                // matching `project_pyre_field_type`).  String literals
                // lower through `__str_const` to `ConstValue::ByteStr`
                // (flowspace_adapter, stamped `Ptr(STR)`/`StringRepr`),
                // so a literal flowing into a `&str`/`String` param must
                // meet the same byte `SomeString` — seeding `s_unicode0`
                // here instead raised `str ∪ unicode` at `mergeinputargs`.
                // The foreign `alloc::string::String` TypeDecl also
                // registers struct-field rows, so the registry path below
                // would otherwise seed a `SomeInstance(String)` shell whose
                // field writes poison classdef attr cells with
                // instance-annotated strings.
                if class_root.as_deref() == Some("String") || class_root.as_deref() == Some("str") {
                    cells.push(crate::annotator::model::s_str0());
                    continue;
                }
                if let (Some(root), Some(bk)) = (class_root.as_ref(), bookkeeper) {
                    // A generic param (`&T` where `T: Trait`, incl. a
                    // trait default body's `&Self`) carries the bound
                    // trait's qualified path as `class_root`
                    // (`tyref_generic_trait_bound_root`).  When the
                    // analyzed world has exactly one concrete impl of
                    // that trait, the receiver's only possible shape is
                    // that impl type — substitute its struct root and
                    // seed below as if the param were typed concretely.
                    let root = bk
                        .pyre_trait_unique_impls
                        .borrow()
                        .get(root)
                        .cloned()
                        .unwrap_or_else(|| root.clone());
                    let root = &root;
                    let known = bk
                        .pyre_struct_fields
                        .borrow()
                        .as_ref()
                        .is_some_and(|reg| reg.fields.contains_key(root));
                    if known {
                        // A `&FixedObjectArray` receiver models as its
                        // `_items` element list, not the wrapping struct
                        // (project_pyre_field_type), so an `arr[idx]` access
                        // resolves as a list `getitem` rather than rewriting
                        // to `getattr("__getitem__")`.
                        if majit_ir::descr::canonical_struct_name(root)
                            == "object_array::FixedObjectArray"
                        {
                            cells.push(bk.project_pyre_field_type(root));
                            continue;
                        }
                        let cd = bk.getuniqueclassdef_for_struct_root(root).map_err(|e| {
                            TyperError::message(format!(
                                "derive_subject_inputcells: startblock.inputargs[{idx}] \
                                 ({var:?}) `Ref` receiver root {root:?} failed struct-root \
                                 ClassDef registration: {e:?}"
                            ))
                        })?;
                        cells.push(crate::annotator::model::SomeValue::Instance(
                            crate::annotator::model::SomeInstance::new(
                                Some(cd),
                                false,
                                std::collections::BTreeMap::new(),
                            ),
                        ));
                        continue;
                    }
                }
            }
            cells.push(shell);
            continue;
        }
        // No further fallback: every typed parameter emits the Input
        // op alongside the inputargs registration; reaching here implies
        // the inputarg has neither a published `Variable.annotation`
        // shell nor a startblock Input op.
        return Err(TyperError::message(format!(
            "derive_subject_inputcells: startblock.inputargs[{idx}] \
             ({var:?}) has no matching `OpKind::Input {{ ty }}` op at \
             the startblock and no `Variable.annotation` shell — \
             front-end producer divergence (every typed parameter emits \
             the Input op alongside the inputargs registration; see \
             `front::mir`)"
        )));
    }
    Ok(cells)
}

/// `remove_dead_blocks` (`translator/simplify.py`) — the set of blocks
/// reachable from `startblock` by following `Block.exits`.  The rtyper
/// only annotates reachable blocks, so the flowspace adapter must drop
/// everything else.  The monotonic front-end leaves model-unreachable
/// `on_unwind` cleanup blocks in the graph: `lower_call` / `Drop` /
/// `Assert` forward past the Rust panic-table unwind edge
/// (`front::mir`), and the orphan cleanup block is closed by `set_raise`
/// (`model.rs::set_raise`).  `remove_dead_blocks` drops it before rtyping,
/// exactly as RPython annotates only the startblock-reachable closure;
/// without this prune the orphan block would be rtyped even though it can
/// never execute.  The `Block.dead`
/// flag covers only the framestate path's explicit marking
/// (`mir.rs::lower_framestate`); startblock reachability is the general
/// case the `dead` skip sites below also honour.
fn reachable_block_ids(legacy: &FunctionGraph) -> std::collections::HashSet<BlockId> {
    // Index lookup instead of `FunctionGraph::block` (a dense
    // `blocks[id.0]` projection): final blocks reached as link targets
    // need no `Block` entry of their own (test fixtures omit them), and
    // a final block contributes no exits anyway.  Mirrors `iterblocks`'
    // `exits[::-1]` stack order.
    let by_id: HashMap<BlockId, &crate::model::Block> =
        legacy.blocks.iter().map(|b| (b.id, b)).collect();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![legacy.startblock];
    while let Some(id) = stack.pop() {
        if seen.insert(id) {
            if let Some(block) = by_id.get(&id) {
                stack.extend(block.exits.iter().rev().map(|e| e.target));
            }
        }
    }
    seen
}

/// One-way conversion from the legacy `crate::model::FunctionGraph`
/// into a `flowspace::FunctionGraph` whose blocks carry `Hlvalue`
/// operands and per-value `SomeValue` annotations on its `Variable`s.
///
/// Two-pass topology assembly:
///
/// 1. **Pass 1** — allocate one `flowspace::BlockRef` per legacy
///    non-final block, allocating fresh `Variable`s for each block's
///    inputargs. Assemble the `flowspace::FunctionGraph` via
///    `FunctionGraph::with_return_var`, supplying the canonical
///    returnblock inputarg so the rtyper's `getreturnvar`
///    (`rtyper.rs:1633-1638`) finds a real return `Variable`.
/// 2. **Pass 2** — for each non-final block, translate `operations` via
///    [`translate_op`], translate `exits` (link args + targets +
///    exitcase) via [`link_arg_to_hlvalue`] / [`exitcase_to_hlvalue`],
///    and translate `exitswitch` via the `value_map`.
///
/// Topology assembly delegates per-OpKind operation translation to
/// [`translate_op`], which means any
/// legacy graph carrying an
/// unported OpKind variant surfaces a fail-loud `TyperError` from this
/// function. Trivial graphs (only `Input` / `ConstInt` / `ConstFloat`
/// op definitions) flow through cleanly.
///
/// Addpendingblock conversion — production path no longer
/// pre-seeds `Variable.annotation` from `legacy_annotator::annotate`.
/// Once the cutover entry queues the subject's startblock onto the
/// orthodox `addpendingblock` queue
/// (`cutover.rs:specialize_legacy_graph_with_registry_returning_value_to_var`),
/// `complete_pending_blocks` drives `flowin` which writes
/// `Variable.annotation` for every reachable inputarg and op result.
/// Carrying the legacy pre-seed alongside flowin caused
/// `setbinding: new value does not contain old` panics at
/// `annrpython.rs:459` whenever flowin's `follow_link` computed a
/// narrower annotation (e.g., constant-tracking `SomeInteger{const,
/// nonneg}`) than legacy_annotator's wider lift.
///
/// Test fixtures that hand-roll minimal SSA graphs without
/// production-shape `OpKind::Input { ty }` ops must seed
/// `legacy.variable(vid).annotation` directly via
/// `legacy_annotator::setbinding(&var, ValueType::…)` before calling
/// this function so `seed_variable` reads the right shell.
pub fn function_graph_to_flowspace(
    legacy: &FunctionGraph,
    // Call resolution plumbing — see [`translate_op`].
    call_registry: &crate::translator::rtyper::pyre_call_registry::PyreCallRegistry,
) -> Result<FlowspaceAdapterOutput, TyperError> {
    let mut value_to_var: LegacyToTyped = HashMap::new();
    let mut constant_hlvalues: HashMap<Variable, Hlvalue> = HashMap::new();
    let mut constant_concretetypes: HashMap<Variable, LowLevelType> = HashMap::new();

    // `remove_dead_blocks` reachable set — blocks not reachable from
    // `startblock` are dropped from every pass below alongside the
    // explicit `dead` flag.  See [`reachable_block_ids`].
    let reachable = reachable_block_ids(legacy);

    for legacy_block in &legacy.blocks {
        // Dead / unreachable blocks are pruned from the flowspace
        // (`remove_dead_blocks` parity), so their const-defines are never
        // referenced — skip them here too rather than stamp concretetypes
        // onto orphan cells.
        if legacy_block.dead || !reachable.contains(&legacy_block.id) {
            continue;
        }
        for legacy_op in &legacy_block.operations {
            if let Some(hlvalue) = legacy_const_define_hlvalue(legacy_op) {
                // `legacy_const_define_hlvalue` only returns `Some` for ops
                // with a result Variable, so this is always present here.
                let Some(result_var) = legacy_op.result.as_ref() else {
                    continue;
                };
                if let Hlvalue::Constant(c) = &hlvalue {
                    if let Some(ct) = &c.concretetype {
                        constant_concretetypes.insert(result_var.clone(), ct.clone());
                        // Also stamp the lltype onto the legacy graph's
                        // orphan Variable cell for this const-define
                        // result.  The rtyper consumes `Hlvalue::Constant`
                        // surfaces for const-defines and never reads the
                        // legacy Variable cell directly, so the write is
                        // additive — `RPythonTyper.specialize` won't
                        // overwrite this slot.  Downstream consumers
                        // reading `FunctionGraph::concretetype_of(&v)`
                        // (RPython parity for `getkind(v.concretetype)`)
                        // then see the const kind inline, without
                        // depending on the post-rtyper
                        // `apply_to_graph(constant_concretetypes, …)`
                        // bridge.
                        result_var.set_concretetype(Some(ct.clone()));
                    }
                }
                constant_hlvalues.insert(result_var.clone(), hlvalue);
            }
        }
    }

    // A flowspace graph contains only blocks reachable from
    // `startblock`: upstream builds graphs by abstract interpretation,
    // so an unreachable block cannot exist, and every downstream
    // consumer (checkgraph / annotator / rtyper) walks `iterblocks()`
    // — a reachability DFS (`flowspace/model.rs:4011`, model.py).  The
    // legacy MIR graph, by contrast, keeps lowered-but-unreachable
    // blocks: every `on_unwind` edge is dropped at lowering while the
    // `UnwindResume`/`Abort` terminator still lowers via `set_raise`,
    // leaving a predecessor-less block whose orphan `[etype, evalue]`
    // Link.args are defined by no inputarg and no op result.  Convert
    // only the reachable closure — the same `reachable` set the const
    // prepass above filtered on, so a block is translated iff its
    // const-defines were seeded.  A *reachable* block with the same
    // orphan shape still fails the operand-definedness check in
    // `link_arg_to_hlvalue` — SSA-definedness is not relaxed.

    // ──────────────────────────────────────────────────────────────
    // Pass 1 — allocate fresh `flowspace::BlockRef` for every legacy
    // reachable non-final block. The legacy `returnblock` and
    // `exceptblock` are skipped here; `FunctionGraph::with_return_var`
    // allocates the canonical flowspace finals, and the block_map is
    // populated with those after graph construction.
    // ──────────────────────────────────────────────────────────────

    let mut block_map: HashMap<BlockId, BlockRef> = HashMap::new();
    let mut block_inputarg_vars: HashMap<BlockId, Vec<(Variable, Variable)>> = HashMap::new();

    for legacy_block in &legacy.blocks {
        // `dead` / unreachable blocks are removed before annotation
        // (`remove_dead_blocks` parity).  The framestate path marks
        // model-unreachable orphan `on_unwind` blocks `dead`; the
        // monotonic path leaves them unmarked but startblock-unreachable,
        // so `reachable` is what prunes them there.  Skipping Pass 1 alloc
        // keeps them out of `block_map`, which Pass 2 must mirror (it
        // indexes `block_map`).  No reachable block can target a pruned
        // block — a target is reachable by construction — so the mirror
        // stays consistent.
        if legacy_block.dead
            || !reachable.contains(&legacy_block.id)
            || legacy_block.id == legacy.returnblock
            || legacy_block.id == legacy.exceptblock
        {
            continue;
        }
        let mut local_inputs: Vec<(Variable, Variable)> =
            Vec::with_capacity(legacy_block.inputargs.len());
        let mut inputargs: Vec<Hlvalue> = Vec::with_capacity(legacy_block.inputargs.len());
        for legacy_var in legacy_block.inputargs.iter() {
            let var = seed_variable(legacy_var);
            value_to_var
                .entry(legacy_var.clone())
                .or_insert_with(|| var.clone());
            local_inputs.push((legacy_var.clone(), var.clone()));
            inputargs.push(Hlvalue::Variable(var));
        }
        block_inputarg_vars.insert(legacy_block.id, local_inputs);
        block_map.insert(legacy_block.id, FlowspaceBlock::shared(inputargs));
    }

    let startblock = block_map.get(&legacy.startblock).cloned().ok_or_else(|| {
        TyperError::message(format!(
            "function_graph_to_flowspace: legacy.startblock {:?} not in legacy.blocks",
            legacy.startblock
        ))
    })?;

    // Resolve the returnblock's inputarg as a fresh final-block
    // Variable. Even when the legacy graph reuses the source slot
    // here, RPython's checkgraph treats target inputargs as definitions
    // in the target block, not as the predecessor's Variable object.
    //
    // RPython `flowmodel.py:281 FunctionGraph.getreturnvar(self)`
    // returns `self.returnblock.inputargs[0]` unconditionally — there
    // is no fallback for an empty `inputargs` list, and a malformed
    // graph raises `IndexError` at this site rather than fabricating a
    // fresh Variable.  Pyre's `model::FunctionGraph::with_return_var`
    // (`model.rs:983-988`) builds the returnblock with `inputargs:
    // vec![return_value]` by invariant, so the lookup is guaranteed to
    // succeed; surface the violation as a `TyperError` instead of
    // silently producing a `Variable::new()` placeholder.
    let return_var = legacy
        .blocks
        .iter()
        .find(|b| b.id == legacy.returnblock)
        .and_then(|b| {
            let legacy_var = b.inputargs.first()?;
            Some(legacy_var.clone())
        })
        .map(|legacy_var| {
            let var = seed_variable(&legacy_var);
            value_to_var
                .entry(legacy_var)
                .or_insert_with(|| var.clone());
            Hlvalue::Variable(var)
        })
        .ok_or_else(|| {
            TyperError::message(format!(
                "function_graph_to_flowspace: legacy graph {:?} has no \
                 returnblock {:?} with at least one inputarg — \
                 `model::FunctionGraph::with_return_var` (model.rs:983-988) \
                 builds the returnblock with `inputargs: vec![return_value]` \
                 by invariant; matches RPython `flowmodel.py:281 \
                 getreturnvar()` which indexes `returnblock.inputargs[0]` \
                 without a fallback",
                legacy.name, legacy.returnblock,
            ))
        })?;

    let graph = FlowspaceGraph::with_return_var(legacy.name.clone(), startblock, return_var);
    let returnblock_ref = graph.returnblock.clone();
    let exceptblock_ref = graph.exceptblock.clone();

    if let Some(legacy_exceptblock) = legacy.blocks.iter().find(|b| b.id == legacy.exceptblock) {
        if legacy_exceptblock.inputargs.len() == 2 {
            let mut except_inputargs = Vec::with_capacity(2);
            for legacy_var in legacy_exceptblock.inputargs.iter() {
                let var = seed_variable(legacy_var);
                value_to_var
                    .entry(legacy_var.clone())
                    .or_insert_with(|| var.clone());
                except_inputargs.push(Hlvalue::Variable(var));
            }
            exceptblock_ref.borrow_mut().inputargs = except_inputargs;
        }
    }

    let graph_ref = Rc::new(RefCell::new(graph));

    // Map the canonical finals so any legacy Link targeting them
    // resolves to the flowspace finals constructed above.
    block_map.insert(legacy.returnblock, returnblock_ref);
    block_map.insert(legacy.exceptblock, exceptblock_ref);

    // ──────────────────────────────────────────────────────────────
    // Pass 2 — fill operations + exits + exitswitch for each reachable
    // non-final legacy block. Final blocks (returnblock / exceptblock)
    // are already terminal in flowspace — `mark_final()` was set by
    // `FunctionGraph::with_return_var`.
    // ──────────────────────────────────────────────────────────────

    for legacy_block in &legacy.blocks {
        // Mirror Pass 1's skip set exactly — a `dead` / unreachable block
        // has no `block_map` entry, so translating it would panic at the
        // index below.  See the Pass 1 comment for the `remove_dead_blocks`
        // rationale.
        if legacy_block.dead
            || !reachable.contains(&legacy_block.id)
            || legacy_block.id == legacy.returnblock
            || legacy_block.id == legacy.exceptblock
        {
            continue;
        }
        let block_ref = block_map[&legacy_block.id].clone();
        let mut value_map = constant_hlvalues.clone();
        let mut name_to_value: HashMap<String, Hlvalue> = HashMap::new();

        if let Some(inputs) = block_inputarg_vars.get(&legacy_block.id) {
            for (legacy_var, var) in inputs {
                let hlvalue = Hlvalue::Variable(var.clone());
                value_map.insert(legacy_var.clone(), hlvalue.clone());
                if let Some(name) = legacy.value_name_for(legacy_var) {
                    name_to_value.entry(name.to_string()).or_insert(hlvalue);
                }
            }
        }
        for legacy_op in &legacy_block.operations {
            if let (
                Some(result_var),
                OpKind::Input {
                    name,
                    ty: _,
                    class_root: _,
                },
            ) = (legacy_op.result.as_ref(), &legacy_op.kind)
            {
                if legacy_block.inputargs.contains(result_var) {
                    if let Some(existing) = value_map.get(result_var).cloned() {
                        name_to_value.entry(name.clone()).or_insert(existing);
                    }
                }
            }
        }

        // Translate operations.
        let mut translated_ops: Vec<FlowspaceOp> = Vec::new();
        for legacy_op in &legacy_block.operations {
            if let Some(hlvalue) = legacy_const_define_hlvalue(legacy_op) {
                if let Some(result_var) = legacy_op.result.as_ref() {
                    value_map.insert(result_var.clone(), hlvalue.clone());
                    if let Some(name) = legacy.value_name_for(result_var) {
                        name_to_value.insert(name.to_string(), hlvalue);
                    }
                }
                translated_ops.extend(translate_op(legacy_op, &value_map, call_registry)?);
                continue;
            }

            if let (
                Some(result_var),
                OpKind::Input {
                    name,
                    ty: _,
                    class_root: _,
                },
            ) = (legacy_op.result.as_ref(), &legacy_op.kind)
            {
                if !value_map.contains_key(result_var) {
                    if let Some(alias) = name_to_value.get(name).cloned() {
                        // Same-block name match: alias the body `Input`
                        // result to the prior `Hlvalue` for `name`.
                        // Mirrors `front::mir`'s same-block
                        // LOAD_FAST dedup that the front already
                        // enforces (no SSA divergence at this site).
                        if let Hlvalue::Variable(var) = &alias {
                            value_to_var
                                .entry(result_var.clone())
                                .or_insert_with(|| var.clone());
                        }
                        value_map.insert(result_var.clone(), alias);
                    } else {
                        // Cross-block body `Input`: name not in the
                        // current block's `name_to_value`.  RPython
                        // flowspace has no analogue for this — every
                        // local cross-block reference is threaded via
                        // the predecessor `Link.args` and the target
                        // block's `inputargs` (`flowcontext.py:872-884
                        // LOAD_FAST` writes the local into
                        // `self.locals_w`; cross-block reads always go
                        // through the target block's pre-allocated
                        // `inputargs[]`, never via a fresh Variable).
                        //
                        // Fail-loud here surfaces the producer-side
                        // gap: either the target block's inputargs were
                        // not extended to carry `name`, or the
                        // predecessor link's args do not include the
                        // slot producer.  The dual-gate at
                        // `cutover.rs:439 is_known_unported` matches
                        // the substring `"adapter cross-block body
                        // Input"` and Skip-classifies the graph,
                        // routing it through `legacy_state` until
                        // cross-block locals threading covers
                        // every shape.
                        return Err(TyperError::message(format!(
                            "translate_op: adapter cross-block body Input — \
                             name {name:?} (result {result_var:?}) was not threaded \
                             through Link.args / target inputargs by the \
                             predecessor block.  RPython has no body-`Input` \
                             op (flowcontext.py:872-884 LOAD_FAST writes locals \
                             into self.locals_w; cross-block reads go via the \
                             target block's pre-allocated inputargs).  Producer \
                             gap — either Cat 2.1 cross-block locals threading \
                             missed this shape, or the front-end's body-`Input` \
                             emission needs to be extended to predeclare the \
                             name in the predecessor link."
                        )));
                    }
                }
                translated_ops.extend(translate_op(legacy_op, &value_map, call_registry)?);
                continue;
            }

            // Skip Abort here for the same reason
            // `build_value_to_variable_map` skips it — `translate_op`
            // emits no flowspace op for Abort, so seeding its result
            // would leave the consumer's flowspace arg referencing a
            // never-defined Variable.  Letting `lookup_operand` fail
            // at the first consumer surfaces the orthodox
            // "undefined operand" message that
            // `is_known_unported` classifies as Skip at the
            // producer-adjacent site.
            if let Some(result_var) = legacy_op.result.as_ref()
                && !value_map.contains_key(result_var)
                && !matches!(legacy_op.kind, OpKind::Abort { .. })
            {
                // The op result is `result_var`'s sole definition site, so
                // its freshly-seeded flowspace Variable is the authority for
                // `value_to_var` — `insert`, not `or_insert_with`. A use of
                // `result_var` reached earlier in block-storage order (e.g.
                // a `*_ovf` raising op whose result the `checked_arith` front
                // rewrite also threads through a link) may already have
                // seeded a *different*, block-local typed Variable under the
                // same legacy key. Keeping that earlier entry splits the
                // identity: `specialize_block` writes the `concretetype` onto
                // this op-result Variable (the one `value_map` carries and
                // `translate_op` emits), while the dual gate reads the stale
                // earlier entry — a spurious `real=Unknown` divergence. The
                // per-block-Variable invariant still holds: this seed stays
                // local to the defining block; only the legacy→typed map is
                // repointed to the definition.
                let var = seed_variable(result_var);
                value_to_var.insert(result_var.clone(), var.clone());
                value_map.insert(result_var.clone(), Hlvalue::Variable(var));
            }
            translated_ops.extend(translate_op(legacy_op, &value_map, call_registry)?);
            if let Some(result_var) = legacy_op.result.as_ref() {
                if let Some(name) = legacy.value_name_for(result_var) {
                    if let Some(value) = value_map.get(result_var).cloned() {
                        name_to_value.insert(name.to_string(), value);
                    }
                }
            }
        }

        // Translate exits.
        let mut translated_exits: Vec<flowspace_model::LinkRef> =
            Vec::with_capacity(legacy_block.exits.len());
        for legacy_link in &legacy_block.exits {
            let mut link_value_map = value_map.clone();
            let last_exception = legacy_link
                .last_exception
                .as_ref()
                .map(|arg| link_extravar_to_hlvalue(arg, &mut link_value_map, &mut value_to_var))
                .transpose()?;
            let last_exc_value = legacy_link
                .last_exc_value
                .as_ref()
                .map(|arg| link_extravar_to_hlvalue(arg, &mut link_value_map, &mut value_to_var))
                .transpose()?;
            let target = block_map.get(&legacy_link.target).cloned().ok_or_else(|| {
                TyperError::message(format!(
                    "function_graph_to_flowspace: legacy link target {:?} not found in \
                     legacy.blocks (block id={:?})",
                    legacy_link.target, legacy_block.id,
                ))
            })?;
            let args: Vec<Hlvalue> = legacy_link
                .args
                .iter()
                .enumerate()
                .map(|(idx, arg)| {
                    link_arg_to_hlvalue(
                        arg,
                        &link_value_map,
                        legacy_block.id,
                        legacy_link.target,
                        idx,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let exitcase = exitcase_to_hlvalue(legacy_link.exitcase.as_ref());
            let mut link = FlowspaceLink::new(args, Some(target), exitcase);
            // RPython `Link.__init__` (`flowspace/model.rs:Link::new`) leaves
            // `llexitcase` unset; `RPythonTyper._convert_link`
            // (`rpython/rtyper/rtyper.py:353-360`) populates it during
            // specialization via `r_case.convert_const(link.exitcase)`.
            // Pre-seeding `llexitcase` here would diverge from upstream
            // `pre-rtyper` graph shape; the rtyper overwrites it anyway.
            link.extravars(last_exception, last_exc_value);
            link.prevblock = Some(Rc::downgrade(&block_ref));
            translated_exits.push(link.into_ref());
        }

        // Translate exitswitch.
        let translated_exitswitch = match &legacy_block.exitswitch {
            None => None,
            Some(ExitSwitch::Value(var)) => {
                Some(value_map.get(var).cloned().ok_or_else(|| {
                    // Inline counterpart of `lookup_operand` for the
                    // block.exitswitch path (no enclosing
                    // SpaceOperation). Required substring
                    // `"undefined operand"` is preserved
                    // verbatim for `is_known_unported`
                    // (`cutover.rs:441`).
                    TyperError::message(format!(
                        "translate_op: undefined operand {var:?} as block.exitswitch — \
                         adapter invariant broken (every referenced operand must be \
                         defined as a block inputarg or op result)",
                    ))
                })?)
            }
            Some(ExitSwitch::LastException) => Some(Hlvalue::Constant(c_last_exception())),
            // `ExitSwitch::Fused` is produced only by the JIT codewriter's
            // `jtransform::optimize_goto_if_not` (post-rtyper); the
            // pre-rtyper legacy graph this adapter translates never carries
            // a fused exitswitch.
            Some(ExitSwitch::Fused { .. }) => {
                unreachable!("fused exitswitch cannot reach the pre-rtyper flowspace adapter")
            }
        };

        // Commit to the flowspace::Block. Borrow scope is tight to
        // avoid alias-clash with link.prevblock's Weak above.
        {
            let mut block = block_ref.borrow_mut();
            block.operations = translated_ops;
            block.exits = translated_exits;
            block.exitswitch = translated_exitswitch;
        }
    }

    flowspace_model::checkgraph(&graph_ref.borrow());

    Ok(FlowspaceAdapterOutput {
        graph: graph_ref,
        value_to_var,
        constant_concretetypes,
        #[cfg(test)]
        block_map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::model::{KnownType, SomeObjectTrait, SomeValue};
    use crate::model::{
        Block, BlockId, FunctionGraph as LegacyGraph, OpKind, SpaceOperation, ValueType,
    };
    use crate::translator::rtyper::legacy_annotator::setbinding;
    use crate::translator::rtyper::pyre_call_registry::PyreCallRegistry;

    /// Mint `n` fresh values on `graph`, returned indexed `0..n` so
    /// fixtures can refer to operands / results / inputargs positionally.
    fn mint_vars(graph: &mut LegacyGraph, n: usize) -> Vec<crate::flowspace::model::Variable> {
        (0..n).map(|_| graph.alloc_value_var()).collect()
    }

    /// Project positional indices to held Variables for
    /// `Block { inputargs, .. }` literals.
    fn block_inputargs(
        vars: &[crate::flowspace::model::Variable],
        vids: &[usize],
    ) -> Vec<crate::flowspace::model::Variable> {
        vids.iter().map(|&i| vars[i].clone()).collect()
    }

    /// Helper: empty `PyreCallRegistry` for tests that don't exercise
    /// the Call resolution path.  The registry's
    /// bookkeeper is freshly minted because translate_op tests don't
    /// share state with an enclosing annotator.
    fn empty_call_registry() -> PyreCallRegistry {
        PyreCallRegistry::new(Rc::new(Bookkeeper::new()))
    }

    #[test]
    fn valuetype_int_lifts_to_someinteger_default() {
        let s = valuetype_to_someshell(&ValueType::Int).expect("Int must project");
        match s {
            SomeValue::Integer(i) => {
                assert_eq!(i.knowntype(), KnownType::Int);
                assert!(!i.nonneg, "default SomeInteger.nonneg must be false");
                assert!(!i.unsigned, "default SomeInteger.unsigned must be false");
            }
            other => panic!("ValueType::Int must lift to SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn valuetype_float_lifts_to_somefloat_default() {
        let s = valuetype_to_someshell(&ValueType::Float).expect("Float must project");
        match s {
            SomeValue::Float(f) => {
                assert_eq!(f.knowntype(), KnownType::Float);
                assert!(f.immutable(), "SomeFloat is immutable per model.py:164-183");
            }
            other => panic!("ValueType::Float must lift to SomeFloat, got {other:?}"),
        }
    }

    #[test]
    fn valuetype_ref_lifts_to_someinstance_classdef_none() {
        let s = valuetype_to_someshell(&ValueType::Ref(None)).expect("Ref must project");
        match s {
            SomeValue::Instance(inst) => {
                assert!(
                    inst.classdef.is_none(),
                    "Ref placeholder must carry classdef=None (model.py:438 default)"
                );
                assert!(
                    !inst.can_be_none,
                    "SomeInstance.__init__ default `can_be_None=False` (model.py:438)"
                );
                assert!(
                    inst.flags.is_empty(),
                    "SomeInstance.__init__ default `flags={{}}` (model.py:438)"
                );
            }
            other => panic!(
                "ValueType::Ref must lift to SomeValue::Instance(classdef=None), got {other:?}"
            ),
        }
    }

    #[test]
    fn valuetype_void_lifts_to_impossible_lattice_bottom() {
        let s = valuetype_to_someshell(&ValueType::Void).expect("Void must project");
        assert!(
            matches!(s, SomeValue::Impossible),
            "ValueType::Void must lift to SomeValue::Impossible, got {s:?}"
        );
    }

    #[test]
    fn valuetype_state_lifts_to_someinstance_classdef_none() {
        // TODO(state-shell): pyre-only `State` (JIT state pointer)
        // — temporary fallback shell.
        let s = valuetype_to_someshell(&ValueType::State).expect("State must project");
        match s {
            SomeValue::Instance(inst) => {
                assert!(
                    inst.classdef.is_none(),
                    "State placeholder must carry classdef=None"
                );
            }
            other => panic!(
                "ValueType::State must lift to SomeValue::Instance(classdef=None), got {other:?}"
            ),
        }
    }

    #[test]
    fn valuetype_unknown_returns_none_for_failloud_at_bindingrepr() {
        // `Unknown` is an annotation gap with no
        // annotation-stage shell.  Returning `None` leaves
        // `Variable.annotation` empty so `bindingrepr` panics with
        // `KeyError: no binding for arg`
        // (`annotator/annrpython.rs:418`), surfacing the producer-side
        // gap instead of silently bridging it to `GcRef` via a
        // fabricated `SomeInstance(None)` shell — that bridging
        // conflated the annotation-stage lattice node with the
        // **legacy** `resolve_types(Unknown) -> ConcreteType::Unknown
        // -> GcRef` resolver-stage backfill.
        assert!(
            valuetype_to_someshell(&ValueType::Unknown).is_none(),
            "ValueType::Unknown must return None — annotation gap, no \
             annotation-stage shell"
        );
    }

    #[test]
    fn seed_variable_attaches_lifted_annotation_observable_via_clone() {
        let mut graph = LegacyGraph::new("seed_test");
        let vars = mint_vars(&mut graph, 8); // vars[0..8]
        let legacy_var = vars[7].clone();
        setbinding(&legacy_var, ValueType::Int);
        let var = seed_variable(&legacy_var);

        // Reference semantics: the annotation Rc-shares across clones
        // (flowspace/model.rs:2010-2018), so a clone observes the same
        // shell instance.
        let clone = var.clone();
        let clone_ann = clone.annotation.borrow();
        let shell = clone_ann
            .as_ref()
            .expect("seed_variable must populate annotation slot");
        assert!(
            matches!(shell.as_ref(), SomeValue::Integer(_)),
            "annotation must round-trip the lifted SomeInteger shell"
        );
    }

    #[test]
    fn seed_variable_unannotated_input_leaves_annotation_empty_for_failloud() {
        // When the legacy Variable carries no published
        // SomeValue shell, the seed MUST NOT fabricate a
        // SomeInstance(classdef=None) — that would silently bridge an
        // annotation gap to GcRef via the resolver-stage backfill at
        // the wrong layer. Instead, leave Variable.annotation empty so
        // `bindingrepr` panics with `KeyError: no binding for arg`
        // (annotator/annrpython.rs:418), surfacing the producer-side
        // gap as a fail-loud signal.
        let legacy_var = Variable::new();
        let var = seed_variable(&legacy_var);
        let ann = var.annotation.borrow();
        assert!(
            ann.is_none(),
            "Unannotated input must leave annotation empty (Cat 2.4 fail-loud), \
             got {:?}",
            ann.as_ref()
        );
    }

    fn legacy_graph_with_inputarg_and_result(
        input: usize,
        result: usize,
    ) -> (LegacyGraph, Vec<crate::flowspace::model::Variable>) {
        let mut graph = LegacyGraph::new("test");
        let vars = mint_vars(&mut graph, input.max(result) + 1);
        let inputargs = block_inputargs(&vars, &[input]);
        let result_var = vars[result].clone();
        let mut block = Block {
            id: BlockId(0),
            inputargs,
            operations: vec![SpaceOperation {
                result: Some(result_var),
                kind: OpKind::ConstInt(0),
            }],
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];
        (graph, vars)
    }

    #[test]
    fn build_value_to_variable_map_seeds_inputargs_and_op_results() {
        let (graph, vars) = legacy_graph_with_inputarg_and_result(1, 2);

        setbinding(&vars[1], ValueType::Int);
        setbinding(&vars[2], ValueType::Ref(None));
        let map = build_value_to_variable_map(&graph);

        assert_eq!(
            map.len(),
            2,
            "map must seed both the inputarg and the op result"
        );
        assert!(
            matches!(
                map[&vars[1]]
                    .annotation
                    .borrow()
                    .as_ref()
                    .map(|s| s.as_ref().clone()),
                Some(SomeValue::Integer(_))
            ),
            "inputarg slot 1 (Int) must be seeded with SomeInteger"
        );
        assert!(
            matches!(
                map[&vars[2]]
                    .annotation
                    .borrow()
                    .as_ref()
                    .map(|s| s.as_ref().clone()),
                Some(SomeValue::Instance(_))
            ),
            "op-result slot 2 (Ref) must be seeded with SomeInstance(classdef=None)"
        );
    }

    #[test]
    fn build_value_to_variable_map_dedupes_by_value_id() {
        // Two ops both reading the same inputarg (legacy graphs are SSA
        // — every slot has one definition, but multiple readers).
        // Must produce one Variable identity per slot.
        let mut graph = LegacyGraph::new("dedup_test");
        let vars = mint_vars(&mut graph, 4); // vars[0..4]
        let inputargs = block_inputargs(&vars, &[1]);
        let result2_var = vars[2].clone();
        let result3_var = vars[3].clone();
        let mut block = Block {
            id: BlockId(0),
            inputargs,
            operations: vec![
                SpaceOperation {
                    result: Some(result2_var),
                    kind: OpKind::ConstInt(7),
                },
                SpaceOperation {
                    result: Some(result3_var),
                    kind: OpKind::ConstInt(11),
                },
            ],
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];

        setbinding(&vars[1], ValueType::Int);
        setbinding(&vars[2], ValueType::Int);
        setbinding(&vars[3], ValueType::Int);
        let map = build_value_to_variable_map(&graph);

        assert_eq!(map.len(), 3, "three distinct slots → three Variables");
        // The identity invariant: the inputarg's Variable is one fresh
        // identity, the two op results are two more fresh identities, and
        // they don't collide.
        assert_ne!(map[&vars[1]], map[&vars[2]]);
        assert_ne!(map[&vars[1]], map[&vars[3]]);
        assert_ne!(map[&vars[2]], map[&vars[3]]);
    }

    #[test]
    fn build_value_to_variable_map_aliases_input_rebind_to_inputarg() {
        // Pyre's surface front emits a leading `Input{name}` op whose
        // result IS a block.inputarg, plus follow-up `Input{same name}`
        // ops with FRESH result slots for body-side rebinds. The
        // adapter must alias the rebind result to the canonical
        // inputarg Variable so `setup_block_entry`'s
        // `concretetype` write reaches both — otherwise the body's
        // BinOp lookup hits a fresh Variable with no concretetype and
        // trips genop's "wrong level!" assertion.
        let mut graph = LegacyGraph::new("rebind_alias");
        let vars = mint_vars(&mut graph, 3); // vars[0..3]
        let mut block = Block {
            id: BlockId(0),
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![
                // Leading definition: result IS the inputarg.
                SpaceOperation {
                    result: Some(vars[1].clone()),
                    kind: OpKind::Input {
                        name: "x".to_string(),
                        ty: ValueType::Int,
                        class_root: None,
                    },
                },
                // Rebind: result is fresh; same name → alias to vars[1]'s Variable.
                SpaceOperation {
                    result: Some(vars[2].clone()),
                    kind: OpKind::Input {
                        name: "x".to_string(),
                        ty: ValueType::Int,
                        class_root: None,
                    },
                },
            ],
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];

        setbinding(&vars[1], ValueType::Int);
        setbinding(&vars[2], ValueType::Int);
        let map = build_value_to_variable_map(&graph);
        assert_eq!(
            map[&vars[1]], map[&vars[2]],
            "Input rebind result must alias to inputarg Variable identity"
        );
    }

    // ───── dispatcher + skip arms + fail-loud ─────

    #[test]
    fn build_value_to_hlvalue_map_inlines_const_defines() {
        let mut graph = LegacyGraph::new("const_inline");
        let vars = mint_vars(&mut graph, 4); // vars[0..4]
        let inputargs = block_inputargs(&vars, &[1]);
        let result2_var = vars[2].clone();
        let result3_var = vars[3].clone();
        let mut block = Block {
            id: BlockId(0),
            inputargs,
            operations: vec![
                SpaceOperation {
                    result: Some(result2_var),
                    kind: OpKind::ConstInt(42),
                },
                SpaceOperation {
                    result: Some(result3_var),
                    kind: OpKind::ConstFloat(0xC000_0000_0000_0000), // f64::from_bits → -2.0
                },
            ],
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        };
        block.id = graph.startblock;
        graph.blocks = vec![block];

        setbinding(&vars[1], ValueType::Int);
        setbinding(&vars[2], ValueType::Int);
        setbinding(&vars[3], ValueType::Float);
        let var_map = build_value_to_variable_map(&graph);
        let hl_map = build_value_to_hlvalue_map(&graph, &var_map);

        // Inputarg keeps its Variable identity.
        assert!(matches!(hl_map[&vars[1]], Hlvalue::Variable(_)));

        // ConstInt define is inlined as Hlvalue::Constant(Int).
        match &hl_map[&vars[2]] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(n) => assert_eq!(*n, 42),
                other => panic!("result 2 must be ConstValue::Int, got {other:?}"),
            },
            other => panic!("result 2 must be inlined as Hlvalue::Constant, got {other:?}"),
        }

        // ConstFloat define is inlined as Hlvalue::Constant(Float).
        match &hl_map[&vars[3]] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Float(bits) => assert_eq!(*bits, 0xC000_0000_0000_0000),
                other => panic!("result 3 must be ConstValue::Float, got {other:?}"),
            },
            other => panic!("result 3 must be inlined as Hlvalue::Constant, got {other:?}"),
        }
    }

    #[test]
    fn translate_op_skips_input_define() {
        let value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        let op = SpaceOperation {
            result: Some(vars[1].clone()),
            kind: OpKind::Input {
                name: "x".to_string(),
                ty: ValueType::Int,
                class_root: None,
            },
        };
        let result = translate_op(&op, &value_map, &empty_call_registry())
            .expect("Input must translate to skip");
        assert!(
            result.is_empty(),
            "Input define has no SpaceOperation analogue (handled by block \
             topology via block.inputargs); translate_op must yield empty Vec"
        );
    }

    #[test]
    fn derive_subject_inputcells_projects_each_typed_input_op() {
        let mut graph = LegacyGraph::new("subject");
        let entry = graph.startblock;
        let x_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "x".to_string(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let y_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "y".to_string(),
                    ty: ValueType::Float,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let z_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "z".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, x_var);
        graph.push_inputarg_var(entry, y_var);
        graph.push_inputarg_var(entry, z_var);

        let cells = derive_subject_inputcells(&graph, None)
            .expect("typed Input ops must project to definite SomeValue cells");
        assert_eq!(cells.len(), 3);
        assert!(
            matches!(cells[0], SomeValue::Integer(_)),
            "x: Int -> SomeInteger, got {:?}",
            cells[0],
        );
        assert!(
            matches!(cells[1], SomeValue::Float(_)),
            "y: Float -> SomeFloat, got {:?}",
            cells[1],
        );
        assert!(
            matches!(cells[2], SomeValue::Instance(_)),
            "z: Ref -> SomeInstance(classdef=None), got {:?}",
            cells[2],
        );
    }

    #[test]
    fn derive_subject_inputcells_fails_loud_on_inputarg_without_input_op() {
        let mut graph = LegacyGraph::new("subject");
        let entry = graph.startblock;
        let orphan = graph.alloc_value_var();
        graph.push_inputarg_var(entry, orphan);
        let err = derive_subject_inputcells(&graph, None)
            .expect_err("inputarg without matching Input op must surface as TyperError");
        let msg = format!("{err}");
        assert!(
            msg.contains("no matching `OpKind::Input"),
            "error must name the missing Input op invariant, got: {msg}"
        );
    }

    #[test]
    fn derive_subject_inputcells_fails_loud_on_unknown_valuetype() {
        let mut graph = LegacyGraph::new("subject");
        let entry = graph.startblock;
        let var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "u".to_string(),
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, var);
        let err = derive_subject_inputcells(&graph, None)
            .expect_err("ValueType::Unknown has no SomeValue projection");
        let msg = format!("{err}");
        assert!(
            msg.contains("Unknown") && msg.contains("None"),
            "error must mention Unknown + missing projection, got: {msg}"
        );
    }

    #[test]
    fn translate_op_skips_const_int_define() {
        let value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        let op = SpaceOperation {
            result: Some(vars[1].clone()),
            kind: OpKind::ConstInt(7),
        };
        let result = translate_op(&op, &value_map, &empty_call_registry())
            .expect("ConstInt must translate to skip");
        assert!(
            result.is_empty(),
            "ConstInt define is inlined by build_value_to_hlvalue_map; \
             translate_op must yield empty Vec"
        );
    }

    #[test]
    fn translate_op_skips_const_float_define() {
        let value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        let op = SpaceOperation {
            result: Some(vars[1].clone()),
            kind: OpKind::ConstFloat(0),
        };
        let result = translate_op(&op, &value_map, &empty_call_registry())
            .expect("ConstFloat must translate to skip");
        assert!(result.is_empty());
    }

    #[test]
    fn translate_op_binop_lowers_to_passthrough_spaceop() {
        // BinOp arm: `add` / `sub` / `lt` / ... pass through to a
        // flowspace SpaceOperation with the same opname; lhs/rhs args
        // get resolved via lookup_operand and the result Hlvalue via
        // resolve_result_hlvalue.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let lhs_var = Hlvalue::Variable(Variable::new());
        let rhs_var = Hlvalue::Variable(Variable::new());
        let result_var = Hlvalue::Variable(Variable::new());
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), lhs_var.clone());
        value_map.insert(vars[2].clone(), rhs_var.clone());
        value_map.insert(vars[3].clone(), result_var.clone());

        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: vars[1].clone(),
                rhs: vars[2].clone(),
                result_ty: ValueType::Int,
            },
        };
        let translated =
            translate_op(&op, &value_map, &empty_call_registry()).expect("BinOp arm must lower");
        assert_eq!(translated.len(), 1, "BinOp lowers to exactly one SpaceOp");
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "add", "opname passes through unchanged");
        assert_eq!(lowered.args.len(), 2);
    }

    #[test]
    fn translate_op_binop_undefined_lhs_surfaces_invariant_break() {
        // BinOp arm threads operand lookups through lookup_operand, so a
        // missing lhs surfaces the "adapter invariant broken" message
        // rather than a silent panic.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 100); // vars[0..100]
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: vars[99].clone(), // not in value_map
                rhs: vars[2].clone(),
                result_ty: ValueType::Int,
            },
        };
        let err = translate_op(&op, &value_map, &empty_call_registry())
            .expect_err("undefined BinOp operand must surface invariant break");
        let msg = format!("{err}");
        assert!(msg.contains("undefined operand"));
    }

    #[test]
    fn translate_op_call_function_path_lowers_to_simple_call() {
        // Call::FunctionPath → `simple_call(callable_host, args...)` per
        // `flowspace/operation.py:663 SimpleCall.opname = 'simple_call'`.
        // The callable Constant wraps the `PyreCallRegistry` entry's
        // synthetic `HostObject::UserFunction` so the rtyper's
        // `bookkeeper.getdesc` short-circuits onto the registered
        // FunctionDesc.
        use crate::flowspace::argument::Signature;
        use crate::translator::rtyper::pyre_call_registry::FunctionPathKey;
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let registry = empty_call_registry();
        registry.get_or_register(
            FunctionPathKey::from_segments(["a", "b"]),
            Signature::new(vec!["x".into()], None, None),
        );
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["a".into(), "b".into()],
                },
                args: vec![vars[1].clone()],
                result_ty: ValueType::Int,
            },
        };
        let translated =
            translate_op(&op, &value_map, &registry).expect("Call::FunctionPath must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "simple_call");
        assert_eq!(lowered.args.len(), 2, "callable + 1 arg");
        let Hlvalue::Constant(ref callable) = lowered.args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("FunctionPath callable must be ConstValue::HostObject");
        };
        // Synthetic GraphFunc takes the last path segment as its
        // `__name__`, mirroring upstream `func.__name__` (the leaf
        // identifier, not the dotted module path).
        assert_eq!(host.qualname(), "b");
    }

    #[test]
    fn translate_op_call_function_path_exact_registry_wins_over_leaf_collision() {
        // Two user fns share the bare leaf `shared_leaf` under different
        // module paths, so `lookup_with_leaf_match` is ambiguous (distinct
        // HostObjects → it returns None).  The exact `call_registry.lookup`
        // runs first in the resolution order, so a FunctionPath spelling
        // the full `othermod::shared_leaf` path resolves to that exact
        // entry; were resolution to fall through to the leaf-match
        // fallback it would error instead.  Locks in the
        // exact-before-leaf-match precedence the resolver depends on.
        use crate::flowspace::argument::Signature;
        use crate::translator::rtyper::pyre_call_registry::FunctionPathKey;
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let registry = empty_call_registry();
        let other_entry = registry.get_or_register(
            FunctionPathKey::from_segments(["othermod", "shared_leaf"]),
            Signature::new(vec!["x".into()], None, None),
        );
        let mymod_entry = registry.get_or_register(
            FunctionPathKey::from_segments(["mymod", "shared_leaf"]),
            Signature::new(vec!["x".into()], None, None),
        );
        assert_ne!(
            other_entry.host_object, mymod_entry.host_object,
            "colliding-leaf entries must be distinct HostObjects"
        );
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["othermod".into(), "shared_leaf".into()],
                },
                args: vec![vars[1].clone()],
                result_ty: ValueType::Int,
            },
        };
        let translated =
            translate_op(&op, &value_map, &registry).expect("exact FunctionPath must lower");
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "simple_call");
        let Hlvalue::Constant(ref callable) = lowered.args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("FunctionPath callable must be ConstValue::HostObject");
        };
        assert_eq!(
            host, &other_entry.host_object,
            "exact registry lookup must win over the leaf-match fallback"
        );
        assert_ne!(
            host, &mymod_entry.host_object,
            "must not resolve to the colliding-leaf sibling"
        );
    }

    #[test]
    fn translate_op_call_function_path_simple_call_exc_class_beats_leaf_match() {
        // Branch 3c (`simple_call(<exc class>)` raise reconstruction)
        // must win over the leaf-match registry fallback so an exception
        // class sharing a leaf with a registered user function still
        // resolves to the builtin class HostObject, not the user fn.
        // Without the ordering, `["simple_call", "ValueError"]` would be
        // captured by a registered `[mymod, ValueError]` free fn through
        // `lookup_with_leaf_match`.
        use crate::flowspace::argument::Signature;
        use crate::translator::rtyper::pyre_call_registry::FunctionPathKey;
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let registry = empty_call_registry();
        // Free-fn-shaped (snake_case module) candidate whose leaf
        // `ValueError` collides with the exception class name.
        registry.get_or_register(
            FunctionPathKey::from_segments(["mymod", "ValueError"]),
            Signature::new(vec!["msg".into()], None, None),
        );
        // Sanity: leaf-match alone would resolve the colliding user fn.
        let leaf_hit = registry
            .lookup_with_leaf_match(&FunctionPathKey::from_segments([
                "simple_call",
                "ValueError",
            ]))
            .expect("leaf-match must find the colliding user fn");
        assert!(
            leaf_hit.host_object.is_user_function(),
            "leaf-match fallback resolves the registered user fn"
        );
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["simple_call".into(), "ValueError".into()],
                },
                args: vec![vars[1].clone()],
                result_ty: ValueType::Ref(None),
            },
        };
        let translated =
            translate_op(&op, &value_map, &registry).expect("simple_call(<exc class>) must lower");
        assert_eq!(translated.len(), 1);
        let Hlvalue::Constant(ref callable) = translated[0].args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("callable must be ConstValue::HostObject");
        };
        assert!(
            !host.is_user_function(),
            "exc_class branch must resolve the builtin class, not the leaf-match user fn"
        );
        let expected = HOST_ENV
            .lookup_builtin("ValueError")
            .expect("bootstrap_builtin_exceptions must register ValueError");
        assert_eq!(
            host, &expected,
            "callable must be the builtin ValueError class HostObject"
        );
    }

    #[test]
    fn translate_op_call_function_path_falls_back_to_host_env_builtin() {
        // Single-segment FunctionPath unregistered in PyreCallRegistry
        // falls back to HOST_ENV.lookup_builtin(name), letting frontend
        // `Expr::Cast` lowering emit
        // `Call { target: FunctionPath { segments: vec!["int"] }, args }`
        // and route through `BuiltinFunctionRepr.rtype_simple_call →
        // BUILTIN_TYPER["int"] → rtype_builtin_int`.  Mirrors upstream
        // `flowspace/flowcontext.py:LOAD_GLOBAL` resolving against
        // `__builtin__.__dict__` after `frame.globals` misses.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["int".into()],
                },
                args: vec![vars[1].clone()],
                result_ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("Call::FunctionPath single-segment must fall back to HOST_ENV");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "simple_call");
        assert_eq!(lowered.args.len(), 2);
        let Hlvalue::Constant(ref callable) = lowered.args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("FunctionPath callable must be ConstValue::HostObject");
        };
        // HOST_ENV.lookup_builtin("int") returns the shared `int` class
        // HostObject — the same identity BUILTIN_TYPER keys its
        // `rtype_builtin_int` entry on.
        let expected = HOST_ENV
            .lookup_builtin("int")
            .expect("HOST_ENV bootstrap must register __builtin__.int");
        assert_eq!(host, &expected);
    }

    #[test]
    fn translate_op_call_function_path_resolves_host_module_attr_for_lltype_cast() {
        // Multi-segment FunctionPath unregistered in PyreCallRegistry
        // falls back to Layer 3 (HOST_ENV.import_module + module_get).
        // Mirrors upstream `LOAD_GLOBAL lltype` → `LOAD_ATTR cast_ptr_\
        // to_int` chain (`flowcontext.py:861-866`).  The resolved
        // HostObject is the same shared identity that BUILTIN_TYPER
        // keys its `rtype_cast_ptr_to_int` typer on
        // (`rbuiltin.py:543-548`).
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec![
                        "rpython".into(),
                        "rtyper".into(),
                        "lltypesystem".into(),
                        "lltype".into(),
                        "cast_ptr_to_int".into(),
                    ],
                },
                args: vec![vars[1].clone()],
                result_ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("Layer 3 host-module attr lookup must resolve lltype.cast_ptr_to_int");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "simple_call");
        let Hlvalue::Constant(ref callable) = translated[0].args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("FunctionPath callable must be ConstValue::HostObject");
        };
        let expected = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .and_then(|m| m.module_get("cast_ptr_to_int"))
            .expect("populate_host_env must register lltype.cast_ptr_to_int");
        assert_eq!(host, &expected);
    }

    #[test]
    fn translate_op_call_function_path_rejects_unregistered_multi_segment_path() {
        // Defense-in-depth: an arbitrary multi-segment user path
        // (`some.unknown.module.path`) that misses every layer —
        // PyreCallRegistry, HOST_ENV single-segment builtin, and
        // HOST_ENV module attr (because `some.unknown.module` is not
        // curated in `populate_host_env`) — surfaces a `TyperError`
        // rather than a silent host-attribute resolution.  This pins
        // the implicit HOST_ENV-curation bound that the Layer 3
        // fallback relies on.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec![
                        "some".into(),
                        "unknown".into(),
                        "module".into(),
                        "path".into(),
                    ],
                },
                args: vec![],
                result_ty: ValueType::Int,
            },
        };
        let err = translate_op(&op, &value_map, &empty_call_registry())
            .expect_err("Unregistered FunctionPath must surface TyperError, not silently resolve");
        let msg = format!("{err}");
        assert!(
            msg.contains("not registered in PyreCallRegistry"),
            "error must name the missing-registration invariant, got: {msg}"
        );
    }

    #[test]
    fn translate_op_cast_pointer_marker_rebuilds_two_arg_upstream_call() {
        // `__cast_pointer/<Root>` marker (front::mir
        // `cast_pointer_marker_op`) reconstructs the upstream 2-arg
        // `cast_pointer(PTRTYPE, ptr)` shape (lltype.py:964-968):
        // constant callable + constant interned target class, then the
        // pointer operand.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 10);
        let operand = Variable::new();
        value_map.insert(vars[1].clone(), Hlvalue::Variable(operand.clone()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::FunctionPath {
                    segments: vec!["__cast_pointer".into(), "W_CastTarget".into()],
                },
                args: vec![vars[1].clone()],
                result_ty: ValueType::Ref(Some("W_CastTarget".into())),
            },
        };
        let registry = empty_call_registry();
        let translated = translate_op(&op, &value_map, &registry)
            .expect("__cast_pointer marker must lower to simple_call");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "simple_call");
        assert_eq!(lowered.args.len(), 3);
        let Hlvalue::Constant(ref callable) = lowered.args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("cast_pointer callable must be ConstValue::HostObject");
        };
        let expected = HOST_ENV
            .import_module("rpython.rtyper.lltypesystem.lltype")
            .and_then(|m| m.module_get("cast_pointer"))
            .expect("populate_host_env must register lltype.cast_pointer");
        assert_eq!(host, &expected);
        let Hlvalue::Constant(ref class_const) = lowered.args[1] else {
            panic!("target class must be a Constant");
        };
        let ConstValue::HostObject(ref class_host) = class_const.value else {
            panic!("target class must be ConstValue::HostObject");
        };
        // Interned by qualname — every cast site shares one HostObject
        // identity, so `getdesc` resolves them to one ClassDesc.
        let interned = registry
            .bookkeeper()
            .intern_class_by_qualname("W_CastTarget");
        assert_eq!(class_host, &interned);
        assert!(matches!(lowered.args[2], Hlvalue::Variable(ref v) if *v == operand));
    }

    #[test]
    fn translate_op_call_synthetic_transparent_ctor_lowers_to_simple_call() {
        // Call::SyntheticTransparentCtor mirrors Rust's `Class { fields }`
        // ctor — flowspace receives a `simple_call(class_const, fields)`
        // shape just like FunctionPath; rtyper's InstanceRepr handles it.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::SyntheticTransparentCtor {
                    name: "Point".into(),
                    owner_path: Vec::new(),
                },
                args: vec![vars[1].clone()],
                result_ty: ValueType::Ref(None),
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("Call::SyntheticTransparentCtor must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].opname, "simple_call");
        let Hlvalue::Constant(ref callable) = translated[0].args[0] else {
            panic!("simple_call callable must be a Constant");
        };
        let ConstValue::HostObject(ref host) = callable.value else {
            panic!("ctor callable must be ConstValue::HostObject");
        };
        assert_eq!(host.qualname(), "Point");
    }

    #[test]
    fn translate_op_isinstance_lowers_to_flowspace_isinstance() {
        // OpKind::IsInstance arrives from the tuple-struct match
        // cascade (`front/ast.rs:7467`).  Emit a single
        // `isinstance(obj, cls)` flowspace op so the rtyper dispatches
        // to `InstanceRepr::rtype_isinstance`.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 4); // vars[0..4]
        let obj_hl_var = Variable::new();
        let cls_hl_var = Variable::new();
        let result_hl_var = Variable::new();
        value_map.insert(vars[1].clone(), Hlvalue::Variable(obj_hl_var.clone()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(cls_hl_var.clone()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(result_hl_var.clone()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::IsInstance {
                obj: vars[1].clone(),
                class_carrier: vars[2].clone(),
                result_ty: ValueType::Bool,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("OpKind::IsInstance must lower");
        assert_eq!(translated.len(), 1);
        assert_eq!(
            translated[0].opname, "isinstance",
            "IsInstance must emit the flowspace `isinstance` opname",
        );
        assert_eq!(translated[0].args.len(), 2, "isinstance args: [obj, cls]");
        match &translated[0].args[0] {
            Hlvalue::Variable(v) => assert_eq!(v, &obj_hl_var, "args[0] must be obj"),
            other => panic!("args[0] must be Variable, got {other:?}"),
        }
        match &translated[0].args[1] {
            Hlvalue::Variable(v) => {
                assert_eq!(v, &cls_hl_var, "args[1] must be class_carrier")
            }
            other => panic!("args[1] must be Variable, got {other:?}"),
        }
        match &translated[0].result {
            Hlvalue::Variable(v) => {
                assert_eq!(v, &result_hl_var, "result must follow value_map mapping")
            }
            other => panic!("result must be Variable, got {other:?}"),
        }
    }

    #[test]
    fn translate_op_call_method_chains_getattr_simple_call() {
        // Call::Method `obj.method(args)` → 2-op chain `[getattr(obj,
        // "method") -> meth, simple_call(meth, args[1..])]`, mirroring
        // `flowspace/flowcontext.py: LOAD_ATTR + CALL_FUNCTION` shape.
        // args[0] is the receiver (matches Rust method-call lowering).
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new())); // receiver
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new())); // arg
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new())); // result
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::method("push", Some("Vec".into())),
                args: vec![vars[1].clone(), vars[2].clone()],
                result_ty: ValueType::Int,
            },
        };
        let translated =
            translate_op(&op, &value_map, &empty_call_registry()).expect("Call::Method must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getattr");
        assert_eq!(translated[1].opname, "simple_call");
        // getattr's name arg is the method name as a byte string.
        let Hlvalue::Constant(ref name_const) = translated[0].args[1] else {
            panic!("getattr method-name arg must be a Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"push"
        ));
        // Bound-method Variable identity threads from getattr.result into
        // simple_call.args[0].
        let Hlvalue::Variable(ref m1) = translated[0].result else {
            panic!("getattr result must be Variable");
        };
        let Hlvalue::Variable(ref m2) = translated[1].args[0] else {
            panic!("simple_call's first arg must be Variable (bound method)");
        };
        assert_eq!(
            m1.id(),
            m2.id(),
            "bound method Variable identity must thread"
        );
        // simple_call args = [bound_method, args[1..]]
        assert_eq!(translated[1].args.len(), 2);
    }

    #[test]
    fn translate_op_call_indirect_surfaces_rclass_invariant() {
        // Call::Indirect must be lowered to VtableMethodPtr +
        // IndirectCall by `rclass.rs` before reaching the flowspace
        // adapter. Reaching here means the rclass rewrite didn't run;
        // surface the structural invariant break.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::Call {
                target: crate::model::CallTarget::Indirect {
                    trait_root: "MyTrait".into(),
                    method_name: "do_it".into(),
                },
                args: vec![vars[1].clone()],
                result_ty: ValueType::Int,
            },
        };
        let err = translate_op(&op, &value_map, &empty_call_registry())
            .expect_err("Call::Indirect must surface rclass invariant break");
        let msg = format!("{err}");
        assert!(
            msg.contains("Indirect") && msg.contains("rclass"),
            "fail-loud must cite Indirect + rclass.rs, got: {msg}"
        );
    }

    #[test]
    fn translate_op_indirect_call_surfaces_rpbc_invariant() {
        // OpKind::IndirectCall must be lowered by `rpbc.rs:1481-1490`
        // (the rtyper-equivalent layer that owns the graph registry
        // and can resolve CallPath → ConstValue::Graphs(Vec<usize>))
        // before reaching the flowspace adapter. Synthesising
        // `ConstValue::List(byte_str)` here would break
        // `graphanalyze.rs:333` indirect-call analysis (any non-Graphs
        // ConstValue falls back to `top_result()`); fail-loud is the
        // parity-correct behaviour.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::IndirectCall {
                funcptr: vars[1].clone(),
                args: vec![vars[2].clone()],
                graphs: None,
                result_ty: ValueType::Int,
            },
        };
        let err = translate_op(&op, &value_map, &empty_call_registry())
            .expect_err("IndirectCall must surface rpbc.rs invariant break");
        let msg = format!("{err}");
        assert!(
            msg.contains("IndirectCall") && msg.contains("rpbc.rs"),
            "fail-loud must cite IndirectCall + rpbc.rs:1481, got: {msg}"
        );
    }

    #[test]
    fn translate_op_field_read_lowers_to_getattr() {
        // FieldRead → flowspace `getattr(base, ConstValue::ByteStr(name))`
        // mirroring `flowspace/operation.py:617 GetAttr.opname = 'getattr'`.
        // The rtyper later dispatches via `rtype_getattr` based on the
        // base operand's resolved repr (InstanceRepr / etc.).
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let base_var = Hlvalue::Variable(Variable::new());
        let result_var = Hlvalue::Variable(Variable::new());
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), base_var.clone());
        value_map.insert(vars[2].clone(), result_var.clone());

        let op = SpaceOperation {
            result: Some(vars[2].clone()),
            kind: OpKind::FieldRead {
                base: vars[1].clone(),
                field: crate::model::FieldDescriptor::new("f", Some("Owner".into())),
                ty: ValueType::Int,
                pure: false,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("FieldRead arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "getattr");
        assert_eq!(lowered.args.len(), 2);
        let Hlvalue::Constant(ref name_const) = lowered.args[1] else {
            panic!("FieldRead lowering must pass field name as Hlvalue::Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"f"
        ));
    }

    #[test]
    fn translate_op_field_write_lowers_to_setattr() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: vars[1].clone(),
                field: crate::model::FieldDescriptor::new("g", Some("Owner".into())),
                value: crate::model::LinkArg::Value(vars[2].clone()),
                ty: ValueType::Int,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("FieldWrite arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "setattr");
        assert_eq!(lowered.args.len(), 3);
        let Hlvalue::Constant(ref name_const) = lowered.args[1] else {
            panic!("FieldWrite lowering must pass field name as Hlvalue::Constant");
        };
        assert!(matches!(
            name_const.value,
            ConstValue::ByteStr(ref bytes) if bytes == b"g"
        ));
    }

    #[test]
    fn translate_op_array_read_lowers_to_getitem() {
        // ArrayRead → flowspace `getitem(base, index)` mirroring
        // `flowspace/operation.py: GetItem.opname = 'getitem'`. RTyper's
        // `rtype_getitem` dispatches via the receiver's resolved repr
        // (ListRepr / TupleRepr / FixedSizeArrayRepr) and lowers to
        // `getarrayitem_gc_*`.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::ArrayRead {
                base: vars[1].clone(),
                index: vars[2].clone(),
                item_ty: ValueType::Int,
                array_type_id: None,
                nolength: false,
                pure: false,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("ArrayRead arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "getitem");
        assert_eq!(lowered.args.len(), 2);
    }

    #[test]
    fn translate_op_array_write_lowers_to_setitem() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: None,
            kind: OpKind::ArrayWrite {
                base: vars[1].clone(),
                index: vars[2].clone(),
                value: crate::model::LinkArg::Value(vars[3].clone()),
                item_ty: ValueType::Int,
                array_type_id: None,
                nolength: false,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("ArrayWrite arm must lower");
        assert_eq!(translated.len(), 1);
        let lowered = &translated[0];
        assert_eq!(lowered.opname, "setitem");
        assert_eq!(lowered.args.len(), 3);
    }

    #[test]
    fn translate_op_interior_field_read_unfolds_to_getitem_getattr_chain() {
        // InteriorFieldRead → `getitem(base, index)` chained into
        // `getattr(elem, field_name)`, mirroring `effectinfo.py:313-340`'s
        // implicit `readarray + readinteriorfield` effects. Two flowspace
        // ops surface from one legacy op; the rtyper sees the chain.
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: Some(vars[3].clone()),
            kind: OpKind::InteriorFieldRead {
                base: vars[1].clone(),
                index: vars[2].clone(),
                field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                item_ty: ValueType::Int,
                array_type_id: None,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("InteriorFieldRead arm must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getitem");
        assert_eq!(translated[0].args.len(), 2);
        assert_eq!(translated[1].opname, "getattr");
        assert_eq!(translated[1].args.len(), 2);
        // The second op's first arg must be the element variable
        // produced by the first op.
        let Hlvalue::Variable(ref elem_v1) = translated[0].result else {
            panic!("getitem result must be a Variable");
        };
        let Hlvalue::Variable(ref elem_v2) = translated[1].args[0] else {
            panic!("getattr base arg must be the chained element Variable");
        };
        assert_eq!(
            elem_v1.id(),
            elem_v2.id(),
            "elem Variable identity must thread"
        );
    }

    #[test]
    fn op_canraise_covers_getitem_bearing_and_inplace_binops() {
        // `InteriorField*` unfolds to a getitem (raising) + getattr/setattr,
        // so as a `?` tail it carries the getitem's `[IndexError, KeyError,
        // Exception]` (operation.py:727-730).
        let interior = OpKind::InteriorFieldRead {
            base: Variable::new(),
            index: Variable::new(),
            field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
            item_ty: ValueType::Int,
            array_type_id: None,
        };
        assert!(
            op_canraise(&interior),
            "InteriorFieldRead carries the unfolded getitem's canraise"
        );

        // Compound-assign names reach `op_canraise` BEFORE
        // `normalize_binop_name` maps them to `inplace_*`.
        // `inplace_div`/`inplace_add`/`inplace_lshift` carry
        // ZeroDivisionError/OverflowError/ValueError; `inplace_and` and
        // plain `add` are `[]` (operation.py:751-756).
        let binop = |name: &str| OpKind::BinOp {
            op: name.to_string(),
            lhs: Variable::new(),
            rhs: Variable::new(),
            result_ty: ValueType::Int,
        };
        assert!(op_canraise(&binop("div_assign")));
        assert!(op_canraise(&binop("add_assign")));
        assert!(op_canraise(&binop("lshift_assign")));
        assert!(op_canraise(&binop("div")));
        assert!(!op_canraise(&binop("bitand_assign")));
        assert!(!op_canraise(&binop("add")));

        // `core::*` method bridges that `translate_op` lowers to pure
        // flowspace ops must classify as non-raising, so a `?` tail does
        // not install an unreachable exception edge.  Drift here is what
        // `nonraising_core_bridge_opname` exists to prevent.
        let core_call = |segs: &[&str], argc: usize| OpKind::Call {
            target: crate::model::CallTarget::FunctionPath {
                segments: segs.iter().map(|s| s.to_string()).collect(),
            },
            args: (0..argc).map(|_| Variable::new()).collect(),
            result_ty: ValueType::Int,
        };
        assert!(!op_canraise(&core_call(
            &["core", "cmp", "PartialEq", "eq"],
            2
        )));
        assert!(!op_canraise(&core_call(
            &["core", "cmp", "PartialOrd", "lt"],
            2
        )));
        assert!(!op_canraise(&core_call(&["core", "slice", "len"], 1)));
        assert!(!op_canraise(&core_call(&["core", "slice", "iter"], 1)));
        assert!(!op_canraise(&core_call(
            &["core", "num", "wrapping_mul"],
            2
        )));
        // `min`/`max` lower to a raising `simple_call`, and a wrong arg
        // count falls through to the general raising `Call` arm.
        assert!(op_canraise(&core_call(&["core", "cmp", "min"], 2)));
        assert!(op_canraise(&core_call(&["core", "cmp", "max"], 2)));
        assert!(op_canraise(&core_call(&["core", "slice", "len"], 2)));
    }

    #[test]
    fn translate_op_interior_field_write_unfolds_to_getitem_setattr_chain() {
        let mut value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 11); // vars[0..11]
        value_map.insert(vars[1].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[2].clone(), Hlvalue::Variable(Variable::new()));
        value_map.insert(vars[3].clone(), Hlvalue::Variable(Variable::new()));
        let op = SpaceOperation {
            result: None,
            kind: OpKind::InteriorFieldWrite {
                base: vars[1].clone(),
                index: vars[2].clone(),
                field: crate::model::FieldDescriptor::new("y", Some("Point".into())),
                value: vars[3].clone(),
                item_ty: ValueType::Int,
                array_type_id: None,
            },
        };
        let translated = translate_op(&op, &value_map, &empty_call_registry())
            .expect("InteriorFieldWrite arm must lower");
        assert_eq!(translated.len(), 2);
        assert_eq!(translated[0].opname, "getitem");
        assert_eq!(translated[1].opname, "setattr");
        assert_eq!(translated[1].args.len(), 3);
    }

    #[test]
    fn translate_op_undefined_operand_surfaces_invariant_break() {
        // The lookup_operand helper is shared across every arm with
        // operands. Validate it surfaces a clear "adapter
        // invariant broken" message and embeds the enriched diagnostic
        // context (op variant + arg role).
        let value_map: HashMap<Variable, Hlvalue> = HashMap::new();
        let mut graph = LegacyGraph::new("translate_op_fixture");
        let vars = mint_vars(&mut graph, 101); // vars[0..101]
        let op = SpaceOperation {
            result: Some(vars[100].clone()),
            kind: OpKind::BinOp {
                op: "add".to_string(),
                lhs: vars[99].clone(),
                rhs: vars[0].clone(),
                result_ty: ValueType::Int,
            },
        };
        let err = lookup_operand(&value_map, &vars[99], &op, "lhs")
            .expect_err("undefined operand lookup must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("undefined operand") && msg.contains("invariant"),
            "fail-loud message must explain the invariant, got: {msg}"
        );
        assert!(
            msg.contains("as lhs of BinOp") && msg.contains("result Some(Variable("),
            "verbose diagnostic must include arg role + op variant + result variable, got: {msg}"
        );
    }

    // ───── topology assembly ─────

    fn link_to_returnblock(args: Vec<LinkArg>, returnblock_id: BlockId) -> crate::model::Link {
        let mut link = crate::model::Link::new_mixed(args, returnblock_id, None);
        link.prevblock = None;
        link
    }

    fn legacy_minimal_identity_return_graph()
    -> (LegacyGraph, Vec<crate::flowspace::model::Variable>) {
        // Smallest valid legacy graph: one inputarg, returns it
        // directly. Must produce a flowspace::FunctionGraph
        // whose startblock has the single inputarg Variable,
        // exits→returnblock, and the returnblock's inputarg is the same
        // Variable identity (so RPythonTyper.getreturnvar resolves
        // correctly).
        //
        // RPython convention: returnblock canonically has one inputarg
        // (`flowspace/model.py:13-18`). True void returns use a
        // `SomeNone` / `Void`-typed argument; pyre's legacy graph
        // mirrors that by always emitting a single slot in the
        // returnblock's inputargs.
        let mut graph = LegacyGraph::new("identity_return");
        let vars = mint_vars(&mut graph, 2); // vars[0..2]
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(vars[1].clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];
        (graph, vars)
    }

    #[test]
    fn function_graph_to_flowspace_minimal_identity_return_assembles_graph() {
        let (legacy, vars) = legacy_minimal_identity_return_graph();

        let output = function_graph_to_flowspace(&legacy, &empty_call_registry())
            .expect("minimal graph must assemble");

        // value_to_var must contain the inputarg.
        assert!(
            output.value_to_var.contains_key(&vars[1]),
            "value_to_var must seed the legacy inputarg"
        );

        // block_map must contain startblock + returnblock + exceptblock.
        assert!(output.block_map.contains_key(&legacy.startblock));
        assert!(output.block_map.contains_key(&legacy.returnblock));
        assert!(output.block_map.contains_key(&legacy.exceptblock));

        // The flowspace graph's startblock is the same BlockRef as the
        // mapped legacy.startblock.
        let graph = output.graph.borrow();
        assert!(Rc::ptr_eq(
            &graph.startblock,
            &output.block_map[&legacy.startblock],
        ));

        // The flowspace graph's returnblock is the mapped legacy.returnblock.
        assert!(Rc::ptr_eq(
            &graph.returnblock,
            &output.block_map[&legacy.returnblock],
        ));

        // The startblock has exactly one exit, targeting the returnblock.
        let startblock = graph.startblock.borrow();
        assert_eq!(startblock.exits.len(), 1);
        let exit = startblock.exits[0].borrow();
        assert!(Rc::ptr_eq(
            exit.target.as_ref().expect("link must have target"),
            &graph.returnblock,
        ));
    }

    #[test]
    fn function_graph_to_flowspace_returnvar_identity_preserved() {
        // When the returnblock has an inputarg slot, the flowspace
        // graph's returnblock must use the SAME Variable identity (so
        // RPythonTyper.getreturnvar finds the right Variable —
        // rtyper.rs:1633-1638).
        let mut graph = LegacyGraph::new("with_return_var");
        let vars = mint_vars(&mut graph, 3); // vars[0..3]
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(vars[1].clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[2]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let output = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect("graph must assemble");

        // The flowspace returnblock's single inputarg must be the
        // Variable we seeded for slot 2.
        let flowspace_graph = output.graph.borrow();
        let returnblock_ref = &flowspace_graph.returnblock;
        let returnblock = returnblock_ref.borrow();
        assert_eq!(
            returnblock.inputargs.len(),
            1,
            "returnblock must carry the single return-value inputarg"
        );
        match &returnblock.inputargs[0] {
            Hlvalue::Variable(v) => {
                let expected = &output.value_to_var[&vars[2]];
                assert_eq!(
                    v, expected,
                    "returnblock inputarg must preserve Variable identity from value_to_var"
                );
            }
            other => panic!("returnblock inputarg must be a Variable, got {other:?}"),
        }
    }

    #[test]
    fn function_graph_to_flowspace_const_define_op_inlined_in_link_args() {
        // ConstInt(7) defines vars[2]. build_value_to_hlvalue_map
        // inlines it into Link.args as Hlvalue::Constant — link
        // translation must use that
        // mapping rather than wrapping the unused Variable.
        let mut graph = LegacyGraph::new("const_link_arg");
        let vars = mint_vars(&mut graph, 4); // vars[0..4]
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![SpaceOperation {
                result: Some(vars[2].clone()),
                kind: OpKind::ConstInt(7),
            }],
            exitswitch: None,
            // Return vars[2], the ConstInt define.
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(vars[2].clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[3]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let output = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect("graph must assemble");

        let flowspace_graph = output.graph.borrow();
        let startblock = flowspace_graph.startblock.borrow();
        // ConstInt define is a skip arm — operations must be empty.
        assert!(
            startblock.operations.is_empty(),
            "ConstInt define has no flowspace::SpaceOperation analogue"
        );
        // The exit's link arg must be the inlined Constant.
        let exit = startblock.exits[0].borrow();
        assert_eq!(exit.args.len(), 1);
        match exit.args[0].as_ref().expect("link arg is Some") {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(n) => assert_eq!(*n, 7),
                other => panic!("link arg constant must be Int, got {other:?}"),
            },
            other => panic!("link arg must be Hlvalue::Constant, got {other:?}"),
        }
    }

    #[test]
    fn function_graph_to_flowspace_exception_link_materialises_extravars_before_args() {
        // RPython checkgraph defines exception-link extravars before
        // validating link.args. Pyre's legacy graph represents those
        // as fresh slots whose only definition site is the link.
        let mut graph = LegacyGraph::new("canraise_with_extravars");
        let vars = mint_vars(&mut graph, 12); // vars[0..12], extravars live at 10/11
        let startblock = Block {
            id: graph.startblock,
            inputargs: block_inputargs(&vars, &[1, 2]),
            operations: vec![SpaceOperation {
                result: Some(vars[3].clone()),
                kind: OpKind::BinOp {
                    op: "add".to_string(),
                    lhs: vars[1].clone(),
                    rhs: vars[2].clone(),
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: Some(crate::model::ExitSwitch::LastException),
            exits: vec![
                link_to_returnblock(vec![LinkArg::Value(vars[3].clone())], graph.returnblock),
                crate::model::Link::new_mixed(
                    vec![
                        LinkArg::Value(vars[10].clone()),
                        LinkArg::Value(vars[11].clone()),
                    ],
                    graph.exceptblock,
                    Some(crate::model::exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::Value(vars[10].clone())),
                    Some(LinkArg::Value(vars[11].clone())),
                ),
            ],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[4]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        let exceptblock = Block {
            id: graph.exceptblock,
            inputargs: block_inputargs(&vars, &[10, 11]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock, exceptblock];

        let output = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect("exception graph assembles");
        let flowspace_graph = output.graph.borrow();
        let startblock = flowspace_graph.startblock.borrow();
        let exc_link = startblock.exits[1].borrow();
        assert!(exc_link.last_exception.is_some());
        assert!(exc_link.last_exc_value.is_some());
        assert_eq!(
            exc_link.args[0].as_ref(),
            exc_link.last_exception.as_ref(),
            "exception type arg must reuse link.last_exception Variable"
        );
        assert_eq!(
            exc_link.args[1].as_ref(),
            exc_link.last_exc_value.as_ref(),
            "exception value arg must reuse link.last_exc_value Variable"
        );
    }

    #[test]
    fn function_graph_to_flowspace_unported_opkind_surfaces_failloud() {
        // A graph carrying a still-fail-loud OpKind (Call::Indirect —
        // requires rclass.rs lowering to VtableMethodPtr + IndirectCall
        // before reaching the adapter) must surface that op's
        // translate_op error from inside Pass 2, not silently emit a
        // partial graph.
        let mut graph = LegacyGraph::new("unported_op");
        let vars = mint_vars(&mut graph, 4); // vars[0..4]
        let inputargs = block_inputargs(&vars, &[1]);
        let arg_var = vars[1].clone();
        let startblock = Block {
            id: graph.startblock,
            inputargs,
            operations: vec![SpaceOperation {
                result: Some(vars[2].clone()),
                kind: OpKind::Call {
                    target: crate::model::CallTarget::Indirect {
                        trait_root: "MyTrait".into(),
                        method_name: "do_it".into(),
                    },
                    args: vec![arg_var],
                    result_ty: ValueType::Int,
                },
            }],
            exitswitch: None,
            exits: vec![link_to_returnblock(
                vec![LinkArg::Value(vars[2].clone())],
                graph.returnblock,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: graph.returnblock,
            inputargs: block_inputargs(&vars, &[3]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        graph.blocks = vec![startblock, returnblock];

        let err = function_graph_to_flowspace(&graph, &empty_call_registry())
            .expect_err("unported OpKind must surface as TyperError");
        let msg = format!("{err}");
        assert!(
            msg.contains("Indirect") && msg.contains("rclass"),
            "unported-op error must propagate translate_op's fail-loud message, got: {msg}"
        );
    }

    #[test]
    fn function_graph_to_flowspace_skips_unreachable_raise_block() {
        // A predecessor-less unwind block carrying `set_raise`'s orphan
        // `[etype, evalue]` Link.args — the MIR lowering shape left
        // behind when every inbound `on_unwind` edge is dropped — must
        // not reject the graph: only the reachable closure is
        // converted, mirroring upstream `iterblocks` (a flowspace
        // graph cannot contain unreachable blocks at all).
        let (mut legacy, _vars) = legacy_minimal_identity_return_graph();
        let orphan_etype = legacy.alloc_value_var();
        let orphan_evalue = legacy.alloc_value_var();
        let exceptblock = legacy.exceptblock;
        legacy.blocks.push(Block {
            id: BlockId(3),
            inputargs: vec![],
            operations: vec![],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![LinkArg::Value(orphan_etype), LinkArg::Value(orphan_evalue)],
                exceptblock,
                None,
            )],
            framestate: None,
            dead: false,
        });

        let output = function_graph_to_flowspace(&legacy, &empty_call_registry())
            .expect("unreachable raise block must not reject the graph");
        // start → return only; the orphan block was never converted.
        assert_eq!(
            output.graph.borrow().iterblocks().len(),
            2,
            "converted graph must contain exactly the reachable closure"
        );
    }

    #[test]
    fn function_graph_to_flowspace_rejects_reachable_orphan_raise_args() {
        // A *reachable* block whose exception-link args are defined by
        // no inputarg and no op result must still fail loud — the
        // reachable-closure conversion does not relax SSA-definedness.
        let mut legacy = LegacyGraph::new("reachable_orphan_raise");
        let vars = mint_vars(&mut legacy, 9);
        let orphan_etype = vars[7].clone();
        let orphan_evalue = vars[8].clone();
        let exceptblock = legacy.exceptblock;
        let startblock = Block {
            id: legacy.startblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![crate::model::Link::new_mixed(
                vec![LinkArg::Value(orphan_etype), LinkArg::Value(orphan_evalue)],
                exceptblock,
                None,
            )],
            framestate: None,
            dead: false,
        };
        let returnblock = Block {
            id: legacy.returnblock,
            inputargs: block_inputargs(&vars, &[1]),
            operations: vec![],
            exitswitch: None,
            exits: vec![],
            framestate: None,
            dead: false,
        };
        legacy.blocks = vec![startblock, returnblock];

        let err = function_graph_to_flowspace(&legacy, &empty_call_registry())
            .expect_err("reachable orphan raise args must stay fail-loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("undefined operand"),
            "reachable orphan must keep the operand-definedness error, got: {msg}"
        );
    }

    #[test]
    fn opkind_variant_name_covers_core_variants() {
        // Sanity: variant_name maps each core OpKind to a stable string.
        // Any new OpKind variant added without a corresponding arm here
        // surfaces as "<unknown OpKind variant>" in fail-loud messages,
        // which prompts the developer to extend this table.
        assert_eq!(opkind_variant_name(&OpKind::ConstInt(0)), "ConstInt");
        assert_eq!(opkind_variant_name(&OpKind::ConstFloat(0)), "ConstFloat");
        assert_eq!(
            opkind_variant_name(&OpKind::Input {
                name: "x".into(),
                ty: ValueType::Int,
                class_root: None,
            }),
            "Input"
        );
    }
}
