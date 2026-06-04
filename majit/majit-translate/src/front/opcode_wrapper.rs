//! Mechanically-synthesized opcode-arm wrapper graphs.
//!
//! The opcode dispatcher (`execute_opcode_step`) is one big Rust `match`
//! whose every arm is one of three closed shapes:
//!
//! - **tail-call** — `execute_<op>(dispatcher params)`,
//! - **wrapped-unit-variant return** — `Ok(StepResult::Continue)`,
//! - **raise-stub return** — `Err(PyError::type_error("…").into())`.
//!
//! Each shape lowers to a fixed dispatcher-shaped wrapper graph (the
//! same five `OpKind::Input` prologue followed by a shape-specific tail).
//! These builders are the single source of truth for that lowering so
//! the syn-AST dispatch extractor (`parse::extract_match_arms`) and the
//! MIR dispatch extractor (`front::mir_dispatch`) produce byte-identical
//! body graphs from their two different front-ends.  The builders carry
//! no `syn` dependency: callers hand them the already-extracted
//! `(param name, ValueType)` prologue and the shape data.

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, OpKind, UnsupportedLiteralKind, ValueType};
use crate::parse::CallPath;

/// A constant-return arm body of the form `Wrapper(Owner::Variant)` —
/// e.g. `Ok(StepResult::Continue)`.  `wrapper_*` is the transparent
/// result/option ctor (`Ok`/`Err`/`Some`); `inner_*` is the synthetic
/// unit-variant ctor it wraps.  Carries the owner/leaf split so the
/// builder emits the exact two-`Call` chain directly.
pub(crate) struct WrappedUnitVariantReturn {
    pub(crate) wrapper_owner: Vec<String>,
    pub(crate) wrapper_name: String,
    pub(crate) inner_owner: Vec<String>,
    pub(crate) inner_name: String,
}

/// A raise-stub arm body of the form `Wrapper(FnPath(<lit>).method())` —
/// e.g. `Err(crate::PyError::type_error("async not yet implemented").into())`.
/// The string-literal argument lowers to an untranslatable `Abort` (so
/// the opcode always deopts to the interpreter rather than JITing),
/// wrapped by the constructor function call, an adapter method call, and
/// a transparent result/option ctor.
pub(crate) struct RaiseStubReturn {
    pub(crate) fn_segments: Vec<String>,
    pub(crate) method_name: String,
    pub(crate) wrapper_owner: Vec<String>,
    pub(crate) wrapper_name: String,
    pub(crate) literal: UnsupportedLiteralKind,
}

/// Push one `OpKind::Input` per dispatcher parameter (in order, typed by
/// the caller) as the startblock inputargs — the shared prologue of
/// every synthesized opcode-arm wrapper.  Returns the ordered
/// `(param-name, var)` pairs so a tail-call builder can forward the
/// requested subset by a linear scan over the fixed dispatcher param
/// list (the same shape as `mir_dispatch`'s `input_names`).  The
/// runtime seeds the dispatch entry from the full dispatcher register
/// layout, so a wrapper must keep every dispatcher param as an inputarg
/// even when the body reads only a subset (or none).
fn push_dispatcher_inputargs(
    graph: &mut FunctionGraph,
    params: &[(String, ValueType)],
) -> Vec<(String, Variable)> {
    let block = graph.startblock;
    let mut param_vars: Vec<(String, Variable)> = Vec::with_capacity(params.len());
    for (pname, pty) in params {
        if let Some(var) = graph.push_op_var(
            block,
            OpKind::Input {
                name: pname.clone(),
                ty: pty.clone(),
                class_root: None,
            },
            true,
        ) {
            graph.name_value_var(&var, pname.clone());
            graph.push_inputarg_var(block, var.clone());
            param_vars.push((pname.clone(), var));
        }
    }
    param_vars
}

/// Build the dispatcher-shaped wrapper graph for a single-tail-call arm.
/// Shape: one `OpKind::Input` per dispatcher parameter (the startblock
/// inputargs), then one `OpKind::Call` to `handler_path` forwarding the
/// mapped inputarg vars with `result_ty: Unknown`, then a return of the
/// call result.
pub(crate) fn build_tail_call_wrapper(
    name: &str,
    params: &[(String, ValueType)],
    handler_path: &CallPath,
    forwarded: &[String],
) -> FunctionGraph {
    let mut graph = FunctionGraph::new(name.to_string());
    let block = graph.startblock;
    let param_vars = push_dispatcher_inputargs(&mut graph, params);
    let args: Vec<Variable> = forwarded
        .iter()
        .map(|pname| {
            param_vars
                .iter()
                .find(|(n, _)| n == pname)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| {
                    panic!(
                        "opcode arm `{name}`: tail-call forwards `{pname}`, \
                         which is not a dispatcher parameter"
                    )
                })
        })
        .collect();
    let result = graph.push_op_var(
        block,
        OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: handler_path.segments.clone(),
            },
            args,
            result_ty: ValueType::Unknown,
        },
        true,
    );
    graph.set_return(block, result);
    graph
}

/// Build the dispatcher-shaped wrapper graph for a constant-return arm
/// (`Wrapper(Owner::Variant)`, e.g. `Ok(StepResult::Continue)`).  Shape:
/// the dispatcher inputargs, then a 0-arg `SyntheticTransparentCtor` Call
/// for the inner unit-variant, then a 1-arg `SyntheticTransparentCtor`
/// Call for the outer wrapper, then a return of the wrapper result.  Both
/// ctor calls carry `result_ty: Unknown`; downstream `unit_variant_fold`
/// folds the inner to a `ConstRef` and `jtransform` elides the
/// transparent wrapper.
pub(crate) fn build_wrapped_unit_variant_wrapper(
    name: &str,
    params: &[(String, ValueType)],
    ret: &WrappedUnitVariantReturn,
) -> FunctionGraph {
    let mut graph = FunctionGraph::new(name.to_string());
    push_dispatcher_inputargs(&mut graph, params);
    let block = graph.startblock;
    let inner = graph
        .push_op_var(
            block,
            OpKind::Call {
                target: CallTarget::synthetic_transparent_ctor_with_owner(
                    ret.inner_owner.clone(),
                    ret.inner_name.clone(),
                ),
                args: Vec::new(),
                result_ty: ValueType::Unknown,
            },
            true,
        )
        .expect("SyntheticTransparentCtor Call has has_result=true");
    let outer = graph.push_op_var(
        block,
        OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor_with_owner(
                ret.wrapper_owner.clone(),
                ret.wrapper_name.clone(),
            ),
            args: vec![inner],
            result_ty: ValueType::Unknown,
        },
        true,
    );
    graph.set_return(block, outer);
    graph
}

/// Build the dispatcher-shaped wrapper graph for a raise-stub arm
/// (`Wrapper(FnPath(<lit>).method())`, e.g.
/// `Err(crate::PyError::type_error("…").into())`).  Shape: the dispatcher
/// inputargs, then an `Abort` for the untranslatable literal, then a
/// `FunctionPath` Call (the ctor fn) consuming it, then a `Method` Call
/// (the `.into()` adapter), then the transparent result/option wrapper
/// ctor, then a return of the wrapper result.  The `Abort` makes any
/// trace that reaches this opcode deopt to the interpreter — correct for
/// stub opcodes that always raise.
pub(crate) fn build_raise_stub_wrapper(
    name: &str,
    params: &[(String, ValueType)],
    stub: &RaiseStubReturn,
) -> FunctionGraph {
    use crate::model::UnknownKind;
    let mut graph = FunctionGraph::new(name.to_string());
    push_dispatcher_inputargs(&mut graph, params);
    let block = graph.startblock;
    let literal = graph
        .push_op_var(
            block,
            OpKind::Abort {
                kind: UnknownKind::UnsupportedLiteral {
                    variant: stub.literal.clone(),
                },
            },
            true,
        )
        .expect("Abort has has_result=true");
    let constructed = graph
        .push_op_var(
            block,
            OpKind::Call {
                target: CallTarget::function_path(stub.fn_segments.clone()),
                args: vec![literal],
                result_ty: ValueType::Unknown,
            },
            true,
        )
        .expect("FunctionPath Call has has_result=true");
    let converted = graph
        .push_op_var(
            block,
            OpKind::Call {
                target: CallTarget::method(stub.method_name.clone(), None),
                args: vec![constructed],
                result_ty: ValueType::Unknown,
            },
            true,
        )
        .expect("Method Call has has_result=true");
    let wrapped = graph.push_op_var(
        block,
        OpKind::Call {
            target: CallTarget::synthetic_transparent_ctor_with_owner(
                stub.wrapper_owner.clone(),
                stub.wrapper_name.clone(),
            ),
            args: vec![converted],
            result_ty: ValueType::Unknown,
        },
        true,
    );
    graph.set_return(block, wrapped);
    graph
}
