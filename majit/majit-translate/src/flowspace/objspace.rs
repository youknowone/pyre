//! Main interface for flow graph creation: [`build_flow`].
//!
//! RPython upstream: `rpython/flowspace/objspace.py` (53 LOC).
//!
//! Rust adaptation (parity rule #1):
//!
//! * Upstream `_assert_rpythonic(func)` inspects `func.__code__.co_cellvars`,
//!   `func._not_rpython_`, `func.__doc__`, and `CO_NEWLOCALS` on the
//!   live Python function. The Rust port reads each of those through
//!   the parsed [`HostCode`] / [`GraphFunc`] counterparts:
//!   - `func.__code__.co_cellvars` → `HostCode.co_cellvars`
//!     (populated from `rustpython_compiler_core::CodeObject.cellvars`).
//!   - `func._not_rpython_` → `GraphFunc.not_rpython` — the
//!     `@not_rpython` decorator attribute lifted to a bool field.
//!   - `func.__doc__` starting with `"NOT_RPYTHON"` →
//!     `HostCode.consts.first()` when that constant is a string. This
//!     mirrors CPython's convention that a function's docstring is
//!     stored at `co_consts[0]` when it's a bare string literal.
//! * Upstream returns the *same* `FunctionGraph` instance that
//!   `PyGraph(func, code)` constructed. The Rust port re-exports
//!   `ctx.graph` after `FlowContext::build_flow()` finishes.

use super::bytecode::{ConstantData, HostCode};
use super::flowcontext::{FlowContext, FlowContextError, fixeggblocks};
use super::generator::{make_generator_entry_graph, tweak_generator_graph};
use super::model::{FunctionGraph, GraphFunc};
use super::pygraph::PyGraph;

/// RPython `CO_NEWLOCALS` compile flag (0x0002). Used to verify that a
/// code object allocates its own `f_locals` dict rather than sharing
/// the caller's — RPython functions must always set this.
pub const CO_NEWLOCALS: u32 = 0x0002;

/// RPython `objspace.py:14-35` — `_assert_rpythonic(func)`.
///
/// Raises a structural error when `func` cannot be flow-analysed.
/// Equivalent upstream exceptions collapse into [`FlowContextError`]
/// variants here.
pub fn assert_rpythonic(func: &GraphFunc) -> Result<(), FlowContextError> {
    // upstream lines 16-20: `try: func.__code__.co_cellvars except
    // AttributeError: raise ValueError(...)`. In Rust the code slot is
    // `Option<Box<HostCode>>`; a missing `code` maps to the same error.
    let code = func.code.as_ref().ok_or_else(|| {
        FlowContextError::Flowing(super::flowcontext::FlowingError::new(format!(
            "{} is not RPython: it is likely an unexpected built-in \
             function or type",
            func.name
        )))
    })?;

    // upstream line 21-22: `if getattr(func, "_not_rpython_", False):
    //                          raise ValueError(...)`.
    if func.not_rpython {
        return Err(FlowContextError::Flowing(
            super::flowcontext::FlowingError::new(format!(
                "{} is tagged as @not_rpython",
                func.name
            )),
        ));
    }

    // upstream line 23-24: `if func.__doc__ and
    //         func.__doc__.lstrip().startswith('NOT_RPYTHON'): raise`.
    // CPython stores the docstring at `co_consts[0]` when it is a bare
    // string literal.
    if let Some(ConstantData::Str { value, .. }) = code.consts.first() {
        if value.trim_start().starts_with("NOT_RPYTHON") {
            return Err(FlowContextError::Flowing(
                super::flowcontext::FlowingError::new(format!(
                    "{} is tagged as NOT_RPYTHON",
                    func.name
                )),
            ));
        }
    }

    // upstream line 25-32: `if func.__code__.co_cellvars: raise
    //     ValueError("RPython functions cannot create closures ...")`.
    if !code.co_cellvars.is_empty() {
        return Err(FlowContextError::Flowing(
            super::flowcontext::FlowingError::new(format!(
                "RPython functions cannot create closures\n\
                 Possible causes:\n\
                 \x20   Function is inner function\n\
                 \x20   Function uses generator expressions\n\
                 \x20   Lambda expressions\n\
                 in {}",
                func.name
            )),
        ));
    }

    // upstream line 33-35: `if not (func.__code__.co_flags & CO_NEWLOCALS)`.
    if code.co_flags & CO_NEWLOCALS == 0 {
        return Err(FlowContextError::Flowing(
            super::flowcontext::FlowingError::new(
                "The code object for a RPython function should have \
                 the flag CO_NEWLOCALS set.",
            ),
        ));
    }

    Ok(())
}

/// RPython `objspace.py:38-53` — `build_flow(func)`.
///
/// Create the flow graph (in SSA form) for the function. Ownership of
/// the returned `FunctionGraph` transfers to the caller.
pub fn build_flow(func: GraphFunc) -> Result<FunctionGraph, FlowContextError> {
    assert_rpythonic(&func)?;

    let code: HostCode = *func
        .code
        .clone()
        .expect("assert_rpythonic guarantees code is Some");

    // upstream: `if isgeneratorfunction(func) and
    //              not hasattr(func, '_generator_next_method_of_'):
    //                return make_generator_entry_graph(func)`.
    //
    // HostCode.is_generator() reads the CO_GENERATOR flag — the Rust
    // equivalent of CPython's `inspect.isgeneratorfunction`.
    if code.is_generator() && func._generator_next_method_of_.is_none() {
        return make_generator_entry_graph(func);
    }

    let pygraph = PyGraph::new(func, &code);
    // pygraph.graph is an `Rc<RefCell<FunctionGraph>>` for annotator
    // sharing; here we own the only reference, so unwrap back to the
    // value-form FlowContext expects.
    let graph = std::rc::Rc::try_unwrap(pygraph.graph)
        .expect("objspace.build_flow: PyGraph.graph should be unique")
        .into_inner();
    let mut ctx = FlowContext::new(graph, code);
    ctx.build_flow()?;
    fixeggblocks(&mut ctx.graph);

    if ctx.pycode.is_generator() {
        tweak_generator_graph(&mut ctx.graph)?;
    }

    Ok(ctx.graph)
}

#[cfg(test)]
mod tests {
    use super::super::bytecode::CO_GENERATOR;
    use super::super::model::{ConstValue, Constant};
    use super::*;

    fn empty_globals() -> Constant {
        Constant::new(ConstValue::Dict(Default::default()))
    }

    #[test]
    fn assert_rpythonic_rejects_missing_code() {
        // GraphFunc default has no code attached.
        let func = GraphFunc::new("f", empty_globals());
        let err = assert_rpythonic(&func).unwrap_err();
        match err {
            FlowContextError::Flowing(err) => {
                assert!(err.message.contains("is not RPython"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn assert_rpythonic_rejects_closure_bearing_func() {
        let mut func = GraphFunc::new("inner", empty_globals());
        // attach a dummy HostCode with CO_NEWLOCALS set so co_cellvars
        // is the check that fires first.
        let mut code = make_empty_host_code();
        code.co_flags = CO_NEWLOCALS;
        code.co_cellvars.push("x".to_string());
        func.code = Some(Box::new(code));

        let err = assert_rpythonic(&func).unwrap_err();
        match err {
            FlowContextError::Flowing(err) => {
                assert!(err.message.contains("cannot create closures"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn assert_rpythonic_rejects_not_rpython_marker() {
        let mut func = GraphFunc::new("marked", empty_globals());
        let mut code = make_empty_host_code();
        code.co_flags = CO_NEWLOCALS;
        func.code = Some(Box::new(code));
        func.not_rpython = true;

        let err = assert_rpythonic(&func).unwrap_err();
        match err {
            FlowContextError::Flowing(err) => {
                assert!(err.message.contains("@not_rpython"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn assert_rpythonic_rejects_not_rpython_docstring() {
        let mut func = GraphFunc::new("doc", empty_globals());
        let mut code = make_empty_host_code();
        code.co_flags = CO_NEWLOCALS;
        code.consts.push(ConstantData::Str {
            value: "NOT_RPYTHON: skip me".to_string().into(),
        });
        func.code = Some(Box::new(code));

        let err = assert_rpythonic(&func).unwrap_err();
        match err {
            FlowContextError::Flowing(err) => {
                assert!(err.message.contains("NOT_RPYTHON"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn assert_rpythonic_rejects_missing_co_newlocals() {
        let mut func = GraphFunc::new("toplevel", empty_globals());
        let mut code = make_empty_host_code();
        code.co_flags = 0; // no CO_NEWLOCALS
        func.code = Some(Box::new(code));

        let err = assert_rpythonic(&func).unwrap_err();
        match err {
            FlowContextError::Flowing(err) => {
                assert!(err.message.contains("CO_NEWLOCALS"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn build_flow_bootstraps_generator_entry_graph() {
        let mut func = GraphFunc::new("gen", empty_globals());
        let mut code = make_empty_host_code();
        code.co_flags = CO_NEWLOCALS | CO_GENERATOR;
        func.code = Some(Box::new(code));

        let graph = build_flow(func).expect("generator bootstrap graph");
        let ops = &graph.startblock.borrow().operations;
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opname, "simple_call");
        assert_eq!(ops[1].opname, "simple_call");
    }

    fn make_empty_host_code() -> HostCode {
        use super::super::argument::Signature;
        HostCode {
            id: HostCode::fresh_identity(),
            co_name: "f".to_string(),
            co_filename: "<test>".to_string(),
            co_firstlineno: 1,
            co_nlocals: 0,
            co_argcount: 0,
            co_stacksize: 0,
            co_flags: 0,
            co_code: rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            co_varnames: Vec::new(),
            co_freevars: Vec::new(),
            co_cellvars: Vec::new(),
            consts: Vec::new(),
            names: Vec::new(),
            co_lnotab: Vec::new(),
            exceptiontable: Vec::new().into_boxed_slice(),
            signature: Signature::new(Vec::new(), None, None),
        }
    }
}
