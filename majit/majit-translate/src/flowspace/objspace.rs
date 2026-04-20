//! Main interface for flow graph creation: [`build_flow`].
//!
//! RPython upstream: `rpython/flowspace/objspace.py` (53 LOC).
//!
//! Rust adaptation (parity rule #1):
//!
//! * Upstream `_assert_rpythonic(func)` inspects `func.__code__.co_cellvars`,
//!   `func._not_rpython_`, `func.__doc__`, and `CO_NEWLOCALS` on the
//!   live Python function. The Rust port receives a [`GraphFunc`]
//!   whose `code` slot carries an already-parsed `HostCode`; the
//!   equivalent checks run against `HostCode.co_flags` (for
//!   `CO_NEWLOCALS`) and `GraphFunc.code.co_freevars` (closures).
//!   `_not_rpython_` / `NOT_RPYTHON` docstring markers are not
//!   carried by the Rust object model yet and are documented as a
//!   Phase 5 gap.
//! * Upstream returns the *same* `FunctionGraph` instance that
//!   `PyGraph(func, code)` constructed. The Rust port re-exports
//!   `ctx.graph` after `FlowContext::build_flow()` finishes.

use super::bytecode::HostCode;
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
///
/// Phase 3 F3.7 gap: `_not_rpython_` marker + `NOT_RPYTHON` docstring
/// detection require carrying those flags on `GraphFunc`, which lands
/// alongside the annotator in Phase 5.
pub fn assert_rpythonic(func: &GraphFunc) -> Result<(), FlowContextError> {
    let code = func.code.as_ref().ok_or_else(|| {
        FlowContextError::Flowing(super::flowcontext::FlowingError::new(format!(
            "{} is not RPython: GraphFunc carries no HostCode",
            func.name
        )))
    })?;

    // upstream: `if func.__code__.co_cellvars: raise ValueError(
    //     "RPython functions cannot create closures")`.
    //
    // HostCode stores cellvars implicitly inside co_freevars / code-unit
    // constants; we check the closure pre-set on GraphFunc which is
    // populated from `func.__closure__` at construction time.
    if !func.closure.is_empty() {
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

    // upstream: `if not (func.__code__.co_flags & CO_NEWLOCALS)`.
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
    // equivalent of CPython's `inspect.isgeneratorfunction`. The second
    // condition guards the recursive case where `tweak_generator_graph`
    // re-enters `build_flow` after attaching `_generator_next_method_of_`;
    // we have no such attribute on GraphFunc, so the recursive branch
    // is currently unreachable and will be wired when the annotator
    // lands in Phase 5.
    if code.is_generator() {
        return Ok(make_generator_entry_graph(func));
    }

    let pygraph = PyGraph::new(func, &code);
    let graph = pygraph.graph;
    let mut ctx = FlowContext::new(graph, code);
    ctx.build_flow()?;
    fixeggblocks(&mut ctx.graph);

    if ctx.pycode.is_generator() {
        tweak_generator_graph(&mut ctx.graph);
    }

    Ok(ctx.graph)
}

#[cfg(test)]
mod tests {
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
        // attach a dummy HostCode with CO_NEWLOCALS set so closure is
        // the check that fires first.
        let mut code = make_empty_host_code();
        code.co_flags = CO_NEWLOCALS;
        func.code = Some(Box::new(code));
        func.closure.push(Constant::new(ConstValue::Int(1)));

        let err = assert_rpythonic(&func).unwrap_err();
        match err {
            FlowContextError::Flowing(err) => {
                assert!(err.message.contains("cannot create closures"));
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

    fn make_empty_host_code() -> HostCode {
        use super::super::argument::Signature;
        HostCode {
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
            consts: Vec::new(),
            names: Vec::new(),
            co_lnotab: Vec::new(),
            exceptiontable: Vec::new().into_boxed_slice(),
            signature: Signature::new(Vec::new(), None, None),
        }
    }
}
