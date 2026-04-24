//! Bundle an adapter-produced `FunctionGraph` into the
//! `(HostObject, PyGraph)` pair the annotator pipeline expects.
//!
//! Upstream analogue — `rpython/translator/interactive.py:25-26`:
//!
//! ```python
//! graph = self.context.buildflowgraph(entry_point)
//! self.context._prebuilt_graphs[entry_point] = graph
//! ```
//!
//! Line 25 runs upstream `build_flow` on Python bytecode and wraps the
//! resulting `FunctionGraph` inside a `PyGraph`. Line 26 seeds the
//! translator's prebuilt-graph cache so subsequent
//! `buildflowgraph(same entry_point)` calls short-circuit without
//! re-building.
//!
//! The Rust-source counterpart has no bytecode, so
//! `build_flow_from_rust` replaces line 25's work; this helper packages
//! the same `(host, pygraph)` pair that line 26 inserts into the cache.
//! Seeding the cache stays the caller's responsibility so this module
//! does not need to depend on `TranslationContext`.
//!
//! The synthetic [`HostCode`] populated here is the minimum needed for
//! upstream `cpython_code_signature` (`flowspace/bytecode.py`) to read
//! back the right argnames — `co_argcount`, `co_varnames`, `co_flags`.
//! `co_code` is empty because the function has no bytecode. Callers
//! that later introspect the code object (e.g. `is_generator`) will
//! see `CO_GENERATOR` unset, which is the correct Rust-fn answer.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use syn::{FnArg, ItemFn, Pat, PatIdent};

use super::build_flow::{AdapterError, build_flow_from_rust};
use crate::flowspace::bytecode::HostCode;
use crate::flowspace::model::{ConstValue, Constant, GraphFunc, HostObject};
use crate::flowspace::pygraph::PyGraph;

/// Walk `item_fn`, run the Rust-AST adapter, and return the
/// `(HostObject, PyGraph)` pair that the upstream translator cache
/// expects. The caller is responsible for seeding
/// `TranslationContext._prebuilt_graphs` with the returned pair, exactly
/// as `interactive.py:26` does:
///
/// ```ignore
/// let (host, pygraph) = build_host_function_from_rust(&item_fn)?;
/// translator
///     ._prebuilt_graphs
///     .borrow_mut()
///     .insert(host.clone(), pygraph);
/// ```
pub fn build_host_function_from_rust(
    item_fn: &ItemFn,
) -> Result<(HostObject, Rc<PyGraph>), AdapterError> {
    let argnames = extract_argnames(item_fn)?;
    let mut graph = build_flow_from_rust(item_fn)?;

    let name = item_fn.sig.ident.to_string();
    let nlocals = argnames.len() as u32;
    let host_code = HostCode::new(
        argnames.len() as u32,
        nlocals,
        0,
        0,
        rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
        Vec::new(),
        Vec::new(),
        argnames.clone(),
        "<rust-source>".to_string(),
        name.clone(),
        0,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new().into_boxed_slice(),
    );
    let empty_globals = Constant::new(ConstValue::Dict(Default::default()));
    let gf = GraphFunc::from_host_code(host_code.clone(), empty_globals, Vec::new());
    let host = HostObject::new_user_function(gf.clone());

    // upstream `PyGraph.__init__` (pygraph.py:20) assigns
    // `FunctionGraph.func = func` via `super().__init__`. Mirror that so
    // downstream helpers (`FlowContext::new`, `FunctionDesc.getuniquegraph`)
    // see the same GraphFunc the HostObject exposes.
    graph.func = Some(gf.clone());

    let pygraph = Rc::new(PyGraph {
        graph: Rc::new(RefCell::new(graph)),
        signature: RefCell::new(host_code.signature.clone()),
        // upstream `PyGraph.__init__`: `self.defaults =
        // func.__defaults__ or ()`. Rust-source adapter does not yet
        // surface default values; use the empty tuple shape.
        defaults: RefCell::new(Some(Vec::new())),
        access_directly: Cell::new(false),
        func: gf,
    });
    Ok((host, pygraph))
}

/// Extract the formal-parameter identifiers from a `syn::ItemFn`,
/// mirroring `collect_params` in `build_flow.rs`. Duplicated rather
/// than shared because the two callers consume different outputs — the
/// adapter needs `Hlvalue`s for the startblock, while this helper needs
/// the plain `String` names for `HostCode::co_varnames`.
fn extract_argnames(item_fn: &ItemFn) -> Result<Vec<String>, AdapterError> {
    let mut out = Vec::new();
    for input in &item_fn.sig.inputs {
        let ident = match input {
            FnArg::Receiver(_) => "self".to_string(),
            FnArg::Typed(pat_type) => match &*pat_type.pat {
                Pat::Ident(PatIdent {
                    ident,
                    by_ref: None,
                    subpat: None,
                    ..
                }) => ident.to_string(),
                _ => {
                    return Err(AdapterError::InvalidSignature {
                        reason: "parameter pattern must be a plain identifier".into(),
                    });
                }
            },
        };
        out.push(ident);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> ItemFn {
        syn::parse_str::<ItemFn>(src).expect("test fixture must parse")
    }

    #[test]
    fn zero_arg_function_produces_matching_signature() {
        let item = parse("fn zero() -> i64 { 1 }");
        let (host, pygraph) = build_host_function_from_rust(&item).expect("adapter");

        assert_eq!(host.qualname(), "zero");
        assert!(host.is_user_function());

        let sig = pygraph.signature.borrow();
        assert!(sig.argnames.is_empty());
        assert!(sig.varargname.is_none());
        assert!(sig.kwargname.is_none());

        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_argcount, 0);
        assert_eq!(code.co_nlocals, 0);
        assert!(code.co_varnames.is_empty());
    }

    #[test]
    fn two_arg_function_preserves_order_and_identity() {
        let item = parse("fn add(a: i64, b: i64) -> i64 { a + b }");
        let (host, pygraph) = build_host_function_from_rust(&item).expect("adapter");

        let sig = pygraph.signature.borrow();
        assert_eq!(sig.argnames, vec!["a".to_string(), "b".to_string()]);

        // FunctionGraph.func points at the same GraphFunc the
        // HostObject wraps — parity with upstream PyGraph.__init__.
        let graph_func_id = pygraph
            .graph
            .borrow()
            .func
            .as_ref()
            .expect("graph.func set")
            .id;
        let host_func_id = host.user_function().expect("user function").id;
        assert_eq!(graph_func_id, host_func_id);
    }

    #[test]
    fn startblock_inputargs_match_argnames() {
        let item = parse("fn add(a: i64, b: i64) -> i64 { a + b }");
        let (_host, pygraph) = build_host_function_from_rust(&item).expect("adapter");
        let inputargs = pygraph.graph.borrow().startblock.borrow().inputargs.clone();
        assert_eq!(inputargs.len(), 2);
        // Adapter builds startblock with named Variables — the names
        // come from the Rust parameter identifiers via `collect_params`.
        // `Variable::rename` (model.rs:2050) always trails the prefix
        // with `_` for valid-Python-identifier parity.
        for (expected, arg) in ["a_", "b_"].iter().zip(inputargs.iter()) {
            match arg {
                crate::flowspace::model::Hlvalue::Variable(v) => {
                    assert_eq!(v.name_prefix(), *expected);
                }
                other => panic!("expected Variable, got {other:?}"),
            }
        }
    }

    #[test]
    fn rejects_tuple_pattern_parameter() {
        // Matches `collect_params` in `build_flow.rs` — only plain
        // identifier patterns are accepted.
        let item = parse("fn f((a, b): (i64, i64)) -> i64 { a + b }");
        let err = build_host_function_from_rust(&item).unwrap_err();
        match err {
            AdapterError::InvalidSignature { reason } => {
                assert!(reason.contains("plain identifier"), "reason: {reason}");
            }
            other => panic!("expected InvalidSignature, got {other:?}"),
        }
    }

    #[test]
    fn seeds_into_translator_prebuilt_graphs_roundtrip() {
        use crate::translator::translator::TranslationContext;

        let item = parse("fn add(a: i64, b: i64) -> i64 { a + b }");
        let (host, pygraph) = build_host_function_from_rust(&item).expect("adapter");

        let ctx = TranslationContext::new();
        ctx._prebuilt_graphs
            .borrow_mut()
            .insert(host.clone(), pygraph.clone());

        // `buildflowgraph` must return the prebuilt graph unchanged
        // and leave no residual entry in the cache (upstream
        // `translator.py:50-51` pops).
        let retrieved = ctx.buildflowgraph(host.clone(), false).expect("prebuilt");
        assert!(Rc::ptr_eq(&retrieved, &pygraph));
        assert!(!ctx._prebuilt_graphs.borrow().contains_key(&host));
    }
}
